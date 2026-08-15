from __future__ import annotations

from decimal import Decimal

from .context import (
    ActiveOrder,
    IntentAction,
    OrderIntent,
    PlanningContext,
    PlanningResult,
    PositionSide,
    StrategyPlanningPort,
    ZERO,
)
from .core_tactical import plan_leg


def _action_family(action: IntentAction) -> str:
    if action in {IntentAction.OPEN, IntentAction.INCREASE}:
        return "ENTRY"
    if action in {IntentAction.REDUCE, IntentAction.CLOSE}:
        return "EXIT"
    return action.value


def _compatible_family(desired: OrderIntent, active: ActiveOrder) -> bool:
    return (
        desired.symbol == active.symbol
        and desired.position_side is active.position_side
        and desired.order_side is active.order_side
        and desired.reduce_only == active.reduce_only
        and desired.bucket is active.bucket
        and _action_family(desired.action) == _action_family(active.action)
        and desired.order_type is active.order_type
        and desired.time_in_force is active.time_in_force
        and (active.layer == 0 or desired.layer == active.layer)
        and desired.tactical_lot_id == active.tactical_lot_id
    )


def _same_enough(context: PlanningContext, desired: OrderIntent, active: ActiveOrder) -> bool:
    if not _compatible_family(desired, active):
        return False
    price_tolerance = context.market.tick_size * Decimal(
        context.config.replace_price_tolerance_ticks
    )
    qty_tolerance = context.market.qty_step * Decimal(
        context.config.replace_qty_tolerance_steps
    )
    return (
        abs(desired.price - active.price) <= price_tolerance
        and abs(desired.quantity - active.quantity) <= qty_tolerance
    )


def _replacement_debounced(
    context: PlanningContext,
    desired: OrderIntent,
    active: ActiveOrder,
) -> bool:
    if not _compatible_family(desired, active):
        return False
    age_seconds = (context.market.timestamp - active.created_at).total_seconds()
    return 0 <= age_seconds < context.config.replace_min_age_seconds


def _enforce_exposure(
    context: PlanningContext,
    orders: tuple[OrderIntent, ...],
) -> tuple[tuple[OrderIntent, ...], tuple[str, ...]]:
    equity = max(context.wallet.equity, Decimal("0.00000001"))
    mark = context.market.mark
    leverage = context.wallet.leverage
    accepted: list[OrderIntent] = []
    diagnostics: list[str] = []
    projected_long_notional = context.wallet.long.quantity * mark
    projected_short_notional = context.wallet.short.quantity * mark

    # Current available balance already reserves active order margin. Add back only orders
    # managed by this symbol so the ideal set can replace them without double counting.
    replaceable_margin = sum(
        (
            active.notional / leverage
            for active in context.wallet.active_orders
            if active.symbol == context.market.symbol and not active.reduce_only
        ),
        ZERO,
    )
    margin_budget = max(context.wallet.available_balance + replaceable_margin, ZERO)
    reserved_margin = ZERO

    for order in sorted(
        orders,
        key=lambda item: (
            item.reduce_only is False,
            item.position_side.value,
            item.layer,
            item.intent_id,
        ),
    ):
        if order.reduce_only:
            accepted.append(order)
            continue

        incremental_notional = order.notional
        if order.position_side is PositionSide.LONG:
            candidate_long = projected_long_notional + incremental_notional
            candidate_short = projected_short_notional
            side_ratio = candidate_long / equity
            side_cap = context.config.max_wallet_exposure_long
        else:
            candidate_long = projected_long_notional
            candidate_short = projected_short_notional + incremental_notional
            side_ratio = candidate_short / equity
            side_cap = context.config.max_wallet_exposure_short
        gross_ratio = (candidate_long + candidate_short) / equity
        order_margin = incremental_notional / leverage
        maintenance = incremental_notional * context.config.maintenance_margin_rate
        margin_ok = reserved_margin + order_margin + maintenance <= margin_budget

        if (
            side_ratio <= side_cap
            and gross_ratio <= context.config.max_gross_wallet_exposure
            and margin_ok
        ):
            accepted.append(order)
            projected_long_notional, projected_short_notional = candidate_long, candidate_short
            reserved_margin += order_margin + maintenance
        else:
            diagnostics.append(
                f"{order.position_side.value}:rejected:{order.intent_id}:exposure_or_margin"
            )
    return tuple(accepted), tuple(diagnostics)


class PureHedgePlanner(StrategyPlanningPort):
    """Deterministic, side-isolated strategy planner with explicit order diff classes."""

    def plan(self, context: PlanningContext) -> PlanningResult:
        long_orders, long_state, long_target, long_diag = plan_leg(
            context,
            PositionSide.LONG,
        )
        short_orders, short_state, short_target, short_diag = plan_leg(
            context,
            PositionSide.SHORT,
        )
        desired, risk_diag = _enforce_exposure(context, tuple(long_orders + short_orders))

        managed_active = tuple(
            active
            for active in context.wallet.active_orders
            if active.symbol == context.market.symbol
        )
        foreign_active = tuple(
            active
            for active in context.wallet.active_orders
            if active.symbol != context.market.symbol
        )
        kept_active: set[str] = {active.order_id for active in foreign_active}
        matched_managed: set[str] = set()
        submit: list[OrderIntent] = []
        modify: set[str] = set()
        debounce_diag: list[str] = []

        for order in desired:
            exact = next(
                (
                    active
                    for active in managed_active
                    if active.order_id not in matched_managed
                    and _same_enough(context, order, active)
                ),
                None,
            )
            if exact is not None:
                matched_managed.add(exact.order_id)
                kept_active.add(exact.order_id)
                continue

            compatible = next(
                (
                    active
                    for active in managed_active
                    if active.order_id not in matched_managed
                    and _compatible_family(order, active)
                ),
                None,
            )
            if compatible is not None and _replacement_debounced(
                context,
                order,
                compatible,
            ):
                matched_managed.add(compatible.order_id)
                kept_active.add(compatible.order_id)
                debounce_diag.append(
                    f"{order.position_side.value}:replacement_debounced:"
                    f"{compatible.order_id}"
                )
                continue
            if compatible is not None:
                matched_managed.add(compatible.order_id)
                modify.add(compatible.order_id)
            submit.append(order)

        gross_ratio = context.wallet.gross_notional(context.market.mark) / max(
            context.wallet.equity,
            Decimal("0.00000001"),
        )
        unstuck_sides = {
            order.position_side
            for order in desired
            if order.action is IntentAction.UNSTUCK
        }
        risk_cancel: set[str] = set()
        delete: set[str] = set()
        for active in managed_active:
            if active.order_id in matched_managed:
                continue
            net_repair_active = (
                abs(long_target.net_gap_quantity * context.market.mark)
                / max(context.wallet.equity, Decimal("0.00000001"))
                >= context.config.net_repair_threshold
            )
            opposing_net_repair = net_repair_active and (
                (
                    long_target.net_gap_quantity > ZERO
                    and active.position_side is PositionSide.SHORT
                )
                or (
                    long_target.net_gap_quantity < ZERO
                    and active.position_side is PositionSide.LONG
                )
            )
            if not active.reduce_only and (
                gross_ratio >= context.config.max_gross_wallet_exposure
                or active.position_side in unstuck_sides
                or opposing_net_repair
            ):
                risk_cancel.add(active.order_id)
            else:
                delete.add(active.order_id)

        cancellations = tuple(sorted(modify | delete | risk_cancel))
        submit.sort(
            key=lambda item: (
                item.position_side.value,
                item.reduce_only,
                item.layer,
                item.price,
                item.intent_id,
            )
        )
        return PlanningResult(
            ideal_orders=desired,
            submit_orders=tuple(submit),
            cancel_order_ids=cancellations,
            kept_order_ids=tuple(sorted(kept_active)),
            long_state=long_state,
            short_state=short_state,
            diagnostics=tuple(
                long_diag + short_diag + risk_diag + tuple(debounce_diag)
            ),
            modify_order_ids=tuple(sorted(modify)),
            delete_order_ids=tuple(sorted(delete)),
            risk_cancel_order_ids=tuple(sorted(risk_cancel)),
            target_net_quantity=long_target.target_net_quantity,
            net_gap_quantity=long_target.net_gap_quantity,
            long_target_quantity=long_target.total_quantity,
            short_target_quantity=short_target.total_quantity,
        )
