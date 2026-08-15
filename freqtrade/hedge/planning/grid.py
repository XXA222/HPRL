from __future__ import annotations

from decimal import Decimal
from hashlib import sha256

from .context import (
    IntentAction,
    OrderIntent,
    OrderSide,
    PlanningContext,
    PositionBucket,
    PositionSide,
    ZERO,
    q_down,
    q_up,
)
from .target import TargetPosition


def _entry_side(side: PositionSide) -> OrderSide:
    return OrderSide.BUY if side is PositionSide.LONG else OrderSide.SELL


def _entry_price(context: PlanningContext, side: PositionSide, layer: int) -> Decimal:
    cfg = context.config
    spacing = cfg.grid_spacing * (cfg.grid_spacing_growth ** max(layer - 1, 0))
    raw = context.market.mark * (
        Decimal("1") - spacing if side is PositionSide.LONG else Decimal("1") + spacing
    )
    if raw <= ZERO:
        return ZERO
    if side is PositionSide.LONG:
        return q_down(raw, context.market.tick_size)
    return q_up(raw, context.market.tick_size)


def _tactical_lot_id(
    context: PlanningContext,
    side: PositionSide,
    layer: int,
    price: Decimal,
) -> str:
    raw = (
        f"{context.market.symbol}|{side.value}|{layer}|"
        f"{context.market.timestamp.isoformat()}|{price}"
    ).encode()
    return "lot-" + sha256(raw).hexdigest()[:24]


def _cap_order_quantity(
    context: PlanningContext,
    quantity: Decimal,
    price: Decimal,
) -> Decimal:
    cap = context.config.max_single_order_notional
    if cap <= ZERO:
        return q_down(quantity, context.market.qty_step)
    return q_down(min(quantity, cap / price), context.market.qty_step)


def _tradable(context: PlanningContext, quantity: Decimal, price: Decimal) -> bool:
    return (
        quantity > ZERO
        and quantity >= context.market.min_qty
        and price > ZERO
        and quantity * price >= context.market.min_notional
    )


def _preserve_active_entries(
    context: PlanningContext,
    side: PositionSide,
    target: TargetPosition,
) -> tuple[OrderIntent, ...]:
    """Keep bounded active entries while new entry placement is intentionally blocked."""
    cfg = context.config
    if cfg.max_pending_entries <= 0:
        return ()
    leg = context.wallet.leg(side)
    budget = q_down(
        max(target.total_quantity - leg.quantity, ZERO),
        context.market.qty_step,
    )
    qty_tolerance = context.market.qty_step * Decimal(cfg.replace_qty_tolerance_steps)
    preserved: list[OrderIntent] = []
    preserved_qty = ZERO
    fallback_layer = max(context.state(side).grid_layers_filled + 1, 1)
    for active in sorted(
        (
            item
            for item in context.wallet.active_orders
            if item.symbol == context.market.symbol
            and item.position_side is side
            and not item.reduce_only
        ),
        key=lambda item: (item.layer, item.price, item.order_id),
    ):
        remaining_budget = max(budget + qty_tolerance - preserved_qty, ZERO)
        desired_qty = q_down(min(active.quantity, remaining_budget), context.market.qty_step)
        layer = active.layer if active.layer > 0 else fallback_layer
        if layer > cfg.max_grid_layers or not _tradable(
            context,
            desired_qty,
            active.price,
        ):
            continue
        preserved.append(
            OrderIntent(
                intent_id=active.client_order_id or f"preserved-{active.order_id}",
                symbol=active.symbol,
                position_side=active.position_side,
                order_side=active.order_side,
                action=active.action,
                bucket=active.bucket,
                quantity=desired_qty,
                price=active.price,
                reduce_only=False,
                order_type=active.order_type,
                time_in_force=active.time_in_force,
                layer=layer,
                reason="preserve_active_entry",
                tactical_lot_id=active.tactical_lot_id,
            )
        )
        preserved_qty += desired_qty
        if len(preserved) >= cfg.max_pending_entries:
            break
    return tuple(preserved)


def build_entry_grid(
    context: PlanningContext,
    side: PositionSide,
    target: TargetPosition,
    *,
    allow_new_entries: bool = True,
) -> tuple[OrderIntent, ...]:
    """Build a finite target grid; preserve active entries only while placement is blocked."""
    cfg = context.config
    leg = context.wallet.leg(side)
    state = context.state(side)

    if not allow_new_entries:
        return _preserve_active_entries(context, side, target)

    missing = q_down(
        max(target.total_quantity - leg.quantity, ZERO),
        context.market.qty_step,
    )
    highest_layer = state.grid_layers_filled
    order_slots_left = cfg.max_pending_entries
    layers_left = max(cfg.max_grid_layers - highest_layer, 0)
    if missing < context.market.min_qty or layers_left <= 0 or order_slots_left <= 0:
        return ()

    allocations: list[tuple[int, Decimal, Decimal, str]] = []
    allocated = ZERO
    next_layer = highest_layer + 1

    if leg.quantity == ZERO and state.grid_layers_filled == 0:
        first_qty = q_down(
            min(missing * cfg.initial_entry_fraction, missing),
            context.market.qty_step,
        )
        first_price = (
            q_down(context.market.bid, context.market.tick_size)
            if side is PositionSide.LONG
            else q_up(context.market.ask, context.market.tick_size)
        )
        if _tradable(context, first_qty, first_price):
            allocations.append((next_layer, first_qty, first_price, "initial_entry"))
            allocated += first_qty
            next_layer += 1
            layers_left -= 1

    remaining = q_down(max(missing - allocated, ZERO), context.market.qty_step)
    if remaining > ZERO and layers_left > 0:
        raw_weights = [cfg.grid_qty_growth**i for i in range(layers_left)]
        weight_sum = sum(raw_weights, ZERO)
        grid_allocated = ZERO
        for offset, weight in enumerate(raw_weights):
            layer = next_layer + offset
            qty = q_down(
                max(remaining - grid_allocated, ZERO)
                if offset == layers_left - 1
                else remaining * weight / weight_sum,
                context.market.qty_step,
            )
            price = _entry_price(context, side, layer)
            if not _tradable(context, qty, price):
                continue
            allocations.append((layer, qty, price, "bounded_grid_entry"))
            grid_allocated += qty

    intents: list[OrderIntent] = []
    core_remaining = q_down(
        max(target.core_quantity - leg.core_quantity, ZERO),
        context.market.qty_step,
    )
    for layer, qty, price, reason in allocations:
        core_qty = q_down(min(qty, core_remaining), context.market.qty_step)
        tactical_qty = q_down(max(qty - core_qty, ZERO), context.market.qty_step)
        created_core = ZERO
        for bucket, bucket_qty in (
            (PositionBucket.CORE, core_qty),
            (PositionBucket.TACTICAL, tactical_qty),
        ):
            bucket_qty = _cap_order_quantity(context, bucket_qty, price)
            if not _tradable(context, bucket_qty, price):
                continue
            if len(intents) >= order_slots_left:
                break
            action = (
                IntentAction.OPEN
                if leg.quantity == ZERO and not intents
                else IntentAction.INCREASE
            )
            intents.append(
                OrderIntent.deterministic(
                    symbol=context.market.symbol,
                    position_side=side,
                    order_side=_entry_side(side),
                    action=action,
                    bucket=bucket,
                    quantity=bucket_qty,
                    price=price,
                    reduce_only=False,
                    layer=layer,
                    reason=reason,
                    epoch=context.market.timestamp.isoformat(),
                    tactical_lot_id=(
                        _tactical_lot_id(context, side, layer, price)
                        if bucket is PositionBucket.TACTICAL
                        else None
                    ),
                )
            )
            if bucket is PositionBucket.CORE:
                created_core += bucket_qty
        core_remaining = q_down(
            max(core_remaining - created_core, ZERO),
            context.market.qty_step,
        )
    return tuple(intents)
