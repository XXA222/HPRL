from __future__ import annotations

from datetime import timedelta
from decimal import Decimal

from .context import (
    IntentAction,
    OrderIntent,
    OrderSide,
    OrderType,
    PlanningContext,
    PositionBucket,
    PositionSide,
    TimeInForce,
    ZERO,
    q_down,
)
from .target import TargetPosition


def unstuck_budget_keys(timestamp) -> tuple[str, str]:
    day = timestamp.date().isoformat()
    year, week, _ = timestamp.isocalendar()
    return day, f"{year}-W{week:02d}"


def _oldest_tactical_lot(context: PlanningContext, side: PositionSide):
    lots = context.wallet.leg(side).tactical_lots
    return min(lots, key=lambda item: (item.opened_at, item.lot_id)) if lots else None


def build_unstuck_intent(
    context: PlanningContext,
    side: PositionSide,
    target: TargetPosition,
) -> OrderIntent | None:
    """Build a budgeted reduce-only action which proves a minimum gross-risk improvement."""
    cfg = context.config
    state = context.state(side)
    equity = max(context.wallet.equity, Decimal("0.00000001"))
    gross_ratio = context.wallet.gross_notional(context.market.mark) / equity
    oldest_lot = _oldest_tactical_lot(context, side)
    aged = (
        oldest_lot is not None
        and cfg.unstuck_max_holding_seconds > 0
        and context.market.timestamp - oldest_lot.opened_at
        >= timedelta(seconds=cfg.unstuck_max_holding_seconds)
    )
    if gross_ratio < cfg.unstuck_trigger_gross_exposure and not aged:
        return None

    if (
        state.last_unstuck_at is not None
        and context.market.timestamp - state.last_unstuck_at
        < timedelta(seconds=cfg.unstuck_min_cooldown_seconds)
    ):
        return None

    leg = context.wallet.leg(side)
    core_floor = min(leg.core_quantity, target.core_quantity * cfg.core_min_fraction)
    reducible = max(leg.quantity - core_floor, ZERO)
    preferred = oldest_lot.quantity if oldest_lot is not None else reducible
    price = context.market.bid if side is PositionSide.LONG else context.market.ask
    quantity_cap = min(reducible * cfg.unstuck_reduce_fraction, preferred)
    if cfg.max_single_order_notional > ZERO:
        quantity_cap = min(quantity_cap, cfg.max_single_order_notional / price)
    qty = q_down(quantity_cap, context.market.qty_step)
    if (
        qty <= ZERO
        or qty < context.market.min_qty
        or qty * price < context.market.min_notional
    ):
        return None

    improvement = qty * context.market.mark / equity
    if improvement < cfg.unstuck_min_risk_improvement:
        return None

    average = (
        oldest_lot.average_price
        if oldest_lot is not None
        else leg.tactical_average_price
        if leg.tactical_quantity > ZERO
        else leg.core_average_price
    )
    estimated_pnl = (price - average) * qty * side.direction
    estimated_loss = max(-estimated_pnl, ZERO)
    day_key, week_key = unstuck_budget_keys(context.market.timestamp)
    daily_used = state.unstuck_daily_loss if state.unstuck_budget_day == day_key else ZERO
    weekly_used = state.unstuck_weekly_loss if state.unstuck_budget_week == week_key else ZERO
    if daily_used + estimated_loss > equity * cfg.unstuck_daily_loss_budget:
        return None
    if weekly_used + estimated_loss > equity * cfg.unstuck_weekly_loss_budget:
        return None

    order_side = OrderSide.SELL if side is PositionSide.LONG else OrderSide.BUY
    return OrderIntent.deterministic(
        symbol=context.market.symbol,
        position_side=side,
        order_side=order_side,
        action=IntentAction.UNSTUCK,
        bucket=(
            PositionBucket.TACTICAL
            if oldest_lot is not None or leg.tactical_quantity > ZERO
            else PositionBucket.CORE
        ),
        quantity=qty,
        price=price,
        reduce_only=True,
        order_type=OrderType.LIMIT if cfg.unstuck_limit_only else OrderType.MARKET,
        time_in_force=TimeInForce.GTC if cfg.unstuck_limit_only else TimeInForce.IOC,
        reason="aged_or_gross_exposure_unstuck",
        epoch=context.market.timestamp.isoformat(),
        tactical_lot_id=oldest_lot.lot_id if oldest_lot is not None else None,
    )
