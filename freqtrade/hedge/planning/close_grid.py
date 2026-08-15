from __future__ import annotations

from decimal import Decimal

from .context import (
    IntentAction,
    OrderIntent,
    OrderSide,
    PlanningContext,
    PositionBucket,
    PositionSide,
    TacticalLot,
    ZERO,
    q_down,
    q_up,
)
from .target import TargetPosition


def _close_side(side: PositionSide) -> OrderSide:
    return OrderSide.SELL if side is PositionSide.LONG else OrderSide.BUY


def _close_price(context: PlanningContext, side: PositionSide, raw_price: Decimal) -> Decimal:
    if raw_price <= ZERO:
        return ZERO
    if side is PositionSide.LONG:
        return q_up(raw_price, context.market.tick_size)
    return q_down(raw_price, context.market.tick_size)



def _cap_close_quantity(
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


def _price_for_average(
    context: PlanningContext,
    side: PositionSide,
    average: Decimal,
    layer: int,
) -> Decimal:
    base = average if average > ZERO else context.market.mark
    movement = context.config.take_profit_spacing * Decimal(max(layer, 1))
    raw = base * (
        Decimal("1") + movement
        if side is PositionSide.LONG
        else Decimal("1") - movement
    )
    return _close_price(context, side, raw)


def _lot_orders(
    context: PlanningContext,
    side: PositionSide,
    lots: tuple[TacticalLot, ...],
    budget: Decimal,
) -> tuple[list[OrderIntent], Decimal]:
    if not lots or budget <= ZERO:
        return [], ZERO
    total = sum((lot.quantity for lot in lots), ZERO)
    if total <= ZERO:
        return [], ZERO
    created = ZERO
    orders: list[OrderIntent] = []
    sorted_lots = sorted(lots, key=lambda item: (item.opened_at, item.layer, item.lot_id))
    for index, lot in enumerate(sorted_lots, 1):
        remaining = q_down(max(budget - created, ZERO), context.market.qty_step)
        if remaining <= ZERO:
            break
        proportional = budget * lot.quantity / total
        qty = q_down(
            min(lot.quantity, remaining if index == len(sorted_lots) else proportional),
            context.market.qty_step,
        )
        layer = max(lot.layer, 1)
        price = _price_for_average(context, side, lot.average_price, layer)
        qty = _cap_close_quantity(context, qty, price)
        if not _tradable(context, qty, price):
            continue
        orders.append(
            OrderIntent.deterministic(
                symbol=context.market.symbol,
                position_side=side,
                order_side=_close_side(side),
                action=IntentAction.REDUCE,
                bucket=PositionBucket.TACTICAL,
                quantity=qty,
                price=price,
                reduce_only=True,
                layer=layer,
                reason="tactical_lot_take_profit",
                epoch=context.market.timestamp.isoformat(),
                tactical_lot_id=lot.lot_id,
            )
        )
        created += qty
    return orders, created


def _aggregate_orders(
    context: PlanningContext,
    side: PositionSide,
    *,
    bucket: PositionBucket,
    average_price: Decimal,
    budget: Decimal,
) -> list[OrderIntent]:
    cfg = context.config
    if budget <= ZERO or cfg.take_profit_layers <= 0:
        return []
    per_layer = q_down(budget / Decimal(cfg.take_profit_layers), context.market.qty_step)
    remaining = budget
    output: list[OrderIntent] = []
    for layer in range(1, cfg.take_profit_layers + 1):
        qty = q_down(
            remaining if layer == cfg.take_profit_layers else per_layer,
            context.market.qty_step,
        )
        price = _price_for_average(context, side, average_price, layer)
        qty = _cap_close_quantity(context, qty, price)
        if not _tradable(context, qty, price):
            continue
        output.append(
            OrderIntent.deterministic(
                symbol=context.market.symbol,
                position_side=side,
                order_side=_close_side(side),
                action=IntentAction.REDUCE,
                bucket=bucket,
                quantity=qty,
                price=price,
                reduce_only=True,
                layer=layer,
                reason="layered_take_profit",
                epoch=context.market.timestamp.isoformat(),
            )
        )
        remaining = q_down(max(remaining - qty, ZERO), context.market.qty_step)
    return output


def build_close_grid(
    context: PlanningContext,
    side: PositionSide,
    target: TargetPosition,
) -> tuple[OrderIntent, ...]:
    cfg = context.config
    leg = context.wallet.leg(side)
    if leg.quantity <= ZERO or cfg.take_profit_layers <= 0:
        return ()

    protected_core = min(leg.core_quantity, target.core_quantity * cfg.core_min_fraction)
    raw_reducible = max(leg.quantity - protected_core, ZERO)
    budget = q_down(raw_reducible * cfg.tactical_reduce_fraction, context.market.qty_step)
    if budget < context.market.min_qty:
        return ()

    orders: list[OrderIntent] = []
    tactical_budget = q_down(min(budget, leg.tactical_quantity), context.market.qty_step)
    lot_orders, lot_created = _lot_orders(
        context,
        side,
        leg.tactical_lots,
        tactical_budget,
    )
    orders.extend(lot_orders)

    remaining_tactical = q_down(
        max(tactical_budget - lot_created, ZERO),
        context.market.qty_step,
    )
    if remaining_tactical > ZERO:
        orders.extend(
            _aggregate_orders(
                context,
                side,
                bucket=PositionBucket.TACTICAL,
                average_price=leg.tactical_average_price or leg.average_price,
                budget=remaining_tactical,
            )
        )

    created_tactical = sum(
        (item.quantity for item in orders if item.bucket is PositionBucket.TACTICAL),
        ZERO,
    )
    core_budget = q_down(max(budget - created_tactical, ZERO), context.market.qty_step)
    if core_budget > ZERO:
        core_budget = min(core_budget, max(leg.core_quantity - protected_core, ZERO))
        orders.extend(
            _aggregate_orders(
                context,
                side,
                bucket=PositionBucket.CORE,
                average_price=leg.core_average_price or leg.average_price,
                budget=q_down(core_budget, context.market.qty_step),
            )
        )
    return tuple(orders)
