from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from .context import D, ONE, ZERO, PlanningContext, PositionSide, q_down


@dataclass(frozen=True, slots=True)
class TargetPosition:
    """Per-leg target plus account-level net-position objective."""

    side: PositionSide
    core_quantity: Decimal
    tactical_quantity: Decimal
    maximum_quantity: Decimal
    current_quantity: Decimal = ZERO
    quantity_gap: Decimal = ZERO
    target_net_quantity: Decimal = ZERO
    net_gap_quantity: Decimal = ZERO

    @property
    def total_quantity(self) -> Decimal:
        return self.core_quantity + self.tactical_quantity

    @property
    def progress(self) -> Decimal:
        if self.total_quantity <= ZERO:
            return ONE if self.current_quantity <= ZERO else ZERO
        return min(max(self.current_quantity / self.total_quantity, ZERO), ONE)


def _target_net_quantity(context: PlanningContext) -> Decimal:
    if context.target_net_quantity is not None:
        return q_down(context.target_net_quantity, context.market.qty_step)
    equity = max(context.wallet.equity, ZERO)
    raw = equity * context.config.target_net_wallet_exposure / context.market.mark
    # q_down truncates toward zero for Decimal and is therefore conservative for both signs.
    return q_down(raw, context.market.qty_step)


def calculate_target(context: PlanningContext, side: PositionSide) -> TargetPosition:
    cfg = context.config
    market = context.market
    current = context.wallet.leg(side).quantity
    target_net = _target_net_quantity(context)
    current_net = context.wallet.long.quantity - context.wallet.short.quantity
    net_gap = target_net - current_net

    if not cfg.enabled(side):
        return TargetPosition(
            side=side,
            core_quantity=ZERO,
            tactical_quantity=ZERO,
            maximum_quantity=ZERO,
            current_quantity=current,
            quantity_gap=-current,
            target_net_quantity=target_net,
            net_gap_quantity=net_gap,
        )

    equity = max(context.wallet.equity, ZERO)
    price = market.mark
    core = equity * cfg.core_exposure(side) / price
    signal = max(-ONE, min(ONE, D(context.signal(side))))
    tactical = equity * cfg.tactical_exposure(side) * max(signal, ZERO) / price
    maximum = equity * cfg.side_cap(side) / price

    # When account net exposure materially misses its configured objective, suppress new
    # tactical risk on the opposing side. Existing opposing inventory remains reducible.
    if equity > ZERO:
        net_gap_ratio = abs(net_gap * price) / equity
        opposing = (
            net_gap > ZERO and side is PositionSide.SHORT
        ) or (
            net_gap < ZERO and side is PositionSide.LONG
        )
        if opposing and net_gap_ratio >= cfg.net_repair_threshold:
            tactical = ZERO

    core = q_down(core, market.qty_step)
    tactical = q_down(tactical, market.qty_step)
    maximum = q_down(maximum, market.qty_step)
    if core > maximum:
        core = maximum
        tactical = ZERO
    elif core + tactical > maximum:
        tactical = q_down(maximum - core, market.qty_step)
    total = core + tactical
    return TargetPosition(
        side=side,
        core_quantity=core,
        tactical_quantity=tactical,
        maximum_quantity=maximum,
        current_quantity=current,
        quantity_gap=q_down(total - current, market.qty_step),
        target_net_quantity=target_net,
        net_gap_quantity=q_down(net_gap, market.qty_step),
    )
