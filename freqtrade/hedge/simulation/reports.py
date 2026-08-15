from __future__ import annotations

from decimal import Decimal

from freqtrade.hedge.planning.context import PositionSide, ZERO
from .cross_wallet import CrossWallet, MutableLeg


def _lot_net(leg: MutableLeg) -> Decimal:
    if not leg.tactical_lots:
        return leg.tactical_realized_pnl
    return sum(
        (lot.realized_pnl + lot.funding - lot.fees for lot in leg.tactical_lots.values()),
        ZERO,
    )


def _effective_core_cost(leg: MutableLeg) -> Decimal:
    if leg.core_quantity <= ZERO:
        return ZERO
    realized_credit = max(_lot_net(leg), ZERO)
    return leg.core_average_price - realized_credit / leg.core_quantity


def _target_progress(current: Decimal, target: Decimal) -> Decimal:
    if target <= ZERO:
        return Decimal("1") if current <= ZERO else ZERO
    return min(max(current / target, ZERO), Decimal("1"))


def build_report(
    wallet: CrossWallet,
    final_mark: Decimal,
    *,
    target_net_quantity: Decimal = ZERO,
    net_gap_quantity: Decimal = ZERO,
    long_target_quantity: Decimal = ZERO,
    short_target_quantity: Decimal = ZERO,
) -> dict[str, Decimal | int | str | bool]:
    long_core_initial = wallet.core_cost_basis_initial.get(PositionSide.LONG, ZERO)
    short_core_initial = wallet.core_cost_basis_initial.get(PositionSide.SHORT, ZERO)
    long_core_change = (
        wallet.long.core_average_price - long_core_initial if long_core_initial else ZERO
    )
    short_core_change = (
        wallet.short.core_average_price - short_core_initial if short_core_initial else ZERO
    )
    long_trading = wallet.long.realized_pnl + wallet.long.immutable().unrealized_pnl(final_mark)
    short_trading = wallet.short.realized_pnl + wallet.short.immutable().unrealized_pnl(final_mark)
    long_net = long_trading + wallet.long_funding - wallet.long_fees_paid
    short_net = short_trading + wallet.short_funding - wallet.short_fees_paid
    tactical_gross = wallet.long.tactical_realized_pnl + wallet.short.tactical_realized_pnl
    tactical_net = _lot_net(wallet.long) + _lot_net(wallet.short)
    if not wallet.long.tactical_lots and not wallet.short.tactical_lots:
        tactical_net = tactical_gross + wallet.tactical_funding - wallet.tactical_fees_paid
    final_equity = wallet.equity(final_mark)
    reconciliation_error = final_equity - wallet.initial_balance - long_net - short_net
    if abs(reconciliation_error) < Decimal("1e-18"):
        reconciliation_error = ZERO
    open_lots = sum(
        1
        for leg in (wallet.long, wallet.short)
        for lot in leg.tactical_lots.values()
        if lot.quantity > ZERO
    )
    closed_lots = sum(
        1
        for leg in (wallet.long, wallet.short)
        for lot in leg.tactical_lots.values()
        if lot.quantity == ZERO and lot.closed_quantity > ZERO
    )
    current_net_qty = wallet.long.quantity - wallet.short.quantity
    total_pnl = final_equity - wallet.initial_balance
    total_return_ratio = (
        total_pnl / wallet.initial_balance if wallet.initial_balance > ZERO else ZERO
    )
    return {
        "initial_balance": wallet.initial_balance,
        "total_pnl": total_pnl,
        "total_return_ratio": total_return_ratio,
        "final_long_quantity": wallet.long.quantity,
        "final_short_quantity": wallet.short.quantity,
        "long_pnl": long_net,
        "short_pnl": short_net,
        "long_trading_pnl": long_trading,
        "short_trading_pnl": short_trading,
        "core_cost_change_long": long_core_change,
        "core_cost_change_short": short_core_change,
        "core_effective_cost_long": _effective_core_cost(wallet.long),
        "core_effective_cost_short": _effective_core_cost(wallet.short),
        "tactical_trading_pnl": tactical_net,
        "tactical_trading_pnl_gross": tactical_gross,
        "tactical_open_lot_count": open_lots,
        "tactical_closed_lot_count": closed_lots,
        "gross_peak": wallet.gross_peak,
        "gross_peak_ratio": (
            wallet.gross_peak / wallet.initial_balance
            if wallet.initial_balance > ZERO
            else ZERO
        ),
        "net_exposure": wallet.net_notional(final_mark),
        "funding": wallet.funding_paid,
        "fees": wallet.fees_paid,
        "maker_fees": wallet.maker_fees_paid,
        "taker_fees": wallet.fees_paid - wallet.maker_fees_paid,
        "maker_fill_count": wallet.maker_fill_count,
        "taker_fill_count": wallet.taker_fill_count,
        "long_fees": wallet.long_fees_paid,
        "short_fees": wallet.short_fees_paid,
        "long_funding": wallet.long_funding,
        "short_funding": wallet.short_funding,
        "dual_leg_duration_seconds": wallet.hedge_duration_seconds,
        "add_count": wallet.long.add_count + wallet.short.add_count,
        "reduce_count": wallet.long.reduce_count + wallet.short.reduce_count,
        "long_add_count": wallet.long.add_count,
        "short_add_count": wallet.short.add_count,
        "long_reduce_count": wallet.long.reduce_count,
        "short_reduce_count": wallet.short.reduce_count,
        "max_drawdown": wallet.max_drawdown,
        "final_equity": final_equity,
        "final_balance": wallet.balance,
        "final_gross_notional": wallet.gross_notional(final_mark),
        "available_balance": wallet.available_balance(final_mark),
        "active_order_margin": wallet.active_order_margin(),
        "maintenance_margin": wallet.maintenance_margin(final_mark),
        "margin_ratio": wallet.margin_ratio(final_mark),
        "liquidation_buffer": wallet.liquidation_buffer(final_mark),
        "liquidation_buffer_ratio": wallet.liquidation_buffer_ratio(final_mark),
        "liquidated": wallet.liquidated,
        "liquidation_count": wallet.liquidation_count,
        "liquidation_warning": wallet.liquidation_warning(final_mark),
        "target_net_quantity": target_net_quantity,
        "current_net_quantity": current_net_qty,
        "net_gap_quantity": target_net_quantity - current_net_qty,
        "planning_net_gap_quantity": net_gap_quantity,
        "long_target_quantity": long_target_quantity,
        "short_target_quantity": short_target_quantity,
        "long_target_progress": _target_progress(wallet.long.quantity, long_target_quantity),
        "short_target_progress": _target_progress(wallet.short.quantity, short_target_quantity),
        "pnl_reconciliation_error": reconciliation_error,
    }
