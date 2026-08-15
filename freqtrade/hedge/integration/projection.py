"""Projection bridge from direction-two facts into central and direction-three models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

from freqtrade.enums.hedge import PositionAction, PositionSide
from freqtrade.hedge.exchange.base import (
    Direction2HealthFact,
    OrderFact,
    ReadonlyAccountView,
)
from freqtrade.hedge.position_book import PositionRecord
from freqtrade.hedge.readonly.service import ReadonlyRuntimeSnapshot
from freqtrade.hedge.risk.models import AccountRiskSnapshot, PendingOrderRisk
from freqtrade.hedge.risk.portfolio import PositionRiskLeg, build_risk_portfolio
from freqtrade.hedge.symbols import canonicalize_symbol


@dataclass(frozen=True, slots=True)
class CentralRuntimeProjection:
    positions: tuple[PositionRecord, ...]
    risk: AccountRiskSnapshot | None
    reconciliation_status: str
    reconciliation_at: datetime | None
    reconciliation_details: tuple[str, ...]
    stream_state: str
    stream_last_event_at: datetime | None
    stream_reconnect_count: int
    checks: dict[str, bool]
    reasons: tuple[str, ...]
    source_version: str
    source_event_time: datetime
    stale: bool


def _position_side(value: str) -> PositionSide:
    return PositionSide(str(value).upper())


def _order_action(order: OrderFact) -> PositionAction:
    side = _position_side(order.position_side)
    order_side = order.side.upper()
    increases = (side is PositionSide.LONG and order_side == "BUY") or (
        side is PositionSide.SHORT and order_side == "SELL"
    )
    return PositionAction.INCREASE if increases else PositionAction.REDUCE


def _reference_price(order: OrderFact, positions: tuple[PositionRiskLeg, ...]) -> Decimal | None:
    if order.average_price > 0:
        return order.average_price
    for leg in positions:
        if leg.symbol == canonicalize_symbol(order.symbol) and leg.position_side is _position_side(order.position_side):
            if leg.mark_price > 0:
                return leg.mark_price
    return None


def _risk_from_view(
    view: ReadonlyAccountView,
    health: Direction2HealthFact,
) -> AccountRiskSnapshot | None:
    account = view.account_snapshot
    if account is None:
        return None
    equity = account.total_margin_balance
    if equity <= 0:
        equity = account.total_wallet_balance + account.total_unrealized_pnl
    if equity <= 0:
        return None

    maintenance_total = max(account.total_maintenance_margin, Decimal("0"))
    position_facts = tuple(item for item in view.positions if item.quantity != 0)
    total_notional = sum((abs(item.quantity) * item.mark_price for item in position_facts), Decimal("0"))
    positions: list[PositionRiskLeg] = []
    for item in position_facts:
        quantity = abs(item.quantity)
        mark = item.mark_price if item.mark_price > 0 else item.entry_price
        if quantity <= 0 or mark <= 0:
            continue
        maintenance = (
            maintenance_total * (quantity * mark / total_notional)
            if total_notional > 0
            else Decimal("0")
        )
        positions.append(
            PositionRiskLeg(
                account_id=item.account_id,
                symbol=canonicalize_symbol(item.symbol),
                position_side=_position_side(item.position_side),
                quantity=quantity,
                mark_price=mark,
                leverage=Decimal(max(item.leverage, 1)),
                maintenance_margin=maintenance,
                liquidation_price=(item.liquidation_price if item.liquidation_price and item.liquidation_price > 0 else None),
            )
        )
    normalized_positions = tuple(positions)
    pending: list[PendingOrderRisk] = []
    for item in view.active_orders:
        remaining = max(item.original_quantity - item.cumulative_filled_quantity, Decimal("0"))
        if remaining <= 0:
            continue
        reference = _reference_price(item, normalized_positions)
        if reference is None or reference <= 0:
            continue
        leverage = next(
            (
                leg.leverage
                for leg in normalized_positions
                if leg.symbol == canonicalize_symbol(item.symbol)
                and leg.position_side is _position_side(item.position_side)
            ),
            Decimal("1"),
        )
        pending.append(
            PendingOrderRisk(
                account_id=item.account_id,
                symbol=canonicalize_symbol(item.symbol),
                position_side=_position_side(item.position_side),
                action=_order_action(item),
                remaining_quantity=remaining,
                reference_price=reference,
                leverage=leverage,
            )
        )

    observed_ms = int(view.observed_at.timestamp() * 1000)
    portfolio = build_risk_portfolio(
        account_id=view.account_id,
        equity=equity,
        wallet_balance=max(account.total_wallet_balance, Decimal("0")),
        available_balance=max(account.total_available_balance, Decimal("0")),
        positions=normalized_positions,
        pending_orders=tuple(pending),
        initial_margin=max(account.total_initial_margin, Decimal("0")),
        maintenance_margin=maintenance_total,
        risk_data_valid=health.rest_fresh and health.configuration_valid,
        observed_at_ms=observed_ms,
        source_version=max(view.revision, 0),
        exchange_time_ms=observed_ms,
        strict_completeness=False,
    )
    return portfolio.account


def build_central_projection(
    account_view: ReadonlyAccountView,
    snapshot: ReadonlyRuntimeSnapshot,
) -> CentralRuntimeProjection:
    health = snapshot.direction2_health
    positions = tuple(
        sorted(
            (
                PositionRecord(
                    symbol=canonicalize_symbol(item.symbol),
                    position_side=item.position_side,
                    amount=abs(item.quantity),
                    entry_price=item.entry_price,
                    mark_price=item.mark_price,
                    unrealized_pnl=item.unrealized_pnl,
                    leverage=item.leverage,
                    source=item.source,
                    exchange="binance",
                    account_id=item.account_id,
                )
                for item in account_view.positions
                if item.quantity != 0 and str(item.position_side).upper() in {"LONG", "SHORT"}
            ),
            key=lambda item: (item.symbol, item.position_side.value),
        )
    )
    risk = _risk_from_view(account_view, health)
    latest_calibration = max(
        (
            item
            for item in (
                snapshot.last_fast_calibration,
                snapshot.last_full_calibration,
                snapshot.last_reconnect_calibration,
            )
            if item is not None
        ),
        key=lambda item: item.completed_at,
        default=None,
    )
    reconciliation_ok = bool(
        latest_calibration is not None
        and latest_calibration.consistent
        and health.reconciliation_consistent
    )
    stream_state = (
        "CONNECTED"
        if health.stream_connected and health.stream_fresh
        else "STALE"
        if health.stream_connected
        else "DISCONNECTED"
    )
    checks = {
        "common.persistence_healthy": True,
        "exchange.readonly_service_bound": True,
        "exchange.rest_calibrated": health.rest_fresh,
        "exchange.user_stream_fresh": health.stream_connected and health.stream_fresh,
        "exchange.reconciliation_converged": reconciliation_ok,
        "exchange.risk_snapshot_valid": risk is not None and risk.effective_risk_data_valid,
    }
    reasons = tuple(
        dict.fromkeys(
            str(item)
            for item in health.reason_codes
            if str(item) and str(item) != "CONSISTENT"
        )
    )
    if risk is None:
        reasons = (*reasons, "RISK_SNAPSHOT_UNAVAILABLE")
    elif not risk.effective_risk_data_valid:
        reasons = (*reasons, *risk.risk_data_errors, "RISK_DATA_INVALID")
    return CentralRuntimeProjection(
        positions=positions,
        risk=risk,
        reconciliation_status=("HEALTHY" if reconciliation_ok else "DRIFT" if latest_calibration else "UNKNOWN"),
        reconciliation_at=(None if latest_calibration is None else latest_calibration.completed_at),
        reconciliation_details=(
            ()
            if latest_calibration is None
            else (
                f"run_id={latest_calibration.run_id}",
                f"kind={latest_calibration.kind.value}",
                f"diff_count={latest_calibration.diff_count}",
                latest_calibration.reason,
            )
        ),
        stream_state=stream_state,
        stream_last_event_at=health.last_stream_event_at,
        stream_reconnect_count=snapshot.stream_health.reconnect_count,
        checks=checks,
        reasons=tuple(dict.fromkeys(reasons)),
        source_version=str(max(account_view.revision, 0)),
        source_event_time=account_view.observed_at,
        stale=not health.rest_fresh,
    )
