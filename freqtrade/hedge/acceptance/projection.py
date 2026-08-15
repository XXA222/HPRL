from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select

from freqtrade.hedge.acceptance.facts import build_fact_plane
from freqtrade.hedge.acceptance.models import (
    BalanceValue,
    FactPlane,
    FillValue,
    OrderValue,
    PositionValue,
)


def build_memory_plane(repository: Any, runtime_snapshot: Any, *, account_id: str) -> FactPlane:
    account_view = runtime_snapshot.account_view
    if account_view is None:
        raise RuntimeError("readonly runtime has no account_view")
    fills: dict[tuple[str, str, str], Any] = {}
    income_payloads: dict[str, dict[str, Any]] = {}
    for batch in getattr(repository, "batches", ()):
        for fill in batch.fills:
            fills[fill.key] = fill
        for event in batch.account_events:
            payload = dict(event.payload)
            if payload.get("incomeType"):
                income_payloads[event.identity] = payload
    return build_fact_plane(
        account_id=account_id,
        observed_at=account_view.observed_at,
        positions=account_view.positions,
        balances=account_view.balances,
        orders=account_view.active_orders,
        fills=tuple(fills.values()),
        income=tuple(income_payloads.values()),
    )


def build_database_plane(session_factory: Any, *, account_id: str) -> FactPlane:
    from freqtrade.persistence.hedge_models import (
        AccountEvent,
        AuditEvent,
        CurrentOrderProjection,
        FillEvent,
        PositionSnapshot,
    )

    positions: dict[str, PositionValue] = {}
    orders: dict[str, OrderValue] = {}
    fills: dict[str, FillValue] = {}
    balances: dict[str, BalanceValue] = {}
    income_payloads: list[dict[str, Any]] = []
    observed_at = datetime.now(UTC)

    with session_factory() as session:
        position_rows = session.scalars(
            select(PositionSnapshot).where(
                PositionSnapshot.account_id == account_id,
                PositionSnapshot.is_current.is_(True),
            )
        ).all()
        for row in position_rows:
            value = PositionValue(
                account_id=row.account_id,
                symbol=row.symbol,
                position_side=row.position_side,
                quantity=Decimal(row.quantity),
                entry_price=Decimal(row.entry_price),
                unrealized_pnl=Decimal(row.unrealized_pnl),
            )
            positions[value.key] = value

        order_rows = session.scalars(
            select(CurrentOrderProjection).where(
                CurrentOrderProjection.account_id == account_id,
                CurrentOrderProjection.is_terminal.is_(False),
            )
        ).all()
        for row in order_rows:
            value = OrderValue(
                account_id=row.account_id,
                symbol=row.symbol,
                position_side=row.position_side,
                exchange_order_id=row.exchange_order_id,
                client_order_id=row.client_order_id or "",
                status=row.status,
                cumulative_filled_quantity=Decimal(row.executed_quantity),
                active=not row.is_terminal,
            )
            orders[value.key] = value

        fill_rows = session.scalars(
            select(FillEvent).where(FillEvent.account_id == account_id)
        ).all()
        for row in fill_rows:
            value = FillValue(
                account_id=row.account_id,
                symbol=row.symbol,
                position_side=row.position_side,
                exchange_trade_id=row.exchange_trade_id,
                exchange_order_id=row.exchange_order_id,
                quantity=Decimal(row.quantity),
                price=Decimal(row.price),
                commission=Decimal(row.fee_amount),
                realized_pnl=Decimal(row.realized_pnl),
            )
            fills[value.key] = value

        account_events = session.scalars(
            select(AccountEvent).where(AccountEvent.account_id == account_id)
        ).all()
        for row in account_events:
            payload = json.loads(row.raw_payload_json or "{}")
            if isinstance(payload, dict) and payload.get("incomeType"):
                income_payloads.append(payload)

        audit_rows = session.scalars(
            select(AuditEvent)
            .where(
                AuditEvent.account_id == account_id,
                AuditEvent.event_type == "BALANCE_SNAPSHOT",
            )
            .order_by(AuditEvent.occurred_at.asc(), AuditEvent.id.asc())
        ).all()
        for row in audit_rows:
            payload = json.loads(row.payload_json or "{}")
            if not isinstance(payload, dict):
                continue
            asset = str(payload.get("asset") or row.entity_id or "").upper()
            if not asset:
                continue
            value = BalanceValue(
                account_id=account_id,
                asset=asset,
                wallet_balance=Decimal(str(payload.get("wallet_balance") or "0")),
                available_balance=Decimal(str(payload.get("available_balance") or "0")),
                unrealized_pnl=Decimal(str(payload.get("unrealized_pnl") or "0")),
            )
            balances[value.key] = value

    plane = build_fact_plane(
        account_id=account_id,
        observed_at=observed_at,
        income=tuple(income_payloads),
    )
    return FactPlane(
        account_id=account_id,
        observed_at=observed_at,
        positions=positions,
        balances=balances,
        active_orders=orders,
        fills=fills,
        income=plane.income,
    )


def count_unrecovered_unknown_orders(session_factory: Any, *, account_id: str) -> int:
    from sqlalchemy import func

    from freqtrade.persistence.hedge_models import ExecutionOrderStateRow

    with session_factory() as session:
        value = session.scalar(
            select(func.count())
            .select_from(ExecutionOrderStateRow)
            .where(
                ExecutionOrderStateRow.account_id == account_id,
                ExecutionOrderStateRow.lifecycle_status == "UNKNOWN",
            )
        )
        return int(value or 0)
