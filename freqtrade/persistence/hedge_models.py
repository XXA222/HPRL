"""Persistent hedge ledger models.

The ledger is deliberately exchange-agnostic. Monetary and quantity values are
stored as canonical decimal strings so SQLite and PostgreSQL produce identical
results without binary floating-point drift.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from freqtrade.persistence.hedge_contracts import (
    CONTRACTS_VERSION,
    EVENT_VERSION,
    PAYLOAD_VERSION,
    SCHEMA_VERSION,
)


class HedgeModelBase(DeclarativeBase):
    """Isolated metadata registry for the Hedge event ledger.

    Keeping Hedge tables out of Freqtrade's native ``ModelBase`` prevents
    upstream database-copy and schema-count utilities from treating extension
    tables as native core models.
    """


LEDGER_RECORD_VERSION = 3
VALID_POSITION_SIDES = frozenset({"LONG", "SHORT"})
VALID_FACT_SOURCES = frozenset({"REST", "WEBSOCKET", "RECOVERY", "MIGRATION", "LOCAL"})
VALID_ACCOUNT_EVENT_TYPES = frozenset({"FUNDING", "FEE", "BALANCE", "TRANSFER"})
VALID_INTENT_ACTIONS = frozenset({"OPEN", "INCREASE", "REDUCE", "CLOSE", "AMEND", "CANCEL"})


def utcnow() -> datetime:
    """Return a timezone-naive UTC timestamp, matching Freqtrade persistence."""

    return datetime.now(UTC).replace(tzinfo=None)


def canonical_decimal(value: Decimal | str | int | float | None, *, default: str = "0") -> str:
    """Return a finite, non-exponent decimal string.

    Floats are converted through ``str`` to avoid importing their binary tail.
    """

    if value is None:
        value = default
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid decimal value: {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError(f"Decimal value must be finite: {value!r}")
    if parsed == 0:
        return "0"
    rendered = format(parsed, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def decimal_value(value: str | Decimal | int | float | None) -> Decimal:
    """Parse a stored decimal string and reject NaN/Infinity."""

    return Decimal(canonical_decimal(value))


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically for checksums and audit comparison."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def new_event_id() -> str:
    return str(uuid4())


class OrderIntent(HedgeModelBase):
    """A durable, idempotent request to create or amend an exchange order."""

    __tablename__ = "hedge_order_intents"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_hedge_order_intents_idempotency_key"),
        Index(
            "ix_hedge_order_intents_account_symbol_side",
            "exchange",
            "account_id",
            "symbol",
            "position_side",
        ),
        CheckConstraint("position_side IN (\'LONG\', \'SHORT\')", name="ck_hedge_intent_side"),
        CheckConstraint("side IN (\'BUY\', \'SELL\')", name="ck_hedge_intent_order_side"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    intent_id: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True, default=new_event_id
    )
    account_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    position_side: Mapped[str] = mapped_column(String(8), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    order_type: Mapped[str] = mapped_column(String(32), nullable=False)
    time_in_force: Mapped[str | None] = mapped_column(String(16), nullable=True)
    requested_quantity: Mapped[str] = mapped_column(String(80), nullable=False)
    requested_price: Mapped[str | None] = mapped_column(String(80), nullable=True)
    reduce_only: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PLANNED", index=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    strategy_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    strategy_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    decision_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    correlation_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    target_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    approved_quantity: Mapped[str | None] = mapped_column(String(80), nullable=True)
    risk_snapshot_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    reason_codes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    rules_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    approved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    request_payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow, onupdate=utcnow
    )
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class OrderSnapshot(HedgeModelBase):
    """Immutable REST/WebSocket/local fact snapshot for one exchange order."""

    __tablename__ = "hedge_order_snapshots"
    __table_args__ = (
        UniqueConstraint("snapshot_key", name="uq_hedge_order_snapshots_snapshot_key"),
        UniqueConstraint("fact_key", name="uq_hedge_order_snapshots_fact_key"),
        CheckConstraint("position_side IN ('LONG', 'SHORT')", name="ck_hedge_order_side"),
        CheckConstraint("side IN ('BUY', 'SELL')", name="ck_hedge_order_exchange_side"),
        Index(
            "ix_hedge_order_snapshots_lookup",
            "account_id",
            "symbol",
            "position_side",
            "exchange_order_id",
            "source_event_time",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_key: Mapped[str] = mapped_column(String(255), nullable=False)
    fact_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    position_side: Mapped[str] = mapped_column(String(8), nullable=False)
    exchange_order_id: Mapped[str] = mapped_column(String(255), nullable=False)
    client_order_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    intent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("hedge_order_intents.intent_id"), nullable=True
    )
    correlation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    order_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    original_quantity: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    executed_quantity: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    remaining_quantity: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    cumulative_quote: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    average_price: Mapped[str | None] = mapped_column(String(80), nullable=True)
    last_fill_quantity: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    sequence_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload_version: Mapped[int] = mapped_column(Integer, nullable=False, default=PAYLOAD_VERSION)
    source_event_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    is_terminal: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    raw_payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class FillEvent(HedgeModelBase):
    """Immutable exchange fill, deduplicated within exchange/account/symbol."""

    __tablename__ = "hedge_fill_events"
    __table_args__ = (
        UniqueConstraint(
            "exchange",
            "account_id",
            "symbol",
            "exchange_trade_id",
            name="uq_hedge_fill_exchange_account_symbol_trade",
        ),
        CheckConstraint("position_side IN ('LONG', 'SHORT')", name="ck_hedge_fill_side"),
        CheckConstraint("side IN ('BUY', 'SELL')", name="ck_hedge_fill_exchange_side"),
        Index("ix_hedge_fill_projection", "account_id", "symbol", "position_side", "event_time"),
        Index(
            "ix_hedge_fill_order",
            "exchange",
            "account_id",
            "symbol",
            "exchange_order_id",
            "event_time",
        ),
        Index(
            "ix_hedge_fill_sequence",
            "exchange",
            "account_id",
            "symbol",
            "position_side",
            "event_time",
            "sequence_number",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True, default=new_event_id
    )
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    position_side: Mapped[str] = mapped_column(String(8), nullable=False)
    exchange_trade_id: Mapped[str] = mapped_column(String(255), nullable=False)
    exchange_order_id: Mapped[str] = mapped_column(String(255), nullable=False)
    client_order_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    intent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("hedge_order_intents.intent_id"), nullable=True
    )
    correlation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    quantity: Mapped[str] = mapped_column(String(80), nullable=False)
    price: Mapped[str] = mapped_column(String(80), nullable=False)
    quote_quantity: Mapped[str] = mapped_column(String(80), nullable=False)
    fee_amount: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    fee_currency: Mapped[str | None] = mapped_column(String(32), nullable=True)
    realized_pnl: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    liquidity_role: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    sequence_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    projection_status: Mapped[str] = mapped_column(String(16), nullable=False, default="APPLIED")
    projection_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class PositionSnapshot(HedgeModelBase):
    """Immutable position fact/projection snapshot.

    At most one active current row may exist for an account/symbol/side tuple.
    """

    __tablename__ = "hedge_position_snapshots"
    __table_args__ = (
        CheckConstraint("position_side IN ('LONG', 'SHORT')", name="ck_hedge_position_side"),
        UniqueConstraint("snapshot_key", name="uq_hedge_position_snapshots_snapshot_key"),
        UniqueConstraint("fact_key", name="uq_hedge_position_snapshots_fact_key"),
        Index(
            "uq_hedge_position_current_active",
            "exchange",
            "account_id",
            "symbol",
            "position_side",
            unique=True,
            sqlite_where=text("is_current = 1 AND is_active = 1"),
            postgresql_where=text("is_current AND is_active"),
        ),
        Index(
            "uq_hedge_position_current",
            "exchange",
            "account_id",
            "symbol",
            "position_side",
            unique=True,
            sqlite_where=text("is_current = 1"),
            postgresql_where=text("is_current"),
        ),
        Index(
            "ix_hedge_position_history",
            "exchange",
            "account_id",
            "symbol",
            "position_side",
            "source_event_time",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_key: Mapped[str] = mapped_column(String(255), nullable=False)
    fact_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    venue_symbol: Mapped[str | None] = mapped_column(String(128), nullable=True)
    position_side: Mapped[str] = mapped_column(String(8), nullable=False)
    quantity: Mapped[str] = mapped_column(String(80), nullable=False)
    entry_price: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    mark_price: Mapped[str | None] = mapped_column(String(80), nullable=True)
    notional: Mapped[str | None] = mapped_column(String(80), nullable=True)
    realized_pnl: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    unrealized_pnl: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    liquidation_price: Mapped[str | None] = mapped_column(String(80), nullable=True)
    leverage: Mapped[str] = mapped_column(String(80), nullable=False, default="1")
    margin_mode: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    sequence_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    risk_source_snapshot_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_event_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_current: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    raw_payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class AccountRiskSnapshot(HedgeModelBase):
    """Immutable account-wide risk snapshot."""

    __tablename__ = "hedge_account_risk_snapshots"
    __table_args__ = (
        UniqueConstraint("snapshot_key", name="uq_hedge_account_risk_snapshot_key"),
        UniqueConstraint("fact_key", name="uq_hedge_account_risk_fact_key"),
        Index("ix_hedge_account_risk_time", "account_id", "source_event_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    snapshot_key: Mapped[str] = mapped_column(String(255), nullable=False)
    fact_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    equity: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    wallet_balance: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    available_balance: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    margin_balance: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    total_initial_margin: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    total_maintenance_margin: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    gross_long_notional: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    gross_short_notional: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    gross_exposure: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    net_exposure: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    pending_risk: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    margin_utilization: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    liquidation_buffer: Mapped[str | None] = mapped_column(String(80), nullable=True)
    risk_state: Mapped[str] = mapped_column(String(16), nullable=False, default="HALT")
    risk_data_valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_snapshot_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    rules_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reason_codes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    projected_risk_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    source_event_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    raw_payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class AccountEvent(HedgeModelBase):
    """Funding, fee, balance and transfer event."""

    __tablename__ = "hedge_account_events"
    __table_args__ = (
        UniqueConstraint("event_key", name="uq_hedge_account_events_event_key"),
        UniqueConstraint("fact_key", name="uq_hedge_account_events_fact_key"),
        Index("ix_hedge_account_events_time", "account_id", "event_time"),
        Index("ix_hedge_account_events_symbol", "account_id", "symbol", "event_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True, default=new_event_id
    )
    event_key: Mapped[str] = mapped_column(String(255), nullable=False)
    fact_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(16), nullable=False)
    asset: Mapped[str] = mapped_column(String(32), nullable=False)
    amount: Mapped[str] = mapped_column(String(80), nullable=False)
    balance_after: Mapped[str | None] = mapped_column(String(80), nullable=True)
    symbol: Mapped[str | None] = mapped_column(String(128), nullable=True)
    position_side: Mapped[str | None] = mapped_column(String(8), nullable=True)
    related_order_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    related_trade_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    transfer_direction: Mapped[str | None] = mapped_column(String(16), nullable=True)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    raw_payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class ReconciliationRun(HedgeModelBase):
    """One reconciliation pass across local and exchange facts."""

    __tablename__ = "hedge_reconciliation_runs"
    __table_args__ = (Index("ix_hedge_reconciliation_account_started", "account_id", "started_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True, default=new_event_id
    )
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    scope: Mapped[str] = mapped_column(String(32), nullable=False, default="FULL")
    trigger: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="RUNNING")
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    diff_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    severe_diff_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    automatic_repairs: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    summary_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class ReconciliationDiff(HedgeModelBase):
    """A single typed discrepancy found during reconciliation."""

    __tablename__ = "hedge_reconciliation_diffs"
    __table_args__ = (
        UniqueConstraint("run_id", "diff_key", name="uq_hedge_reconciliation_run_diff_key"),
        Index("ix_hedge_reconciliation_diff_run", "run_id", "severity"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("hedge_reconciliation_runs.run_id"), nullable=False
    )
    diff_key: Mapped[str] = mapped_column(String(255), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_key: Mapped[str] = mapped_column(String(255), nullable=False)
    field_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    local_value_json: Mapped[str] = mapped_column(Text, nullable=False, default="null")
    exchange_value_json: Mapped[str] = mapped_column(Text, nullable=False, default="null")
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    resolution: Mapped[str] = mapped_column(String(32), nullable=False, default="UNRESOLVED")
    repair_action: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class StrategySideState(HedgeModelBase):
    """Durable strategy state isolated by account, symbol and hedge side."""

    __tablename__ = "hedge_strategy_side_states"
    __table_args__ = (
        UniqueConstraint(
            "exchange",
            "account_id",
            "symbol",
            "position_side",
            "strategy_name",
            name="uq_hedge_strategy_side_state",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    position_side: Mapped[str] = mapped_column(String(8), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(128), nullable=False)
    state_name: Mapped[str] = mapped_column(String(64), nullable=False, default="IDLE")
    state_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    last_decision_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow, onupdate=utcnow
    )
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class EventOutbox(HedgeModelBase):
    """Transactional outbox row. It only becomes visible after DB commit."""

    __tablename__ = "hedge_event_outbox"
    __table_args__ = (
        UniqueConstraint("event_id", name="uq_hedge_event_outbox_event_id"),
        Index("ix_hedge_event_outbox_pending", "status", "available_at", "id"),
        UniqueConstraint(
            "aggregate_type",
            "aggregate_id",
            "aggregate_sequence",
            name="uq_hedge_outbox_aggregate_sequence",
        ),
        Index("ix_hedge_event_outbox_aggregate", "aggregate_type", "aggregate_id", "id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    event_id: Mapped[str] = mapped_column(String(36), nullable=False, default=new_event_id)
    aggregate_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(255), nullable=False)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    aggregate_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    correlation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    causation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload_version: Mapped[int] = mapped_column(Integer, nullable=False, default=PAYLOAD_VERSION)
    event_version: Mapped[int] = mapped_column(Integer, nullable=False, default=EVENT_VERSION)
    contracts_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default=CONTRACTS_VERSION
    )
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False, default=SCHEMA_VERSION)
    exchange_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    observed_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    headers_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    available_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    lock_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    resolution_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    resolved_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    resolution_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow, onupdate=utcnow
    )
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class CurrentOrderProjection(HedgeModelBase):
    """Mutable latest order projection rebuilt from immutable order/fill facts."""

    __tablename__ = "hedge_current_orders"
    __table_args__ = (
        CheckConstraint("position_side IN ('LONG', 'SHORT')", name="ck_hedge_current_order_side"),
        CheckConstraint("side IN ('BUY', 'SELL')", name="ck_hedge_current_order_exchange_side"),
        UniqueConstraint(
            "exchange",
            "account_id",
            "symbol",
            "position_side",
            "exchange_order_id",
            name="uq_hedge_current_order_identity",
        ),
        Index("ix_hedge_current_order_status", "account_id", "status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    position_side: Mapped[str] = mapped_column(String(8), nullable=False)
    exchange_order_id: Mapped[str] = mapped_column(String(255), nullable=False)
    client_order_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    intent_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    side: Mapped[str] = mapped_column(String(8), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    original_quantity: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    executed_quantity: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    remaining_quantity: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    cumulative_quote: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    average_price: Mapped[str | None] = mapped_column(String(80), nullable=True)
    is_terminal: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    source_snapshot_key: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_event_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    sequence_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow, onupdate=utcnow
    )
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class TargetPosition(HedgeModelBase):
    __tablename__ = "hedge_target_positions"
    __table_args__ = (
        CheckConstraint("position_side IN ('LONG', 'SHORT')", name="ck_hedge_target_side"),
        UniqueConstraint(
            "exchange", "account_id", "symbol", "position_side", "strategy_id", "cycle_id",
            name="uq_hedge_target_position_cycle",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    target_id: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True, default=new_event_id
    )
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    position_side: Mapped[str] = mapped_column(String(8), nullable=False)
    target_quantity: Mapped[str] = mapped_column(String(80), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(128), nullable=False)
    cycle_id: Mapped[str] = mapped_column(String(128), nullable=False)
    correlation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    target_payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class CorePositionState(HedgeModelBase):
    __tablename__ = "hedge_core_position_states"
    __table_args__ = (
        CheckConstraint("position_side IN ('LONG', 'SHORT')", name="ck_hedge_core_side"),
        UniqueConstraint(
            "exchange", "account_id", "symbol", "position_side",
            name="uq_hedge_core_position_state",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    position_side: Mapped[str] = mapped_column(String(8), nullable=False)
    core_quantity: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    core_floor: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    effective_cost: Mapped[str | None] = mapped_column(String(80), nullable=True)
    realized_profit_credit: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    state_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        nullable=False,
        default=utcnow,
        onupdate=utcnow,
    )
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class TacticalLot(HedgeModelBase):
    __tablename__ = "hedge_tactical_lots"
    __table_args__ = (
        UniqueConstraint("lot_id", name="uq_hedge_tactical_lot_id"),
        CheckConstraint(
            "position_side IN ('LONG', 'SHORT')",
            name="ck_hedge_tactical_side",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    lot_id: Mapped[str] = mapped_column(String(36), nullable=False, default=new_event_id)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    position_side: Mapped[str] = mapped_column(String(8), nullable=False)
    strategy_name: Mapped[str] = mapped_column(String(128), nullable=False)
    lot_type: Mapped[str] = mapped_column(String(32), nullable=False)
    quantity: Mapped[str] = mapped_column(String(80), nullable=False)
    entry_price: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="OPEN")
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class ExecutionOrderStateRow(HedgeModelBase):
    """Authoritative current state for one execution order.

    The immutable intent/fill/audit rows remain the evidence ledger.  This table
    is the restart-safe current projection consumed by ``ExecutionStorePort``.
    """

    __tablename__ = "hedge_execution_order_states"
    __table_args__ = (
        UniqueConstraint("client_order_id", name="uq_hedge_execution_order_client_id"),
        UniqueConstraint("intent_id", name="uq_hedge_execution_order_intent_id"),
        UniqueConstraint(
            "idempotency_key", name="uq_hedge_execution_order_idempotency_key"
        ),
        Index(
            "ix_hedge_execution_order_leg_state",
            "account_id",
            "symbol",
            "position_side",
            "lifecycle_status",
        ),
        CheckConstraint(
            "position_side IN ('LONG', 'SHORT')",
            name="ck_hedge_execution_order_position_side",
        ),
        CheckConstraint(
            "lifecycle_version >= 0",
            name="ck_hedge_execution_order_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_order_id: Mapped[str] = mapped_column(String(256), nullable=False)
    intent_id: Mapped[str] = mapped_column(String(36), nullable=False)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False, default="binance")
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    position_side: Mapped[str] = mapped_column(String(8), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    order_type: Mapped[str] = mapped_column(String(32), nullable=False)
    quantity: Mapped[str] = mapped_column(String(80), nullable=False)
    limit_price: Mapped[str | None] = mapped_column(String(80), nullable=True)
    reduce_only: Mapped[bool] = mapped_column(Boolean, nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    action_group_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    approved_quantity: Mapped[str] = mapped_column(String(80), nullable=False)
    risk_reason_codes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    lifecycle_status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    lifecycle_filled_quantity: Mapped[str] = mapped_column(String(80), nullable=False)
    lifecycle_average_price: Mapped[str | None] = mapped_column(String(80), nullable=True)
    lifecycle_exchange_order_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    lifecycle_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    lifecycle_version: Mapped[int] = mapped_column(Integer, nullable=False)
    approved_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    lifecycle_updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow, onupdate=utcnow
    )
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class ExecutionIdempotencyRow(HedgeModelBase):
    """Restart-safe reservation/result pointer for execution idempotency."""

    __tablename__ = "hedge_execution_idempotency"
    __table_args__ = (
        CheckConstraint(
            "state IN ('IN_FLIGHT', 'COMPLETED')",
            name="ck_hedge_execution_idempotency_state",
        ),
        CheckConstraint(
            "(state = 'IN_FLIGHT' AND client_order_id IS NULL "
            "AND lease_owner IS NOT NULL AND lease_expires_at IS NOT NULL) OR "
            "(state = 'COMPLETED' AND client_order_id IS NOT NULL "
            "AND lease_owner IS NULL AND lease_expires_at IS NULL)",
            name="ck_hedge_execution_idempotency_shape",
        ),
        Index("ix_hedge_execution_idempotency_lease", "state", "lease_expires_at"),
    )

    idempotency_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    client_order_id: Mapped[str | None] = mapped_column(
        String(256),
        ForeignKey(
            "hedge_execution_order_states.client_order_id",
            name="fk_hedge_execution_idempotency_order",
            ondelete="RESTRICT",
            deferrable=True,
            initially="DEFERRED",
        ),
        nullable=True,
    )
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow, onupdate=utcnow
    )
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class PaperRuntimeCheckpointRow(HedgeModelBase):
    """Latest durable Paper/Shadow runtime checkpoint for one source namespace."""

    __tablename__ = "hedge_paper_runtime_checkpoints"
    __table_args__ = (
        UniqueConstraint(
            "exchange",
            "account_id",
            "symbol",
            "source",
            name="uq_hedge_paper_checkpoint_identity",
        ),
        CheckConstraint(
            "source IN ('PAPER', 'SHADOW')",
            name="ck_hedge_paper_checkpoint_source",
        ),
        Index(
            "ix_hedge_paper_checkpoint_updated",
            "account_id",
            "updated_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    state_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow, onupdate=utcnow
    )
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class ActionGroupRow(HedgeModelBase):
    """Durable state for a recoverable multi-leg execution action group."""

    __tablename__ = "hedge_action_groups"
    __table_args__ = (
        UniqueConstraint("action_group_id", name="uq_hedge_action_group_id"),
        Index(
            "ix_hedge_action_group_account_symbol",
            "account_id",
            "symbol",
            "updated_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    action_group_id: Mapped[str] = mapped_column(String(36), nullable=False)
    action_type: Mapped[str] = mapped_column(String(64), nullable=False)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    symbol: Mapped[str] = mapped_column(String(128), nullable=False)
    members_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class RiskApprovalCommitRow(HedgeModelBase):
    """Durable ownership transfer for an approved risk reservation."""

    __tablename__ = "hedge_risk_approval_commits"
    __table_args__ = (
        UniqueConstraint(
            "decision_id",
            name="uq_hedge_risk_approval_commit_decision",
        ),
        UniqueConstraint(
            "idempotency_key",
            name="uq_hedge_risk_approval_commit_idempotency",
        ),
        Index(
            "ix_hedge_risk_approval_commit_correlation",
            "correlation_id",
            "committed_at_ms",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    decision_id: Mapped[str] = mapped_column(String(128), nullable=False)
    intent_id: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    correlation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    risk_snapshot_id: Mapped[str] = mapped_column(String(255), nullable=False)
    request_json: Mapped[str] = mapped_column(Text, nullable=False)
    risk_snapshot_json: Mapped[str] = mapped_column(Text, nullable=False)
    rules_version: Mapped[str] = mapped_column(String(64), nullable=False)
    fencing_token: Mapped[int] = mapped_column(Integer, nullable=False)
    approved_quantity: Mapped[str] = mapped_column(String(80), nullable=False)
    approved_notional: Mapped[str] = mapped_column(String(80), nullable=False)
    evaluated_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    committed_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    durable_reference: Mapped[str] = mapped_column(String(255), nullable=False)
    target_snapshot_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    intent_expires_at_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class ExecutionDailyBudgetRow(HedgeModelBase):
    """Authoritative daily execution budget stored with the execution ledger."""

    __tablename__ = "hedge_r5_daily_budgets"
    __table_args__ = (
        UniqueConstraint("account_id", "utc_date", name="uq_hedge_r5_daily_budget_scope"),
        Index("ix_hedge_r5_daily_budget_scope", "account_id", "utc_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    utc_date: Mapped[str] = mapped_column(String(10), nullable=False)
    orders: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    turnover: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    realized_loss: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    gross_peak: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    net_peak: Mapped[str] = mapped_column(String(80), nullable=False, default="0")
    open_orders_peak: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=utcnow, onupdate=utcnow)
    record_version: Mapped[int] = mapped_column(Integer, nullable=False, default=LEDGER_RECORD_VERSION)


class ExecutionBudgetReservationRow(HedgeModelBase):
    """PREPARED/CONFIRMED/RELEASED reservation for a complete write batch."""

    __tablename__ = "hedge_r5_budget_reservations"
    __table_args__ = (
        CheckConstraint("state IN ('PREPARED','CONFIRMED','RELEASED')", name="ck_hedge_r5_budget_reservation_state"),
        Index("ix_hedge_r5_budget_reservation_scope", "account_id", "utc_date", "state"),
    )

    reservation_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    utc_date: Mapped[str] = mapped_column(String(10), nullable=False)
    orders: Mapped[int] = mapped_column(Integer, nullable=False)
    turnover: Mapped[str] = mapped_column(String(80), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="PREPARED")
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=utcnow, onupdate=utcnow)
    record_version: Mapped[int] = mapped_column(Integer, nullable=False, default=LEDGER_RECORD_VERSION)


class ExecutionIncomeEventRow(HedgeModelBase):
    """Idempotent Binance income fact used by the authoritative daily-loss gate."""

    __tablename__ = "hedge_r5_income_events"
    __table_args__ = (
        UniqueConstraint("exchange", "account_id", "external_event_id", name="uq_hedge_r5_income_event"),
        Index("ix_hedge_r5_income_scope", "account_id", "occurred_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False, default="binance")
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    income_type: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(128), nullable=True)
    asset: Mapped[str | None] = mapped_column(String(32), nullable=True)
    amount: Mapped[str] = mapped_column(String(80), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=utcnow)
    record_version: Mapped[int] = mapped_column(Integer, nullable=False, default=LEDGER_RECORD_VERSION)


class ControlOperationRow(HedgeModelBase):
    """Durable idempotency and result record for dangerous control operations."""

    __tablename__ = "hedge_control_operations"
    __table_args__ = (
        UniqueConstraint(
            "account_id",
            "idempotency_key",
            name="uq_hedge_control_operation_idempotency",
        ),
        UniqueConstraint(
            "operation_id",
            name="uq_hedge_control_operation_id",
        ),
        Index(
            "ix_hedge_control_operation_account_time",
            "account_id",
            "created_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    operation_id: Mapped[str] = mapped_column(String(36), nullable=False)
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    symbol: Mapped[str | None] = mapped_column(String(128), nullable=True)
    actor: Mapped[str] = mapped_column(String(128), nullable=False)
    actor_role: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    request_json: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(String(32), nullable=False)
    outcome: Mapped[str | None] = mapped_column(String(32), nullable=True)
    outcome_code: Mapped[str | None] = mapped_column(String(128), nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=False), nullable=True
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class AuditEvent(HedgeModelBase):
    __tablename__ = "hedge_audit_events"
    __table_args__ = (Index("ix_hedge_audit_account_time", "account_id", "occurred_at"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    audit_id: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True, default=new_event_id
    )
    account_id: Mapped[str] = mapped_column(String(128), nullable=False)
    exchange: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(128), nullable=False)
    entity_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    entity_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    severity: Mapped[str] = mapped_column(String(16), nullable=False, default="INFO")
    reason_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    correlation_id: Mapped[str] = mapped_column(String(128), nullable=False)
    actor: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), nullable=False, default=utcnow
    )
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


class SchemaMigrationRecord(HedgeModelBase):
    """Durable migration journal used for idempotence and interrupted recovery."""

    __tablename__ = "hedge_schema_migrations"

    migration_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="PENDING")
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    failed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    backup_reference: Mapped[str | None] = mapped_column(Text, nullable=True)
    conflict_report_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    details_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    runner_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=LEDGER_RECORD_VERSION
    )


HEDGE_MODEL_CLASSES = (
    OrderIntent,
    OrderSnapshot,
    FillEvent,
    PositionSnapshot,
    AccountRiskSnapshot,
    AccountEvent,
    ReconciliationRun,
    ReconciliationDiff,
    StrategySideState,
    EventOutbox,
    CurrentOrderProjection,
    TargetPosition,
    CorePositionState,
    TacticalLot,
    ExecutionOrderStateRow,
    ExecutionIdempotencyRow,
    PaperRuntimeCheckpointRow,
    ActionGroupRow,
    RiskApprovalCommitRow,
    ExecutionDailyBudgetRow,
    ExecutionBudgetReservationRow,
    ExecutionIncomeEventRow,
    ControlOperationRow,
    AuditEvent,
    SchemaMigrationRecord,
)


def create_hedge_tables(engine: Engine) -> None:
    """Create only hedge-ledger tables and indexes."""

    HedgeModelBase.metadata.create_all(
        engine, tables=[model.__table__ for model in HEDGE_MODEL_CLASSES]
    )
