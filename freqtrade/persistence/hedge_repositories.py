"""Repositories for the hedge event ledger and restart projections."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any

from sqlalchemy import Select, delete, func, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from freqtrade.persistence.hedge_contracts import (
    EventMetadata,
    TERMINAL_ORDER_STATUSES,
    canonical_symbol,
    normalize_order_status,
    stable_fact_key,
)
from freqtrade.persistence.hedge_models import (
    AccountEvent,
    AccountRiskSnapshot,
    AuditEvent,
    CorePositionState,
    CurrentOrderProjection,
    EventOutbox,
    FillEvent,
    OrderIntent,
    OrderSnapshot,
    PositionSnapshot,
    ReconciliationDiff,
    ReconciliationRun,
    StrategySideState,
    TacticalLot,
    TargetPosition,
    VALID_ACCOUNT_EVENT_TYPES,
    VALID_FACT_SOURCES,
    VALID_INTENT_ACTIONS,
    canonical_decimal,
    canonical_json,
    decimal_value,
    new_event_id,
    utcnow,
)


@dataclass(frozen=True)
class FillApplyResult:
    accepted: bool
    duplicate: bool
    fill_event_id: str
    action: str
    position_quantity: str
    position_entry_price: str
    cumulative_order_quantity: str
    cumulative_order_quote: str
    order_status: str | None = None
    projection_blocked: bool = False
    reason_code: str | None = None


@dataclass(frozen=True)
class RecoveredOrder:
    account_id: str
    exchange: str
    symbol: str
    position_side: str
    exchange_order_id: str
    client_order_id: str | None
    action: str | None
    status: str
    original_quantity: str
    executed_quantity: str
    remaining_quantity: str
    cumulative_quote: str
    average_price: str | None
    is_terminal: bool
    source_event_time: datetime


@dataclass(frozen=True)
class RecoveredPosition:
    account_id: str
    exchange: str
    symbol: str
    position_side: str
    quantity: str
    entry_price: str
    realized_pnl: str
    mark_price: str | None
    notional: str | None
    unrealized_pnl: str
    liquidation_price: str | None
    leverage: str
    margin_mode: str | None
    risk_source_snapshot_key: str | None
    last_event_time: datetime
    last_sequence_number: int | None = None


@dataclass(frozen=True)
class LedgerRecovery:
    orders: tuple[RecoveredOrder, ...] = field(default_factory=tuple)
    positions: tuple[RecoveredPosition, ...] = field(default_factory=tuple)


class ProjectionInvariantError(ValueError):
    """A persisted exchange fact cannot be safely applied to a local projection."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str,
        entity_key: str,
        offending_event_id: str | None = None,
    ):
        super().__init__(message)
        self.reason_code = reason_code
        self.entity_key = entity_key
        self.offending_event_id = offending_event_id


def _normalize_side(value: str) -> str:
    normalized = value.upper()
    if normalized not in {"LONG", "SHORT"}:
        raise ValueError(f"Unsupported position side: {value!r}")
    return normalized


def _normalize_order_side(value: str) -> str:
    normalized = value.upper()
    if normalized not in {"BUY", "SELL"}:
        raise ValueError(f"Unsupported order side: {value!r}")
    return normalized


def _normalize_source(value: str) -> str:
    normalized = value.upper()
    if normalized not in VALID_FACT_SOURCES:
        raise ValueError(f"Unsupported fact source: {value!r}")
    return normalized



VALID_INTENT_TRANSITIONS: dict[str, frozenset[str]] = {
    "PLANNED": frozenset({"APPROVED", "REJECTED", "CANCELED", "EXPIRED"}),
    "APPROVED": frozenset({"PREPARED", "REJECTED", "CANCELED", "EXPIRED"}),
    "PREPARED": frozenset({"SUBMITTING", "CANCELED", "EXPIRED"}),
    "SUBMITTING": frozenset({"ACKNOWLEDGED", "UNKNOWN", "REJECTED", "CANCELED"}),
    "ACKNOWLEDGED": frozenset({"PARTIAL", "FILLED", "UNKNOWN", "CANCELED"}),
    "PARTIAL": frozenset({"PARTIAL", "FILLED", "UNKNOWN", "CANCELED"}),
    "UNKNOWN": frozenset({"ACKNOWLEDGED", "PARTIAL", "FILLED", "CANCELED", "REJECTED"}),
    "REJECTED": frozenset(),
    "FILLED": frozenset(),
    "CANCELED": frozenset(),
    "EXPIRED": frozenset(),
}


def _normalize_action(value: str) -> str:
    normalized = value.upper()
    if normalized not in VALID_INTENT_ACTIONS:
        raise ValueError(f"Unsupported intent action: {value!r}")
    return normalized


def _default_fill_action(position_side: str, order_side: str) -> str:
    return "INCREASE" if _signed_quantity(position_side, order_side, Decimal("1")) > 0 else "REDUCE"


def _validate_action_direction(
    position_side: str,
    order_side: str,
    action: str,
    *,
    reduce_only: bool | None = None,
) -> None:
    increases_position = (
        _signed_quantity(position_side, order_side, Decimal("1")) > 0
    )
    if action in {"OPEN", "INCREASE"} and not increases_position:
        raise ValueError(f"{action} action has reducing order direction")
    if action in {"REDUCE", "CLOSE"} and increases_position:
        raise ValueError(f"{action} action has increasing order direction")
    if reduce_only is True and increases_position:
        raise ValueError("Reduce-only order intent cannot increase a position")
    if reduce_only is False and action in {"REDUCE", "CLOSE"}:
        raise ValueError(f"{action} order intent must be reduce-only")


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _required_key(name: str, value: str, *, max_length: int = 255) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    if len(normalized) > max_length:
        raise ValueError(f"{name} must not exceed {max_length} characters")
    return normalized


def _bounded_key(prefix: str, *parts: object, max_length: int = 255) -> str:
    raw = "|".join((prefix, *(str(part) for part in parts)))
    if len(raw) <= max_length:
        return raw
    digest = sha256(raw.encode("utf-8")).hexdigest()
    compact = f"{prefix}:{digest}"
    return compact if len(compact) <= max_length else digest[:max_length]


def _signed_quantity(position_side: str, order_side: str, quantity: Decimal) -> Decimal:
    if position_side == "LONG":
        return quantity if order_side == "BUY" else -quantity
    return quantity if order_side == "SELL" else -quantity


def _position_after_fill(
    old_quantity: Decimal,
    old_entry_price: Decimal,
    position_side: str,
    order_side: str,
    fill_quantity: Decimal,
    fill_price: Decimal,
) -> tuple[Decimal, Decimal]:
    delta = _signed_quantity(position_side, order_side, fill_quantity)
    raw_quantity = old_quantity + delta
    new_quantity = max(Decimal("0"), raw_quantity)
    if delta > 0:
        old_cost = old_quantity * old_entry_price
        new_cost = fill_quantity * fill_price
        entry_price = (old_cost + new_cost) / new_quantity if new_quantity else Decimal("0")
    elif new_quantity == 0:
        entry_price = Decimal("0")
    else:
        entry_price = old_entry_price
    return new_quantity, entry_price


def _immutable_mismatches(existing: Any, expected: dict[str, Any]) -> list[str]:
    return [name for name, value in expected.items() if getattr(existing, name) != value]


def _raise_immutable_conflict(label: str, key: str, mismatches: list[str]) -> None:
    raise ValueError(
        f"{label} conflict for key {key!r}; changed fields: {', '.join(mismatches)}"
    )


class HedgeLedgerRepository:
    """Unit-of-work repository.

    Methods flush but do not commit. The caller controls the transaction, which
    ensures ledger rows, projections and outbox messages commit or roll back
    together.
    """

    def __init__(self, session: Session):
        self.session = session

    def _insert_immutable(
        self,
        row: Any,
        find_existing: Any,
        verify_existing: Any,
    ) -> Any | None:
        """Flush an immutable row without breaking SQLite outer rollback semantics."""

        if self.session.get_bind().dialect.name != "postgresql":
            self.session.add(row)
            self.session.flush([row])
            return None
        try:
            with self.session.begin_nested():
                self.session.add(row)
                self.session.flush([row])
        except IntegrityError:
            existing = find_existing()
            if existing is None:
                raise
            verify_existing(existing)
            return existing
        return None

    def create_order_intent(
        self,
        *,
        account_id: str,
        exchange: str,
        symbol: str,
        position_side: str,
        action: str,
        side: str,
        order_type: str,
        requested_quantity: Decimal | str | int | float,
        idempotency_key: str,
        requested_price: Decimal | str | int | float | None = None,
        time_in_force: str | None = None,
        reduce_only: bool = False,
        status: str = "PLANNED",
        strategy_name: str | None = None,
        strategy_version: str | None = None,
        decision_id: str | None = None,
        correlation_id: str | None = None,
        target_snapshot: dict[str, Any] | None = None,
        expires_at: datetime | None = None,
        payload: dict[str, Any] | None = None,
    ) -> tuple[OrderIntent, bool]:
        account_id = _required_key("Account id", account_id, max_length=128)
        exchange = _required_key("Exchange", exchange, max_length=64).lower()
        symbol = canonical_symbol(symbol)
        normalized_key = _required_key("Order intent idempotency key", idempotency_key)
        normalized_position_side = _normalize_side(position_side)
        normalized_order_side = _normalize_order_side(side)
        normalized_quantity = decimal_value(requested_quantity)
        normalized_price = decimal_value(requested_price) if requested_price is not None else None
        normalized_status = normalize_order_status(status)
        normalized_correlation = _required_key(
            "Correlation id", correlation_id or normalized_key, max_length=128
        )
        normalized_expires = _naive_utc(expires_at) if expires_at else None
        if normalized_quantity <= 0:
            raise ValueError("Order intent quantity must be positive")
        if normalized_price is not None and normalized_price <= 0:
            raise ValueError("Order intent price must be positive")
        if normalized_expires is not None and normalized_expires <= utcnow():
            raise ValueError("Order intent expires_at must be in the future")

        normalized_action = _normalize_action(action)
        _validate_action_direction(
            normalized_position_side,
            normalized_order_side,
            normalized_action,
            reduce_only=reduce_only,
        )
        expected = {
            "account_id": account_id,
            "exchange": exchange,
            "symbol": symbol,
            "position_side": normalized_position_side,
            "action": normalized_action,
            "side": normalized_order_side,
            "order_type": order_type.upper(),
            "requested_quantity": canonical_decimal(normalized_quantity),
            "requested_price": (
                canonical_decimal(normalized_price) if normalized_price is not None else None
            ),
            "time_in_force": time_in_force,
            "reduce_only": reduce_only,
            "request_payload_json": canonical_json(payload or {}),
            "correlation_id": normalized_correlation,
            "target_snapshot_json": canonical_json(target_snapshot or {}),
            "expires_at": normalized_expires,
        }

        def find_existing() -> OrderIntent | None:
            return self.session.scalar(
                select(OrderIntent).where(OrderIntent.idempotency_key == normalized_key)
            )

        def verify_existing(existing: OrderIntent) -> None:
            mismatches = _immutable_mismatches(existing, expected)
            if mismatches:
                _raise_immutable_conflict("Order intent idempotency", normalized_key, mismatches)

        existing = find_existing()
        if existing is not None:
            verify_existing(existing)
            return existing, False

        intent_id = new_event_id()
        intent = OrderIntent(
            intent_id=intent_id,
            **expected,
            status=normalized_status,
            idempotency_key=normalized_key,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            decision_id=decision_id,
        )
        existing = self._insert_immutable(intent, find_existing, verify_existing)
        if existing is not None:
            return existing, False

        self._enqueue(
            aggregate_type="OrderIntent",
            aggregate_id=intent_id,
            event_type="hedge.order_intent.created",
            payload={
                "intent_id": intent_id,
                "account_id": account_id,
                "exchange": exchange,
                "symbol": symbol,
                "position_side": normalized_position_side,
                "idempotency_key": normalized_key,
            },
            metadata=EventMetadata(correlation_id=normalized_correlation),
        )
        self.session.flush()
        return intent, True

    def transition_order_intent(
        self,
        *,
        intent_id: str,
        new_status: str,
        expected_revision: int | None = None,
        approved_by: str | None = None,
        approved_quantity: Decimal | str | int | float | None = None,
        risk_snapshot_id: str | None = None,
        reason_codes: Iterable[str] | None = None,
        rules_version: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> OrderIntent:
        normalized_id = _required_key("Order intent id", intent_id, max_length=36)
        target = normalize_order_status(new_status)
        row = self.session.scalar(
            select(OrderIntent).where(OrderIntent.intent_id == normalized_id)
        )
        if row is None:
            raise KeyError(f"Order intent not found: {normalized_id}")
        current = normalize_order_status(row.status)
        if row.expires_at is not None and row.expires_at <= utcnow() and target != "EXPIRED":
            raise ValueError("Expired order intent can only transition to EXPIRED")
        if target == current:
            return row
        allowed = VALID_INTENT_TRANSITIONS.get(current)
        if allowed is None or target not in allowed:
            raise ValueError(f"Invalid order intent transition: {current} -> {target}")
        compare_revision = row.revision if expected_revision is None else expected_revision
        values: dict[str, Any] = {
            "status": target,
            "revision": compare_revision + 1,
            "updated_at": utcnow(),
            "error_code": error_code,
            "error_message": error_message,
        }
        if target == "APPROVED":
            approved_quantity = (
                row.requested_quantity if approved_quantity is None else approved_quantity
            )
            risk_snapshot_id = risk_snapshot_id or "legacy-unbound-risk"
            rules_version = rules_version or "legacy-approval-v0"
            reason_codes = tuple(reason_codes or ())
            if not reason_codes and risk_snapshot_id == "legacy-unbound-risk":
                reason_codes = ("LEGACY_APPROVAL",)
            approved = decimal_value(approved_quantity)
            requested = decimal_value(row.requested_quantity)
            if approved <= 0 or approved > requested:
                raise ValueError("Approved quantity must be positive and <= requested quantity")
            values.update(
                {
                    "approved_by": approved_by,
                    "approved_at": utcnow(),
                    "approved_quantity": canonical_decimal(approved),
                    "risk_snapshot_id": risk_snapshot_id,
                    "reason_codes_json": canonical_json(sorted(set(reason_codes or ()))),
                    "rules_version": rules_version,
                }
            )
        result = self.session.execute(
            update(OrderIntent)
            .where(
                OrderIntent.intent_id == normalized_id,
                OrderIntent.revision == compare_revision,
                OrderIntent.status == row.status,
            )
            .values(**values)
        )
        if result.rowcount != 1:
            raise ValueError(f"Order intent revision mismatch: expected {compare_revision}")
        self.session.expire_all()
        updated = self.session.scalar(
            select(OrderIntent).where(OrderIntent.intent_id == normalized_id)
        )
        if updated is None:
            raise RuntimeError("Order intent disappeared after transition")
        self._enqueue(
            aggregate_type="OrderIntent",
            aggregate_id=normalized_id,
            event_type="hedge.order_intent.status_changed",
            payload={
                "intent_id": normalized_id,
                "from_status": current,
                "to_status": target,
                "revision": updated.revision,
            },
            metadata=EventMetadata(
                correlation_id=updated.correlation_id,
                causation_id=normalized_id,
            ),
        )
        self.session.flush()
        return updated

    def append_order_snapshot(
        self,
        *,
        snapshot_key: str,
        account_id: str,
        exchange: str,
        symbol: str,
        position_side: str,
        exchange_order_id: str,
        side: str,
        status: str,
        action: str | None = None,
        original_quantity: Decimal | str | int | float,
        executed_quantity: Decimal | str | int | float,
        cumulative_quote: Decimal | str | int | float,
        source: str,
        source_event_time: datetime,
        client_order_id: str | None = None,
        intent_id: str | None = None,
        order_type: str | None = None,
        average_price: Decimal | str | int | float | None = None,
        last_fill_quantity: Decimal | str | int | float = "0",
        sequence_number: int | None = None,
        source_version: str | None = None,
        correlation_id: str | None = None,
        payload_version: int = 1,
        is_terminal: bool | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> tuple[OrderSnapshot, bool]:
        snapshot_key = _required_key("Order snapshot key", snapshot_key)
        account_id = _required_key("Account id", account_id, max_length=128)
        exchange = _required_key("Exchange", exchange, max_length=64).lower()
        symbol = canonical_symbol(symbol)
        exchange_order_id = _required_key("Exchange order id", exchange_order_id)
        original = decimal_value(original_quantity)
        executed = decimal_value(executed_quantity)
        cumulative = decimal_value(cumulative_quote)
        last_fill = decimal_value(last_fill_quantity)
        average = decimal_value(average_price) if average_price is not None else None
        normalized_time = _naive_utc(source_event_time)
        normalized_status = normalize_order_status(status)
        if min(original, executed, cumulative, last_fill) < 0:
            raise ValueError("Order snapshot quantities must not be negative")
        if original > 0 and executed > original:
            raise ValueError("Executed quantity must not exceed original quantity")
        if average is not None and average <= 0:
            raise ValueError("Average price must be positive")
        if sequence_number is not None and sequence_number < 0:
            raise ValueError("Sequence number must not be negative")
        if payload_version <= 0:
            raise ValueError("Payload version must be positive")
        normalized_source = _normalize_source(source)
        remaining = max(original - executed, Decimal("0"))
        fact_key = stable_fact_key(
            "order",
            exchange,
            account_id,
            symbol,
            position_side,
            exchange_order_id,
            normalized_source,
            source_version or "",
            sequence_number if sequence_number is not None else "",
            normalized_time.isoformat(),
        )
        expected = {
            "fact_key": fact_key,
            "account_id": account_id,
            "exchange": exchange,
            "symbol": symbol,
            "position_side": _normalize_side(position_side),
            "exchange_order_id": exchange_order_id,
            "client_order_id": client_order_id,
            "intent_id": intent_id,
            "correlation_id": correlation_id,
            "side": _normalize_order_side(side),
            "action": _normalize_action(action) if action is not None else None,
            "order_type": order_type,
            "status": normalized_status,
            "original_quantity": canonical_decimal(original),
            "executed_quantity": canonical_decimal(executed),
            "remaining_quantity": canonical_decimal(remaining),
            "cumulative_quote": canonical_decimal(cumulative),
            "average_price": canonical_decimal(average) if average is not None else None,
            "last_fill_quantity": canonical_decimal(last_fill),
            "source": normalized_source,
            "sequence_number": sequence_number,
            "source_version": source_version,
            "payload_version": payload_version,
            "source_event_time": normalized_time,
            "is_terminal": (
                normalized_status in TERMINAL_ORDER_STATUSES
                if is_terminal is None
                else is_terminal
            ),
            "raw_payload_json": canonical_json(raw_payload or {}),
        }

        def find_existing() -> OrderSnapshot | None:
            return self.session.scalar(
                select(OrderSnapshot).where(
                    (OrderSnapshot.fact_key == fact_key)
                    | (OrderSnapshot.snapshot_key == snapshot_key)
                )
            )

        def verify_existing(existing: OrderSnapshot) -> None:
            mismatches = _immutable_mismatches(existing, expected)
            if mismatches:
                _raise_immutable_conflict("Order snapshot", fact_key, mismatches)

        existing = find_existing()
        if existing is not None:
            verify_existing(existing)
            self._upsert_current_order_projection(existing)
            return existing, False

        snapshot = OrderSnapshot(snapshot_key=snapshot_key, **expected)
        existing = self._insert_immutable(snapshot, find_existing, verify_existing)
        if existing is not None:
            self._upsert_current_order_projection(existing)
            return existing, False
        self._upsert_current_order_projection(snapshot)
        self._enqueue(
            aggregate_type="OrderSnapshot",
            aggregate_id=snapshot_key,
            event_type="hedge.order.fact.recorded",
            payload={
                "snapshot_key": snapshot_key,
                "fact_key": fact_key,
                "account_id": account_id,
                "exchange": exchange,
                "symbol": symbol,
                "position_side": snapshot.position_side,
                "exchange_order_id": exchange_order_id,
                "status": snapshot.status,
                "source": normalized_source,
            },
            metadata=EventMetadata(
                correlation_id=correlation_id or snapshot_key,
                exchange_time=normalized_time,
                observed_time=snapshot.observed_at,
                payload_version=payload_version,
            ),
        )
        self.session.flush()
        return snapshot, True

    def _upsert_current_order_projection(
        self,
        snapshot: OrderSnapshot,
    ) -> CurrentOrderProjection:
        predicate = (
            CurrentOrderProjection.exchange == snapshot.exchange,
            CurrentOrderProjection.account_id == snapshot.account_id,
            CurrentOrderProjection.symbol == snapshot.symbol,
            CurrentOrderProjection.position_side == snapshot.position_side,
            CurrentOrderProjection.exchange_order_id == snapshot.exchange_order_id,
        )
        current = self.session.scalar(select(CurrentOrderProjection).where(*predicate))
        incoming_key = (
            snapshot.source_event_time,
            snapshot.sequence_number if snapshot.sequence_number is not None else -1,
            snapshot.id or 0,
        )
        if current is not None:
            current_key = (
                current.source_event_time,
                current.sequence_number if current.sequence_number is not None else -1,
                current.revision,
            )
            if incoming_key < current_key:
                return current
            current.client_order_id = snapshot.client_order_id
            current.intent_id = snapshot.intent_id
            current.action = snapshot.action
            current.side = snapshot.side
            current.status = snapshot.status
            current.original_quantity = snapshot.original_quantity
            current.executed_quantity = snapshot.executed_quantity
            current.remaining_quantity = snapshot.remaining_quantity
            current.cumulative_quote = snapshot.cumulative_quote
            current.average_price = snapshot.average_price
            current.is_terminal = snapshot.is_terminal
            current.source_snapshot_key = snapshot.snapshot_key
            current.source_event_time = snapshot.source_event_time
            current.sequence_number = snapshot.sequence_number
            current.revision += 1
            current.updated_at = utcnow()
            self.session.flush([current])
            return current

        current = CurrentOrderProjection(
            exchange=snapshot.exchange,
            account_id=snapshot.account_id,
            symbol=snapshot.symbol,
            position_side=snapshot.position_side,
            exchange_order_id=snapshot.exchange_order_id,
            client_order_id=snapshot.client_order_id,
            intent_id=snapshot.intent_id,
            action=snapshot.action,
            side=snapshot.side,
            status=snapshot.status,
            original_quantity=snapshot.original_quantity,
            executed_quantity=snapshot.executed_quantity,
            remaining_quantity=snapshot.remaining_quantity,
            cumulative_quote=snapshot.cumulative_quote,
            average_price=snapshot.average_price,
            is_terminal=snapshot.is_terminal,
            source_snapshot_key=snapshot.snapshot_key,
            source_event_time=snapshot.source_event_time,
            sequence_number=snapshot.sequence_number,
            revision=1,
        )
        self.session.add(current)
        self.session.flush([current])
        return current

    def _current_order(
        self,
        *,
        exchange: str,
        account_id: str,
        symbol: str,
        position_side: str,
        exchange_order_id: str,
    ) -> CurrentOrderProjection | None:
        return self.session.scalar(
            select(CurrentOrderProjection).where(
                CurrentOrderProjection.exchange == exchange,
                CurrentOrderProjection.account_id == account_id,
                CurrentOrderProjection.symbol == symbol,
                CurrentOrderProjection.position_side == position_side,
                CurrentOrderProjection.exchange_order_id == exchange_order_id,
            )
        )

    def record_account_event(
        self,
        *,
        event_key: str,
        account_id: str,
        exchange: str,
        event_type: str,
        asset: str,
        amount: Decimal | str | int | float,
        source: str,
        event_time: datetime,
        balance_after: Decimal | str | int | float | None = None,
        symbol: str | None = None,
        position_side: str | None = None,
        related_order_id: str | None = None,
        related_trade_id: str | None = None,
        transfer_direction: str | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> tuple[AccountEvent, bool]:
        event_key = _required_key("Account event key", event_key)
        account_id = _required_key("Account id", account_id, max_length=128)
        exchange = _required_key("Exchange", exchange, max_length=64).lower()
        asset = _required_key("Asset", asset, max_length=32)
        normalized_type = event_type.upper()
        if normalized_type not in VALID_ACCOUNT_EVENT_TYPES:
            raise ValueError(f"Unsupported account event type: {event_type!r}")
        normalized_source = _normalize_source(source)
        normalized_position_side = _normalize_side(position_side) if position_side else None
        normalized_transfer_direction = transfer_direction.upper() if transfer_direction else None
        if normalized_type == "TRANSFER" and normalized_transfer_direction not in {
            None,
            "IN",
            "OUT",
            "INTERNAL",
        }:
            raise ValueError("Transfer events require direction IN, OUT, or INTERNAL")
        if normalized_type != "TRANSFER" and normalized_transfer_direction is not None:
            raise ValueError("Transfer direction is only valid for TRANSFER events")
        normalized_time = _naive_utc(event_time)
        canonical_event_symbol = canonical_symbol(symbol) if symbol else None
        fact_key = stable_fact_key(
            "account-event",
            exchange,
            account_id,
            normalized_type,
            asset.upper(),
            canonical_decimal(amount),
            canonical_event_symbol or "",
            normalized_position_side or "",
            related_order_id or "",
            related_trade_id or "",
            normalized_transfer_direction or "",
            normalized_source,
            normalized_time.isoformat(),
        )
        expected = {
            "fact_key": fact_key,
            "account_id": account_id,
            "exchange": exchange,
            "event_type": normalized_type,
            "asset": asset,
            "amount": canonical_decimal(amount),
            "balance_after": (
                canonical_decimal(balance_after) if balance_after is not None else None
            ),
            "symbol": canonical_event_symbol,
            "position_side": normalized_position_side,
            "related_order_id": related_order_id,
            "related_trade_id": related_trade_id,
            "transfer_direction": normalized_transfer_direction,
            "source": normalized_source,
            "event_time": normalized_time,
            "raw_payload_json": canonical_json(raw_payload or {}),
        }

        def find_existing() -> AccountEvent | None:
            return self.session.scalar(
                select(AccountEvent).where(
                    (AccountEvent.fact_key == fact_key) | (AccountEvent.event_key == event_key)
                )
            )

        def verify_existing(existing: AccountEvent) -> None:
            mismatches = _immutable_mismatches(existing, expected)
            if mismatches:
                _raise_immutable_conflict("Account event", event_key, mismatches)

        existing = find_existing()
        if existing is not None:
            verify_existing(existing)
            return existing, False
        event_id = new_event_id()
        row = AccountEvent(
            event_id=event_id,
            event_key=event_key,
            **expected,
        )
        existing = self._insert_immutable(row, find_existing, verify_existing)
        if existing is not None:
            return existing, False
        self._enqueue(
            aggregate_type="AccountEvent",
            aggregate_id=event_id,
            event_type=f"hedge.account.{normalized_type.lower()}",
            payload={
                "event_id": event_id,
                "event_key": event_key,
                "account_id": account_id,
                "event_type": normalized_type,
                "asset": asset,
                "amount": row.amount,
                "fact_key": fact_key,
            },
            metadata=EventMetadata(
                correlation_id=related_trade_id or related_order_id or event_key,
                exchange_time=normalized_time,
            ),
        )
        self.session.flush()
        return row, True

    def append_position_snapshot(
        self,
        *,
        snapshot_key: str,
        account_id: str,
        exchange: str,
        symbol: str,
        position_side: str,
        quantity: Decimal | str | int | float,
        entry_price: Decimal | str | int | float,
        source: str,
        source_event_time: datetime,
        sequence_number: int | None = None,
        source_version: str | None = None,
        venue_symbol: str | None = None,
        mark_price: Decimal | str | int | float | None = None,
        notional: Decimal | str | int | float | None = None,
        realized_pnl: Decimal | str | int | float = "0",
        unrealized_pnl: Decimal | str | int | float = "0",
        liquidation_price: Decimal | str | int | float | None = None,
        leverage: Decimal | str | int | float = "1",
        margin_mode: str | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> tuple[PositionSnapshot, bool]:
        normalized_snapshot_key = _required_key("Position snapshot key", snapshot_key)
        account_id = _required_key("Account id", account_id, max_length=128)
        exchange = _required_key("Exchange", exchange, max_length=64).lower()
        canonical = canonical_symbol(symbol)
        normalized_side = _normalize_side(position_side)
        normalized_source = _normalize_source(source)
        normalized_time = _naive_utc(source_event_time)
        normalized_quantity = decimal_value(quantity)
        normalized_entry_price = decimal_value(entry_price)
        normalized_leverage = decimal_value(leverage)
        normalized_mark_price = decimal_value(mark_price) if mark_price is not None else None
        normalized_notional = decimal_value(notional) if notional is not None else None
        normalized_liquidation = (
            decimal_value(liquidation_price) if liquidation_price is not None else None
        )
        if sequence_number is not None and sequence_number < 0:
            raise ValueError("Position sequence number must not be negative")
        if normalized_quantity < 0:
            raise ValueError("Position quantity must not be negative")
        if normalized_entry_price < 0 or (
            normalized_quantity > 0 and normalized_entry_price == 0
        ):
            raise ValueError("Active position entry price must be positive")
        if normalized_leverage <= 0:
            raise ValueError("Position leverage must be positive")
        if normalized_mark_price is not None and normalized_mark_price < 0:
            raise ValueError("Position mark price must not be negative")
        if normalized_notional is not None and normalized_notional < 0:
            raise ValueError("Position notional must not be negative")
        if normalized_liquidation is not None and normalized_liquidation < 0:
            raise ValueError("Position liquidation price must not be negative")

        fact_key = stable_fact_key(
            "position",
            exchange,
            account_id,
            canonical,
            normalized_side,
            normalized_source,
            source_version or "",
            sequence_number if sequence_number is not None else "",
            normalized_time.isoformat(),
        )
        expected = {
            "fact_key": fact_key,
            "account_id": account_id,
            "exchange": exchange,
            "symbol": canonical,
            "venue_symbol": venue_symbol or symbol,
            "position_side": normalized_side,
            "quantity": canonical_decimal(normalized_quantity),
            "entry_price": canonical_decimal(normalized_entry_price),
            "mark_price": (
                canonical_decimal(normalized_mark_price)
                if normalized_mark_price is not None
                else None
            ),
            "notional": (
                canonical_decimal(normalized_notional)
                if normalized_notional is not None
                else None
            ),
            "realized_pnl": canonical_decimal(realized_pnl),
            "unrealized_pnl": canonical_decimal(unrealized_pnl),
            "liquidation_price": (
                canonical_decimal(normalized_liquidation)
                if normalized_liquidation is not None
                else None
            ),
            "leverage": canonical_decimal(normalized_leverage),
            "margin_mode": margin_mode.upper() if margin_mode else None,
            "source": normalized_source,
            "sequence_number": sequence_number,
            "source_version": source_version,
            "risk_source_snapshot_key": normalized_snapshot_key,
            "source_event_time": normalized_time,
            "is_active": normalized_quantity != 0,
            "is_current": False,
            "raw_payload_json": canonical_json(raw_payload or {}),
        }

        def find_existing() -> PositionSnapshot | None:
            return self.session.scalar(
                select(PositionSnapshot).where(
                    (PositionSnapshot.fact_key == fact_key)
                    | (PositionSnapshot.snapshot_key == normalized_snapshot_key)
                )
            )

        def verify_existing(existing: PositionSnapshot) -> None:
            mismatches = _immutable_mismatches(existing, expected)
            if mismatches:
                _raise_immutable_conflict("Position snapshot", fact_key, mismatches)

        existing = find_existing()
        if existing is not None:
            verify_existing(existing)
            return existing, False

        self._lock_position_scope(
            exchange=exchange,
            account_id=account_id,
            symbol=canonical,
            position_side=normalized_side,
        )
        row = PositionSnapshot(snapshot_key=normalized_snapshot_key, **expected)
        existing = self._insert_immutable(row, find_existing, verify_existing)
        if existing is not None:
            return existing, False
        projection = self._rebuild_position_scope(
            exchange=exchange,
            account_id=account_id,
            symbol=canonical,
            position_side=normalized_side,
            projection_trigger_id=f"fact:{row.fact_key}",
        )
        metadata = EventMetadata(
            correlation_id=normalized_snapshot_key,
            exchange_time=normalized_time,
            observed_time=row.observed_at,
        )
        self._enqueue(
            aggregate_type="PositionSnapshot",
            aggregate_id=normalized_snapshot_key,
            event_type="hedge.position.fact.recorded",
            payload={
                "snapshot_key": normalized_snapshot_key,
                "fact_key": fact_key,
                "exchange": exchange,
                "account_id": account_id,
                "symbol": canonical,
                "position_side": normalized_side,
                "source": normalized_source,
                "projection_snapshot_key": projection.snapshot_key,
            },
            metadata=metadata,
        )
        self._enqueue(
            aggregate_type="PositionProjection",
            aggregate_id=_bounded_key(
                "position", exchange, account_id, canonical, normalized_side
            ),
            event_type="hedge.position.projected",
            payload={
                "exchange": exchange,
                "account_id": account_id,
                "symbol": canonical,
                "position_side": normalized_side,
                "quantity": projection.quantity,
                "entry_price": projection.entry_price,
                "source_position_snapshot_key": normalized_snapshot_key,
            },
            metadata=metadata,
        )
        self.session.flush()
        return row, True

    def append_account_risk_snapshot(
        self,
        *,
        snapshot_key: str,
        account_id: str,
        exchange: str,
        source: str,
        source_event_time: datetime,
        equity: Decimal | str | int | float | None = None,
        wallet_balance: Decimal | str | int | float = "0",
        available_balance: Decimal | str | int | float = "0",
        margin_balance: Decimal | str | int | float = "0",
        total_initial_margin: Decimal | str | int | float = "0",
        total_maintenance_margin: Decimal | str | int | float = "0",
        gross_long_notional: Decimal | str | int | float | None = None,
        gross_short_notional: Decimal | str | int | float | None = None,
        gross_exposure: Decimal | str | int | float | None = None,
        net_exposure: Decimal | str | int | float = "0",
        pending_risk: Decimal | str | int | float = "0",
        margin_utilization: Decimal | str | int | float = "0",
        liquidation_buffer: Decimal | str | int | float | None = None,
        risk_state: str = "HALT",
        risk_data_valid: bool = False,
        source_snapshot_id: str | None = None,
        source_version: str | None = None,
        rules_version: str | None = None,
        reason_codes: Iterable[str] | None = None,
        projected_risk: dict[str, Any] | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> tuple[AccountRiskSnapshot, bool]:
        snapshot_key = _required_key("Account risk snapshot key", snapshot_key)
        account_id = _required_key("Account id", account_id, max_length=128)
        exchange = _required_key("Exchange", exchange, max_length=64).lower()
        normalized_source = _normalize_source(source)
        normalized_time = _naive_utc(source_event_time)
        net_value = decimal_value(net_exposure)
        gross_total_input = (
            decimal_value(gross_exposure) if gross_exposure is not None else None
        )
        if gross_long_notional is None and gross_short_notional is None:
            gross_total_input = gross_total_input or Decimal("0")
            gross_long = (gross_total_input + net_value) / Decimal("2")
            gross_short = (gross_total_input - net_value) / Decimal("2")
        else:
            gross_long = decimal_value(gross_long_notional)
            gross_short = decimal_value(gross_short_notional)
        gross_total = (
            gross_total_input
            if gross_total_input is not None
            else gross_long + gross_short
        )
        wallet_value = decimal_value(wallet_balance)
        margin_value = decimal_value(margin_balance)
        equity_value = (
            decimal_value(equity)
            if equity is not None
            else max(wallet_value, margin_value)
        )
        numbers = {
            "equity": equity_value,
            "wallet_balance": wallet_value,
            "available_balance": decimal_value(available_balance),
            "margin_balance": margin_value,
            "total_initial_margin": decimal_value(total_initial_margin),
            "total_maintenance_margin": decimal_value(total_maintenance_margin),
            "gross_long_notional": gross_long,
            "gross_short_notional": gross_short,
            "gross_exposure": gross_total,
            "net_exposure": net_value,
            "pending_risk": decimal_value(pending_risk),
            "margin_utilization": decimal_value(margin_utilization),
        }
        if any(value < 0 for name, value in numbers.items() if name != "net_exposure"):
            raise ValueError("Account risk non-directional values must not be negative")
        if gross_total != gross_long + gross_short:
            raise ValueError("Gross exposure must equal gross long plus gross short")
        normalized_state = risk_state.upper()
        if normalized_state == "READY" and not risk_data_valid and numbers["equity"] > 0:
            risk_data_valid = True
            reason_codes = tuple(reason_codes or ()) + ("LEGACY_RISK_VALIDATED",)
        if risk_data_valid and numbers["equity"] <= 0:
            raise ValueError("Valid risk data requires positive equity")
        if not risk_data_valid and normalized_state == "READY":
            raise ValueError("Invalid risk data cannot be READY")
        reason_list = sorted(set(reason_codes or ()))
        fact_key = stable_fact_key(
            "account-risk",
            exchange,
            account_id,
            normalized_source,
            source_version or "",
            normalized_time.isoformat(),
        )
        expected = {
            "fact_key": fact_key,
            "account_id": account_id,
            "exchange": exchange,
            **{name: canonical_decimal(value) for name, value in numbers.items()},
            "liquidation_buffer": (
                canonical_decimal(liquidation_buffer) if liquidation_buffer is not None else None
            ),
            "risk_state": normalized_state,
            "risk_data_valid": risk_data_valid,
            "source_snapshot_id": source_snapshot_id,
            "source_version": source_version,
            "rules_version": rules_version,
            "reason_codes_json": canonical_json(reason_list),
            "projected_risk_json": canonical_json(projected_risk or {}),
            "source": normalized_source,
            "source_event_time": normalized_time,
            "raw_payload_json": canonical_json(raw_payload or {}),
        }

        def find_existing() -> AccountRiskSnapshot | None:
            return self.session.scalar(
                select(AccountRiskSnapshot).where(
                    (AccountRiskSnapshot.fact_key == fact_key)
                    | (AccountRiskSnapshot.snapshot_key == snapshot_key)
                )
            )

        def verify_existing(existing: AccountRiskSnapshot) -> None:
            mismatches = _immutable_mismatches(existing, expected)
            if mismatches:
                _raise_immutable_conflict("Account risk snapshot", fact_key, mismatches)

        existing = find_existing()
        if existing is not None:
            verify_existing(existing)
            return existing, False

        row = AccountRiskSnapshot(snapshot_key=snapshot_key, **expected)
        existing = self._insert_immutable(row, find_existing, verify_existing)
        if existing is not None:
            return existing, False
        self._enqueue(
            aggregate_type="AccountRiskSnapshot",
            aggregate_id=snapshot_key,
            event_type="hedge.account_risk.fact.recorded",
            payload={
                "snapshot_key": snapshot_key,
                "fact_key": fact_key,
                "account_id": account_id,
                "risk_state": row.risk_state,
                "risk_data_valid": row.risk_data_valid,
                "source": normalized_source,
            },
            metadata=EventMetadata(
                correlation_id=snapshot_key,
                exchange_time=normalized_time,
                observed_time=row.observed_at,
            ),
        )
        self.session.flush()
        return row, True

    def start_reconciliation(
        self,
        *,
        account_id: str,
        exchange: str,
        trigger: str,
        scope: str = "FULL",
    ) -> ReconciliationRun:
        run_id = new_event_id()
        run = ReconciliationRun(
            run_id=run_id,
            account_id=account_id,
            exchange=exchange,
            trigger=trigger.upper(),
            scope=scope.upper(),
            status="RUNNING",
        )
        self.session.add(run)
        self._enqueue(
            aggregate_type="ReconciliationRun",
            aggregate_id=run_id,
            event_type="hedge.reconciliation.started",
            payload={
                "run_id": run_id,
                "account_id": account_id,
                "trigger": run.trigger,
                "scope": run.scope,
            },
        )
        self.session.flush()
        return run

    def add_reconciliation_diff(
        self,
        *,
        run_id: str,
        diff_key: str,
        entity_type: str,
        entity_key: str,
        severity: str,
        field_name: str | None = None,
        local_value: Any = None,
        exchange_value: Any = None,
        resolution: str = "UNRESOLVED",
        repair_action: str | None = None,
    ) -> tuple[ReconciliationDiff, bool]:
        existing = self.session.scalar(
            select(ReconciliationDiff).where(
                ReconciliationDiff.run_id == run_id,
                ReconciliationDiff.diff_key == diff_key,
            )
        )
        if existing is not None:
            return existing, False
        row = ReconciliationDiff(
            run_id=run_id,
            diff_key=diff_key,
            entity_type=entity_type,
            entity_key=entity_key,
            field_name=field_name,
            local_value_json=canonical_json(local_value),
            exchange_value_json=canonical_json(exchange_value),
            severity=severity.upper(),
            resolution=resolution.upper(),
            repair_action=repair_action,
            resolved_at=(
                utcnow() if resolution.upper() not in {"UNRESOLVED", "PENDING"} else None
            ),
        )
        self.session.add(row)
        run = self.session.scalar(
            select(ReconciliationRun).where(ReconciliationRun.run_id == run_id)
        )
        if run is None:
            raise ValueError(f"Unknown reconciliation run: {run_id}")
        run.diff_count += 1
        if severity.upper() in {"ERROR", "CRITICAL"}:
            run.severe_diff_count += 1
        self._enqueue(
            aggregate_type="ReconciliationDiff",
            aggregate_id=_bounded_key("reconciliation-diff", run_id, diff_key),
            event_type="hedge.reconciliation.diff.recorded",
            payload={
                "run_id": run_id,
                "diff_key": diff_key,
                "entity_type": entity_type,
                "entity_key": entity_key,
                "severity": row.severity,
            },
        )
        self.session.flush()
        return row, True

    def complete_reconciliation(
        self,
        *,
        run_id: str,
        status: str,
        summary: dict[str, Any] | None = None,
        error_message: str | None = None,
    ) -> ReconciliationRun:
        run = self.session.scalar(
            select(ReconciliationRun).where(ReconciliationRun.run_id == run_id)
        )
        if run is None:
            raise ValueError(f"Unknown reconciliation run: {run_id}")
        run.status = status.upper()
        run.completed_at = utcnow()
        run.summary_json = canonical_json(summary or {})
        run.error_message = error_message
        self._enqueue(
            aggregate_type="ReconciliationRun",
            aggregate_id=run_id,
            event_type="hedge.reconciliation.completed",
            payload={
                "run_id": run_id,
                "status": run.status,
                "diff_count": run.diff_count,
                "severe_diff_count": run.severe_diff_count,
            },
        )
        self.session.flush()
        return run

    def upsert_strategy_side_state(
        self,
        *,
        exchange: str = "binance",
        account_id: str,
        symbol: str,
        position_side: str,
        strategy_name: str,
        state_name: str,
        state: dict[str, Any],
        last_decision_id: str | None = None,
        cooldown_until: datetime | None = None,
        expected_revision: int | None = None,
    ) -> StrategySideState:
        """Create or atomically compare-and-swap one strategy-side state."""

        exchange = _required_key("Exchange", exchange, max_length=64).lower()
        account_id = _required_key("Account id", account_id, max_length=128)
        symbol = canonical_symbol(symbol)
        strategy_name = _required_key("Strategy name", strategy_name, max_length=128)
        normalized_side = _normalize_side(position_side)
        normalized_state_name = _required_key(
            "Strategy state name",
            state_name.upper(),
            max_length=64,
        )
        if expected_revision is not None and expected_revision < 0:
            raise ValueError("Expected revision must not be negative")
        normalized_cooldown = _naive_utc(cooldown_until) if cooldown_until else None
        state_json = canonical_json(state)
        predicate = (
            StrategySideState.exchange == exchange,
            StrategySideState.account_id == account_id,
            StrategySideState.symbol == symbol,
            StrategySideState.position_side == normalized_side,
            StrategySideState.strategy_name == strategy_name,
        )

        def load() -> StrategySideState | None:
            return self.session.scalar(select(StrategySideState).where(*predicate))

        row = load()
        if row is None:
            if expected_revision not in {None, 0}:
                raise ValueError("Strategy side state revision mismatch: state does not exist")
            candidate = StrategySideState(
                exchange=exchange,
                account_id=account_id,
                symbol=symbol,
                position_side=normalized_side,
                strategy_name=strategy_name,
                state_name=normalized_state_name,
                state_json=state_json,
                last_decision_id=last_decision_id,
                cooldown_until=normalized_cooldown,
                revision=1,
                updated_at=utcnow(),
            )
            if self.session.get_bind().dialect.name != "postgresql":
                self.session.add(candidate)
                self.session.flush([candidate])
                row = candidate
                self._enqueue_strategy_state(row)
                self.session.flush()
                return row
            try:
                with self.session.begin_nested():
                    self.session.add(candidate)
                    self.session.flush([candidate])
            except IntegrityError:
                row = load()
                if row is None:
                    raise
            else:
                row = candidate
                self._enqueue_strategy_state(row)
                self.session.flush()
                return row

        assert row is not None
        compare_revision = row.revision if expected_revision is None else expected_revision
        now = utcnow()
        result = self.session.execute(
            update(StrategySideState)
            .where(*predicate, StrategySideState.revision == compare_revision)
            .values(
                state_name=normalized_state_name,
                state_json=state_json,
                last_decision_id=last_decision_id,
                cooldown_until=normalized_cooldown,
                revision=compare_revision + 1,
                updated_at=now,
            )
        )
        if result.rowcount != 1:
            self.session.expire_all()
            current = load()
            found = current.revision if current is not None else "missing"
            raise ValueError(
                "Strategy side state revision mismatch: "
                f"expected {compare_revision}, found {found}"
            )
        self.session.expire_all()
        updated_row = load()
        if updated_row is None:
            raise RuntimeError("Strategy side state disappeared after successful CAS")
        self._enqueue_strategy_state(updated_row)
        self.session.flush()
        return updated_row

    def _enqueue_strategy_state(self, row: StrategySideState) -> None:
        self._enqueue(
            aggregate_type="StrategySideState",
            aggregate_id=_bounded_key(
                "strategy-side",
                row.exchange,
                row.account_id,
                row.symbol,
                row.position_side,
                row.strategy_name,
            ),
            event_type="hedge.strategy_side_state.updated",
            payload={
                "exchange": row.exchange,
                "account_id": row.account_id,
                "symbol": row.symbol,
                "position_side": row.position_side,
                "strategy_name": row.strategy_name,
                "state_name": row.state_name,
                "revision": row.revision,
            },
        )

    def record_target_position(
        self,
        *,
        exchange: str,
        account_id: str,
        symbol: str,
        position_side: str,
        target_quantity: Decimal | str | int | float,
        reason: str,
        strategy_id: str,
        cycle_id: str,
        correlation_id: str,
        payload: dict[str, Any] | None = None,
    ) -> tuple[TargetPosition, bool]:
        exchange = _required_key("Exchange", exchange, max_length=64).lower()
        account_id = _required_key("Account id", account_id, max_length=128)
        symbol = canonical_symbol(symbol)
        position_side = _normalize_side(position_side)
        strategy_id = _required_key("Strategy id", strategy_id, max_length=128)
        cycle_id = _required_key("Cycle id", cycle_id, max_length=128)
        correlation_id = _required_key("Correlation id", correlation_id, max_length=128)
        quantity = decimal_value(target_quantity)
        if quantity < 0:
            raise ValueError("Target quantity must not be negative")
        existing = self.session.scalar(
            select(TargetPosition).where(
                TargetPosition.exchange == exchange,
                TargetPosition.account_id == account_id,
                TargetPosition.symbol == symbol,
                TargetPosition.position_side == position_side,
                TargetPosition.strategy_id == strategy_id,
                TargetPosition.cycle_id == cycle_id,
            )
        )
        expected = {
            "target_quantity": canonical_decimal(quantity),
            "reason": reason,
            "correlation_id": correlation_id,
            "target_payload_json": canonical_json(payload or {}),
        }
        if existing is not None:
            mismatches = _immutable_mismatches(existing, expected)
            if mismatches:
                _raise_immutable_conflict("Target position", cycle_id, mismatches)
            return existing, False
        row = TargetPosition(
            exchange=exchange,
            account_id=account_id,
            symbol=symbol,
            position_side=position_side,
            strategy_id=strategy_id,
            cycle_id=cycle_id,
            **expected,
        )
        self.session.add(row)
        self.session.flush([row])
        self._enqueue(
            aggregate_type="TargetPosition",
            aggregate_id=row.target_id,
            event_type="hedge.target_position.recorded",
            payload={
                "target_id": row.target_id,
                "account_id": account_id,
                "symbol": symbol,
                "position_side": position_side,
                "target_quantity": row.target_quantity,
                "cycle_id": cycle_id,
            },
            metadata=EventMetadata(correlation_id=correlation_id),
        )
        return row, True

    def upsert_core_position_state(
        self,
        *,
        exchange: str,
        account_id: str,
        symbol: str,
        position_side: str,
        core_quantity: Decimal | str | int | float,
        core_floor: Decimal | str | int | float,
        effective_cost: Decimal | str | int | float | None = None,
        realized_profit_credit: Decimal | str | int | float = "0",
        state: dict[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> CorePositionState:
        exchange = _required_key("Exchange", exchange, max_length=64).lower()
        account_id = _required_key("Account id", account_id, max_length=128)
        symbol = canonical_symbol(symbol)
        position_side = _normalize_side(position_side)
        core_qty = decimal_value(core_quantity)
        floor = decimal_value(core_floor)
        credit = decimal_value(realized_profit_credit)
        cost = decimal_value(effective_cost) if effective_cost is not None else None
        if core_qty < 0 or floor < 0 or credit < 0:
            raise ValueError("Core position values must not be negative")
        if floor > core_qty:
            raise ValueError("Core floor must not exceed core quantity")
        predicate = (
            CorePositionState.exchange == exchange,
            CorePositionState.account_id == account_id,
            CorePositionState.symbol == symbol,
            CorePositionState.position_side == position_side,
        )
        row = self.session.scalar(select(CorePositionState).where(*predicate))
        if row is None:
            if expected_revision not in {None, 0}:
                raise ValueError("Core position revision mismatch: state does not exist")
            row = CorePositionState(
                exchange=exchange,
                account_id=account_id,
                symbol=symbol,
                position_side=position_side,
                core_quantity=canonical_decimal(core_qty),
                core_floor=canonical_decimal(floor),
                effective_cost=canonical_decimal(cost) if cost is not None else None,
                realized_profit_credit=canonical_decimal(credit),
                state_json=canonical_json(state or {}),
                revision=1,
            )
            self.session.add(row)
        else:
            compare = row.revision if expected_revision is None else expected_revision
            result = self.session.execute(
                update(CorePositionState)
                .where(*predicate, CorePositionState.revision == compare)
                .values(
                    core_quantity=canonical_decimal(core_qty),
                    core_floor=canonical_decimal(floor),
                    effective_cost=canonical_decimal(cost) if cost is not None else None,
                    realized_profit_credit=canonical_decimal(credit),
                    state_json=canonical_json(state or {}),
                    revision=compare + 1,
                    updated_at=utcnow(),
                )
            )
            if result.rowcount != 1:
                raise ValueError(f"Core position revision mismatch: expected {compare}")
            self.session.expire_all()
            row = self.session.scalar(select(CorePositionState).where(*predicate))
            if row is None:
                raise RuntimeError("Core position state disappeared after CAS")
        self.session.flush([row])
        self._enqueue(
            aggregate_type="CorePositionState",
            aggregate_id=_bounded_key(
                "core", exchange, account_id, symbol, position_side
            ),
            event_type="hedge.core_position.updated",
            payload={
                "exchange": exchange,
                "account_id": account_id,
                "symbol": symbol,
                "position_side": position_side,
                "core_quantity": row.core_quantity,
                "core_floor": row.core_floor,
                "revision": row.revision,
            },
        )
        return row

    def record_tactical_lot(
        self,
        *,
        exchange: str,
        account_id: str,
        symbol: str,
        position_side: str,
        strategy_name: str,
        lot_type: str,
        quantity: Decimal | str | int | float,
        entry_price: Decimal | str | int | float,
        opened_at: datetime,
        lot_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[TacticalLot, bool]:
        lot_id = lot_id or new_event_id()
        existing = self.session.scalar(select(TacticalLot).where(TacticalLot.lot_id == lot_id))
        quantity_value = decimal_value(quantity)
        price_value = decimal_value(entry_price)
        if quantity_value <= 0 or price_value <= 0:
            raise ValueError("Tactical lot quantity and entry price must be positive")
        expected = {
            "exchange": _required_key("Exchange", exchange, max_length=64).lower(),
            "account_id": _required_key("Account id", account_id, max_length=128),
            "symbol": canonical_symbol(symbol),
            "position_side": _normalize_side(position_side),
            "strategy_name": _required_key("Strategy name", strategy_name, max_length=128),
            "lot_type": _required_key("Lot type", lot_type.upper(), max_length=32),
            "quantity": canonical_decimal(quantity_value),
            "entry_price": canonical_decimal(price_value),
            "opened_at": _naive_utc(opened_at),
            "metadata_json": canonical_json(metadata or {}),
        }
        if existing is not None:
            mismatches = _immutable_mismatches(existing, expected)
            if mismatches:
                _raise_immutable_conflict("Tactical lot", lot_id, mismatches)
            return existing, False
        row = TacticalLot(lot_id=lot_id, **expected)
        self.session.add(row)
        self.session.flush([row])
        return row, True

    def record_audit_event(
        self,
        *,
        account_id: str,
        exchange: str,
        event_type: str,
        correlation_id: str,
        entity_type: str | None = None,
        entity_id: str | None = None,
        severity: str = "INFO",
        reason_code: str | None = None,
        actor: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> AuditEvent:
        row = AuditEvent(
            account_id=_required_key("Account id", account_id, max_length=128),
            exchange=_required_key("Exchange", exchange, max_length=64).lower(),
            event_type=_required_key("Audit event type", event_type, max_length=128),
            entity_type=entity_type,
            entity_id=entity_id,
            severity=severity.upper(),
            reason_code=reason_code,
            correlation_id=_required_key("Correlation id", correlation_id, max_length=128),
            actor=actor,
            payload_json=canonical_json(payload or {}),
        )
        self.session.add(row)
        self.session.flush([row])
        self._enqueue(
            aggregate_type="AuditEvent",
            aggregate_id=row.audit_id,
            event_type="hedge.audit.recorded",
            payload={
                "audit_id": row.audit_id,
                "event_type": row.event_type,
                "severity": row.severity,
                "reason_code": row.reason_code,
            },
            metadata=EventMetadata(correlation_id=row.correlation_id),
        )
        return row

    def resolve_reconciliation_diff(
        self,
        *,
        run_id: str,
        diff_key: str,
        resolution: str,
        resolved_by: str,
        note: str | None = None,
    ) -> ReconciliationDiff:
        row = self.session.scalar(
            select(ReconciliationDiff).where(
                ReconciliationDiff.run_id == run_id,
                ReconciliationDiff.diff_key == diff_key,
            )
        )
        if row is None:
            raise KeyError(f"Reconciliation diff not found: {run_id}/{diff_key}")
        row.resolution = resolution.upper()
        row.resolved_at = utcnow()
        row.resolved_by = resolved_by
        row.resolution_note = note
        self.session.flush([row])
        return row

    def apply_fill(
        self,
        *,
        exchange: str,
        account_id: str,
        symbol: str,
        position_side: str,
        exchange_trade_id: str,
        exchange_order_id: str,
        side: str,
        quantity: Decimal | str | int | float,
        price: Decimal | str | int | float,
        source: str,
        event_time: datetime,
        action: str | None = None,
        sequence_number: int | None = None,
        allow_overclose: bool = False,
        quote_quantity: Decimal | str | int | float | None = None,
        client_order_id: str | None = None,
        intent_id: str | None = None,
        correlation_id: str | None = None,
        fee_amount: Decimal | str | int | float = "0",
        fee_currency: str | None = None,
        realized_pnl: Decimal | str | int | float = "0",
        liquidity_role: str | None = None,
        raw_payload: dict[str, Any] | None = None,
    ) -> FillApplyResult:
        exchange = _required_key("Exchange", exchange, max_length=64).lower()
        account_id = _required_key("Account id", account_id, max_length=128)
        symbol = canonical_symbol(symbol)
        normalized_position_side = _normalize_side(position_side)
        normalized_order_side = _normalize_order_side(side)
        normalized_source = _normalize_source(source)
        normalized_time = _naive_utc(event_time)
        normalized_correlation = correlation_id or intent_id or exchange_trade_id
        normalized_action = (
            _normalize_action(action)
            if action is not None
            else _default_fill_action(normalized_position_side, normalized_order_side)
        )
        _validate_action_direction(
            normalized_position_side,
            normalized_order_side,
            normalized_action,
        )
        exchange_trade_id = _required_key("Exchange trade id", exchange_trade_id)
        exchange_order_id = _required_key("Exchange order id", exchange_order_id)
        if sequence_number is not None and sequence_number < 0:
            raise ValueError("Fill sequence number must not be negative")

        fill_quantity = decimal_value(quantity)
        fill_price = decimal_value(price)
        fill_fee = decimal_value(fee_amount)
        fill_realized_pnl = decimal_value(realized_pnl)
        if fill_quantity <= 0:
            raise ValueError("Fill quantity must be positive")
        if fill_price <= 0:
            raise ValueError("Fill price must be positive")
        if fill_fee < 0:
            raise ValueError("Fill fee amount must not be negative")
        quote = (
            decimal_value(quote_quantity)
            if quote_quantity is not None
            else fill_quantity * fill_price
        )
        if quote < 0:
            raise ValueError("Fill quote quantity must not be negative")

        self._lock_position_scope(
            exchange=exchange,
            account_id=account_id,
            symbol=symbol,
            position_side=normalized_position_side,
        )

        expected_fill = {
            "position_side": normalized_position_side,
            "exchange_order_id": exchange_order_id,
            "side": normalized_order_side,
            "action": normalized_action,
            "quantity": canonical_decimal(fill_quantity),
            "price": canonical_decimal(fill_price),
            "quote_quantity": canonical_decimal(quote),
            "fee_amount": canonical_decimal(fill_fee),
            "fee_currency": fee_currency,
            "realized_pnl": canonical_decimal(fill_realized_pnl),
            "sequence_number": sequence_number,
            "event_time": normalized_time,
        }

        def verify_existing(existing: FillEvent) -> None:
            mismatches = [
                name for name, value in expected_fill.items() if getattr(existing, name) != value
            ]
            if mismatches:
                raise ValueError(
                    "Conflicting duplicate fill for exchange trade id "
                    f"{exchange_trade_id!r}; changed fields: {', '.join(mismatches)}"
                )

        def find_existing() -> FillEvent | None:
            return self.session.scalar(
                select(FillEvent).where(
                    FillEvent.exchange == exchange,
                    FillEvent.account_id == account_id,
                    FillEvent.symbol == symbol,
                    FillEvent.exchange_trade_id == exchange_trade_id,
                )
            )

        existing = find_existing()
        if existing is not None:
            verify_existing(existing)
            return self._duplicate_fill_result(existing=existing)

        event_id = new_event_id()
        fill = FillEvent(
            event_id=event_id,
            exchange=exchange,
            account_id=account_id,
            symbol=symbol,
            position_side=normalized_position_side,
            exchange_trade_id=exchange_trade_id,
            exchange_order_id=exchange_order_id,
            client_order_id=client_order_id,
            intent_id=intent_id,
            correlation_id=normalized_correlation,
            side=normalized_order_side,
            action=normalized_action,
            quantity=canonical_decimal(fill_quantity),
            price=canonical_decimal(fill_price),
            quote_quantity=canonical_decimal(quote),
            fee_amount=canonical_decimal(fill_fee),
            fee_currency=fee_currency,
            realized_pnl=canonical_decimal(fill_realized_pnl),
            liquidity_role=liquidity_role,
            source=normalized_source,
            sequence_number=sequence_number,
            event_time=normalized_time,
            projection_status="APPLIED",
            raw_payload_json=canonical_json(raw_payload or {}),
        )
        existing = self._insert_immutable(fill, find_existing, verify_existing)
        if existing is not None:
            return self._duplicate_fill_result(existing=existing)

        cumulative_quantity, cumulative_quote = self.order_fill_totals(
            exchange=exchange,
            account_id=account_id,
            symbol=symbol,
            exchange_order_id=exchange_order_id,
        )
        cumulative_qty_value = decimal_value(cumulative_quantity)
        cumulative_quote_value = decimal_value(cumulative_quote)
        current_order = self._current_order(
            exchange=exchange,
            account_id=account_id,
            symbol=symbol,
            position_side=normalized_position_side,
            exchange_order_id=exchange_order_id,
        )
        intent = (
            self.session.scalar(select(OrderIntent).where(OrderIntent.intent_id == intent_id))
            if intent_id
            else None
        )
        original_quantity = max(
            cumulative_qty_value,
            decimal_value(current_order.original_quantity) if current_order else Decimal("0"),
            decimal_value(intent.requested_quantity) if intent else Decimal("0"),
        )
        order_status = (
            "FILLED"
            if original_quantity > 0 and cumulative_qty_value >= original_quantity
            else "PARTIAL"
        )
        average_price = (
            cumulative_quote_value / cumulative_qty_value
            if cumulative_qty_value > 0
            else Decimal("0")
        )
        self.append_order_snapshot(
            snapshot_key=_bounded_key(
                "fill-order",
                exchange,
                account_id,
                symbol,
                exchange_order_id,
                event_id,
            ),
            account_id=account_id,
            exchange=exchange,
            symbol=symbol,
            position_side=normalized_position_side,
            exchange_order_id=exchange_order_id,
            client_order_id=(
                client_order_id
                or (current_order.client_order_id if current_order else None)
            ),
            intent_id=intent_id,
            correlation_id=normalized_correlation,
            side=normalized_order_side,
            action=normalized_action,
            order_type=None,
            status=order_status,
            original_quantity=original_quantity,
            executed_quantity=cumulative_qty_value,
            cumulative_quote=cumulative_quote_value,
            average_price=average_price,
            last_fill_quantity=fill_quantity,
            source="LOCAL",
            source_event_time=normalized_time,
            sequence_number=sequence_number,
            source_version=exchange_trade_id,
            raw_payload={"source_fill_event_id": event_id},
        )
        if intent is not None:
            self._advance_intent_from_fill(intent, order_status)

        projection_blocked = False
        reason_code = None
        try:
            position = self._rebuild_position_scope(
                exchange=exchange,
                account_id=account_id,
                symbol=symbol,
                position_side=normalized_position_side,
                projection_trigger_id=f"fill:{event_id}",
                strict_overclose=not allow_overclose,
            )
        except ProjectionInvariantError as exc:
            projection_blocked = True
            reason_code = exc.reason_code
            offender = (
                self.session.scalar(
                    select(FillEvent).where(FillEvent.event_id == exc.offending_event_id)
                )
                if exc.offending_event_id
                else fill
            )
            offender = offender or fill
            offender.projection_status = "BLOCKED"
            offender.projection_error = str(exc)
            self._record_projection_failure(
                fill=offender,
                reason_code=exc.reason_code,
                message=str(exc),
                entity_key=exc.entity_key,
            )
            try:
                position = self._rebuild_position_scope(
                    exchange=exchange,
                    account_id=account_id,
                    symbol=symbol,
                    position_side=normalized_position_side,
                    projection_trigger_id=f"blocked-replay:{event_id}",
                    strict_overclose=True,
                    include_blocked=False,
                )
            except RuntimeError:
                position = self._current_position(
                    exchange,
                    account_id,
                    symbol,
                    normalized_position_side,
                )

        if fill_fee > 0 and fee_currency:
            self.record_account_event(
                event_key=_bounded_key(
                    "fill-fee", exchange, account_id, symbol, exchange_trade_id
                ),
                account_id=account_id,
                exchange=exchange,
                event_type="FEE",
                asset=fee_currency,
                amount=canonical_decimal(-fill_fee),
                symbol=symbol,
                position_side=normalized_position_side,
                related_order_id=exchange_order_id,
                related_trade_id=exchange_trade_id,
                source=normalized_source,
                event_time=normalized_time,
                raw_payload={"fill_event_id": event_id},
            )
        metadata = EventMetadata(
            correlation_id=normalized_correlation,
            causation_id=intent_id,
            exchange_time=normalized_time,
            observed_time=fill.observed_at,
        )
        self._enqueue(
            aggregate_type="FillEvent",
            aggregate_id=event_id,
            event_type=(
                "hedge.fill.projection_blocked"
                if projection_blocked
                else "hedge.fill.accepted"
            ),
            payload={
                "event_id": event_id,
                "exchange": exchange,
                "account_id": account_id,
                "symbol": symbol,
                "position_side": normalized_position_side,
                "action": normalized_action,
                "exchange_trade_id": exchange_trade_id,
                "quantity": canonical_decimal(fill_quantity),
                "price": canonical_decimal(fill_price),
                "projection_status": fill.projection_status,
                "reason_code": reason_code,
            },
            metadata=metadata,
        )
        if not projection_blocked and position is not None:
            self._enqueue(
                aggregate_type="PositionProjection",
                aggregate_id=_bounded_key(
                    "position",
                    exchange,
                    account_id,
                    symbol,
                    normalized_position_side,
                ),
                event_type="hedge.position.projected",
                payload={
                    "exchange": exchange,
                    "account_id": account_id,
                    "symbol": symbol,
                    "position_side": normalized_position_side,
                    "quantity": position.quantity,
                    "entry_price": position.entry_price,
                    "source_fill_event_id": event_id,
                },
                metadata=metadata,
            )
        self.session.flush()
        return FillApplyResult(
            accepted=True,
            duplicate=False,
            fill_event_id=event_id,
            action=normalized_action,
            position_quantity=position.quantity if position else "0",
            position_entry_price=position.entry_price if position else "0",
            cumulative_order_quantity=cumulative_quantity,
            cumulative_order_quote=cumulative_quote,
            order_status=order_status,
            projection_blocked=projection_blocked,
            reason_code=reason_code,
        )

    def _advance_intent_from_fill(self, intent: OrderIntent, order_status: str) -> None:
        target = normalize_order_status(order_status)
        current = normalize_order_status(intent.status)
        if current in TERMINAL_ORDER_STATUSES or current == target:
            return
        intent.status = target
        intent.revision += 1
        intent.updated_at = utcnow()
        self._enqueue(
            aggregate_type="OrderIntent",
            aggregate_id=intent.intent_id,
            event_type="hedge.order_intent.fill_progress",
            payload={
                "intent_id": intent.intent_id,
                "from_status": current,
                "to_status": target,
                "revision": intent.revision,
            },
            metadata=EventMetadata(
                correlation_id=intent.correlation_id,
                causation_id=intent.intent_id,
            ),
        )

    def _record_projection_failure(
        self,
        *,
        fill: FillEvent,
        reason_code: str,
        message: str,
        entity_key: str,
    ) -> None:
        run = self.start_reconciliation(
            account_id=fill.account_id,
            exchange=fill.exchange,
            trigger="FILL_PROJECTION",
            scope="POSITION",
        )
        self.add_reconciliation_diff(
            run_id=run.run_id,
            diff_key=_bounded_key("projection", fill.event_id),
            entity_type="POSITION",
            entity_key=entity_key,
            severity="CRITICAL",
            local_value={"projection_status": "BLOCKED"},
            exchange_value={
                "trade_id": fill.exchange_trade_id,
                "quantity": fill.quantity,
                "side": fill.side,
            },
            repair_action="REST_RECONCILE_AND_HALT",
        )
        self.complete_reconciliation(
            run_id=run.run_id,
            status="FAILED",
            summary={"reason_code": reason_code, "message": message},
            error_message=message,
        )
        self.record_audit_event(
            account_id=fill.account_id,
            exchange=fill.exchange,
            event_type="FILL_PROJECTION_BLOCKED",
            entity_type="FillEvent",
            entity_id=fill.event_id,
            severity="CRITICAL",
            reason_code=reason_code,
            correlation_id=fill.correlation_id or fill.event_id,
            payload={"message": message, "entity_key": entity_key},
        )

    def order_fill_totals(
        self,
        *,
        exchange: str,
        account_id: str,
        symbol: str,
        exchange_order_id: str,
    ) -> tuple[str, str]:
        rows = self.session.scalars(
            select(FillEvent).where(
                FillEvent.exchange == exchange,
                FillEvent.account_id == account_id,
                FillEvent.symbol == symbol,
                FillEvent.exchange_order_id == exchange_order_id,
            )
        ).all()
        quantity = sum((decimal_value(row.quantity) for row in rows), Decimal("0"))
        quote = sum((decimal_value(row.quote_quantity) for row in rows), Decimal("0"))
        return canonical_decimal(quantity), canonical_decimal(quote)

    def _duplicate_fill_result(
        self,
        *,
        existing: FillEvent,
    ) -> FillApplyResult:
        position = self._current_position(
            existing.exchange,
            existing.account_id,
            existing.symbol,
            existing.position_side,
        )
        cumulative_quantity, cumulative_quote = self.order_fill_totals(
            exchange=existing.exchange,
            account_id=existing.account_id,
            symbol=existing.symbol,
            exchange_order_id=existing.exchange_order_id,
        )
        current_order = self._current_order(
            exchange=existing.exchange,
            account_id=existing.account_id,
            symbol=existing.symbol,
            position_side=existing.position_side,
            exchange_order_id=existing.exchange_order_id,
        )
        return FillApplyResult(
            accepted=False,
            duplicate=True,
            fill_event_id=existing.event_id,
            action=existing.action or _default_fill_action(
                existing.position_side, existing.side
            ),
            position_quantity=position.quantity if position else "0",
            position_entry_price=position.entry_price if position else "0",
            cumulative_order_quantity=cumulative_quantity,
            cumulative_order_quote=cumulative_quote,
            order_status=current_order.status if current_order else None,
            projection_blocked=existing.projection_status == "BLOCKED",
            reason_code="PROJECTION_BLOCKED" if existing.projection_status == "BLOCKED" else None,
        )

    def _lock_position_scope(
        self,
        *,
        exchange: str,
        account_id: str,
        symbol: str,
        position_side: str,
    ) -> None:
        bind = self.session.get_bind()
        if bind.dialect.name != "postgresql":
            return
        scope = f"{exchange}|{account_id}|{symbol}|{position_side}"
        self.session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:scope, 0))"),
            {"scope": scope},
        )

    def _rebuild_position_scope(
        self,
        *,
        exchange: str,
        account_id: str,
        symbol: str,
        position_side: str,
        projection_trigger_id: str,
        strict_overclose: bool = True,
        include_blocked: bool = True,
    ) -> PositionSnapshot:
        fact = self.session.scalar(
            select(PositionSnapshot)
            .where(
                PositionSnapshot.exchange == exchange,
                PositionSnapshot.account_id == account_id,
                PositionSnapshot.symbol == symbol,
                PositionSnapshot.position_side == position_side,
                PositionSnapshot.source.in_(("REST", "WEBSOCKET", "MIGRATION")),
            )
            .order_by(
                PositionSnapshot.source_event_time.desc(),
                func.coalesce(PositionSnapshot.sequence_number, -1).desc(),
                PositionSnapshot.id.desc(),
            )
            .limit(1)
        )
        quantity = decimal_value(fact.quantity) if fact else Decimal("0")
        entry_price = decimal_value(fact.entry_price) if fact else Decimal("0")
        realized_pnl = decimal_value(fact.realized_pnl) if fact else Decimal("0")
        base_time = fact.source_event_time if fact else None
        base_sequence = (
            fact.sequence_number if fact and fact.sequence_number is not None else -1
        )
        last_event_time = base_time
        last_sequence = base_sequence if fact else None
        fill_stmt = select(FillEvent).where(
            FillEvent.exchange == exchange,
            FillEvent.account_id == account_id,
            FillEvent.symbol == symbol,
            FillEvent.position_side == position_side,
        )
        if not include_blocked:
            fill_stmt = fill_stmt.where(FillEvent.projection_status != "BLOCKED")
        fills = self.session.scalars(
            fill_stmt.order_by(
                FillEvent.event_time,
                func.coalesce(FillEvent.sequence_number, -1),
                FillEvent.exchange_trade_id,
                FillEvent.id,
            )
        ).all()
        applied_fills = 0
        for row in fills:
            row_sequence = row.sequence_number if row.sequence_number is not None else -1
            if base_time is not None and (
                row.event_time < base_time
                or (row.event_time == base_time and row_sequence <= base_sequence)
            ):
                continue
            fill_quantity = decimal_value(row.quantity)
            projected_quantity = quantity + _signed_quantity(
                position_side, row.side, fill_quantity
            )
            if projected_quantity < 0 and strict_overclose:
                entity_key = f"{exchange}|{account_id}|{symbol}|{position_side}"
                row.projection_status = "BLOCKED"
                row.projection_error = (
                    f"Fill would over-close position scope {entity_key}: "
                    f"trade_id={row.exchange_trade_id}"
                )
                raise ProjectionInvariantError(
                    row.projection_error,
                    reason_code="OVER_CLOSE_FILL",
                    entity_key=entity_key,
                    offending_event_id=row.event_id,
                )
            quantity, entry_price = _position_after_fill(
                quantity,
                entry_price,
                position_side,
                row.side,
                fill_quantity,
                decimal_value(row.price),
            )
            realized_pnl += decimal_value(row.realized_pnl)
            last_event_time = row.event_time
            last_sequence = row.sequence_number
            applied_fills += 1
        if include_blocked:
            for row in fills:
                if row.projection_status == "BLOCKED":
                    row.projection_status = "APPLIED"
                    row.projection_error = None
        if last_event_time is None:
            raise RuntimeError("Cannot project a position without a fact or fill")

        previous = self._current_position(exchange, account_id, symbol, position_side)
        self.session.execute(
            update(PositionSnapshot)
            .where(
                PositionSnapshot.exchange == exchange,
                PositionSnapshot.account_id == account_id,
                PositionSnapshot.symbol == symbol,
                PositionSnapshot.position_side == position_side,
                PositionSnapshot.is_current.is_(True),
            )
            .values(is_current=False)
        )
        snapshot_key = _bounded_key(
            "projection",
            exchange,
            account_id,
            symbol,
            position_side,
            projection_trigger_id,
        )
        projection = PositionSnapshot(
            snapshot_key=snapshot_key,
            fact_key=stable_fact_key("projection", snapshot_key),
            account_id=account_id,
            exchange=exchange,
            symbol=symbol,
            venue_symbol=fact.venue_symbol if fact else symbol,
            position_side=position_side,
            quantity=canonical_decimal(quantity),
            entry_price=canonical_decimal(entry_price),
            mark_price=fact.mark_price if fact else None,
            notional=(
                canonical_decimal(quantity * decimal_value(fact.mark_price))
                if fact and fact.mark_price is not None
                else (fact.notional if fact else None)
            ),
            realized_pnl=canonical_decimal(realized_pnl),
            unrealized_pnl=fact.unrealized_pnl if fact else "0",
            liquidation_price=fact.liquidation_price if fact else None,
            leverage=fact.leverage if fact else "1",
            margin_mode=fact.margin_mode if fact else None,
            source="LOCAL",
            sequence_number=last_sequence,
            source_version=(fact.source_version if fact else None),
            risk_source_snapshot_key=(fact.snapshot_key if fact else None),
            source_event_time=last_event_time,
            is_active=quantity > 0,
            is_current=True,
            raw_payload_json=canonical_json(
                {
                    "projection_trigger_id": projection_trigger_id,
                    "replayed_fill_count": applied_fills,
                    "base_snapshot_key": fact.snapshot_key if fact else None,
                    "previous_quantity": previous.quantity if previous else "0",
                }
            ),
        )
        self.session.add(projection)
        self.session.flush([projection])
        return projection

    def recover_projection(
        self,
        *,
        account_id: str,
        symbol: str | None = None,
    ) -> LedgerRecovery:
        canonical_filter = canonical_symbol(symbol) if symbol is not None else None
        order_stmt: Select[tuple[OrderSnapshot]] = select(OrderSnapshot).where(
            OrderSnapshot.account_id == account_id
        )
        fill_stmt: Select[tuple[FillEvent]] = select(FillEvent).where(
            FillEvent.account_id == account_id,
            FillEvent.projection_status != "BLOCKED",
        )
        if canonical_filter is not None:
            order_stmt = order_stmt.where(OrderSnapshot.symbol == canonical_filter)
            fill_stmt = fill_stmt.where(FillEvent.symbol == canonical_filter)
        order_rows = self.session.scalars(
            order_stmt.order_by(OrderSnapshot.source_event_time, OrderSnapshot.id)
        ).all()
        latest_orders: dict[tuple[str, str, str, str], OrderSnapshot] = {}
        for row in order_rows:
            key = (row.exchange, row.symbol, row.position_side, row.exchange_order_id)
            previous = latest_orders.get(key)
            if previous is None or (
                row.source_event_time,
                row.sequence_number if row.sequence_number is not None else -1,
                row.id,
            ) >= (
                previous.source_event_time,
                previous.sequence_number if previous.sequence_number is not None else -1,
                previous.id,
            ):
                latest_orders[key] = row

        recovered_orders = tuple(
            RecoveredOrder(
                account_id=row.account_id,
                exchange=row.exchange,
                symbol=row.symbol,
                position_side=row.position_side,
                exchange_order_id=row.exchange_order_id,
                client_order_id=row.client_order_id,
                action=row.action,
                status=row.status,
                original_quantity=row.original_quantity,
                executed_quantity=row.executed_quantity,
                remaining_quantity=row.remaining_quantity,
                cumulative_quote=row.cumulative_quote,
                average_price=row.average_price,
                is_terminal=row.is_terminal,
                source_event_time=row.source_event_time,
            )
            for row in sorted(
                latest_orders.values(),
                key=lambda item: (
                    item.exchange,
                    item.symbol,
                    item.position_side,
                    item.exchange_order_id,
                ),
            )
        )

        fact_stmt: Select[tuple[PositionSnapshot]] = select(PositionSnapshot).where(
            PositionSnapshot.account_id == account_id,
            PositionSnapshot.source.in_(("REST", "WEBSOCKET", "MIGRATION")),
        )
        if canonical_filter is not None:
            fact_stmt = fact_stmt.where(PositionSnapshot.symbol == canonical_filter)
        fact_rows = self.session.scalars(
            fact_stmt.order_by(PositionSnapshot.source_event_time, PositionSnapshot.id)
        ).all()
        projection: dict[tuple[str, str, str], dict[str, Any]] = {}
        for fact in fact_rows:
            key = (fact.exchange, fact.symbol, fact.position_side)
            previous = projection.get(key)
            fact_sequence = fact.sequence_number if fact.sequence_number is not None else -1
            if previous is None or (fact.source_event_time, fact_sequence, fact.id) >= (
                previous["base_time"],
                previous["base_sequence"],
                previous["base_id"],
            ):
                projection[key] = {
                    "quantity": decimal_value(fact.quantity),
                    "entry_price": decimal_value(fact.entry_price),
                    "realized_pnl": decimal_value(fact.realized_pnl),
                    "mark_price": fact.mark_price,
                    "notional": fact.notional,
                    "unrealized_pnl": fact.unrealized_pnl,
                    "liquidation_price": fact.liquidation_price,
                    "leverage": fact.leverage,
                    "margin_mode": fact.margin_mode,
                    "risk_source_snapshot_key": fact.snapshot_key,
                    "last_event_time": fact.source_event_time,
                    "last_sequence_number": fact.sequence_number,
                    "base_time": fact.source_event_time,
                    "base_sequence": fact_sequence,
                    "base_id": fact.id,
                }

        fill_rows = self.session.scalars(
            fill_stmt.order_by(
                FillEvent.event_time,
                func.coalesce(FillEvent.sequence_number, -1),
                FillEvent.exchange_trade_id,
                FillEvent.id,
            )
        ).all()
        for fill in fill_rows:
            key = (fill.exchange, fill.symbol, fill.position_side)
            state = projection.setdefault(
                key,
                {
                    "quantity": Decimal("0"),
                    "entry_price": Decimal("0"),
                    "realized_pnl": Decimal("0"),
                    "mark_price": None,
                    "notional": None,
                    "unrealized_pnl": "0",
                    "liquidation_price": None,
                    "leverage": "1",
                    "margin_mode": None,
                    "risk_source_snapshot_key": None,
                    "last_event_time": fill.event_time,
                    "last_sequence_number": fill.sequence_number,
                    "base_time": None,
                    "base_sequence": -1,
                    "base_id": -1,
                },
            )
            fill_sequence = fill.sequence_number if fill.sequence_number is not None else -1
            if state["base_time"] is not None and (
                fill.event_time < state["base_time"]
                or (
                    fill.event_time == state["base_time"]
                    and fill_sequence <= state["base_sequence"]
                )
            ):
                continue
            projected = state["quantity"] + _signed_quantity(
                fill.position_side,
                fill.side,
                decimal_value(fill.quantity),
            )
            if projected < 0:
                entity_key = (
                    f"{fill.exchange}|{account_id}|{fill.symbol}|{fill.position_side}"
                )
                raise ProjectionInvariantError(
                    f"Recovery encountered over-close fill {fill.exchange_trade_id}",
                    reason_code="OVER_CLOSE_FILL",
                    entity_key=entity_key,
                )
            quantity, entry_price = _position_after_fill(
                state["quantity"],
                state["entry_price"],
                fill.position_side,
                fill.side,
                decimal_value(fill.quantity),
                decimal_value(fill.price),
            )
            state["quantity"] = quantity
            state["entry_price"] = entry_price
            state["realized_pnl"] += decimal_value(fill.realized_pnl)
            state["last_event_time"] = fill.event_time
            state["last_sequence_number"] = fill.sequence_number
            if state["mark_price"] is not None:
                state["notional"] = canonical_decimal(
                    quantity * decimal_value(state["mark_price"])
                )

        recovered_positions = tuple(
            RecoveredPosition(
                account_id=account_id,
                exchange=exchange,
                symbol=position_symbol,
                position_side=position_side,
                quantity=canonical_decimal(state["quantity"]),
                entry_price=canonical_decimal(state["entry_price"]),
                realized_pnl=canonical_decimal(state["realized_pnl"]),
                mark_price=state["mark_price"],
                notional=state["notional"],
                unrealized_pnl=state["unrealized_pnl"],
                liquidation_price=state["liquidation_price"],
                leverage=state["leverage"],
                margin_mode=state["margin_mode"],
                risk_source_snapshot_key=state["risk_source_snapshot_key"],
                last_event_time=state["last_event_time"],
                last_sequence_number=state["last_sequence_number"],
            )
            for (exchange, position_symbol, position_side), state in sorted(projection.items())
        )
        return LedgerRecovery(orders=recovered_orders, positions=recovered_positions)

    def rebuild_current_projections(
        self,
        *,
        account_id: str,
        symbol: str | None = None,
    ) -> LedgerRecovery:
        canonical_filter = canonical_symbol(symbol) if symbol is not None else None
        recovery = self.recover_projection(account_id=account_id, symbol=canonical_filter)
        changed = 0
        for position in recovery.positions:
            current = self._current_position(
                position.exchange,
                account_id,
                position.symbol,
                position.position_side,
            )
            expected = (
                position.quantity,
                position.entry_price,
                position.realized_pnl,
                position.mark_price,
                position.notional,
                position.unrealized_pnl,
                position.liquidation_price,
                position.leverage,
                position.margin_mode,
                position.risk_source_snapshot_key,
                position.last_event_time,
                position.last_sequence_number,
            )
            actual = (
                current.quantity,
                current.entry_price,
                current.realized_pnl,
                current.mark_price,
                current.notional,
                current.unrealized_pnl,
                current.liquidation_price,
                current.leverage,
                current.margin_mode,
                current.risk_source_snapshot_key,
                current.source_event_time,
                current.sequence_number,
            ) if current else None
            if actual == expected:
                continue
            if current is not None:
                current.is_current = False
            snapshot_key = _bounded_key(
                "recovery",
                position.exchange,
                account_id,
                position.symbol,
                position.position_side,
                position.last_event_time.isoformat(),
                position.last_sequence_number,
                position.quantity,
                position.entry_price,
            )
            self.session.add(
                PositionSnapshot(
                    snapshot_key=snapshot_key,
                    fact_key=stable_fact_key("recovery", snapshot_key),
                    account_id=account_id,
                    exchange=position.exchange,
                    symbol=position.symbol,
                    venue_symbol=position.symbol,
                    position_side=position.position_side,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    mark_price=position.mark_price,
                    notional=position.notional,
                    realized_pnl=position.realized_pnl,
                    unrealized_pnl=position.unrealized_pnl,
                    liquidation_price=position.liquidation_price,
                    leverage=position.leverage,
                    margin_mode=position.margin_mode,
                    source="RECOVERY",
                    sequence_number=position.last_sequence_number,
                    risk_source_snapshot_key=position.risk_source_snapshot_key,
                    source_event_time=position.last_event_time,
                    is_active=decimal_value(position.quantity) > 0,
                    is_current=True,
                    raw_payload_json=canonical_json(
                        {"replayed_from": "hedge_fill_events_and_position_facts"}
                    ),
                )
            )
            changed += 1

        recovered_order_keys: set[tuple[str, str, str, str]] = set()
        for order in recovery.orders:
            recovered_order_keys.add(
                (order.exchange, order.symbol, order.position_side, order.exchange_order_id)
            )
            current = self._current_order(
                exchange=order.exchange,
                account_id=account_id,
                symbol=order.symbol,
                position_side=order.position_side,
                exchange_order_id=order.exchange_order_id,
            )
            values = {
                "client_order_id": order.client_order_id,
                "action": order.action,
                "status": order.status,
                "original_quantity": order.original_quantity,
                "executed_quantity": order.executed_quantity,
                "remaining_quantity": order.remaining_quantity,
                "cumulative_quote": order.cumulative_quote,
                "average_price": order.average_price,
                "is_terminal": order.is_terminal,
                "source_event_time": order.source_event_time,
            }
            if current is not None and all(
                getattr(current, key) == value for key, value in values.items()
            ):
                continue
            if current is None:
                current = CurrentOrderProjection(
                    exchange=order.exchange,
                    account_id=account_id,
                    symbol=order.symbol,
                    position_side=order.position_side,
                    exchange_order_id=order.exchange_order_id,
                    side="BUY" if order.position_side == "LONG" else "SELL",
                    revision=0,
                    **values,
                )
                self.session.add(current)
            else:
                for key, value in values.items():
                    setattr(current, key, value)
            current.revision += 1
            current.updated_at = utcnow()
            changed += 1

        stale_stmt = select(CurrentOrderProjection).where(
            CurrentOrderProjection.account_id == account_id
        )
        if canonical_filter is not None:
            stale_stmt = stale_stmt.where(CurrentOrderProjection.symbol == canonical_filter)
        for current in self.session.scalars(stale_stmt).all():
            key = (
                current.exchange,
                current.symbol,
                current.position_side,
                current.exchange_order_id,
            )
            if key not in recovered_order_keys:
                self.session.delete(current)
                changed += 1

        if changed:
            self._enqueue(
                aggregate_type="LedgerRecovery",
                aggregate_id=_bounded_key("ledger-recovery", account_id, canonical_filter or "*"),
                event_type="hedge.ledger.recovered",
                payload={
                    "account_id": account_id,
                    "symbol": canonical_filter,
                    "orders": len(recovery.orders),
                    "positions": len(recovery.positions),
                    "changed_projections": changed,
                },
                metadata=EventMetadata(
                    correlation_id=_bounded_key(
                        "recovery", account_id, canonical_filter or "*"
                    )
                ),
            )
        self.session.flush()
        return recovery

    def rebuild_current_position_snapshots(
        self,
        *,
        account_id: str,
        symbol: str | None = None,
    ) -> LedgerRecovery:
        """Compatibility wrapper; v1.5 rebuilds both order and position projections."""

        return self.rebuild_current_projections(account_id=account_id, symbol=symbol)

    def _current_position(
        self,
        exchange: str,
        account_id: str,
        symbol: str,
        position_side: str,
    ) -> PositionSnapshot | None:
        return self.session.scalar(
            select(PositionSnapshot)
            .where(
                PositionSnapshot.account_id == account_id,
                PositionSnapshot.exchange == exchange,
                PositionSnapshot.symbol == symbol,
                PositionSnapshot.position_side == position_side,
                PositionSnapshot.is_current.is_(True),
            )
            .order_by(PositionSnapshot.id.desc())
            .limit(1)
        )

    def _enqueue(
        self,
        *,
        aggregate_type: str,
        aggregate_id: str,
        event_type: str,
        payload: dict[str, Any],
        metadata: EventMetadata | None = None,
    ) -> EventOutbox:
        bounded_aggregate_id = (
            aggregate_id
            if len(aggregate_id) <= 255
            else _bounded_key(aggregate_type, aggregate_id)
        )
        metadata = metadata or EventMetadata(
            correlation_id=new_event_id(),
            observed_time=utcnow(),
        )
        if self.session.get_bind().dialect.name == "postgresql":
            scope = f"outbox|{aggregate_type}|{bounded_aggregate_id}"
            self.session.execute(
                text("SELECT pg_advisory_xact_lock(hashtextextended(:scope, 0))"),
                {"scope": scope},
            )
        sequence = self.session.scalar(
            select(func.coalesce(func.max(EventOutbox.aggregate_sequence), 0)).where(
                EventOutbox.aggregate_type == aggregate_type,
                EventOutbox.aggregate_id == bounded_aggregate_id,
            )
        )
        row = EventOutbox(
            aggregate_type=aggregate_type,
            aggregate_id=bounded_aggregate_id,
            event_type=event_type,
            aggregate_sequence=int(sequence or 0) + 1,
            correlation_id=metadata.correlation_id,
            causation_id=metadata.causation_id,
            payload_version=metadata.payload_version,
            event_version=metadata.event_version,
            contracts_version=metadata.contracts_version,
            schema_version=metadata.schema_version,
            exchange_time=(
                _naive_utc(metadata.exchange_time) if metadata.exchange_time else None
            ),
            observed_time=(
                _naive_utc(metadata.observed_time)
                if metadata.observed_time
                else utcnow()
            ),
            payload_json=canonical_json(payload),
            headers_json=canonical_json(metadata.headers()),
        )
        self.session.add(row)
        return row



def current_positions_query(account_id: str) -> Select[tuple[PositionSnapshot]]:
    return select(PositionSnapshot).where(
        PositionSnapshot.account_id == account_id,
        PositionSnapshot.is_current.is_(True),
    )


def count_ledger_rows(session: Session, models: Iterable[type[Any]]) -> dict[str, int]:
    return {
        model.__tablename__: int(session.scalar(select(func.count()).select_from(model)) or 0)
        for model in models
    }
