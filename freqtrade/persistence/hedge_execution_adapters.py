"""SQL adapters for Hedge execution ports.

The execution domain owns contracts and in-memory implementations.  This module is
the persistence-side adapter boundary and may depend on SQLAlchemy ORM models.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import Enum
import socket
from typing import Any, Mapping, Sequence
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from freqtrade.hedge.contracts.events import FillEvent as ContractFillEvent
from freqtrade.hedge.contracts.events import OutboxEvent
from freqtrade.hedge.contracts.types import PositionKey, PositionSide
from freqtrade.hedge.execution.action_group_store import (
    ActionGroupMember,
    ActionGroupMemberState,
    ActionGroupRecord,
)
from freqtrade.hedge.execution.ledger import InMemoryExecutionLedger, PositionProjection
from freqtrade.hedge.execution.idempotency import (
    IdempotencyReservation,
    ReservationState,
)
from freqtrade.hedge.execution.service import (
    ExecutionOrder,
    ExecutionResult,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide as ExecutionPositionSide,
)
from freqtrade.hedge.execution.state_machine import OrderLifecycle, OrderState
from freqtrade.persistence.hedge_models import (
    ActionGroupRow,
    AuditEvent,
    EventOutbox,
    ExecutionIdempotencyRow,
    ExecutionOrderStateRow,
    FillEvent as FillEventRow,
    OrderIntent as OrderIntentRow,
    PositionSnapshot,
    canonical_decimal,
)


def _naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


class SqlActionGroupRepository:
    """SQL-backed action-group adapter with optimistic idempotency semantics."""

    def __init__(self, session_factory: object) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory

    @staticmethod
    def _members_json(members: tuple[ActionGroupMember, ...]) -> str:
        return json.dumps(
            [
                {
                    "position_side": item.position_side.value,
                    "state": item.state.value,
                    "intent_id": item.intent_id,
                    "client_order_id": item.client_order_id,
                    "error": item.error,
                }
                for item in members
            ],
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _aware(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @classmethod
    def _from_row(cls, row: object) -> ActionGroupRecord:
        payload = json.loads(str(getattr(row, "members_json")))
        if not isinstance(payload, list):
            raise ValueError("action group members_json must be a list")
        members = tuple(
            ActionGroupMember(
                position_side=PositionSide(str(item["position_side"])),
                state=ActionGroupMemberState(str(item["state"])),
                intent_id=None if item.get("intent_id") is None else str(item["intent_id"]),
                client_order_id=(
                    None
                    if item.get("client_order_id") is None
                    else str(item["client_order_id"])
                ),
                error=None if item.get("error") is None else str(item["error"]),
            )
            for item in payload
            if isinstance(item, dict)
        )
        if len(members) != len(payload):
            raise ValueError("action group member entry must be an object")
        return ActionGroupRecord(
            action_group_id=UUID(str(getattr(row, "action_group_id"))),
            action_type=str(getattr(row, "action_type")),
            account_id=str(getattr(row, "account_id")),
            symbol=str(getattr(row, "symbol")),
            members=members,
            created_at=cls._aware(getattr(row, "created_at")),
            updated_at=cls._aware(getattr(row, "updated_at")),
        )

    def put(self, group: ActionGroupRecord) -> None:
        if not isinstance(group, ActionGroupRecord):
            raise TypeError("group must be an ActionGroupRecord")
        with self._session_factory.begin() as session:  # type: ignore[operator]
            row = session.scalar(
                select(ActionGroupRow).where(
                    ActionGroupRow.action_group_id == str(group.action_group_id)
                )
            )
            if row is not None:
                if self._from_row(row) != group:
                    raise ValueError("action group already exists with conflicting data")
                return
            session.add(
                ActionGroupRow(
                    action_group_id=str(group.action_group_id),
                    action_type=group.action_type,
                    account_id=group.account_id,
                    symbol=group.symbol,
                    members_json=self._members_json(group.members),
                    created_at=group.created_at.astimezone(UTC).replace(tzinfo=None),
                    updated_at=group.updated_at.astimezone(UTC).replace(tzinfo=None),
                )
            )

    def get(self, action_group_id: UUID) -> ActionGroupRecord | None:
        with self._session_factory() as session:  # type: ignore[operator]
            row = session.scalar(
                select(ActionGroupRow).where(
                    ActionGroupRow.action_group_id == str(action_group_id)
                )
            )
            return None if row is None else self._from_row(row)

    def update_member(
        self,
        action_group_id: UUID,
        member: ActionGroupMember,
    ) -> ActionGroupRecord:
        with self._session_factory.begin() as session:  # type: ignore[operator]
            row = session.scalar(
                select(ActionGroupRow).where(
                    ActionGroupRow.action_group_id == str(action_group_id)
                )
            )
            if row is None:
                raise KeyError(action_group_id)
            current = self._from_row(row)
            found = False
            members: list[ActionGroupMember] = []
            for item in current.members:
                if item.position_side is member.position_side:
                    members.append(member)
                    found = True
                else:
                    members.append(item)
            if not found:
                raise KeyError(member.position_side)
            updated = replace(
                current,
                members=tuple(members),
                updated_at=datetime.now(UTC),
            )
            row.members_json = self._members_json(updated.members)
            row.updated_at = updated.updated_at.replace(tzinfo=None)
            return updated


class SqlExecutionLedger(InMemoryExecutionLedger):
    """Execution ledger adapter backed by SQL and an in-process read cache."""

    def __init__(self, session_factory: object) -> None:
        super().__init__()
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory

    def seed_position(
        self,
        *,
        position_key: PositionKey,
        quantity: Decimal,
        average_entry_price: Decimal,
        realized_pnl: Decimal = Decimal("0"),
        fees: Decimal = Decimal("0"),
        funding: Decimal = Decimal("0"),
        observed_at: datetime | None = None,
    ) -> PositionProjection:
        projection = super().seed_position(
            position_key=position_key,
            quantity=quantity,
            average_entry_price=average_entry_price,
            realized_pnl=realized_pnl,
            fees=fees,
            funding=funding,
            observed_at=observed_at,
        )
        timestamp = observed_at or datetime.now(UTC)
        naive = _naive_utc(timestamp)
        with self._session_factory.begin() as session:  # type: ignore[operator]
            current = session.scalar(
                select(PositionSnapshot).where(
                    PositionSnapshot.exchange == position_key.exchange,
                    PositionSnapshot.account_id == position_key.account_id,
                    PositionSnapshot.symbol == position_key.symbol,
                    PositionSnapshot.position_side == position_key.position_side.value,
                    PositionSnapshot.is_current.is_(True),
                )
            )
            expected = (canonical_decimal(quantity), canonical_decimal(average_entry_price))
            if current is not None and (current.quantity, current.entry_price) == expected:
                return projection
            if current is not None:
                current.is_current = False
            identity = uuid4().hex
            session.add(
                PositionSnapshot(
                    snapshot_key=f"ADOPTION-SEED:{position_key.lock_name}:{identity}",
                    fact_key=None,
                    account_id=position_key.account_id,
                    exchange=position_key.exchange,
                    symbol=position_key.symbol,
                    venue_symbol=position_key.symbol.replace("/", ""),
                    position_side=position_key.position_side.value,
                    quantity=canonical_decimal(quantity),
                    entry_price=canonical_decimal(average_entry_price),
                    mark_price=None,
                    notional=None,
                    realized_pnl=canonical_decimal(realized_pnl),
                    unrealized_pnl="0",
                    liquidation_price=None,
                    leverage="1",
                    margin_mode="cross",
                    source="REST",
                    sequence_number=None,
                    source_version="CLEAN_MAINLINE_ADOPTION_SEED",
                    risk_source_snapshot_key=None,
                    source_event_time=naive,
                    observed_at=naive,
                    is_active=quantity > 0,
                    is_current=True,
                    raw_payload_json=json.dumps(
                        {
                            "authority": "clean-mainline",
                            "fees": str(fees),
                            "funding": str(funding),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
        return projection

    @staticmethod
    def _identity(order: object) -> tuple[str, str, str, str, str]:
        intent = getattr(order, "intent", None)
        client_order_id = str(getattr(order, "client_order_id", "")).strip()
        if intent is None or not client_order_id:
            raise ValueError("order must expose intent and client_order_id")
        metadata = getattr(intent, "metadata", {})
        exchange = str(metadata.get("exchange", "binance")).strip().lower()
        account_id = str(getattr(intent, "account_id", "")).strip()
        symbol = str(getattr(intent, "symbol", "")).strip()
        position_side = str(
            getattr(getattr(intent, "position_side", None), "value", "")
        ).strip()
        if not all((exchange, account_id, symbol, position_side)):
            raise ValueError("execution order identity is incomplete")
        return exchange, account_id, symbol, position_side, client_order_id

    @staticmethod
    def _correlation_id(order: object) -> str:
        intent = getattr(order, "intent")
        group = getattr(intent, "action_group_id", None)
        return str(group or getattr(intent, "intent_id"))

    def record(
        self,
        *,
        order: object,
        event_type: str,
        fill: ContractFillEvent | None = None,
        outbox: OutboxEvent | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        exchange, account_id, symbol, position_side, client_order_id = self._identity(order)
        event_name = str(event_type).strip().upper()
        if not event_name:
            raise ValueError("event_type is required")
        encoded_payload = json.dumps(
            dict(payload or {}),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        correlation_id = self._correlation_id(order)
        lifecycle = getattr(order, "lifecycle")
        intent = getattr(order, "intent")

        with self._session_factory.begin() as session:  # type: ignore[operator]
            # The network-after transaction always carries the latest order state
            # together with Intent/Fill/Audit/Outbox.  Earlier state transitions may
            # exist, but a successful exchange fact cannot be ledgered without its
            # authoritative current projection in the same commit.
            SqlExecutionStore(self._session_factory)._put_in_session(session, order)
            intent_id = str(getattr(intent, "intent_id"))
            existing_intent = session.scalar(
                select(OrderIntentRow).where(OrderIntentRow.intent_id == intent_id)
            )
            increases = str(getattr(getattr(intent, "action", None), "value", "")) in {
                "OPEN",
                "INCREASE",
            }
            long_side = position_side == "LONG"
            order_side = "BUY" if increases == long_side else "SELL"
            request_payload = {
                "account_id": account_id,
                "exchange": exchange,
                "symbol": symbol,
                "position_side": position_side,
                "action": getattr(intent.action, "value", str(intent.action)),
                "quantity": str(intent.quantity),
                "limit_price": (
                    None if intent.limit_price is None else str(intent.limit_price)
                ),
                "reduce_only": bool(intent.reduce_only),
                "client_order_id": client_order_id,
            }
            if existing_intent is None:
                session.add(
                    OrderIntentRow(
                        intent_id=intent_id,
                        account_id=account_id,
                        exchange=exchange,
                        symbol=symbol,
                        position_side=position_side,
                        action=getattr(intent.action, "value", str(intent.action)),
                        side=order_side,
                        order_type=getattr(
                            intent.order_type, "value", str(intent.order_type)
                        ),
                        requested_quantity=canonical_decimal(intent.quantity),
                        requested_price=(
                            None
                            if intent.limit_price is None
                            else canonical_decimal(intent.limit_price)
                        ),
                        reduce_only=bool(intent.reduce_only),
                        status=getattr(lifecycle.status, "value", str(lifecycle.status)),
                        idempotency_key=str(intent.idempotency_key),
                        correlation_id=correlation_id,
                        target_snapshot_json="{}",
                        approved_quantity=canonical_decimal(
                            getattr(order, "approved_quantity")
                        ),
                        reason_codes_json="[]",
                        request_payload_json=json.dumps(
                            request_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        approved_at=_naive_utc(getattr(order, "created_at")),
                    )
                )
            else:
                expected_key = (
                    existing_intent.idempotency_key,
                    existing_intent.account_id,
                    existing_intent.symbol,
                    existing_intent.position_side,
                )
                actual_key = (
                    str(intent.idempotency_key),
                    account_id,
                    symbol,
                    position_side,
                )
                if expected_key != actual_key:
                    raise ValueError("intent identity conflicts with durable SQL evidence")
                existing_intent.status = getattr(
                    lifecycle.status, "value", str(lifecycle.status)
                )
                existing_intent.approved_quantity = canonical_decimal(
                    getattr(order, "approved_quantity")
                )

            if fill is not None:
                key = fill.position_key
                if key is None:
                    raise ValueError("execution FillEvent requires position_key")
                existing = session.scalar(
                    select(FillEventRow).where(
                        FillEventRow.exchange == key.exchange,
                        FillEventRow.account_id == key.account_id,
                        FillEventRow.symbol == key.symbol,
                        FillEventRow.exchange_trade_id == fill.trade_id,
                    )
                )
                if existing is None:
                    session.add(
                        FillEventRow(
                            exchange=key.exchange,
                            account_id=key.account_id,
                            symbol=key.symbol,
                            position_side=key.position_side.value,
                            exchange_trade_id=fill.trade_id,
                            exchange_order_id=(
                                getattr(lifecycle, "exchange_order_id", None)
                                or client_order_id
                            ),
                            client_order_id=fill.client_order_id,
                            intent_id=str(getattr(intent, "intent_id")),
                            correlation_id=correlation_id,
                            side=fill.order_side.value,
                            action=fill.action.value,
                            quantity=canonical_decimal(fill.quantity),
                            price=canonical_decimal(fill.price),
                            quote_quantity=canonical_decimal(fill.quantity * fill.price),
                            fee_amount=canonical_decimal(fill.fee),
                            fee_currency=fill.fee_currency,
                            source="LOCAL",
                            event_time=_naive_utc(fill.exchange_time),
                            observed_at=_naive_utc(fill.observed_time),
                            raw_payload_json=encoded_payload,
                        )
                    )
                else:
                    expected = (
                        canonical_decimal(fill.quantity),
                        canonical_decimal(fill.price),
                        fill.client_order_id,
                    )
                    actual = (
                        existing.quantity,
                        existing.price,
                        existing.client_order_id,
                    )
                    if actual != expected:
                        raise ValueError(
                            "duplicate trade_id contains conflicting fill data"
                        )

            session.add(
                AuditEvent(
                    account_id=account_id,
                    exchange=exchange,
                    event_type=event_name,
                    entity_type="EXECUTION_ORDER",
                    entity_id=client_order_id,
                    severity="INFO",
                    reason_code=getattr(lifecycle, "reason", None),
                    correlation_id=correlation_id,
                    actor=str(getattr(intent, "metadata", {}).get("actor", "execution-runtime")),
                    payload_json=encoded_payload,
                    occurred_at=_naive_utc(getattr(lifecycle, "updated_at")),
                )
            )

            if outbox is not None:
                aggregate_sequence = int(
                    session.scalar(
                        select(
                            func.coalesce(func.max(EventOutbox.aggregate_sequence), 0)
                        ).where(
                            EventOutbox.aggregate_type == "EXECUTION_ORDER",
                            EventOutbox.aggregate_id == client_order_id,
                        )
                    )
                    or 0
                ) + 1
                session.add(
                    EventOutbox(
                        event_id=str(outbox.event_id),
                        aggregate_type="EXECUTION_ORDER",
                        aggregate_id=client_order_id,
                        event_type=outbox.event_type,
                        aggregate_sequence=aggregate_sequence,
                        correlation_id=outbox.correlation_id or correlation_id,
                        payload_version=outbox.payload_version,
                        observed_time=_naive_utc(outbox.occurred_at),
                        payload_json=json.dumps(
                            dict(outbox.payload),
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        ),
                        headers_json=json.dumps(
                            {"source": "PAPER", "durability": "SQL"},
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        occurred_at=_naive_utc(outbox.occurred_at),
                        available_at=_naive_utc(outbox.occurred_at),
                        status=(
                            "PENDING"
                            if outbox.published_at is None
                            else "PUBLISHED"
                        ),
                        attempts=outbox.attempts,
                        published_at=(
                            None
                            if outbox.published_at is None
                            else _naive_utc(outbox.published_at)
                        ),
                    )
                )

        super().record(
            order=order,
            event_type=event_name,
            fill=fill,
            outbox=outbox,
            payload=payload,
        )

    def mark_published(
        self,
        event_id: str,
        *,
        published_at: datetime | None = None,
    ) -> None:
        timestamp = published_at or datetime.now(UTC)
        with self._session_factory.begin() as session:  # type: ignore[operator]
            row = session.scalar(
                select(EventOutbox).where(EventOutbox.event_id == str(event_id))
            )
            if row is None:
                raise KeyError(event_id)
            row.status = "PUBLISHED"
            row.published_at = _naive_utc(timestamp)
            row.attempts = int(row.attempts or 0) + 1
        super().mark_published(event_id, published_at=timestamp)

    def mark_publish_attempt(self, event_id: str) -> None:
        with self._session_factory.begin() as session:  # type: ignore[operator]
            row = session.scalar(
                select(EventOutbox).where(EventOutbox.event_id == str(event_id))
            )
            if row is None:
                raise KeyError(event_id)
            row.attempts = int(row.attempts or 0) + 1
        super().mark_publish_attempt(event_id)


# ---------------------------------------------------------------------------
# Authoritative execution current-state and idempotency adapters
# ---------------------------------------------------------------------------


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _json_encode(value: object) -> object:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Decimal):
        return {"__type__": "decimal", "value": str(value)}
    if isinstance(value, UUID):
        return {"__type__": "uuid", "value": str(value)}
    if isinstance(value, datetime):
        return {"__type__": "datetime", "value": _aware_utc(value).isoformat()}
    if isinstance(value, Enum):
        return {"__type__": "enum", "value": str(value.value)}
    if isinstance(value, Mapping):
        return {str(key): _json_encode(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_encode(item) for item in value]
    raise TypeError(f"unsupported execution metadata value: {type(value).__name__}")


def _json_decode(value: object) -> object:
    if isinstance(value, list):
        return tuple(_json_decode(item) for item in value)
    if not isinstance(value, dict):
        return value
    tag = value.get("__type__")
    if tag == "decimal":
        return Decimal(str(value["value"]))
    if tag == "uuid":
        return UUID(str(value["value"]))
    if tag == "datetime":
        return datetime.fromisoformat(str(value["value"]))
    if tag == "enum":
        # Metadata enums are intentionally restored as their stable wire value.
        return str(value["value"])
    return {str(key): _json_decode(item) for key, item in value.items()}


def _metadata_json(metadata: Mapping[str, Any]) -> str:
    return json.dumps(
        _json_encode(dict(metadata)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _metadata_from_json(raw: str) -> Mapping[str, Any]:
    payload = json.loads(raw or "{}")
    decoded = _json_decode(payload)
    if not isinstance(decoded, Mapping):
        raise ValueError("execution metadata_json must decode to a mapping")
    return decoded


class SqlExecutionStore:
    """SQL-authoritative implementation of ``ExecutionStorePort``.

    The adapter enforces lifecycle monotonicity at the durable boundary.  A
    checkpoint may still contain planner/bucket state, but it is no longer the
    authority for execution-order status or filled quantity.
    """

    def __init__(self, session_factory: object) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory

    @staticmethod
    def _from_row(row: ExecutionOrderStateRow) -> ExecutionOrder:
        intent = OrderIntent(
            account_id=row.account_id,
            symbol=row.symbol,
            position_side=ExecutionPositionSide(row.position_side),
            action=IntentAction(row.action),
            quantity=Decimal(row.quantity),
            idempotency_key=row.idempotency_key,
            order_type=OrderType(row.order_type),
            limit_price=(None if row.limit_price is None else Decimal(row.limit_price)),
            reduce_only=bool(row.reduce_only),
            intent_id=UUID(row.intent_id),
            action_group_id=(
                None if row.action_group_id is None else UUID(row.action_group_id)
            ),
            metadata=_metadata_from_json(row.metadata_json),
        )
        lifecycle = OrderLifecycle(
            status=OrderState(row.lifecycle_status),
            filled_quantity=Decimal(row.lifecycle_filled_quantity),
            average_price=(
                None
                if row.lifecycle_average_price is None
                else Decimal(row.lifecycle_average_price)
            ),
            exchange_order_id=row.lifecycle_exchange_order_id,
            version=int(row.lifecycle_version),
            updated_at=_aware_utc(row.lifecycle_updated_at),
            reason=row.lifecycle_reason,
        )
        return ExecutionOrder(
            intent=intent,
            client_order_id=row.client_order_id,
            approved_quantity=Decimal(row.approved_quantity),
            lifecycle=lifecycle,
            created_at=_aware_utc(row.created_at),
        )

    @staticmethod
    def _canonical_order(order: ExecutionOrder) -> ExecutionOrder:
        canonical_metadata = _metadata_from_json(_metadata_json(order.intent.metadata))
        return replace(
            order,
            intent=replace(order.intent, metadata=canonical_metadata),
        )

    @staticmethod
    def _same(row: ExecutionOrderStateRow, order: ExecutionOrder) -> bool:
        return SqlExecutionStore._from_row(row) == SqlExecutionStore._canonical_order(order)

    @staticmethod
    def _assert_immutable_identity(
        row: ExecutionOrderStateRow,
        order: ExecutionOrder,
    ) -> None:
        current = SqlExecutionStore._from_row(row)
        candidate = SqlExecutionStore._canonical_order(order)
        if (
            current.intent != candidate.intent
            or current.client_order_id != candidate.client_order_id
            or current.approved_quantity != candidate.approved_quantity
            or current.created_at != candidate.created_at
        ):
            raise ValueError(
                "durable execution lifecycle update attempted to mutate order identity"
            )

    @staticmethod
    def _apply(row: ExecutionOrderStateRow, order: ExecutionOrder) -> None:
        intent = order.intent
        lifecycle = order.lifecycle
        row.intent_id = str(intent.intent_id)
        row.account_id = intent.account_id
        row.exchange = str(intent.metadata.get("exchange", "binance")).strip().lower()
        row.symbol = intent.symbol
        row.position_side = intent.position_side.value
        row.action = intent.action.value
        row.order_type = intent.order_type.value
        row.quantity = canonical_decimal(intent.quantity)
        row.limit_price = (
            None if intent.limit_price is None else canonical_decimal(intent.limit_price)
        )
        row.reduce_only = bool(intent.reduce_only)
        row.idempotency_key = intent.idempotency_key
        row.action_group_id = (
            None if intent.action_group_id is None else str(intent.action_group_id)
        )
        row.metadata_json = _metadata_json(intent.metadata)
        row.approved_quantity = canonical_decimal(order.approved_quantity)
        row.risk_reason_codes_json = json.dumps([], separators=(",", ":"))
        row.lifecycle_status = lifecycle.status.value
        row.lifecycle_filled_quantity = canonical_decimal(lifecycle.filled_quantity)
        row.lifecycle_average_price = (
            None
            if lifecycle.average_price is None
            else canonical_decimal(lifecycle.average_price)
        )
        row.lifecycle_exchange_order_id = lifecycle.exchange_order_id
        row.lifecycle_reason = lifecycle.reason
        row.lifecycle_version = lifecycle.version
        row.approved_at = _naive_utc(order.created_at)
        row.lifecycle_updated_at = _naive_utc(lifecycle.updated_at)
        row.created_at = _naive_utc(order.created_at)
        row.updated_at = datetime.now(UTC).replace(tzinfo=None)

    def _put_in_session(self, session: object, order: ExecutionOrder) -> None:
        statement = select(ExecutionOrderStateRow).where(
            ExecutionOrderStateRow.client_order_id == order.client_order_id
        )
        if session.get_bind().dialect.name == "postgresql":  # type: ignore[attr-defined]
            statement = statement.with_for_update()
        row = session.scalar(statement)  # type: ignore[attr-defined]
        if row is None:
            row = ExecutionOrderStateRow(
                client_order_id=order.client_order_id,
                intent_id=str(order.intent.intent_id),
                account_id=order.intent.account_id,
                exchange="binance",
                symbol=order.intent.symbol,
                position_side=order.intent.position_side.value,
                action=order.intent.action.value,
                order_type=order.intent.order_type.value,
                quantity="0",
                limit_price=None,
                reduce_only=False,
                idempotency_key=order.intent.idempotency_key,
                action_group_id=None,
                metadata_json="{}",
                approved_quantity="0",
                risk_reason_codes_json="[]",
                lifecycle_status=order.lifecycle.status.value,
                lifecycle_filled_quantity="0",
                lifecycle_average_price=None,
                lifecycle_exchange_order_id=None,
                lifecycle_reason=None,
                lifecycle_version=order.lifecycle.version,
                approved_at=_naive_utc(order.created_at),
                lifecycle_updated_at=_naive_utc(order.lifecycle.updated_at),
                created_at=_naive_utc(order.created_at),
                updated_at=datetime.now(UTC).replace(tzinfo=None),
            )
            self._apply(row, order)
            session.add(row)  # type: ignore[attr-defined]
            return
        current_version = int(row.lifecycle_version)
        next_version = int(order.lifecycle.version)
        if next_version < current_version:
            raise ValueError("refusing to overwrite a newer durable execution order")
        if next_version == current_version:
            if not self._same(row, order):
                raise ValueError("same lifecycle version contains conflicting durable data")
            return
        self._assert_immutable_identity(row, order)
        self._apply(row, order)

    def put(self, order: ExecutionOrder) -> None:
        if not isinstance(order, ExecutionOrder):
            raise TypeError("order must be an ExecutionOrder")
        with self._session_factory.begin() as session:  # type: ignore[operator]
            self._put_in_session(session, order)

    def get_by_client_order_id(self, client_order_id: str) -> ExecutionOrder | None:
        normalized = str(client_order_id).strip()
        if not normalized:
            raise ValueError("client_order_id is required")
        with self._session_factory() as session:  # type: ignore[operator]
            row = session.scalar(
                select(ExecutionOrderStateRow).where(
                    ExecutionOrderStateRow.client_order_id == normalized
                )
            )
            return None if row is None else self._from_row(row)

    def has_unresolved_unknown(
        self,
        leg_key: tuple[str, str, ExecutionPositionSide],
    ) -> bool:
        account_id, symbol, position_side = leg_key
        with self._session_factory() as session:  # type: ignore[operator]
            count = session.scalar(
                select(func.count(ExecutionOrderStateRow.id)).where(
                    ExecutionOrderStateRow.account_id == account_id,
                    ExecutionOrderStateRow.symbol == symbol,
                    ExecutionOrderStateRow.position_side == position_side.value,
                    ExecutionOrderStateRow.lifecycle_status == OrderState.UNKNOWN.value,
                )
            )
            return bool(count)

    def list_orders(self) -> Sequence[ExecutionOrder]:
        with self._session_factory() as session:  # type: ignore[operator]
            rows = session.scalars(
                select(ExecutionOrderStateRow).order_by(
                    ExecutionOrderStateRow.created_at,
                    ExecutionOrderStateRow.client_order_id,
                )
            ).all()
            return tuple(self._from_row(row) for row in rows)


class SqlExecutionIdempotencyStore:
    """Durable reservation store with expiry-based crash recovery.

    ``COMPLETED`` rows point at the authoritative execution-order projection.
    An abandoned ``IN_FLIGHT`` reservation may be reclaimed only after its
    lease expires; this prevents both blind duplicate submits and permanent
    deadlock after process termination.
    """

    def __init__(
        self,
        session_factory: object,
        execution_store: SqlExecutionStore,
        *,
        lease_seconds: int = 300,
        owner_id: str | None = None,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        if not isinstance(execution_store, SqlExecutionStore):
            raise TypeError("execution_store must be SqlExecutionStore")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self._session_factory = session_factory
        self._execution_store = execution_store
        self._lease = timedelta(seconds=lease_seconds)
        self._owner_id = owner_id or (
            f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
        )

    @staticmethod
    def _key(key: object) -> str:
        if not isinstance(key, str):
            raise TypeError("idempotency key must be a string")
        normalized = key.strip()
        if not normalized or len(normalized) > 255:
            raise ValueError("idempotency key is invalid")
        return normalized

    def reserve(self, key: str) -> IdempotencyReservation[ExecutionResult]:
        normalized = self._key(key)
        now = datetime.now(UTC).replace(tzinfo=None)
        expires = now + self._lease
        try:
            with self._session_factory.begin() as session:  # type: ignore[operator]
                statement = select(ExecutionIdempotencyRow).where(
                    ExecutionIdempotencyRow.idempotency_key == normalized
                )
                if session.get_bind().dialect.name == "postgresql":
                    statement = statement.with_for_update()
                row = session.scalar(statement)
                if row is None:
                    session.add(
                        ExecutionIdempotencyRow(
                            idempotency_key=normalized,
                            state=ReservationState.IN_FLIGHT.value,
                            client_order_id=None,
                            lease_owner=self._owner_id,
                            lease_expires_at=expires,
                            created_at=now,
                            updated_at=now,
                        )
                    )
                    return IdempotencyReservation(ReservationState.NEW)
                if row.state == ReservationState.COMPLETED.value:
                    if not row.client_order_id:
                        raise RuntimeError("completed idempotency row lacks client order id")
                    order = self._execution_store.get_by_client_order_id(row.client_order_id)
                    if order is None:
                        raise RuntimeError("completed idempotency result references missing order")
                    return IdempotencyReservation(
                        ReservationState.COMPLETED,
                        ExecutionResult(order=order, idempotent_replay=True),
                    )
                if row.state != ReservationState.IN_FLIGHT.value:
                    raise RuntimeError(f"unknown idempotency state: {row.state}")
                if row.lease_expires_at is not None and row.lease_expires_at > now:
                    return IdempotencyReservation(ReservationState.IN_FLIGHT)
                row.lease_owner = self._owner_id
                row.lease_expires_at = expires
                row.updated_at = now
                return IdempotencyReservation(ReservationState.NEW)
        except IntegrityError:
            # A concurrent transaction won the unique-key insert.  Read its
            # committed state rather than treating the race as a submission error.
            with self._session_factory() as session:  # type: ignore[operator]
                row = session.get(ExecutionIdempotencyRow, normalized)
                if row is None:
                    raise
                if row.state == ReservationState.COMPLETED.value and row.client_order_id:
                    order = self._execution_store.get_by_client_order_id(row.client_order_id)
                    if order is None:
                        raise RuntimeError("idempotency row references missing order")
                    return IdempotencyReservation(
                        ReservationState.COMPLETED,
                        ExecutionResult(order=order, idempotent_replay=True),
                    )
                return IdempotencyReservation(ReservationState.IN_FLIGHT)

    def complete(self, key: str, value: ExecutionResult) -> None:
        normalized = self._key(key)
        if not isinstance(value, ExecutionResult):
            raise TypeError("value must be an ExecutionResult")
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._session_factory.begin() as session:  # type: ignore[operator]
            statement = select(ExecutionIdempotencyRow).where(
                ExecutionIdempotencyRow.idempotency_key == normalized
            )
            if session.get_bind().dialect.name == "postgresql":
                statement = statement.with_for_update()
            row = session.scalar(statement)
            if row is None:
                raise KeyError(f"idempotency key was not reserved: {normalized}")
            if row.state == ReservationState.COMPLETED.value:
                if row.client_order_id != value.order.client_order_id:
                    raise ValueError("idempotency key completed with conflicting order")
                return
            if row.lease_owner != self._owner_id:
                raise PermissionError(
                    "idempotency reservation is owned by another execution worker"
                )
            if row.lease_expires_at is None or row.lease_expires_at <= now:
                raise TimeoutError(
                    "idempotency reservation lease expired before completion"
                )
            # Persist the execution projection and completion pointer in one
            # database transaction.  A crash can no longer leave a durable
            # order paired with an indefinitely IN_FLIGHT idempotency record.
            self._execution_store._put_in_session(session, value.order)
            row.state = ReservationState.COMPLETED.value
            row.client_order_id = value.order.client_order_id
            row.lease_owner = None
            row.lease_expires_at = None
            row.updated_at = now

    def recover_completed(self, key: str, value: ExecutionResult) -> None:
        """Converge a reservation to a known durable execution result.

        This recovery-only operation is safe because the execution projection
        has a unique idempotency key and immutable order identity.  It is used
        after a process crash when the order fact exists but an older release
        may have left the reservation IN_FLIGHT.
        """

        normalized = self._key(key)
        if not isinstance(value, ExecutionResult):
            raise TypeError("value must be an ExecutionResult")
        if value.order.intent.idempotency_key != normalized:
            raise ValueError("recovery result does not match the idempotency key")
        now = datetime.now(UTC).replace(tzinfo=None)
        with self._session_factory.begin() as session:  # type: ignore[operator]
            statement = select(ExecutionIdempotencyRow).where(
                ExecutionIdempotencyRow.idempotency_key == normalized
            )
            if session.get_bind().dialect.name == "postgresql":
                statement = statement.with_for_update()
            row = session.scalar(statement)
            self._execution_store._put_in_session(session, value.order)
            if row is None:
                session.add(
                    ExecutionIdempotencyRow(
                        idempotency_key=normalized,
                        state=ReservationState.COMPLETED.value,
                        client_order_id=value.order.client_order_id,
                        lease_owner=None,
                        lease_expires_at=None,
                        created_at=now,
                        updated_at=now,
                    )
                )
                return
            if (
                row.state == ReservationState.COMPLETED.value
                and row.client_order_id != value.order.client_order_id
            ):
                raise ValueError("idempotency key completed with conflicting order")
            row.state = ReservationState.COMPLETED.value
            row.client_order_id = value.order.client_order_id
            row.lease_owner = None
            row.lease_expires_at = None
            row.updated_at = now

    def release(self, key: str) -> None:
        normalized = self._key(key)
        with self._session_factory.begin() as session:  # type: ignore[operator]
            session.execute(
                delete(ExecutionIdempotencyRow).where(
                    ExecutionIdempotencyRow.idempotency_key == normalized,
                    ExecutionIdempotencyRow.state == ReservationState.IN_FLIGHT.value,
                    ExecutionIdempotencyRow.lease_owner == self._owner_id,
                )
            )
