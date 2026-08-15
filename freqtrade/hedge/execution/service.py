"""Port-driven hedge execution service with no concrete exchange or database dependency."""

from __future__ import annotations

import hashlib
from collections import deque
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum, StrEnum
from threading import RLock
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Protocol, Sequence
from uuid import UUID, uuid4

from .client_order_id import build_client_order_id, validate_client_order_id
from .idempotency import IdempotencyPort, ReservationState
from .kill_switch import KillSwitch
from .state_machine import InvalidOrderTransition, OrderLifecycle, OrderState

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SYMBOL_ALLOWED = re.compile(r"^[A-Za-z0-9/_:-]+$")
_EXTERNAL_STATES = frozenset(
    {
        OrderState.ACKNOWLEDGED,
        OrderState.PARTIAL,
        OrderState.FILLED,
        OrderState.CANCELED,
        OrderState.REJECTED,
        OrderState.UNKNOWN,
    }
)


class PositionSide(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class IntentAction(StrEnum):
    OPEN = "OPEN"
    INCREASE = "INCREASE"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"

    @property
    def reduces_risk(self) -> bool:
        return self in {IntentAction.REDUCE, IntentAction.CLOSE}


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


def _decimal(value: object, *, field_name: str) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError(f"{field_name} must use an exact decimal value")
    try:
        result = Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


def _aware(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value


def _text(
    value: object,
    *,
    field_name: str,
    max_length: int,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        if optional:
            return None
        raise ValueError(f"{field_name} is required")
    if len(result) > max_length or _CONTROL.search(result):
        raise ValueError(f"{field_name} is invalid")
    return result


def _normalize_symbol(value: object) -> str:
    raw = _text(value, field_name="symbol", max_length=64)
    if raw is None:  # pragma: no cover - required text cannot return None
        raise ValueError("symbol is required")
    raw = raw.upper()
    if not _SYMBOL_ALLOWED.fullmatch(raw):
        raise ValueError("symbol contains unsupported characters")
    parts = raw.split(":")
    if len(parts) > 2:
        raise ValueError("symbol contains multiple settlement suffixes")
    normalized = re.sub(r"[/_-]", "", parts[0])
    if not normalized or not normalized.isascii() or not normalized.isalnum():
        raise ValueError("symbol must contain ASCII alphanumeric characters")
    if len(parts) == 2:
        settle = parts[1]
        if not settle.isascii() or not settle.isalnum() or not normalized.endswith(settle):
            raise ValueError("settlement suffix must match the normalized quote asset")
    return normalized


def _uuid(value: object, *, field_name: str, optional: bool = False) -> UUID | None:
    if value is None and optional:
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"{field_name} must be a UUID") from exc


def _reason_codes(value: object, *, field_name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence, not a string")
    try:
        source = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise TypeError(f"{field_name} must be a sequence") from exc
    if len(source) > 32:
        raise ValueError(f"{field_name} has too many values")
    result: list[str] = []
    seen: set[str] = set()
    for item in source:
        text = _text(item, field_name=field_name, max_length=64)
        if text is None:  # pragma: no cover
            continue
        if text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _stable_sort_key(value: object) -> str:
    return f"{type(value).__qualname__}:{value!r}"


def _freeze_value(value: object, *, depth: int = 0) -> object:
    if depth > 12:
        raise ValueError("metadata nesting is too deep")
    if value is None or isinstance(value, (str, bool, int, Decimal, UUID, date, Enum)):
        if isinstance(value, str) and (
            len(value) > 65536 or _CONTROL.search(value)
        ):
            raise ValueError("metadata string is invalid")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata float must be finite")
        return value
    if isinstance(value, Mapping):
        if len(value) > 1000:
            raise ValueError("metadata mapping is too large")
        converted: dict[str, object] = {}
        for raw_key, raw_value in value.items():
            key = _text(raw_key, field_name="metadata key", max_length=256)
            if key is None:  # pragma: no cover
                continue
            if key in converted:
                raise ValueError("metadata keys collide after normalization")
            converted[key] = _freeze_value(raw_value, depth=depth + 1)
        return MappingProxyType(converted)
    if isinstance(value, (list, tuple)):
        if len(value) > 1000:
            raise ValueError("metadata sequence is too large")
        return tuple(_freeze_value(item, depth=depth + 1) for item in value)
    if isinstance(value, (set, frozenset)):
        if len(value) > 1000:
            raise ValueError("metadata set is too large")
        items = [_freeze_value(item, depth=depth + 1) for item in value]
        return tuple(sorted(items, key=_stable_sort_key))
    raise TypeError(f"unsupported metadata value: {type(value).__name__}")


def _freeze_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    frozen = _freeze_value(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover
        raise TypeError(f"{field_name} must be a mapping")
    return frozen  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class OrderIntent:
    account_id: str
    symbol: str
    position_side: PositionSide
    action: IntentAction
    quantity: Decimal
    idempotency_key: str
    order_type: OrderType = OrderType.LIMIT
    limit_price: Decimal | None = None
    reduce_only: bool = False
    intent_id: UUID = field(default_factory=uuid4)
    action_group_id: UUID | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        account_id = _text(self.account_id, field_name="account_id", max_length=128)
        key = _text(
            self.idempotency_key,
            field_name="idempotency_key",
            max_length=256,
        )
        symbol = _normalize_symbol(self.symbol)
        try:
            side = (
                self.position_side
                if isinstance(self.position_side, PositionSide)
                else PositionSide(self.position_side)
            )
            action = (
                self.action if isinstance(self.action, IntentAction) else IntentAction(self.action)
            )
            order_type = (
                self.order_type
                if isinstance(self.order_type, OrderType)
                else OrderType(self.order_type)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("position_side, action or order_type is invalid") from exc
        quantity = _decimal(self.quantity, field_name="quantity")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        limit_price = None
        if self.limit_price is not None:
            limit_price = _decimal(self.limit_price, field_name="limit_price")
            if limit_price <= 0:
                raise ValueError("limit_price must be positive")
        if order_type is OrderType.LIMIT and limit_price is None:
            raise ValueError("LIMIT intent requires positive limit_price")
        if order_type is OrderType.MARKET and limit_price is not None:
            raise ValueError("MARKET intent must not include limit_price")
        if not isinstance(self.reduce_only, bool):
            raise TypeError("reduce_only must be a boolean")
        reduce_only = self.reduce_only
        if action.reduces_risk:
            reduce_only = True
        elif reduce_only:
            raise ValueError("risk-increasing intent cannot be reduce_only")
        intent_id = _uuid(self.intent_id, field_name="intent_id")
        group_id = _uuid(
            self.action_group_id,
            field_name="action_group_id",
            optional=True,
        )
        metadata = _freeze_mapping(self.metadata, field_name="metadata")
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "idempotency_key", key)
        object.__setattr__(self, "position_side", side)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "order_type", order_type)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "limit_price", limit_price)
        object.__setattr__(self, "reduce_only", reduce_only)
        object.__setattr__(self, "intent_id", intent_id)
        object.__setattr__(self, "action_group_id", group_id)
        object.__setattr__(self, "metadata", metadata)

    @property
    def reduces_risk(self) -> bool:
        return self.action.reduces_risk


@dataclass(frozen=True, slots=True)
class RiskApproval:
    approved: bool
    approved_quantity: Decimal
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise TypeError("approved must be a boolean")
        quantity = _decimal(self.approved_quantity, field_name="approved_quantity")
        if quantity < 0:
            raise ValueError("approved_quantity must not be negative")
        if self.approved and quantity <= 0:
            raise ValueError("approved approval requires positive approved_quantity")
        if not self.approved and quantity != 0:
            raise ValueError("rejected approval requires approved_quantity == 0")
        object.__setattr__(self, "approved_quantity", quantity)
        object.__setattr__(
            self,
            "reason_codes",
            _reason_codes(self.reason_codes, field_name="reason_codes"),
        )


@dataclass(frozen=True, slots=True)
class ApprovedOrderIntent:
    intent: OrderIntent
    approved_quantity: Decimal
    client_order_id: str
    approved_at: datetime
    risk_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.intent, OrderIntent):
            raise TypeError("intent must be an OrderIntent")
        quantity = _decimal(self.approved_quantity, field_name="approved_quantity")
        if quantity <= 0 or quantity > self.intent.quantity:
            raise ValueError("approved_quantity must be in (0, intent.quantity]")
        validate_client_order_id(self.client_order_id)
        approved_at = _aware(self.approved_at, field_name="approved_at")
        codes = _reason_codes(
            self.risk_reason_codes,
            field_name="risk_reason_codes",
        )
        object.__setattr__(self, "approved_quantity", quantity)
        object.__setattr__(self, "approved_at", approved_at)
        object.__setattr__(self, "risk_reason_codes", codes)


@dataclass(frozen=True, slots=True)
class ExternalOrderSnapshot:
    client_order_id: str
    status: OrderState
    filled_quantity: Decimal = Decimal("0")
    average_price: Decimal | None = None
    exchange_order_id: str | None = None
    exchange_trade_id: str | None = None
    last_fill_fee: Decimal = Decimal("0")
    fee_currency: str | None = "USDT"
    reason: str | None = None
    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        client_id = _text(
            self.client_order_id,
            field_name="client_order_id",
            max_length=256,
        )
        try:
            status = self.status if isinstance(self.status, OrderState) else OrderState(self.status)
        except (TypeError, ValueError) as exc:
            raise ValueError("snapshot status is invalid") from exc
        if status not in _EXTERNAL_STATES:
            raise ValueError("snapshot status is not externally observable")
        filled = _decimal(self.filled_quantity, field_name="filled_quantity")
        if filled < 0:
            raise ValueError("filled_quantity must not be negative")
        average = None
        if self.average_price is not None:
            average = _decimal(self.average_price, field_name="average_price")
            if average <= 0 or filled <= 0:
                raise ValueError("average_price requires a positive fill")
        if status is OrderState.ACKNOWLEDGED and filled != 0:
            raise ValueError("ACKNOWLEDGED requires filled == 0")
        if status is OrderState.PARTIAL and filled <= 0:
            raise ValueError("PARTIAL requires positive filled_quantity")
        if status is OrderState.FILLED and filled <= 0:
            raise ValueError("FILLED requires positive filled_quantity")
        if status is OrderState.REJECTED and filled != 0:
            raise ValueError("REJECTED requires filled == 0")
        observed_at = _aware(self.observed_at, field_name="observed_at")
        exchange_order_id = _text(
            self.exchange_order_id,
            field_name="exchange_order_id",
            max_length=256,
            optional=True,
        )
        exchange_trade_id = _text(
            self.exchange_trade_id,
            field_name="exchange_trade_id",
            max_length=256,
            optional=True,
        )
        last_fill_fee = _decimal(self.last_fill_fee, field_name="last_fill_fee")
        if last_fill_fee < 0:
            raise ValueError("last_fill_fee must not be negative")
        fee_currency = _text(
            self.fee_currency,
            field_name="fee_currency",
            max_length=32,
            optional=True,
        )
        if fee_currency is not None:
            fee_currency = fee_currency.upper()
        if last_fill_fee > 0 and (filled <= 0 or exchange_trade_id is None):
            raise ValueError(
                "positive last_fill_fee requires a filled snapshot and exchange_trade_id"
            )
        reason = _text(
            self.reason,
            field_name="reason",
            max_length=1024,
            optional=True,
        )
        object.__setattr__(self, "client_order_id", client_id)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "filled_quantity", filled)
        object.__setattr__(self, "average_price", average)
        object.__setattr__(self, "exchange_order_id", exchange_order_id)
        object.__setattr__(self, "exchange_trade_id", exchange_trade_id)
        object.__setattr__(self, "last_fill_fee", last_fill_fee)
        object.__setattr__(self, "fee_currency", fee_currency)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "observed_at", observed_at)


@dataclass(frozen=True, slots=True)
class ExecutionOrder:
    intent: OrderIntent
    client_order_id: str
    approved_quantity: Decimal
    lifecycle: OrderLifecycle
    created_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.intent, OrderIntent):
            raise TypeError("intent must be an OrderIntent")
        validate_client_order_id(self.client_order_id)
        approved = _decimal(self.approved_quantity, field_name="approved_quantity")
        if approved < 0 or approved > self.intent.quantity:
            raise ValueError("approved_quantity must be in [0, intent.quantity]")
        if not isinstance(self.lifecycle, OrderLifecycle):
            raise TypeError("lifecycle must be an OrderLifecycle")
        if self.lifecycle.filled_quantity > approved:
            raise ValueError("lifecycle fill exceeds approved_quantity")
        if approved == 0 and self.lifecycle.status is not OrderState.REJECTED:
            raise ValueError("zero approved quantity is valid only for REJECTED orders")
        if self.lifecycle.status is OrderState.FILLED:
            if approved <= 0 or self.lifecycle.filled_quantity != approved:
                raise ValueError("FILLED lifecycle must equal approved_quantity")
        if self.lifecycle.status is OrderState.PARTIAL:
            if not (Decimal("0") < self.lifecycle.filled_quantity < approved):
                raise ValueError("PARTIAL lifecycle must be below approved_quantity")
        created_at = _aware(self.created_at, field_name="created_at")
        object.__setattr__(self, "approved_quantity", approved)
        object.__setattr__(self, "created_at", created_at)

    @property
    def leg_key(self) -> tuple[str, str, PositionSide]:
        return (self.intent.account_id, self.intent.symbol, self.intent.position_side)


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    order: ExecutionOrder
    idempotent_replay: bool = False
    message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.order, ExecutionOrder):
            raise TypeError("order must be an ExecutionOrder")
        if not isinstance(self.idempotent_replay, bool):
            raise TypeError("idempotent_replay must be a boolean")
        message = _text(
            self.message,
            field_name="message",
            max_length=2048,
            optional=True,
        )
        object.__setattr__(self, "message", message)


@dataclass(frozen=True, slots=True)
class ExecutionBatchReport:
    """Result of an operational batch such as refresh, resolve or cancel."""

    operation: str
    attempted: int
    results: tuple[ExecutionResult, ...] = ()
    errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        operation = _text(
            self.operation,
            field_name="operation",
            max_length=64,
        )
        if not isinstance(self.attempted, int) or isinstance(self.attempted, bool):
            raise TypeError("attempted must be an integer")
        if self.attempted < 0:
            raise ValueError("attempted must not be negative")
        if any(not isinstance(item, ExecutionResult) for item in self.results):
            raise TypeError("results must contain ExecutionResult values")
        normalized_errors = tuple(
            _text(item, field_name="batch error", max_length=2048) or "unknown"
            for item in self.errors
        )
        if len(self.results) + len(normalized_errors) > self.attempted:
            raise ValueError("batch outcomes exceed attempted operations")
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "errors", normalized_errors)

    @property
    def succeeded(self) -> int:
        return len(self.results)

    @property
    def failed(self) -> int:
        return len(self.errors)

    @property
    def unresolved(self) -> int:
        return sum(
            result.order.lifecycle.status is OrderState.UNKNOWN
            for result in self.results
        )

    @property
    def complete(self) -> bool:
        return self.succeeded + self.failed == self.attempted


class RiskApprovalPort(Protocol):
    def approve(self, intent: OrderIntent) -> RiskApproval: ...


class ExchangeExecutionPort(Protocol):
    def submit_order(self, approved: ApprovedOrderIntent) -> ExternalOrderSnapshot: ...

    def query_order(self, *, client_order_id: str) -> ExternalOrderSnapshot | None: ...

    def list_open_orders(
        self,
        *,
        account_id: str,
        symbol: str,
    ) -> Sequence[ExternalOrderSnapshot]: ...

    def list_recent_fills(
        self,
        *,
        account_id: str,
        symbol: str,
    ) -> Sequence[ExternalOrderSnapshot]: ...

    def cancel_order(self, *, client_order_id: str) -> ExternalOrderSnapshot: ...


class ExecutionStorePort(Protocol):
    def put(self, order: ExecutionOrder) -> None: ...

    def get_by_client_order_id(self, client_order_id: str) -> ExecutionOrder | None: ...

    def has_unresolved_unknown(
        self,
        leg_key: tuple[str, str, PositionSide],
    ) -> bool: ...

    def list_orders(self) -> Sequence[ExecutionOrder]: ...


class AuditPort(Protocol):
    def emit(self, event: str, payload: Mapping[str, Any]) -> None: ...


class ExecutionMetricsPort(Protocol):
    def intent(self, outcome: str) -> None: ...

    def order(self, from_state: str, to_state: str) -> None: ...

    def fill(self, quantity: Decimal) -> None: ...

    def lock(self, leg: str, locked: bool) -> None: ...


class UnknownOrderResolverPort(Protocol):
    def resolve(self, order: ExecutionOrder) -> ExternalOrderSnapshot | None: ...


class InMemoryExecutionStore:
    def __init__(self) -> None:
        self._orders: dict[str, ExecutionOrder] = {}
        self._lock = RLock()

    def put(self, order: ExecutionOrder) -> None:
        if not isinstance(order, ExecutionOrder):
            raise TypeError("order must be an ExecutionOrder")
        with self._lock:
            current = self._orders.get(order.client_order_id)
            if current is not None:
                if order.lifecycle.version < current.lifecycle.version:
                    raise ValueError("refusing to overwrite a newer execution order")
                if (
                    order.lifecycle.version == current.lifecycle.version
                    and order != current
                ):
                    raise ValueError("same lifecycle version contains conflicting data")
            self._orders[order.client_order_id] = order

    def get_by_client_order_id(self, client_order_id: str) -> ExecutionOrder | None:
        with self._lock:
            return self._orders.get(client_order_id)

    def has_unresolved_unknown(
        self,
        leg_key: tuple[str, str, PositionSide],
    ) -> bool:
        with self._lock:
            return any(
                order.leg_key == leg_key and order.lifecycle.status is OrderState.UNKNOWN
                for order in self._orders.values()
            )

    def list_orders(self) -> tuple[ExecutionOrder, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._orders.values(),
                    key=lambda order: (order.created_at, order.client_order_id),
                )
            )


class AllowAllRiskApproval:
    def approve(self, intent: OrderIntent) -> RiskApproval:
        return RiskApproval(True, intent.quantity, ("ALLOW_ALL_FAKE",))


@dataclass(frozen=True, slots=True)
class AuditRecord:
    event: str
    payload: Mapping[str, Any]
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class InMemoryAuditLog:
    """Bounded recent audit projection. Durable audit belongs in an external AuditPort."""

    def __init__(self, *, max_records: int = 5000) -> None:
        if not isinstance(max_records, int) or isinstance(max_records, bool) or max_records < 1:
            raise ValueError("max_records must be a positive integer")
        self.max_records = max_records
        self.records: deque[AuditRecord] = deque(maxlen=max_records)
        self._lock = RLock()

    def emit(self, event: str, payload: Mapping[str, Any]) -> None:
        event_text = _text(event, field_name="event", max_length=128)
        frozen = _freeze_mapping(payload, field_name="payload")
        with self._lock:
            self.records.append(AuditRecord(event_text or "UNKNOWN", frozen))


class NullAudit:
    def emit(self, event: str, payload: Mapping[str, Any]) -> None:
        return None


class NullMetrics:
    def intent(self, outcome: str) -> None:
        return None

    def order(self, from_state: str, to_state: str) -> None:
        return None

    def fill(self, quantity: Decimal) -> None:
        return None

    def lock(self, leg: str, locked: bool) -> None:
        return None


class ExecutionBlockedError(RuntimeError):
    pass


class DefinitiveExchangeOperationError(RuntimeError):
    """The adapter guarantees that the requested side effect did not occur."""


class DefinitiveSubmissionError(DefinitiveExchangeOperationError):
    """The adapter guarantees that no exchange order was created."""


class DefinitiveCancellationError(DefinitiveExchangeOperationError):
    """The adapter guarantees that cancellation was not submitted."""


class IdempotencyConflictError(ExecutionBlockedError):
    pass


class ExecutionService:
    _LOCK_STRIPES = 257

    def __init__(
        self,
        *,
        risk: RiskApprovalPort,
        exchange: ExchangeExecutionPort,
        store: ExecutionStorePort,
        idempotency: IdempotencyPort[ExecutionResult],
        unknown_resolver: UnknownOrderResolverPort,
        kill_switch: KillSwitch,
        audit: AuditPort | None = None,
        metrics: ExecutionMetricsPort | None = None,
    ) -> None:
        self._risk = risk
        self._exchange = exchange
        self._store = store
        self._idempotency = idempotency
        self._resolver = unknown_resolver
        self._kill_switch = kill_switch
        self._audit = audit or NullAudit()
        self._metrics = metrics or NullMetrics()
        self._leg_locks = tuple(RLock() for _ in range(self._LOCK_STRIPES))

    def submit(self, intent: OrderIntent) -> ExecutionResult:
        if not isinstance(intent, OrderIntent):
            raise TypeError("intent must be an OrderIntent")
        with self.leg_guard(intent):
            return self._submit_locked(intent, intent_leg_key(intent))

    @contextmanager
    def leg_guard(
        self,
        value: OrderIntent | ExecutionOrder | tuple[str, str, PositionSide],
    ) -> Iterator[None]:
        if isinstance(value, OrderIntent):
            leg_key = intent_leg_key(value)
        elif isinstance(value, ExecutionOrder):
            leg_key = value.leg_key
        elif (
            isinstance(value, tuple)
            and len(value) == 3
            and isinstance(value[2], PositionSide)
        ):
            leg_key = value
        else:
            raise TypeError("leg guard requires an intent, order or leg key")
        lock = self._leg_lock(leg_key)
        with lock:
            yield

    def _submit_locked(
        self,
        intent: OrderIntent,
        leg_key: tuple[str, str, PositionSide],
    ) -> ExecutionResult:
        client_id = build_client_order_id(
            account_id=intent.account_id,
            symbol=intent.symbol,
            position_side=intent.position_side.value,
            idempotency_key=intent.idempotency_key,
        )
        existing_for_intent = self._store.get_by_client_order_id(client_id)
        reservation = self._idempotency.reserve(intent.idempotency_key)
        if reservation.state is ReservationState.COMPLETED:
            if reservation.value is None:
                raise RuntimeError("completed idempotency reservation has no value")
            if not self._same_intent(reservation.value.order.intent, intent):
                raise IdempotencyConflictError(
                    "idempotency key was already used for a different intent"
                )
            latest = self._store.get_by_client_order_id(
                reservation.value.order.client_order_id
            )
            return replace(
                reservation.value,
                order=latest or reservation.value.order,
                idempotent_replay=True,
            )
        if reservation.state is ReservationState.IN_FLIGHT:
            raise ExecutionBlockedError("idempotency key is already in flight")

        side_effect_started = False
        self._safe_metric("intent", "received")
        try:
            existing = existing_for_intent
            if existing is not None:
                result = self._recover_existing_execution(existing, intent)
                self._idempotency.complete(intent.idempotency_key, result)
                return result

            self._kill_switch.assert_allowed(reduces_risk=intent.reduces_risk)
            if (
                not intent.reduces_risk
                and self._store.has_unresolved_unknown(leg_key)
            ):
                self._safe_metric("lock", self._leg_label(leg_key), True)
                raise ExecutionBlockedError(
                    "unresolved UNKNOWN order blocks new risk on this leg"
                )

            approval = self._risk.approve(intent)
            if not isinstance(approval, RiskApproval):
                raise TypeError("risk adapter must return RiskApproval")
            if approval.approved_quantity > intent.quantity:
                raise ValueError("risk approval cannot exceed requested quantity")
            if not approval.approved:
                result = self._rejected_before_submit(intent, approval)
                self._idempotency.complete(intent.idempotency_key, result)
                return result

            now = datetime.now(UTC)
            order = ExecutionOrder(
                intent=intent,
                client_order_id=client_id,
                approved_quantity=approval.approved_quantity,
                lifecycle=OrderLifecycle(updated_at=now),
                created_at=now,
            )
            self._store.put(order)
            self._safe_audit("ORDER_INTENT_PREPARED", self._audit_payload(order))
            order = self._transition(order, OrderState.SUBMITTING)
            approved = ApprovedOrderIntent(
                intent=intent,
                approved_quantity=order.approved_quantity,
                client_order_id=client_id,
                approved_at=now,
                risk_reason_codes=approval.reason_codes,
            )

            side_effect_started = True
            try:
                snapshot = self._exchange.submit_order(approved)
            except DefinitiveSubmissionError as exc:
                order = self._transition(
                    order,
                    OrderState.REJECTED,
                    reason=f"submit_rejected:{type(exc).__name__}",
                )
                self._safe_audit("ORDER_SUBMIT_REJECTED", self._audit_payload(order))
                result = ExecutionResult(
                    order,
                    message="submission was definitively rejected",
                )
                self._idempotency.complete(intent.idempotency_key, result)
                return result
            except Exception as exc:
                result = self._submission_unknown(order, exc)
                self._idempotency.complete(intent.idempotency_key, result)
                return result

            if not isinstance(snapshot, ExternalOrderSnapshot):
                result = self._invalid_submission_ack(
                    order,
                    TypeError("invalid snapshot"),
                )
            else:
                try:
                    order = self._apply_snapshot_locked(order, snapshot)
                except Exception as exc:
                    result = self._invalid_submission_ack(order, exc)
                else:
                    if order.lifecycle.status is OrderState.UNKNOWN:
                        recovered = self._recover_safely(order)
                        if recovered is not None:
                            try:
                                order = self._apply_snapshot_locked(order, recovered)
                            except Exception as recovery_error:
                                self._audit_invalid_recovery(
                                    order,
                                    recovery_error,
                                )
                        result = ExecutionResult(
                            order,
                            message="submit returned UNKNOWN; recovery queried",
                        )
                    else:
                        result = ExecutionResult(order)
            self._idempotency.complete(intent.idempotency_key, result)
            return result
        except Exception:
            if not side_effect_started:
                self._idempotency.release(intent.idempotency_key)
            raise

    def _recover_existing_execution(
        self,
        existing: ExecutionOrder,
        intent: OrderIntent,
    ) -> ExecutionResult:
        if not self._same_intent(existing.intent, intent):
            raise IdempotencyConflictError(
                "deterministic client order id belongs to a different intent"
            )
        order = existing
        if order.lifecycle.status in {OrderState.PREPARED, OrderState.SUBMITTING}:
            order = self._transition(
                order,
                OrderState.UNKNOWN,
                reason="recovered_existing_execution_record",
            )
        if order.lifecycle.status is OrderState.UNKNOWN:
            recovered = self._recover_safely(order)
            if recovered is not None:
                try:
                    order = self._apply_snapshot_locked(order, recovered)
                except Exception as exc:
                    self._audit_invalid_recovery(order, exc)
        self._safe_metric("intent", "existing_record_replay")
        self._safe_audit(
            "ORDER_EXISTING_RECORD_REPLAYED",
            self._audit_payload(order),
        )
        return ExecutionResult(
            order,
            idempotent_replay=True,
            message="existing execution record recovered without resubmission",
        )

    def _submission_unknown(
        self,
        order: ExecutionOrder,
        error: Exception,
    ) -> ExecutionResult:
        reason = (
            "submit_timeout"
            if isinstance(error, TimeoutError)
            else f"submit_error:{type(error).__name__}"
        )
        order = self._transition(order, OrderState.UNKNOWN, reason=reason)
        self._safe_metric("lock", self._leg_label(order.leg_key), True)
        self._safe_audit("ORDER_SUBMIT_UNKNOWN", self._audit_payload(order))
        recovered = self._recover_safely(order)
        if recovered is not None:
            try:
                order = self._apply_snapshot_locked(order, recovered)
            except Exception as recovery_error:
                self._audit_invalid_recovery(order, recovery_error)
        message = (
            "submit timed out; queried before any retry"
            if isinstance(error, TimeoutError)
            else "submit outcome uncertain; queried before any retry"
        )
        return ExecutionResult(order, message=message)

    def _invalid_submission_ack(
        self,
        order: ExecutionOrder,
        error: Exception,
    ) -> ExecutionResult:
        latest = self.get_order(order.client_order_id)
        if latest.lifecycle.terminal:
            return ExecutionResult(
                latest,
                message="invalid acknowledgement ignored after terminal fact",
            )
        if latest.lifecycle.status is not OrderState.UNKNOWN:
            order = self._transition(
                latest,
                OrderState.UNKNOWN,
                reason=f"invalid_ack:{type(error).__name__}",
            )
        else:
            order = latest
        recovered = self._recover_safely(order)
        if recovered is not None:
            try:
                order = self._apply_snapshot_locked(order, recovered)
            except Exception as recovery_error:
                self._audit_invalid_recovery(order, recovery_error)
        return ExecutionResult(order, message="invalid acknowledgement; recovery queried")

    def resolve_unknown(self, client_order_id: str) -> ExecutionResult:
        order = self.get_order(client_order_id)
        with self.leg_guard(order):
            order = self.get_order(client_order_id)
            if order.lifecycle.status is not OrderState.UNKNOWN:
                return ExecutionResult(order, message="order is not UNKNOWN")
            snapshot = self._recover_safely(order)
            if snapshot is None:
                return ExecutionResult(order, message="UNKNOWN remains unresolved")
            try:
                order = self._apply_snapshot_locked(order, snapshot)
            except Exception as exc:
                self._audit_invalid_recovery(order, exc)
                return ExecutionResult(
                    order,
                    message="UNKNOWN recovery fact was invalid",
                )
            result = ExecutionResult(order, message="UNKNOWN resolved")
            self._idempotency.complete(order.intent.idempotency_key, result)
            return result

    def cancel(self, client_order_id: str) -> ExecutionResult:
        order = self.get_order(client_order_id)
        with self.leg_guard(order):
            order = self.get_order(client_order_id)
            if order.lifecycle.terminal:
                return ExecutionResult(order, message="already terminal")
            try:
                snapshot = self._exchange.cancel_order(
                    client_order_id=client_order_id
                )
            except DefinitiveCancellationError:
                raise
            except Exception as exc:
                reason = (
                    "cancel_timeout"
                    if isinstance(exc, TimeoutError)
                    else f"cancel_error:{type(exc).__name__}"
                )
                order = self._transition(order, OrderState.UNKNOWN, reason=reason)
                snapshot = self._recover_safely(order)
                if snapshot is None:
                    result = ExecutionResult(
                        order,
                        message="cancel UNKNOWN remains unresolved",
                    )
                    self._idempotency.complete(order.intent.idempotency_key, result)
                    return result
            if not isinstance(snapshot, ExternalOrderSnapshot):
                result = self._invalid_cancel_ack(
                    order,
                    TypeError("cancel adapter returned an invalid snapshot"),
                )
                self._idempotency.complete(order.intent.idempotency_key, result)
                return result
            try:
                order = self._apply_snapshot_locked(order, snapshot)
            except Exception as exc:
                result = self._invalid_cancel_ack(order, exc)
                self._idempotency.complete(order.intent.idempotency_key, result)
                return result
            if order.lifecycle.status is OrderState.UNKNOWN:
                recovered = self._recover_safely(order)
                if recovered is None:
                    result = ExecutionResult(
                        order,
                        message="cancel returned UNKNOWN; recovery unresolved",
                    )
                    self._idempotency.complete(order.intent.idempotency_key, result)
                    return result
                try:
                    order = self._apply_snapshot_locked(order, recovered)
                except Exception as exc:
                    self._audit_invalid_recovery(order, exc)
                    result = ExecutionResult(
                        order,
                        message="cancel recovery fact was invalid",
                    )
                    self._idempotency.complete(order.intent.idempotency_key, result)
                    return result
            result = ExecutionResult(order)
            self._idempotency.complete(order.intent.idempotency_key, result)
            return result

    def _invalid_cancel_ack(
        self,
        order: ExecutionOrder,
        error: Exception,
    ) -> ExecutionResult:
        latest = self.get_order(order.client_order_id)
        if latest.lifecycle.terminal:
            return ExecutionResult(
                latest,
                message="invalid cancel acknowledgement ignored after terminal fact",
            )
        if latest.lifecycle.status is not OrderState.UNKNOWN:
            latest = self._transition(
                latest,
                OrderState.UNKNOWN,
                reason=f"invalid_cancel_ack:{type(error).__name__}",
            )
        recovered = self._recover_safely(latest)
        if recovered is None:
            return ExecutionResult(
                latest,
                message="invalid cancel acknowledgement; recovery unresolved",
            )
        try:
            latest = self._apply_snapshot_locked(latest, recovered)
        except Exception as recovery_error:
            self._audit_invalid_recovery(latest, recovery_error)
            return ExecutionResult(
                latest,
                message="invalid cancel acknowledgement and recovery fact",
            )
        return ExecutionResult(
            latest,
            message="invalid cancel acknowledgement; recovery queried",
        )

    def get_order(self, client_order_id: str) -> ExecutionOrder:
        if not isinstance(client_order_id, str):
            raise TypeError("client_order_id must be a string")
        order = self._store.get_by_client_order_id(client_order_id)
        if order is None:
            raise KeyError(client_order_id)
        return order

    def list_orders(
        self,
        *,
        account_id: str | None = None,
        symbol: str | None = None,
        position_side: PositionSide | str | None = None,
        statuses: Sequence[OrderState | str] | None = None,
        action_group_id: UUID | str | None = None,
        include_terminal: bool = True,
        limit: int | None = None,
    ) -> tuple[ExecutionOrder, ...]:
        """Return a deterministic, filtered execution view for control-plane use."""
        if not isinstance(include_terminal, bool):
            raise TypeError("include_terminal must be a boolean")
        if limit is not None:
            if not isinstance(limit, int) or isinstance(limit, bool):
                raise TypeError("limit must be an integer")
            if limit <= 0:
                raise ValueError("limit must be positive")
        account = (
            None
            if account_id is None
            else _text(account_id, field_name="account_id", max_length=128)
        )
        normalized_symbol = None if symbol is None else _normalize_symbol(symbol)
        side = None
        if position_side is not None:
            try:
                side = (
                    position_side
                    if isinstance(position_side, PositionSide)
                    else PositionSide(position_side)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("position_side is invalid") from exc
        normalized_statuses: frozenset[OrderState] | None = None
        if statuses is not None:
            if isinstance(statuses, (str, bytes)):
                raise TypeError("statuses must be a sequence")
            try:
                normalized_statuses = frozenset(
                    item if isinstance(item, OrderState) else OrderState(item)
                    for item in statuses
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("statuses contains an invalid state") from exc
        group_id = None
        if action_group_id is not None:
            group_id = _uuid(action_group_id, field_name="action_group_id")

        selected: list[ExecutionOrder] = []
        for order in reversed(tuple(self._store.list_orders())):
            if account is not None and order.intent.account_id != account:
                continue
            if normalized_symbol is not None and order.intent.symbol != normalized_symbol:
                continue
            if side is not None and order.intent.position_side is not side:
                continue
            if normalized_statuses is not None and order.lifecycle.status not in normalized_statuses:
                continue
            if group_id is not None and order.intent.action_group_id != group_id:
                continue
            if not include_terminal and order.lifecycle.terminal:
                continue
            selected.append(order)
            if limit is not None and len(selected) >= limit:
                break
        return tuple(selected)

    def action_group_orders(
        self,
        action_group_id: UUID | str,
    ) -> tuple[ExecutionOrder, ...]:
        return self.list_orders(action_group_id=action_group_id)

    def apply_exchange_event(
        self,
        snapshot: ExternalOrderSnapshot,
    ) -> ExecutionResult:
        """Apply one normalized exchange/user-stream fact to a known order."""
        if not isinstance(snapshot, ExternalOrderSnapshot):
            raise TypeError("snapshot must be an ExternalOrderSnapshot")
        order = self.get_order(snapshot.client_order_id)
        with self.leg_guard(order):
            latest = self.get_order(snapshot.client_order_id)
            updated = self._apply_snapshot_locked(latest, snapshot)
            result = ExecutionResult(updated, message="exchange event applied")
            self._idempotency.complete(updated.intent.idempotency_key, result)
            return result

    def refresh_order(self, client_order_id: str) -> ExecutionResult:
        """Query current exchange state and merge it without submitting anything."""
        order = self.get_order(client_order_id)
        with self.leg_guard(order):
            order = self.get_order(client_order_id)
            if order.lifecycle.terminal:
                return ExecutionResult(order, message="already terminal")
            snapshot: ExternalOrderSnapshot | None = None
            direct_error: Exception | None = None
            try:
                snapshot = self._exchange.query_order(
                    client_order_id=client_order_id
                )
                if snapshot is not None and not isinstance(
                    snapshot,
                    ExternalOrderSnapshot,
                ):
                    raise TypeError("query_order returned an invalid snapshot")
            except Exception as exc:
                direct_error = exc
                snapshot = None
            if snapshot is None or snapshot.status is OrderState.UNKNOWN:
                recovered = self._recover_safely(order)
                if recovered is not None:
                    snapshot = recovered
            if snapshot is None or snapshot.status is OrderState.UNKNOWN:
                message = "refresh found no conclusive exchange fact"
                if direct_error is not None:
                    message = (
                        "direct refresh failed; fallback recovery remained unresolved"
                    )
                return ExecutionResult(order, message=message)
            try:
                order = self._apply_snapshot_locked(order, snapshot)
            except Exception as exc:
                self._audit_invalid_recovery(order, exc)
                recovered = self._recover_safely(order)
                if recovered is None:
                    return ExecutionResult(
                        order,
                        message="refresh fact was invalid and fallback unresolved",
                    )
                try:
                    order = self._apply_snapshot_locked(order, recovered)
                except Exception as recovery_error:
                    self._audit_invalid_recovery(order, recovery_error)
                    return ExecutionResult(
                        order,
                        message="refresh and fallback facts were invalid",
                    )
            result = ExecutionResult(order, message="order refreshed")
            self._idempotency.complete(order.intent.idempotency_key, result)
            return result

    def refresh_orders(
        self,
        *,
        account_id: str | None = None,
        symbol: str | None = None,
        include_unknown: bool = True,
        limit: int | None = None,
    ) -> ExecutionBatchReport:
        """Refresh all selected non-terminal orders from exchange facts."""
        if not isinstance(include_unknown, bool):
            raise TypeError("include_unknown must be a boolean")
        selected = self.list_orders(
            account_id=account_id,
            symbol=symbol,
            include_terminal=False,
            limit=limit,
        )
        if not include_unknown:
            selected = tuple(
                order
                for order in selected
                if order.lifecycle.status is not OrderState.UNKNOWN
            )
        results: list[ExecutionResult] = []
        errors: list[str] = []
        for order in selected:
            try:
                results.append(self.refresh_order(order.client_order_id))
            except Exception as exc:
                errors.append(
                    f"{order.client_order_id}:{type(exc).__name__}:{exc}"
                )
        return ExecutionBatchReport(
            operation="refresh",
            attempted=len(selected),
            results=tuple(results),
            errors=tuple(errors),
        )

    def resolve_unknowns(
        self,
        *,
        account_id: str | None = None,
        symbol: str | None = None,
        limit: int | None = None,
    ) -> ExecutionBatchReport:
        selected = self.list_orders(
            account_id=account_id,
            symbol=symbol,
            statuses=(OrderState.UNKNOWN,),
            limit=limit,
        )
        results: list[ExecutionResult] = []
        errors: list[str] = []
        for order in selected:
            try:
                results.append(self.resolve_unknown(order.client_order_id))
            except Exception as exc:
                errors.append(
                    f"{order.client_order_id}:{type(exc).__name__}:{exc}"
                )
        return ExecutionBatchReport(
            operation="resolve_unknown",
            attempted=len(selected),
            results=tuple(results),
            errors=tuple(errors),
        )

    def cancel_orders(
        self,
        *,
        account_id: str | None = None,
        symbol: str | None = None,
        position_side: PositionSide | str | None = None,
        limit: int | None = None,
    ) -> ExecutionBatchReport:
        selected = self.list_orders(
            account_id=account_id,
            symbol=symbol,
            position_side=position_side,
            include_terminal=False,
            limit=limit,
        )
        results: list[ExecutionResult] = []
        errors: list[str] = []
        for order in selected:
            try:
                results.append(self.cancel(order.client_order_id))
            except Exception as exc:
                errors.append(
                    f"{order.client_order_id}:{type(exc).__name__}:{exc}"
                )
        return ExecutionBatchReport(
            operation="cancel",
            attempted=len(selected),
            results=tuple(results),
            errors=tuple(errors),
        )

    def apply_snapshot(
        self,
        order: ExecutionOrder,
        snapshot: ExternalOrderSnapshot,
    ) -> ExecutionOrder:
        if not isinstance(order, ExecutionOrder):
            raise TypeError("order must be an ExecutionOrder")
        if not isinstance(snapshot, ExternalOrderSnapshot):
            raise TypeError("snapshot must be an ExternalOrderSnapshot")
        with self.leg_guard(order):
            return self._apply_snapshot_locked(order, snapshot)

    def _apply_snapshot_locked(
        self,
        order: ExecutionOrder,
        snapshot: ExternalOrderSnapshot,
    ) -> ExecutionOrder:
        if snapshot.client_order_id != order.client_order_id:
            raise ValueError("snapshot client_order_id mismatch")
        latest = self._store.get_by_client_order_id(order.client_order_id) or order
        current = latest.lifecycle
        status = snapshot.status
        filled = snapshot.filled_quantity
        if filled > latest.approved_quantity:
            raise ValueError("snapshot fill exceeds approved quantity")
        if filled == latest.approved_quantity and filled > 0:
            if status in {OrderState.PARTIAL, OrderState.CANCELED}:
                status = OrderState.FILLED
        elif (
            current.status is OrderState.CANCELED
            and status is OrderState.PARTIAL
            and filled > current.filled_quantity
        ):
            status = OrderState.CANCELED

        same_fact = (
            status is current.status
            and filled == current.filled_quantity
            and (
                snapshot.average_price is None
                or snapshot.average_price == current.average_price
            )
            and (
                snapshot.exchange_order_id is None
                or snapshot.exchange_order_id == current.exchange_order_id
            )
        )
        if same_fact:
            return latest

        stale = snapshot.observed_at < current.updated_at
        if (
            stale
            and current.status is not OrderState.UNKNOWN
            and status is not OrderState.FILLED
            and filled <= current.filled_quantity
        ):
            self._safe_audit(
                "ORDER_SNAPSHOT_IGNORED_STALE",
                {
                    **self._audit_payload(latest),
                    "snapshot_status": status.value,
                    "snapshot_observed_at": snapshot.observed_at.isoformat(),
                },
            )
            return latest

        occurred_at = current.updated_at if stale else snapshot.observed_at
        previous_state = current.status
        try:
            if current.terminal:
                lifecycle = current.reconcile_terminal_fact(
                    status,
                    ordered_quantity=latest.approved_quantity,
                    filled_quantity=filled,
                    average_price=snapshot.average_price,
                    exchange_order_id=snapshot.exchange_order_id,
                    reason=snapshot.reason,
                    occurred_at=occurred_at,
                )
            else:
                lifecycle = current.transition(
                    status,
                    ordered_quantity=latest.approved_quantity,
                    filled_quantity=filled,
                    average_price=snapshot.average_price,
                    exchange_order_id=snapshot.exchange_order_id,
                    reason=snapshot.reason,
                    occurred_at=occurred_at,
                )
        except InvalidOrderTransition:
            if stale and filled <= current.filled_quantity:
                return latest
            raise
        updated = replace(latest, lifecycle=lifecycle)
        self._store.put(updated)
        self._safe_metric("order", previous_state.value, lifecycle.status.value)
        if lifecycle.filled_quantity > current.filled_quantity:
            self._safe_metric(
                "fill",
                lifecycle.filled_quantity - current.filled_quantity,
            )
        self._safe_metric(
            "lock",
            self._leg_label(updated.leg_key),
            self._store.has_unresolved_unknown(updated.leg_key),
        )
        self._safe_audit("ORDER_STATE_CHANGED", self._audit_payload(updated))
        return updated

    def _transition(
        self,
        order: ExecutionOrder,
        target: OrderState,
        *,
        reason: str | None = None,
    ) -> ExecutionOrder:
        latest = self._store.get_by_client_order_id(order.client_order_id) or order
        current = latest.lifecycle.status
        updated = replace(
            latest,
            lifecycle=latest.lifecycle.transition(
                target,
                ordered_quantity=latest.approved_quantity,
                reason=reason,
            ),
        )
        self._store.put(updated)
        self._safe_metric("order", current.value, target.value)
        return updated

    def _rejected_before_submit(
        self,
        intent: OrderIntent,
        approval: RiskApproval,
    ) -> ExecutionResult:
        now = datetime.now(UTC)
        client_id = build_client_order_id(
            account_id=intent.account_id,
            symbol=intent.symbol,
            position_side=intent.position_side.value,
            idempotency_key=intent.idempotency_key,
        )
        lifecycle = OrderLifecycle(updated_at=now).transition(
            OrderState.REJECTED,
            ordered_quantity=intent.quantity,
            reason=",".join(approval.reason_codes) or "risk_rejected",
            occurred_at=now,
        )
        order = ExecutionOrder(intent, client_id, Decimal("0"), lifecycle, now)
        self._store.put(order)
        self._safe_metric("intent", "risk_rejected")
        self._safe_audit("ORDER_INTENT_REJECTED", self._audit_payload(order))
        return ExecutionResult(order, message="risk rejected")

    def _recover_safely(
        self,
        order: ExecutionOrder,
    ) -> ExternalOrderSnapshot | None:
        try:
            snapshot = self._resolver.resolve(order)
            if snapshot is not None and not isinstance(
                snapshot,
                ExternalOrderSnapshot,
            ):
                raise TypeError("resolver must return ExternalOrderSnapshot or None")
            if snapshot is not None and snapshot.status is OrderState.UNKNOWN:
                self._safe_audit(
                    "ORDER_RECOVERY_REMAINED_UNKNOWN",
                    self._audit_payload(order),
                )
                return None
            return snapshot
        except Exception as exc:
            self._safe_audit(
                "ORDER_RECOVERY_FAILED",
                {**self._audit_payload(order), "error": type(exc).__name__},
            )
            return None

    def _audit_invalid_recovery(
        self,
        order: ExecutionOrder,
        error: Exception,
    ) -> None:
        self._safe_audit(
            "ORDER_RECOVERY_SNAPSHOT_INVALID",
            {**self._audit_payload(order), "error": type(error).__name__},
        )

    def _leg_lock(self, leg_key: tuple[str, str, PositionSide]) -> RLock:
        material = "\x1f".join(
            (leg_key[0], leg_key[1], leg_key[2].value)
        ).encode("utf-8")
        index = int.from_bytes(hashlib.blake2s(material, digest_size=4).digest(), "big")
        return self._leg_locks[index % self._LOCK_STRIPES]

    def _safe_audit(self, event: str, payload: Mapping[str, Any]) -> None:
        try:
            self._audit.emit(event, payload)
        except Exception:
            return None

    def _safe_metric(self, method: str, *args: object) -> None:
        try:
            getattr(self._metrics, method)(*args)
        except Exception:
            return None

    @staticmethod
    def _same_intent(left: OrderIntent, right: OrderIntent) -> bool:
        return (
            left.account_id,
            left.symbol,
            left.position_side,
            left.action,
            left.quantity,
            left.order_type,
            left.limit_price,
            left.reduce_only,
            left.action_group_id,
            left.metadata,
        ) == (
            right.account_id,
            right.symbol,
            right.position_side,
            right.action,
            right.quantity,
            right.order_type,
            right.limit_price,
            right.reduce_only,
            right.action_group_id,
            right.metadata,
        )

    @staticmethod
    def _leg_label(leg_key: tuple[str, str, PositionSide]) -> str:
        return ":".join((leg_key[0], leg_key[1], leg_key[2].value))

    @staticmethod
    def _audit_payload(order: ExecutionOrder) -> dict[str, Any]:
        return {
            "intent_id": str(order.intent.intent_id),
            "client_order_id": order.client_order_id,
            "action_group_id": (
                str(order.intent.action_group_id)
                if order.intent.action_group_id
                else None
            ),
            "account_id": order.intent.account_id,
            "symbol": order.intent.symbol,
            "position_side": order.intent.position_side.value,
            "action": order.intent.action.value,
            "status": order.lifecycle.status.value,
            "quantity": str(order.approved_quantity),
            "filled_quantity": str(order.lifecycle.filled_quantity),
            "reason": order.lifecycle.reason,
        }


def intent_leg_key(intent: OrderIntent) -> tuple[str, str, PositionSide]:
    return (intent.account_id, intent.symbol, intent.position_side)
