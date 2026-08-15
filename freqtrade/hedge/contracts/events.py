"""Versioned immutable facts shared by execution, persistence and telemetry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID, uuid4

from .types import IntentAction, OrderSide, PositionKey, PositionRecord as LegacyPositionRecord


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError(f"{name} must be exact")
    try:
        result = Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is invalid") from exc
    if not result.is_finite() or result < 0 or (positive and result <= 0):
        raise ValueError(f"{name} is outside the valid range")
    return result


def _legacy_decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be exact")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is invalid") from exc
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def _aware(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value


def _milliseconds(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be nonnegative integer milliseconds")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be nonnegative integer milliseconds") from exc
    if result < 0:
        raise ValueError(f"{name} must be nonnegative integer milliseconds")
    return result


@dataclass(frozen=True, slots=True)
class AccountEvent:
    event_id: str
    correlation_id: str
    account_id: str
    event_type: str
    exchange_time_ms: int
    observed_time_ms: int
    hedge_event_version: int = 1


@dataclass(frozen=True, slots=True, init=False)
class FillEvent(AccountEvent):
    """A fill fact accepting both frozen v1 and execution-v2 constructor shapes.

    The original P2-H2 contract used AccountEvent fields plus ``order_id``.
    Direction 5 later introduced side-aware execution fields.  Replacing one
    public shape with the other breaks independent workstreams, so the merged
    contract stores the capability union and validates according to the shape
    supplied by the caller.
    """

    position_key: PositionKey | None = None
    trade_id: str = ""
    client_order_id: str = ""
    order_id: str = ""
    action: IntentAction | None = None
    order_side: OrderSide | None = None
    quantity: Decimal = Decimal("0")
    price: Decimal = Decimal("0")
    fee: Decimal = Decimal("0")
    fee_currency: str = "USDT"
    exchange_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    observed_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    payload_version: int = 1

    _NEW_POSITIONAL = (
        "position_key",
        "trade_id",
        "client_order_id",
        "action",
        "order_side",
        "quantity",
        "price",
        "fee",
        "fee_currency",
        "exchange_time",
        "observed_time",
        "event_id",
        "payload_version",
    )
    _LEGACY_POSITIONAL = (
        "event_id",
        "correlation_id",
        "account_id",
        "event_type",
        "exchange_time_ms",
        "observed_time_ms",
        "hedge_event_version",
        "order_id",
        "quantity",
        "price",
    )

    def __init__(self, *args: object, **kwargs: object) -> None:
        legacy_keys = {
            "correlation_id",
            "account_id",
            "event_type",
            "exchange_time_ms",
            "observed_time_ms",
            "hedge_event_version",
            "order_id",
        }
        is_new = "position_key" in kwargs or (args and isinstance(args[0], PositionKey))
        if not is_new and not (legacy_keys & set(kwargs)):
            # Keep a useful error for incomplete new-style calls.
            is_new = any(key in kwargs for key in ("trade_id", "client_order_id", "action", "order_side"))
        if is_new:
            self._init_execution(args, kwargs)
        else:
            self._init_legacy(args, kwargs)

    @staticmethod
    def _merge_positional(
        names: tuple[str, ...], args: tuple[object, ...], kwargs: dict[str, object]
    ) -> dict[str, object]:
        if len(args) > len(names):
            raise TypeError(f"expected at most {len(names)} positional arguments")
        result = dict(kwargs)
        for name, value in zip(names, args, strict=False):
            if name in result:
                raise TypeError(f"multiple values for argument {name!r}")
            result[name] = value
        return result

    def _init_execution(self, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        values = self._merge_positional(self._NEW_POSITIONAL, args, kwargs)
        allowed = set(self._NEW_POSITIONAL) | {"correlation_id", "event_type", "hedge_event_version"}
        extra = sorted(set(values) - allowed)
        if extra:
            raise TypeError(f"unexpected FillEvent arguments: {', '.join(extra)}")

        required = ("position_key", "trade_id", "client_order_id", "action", "order_side", "quantity", "price")
        missing = [name for name in required if name not in values]
        if missing:
            raise TypeError(f"missing FillEvent arguments: {', '.join(missing)}")

        position_key = values["position_key"]
        if not isinstance(position_key, PositionKey):
            raise TypeError("position_key must be a PositionKey")
        trade_id = str(values["trade_id"]).strip()
        client_order_id = str(values["client_order_id"]).strip()
        if not trade_id or not client_order_id:
            raise ValueError("trade_id and client_order_id are required")
        try:
            action_value = values["action"]
            action = action_value if isinstance(action_value, IntentAction) else IntentAction(action_value)
            side_value = values["order_side"]
            order_side = side_value if isinstance(side_value, OrderSide) else OrderSide(side_value)
        except (TypeError, ValueError) as exc:
            raise ValueError("fill action or order_side is invalid") from exc

        quantity = _decimal(values["quantity"], name="quantity", positive=True)
        price = _decimal(values["price"], name="price", positive=True)
        fee = _decimal(values.get("fee", Decimal("0")), name="fee")
        fee_currency = str(values.get("fee_currency", "USDT")).strip().upper()
        if not fee_currency or len(fee_currency) > 16:
            raise ValueError("fee_currency is invalid")
        exchange_time = _aware(values.get("exchange_time", datetime.now(UTC)), name="exchange_time")
        observed_time = _aware(values.get("observed_time", datetime.now(UTC)), name="observed_time")
        payload_version = values.get("payload_version", 1)
        if isinstance(payload_version, bool) or not isinstance(payload_version, int) or payload_version <= 0:
            raise ValueError("payload_version must be positive")
        event_id = values.get("event_id", uuid4())
        if not isinstance(event_id, (UUID, str)):
            raise TypeError("event_id must be UUID or string")
        correlation_id = str(values.get("correlation_id", event_id)).strip()
        event_type = str(values.get("event_type", "FILL")).strip() or "FILL"
        hedge_event_version = values.get("hedge_event_version", 1)
        if isinstance(hedge_event_version, bool) or not isinstance(hedge_event_version, int) or hedge_event_version <= 0:
            raise ValueError("hedge_event_version must be positive")

        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "correlation_id", correlation_id)
        object.__setattr__(self, "account_id", position_key.account_id)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "exchange_time_ms", int(exchange_time.timestamp() * 1000))
        object.__setattr__(self, "observed_time_ms", int(observed_time.timestamp() * 1000))
        object.__setattr__(self, "hedge_event_version", hedge_event_version)
        object.__setattr__(self, "position_key", position_key)
        object.__setattr__(self, "trade_id", trade_id)
        object.__setattr__(self, "client_order_id", client_order_id)
        object.__setattr__(self, "order_id", client_order_id)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "order_side", order_side)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "fee", fee)
        object.__setattr__(self, "fee_currency", fee_currency)
        object.__setattr__(self, "exchange_time", exchange_time)
        object.__setattr__(self, "observed_time", observed_time)
        object.__setattr__(self, "payload_version", payload_version)

    def _init_legacy(self, args: tuple[object, ...], kwargs: dict[str, object]) -> None:
        values = self._merge_positional(self._LEGACY_POSITIONAL, args, kwargs)
        allowed = set(self._LEGACY_POSITIONAL)
        extra = sorted(set(values) - allowed)
        if extra:
            raise TypeError(f"unexpected legacy FillEvent arguments: {', '.join(extra)}")
        required = ("event_id", "correlation_id", "account_id", "event_type", "exchange_time_ms", "observed_time_ms")
        missing = [name for name in required if name not in values]
        if missing:
            raise TypeError(f"missing legacy FillEvent arguments: {', '.join(missing)}")

        event_id = str(values["event_id"]).strip()
        correlation_id = str(values["correlation_id"]).strip()
        account_id = str(values["account_id"]).strip()
        event_type = str(values["event_type"]).strip()
        if not event_id or not correlation_id or not account_id or not event_type:
            raise ValueError("legacy event text fields must not be empty")
        exchange_ms = _milliseconds(values["exchange_time_ms"], name="exchange_time_ms")
        observed_ms = _milliseconds(values["observed_time_ms"], name="observed_time_ms")
        hedge_event_version = values.get("hedge_event_version", 1)
        if isinstance(hedge_event_version, bool) or not isinstance(hedge_event_version, int) or hedge_event_version <= 0:
            raise ValueError("hedge_event_version must be positive")
        order_id = str(values.get("order_id", "")).strip()
        quantity = _legacy_decimal(values.get("quantity", Decimal("0")), name="quantity")
        price = _legacy_decimal(values.get("price", Decimal("0")), name="price")
        exchange_time = datetime.fromtimestamp(exchange_ms / 1000, tz=UTC)
        observed_time = datetime.fromtimestamp(observed_ms / 1000, tz=UTC)

        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "correlation_id", correlation_id)
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "exchange_time_ms", exchange_ms)
        object.__setattr__(self, "observed_time_ms", observed_ms)
        object.__setattr__(self, "hedge_event_version", hedge_event_version)
        object.__setattr__(self, "position_key", None)
        object.__setattr__(self, "trade_id", order_id or event_id)
        object.__setattr__(self, "client_order_id", order_id)
        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "action", None)
        object.__setattr__(self, "order_side", None)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "fee", Decimal("0"))
        object.__setattr__(self, "fee_currency", "USDT")
        object.__setattr__(self, "exchange_time", exchange_time)
        object.__setattr__(self, "observed_time", observed_time)
        object.__setattr__(self, "payload_version", 1)


@dataclass(frozen=True, slots=True)
class OutboxEvent:
    event_type: str
    payload: Mapping[str, Any]
    correlation_id: str | None = None
    event_id: UUID = field(default_factory=uuid4)
    payload_version: int = 1
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    published_at: datetime | None = None
    attempts: int = 0

    def __post_init__(self) -> None:
        event_type = str(self.event_type).strip().upper()
        if not event_type or len(event_type) > 128:
            raise ValueError("event_type is invalid")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        if not isinstance(self.payload_version, int) or self.payload_version <= 0:
            raise ValueError("payload_version must be positive")
        if not isinstance(self.attempts, int) or self.attempts < 0:
            raise ValueError("attempts must not be negative")
        occurred = _aware(self.occurred_at, name="occurred_at")
        published = None if self.published_at is None else _aware(self.published_at, name="published_at")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "payload", MappingProxyType(dict(self.payload)))
        object.__setattr__(self, "occurred_at", occurred)
        object.__setattr__(self, "published_at", published)


@dataclass(frozen=True, slots=True)
class PositionSnapshot(AccountEvent):
    positions: tuple["LegacyPositionRecord", ...] = ()
