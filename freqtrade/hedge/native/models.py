"""Typed contracts for the Freqtrade-to-Hedge native convergence layer.

The module deliberately depends only on the Python standard library.  It is used by
live runtime adapters, backtesting, Hyperopt, FreqAI, RPC projections and offline
verification without importing exchange clients or database state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Mapping


ZERO = Decimal("0")
ONE = Decimal("1")


def finite_decimal(value: object, *, field_name: str) -> Decimal:
    """Return an exact finite Decimal and reject booleans and binary float noise."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a finite decimal")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be a finite decimal")
    return result


def utc_datetime(value: datetime | None = None) -> datetime:
    result = datetime.now(UTC) if value is None else value
    if result.tzinfo is None:
        return result.replace(tzinfo=UTC)
    return result.astimezone(UTC)


class HedgeSide(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def pairlock_side(self) -> str:
        return self.value.lower()

    @classmethod
    def parse(cls, value: object) -> "HedgeSide":
        if isinstance(value, cls):
            return value
        raw = getattr(value, "value", value)
        return cls(str(raw).upper())


class HedgeBucket(StrEnum):
    CORE = "CORE"
    TACTICAL = "TACTICAL"


class HedgeAction(StrEnum):
    OPEN = "OPEN"
    INCREASE = "INCREASE"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    UNSTUCK = "UNSTUCK"
    CANCEL = "CANCEL"
    RECOVER = "RECOVER"

    @property
    def reduce_only(self) -> bool:
        return self in {self.REDUCE, self.CLOSE, self.UNSTUCK}


class NativeBotMode(StrEnum):
    RUNNING = "RUNNING"
    REDUCE_ONLY = "REDUCE_ONLY"
    STOPPED = "STOPPED"
    RELOAD = "RELOAD"
    UNKNOWN = "UNKNOWN"


class AdmissionCode(StrEnum):
    ALLOWED = "ALLOWED"
    BOT_PAUSED = "BOT_PAUSED"
    BOT_STOPPED = "BOT_STOPPED"
    BOT_RELOAD = "BOT_RELOAD"
    GLOBAL_PAIRLOCK = "GLOBAL_PAIRLOCK"
    PAIR_LOCKED = "PAIR_LOCKED"
    PROTECTION_ERROR = "PROTECTION_ERROR"
    CAPITAL_EXHAUSTED = "CAPITAL_EXHAUSTED"
    ORDER_TOO_LARGE = "ORDER_TOO_LARGE"
    MODEL_NOT_READY = "MODEL_NOT_READY"
    PRODUCER_STALE = "PRODUCER_STALE"
    UNIVERSE_REJECTED = "UNIVERSE_REJECTED"
    STRATEGY_REJECTED = "STRATEGY_REJECTED"
    KILL_SWITCH = "KILL_SWITCH"
    READINESS_BLOCKED = "READINESS_BLOCKED"
    INVALID_INTENT = "INVALID_INTENT"


@dataclass(frozen=True, slots=True)
class NativeOrderIntent:
    """Minimal side-aware order intent consumed by all admission adapters."""

    pair: str
    side: HedgeSide
    action: HedgeAction
    quantity: Decimal
    price: Decimal
    bucket: HedgeBucket = HedgeBucket.TACTICAL
    intent_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        pair = str(self.pair).strip().upper()
        if not pair:
            raise ValueError("pair is required")
        quantity = finite_decimal(self.quantity, field_name="quantity")
        price = finite_decimal(self.price, field_name="price")
        if quantity <= ZERO or price <= ZERO:
            raise ValueError("quantity and price must be positive")
        object.__setattr__(self, "pair", pair)
        object.__setattr__(self, "side", HedgeSide.parse(self.side))
        object.__setattr__(self, "action", HedgeAction(self.action))
        object.__setattr__(self, "bucket", HedgeBucket(self.bucket))
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def reduce_only(self) -> bool:
        return self.action.reduce_only

    @property
    def notional(self) -> Decimal:
        return self.quantity * self.price


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    allowed: bool
    code: AdmissionCode = AdmissionCode.ALLOWED
    reason: str = ""
    reduce_only_exempt: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", AdmissionCode(self.code))
        object.__setattr__(self, "reason", str(self.reason))
        object.__setattr__(self, "metadata", dict(self.metadata))
        if self.allowed and self.code is not AdmissionCode.ALLOWED:
            raise ValueError("allowed decisions must use ALLOWED code")

    @classmethod
    def allow(cls, *, reason: str = "", reduce_only_exempt: bool = False) -> "AdmissionDecision":
        return cls(True, AdmissionCode.ALLOWED, reason, reduce_only_exempt)

    @classmethod
    def block(
        cls,
        code: AdmissionCode,
        reason: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "AdmissionDecision":
        if code is AdmissionCode.ALLOWED:
            raise ValueError("blocking decision cannot use ALLOWED")
        return cls(False, code, reason, False, metadata or {})


@dataclass(frozen=True, slots=True)
class BotStateSnapshot:
    source_state: str
    mode: NativeBotMode
    allow_planner: bool
    allow_new_risk: bool
    allow_reduce_only: bool
    allow_recovery: bool
    cancel_managed_orders: bool
    observed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", NativeBotMode(self.mode))
        object.__setattr__(self, "observed_at", utc_datetime(self.observed_at))


@dataclass(frozen=True, slots=True)
class ProtectionSnapshot:
    pair: str
    side: HedgeSide
    global_locked: bool
    pair_locked: bool
    reasons: tuple[str, ...]
    observed_at: datetime

    @property
    def blocked(self) -> bool:
        return self.global_locked or self.pair_locked


@dataclass(frozen=True, slots=True)
class CapitalSnapshot:
    equity: Decimal
    available_balance: Decimal
    official_capital_limit: Decimal
    hedge_capital_limit: Decimal
    effective_capital_limit: Decimal
    current_gross_notional: Decimal
    remaining_notional: Decimal
    max_single_order_notional: Decimal
    observed_at: datetime

    def __post_init__(self) -> None:
        names = (
            "equity",
            "available_balance",
            "official_capital_limit",
            "hedge_capital_limit",
            "effective_capital_limit",
            "current_gross_notional",
            "remaining_notional",
            "max_single_order_notional",
        )
        for name in names:
            value = finite_decimal(getattr(self, name), field_name=name)
            if value < ZERO:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "observed_at", utc_datetime(self.observed_at))


@dataclass(frozen=True, slots=True)
class LegSnapshot:
    pair: str
    side: HedgeSide
    bucket: HedgeBucket
    quantity: Decimal
    average_price: Decimal
    mark_price: Decimal
    realized_pnl: Decimal = ZERO
    funding: Decimal = ZERO
    fees: Decimal = ZERO
    opened_at: datetime | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pair", str(self.pair).strip().upper())
        object.__setattr__(self, "side", HedgeSide.parse(self.side))
        object.__setattr__(self, "bucket", HedgeBucket(self.bucket))
        for name in (
            "quantity",
            "average_price",
            "mark_price",
            "realized_pnl",
            "funding",
            "fees",
        ):
            value = finite_decimal(getattr(self, name), field_name=name)
            object.__setattr__(self, name, value)
        if self.quantity < ZERO:
            raise ValueError("quantity cannot be negative")
        if self.quantity > ZERO and (self.average_price <= ZERO or self.mark_price <= ZERO):
            raise ValueError("open leg prices must be positive")
        if self.opened_at is not None:
            object.__setattr__(self, "opened_at", utc_datetime(self.opened_at))

    @property
    def profit_ratio(self) -> Decimal:
        if self.quantity <= ZERO:
            return ZERO
        if self.side is HedgeSide.LONG:
            return self.mark_price / self.average_price - ONE
        return self.average_price / self.mark_price - ONE

    @property
    def unrealized_pnl(self) -> Decimal:
        if self.quantity <= ZERO:
            return ZERO
        direction = ONE if self.side is HedgeSide.LONG else -ONE
        return (self.mark_price - self.average_price) * self.quantity * direction


@dataclass(frozen=True, slots=True)
class ExitDecision:
    should_exit: bool
    fraction: Decimal = ZERO
    reason: str = ""
    priority: int = 0
    hard_risk: bool = False

    def __post_init__(self) -> None:
        fraction = finite_decimal(self.fraction, field_name="fraction")
        if fraction < ZERO or fraction > ONE:
            raise ValueError("exit fraction must be between zero and one")
        if self.should_exit and fraction <= ZERO:
            raise ValueError("an exit decision requires a positive fraction")
        if not self.should_exit and fraction != ZERO:
            raise ValueError("non-exit decisions must use zero fraction")
        object.__setattr__(self, "fraction", fraction)

    @classmethod
    def hold(cls, reason: str = "") -> "ExitDecision":
        return cls(False, ZERO, reason, 0, False)


@dataclass(frozen=True, slots=True)
class ModelReadinessSnapshot:
    ready: bool
    model_version: str
    trained_at: datetime | None
    expires_at: datetime | None
    feature_schema: str
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.trained_at is not None:
            object.__setattr__(self, "trained_at", utc_datetime(self.trained_at))
        if self.expires_at is not None:
            object.__setattr__(self, "expires_at", utc_datetime(self.expires_at))


@dataclass(frozen=True, slots=True)
class HedgeEvent:
    event_type: str
    pair: str
    side: HedgeSide | None
    payload: Mapping[str, Any]
    timestamp: datetime
    severity: str = "INFO"
    correlation_id: str = ""

    def __post_init__(self) -> None:
        event_type = str(self.event_type).strip().upper()
        if not event_type:
            raise ValueError("event_type is required")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "pair", str(self.pair).strip().upper())
        if self.side is not None:
            object.__setattr__(self, "side", HedgeSide.parse(self.side))
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "timestamp", utc_datetime(self.timestamp))
        object.__setattr__(self, "severity", str(self.severity).strip().upper() or "INFO")
