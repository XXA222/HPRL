"""Cross-direction ports used by the integrated execution engine."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from enum import StrEnum
from threading import RLock
from typing import ContextManager, Iterator, Mapping, Protocol

from .events import AccountEvent, FillEvent, OutboxEvent, PositionSnapshot
from .types import ApprovedOrderIntent, OrderIntent, OrderSnapshot, PositionKey, PositionRecord


class ReadinessState(StrEnum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    HALT = "HALT"


@dataclass(frozen=True, slots=True)
class ReadinessDecision:
    state: ReadinessState
    reason_codes: tuple[str, ...] = ()
    allow_reduce: bool = True

    @property
    def allow_increase(self) -> bool:
        return self.state is ReadinessState.READY


class ReadinessGatePort(Protocol):
    def evaluate(self, position_key: PositionKey) -> ReadinessDecision: ...


class SingleWriterPort(Protocol):
    def assert_leader(self, *, account_id: str, now: datetime) -> None: ...
    def claim(self, *, account_id: str, owner_id: str) -> bool: ...


class PositionLockPort(Protocol):
    def acquire(
        self,
        position_key: PositionKey | None = None,
        *,
        key: PositionKey | None = None,
        owner_id: str | None = None,
    ) -> ContextManager[None] | bool: ...


class ClockPort(Protocol):
    def now(self) -> datetime: ...
    def now_ms(self) -> int: ...


@dataclass(frozen=True, slots=True)
class MarketRules:
    quantity_step: Decimal = Decimal("0.001")
    price_tick: Decimal = Decimal("0.01")
    minimum_quantity: Decimal = Decimal("0.001")
    minimum_notional: Decimal = Decimal("5")

    def __post_init__(self) -> None:
        for name in ("quantity_step", "price_tick", "minimum_quantity", "minimum_notional"):
            value = getattr(self, name)
            if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
                raise ValueError(f"{name} must be a positive Decimal")

    def normalize_quantity(self, quantity: Decimal) -> Decimal:
        units = (quantity / self.quantity_step).to_integral_value(rounding=ROUND_DOWN)
        return units * self.quantity_step

    def normalize_price(
        self,
        price: Decimal,
        *,
        order_side: str | None = None,
        post_only: bool = False,
    ) -> Decimal:
        """Normalize price without making SELL orders unintentionally more aggressive.

        BUY prices round down. SELL prices round up. The legacy no-side call preserves
        ROUND_DOWN for compatibility. ``post_only`` is recorded in the contract so an
        exchange-specific rule adapter may additionally enforce book non-crossing.
        """
        del post_only
        side = None if order_side is None else str(getattr(order_side, "value", order_side)).upper()
        if side not in {None, "BUY", "SELL"}:
            raise ValueError("order_side must be BUY, SELL, or None")
        rounding = ROUND_UP if side == "SELL" else ROUND_DOWN
        units = (price / self.price_tick).to_integral_value(rounding=rounding)
        return units * self.price_tick


class MarketRulesPort(Protocol):
    def rules_for(self, position_key: PositionKey) -> MarketRules: ...


class ExecutionTransactionPort(Protocol):
    def record(
        self,
        *,
        order: object,
        event_type: str,
        fill: FillEvent | None = None,
        outbox: OutboxEvent | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> None: ...


class EventPublisherPort(Protocol):
    def publish(self, event: OutboxEvent) -> None: ...


class AlwaysReadyGate:
    def evaluate(self, position_key: PositionKey) -> ReadinessDecision:
        return ReadinessDecision(ReadinessState.READY)


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)

    def now_ms(self) -> int:
        return int(self.now().timestamp() * 1000)


class InMemorySingleWriter:
    def __init__(self, *, leader: bool = True) -> None:
        self._leader = leader
        self._lock = RLock()

    def set_leader(self, leader: bool) -> None:
        if not isinstance(leader, bool):
            raise TypeError("leader must be a boolean")
        with self._lock:
            self._leader = leader

    def assert_leader(self, *, account_id: str, now: datetime) -> None:
        del account_id, now
        with self._lock:
            if not self._leader:
                raise RuntimeError("SINGLE_WRITER_LOST")

    def claim(self, *, account_id: str, owner_id: str) -> bool:
        # Compatibility with the original boolean-claim port.  The in-memory
        # implementation has no owner registry; it reports the same leader
        # state enforced by assert_leader().
        del account_id, owner_id
        with self._lock:
            return self._leader


class InMemoryPositionLock:
    def __init__(self, *, stripes: int = 257) -> None:
        if not isinstance(stripes, int) or stripes <= 0:
            raise ValueError("stripes must be positive")
        self._locks = tuple(RLock() for _ in range(stripes))
        self._legacy_owners: dict[PositionKey, str] = {}
        self._owner_lock = RLock()

    def acquire(
        self,
        position_key: PositionKey | None = None,
        *,
        key: PositionKey | None = None,
        owner_id: str | None = None,
    ) -> ContextManager[None] | bool:
        resolved = position_key if position_key is not None else key
        if resolved is None:
            raise TypeError("position_key or key is required")
        if owner_id is not None:
            owner = str(owner_id).strip()
            if not owner:
                raise ValueError("owner_id must not be empty")
            with self._owner_lock:
                current = self._legacy_owners.get(resolved)
                if current not in {None, owner}:
                    return False
                self._legacy_owners[resolved] = owner
                return True
        return self._context(resolved)

    @contextmanager
    def _context(self, position_key: PositionKey) -> Iterator[None]:
        lock = self._locks[hash(position_key) % len(self._locks)]
        with lock:
            yield


class StaticMarketRules:
    def __init__(self, default: MarketRules | None = None) -> None:
        self._default = default or MarketRules()
        self._overrides: dict[PositionKey, MarketRules] = {}

    def set_rules(self, position_key: PositionKey, rules: MarketRules) -> None:
        self._overrides[position_key] = rules

    def rules_for(self, position_key: PositionKey) -> MarketRules:
        return self._overrides.get(position_key, self._default)


class NullExecutionTransaction:
    def record(self, **kwargs: object) -> None:
        del kwargs


class NullEventPublisher:
    def publish(self, event: OutboxEvent) -> None:
        del event


class PositionRepository(Protocol):
    def get(self, key: PositionKey) -> PositionRecord | None: ...
    def save(self, record: PositionRecord) -> None: ...


class OrderIntentRepository(Protocol):
    def save(self, intent: OrderIntent) -> None: ...
    def get_by_idempotency_key(self, key: str) -> OrderIntent | None: ...


class FillRepository(Protocol):
    def save(self, event: FillEvent) -> None: ...


class SnapshotRepository(Protocol):
    def save(self, snapshot: PositionSnapshot) -> None: ...


class OutboxRepository(Protocol):
    def enqueue(self, event: AccountEvent) -> None: ...


class ReadonlyExchangePort(Protocol):
    def fetch_positions(self) -> tuple[PositionRecord, ...]: ...
    def fetch_orders(self) -> tuple[OrderSnapshot, ...]: ...


class ExecutionExchangePort(Protocol):
    def submit(self, intent: ApprovedOrderIntent) -> OrderSnapshot: ...


class RiskEvaluationPort(Protocol):
    def readiness(self) -> ReadinessState: ...
