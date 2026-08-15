"""Side-specific position locks, timeout/deadlock detection and reduce reservations."""

from __future__ import annotations

import math
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Iterator

from freqtrade.enums.hedge import PositionSide
from freqtrade.hedge.concurrency.lock_order import (
    LockOrderTracker,
    PositionLockKey,
    ordered_lock_keys,
)
from freqtrade.hedge.local_reduce_only import calculate_safe_reduce
from freqtrade.hedge.numeric import require_nonnegative


class PositionLockTimeout(TimeoutError):
    pass


class DeadlockDetected(RuntimeError):
    pass


def _positive_timeout(value: float, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a positive finite number.")
    timeout = float(value)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{field_name} must be a positive finite number.")
    return timeout


@dataclass(slots=True)
class _LockEntry:
    lock: threading.RLock = field(default_factory=threading.RLock)
    owner_thread_id: int | None = None
    depth: int = 0


@dataclass(frozen=True, slots=True)
class _ReduceReservationRecord:
    quantity: Decimal
    expires_at_monotonic: float


@dataclass(slots=True)
class ReduceReservation:
    manager: "PositionLockManager"
    key: PositionLockKey
    token: str | None
    requested_quantity: Decimal
    allowed_quantity: Decimal
    reason_code: str
    expires_at_monotonic: float | None = None
    _released: bool = False

    def release(self) -> None:
        if self._released or self.token is None:
            self._released = True
            return
        self.manager.release_reservation(self.key, self.token)
        self._released = True

    def confirm(self) -> None:
        """Remove the local reservation after the exchange/ledger confirms it."""

        self.release()

    @property
    def expired(self) -> bool:
        return (
            self.expires_at_monotonic is not None
            and self.manager.monotonic_time() >= self.expires_at_monotonic
        )

    def __enter__(self) -> "ReduceReservation":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class PositionLockManager:
    def __init__(
        self,
        *,
        default_timeout_seconds: float = 5.0,
        reservation_ttl_seconds: float = 30.0,
        monotonic_clock: Callable[[], float] | None = None,
    ) -> None:
        self._default_timeout = _positive_timeout(
            default_timeout_seconds,
            field_name="default_timeout_seconds",
        )
        self._reservation_ttl = _positive_timeout(
            reservation_ttl_seconds,
            field_name="reservation_ttl_seconds",
        )
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._state_lock = threading.RLock()
        self._entries: dict[PositionLockKey, _LockEntry] = {}
        self._waiting: dict[int, PositionLockKey] = {}
        self._reservations: dict[
            PositionLockKey, dict[str, _ReduceReservationRecord]
        ] = {}
        self._order = LockOrderTracker()

    def monotonic_time(self) -> float:
        value = self._monotonic_clock()
        if isinstance(value, bool):
            raise ValueError("monotonic_clock must return a finite nonnegative number.")
        result = float(value)
        if not math.isfinite(result) or result < 0:
            raise ValueError("monotonic_clock must return a finite nonnegative number.")
        return result

    def _prune_expired_locked(self, now: float) -> int:
        removed = 0
        for key, reservations in tuple(self._reservations.items()):
            for token, record in tuple(reservations.items()):
                if record.expires_at_monotonic <= now:
                    reservations.pop(token, None)
                    removed += 1
            if not reservations:
                self._reservations.pop(key, None)
        return removed

    def prune_expired_reservations(self) -> int:
        now = self.monotonic_time()
        with self._state_lock:
            return self._prune_expired_locked(now)

    def _entry(self, key: PositionLockKey) -> _LockEntry:
        with self._state_lock:
            return self._entries.setdefault(key, _LockEntry())

    def _would_deadlock(self, waiter: int, owner: int | None) -> bool:
        visited: set[int] = set()
        current = owner
        while current is not None and current not in visited:
            if current == waiter:
                return True
            visited.add(current)
            waiting_key = self._waiting.get(current)
            if waiting_key is None:
                return False
            entry = self._entries.get(waiting_key)
            current = None if entry is None else entry.owner_thread_id
        return False

    @contextmanager
    def lock(
        self,
        *,
        account_id: str,
        symbol: str,
        position_side: PositionSide | str,
        exchange: str = "binance",
        timeout_seconds: float | None = None,
    ) -> Iterator[PositionLockKey]:
        key = PositionLockKey(account_id, symbol, position_side, exchange)
        timeout = (
            self._default_timeout
            if timeout_seconds is None
            else _positive_timeout(timeout_seconds, field_name="timeout_seconds")
        )
        self._order.before_acquire(key)
        entry = self._entry(key)
        thread_id = threading.get_ident()
        acquired = entry.lock.acquire(blocking=False)
        if not acquired:
            with self._state_lock:
                self._waiting[thread_id] = key
                if self._would_deadlock(thread_id, entry.owner_thread_id):
                    self._waiting.pop(thread_id, None)
                    raise DeadlockDetected(f"Deadlock detected while waiting for {key}.")
            try:
                acquired = entry.lock.acquire(timeout=timeout)
            finally:
                with self._state_lock:
                    self._waiting.pop(thread_id, None)
        if not acquired:
            raise PositionLockTimeout(f"Timed out waiting for position lock {key}.")
        try:
            with self._state_lock:
                entry.owner_thread_id = thread_id
                entry.depth += 1
            self._order.acquired(key)
            yield key
        finally:
            self._order.released(key)
            with self._state_lock:
                entry.depth -= 1
                if entry.depth == 0:
                    entry.owner_thread_id = None
            entry.lock.release()

    @contextmanager
    def lock_many(
        self,
        keys: list[PositionLockKey] | tuple[PositionLockKey, ...],
        *,
        timeout_seconds: float | None = None,
    ) -> Iterator[tuple[PositionLockKey, ...]]:
        ordered = ordered_lock_keys(keys)
        timeout = (
            self._default_timeout
            if timeout_seconds is None
            else _positive_timeout(timeout_seconds, field_name="timeout_seconds")
        )
        deadline = time.monotonic() + timeout
        contexts = []
        try:
            for key in ordered:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PositionLockTimeout(
                        "Timed out before all position locks could be acquired."
                    )
                context = self.lock(
                    account_id=key.account_id,
                    symbol=key.symbol,
                    position_side=key.position_side,
                    exchange=key.exchange,
                    timeout_seconds=remaining,
                )
                context.__enter__()
                contexts.append(context)
            yield ordered
        finally:
            while contexts:
                contexts.pop().__exit__(None, None, None)

    def pending_reduce_quantity(self, key: PositionLockKey) -> Decimal:
        now = self.monotonic_time()
        with self._state_lock:
            self._prune_expired_locked(now)
            return sum(
                (record.quantity for record in self._reservations.get(key, {}).values()),
                Decimal("0"),
            )

    def reserve_reduce(
        self,
        *,
        account_id: str,
        symbol: str,
        position_side: PositionSide | str,
        requested_quantity: Decimal,
        exchange: str = "binance",
        confirmed_quantity: Decimal,
        existing_pending_reduce_quantity: Decimal = Decimal("0"),
        timeout_seconds: float | None = None,
        pre_reservation_check: Callable[[], None] | None = None,
    ) -> ReduceReservation:
        with self.lock(
            account_id=account_id,
            symbol=symbol,
            position_side=position_side,
            exchange=exchange,
            timeout_seconds=timeout_seconds,
        ) as key:
            if pre_reservation_check is not None:
                pre_reservation_check()
            local_pending = self.pending_reduce_quantity(key)
            known_pending = require_nonnegative(
                existing_pending_reduce_quantity,
                field="existing_pending_reduce_quantity",
            )
            decision = calculate_safe_reduce(
                requested_quantity=requested_quantity,
                confirmed_quantity=confirmed_quantity,
                pending_reduce_quantity=known_pending + local_pending,
            )
            token: str | None = None
            expires_at: float | None = None
            if decision.allowed_quantity > 0:
                token = uuid.uuid4().hex
                expires_at = self.monotonic_time() + self._reservation_ttl
                with self._state_lock:
                    self._prune_expired_locked(self.monotonic_time())
                    self._reservations.setdefault(key, {})[token] = _ReduceReservationRecord(
                        decision.allowed_quantity,
                        expires_at,
                    )
            return ReduceReservation(
                manager=self,
                key=key,
                token=token,
                requested_quantity=decision.requested_quantity,
                allowed_quantity=decision.allowed_quantity,
                reason_code=decision.reason_code,
                expires_at_monotonic=expires_at,
            )

    def release_reservation(self, key: PositionLockKey, token: str) -> None:
        # Reservation accounting is protected by _state_lock. Reacquiring the
        # position lock here can turn harmless cleanup into a timeout/deadlock.
        with self._state_lock:
            reservations = self._reservations.get(key)
            if reservations is None:
                return
            reservations.pop(token, None)
            if not reservations:
                self._reservations.pop(key, None)

    def release_all_reservations(self, *, account_id: str | None = None) -> int:
        normalized = None
        if account_id is not None:
            if not isinstance(account_id, str) or not account_id.strip():
                raise ValueError("account_id must be a non-empty string.")
            normalized = account_id.strip()
        removed = 0
        with self._state_lock:
            for key in tuple(self._reservations):
                if normalized is not None and key.account_id != normalized:
                    continue
                removed += len(self._reservations.pop(key, {}))
        return removed

    def reservation_snapshot(self) -> tuple[dict[str, object], ...]:
        now = self.monotonic_time()
        with self._state_lock:
            self._prune_expired_locked(now)
            rows = [
                {
                    "exchange": key.exchange,
                    "account_id": key.account_id,
                    "symbol": key.symbol,
                    "position_side": key.position_side.value,
                    "token": token,
                    "quantity": str(record.quantity),
                    "expires_in_seconds": max(record.expires_at_monotonic - now, 0.0),
                }
                for key, reservations in self._reservations.items()
                for token, record in reservations.items()
            ]
        return tuple(
            sorted(
                rows,
                key=lambda item: (
                    str(item["exchange"]),
                    str(item["account_id"]),
                    str(item["symbol"]),
                    str(item["position_side"]),
                    str(item["token"]),
                ),
            )
        )
