"""Atomic in-process reservations for live-candidate risk capacity.

The exchange/risk database remains authoritative.  This book closes the process-local
TOCTOU gap between canary admission and exchange submission: concurrent intents reserve
capacity before they may reach the adapter, and reservations expire fail-closed.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from threading import RLock
from typing import Iterable


ZERO = Decimal("0")


class ReservationState(StrEnum):
    HELD = "HELD"
    COMMITTED = "COMMITTED"
    RELEASED = "RELEASED"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True, slots=True)
class ExposureReservation:
    reservation_id: str
    client_order_id: str
    notional: Decimal
    created_at: datetime
    expires_at: datetime
    state: ReservationState = ReservationState.HELD

    def __post_init__(self) -> None:
        if not self.reservation_id.strip() or not self.client_order_id.strip():
            raise ValueError("reservation_id and client_order_id are required")
        value = Decimal(self.notional)
        if not value.is_finite() or value <= ZERO:
            raise ValueError("reservation notional must be finite and positive")
        if self.created_at.tzinfo is None or self.expires_at.tzinfo is None:
            raise ValueError("reservation timestamps must be timezone-aware")
        if self.expires_at <= self.created_at:
            raise ValueError("reservation expiry must follow creation")
        object.__setattr__(self, "notional", value)
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        object.__setattr__(self, "expires_at", self.expires_at.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class ReservationSnapshot:
    held_notional: Decimal
    held_orders: int
    reservations: tuple[ExposureReservation, ...]


class ExposureReservationBook:
    """Thread-safe, idempotent reservation book for candidate live orders."""

    def __init__(self, *, ttl: timedelta = timedelta(seconds=30)) -> None:
        if ttl <= timedelta(0) or ttl > timedelta(minutes=5):
            raise ValueError("reservation ttl must be in (0, 5m]")
        self._ttl = ttl
        self._lock = RLock()
        self._items: dict[str, ExposureReservation] = {}
        self._by_client_order: dict[str, str] = {}
        self._sequence = 0

    def reserve(
        self,
        *,
        client_order_id: str,
        notional: Decimal,
        now: datetime,
        max_total_notional: Decimal,
        max_orders: int,
    ) -> ExposureReservation:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now = now.astimezone(UTC)
        notional = Decimal(notional)
        max_total_notional = Decimal(max_total_notional)
        if notional <= ZERO or max_total_notional < ZERO:
            raise ValueError("invalid reservation notional")
        if max_orders <= 0:
            raise ValueError("max_orders must be positive")
        key = client_order_id.strip()
        if not key:
            raise ValueError("client_order_id is required")
        with self._lock:
            self._expire_locked(now)
            existing_id = self._by_client_order.get(key)
            if existing_id is not None:
                existing = self._items[existing_id]
                if existing.notional != notional:
                    raise ValueError("idempotent reservation notional mismatch")
                if existing.state in {ReservationState.HELD, ReservationState.COMMITTED}:
                    return existing
                # A clientOrderId is an execution idempotency identity.  Once its
                # reservation reaches a terminal state it must never be recycled for
                # another exchange write, even if the notional happens to match.
                raise ValueError("terminal client_order_id reservation cannot be reused")
            active = [
                x
                for x in self._items.values()
                if x.state in {ReservationState.HELD, ReservationState.COMMITTED}
            ]
            if len(active) + 1 > max_orders:
                raise PermissionError("CANARY_RESERVATION_ORDER_LIMIT")
            if sum((x.notional for x in active), ZERO) + notional > max_total_notional:
                raise PermissionError("CANARY_RESERVATION_GROSS_LIMIT")
            self._sequence += 1
            rid = f"canary-{self._sequence:016d}"
            item = ExposureReservation(rid, key, notional, now, now + self._ttl)
            self._items[rid] = item
            self._by_client_order[key] = rid
            return item

    def commit(self, reservation_id: str, *, now: datetime) -> ExposureReservation:
        return self._transition(reservation_id, ReservationState.COMMITTED, now=now)

    def release(self, reservation_id: str, *, now: datetime) -> ExposureReservation:
        return self._transition(reservation_id, ReservationState.RELEASED, now=now)

    def expire(self, *, now: datetime) -> tuple[ExposureReservation, ...]:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self._lock:
            return self._expire_locked(now.astimezone(UTC))

    def snapshot(self, *, now: datetime) -> ReservationSnapshot:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self._lock:
            self._expire_locked(now.astimezone(UTC))
            held = tuple(sorted(
                (
                    x
                    for x in self._items.values()
                    if x.state in {ReservationState.HELD, ReservationState.COMMITTED}
                ),
                key=lambda x: x.reservation_id,
            ))
            return ReservationSnapshot(sum((x.notional for x in held), ZERO), len(held), held)

    def find_by_client_order(
        self, client_order_id: str, *, now: datetime
    ) -> ExposureReservation | None:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        with self._lock:
            self._expire_locked(now.astimezone(UTC))
            rid = self._by_client_order.get(client_order_id.strip())
            return self._items.get(rid) if rid is not None else None

    def _transition(
        self, reservation_id: str, state: ReservationState, *, now: datetime
    ) -> ExposureReservation:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now = now.astimezone(UTC)
        with self._lock:
            self._expire_locked(now)
            current = self._items[reservation_id]
            if current.state is state:
                return current
            allowed = {
                ReservationState.HELD: {ReservationState.COMMITTED, ReservationState.RELEASED},
                ReservationState.COMMITTED: {ReservationState.RELEASED},
            }
            if state not in allowed.get(current.state, set()):
                raise ValueError(
                    f"invalid reservation transition: {current.state.value}->{state.value}"
                )
            updated = replace(current, state=state)
            self._items[reservation_id] = updated
            return updated

    def _expire_locked(self, now: datetime) -> tuple[ExposureReservation, ...]:
        expired: list[ExposureReservation] = []
        for rid, current in tuple(self._items.items()):
            if current.state is ReservationState.HELD and now >= current.expires_at:
                updated = replace(current, state=ReservationState.EXPIRED)
                self._items[rid] = updated
                expired.append(updated)
        return tuple(expired)
