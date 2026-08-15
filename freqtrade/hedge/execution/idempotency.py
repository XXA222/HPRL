"""Port and in-memory implementation for idempotent execution submission."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Generic, Protocol, TypeVar

T = TypeVar("T")


class ReservationState(StrEnum):
    NEW = "NEW"
    IN_FLIGHT = "IN_FLIGHT"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True, slots=True)
class IdempotencyReservation(Generic[T]):
    state: ReservationState
    value: T | None = None


class IdempotencyPort(Protocol, Generic[T]):
    def reserve(self, key: str) -> IdempotencyReservation[T]: ...

    def complete(self, key: str, value: T) -> None: ...

    def release(self, key: str) -> None: ...


def _normalize_key(key: object) -> str:
    if not isinstance(key, str):
        raise TypeError("idempotency key must be a string")
    result = key.strip()
    if not result:
        raise ValueError("idempotency key must not be empty")
    if len(result) > 256 or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError("idempotency key is invalid")
    return result


class InMemoryIdempotencyStore(Generic[T]):
    def __init__(self) -> None:
        self._in_flight: set[str] = set()
        self._completed: dict[str, T] = {}
        self._lock = RLock()

    def reserve(self, key: str) -> IdempotencyReservation[T]:
        normalized = _normalize_key(key)
        with self._lock:
            if normalized in self._completed:
                return IdempotencyReservation(
                    ReservationState.COMPLETED, self._completed[normalized]
                )
            if normalized in self._in_flight:
                return IdempotencyReservation(ReservationState.IN_FLIGHT)
            self._in_flight.add(normalized)
            return IdempotencyReservation(ReservationState.NEW)

    def complete(self, key: str, value: T) -> None:
        normalized = _normalize_key(key)
        with self._lock:
            if normalized not in self._in_flight and normalized not in self._completed:
                raise KeyError(f"idempotency key was not reserved: {normalized}")
            self._completed[normalized] = value
            self._in_flight.discard(normalized)

    def release(self, key: str) -> None:
        normalized = _normalize_key(key)
        with self._lock:
            self._in_flight.discard(normalized)
