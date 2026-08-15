"""Fixed Account -> Position -> Order lock ordering."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from threading import local
from typing import Protocol, TypeVar

from freqtrade.enums.hedge import PositionSide
from freqtrade.hedge.symbols import canonicalize_symbol


class LockOrderViolation(RuntimeError):
    """A thread/task attempted to acquire locks in a non-canonical order."""


class LockLevel(IntEnum):
    ACCOUNT = 10
    POSITION = 20
    ORDER = 30


class OrderedLockKey(Protocol):
    @property
    def sort_key(self) -> tuple[object, ...]: ...


_SIDE_ORDER = {
    PositionSide.LONG: 0,
    PositionSide.SHORT: 1,
    PositionSide.BOTH: 2,
}


def _account_id(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("account_id must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError("account_id must not be empty.")
    return normalized


def _exchange(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("exchange must not be empty.")
    return value.strip().lower()


@dataclass(frozen=True, slots=True)
class AccountLockKey:
    account_id: str
    exchange: str = "binance"

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_id", _account_id(self.account_id))
        object.__setattr__(self, "exchange", _exchange(self.exchange))

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (LockLevel.ACCOUNT, self.exchange, self.account_id)


@dataclass(frozen=True, slots=True)
class PositionLockKey:
    account_id: str
    symbol: str
    position_side: PositionSide
    exchange: str = "binance"

    def __post_init__(self) -> None:
        side = (
            self.position_side
            if isinstance(self.position_side, PositionSide)
            else PositionSide(str(self.position_side).upper())
        )
        if side is PositionSide.BOTH:
            raise ValueError("Position lock side must be LONG or SHORT.")
        object.__setattr__(self, "account_id", _account_id(self.account_id))
        object.__setattr__(self, "exchange", _exchange(self.exchange))
        object.__setattr__(self, "symbol", canonicalize_symbol(self.symbol))
        object.__setattr__(self, "position_side", side)

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (
            LockLevel.POSITION,
            self.exchange,
            self.account_id,
            self.symbol,
            _SIDE_ORDER[self.position_side],
        )


@dataclass(frozen=True, slots=True)
class OrderLockKey:
    account_id: str
    symbol: str
    position_side: PositionSide
    order_id: str
    exchange: str = "binance"

    def __post_init__(self) -> None:
        position = PositionLockKey(
            self.account_id,
            self.symbol,
            self.position_side,
            self.exchange,
        )
        if not isinstance(self.order_id, str) or not self.order_id.strip():
            raise ValueError("order_id must not be empty.")
        object.__setattr__(self, "account_id", position.account_id)
        object.__setattr__(self, "exchange", position.exchange)
        object.__setattr__(self, "symbol", position.symbol)
        object.__setattr__(self, "position_side", position.position_side)
        object.__setattr__(self, "order_id", self.order_id.strip())

    @property
    def sort_key(self) -> tuple[object, ...]:
        return (
            LockLevel.ORDER,
            self.exchange,
            self.account_id,
            self.symbol,
            _SIDE_ORDER[self.position_side],
            self.order_id,
        )


K = TypeVar("K", bound=OrderedLockKey)


def ordered_lock_keys(keys: list[K] | tuple[K, ...]) -> tuple[K, ...]:
    unique = {key: None for key in keys}
    return tuple(sorted(unique, key=lambda item: item.sort_key))


class LockOrderTracker:
    def __init__(self) -> None:
        self._local = local()

    def _stack(self) -> list[OrderedLockKey]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def before_acquire(self, key: OrderedLockKey) -> None:
        stack = self._stack()
        if stack and key not in stack and key.sort_key < stack[-1].sort_key:
            raise LockOrderViolation(
                f"Lock order violation: attempted {key.sort_key} after {stack[-1].sort_key}."
            )

    def acquired(self, key: OrderedLockKey) -> None:
        self._stack().append(key)

    def released(self, key: OrderedLockKey) -> None:
        stack = self._stack()
        if not stack or stack[-1] != key:
            raise LockOrderViolation(f"Locks must be released in reverse order: {key}.")
        stack.pop()
