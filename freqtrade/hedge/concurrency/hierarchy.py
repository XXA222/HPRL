"""Synchronous and asyncio lock managers enforcing Account -> Position -> Order."""

from __future__ import annotations

import asyncio
import math
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import AsyncIterator, Iterator

from freqtrade.hedge.concurrency.lock_order import (
    AccountLockKey,
    LockOrderTracker,
    LockOrderViolation,
    OrderLockKey,
    OrderedLockKey,
    PositionLockKey,
    ordered_lock_keys,
)


class HierarchicalLockTimeout(TimeoutError):
    pass


def _timeout(value: float) -> float:
    if isinstance(value, bool):
        raise ValueError("timeout_seconds must be a positive finite number.")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("timeout_seconds must be a positive finite number.")
    return result


class HierarchicalLockManager:
    def __init__(self, *, default_timeout_seconds: float = 5.0) -> None:
        self._timeout = _timeout(default_timeout_seconds)
        self._state = threading.RLock()
        self._locks: dict[OrderedLockKey, threading.RLock] = {}
        self._order = LockOrderTracker()

    def _lock_for(self, key: OrderedLockKey) -> threading.RLock:
        with self._state:
            return self._locks.setdefault(key, threading.RLock())

    @contextmanager
    def acquire(
        self,
        key: OrderedLockKey,
        *,
        timeout_seconds: float | None = None,
    ) -> Iterator[OrderedLockKey]:
        timeout = self._timeout if timeout_seconds is None else _timeout(timeout_seconds)
        self._order.before_acquire(key)
        lock = self._lock_for(key)
        if not lock.acquire(timeout=timeout):
            raise HierarchicalLockTimeout(f"Timed out waiting for lock {key!r}.")
        self._order.acquired(key)
        try:
            yield key
        finally:
            self._order.released(key)
            lock.release()

    @contextmanager
    def acquire_many(
        self,
        keys: tuple[OrderedLockKey, ...] | list[OrderedLockKey],
        *,
        timeout_seconds: float | None = None,
    ) -> Iterator[tuple[OrderedLockKey, ...]]:
        timeout = self._timeout if timeout_seconds is None else _timeout(timeout_seconds)
        deadline = time.monotonic() + timeout
        ordered = ordered_lock_keys(keys)
        contexts = []
        try:
            for key in ordered:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HierarchicalLockTimeout("Timed out before acquiring all locks.")
                context = self.acquire(key, timeout_seconds=remaining)
                context.__enter__()
                contexts.append(context)
            yield ordered
        finally:
            while contexts:
                contexts.pop().__exit__(None, None, None)


_ASYNC_STACK: ContextVar[tuple[OrderedLockKey, ...]] = ContextVar(
    "hedge_async_lock_stack",
    default=(),
)


class AsyncHierarchicalLockManager:
    def __init__(self, *, default_timeout_seconds: float = 5.0) -> None:
        self._timeout = _timeout(default_timeout_seconds)
        self._state = asyncio.Lock()
        self._locks: dict[OrderedLockKey, asyncio.Lock] = {}

    async def _lock_for(self, key: OrderedLockKey) -> asyncio.Lock:
        async with self._state:
            return self._locks.setdefault(key, asyncio.Lock())

    @asynccontextmanager
    async def acquire(
        self,
        key: OrderedLockKey,
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[OrderedLockKey]:
        timeout = self._timeout if timeout_seconds is None else _timeout(timeout_seconds)
        stack = _ASYNC_STACK.get()
        if key in stack:
            raise LockOrderViolation(f"Async hedge locks are not reentrant: {key!r}.")
        if stack and key.sort_key < stack[-1].sort_key:
            raise LockOrderViolation(
                f"Async lock order violation: {key.sort_key} after {stack[-1].sort_key}."
            )
        lock = await self._lock_for(key)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except TimeoutError as exc:
            raise HierarchicalLockTimeout(f"Timed out waiting for async lock {key!r}.") from exc
        token = _ASYNC_STACK.set((*stack, key))
        try:
            yield key
        finally:
            current = _ASYNC_STACK.get()
            if not current or current[-1] != key:
                raise LockOrderViolation("Async locks must be released in reverse order.")
            _ASYNC_STACK.reset(token)
            lock.release()

    @asynccontextmanager
    async def acquire_many(
        self,
        keys: tuple[OrderedLockKey, ...] | list[OrderedLockKey],
        *,
        timeout_seconds: float | None = None,
    ) -> AsyncIterator[tuple[OrderedLockKey, ...]]:
        """Acquire heterogeneous locks in canonical order without blocking the loop."""

        timeout = self._timeout if timeout_seconds is None else _timeout(timeout_seconds)
        deadline = asyncio.get_running_loop().time() + timeout
        ordered = ordered_lock_keys(keys)
        contexts = []
        try:
            for key in ordered:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise HierarchicalLockTimeout(
                        "Timed out before acquiring all async hedge locks."
                    )
                context = self.acquire(key, timeout_seconds=remaining)
                await context.__aenter__()
                contexts.append(context)
            yield ordered
        finally:
            while contexts:
                await contexts.pop().__aexit__(None, None, None)


__all__ = [
    "AccountLockKey",
    "AsyncHierarchicalLockManager",
    "HierarchicalLockManager",
    "HierarchicalLockTimeout",
    "OrderLockKey",
    "PositionLockKey",
]
