from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any, TypeVar

from .shared_rate_limit import SqliteSharedWeightBudget

from .base import Clock, SystemClock

T = TypeVar("T")
_SYSTEM_RANDOM = random.SystemRandom()


class BinanceTransportError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: int | str | None = None,
        retry_after: float | None = None,
        retryable: bool = False,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.retry_after = retry_after
        self.retryable = retryable
        self.payload = payload


class BinanceRateLimitError(BinanceTransportError):
    pass


class BinancePermissionError(BinanceTransportError):
    pass


class BinanceDataError(BinanceTransportError):
    pass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 30.0
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not math.isfinite(self.base_delay_seconds) or self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be finite and nonnegative")
        if not math.isfinite(self.max_delay_seconds) or self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds must be finite and nonnegative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if not math.isfinite(self.jitter_ratio) or self.jitter_ratio < 0:
            raise ValueError("jitter_ratio must be finite and nonnegative")

    def delay_for(self, attempt: int, *, retry_after: float | None = None) -> float:
        if retry_after is not None:
            parsed_retry_after = float(retry_after)
            if not math.isfinite(parsed_retry_after):
                raise ValueError("retry_after must be finite")
            return max(0.0, parsed_retry_after)
        base = min(
            self.base_delay_seconds * (2 ** max(0, attempt - 1)),
            self.max_delay_seconds,
        )
        if self.jitter_ratio <= 0:
            return base
        jitter = base * self.jitter_ratio
        return max(0.0, base + _SYSTEM_RANDOM.uniform(-jitter, jitter))


def parse_retry_after(
    headers: Mapping[str, str], *, now_epoch_seconds: float | None = None
) -> float | None:
    value = None
    for key, item in headers.items():
        if key.lower() == "retry-after":
            value = str(item).strip()
            break
    if not value:
        return None
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else None
    except ValueError:
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                return None
            now_value = time.time() if now_epoch_seconds is None else now_epoch_seconds
            return max(0.0, parsed.timestamp() - now_value)
        except (TypeError, ValueError, OverflowError):
            return None


class AdaptiveWeightLimiter:
    """Cooperative limiter driven by Binance used-weight response headers."""

    def __init__(
        self,
        *,
        limit_per_minute: int = 2400,
        reserve_weight: int = 100,
        clock: Clock | None = None,
        shared_budget: SqliteSharedWeightBudget | None = None,
    ) -> None:
        if limit_per_minute <= 0:
            raise ValueError("limit_per_minute must be positive")
        if reserve_weight < 0 or reserve_weight >= limit_per_minute:
            raise ValueError("reserve_weight must be in [0, limit_per_minute)")
        self._limit = limit_per_minute
        self._reserve = reserve_weight
        if shared_budget is not None and (
            shared_budget.limit != limit_per_minute
            or shared_budget.reserve != reserve_weight
        ):
            raise ValueError("shared budget limits do not match limiter limits")
        self._clock = clock or SystemClock()
        self._shared_budget = shared_budget
        self._window_started = self._clock.monotonic()
        self._local_weight = 0
        self._remote_weight = 0
        self._lock = asyncio.Lock()

    @property
    def used_weight(self) -> int:
        return max(self._local_weight, self._remote_weight)

    @property
    def usable_capacity(self) -> int:
        return self._limit - self._reserve

    async def acquire(self, weight: int = 1) -> None:
        if weight <= 0:
            raise ValueError("weight must be positive")
        if weight > self.usable_capacity:
            raise ValueError(
                f"weight {weight} exceeds usable per-minute capacity {self.usable_capacity}"
            )
        while True:
            if self._shared_budget is not None:
                decision = await asyncio.to_thread(
                    self._shared_budget.reserve_weight, weight
                )
                if decision.granted:
                    async with self._lock:
                        self._local_weight += weight
                    return
                await self._clock.sleep(decision.retry_after_seconds)
                continue
            async with self._lock:
                now = self._clock.monotonic()
                elapsed = now - self._window_started
                if elapsed >= 60.0 or elapsed < 0:
                    self._window_started = now
                    self._local_weight = 0
                    self._remote_weight = 0
                    elapsed = 0.0
                projected = self.used_weight + weight
                if projected <= self.usable_capacity:
                    self._local_weight += weight
                    return
                delay = max(0.01, 60.0 - elapsed)
            await self._clock.sleep(delay)

    def observe_headers(self, headers: Mapping[str, str]) -> None:
        candidates: list[int] = []
        for key, value in headers.items():
            lowered = key.lower()
            if lowered.startswith("x-mbx-used-weight-") or lowered == "x-mbx-used-weight":
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed >= 0:
                    candidates.append(parsed)
        if candidates:
            observed = max(candidates)
            self._remote_weight = max(self._remote_weight, observed)
            if self._shared_budget is not None:
                self._shared_budget.observe_remote_weight(observed)


async def run_with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: RetryPolicy | None = None,
    clock: Clock | None = None,
    retry_if: Callable[[Exception], bool] | None = None,
) -> T:
    effective_policy = policy or RetryPolicy()
    effective_clock = clock or SystemClock()
    last_error: Exception | None = None
    for attempt in range(1, effective_policy.max_attempts + 1):
        try:
            return await operation()
        except Exception as exc:
            last_error = exc
            should_retry = (
                retry_if(exc)
                if retry_if is not None
                else isinstance(exc, BinanceTransportError) and exc.retryable
            )
            if not should_retry or attempt >= effective_policy.max_attempts:
                raise
            retry_after = (
                exc.retry_after if isinstance(exc, BinanceTransportError) else None
            )
            await effective_clock.sleep(
                effective_policy.delay_for(attempt, retry_after=retry_after)
            )
    if last_error is None:
        raise RuntimeError("Retry loop exited without an operation result")
    raise last_error
