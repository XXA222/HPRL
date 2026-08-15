from __future__ import annotations

import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from statistics import median
from typing import Any

from .base import Clock, SystemClock
from .rate_limit import BinanceDataError


@dataclass(frozen=True, slots=True)
class ClockSample:
    started_monotonic: float
    completed_monotonic: float
    server_time_ms: int
    midpoint_local_ms: float
    round_trip_ms: float
    offset_ms: float


@dataclass(frozen=True, slots=True)
class ClockSyncStatus:
    synchronized: bool
    offset_ms: float
    round_trip_ms: float
    sample_count: int
    max_abs_skew_ms: float


class ClockSynchronizer:
    """NTP-style midpoint clock calibration for Binance signed timestamps."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        max_abs_skew_ms: float = 1000.0,
        sample_count: int = 5,
    ) -> None:
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        if not math.isfinite(max_abs_skew_ms) or max_abs_skew_ms <= 0:
            raise ValueError("max_abs_skew_ms must be finite and positive")
        self._clock = clock or SystemClock()
        self._max_abs_skew_ms = float(max_abs_skew_ms)
        self._sample_count = int(sample_count)
        self._offset_ms = 0.0
        self._rtt_ms = 0.0
        self._synchronized = False

    @property
    def status(self) -> ClockSyncStatus:
        return ClockSyncStatus(
            synchronized=self._synchronized,
            offset_ms=self._offset_ms,
            round_trip_ms=self._rtt_ms,
            sample_count=self._sample_count if self._synchronized else 0,
            max_abs_skew_ms=self._max_abs_skew_ms,
        )

    async def sync(
        self,
        fetch_server_time: Callable[[], Awaitable[int | dict[str, Any]]],
    ) -> ClockSyncStatus:
        samples: list[ClockSample] = []
        for _ in range(self._sample_count):
            started_mono = self._clock.monotonic()
            started_wall_ms = self._clock.now().timestamp() * 1000.0
            raw = await fetch_server_time()
            completed_mono = self._clock.monotonic()
            completed_wall_ms = self._clock.now().timestamp() * 1000.0
            if isinstance(raw, dict):
                raw = raw.get("serverTime")
            try:
                numeric_server_time = float(raw)
            except (TypeError, ValueError) as exc:
                raise BinanceDataError("Invalid Binance server time response") from exc
            if (
                not math.isfinite(numeric_server_time)
                or not numeric_server_time.is_integer()
                or numeric_server_time < 0
            ):
                raise BinanceDataError("Invalid Binance server time response")
            server_time_ms = int(numeric_server_time)
            elapsed = completed_mono - started_mono
            if elapsed < 0:
                raise BinanceDataError("Monotonic clock regressed during synchronization")
            rtt_ms = elapsed * 1000.0
            midpoint_local_ms = (started_wall_ms + completed_wall_ms) / 2.0
            samples.append(
                ClockSample(
                    started_monotonic=started_mono,
                    completed_monotonic=completed_mono,
                    server_time_ms=server_time_ms,
                    midpoint_local_ms=midpoint_local_ms,
                    round_trip_ms=rtt_ms,
                    offset_ms=server_time_ms - midpoint_local_ms,
                )
            )
        best = sorted(samples, key=lambda item: item.round_trip_ms)[: min(3, len(samples))]
        offset = float(median([item.offset_ms for item in best]))
        rtt = float(median([item.round_trip_ms for item in best]))
        self._offset_ms = offset
        self._rtt_ms = rtt
        self._synchronized = True
        if abs(offset) > self._max_abs_skew_ms:
            self._synchronized = False
            raise BinanceDataError(
                f"Clock skew {offset:.3f}ms exceeds limit {self._max_abs_skew_ms:.3f}ms"
            )
        return self.status

    def timestamp_ms(self) -> int:
        if not self._synchronized:
            raise BinanceDataError("Clock is not synchronized")
        return int(self._clock.now().timestamp() * 1000.0 + self._offset_ms)
