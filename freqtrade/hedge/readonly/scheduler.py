from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta

from freqtrade.hedge.exchange.base import (
    AsyncCallback,
    CalibrationKind,
    CalibrationResult,
    Clock,
    SystemClock,
    maybe_await,
)
from freqtrade.hedge.readonly.calibration import ReadonlyCalibration


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CalibrationSchedule:
    fast_interval: timedelta = timedelta(minutes=1)
    full_interval: timedelta = timedelta(minutes=15)
    error_retry_interval: timedelta = timedelta(seconds=30)

    def __post_init__(self) -> None:
        if self.fast_interval.total_seconds() <= 0:
            raise ValueError("fast_interval must be positive")
        if self.full_interval.total_seconds() <= 0:
            raise ValueError("full_interval must be positive")
        if self.error_retry_interval.total_seconds() <= 0:
            raise ValueError("error_retry_interval must be positive")
        if self.full_interval < self.fast_interval:
            raise ValueError("full_interval must be >= fast_interval")


class ReconciliationScheduler:
    def __init__(
        self,
        *,
        calibration: ReadonlyCalibration,
        schedule: CalibrationSchedule | None = None,
        clock: Clock | None = None,
        on_result: AsyncCallback | None = None,
        on_error: AsyncCallback | None = None,
    ) -> None:
        self.calibration = calibration
        self.schedule = schedule or CalibrationSchedule()
        self.clock = clock or SystemClock()
        self._next_fast = self.clock.now()
        self._next_full = self.clock.now()
        self._lock = asyncio.Lock()
        self._on_result = on_result
        self._on_error = on_error

    def reset_after_bootstrap(self) -> None:
        now = self.clock.now()
        self._next_fast = now + self.schedule.fast_interval
        self._next_full = now + self.schedule.full_interval

    def set_callbacks(
        self,
        *,
        on_result: AsyncCallback | None = None,
        on_error: AsyncCallback | None = None,
    ) -> None:
        self._on_result = on_result
        self._on_error = on_error

    async def run_due(self) -> tuple[CalibrationResult, ...]:
        callback_results: list[CalibrationResult] = []
        async with self._lock:
            now = self.clock.now()
            if now >= self._next_full:
                kind = CalibrationKind.FULL
                # Move deadlines before I/O so a failure/callback error cannot
                # produce a tight retry loop against Binance.
                self._next_full = now + self.schedule.full_interval
                self._next_fast = now + self.schedule.fast_interval
            elif now >= self._next_fast:
                kind = CalibrationKind.FAST
                self._next_fast = now + self.schedule.fast_interval
            else:
                return ()

            try:
                result = await self.calibration.run(kind)
            except asyncio.CancelledError:
                raise
            except Exception:
                retry_at = self.clock.now() + self.schedule.error_retry_interval
                if kind is CalibrationKind.FULL:
                    self._next_full = min(self._next_full, retry_at)
                    self._next_fast = min(self._next_fast, retry_at)
                else:
                    self._next_fast = min(self._next_fast, retry_at)
                raise
            callback_results.append(result)

        # A reporting callback must not cause the same REST calibration to run
        # again. Scheduler.run reports callback errors through on_error.
        if self._on_result is not None:
            await maybe_await(self._on_result(callback_results[0]))
        return tuple(callback_results)

    async def _report_error(self, exc: Exception) -> None:
        if self._on_error is None:
            return
        try:
            await maybe_await(self._on_error(exc))
        except Exception:
            logger.exception("Reconciliation scheduler error callback failed")

    async def _wait_until_due(self, stop_event: asyncio.Event) -> None:
        now = self.clock.now()
        next_at = min(self._next_fast, self._next_full)
        delay = max(0.05, (next_at - now).total_seconds())
        sleeper = asyncio.create_task(self.clock.sleep(delay))
        stopper = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            {sleeper, stopper}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if sleeper in done:
            sleeper.result()

    async def run(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self.run_due()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._report_error(exc)
            await self._wait_until_due(stop_event)
