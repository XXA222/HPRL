"""Background lifecycle service for the database single-writer lease."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from freqtrade.hedge.concurrency.database_lease import LeaseLost, LeaseUnavailable
from freqtrade.hedge.concurrency.single_writer import SingleWriterGuard, SingleWriterStatus


class LeaseRunnerState(str, Enum):
    STOPPED = "STOPPED"
    ACQUIRING = "ACQUIRING"
    ACTIVE = "ACTIVE"
    LOST = "LOST"


@dataclass(frozen=True, slots=True)
class LeaseRunnerStatus:
    state: LeaseRunnerState
    writer: SingleWriterStatus
    renewal_count: int
    acquisition_count: int
    last_error: str | None = None

    @property
    def active(self) -> bool:
        return self.state is LeaseRunnerState.ACTIVE and self.writer.valid

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "active": self.active,
            "renewal_count": self.renewal_count,
            "acquisition_count": self.acquisition_count,
            "last_error": self.last_error,
            "writer": self.writer.as_dict(),
        }


class SingleWriterLeaseRunner:
    """Acquire, renew and reacquire a :class:`SingleWriterGuard` lease.

    ``run_once`` provides a deterministic integration point for event loops and
    tests. ``start`` adds a lightweight daemon thread for the normal runtime.
    """

    def __init__(
        self,
        guard: SingleWriterGuard,
        *,
        interval_seconds: float | None = None,
        on_active: Callable[[LeaseRunnerStatus], None] | None = None,
        on_lost: Callable[[LeaseRunnerStatus], None] | None = None,
    ) -> None:
        interval = guard.ttl_ms / 3000 if interval_seconds is None else interval_seconds
        if isinstance(interval, bool):
            raise ValueError("interval_seconds must be a positive finite number.")
        interval = float(interval)
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("interval_seconds must be a positive finite number.")
        if interval * 1000 >= guard.ttl_ms:
            raise ValueError("interval_seconds must be shorter than the lease TTL.")
        self._guard = guard
        self._interval = interval
        self._on_active = on_active
        self._on_lost = on_lost
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = LeaseRunnerState.STOPPED
        self._renewal_count = 0
        self._acquisition_count = 0
        self._last_error: str | None = None

    @property
    def guard(self) -> SingleWriterGuard:
        return self._guard

    @property
    def interval_seconds(self) -> float:
        return self._interval

    @property
    def running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def status(self) -> LeaseRunnerStatus:
        with self._lock:
            state = self._state
            renewal_count = self._renewal_count
            acquisition_count = self._acquisition_count
            last_error = self._last_error
        return LeaseRunnerStatus(
            state,
            self._guard.status(),
            renewal_count,
            acquisition_count,
            last_error,
        )

    def _notify(self, callback: Callable[[LeaseRunnerStatus], None] | None) -> None:
        if callback is None:
            return
        try:
            callback(self.status())
        except Exception:
            # Lifecycle callbacks are observability hooks; they must not stop
            # lease renewal or change writer ownership.
            return

    def run_once(self) -> LeaseRunnerStatus:
        """Acquire when absent/lost, otherwise renew the current lease."""

        previous = self.status()
        previous_active = previous.active
        with self._lock:
            self._state = LeaseRunnerState.ACQUIRING
        try:
            if self._guard.lease is None:
                self._guard.acquire()
                with self._lock:
                    self._acquisition_count += 1
            else:
                self._guard.renew()
                with self._lock:
                    self._renewal_count += 1
        except (LeaseLost, LeaseUnavailable) as exc:
            with self._lock:
                self._state = LeaseRunnerState.LOST
                self._last_error = str(exc)
            current = self.status()
            if previous.state is not LeaseRunnerState.LOST:
                self._notify(self._on_lost)
            return current
        except Exception as exc:
            with self._lock:
                self._state = LeaseRunnerState.LOST
                self._last_error = f"{type(exc).__name__}: {exc}"
            if previous.state is not LeaseRunnerState.LOST:
                self._notify(self._on_lost)
            return self.status()

        with self._lock:
            self._state = LeaseRunnerState.ACTIVE
            self._last_error = None
        current = self.status()
        if not previous_active:
            self._notify(self._on_active)
        return current

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval):
            self.run_once()

    def start(self, *, require_initial_acquire: bool = True) -> LeaseRunnerStatus:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.status()
            self._stop_event.clear()
        initial = self.run_once()
        if require_initial_acquire and not initial.active:
            return initial
        with self._lock:
            self._thread = threading.Thread(
                target=self._run,
                name=f"single-writer:{self._guard.lease_name}",
                daemon=True,
            )
            self._thread.start()
        return self.status()

    def stop(self, *, release: bool = True, join_timeout_seconds: float = 5.0) -> LeaseRunnerStatus:
        if isinstance(join_timeout_seconds, bool):
            raise ValueError("join_timeout_seconds must be a nonnegative finite number.")
        timeout = float(join_timeout_seconds)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("join_timeout_seconds must be a nonnegative finite number.")
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                with self._lock:
                    self._state = LeaseRunnerState.LOST
                    self._last_error = "LEASE_RUNNER_STOP_TIMEOUT"
                return self.status()
        if release:
            self._guard.release()
        with self._lock:
            self._thread = None
            self._state = LeaseRunnerState.STOPPED
            self._last_error = None
        return self.status()

    def __enter__(self) -> "SingleWriterLeaseRunner":
        status = self.start(require_initial_acquire=True)
        if not status.active:
            raise LeaseUnavailable(status.last_error or "Single-writer lease unavailable.")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop(release=True)
