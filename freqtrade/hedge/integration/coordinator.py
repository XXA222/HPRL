"""Central composition root for Binance read-only facts and the control-plane runtime."""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import Future
from threading import Event, RLock, Thread
from typing import Any, Mapping

from freqtrade.hedge.readonly import (
    BinanceReadonlyRuntime,
    build_binance_readonly_runtime_from_freqtrade_config,
)
from freqtrade.hedge.runtime import HedgeProjectionSource, HedgeRuntime

from .projection import build_central_projection
from .repository import InMemoryReadonlyRepository

logger = logging.getLogger(__name__)


class AsyncLoopThread:
    def __init__(self, *, name: str = "freqtrade-hedge-async") -> None:
        self._name = name
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._ready = Event()
        self._lock = RLock()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready.clear()
            self._thread = Thread(target=self._run, name=self._name, daemon=True)
            self._thread.start()
        if not self._ready.wait(timeout=10):
            raise TimeoutError("hedge async event loop did not start")

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            self._loop = None

    def submit(self, coroutine: Any) -> Future[Any]:
        self.start()
        loop = self._loop
        if loop is None:
            raise RuntimeError("hedge async event loop is not available")
        return asyncio.run_coroutine_threadsafe(coroutine, loop)

    def run(self, coroutine: Any, *, timeout: float = 120.0) -> Any:
        return self.submit(coroutine).result(timeout=timeout)

    def stop(self) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            if loop is not None:
                loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=10)
        with self._lock:
            self._thread = None


class HedgeRuntimeCoordinator:
    """Start direction two, project facts, and keep the central Runtime current."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        central_runtime: HedgeRuntime,
        repository: InMemoryReadonlyRepository | None = None,
        readonly_runtime: BinanceReadonlyRuntime | None = None,
        websocket_connector: Any = None,
        http_session: Any = None,
    ) -> None:
        self.config = config
        self.central_runtime = central_runtime
        self.repository = repository or InMemoryReadonlyRepository()
        self._runner = AsyncLoopThread()
        self.readonly_runtime = readonly_runtime or build_binance_readonly_runtime_from_freqtrade_config(
            config=config,
            repository=self.repository,
            websocket_connector=websocket_connector,
            http_session=http_session,
        )
        self._started = False
        self._lock = RLock()

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    def start(self, *, timeout: float = 120.0) -> None:
        with self._lock:
            if self._started:
                return
        self._runner.run(self.readonly_runtime.start(), timeout=timeout)
        with self._lock:
            self._started = True
        self.refresh()

    def refresh(self) -> None:
        if not self.started:
            self.central_runtime.halt(
                "READONLY_COORDINATOR_NOT_STARTED",
                source=HedgeProjectionSource.EXCHANGE,
            )
            return
        try:
            snapshot = self.readonly_runtime.snapshot()
            account_view = self.readonly_runtime.account_view()
            projection = build_central_projection(account_view, snapshot)
            self.central_runtime.publish(
                source=HedgeProjectionSource.EXCHANGE,
                positions=projection.positions,
                risk=projection.risk,
                reconciliation_status=projection.reconciliation_status,
                reconciliation_at=projection.reconciliation_at,
                reconciliation_details=projection.reconciliation_details,
                stream_state=projection.stream_state,
                stream_last_event_at=projection.stream_last_event_at,
                stream_reconnect_count=projection.stream_reconnect_count,
                checks=projection.checks,
                reasons=projection.reasons,
                source_version=projection.source_version,
                source_event_time=projection.source_event_time,
                stale=projection.stale,
            )
        except Exception as exc:
            logger.exception("Failed to refresh integrated hedge runtime")
            self.central_runtime.halt(
                f"READONLY_PROJECTION_FAILED:{type(exc).__name__}",
                source=HedgeProjectionSource.EXCHANGE,
            )

    def calibrate_now(self, *, timeout: float = 120.0) -> None:
        self._runner.run(self.readonly_runtime.calibrate_now(), timeout=timeout)
        self.refresh()

    def request_reconnect(self, *, timeout: float = 30.0) -> None:
        self._runner.run(self.readonly_runtime.request_reconnect(), timeout=timeout)

    def stop(self, *, timeout: float = 60.0) -> None:
        with self._lock:
            was_started = self._started
            self._started = False
        if was_started:
            try:
                self._runner.run(self.readonly_runtime.stop(), timeout=timeout)
            finally:
                self._runner.stop()
        else:
            self._runner.stop()

    close = stop
