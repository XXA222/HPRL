from __future__ import annotations

import asyncio
from datetime import timedelta
from decimal import Decimal

import pytest

from freqtrade.hedge.exchange import (
    AiohttpBinanceRestTransport,
    ReadonlyHistoryCursorRepository,
)
from freqtrade.hedge.exchange.base import CalibrationKind, ReadonlyState
from freqtrade.hedge.readonly import (
    BinanceReadonlyRuntime,
    BinanceReadonlyRuntimeConfig,
    ReadonlyRuntimeSnapshot,
    ReadonlyServiceSoakProvider,
    SoakAccountingTotals,
    build_binance_readonly_runtime,
)

from ._helpers import FakeClock, FakeRepository


class FakeConnector:
    async def connect(self, url):
        raise AssertionError(f"unexpected connect: {url}")


class FakeService:
    def __init__(self, snapshot):
        self._snapshot = snapshot
        self.started = 0
        self.stopped = 0

    async def start(self):
        self.started += 1

    async def stop(self):
        self.stopped += 1

    def runtime_snapshot(self):
        return self._snapshot


class SnapshotStatus:
    def __init__(self, state, reason="TEST"):
        self.state = state
        self.reason = reason


class Snapshot:
    def __init__(self, state):
        self.status = SnapshotStatus(state)


def test_runtime_config_normalizes_symbols_and_keeps_operational_intervals():
    config = BinanceReadonlyRuntimeConfig(
        account_id=" acct ",
        managed_symbols=["ethusdt", "BTCUSDT", "ethusdt"],
        api_key=" key ",
        api_secret=" secret ",
        fast_calibration_interval=timedelta(seconds=30),
        full_calibration_interval=timedelta(minutes=10),
        quantity_tolerance=Decimal("0.0001"),
    )

    assert config.account_id == "acct"
    assert config.managed_symbols == ("BTCUSDT", "ETHUSDT")
    assert config.api_key == "key"
    assert config.fast_calibration_interval == timedelta(seconds=30)
    assert config.event_stale_after is None


def test_runtime_factory_wires_one_clock_and_all_primary_components():
    clock = FakeClock()
    repository = FakeRepository()
    config = BinanceReadonlyRuntimeConfig(
        account_id="acct",
        managed_symbols=["BTCUSDT"],
        api_key="key",
        api_secret="secret",
        futures_base_url="http://localhost:18080",
        spot_base_url="http://localhost:18081",
        websocket_base_url="ws://localhost:18082/ws",
    )

    runtime = build_binance_readonly_runtime(
        config=config,
        repository=repository,
        clock=clock,
        websocket_connector=FakeConnector(),
    )

    assert isinstance(runtime, BinanceReadonlyRuntime)
    assert isinstance(runtime.transport, AiohttpBinanceRestTransport)
    assert runtime.clock is clock
    assert runtime.client.clock is clock
    assert runtime.stream._clock is clock
    assert runtime.calibration.clock is clock
    assert runtime.scheduler.clock is clock
    assert runtime.service.clock is clock
    assert runtime.client.managed_symbols == ("BTCUSDT",)


@pytest.mark.asyncio
async def test_runtime_run_until_owns_start_and_stop_lifecycle():
    stop_event = asyncio.Event()
    stop_event.set()
    runtime = object.__new__(BinanceReadonlyRuntime)
    runtime.service = FakeService(Snapshot(ReadonlyState.READY))

    await runtime.run_until(stop_event)

    assert runtime.service.started == 1
    assert runtime.service.stopped == 1


@pytest.mark.asyncio
async def test_wait_until_ready_returns_snapshot_and_fails_on_halt():
    runtime = object.__new__(BinanceReadonlyRuntime)
    runtime.clock = FakeClock()
    ready = Snapshot(ReadonlyState.READY)
    runtime.service = FakeService(ready)

    assert await runtime.wait_until_ready(timeout_seconds=1) is ready

    runtime.service = FakeService(Snapshot(ReadonlyState.HALT))
    with pytest.raises(RuntimeError, match="HALT"):
        await runtime.wait_until_ready(timeout_seconds=1)


def test_new_runtime_types_are_publicly_importable():
    assert BinanceReadonlyRuntimeConfig is not None
    assert ReadonlyRuntimeSnapshot is not None
    assert ReadonlyServiceSoakProvider is not None
    assert SoakAccountingTotals is not None
    assert ReadonlyHistoryCursorRepository is not None


@pytest.mark.asyncio
async def test_runtime_proxies_manual_calibration_and_reconnect():
    class Service:
        async def calibrate_now(self, kind):
            return ("calibrated", kind)

    class Stream:
        def __init__(self):
            self.reconnects = 0

        async def request_reconnect(self):
            self.reconnects += 1

    runtime = object.__new__(BinanceReadonlyRuntime)
    runtime.service = Service()
    runtime.stream = Stream()

    result = await runtime.calibrate_now(CalibrationKind.FAST)
    await runtime.request_reconnect()

    assert result == ("calibrated", CalibrationKind.FAST)
    assert runtime.stream.reconnects == 1


def test_runtime_factory_applies_clock_skew_and_target_leverage():
    clock = FakeClock()
    config = BinanceReadonlyRuntimeConfig(
        account_id="acct",
        managed_symbols=["BTCUSDT"],
        api_key="key",
        api_secret="secret",
        max_clock_skew_ms=4321,
        target_leverage=7,
    )

    runtime = build_binance_readonly_runtime(
        config=config,
        repository=FakeRepository(),
        clock=clock,
        websocket_connector=FakeConnector(),
    )

    assert runtime.client.clock_sync.status.max_abs_skew_ms == 4321
    assert runtime.service.target_leverage == 7


def test_service_target_leverage_comparison_is_exact_when_configured():
    from freqtrade.hedge.readonly.service import BinanceReadonlyService

    service = object.__new__(BinanceReadonlyService)
    service.target_leverage = 3

    assert service._leverage_configuration_valid({"BTCUSDT:LONG": 3})
    assert not service._leverage_configuration_valid({"BTCUSDT:LONG": 5})

    service.target_leverage = None
    assert service._leverage_configuration_valid({"BTCUSDT:LONG": 5})


def test_runtime_factory_wires_explicit_rest_and_websocket_proxies():
    clock = FakeClock()
    config = BinanceReadonlyRuntimeConfig(
        account_id="acct",
        managed_symbols=["BTCUSDT"],
        api_key="key",
        api_secret="secret",
        rest_proxy_url="http://127.0.0.1:7897",
        websocket_proxy_url="http://127.0.0.1:7898",
    )

    runtime = build_binance_readonly_runtime(
        config=config,
        repository=FakeRepository(),
        clock=clock,
    )

    assert runtime.transport._proxy_url == "http://127.0.0.1:7897"
    assert runtime.stream._connector._proxy_url == "http://127.0.0.1:7898"
