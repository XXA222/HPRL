import asyncio
from types import SimpleNamespace

import pytest

from freqtrade.hedge.exchange.base import (
    CalibrationKind,
    CalibrationResult,
    ReadonlyState,
    StreamHealth,
)
from freqtrade.hedge.readonly.service import BinanceReadonlyService

from ._helpers import FakeClock


class Transport:
    def __init__(self, *, failure=None):
        self.failure = failure
        self.closed = 0

    async def close(self):
        self.closed += 1
        if self.failure is not None:
            raise self.failure


class Client:
    def __init__(self, *, transport=None):
        self.synced = 0
        self.preflight = 0
        self.transport = transport

    async def synchronize_clock(self):
        self.synced += 1

    async def preflight_permissions(self, policy):
        self.preflight += 1


class Calibration:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.last_bundle = SimpleNamespace(positions=(), open_orders=())

    async def run(self, kind):
        self.calls.append(kind)
        return CalibrationResult(
            f"run-{len(self.calls)}",
            kind,
            self.clock.now(),
            self.clock.now(),
            0,
            0,
            0,
            0,
            (),
            (),
            True,
            "CONSISTENT",
        )


class Stream:
    def __init__(self, clock):
        self.clock = clock
        self._health = StreamHealth(
            True,
            clock.now(),
            clock.now(),
            None,
            0,
            0,
            0,
            0,
        )
        self.callbacks = {}
        self.seeded = 0

    @property
    def health(self):
        return self._health

    def set_callbacks(self, **kwargs):
        self.callbacks = kwargs

    def seed_from_rest(self, positions, orders):
        self.seeded += 1

    def mark_calibrated(self, at=None):
        self._health = StreamHealth(
            True,
            self.clock.now(),
            self.clock.now(),
            at or self.clock.now(),
            0,
            0,
            0,
            0,
        )

    async def run(self):
        return None

    async def stop(self):
        return None


def build_service(clock):
    client = Client()
    calibration = Calibration(clock)
    stream = Stream(clock)
    service = BinanceReadonlyService(
        client=client,
        calibration=calibration,
        stream=stream,
        clock=clock,
    )
    return service, calibration, stream


def test_disconnect_cannot_return_ready_without_rest_recalibration():
    clock = FakeClock()
    service, calibration, stream = build_service(clock)

    asyncio.run(service.preflight_and_bootstrap())
    assert service.status.state is ReadonlyState.RECOVERING
    assert calibration.calls == [CalibrationKind.STARTUP]

    asyncio.run(service.on_stream_disconnected())
    assert service.status.state is ReadonlyState.RECOVERING

    asyncio.run(service.on_stream_connected(2))
    assert calibration.calls[-1] is CalibrationKind.RECONNECT
    assert service.status.state is ReadonlyState.READY
    assert stream.seeded == 2


def test_integrity_fault_waits_for_physical_reconnect_before_rest_recovery():
    clock = FakeClock()
    service, calibration, _stream = build_service(clock)

    asyncio.run(service.preflight_and_bootstrap())
    before = len(calibration.calls)
    asyncio.run(
        service.on_integrity_fault(
            "ORDER_CUMULATIVE_FILL_GAP",
            {},
        )
    )
    assert len(calibration.calls) == before
    assert service.status.state is ReadonlyState.RECOVERING

    asyncio.run(service.on_stream_connected(3))
    assert calibration.calls[-1] is CalibrationKind.RECONNECT
    assert service.status.state is ReadonlyState.READY


def test_scheduled_calibration_cannot_clear_reconnect_barrier():
    clock = FakeClock()
    service, _calibration, _stream = build_service(clock)

    asyncio.run(service.preflight_and_bootstrap())
    asyncio.run(service.on_stream_disconnected())
    result = CalibrationResult(
        "scheduled",
        CalibrationKind.FULL,
        clock.now(),
        clock.now(),
        0,
        0,
        0,
        0,
        (),
        (),
        True,
        "CONSISTENT",
    )
    asyncio.run(service.on_scheduled_calibration(result))
    assert service.status.state is ReadonlyState.RECOVERING


class FailingStopStream(Stream):
    async def stop(self):
        raise RuntimeError("stream stop failed")


class FailingPreflightClient(Client):
    async def synchronize_clock(self):
        raise ValueError("preflight root cause")


def test_stop_completes_cleanup_when_stream_stop_fails():
    clock = FakeClock()
    transport = Transport()
    client = Client(transport=transport)
    calibration = Calibration(clock)
    stream = FailingStopStream(clock)
    service = BinanceReadonlyService(
        client=client,
        calibration=calibration,
        stream=stream,
        clock=clock,
    )
    service._started = True

    with pytest.raises(RuntimeError, match="stream stop failed"):
        asyncio.run(service.stop())

    assert transport.closed == 1
    assert service.status.state is ReadonlyState.STOPPED
    assert not service._started
    assert service._tasks == []
    assert service._supervisor_tasks == set()


def test_start_preserves_preflight_error_when_transport_close_also_fails():
    clock = FakeClock()
    transport = Transport(failure=RuntimeError("transport close failed"))
    client = FailingPreflightClient(transport=transport)
    calibration = Calibration(clock)
    stream = Stream(clock)
    service = BinanceReadonlyService(
        client=client,
        calibration=calibration,
        stream=stream,
        clock=clock,
    )

    with pytest.raises(ValueError, match="preflight root cause"):
        asyncio.run(service.start())

    assert transport.closed == 1
    assert service.status.state is ReadonlyState.HALT
    assert not service._started


def test_reconnect_drift_is_verified_before_ready():
    class DriftCalibration(Calibration):
        async def run(self, kind):
            self.calls.append(kind)
            consistent = len(self.calls) > 1
            return CalibrationResult(
                f"run-{len(self.calls)}",
                kind,
                self.clock.now(),
                self.clock.now(),
                0,
                0,
                0,
                0 if consistent else 1,
                (),
                (),
                consistent,
                "CONSISTENT" if consistent else "RECONCILIATION_DRIFT",
            )

    clock = FakeClock()
    client = Client()
    calibration = DriftCalibration(clock)
    stream = Stream(clock)
    service = BinanceReadonlyService(
        client=client,
        calibration=calibration,
        stream=stream,
        clock=clock,
    )

    asyncio.run(service.on_stream_connected(2))

    assert calibration.calls == [
        CalibrationKind.RECONNECT,
        CalibrationKind.FAST,
    ]
    assert service.status.state is ReadonlyState.READY


def test_stale_connected_stream_requests_reconnect_once():
    from datetime import timedelta

    from freqtrade.hedge.readonly.freshness import (
        FreshnessPolicy,
        UserStreamFreshness,
    )

    class ReconnectableStream(Stream):
        def __init__(self, clock):
            super().__init__(clock)
            self.reconnect_requests = 0

        async def request_reconnect(self):
            self.reconnect_requests += 1

    clock = FakeClock()
    stream = ReconnectableStream(clock)
    service = BinanceReadonlyService(
        client=Client(),
        calibration=Calibration(clock),
        stream=stream,
        freshness=UserStreamFreshness(
            FreshnessPolicy(event_stale_after=timedelta(seconds=10))
        ),
        clock=clock,
    )
    stream.mark_calibrated(clock.now())
    clock.advance(11)
    assessment = service.assess_freshness()

    asyncio.run(service._handle_freshness_failure(assessment))
    asyncio.run(service._handle_freshness_failure(assessment))

    assert stream.reconnect_requests == 1
    assert service.status.state is ReadonlyState.RECOVERING


def test_runtime_snapshot_exposes_stream_and_latest_calibrations():
    clock = FakeClock()
    service, calibration, stream = build_service(clock)

    asyncio.run(service.preflight_and_bootstrap())
    asyncio.run(service.on_stream_connected(1))
    snapshot = service.runtime_snapshot()

    assert snapshot.status.state is ReadonlyState.READY
    assert snapshot.stream_health is stream.health
    assert snapshot.last_reconnect_calibration is not None
    assert snapshot.last_reconnect_calibration.kind is CalibrationKind.RECONNECT
    assert calibration.calls[-1] is CalibrationKind.RECONNECT


def test_freshness_recovery_failure_degrades_without_killing_loop():
    from datetime import timedelta

    from freqtrade.hedge.readonly.freshness import (
        FreshnessPolicy,
        UserStreamFreshness,
    )

    class FailingReconnectStream(Stream):
        async def request_reconnect(self):
            raise RuntimeError("temporary reconnect failure")

    clock = FakeClock()
    stream = FailingReconnectStream(clock)
    service = BinanceReadonlyService(
        client=Client(),
        calibration=Calibration(clock),
        stream=stream,
        freshness=UserStreamFreshness(
            FreshnessPolicy(event_stale_after=timedelta(seconds=10))
        ),
        clock=clock,
    )
    stream.mark_calibrated(clock.now())
    clock.advance(11)

    asyncio.run(service._freshness_iteration())

    assert service.status.state is ReadonlyState.DEGRADED
    assert service.status.reason == "FRESHNESS_RECOVERY_FAILED:RuntimeError"
    assert not service._freshness_action_pending


def test_manual_full_calibration_updates_runtime_state_and_result():
    clock = FakeClock()
    service, calibration, _stream = build_service(clock)

    result = asyncio.run(service.calibrate_now(CalibrationKind.FULL))

    assert result.kind is CalibrationKind.FULL
    assert calibration.calls == [CalibrationKind.FULL]
    assert service.status.state is ReadonlyState.READY
    assert service.latest_calibration(CalibrationKind.FULL) is result


def test_manual_calibration_rejects_lifecycle_only_kinds():
    clock = FakeClock()
    service, _calibration, _stream = build_service(clock)

    with pytest.raises(ValueError, match="FAST or FULL"):
        asyncio.run(service.calibrate_now(CalibrationKind.RECONNECT))


def test_newer_reconciliation_drift_is_not_cleared_by_stream_event():
    clock = FakeClock()
    service, _calibration, _stream = build_service(clock)
    service.drift_verification_attempts = 0

    asyncio.run(service.on_stream_connected(1))
    assert service.status.state is ReadonlyState.READY

    clock.advance(1)
    drift = CalibrationResult(
        "newer-full-drift",
        CalibrationKind.FULL,
        clock.now(),
        clock.now(),
        0,
        0,
        0,
        1,
        (),
        (),
        False,
        "RECONCILIATION_DRIFT",
    )
    asyncio.run(service.on_scheduled_calibration(drift))
    assert service.status.state is ReadonlyState.DEGRADED

    asyncio.run(service.on_stream_event(clock.now()))
    assert service.status.state is ReadonlyState.DEGRADED
    assert service._latest_health_calibration() is drift


def test_account_view_exposes_latest_calibration_facts():
    from ._helpers import order, position

    clock = FakeClock()
    service, calibration, _stream = build_service(clock)
    position_fact = position(clock)
    order_fact = order(clock)
    calibration.last_bundle = SimpleNamespace(
        positions=(position_fact,),
        open_orders=(order_fact,),
        balances=(),
        account_snapshot=None,
        configuration=None,
        collection_completed_at=clock.now(),
    )

    view = service.account_view()

    assert view.account_id == "unknown"
    assert view.positions == (position_fact,)
    assert view.active_orders == (order_fact,)
    assert view.revision == 0
    assert service.runtime_snapshot().account_view == view
