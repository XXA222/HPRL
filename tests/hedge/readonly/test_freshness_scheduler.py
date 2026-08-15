import asyncio
from datetime import datetime, timedelta

import pytest

from freqtrade.hedge.exchange.base import (
    CalibrationKind,
    CalibrationResult,
    ReadonlyState,
    StreamHealth,
)
from freqtrade.hedge.readonly.freshness import (
    FreshnessPolicy,
    UserStreamFreshness,
)
from freqtrade.hedge.readonly.scheduler import (
    CalibrationSchedule,
    ReconciliationScheduler,
)

from ._helpers import FakeClock


def health(
    clock,
    *,
    connected=True,
    connected_at=True,
    event_at=True,
    calibration_at=True,
    generation=0,
):
    return StreamHealth(
        connected,
        clock.now() if connected_at else None,
        clock.now() if event_at else None,
        clock.now() if calibration_at else None,
        generation,
        0,
        0,
        0,
    )


def test_disconnected_and_stale_stream_are_not_ready():
    clock = FakeClock()
    checker = UserStreamFreshness(
        FreshnessPolicy(event_stale_after=timedelta(seconds=10))
    )
    disconnected = StreamHealth(False, None, None, None, 0, 0, 0, 0)
    assert checker.assess(
        disconnected,
        now=clock.now(),
    ).state is ReadonlyState.RECOVERING

    uncalibrated = health(clock, calibration_at=False)
    assert checker.assess(
        uncalibrated,
        now=clock.now(),
    ).state is ReadonlyState.CALIBRATING

    current = health(clock)
    clock.advance(11)
    assert checker.assess(current, now=clock.now()).reason == "USER_STREAM_STALE"


class Calibration:
    def __init__(self, clock):
        self.clock = clock
        self.kinds = []

    async def run(self, kind):
        self.kinds.append(kind)
        return CalibrationResult(
            str(len(self.kinds)),
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


def test_scheduler_runs_full_then_fast_with_fake_clock():
    clock = FakeClock()
    calibration = Calibration(clock)
    scheduler = ReconciliationScheduler(
        calibration=calibration,
        schedule=CalibrationSchedule(
            timedelta(seconds=10),
            timedelta(seconds=30),
        ),
        clock=clock,
    )
    asyncio.run(scheduler.run_due())
    assert calibration.kinds == [CalibrationKind.FULL]

    clock.advance(11)
    asyncio.run(scheduler.run_due())
    assert calibration.kinds[-1] is CalibrationKind.FAST

    clock.advance(20)
    asyncio.run(scheduler.run_due())
    assert calibration.kinds[-1] is CalibrationKind.FULL


def test_connected_stream_with_no_business_events_eventually_becomes_stale():
    clock = FakeClock()
    checker = UserStreamFreshness(
        FreshnessPolicy(event_stale_after=timedelta(seconds=10))
    )
    current = health(clock, event_at=False)
    clock.advance(11)
    assessment = checker.assess(current, now=clock.now())
    assert assessment.reason == "USER_STREAM_STALE"
    assert not assessment.fresh


def test_scheduler_failure_uses_backoff_instead_of_tight_retry_loop():
    class FailingCalibration:
        def __init__(self):
            self.calls = 0

        async def run(self, kind):
            self.calls += 1
            raise RuntimeError("REST unavailable")

    clock = FakeClock()
    calibration = FailingCalibration()
    scheduler = ReconciliationScheduler(
        calibration=calibration,
        schedule=CalibrationSchedule(
            timedelta(seconds=10),
            timedelta(seconds=30),
            timedelta(seconds=5),
        ),
        clock=clock,
    )
    with pytest.raises(RuntimeError):
        asyncio.run(scheduler.run_due())
    assert calibration.calls == 1

    assert asyncio.run(scheduler.run_due()) == ()
    assert calibration.calls == 1
    clock.advance(4.9)
    assert asyncio.run(scheduler.run_due()) == ()
    clock.advance(0.1)
    with pytest.raises(RuntimeError):
        asyncio.run(scheduler.run_due())
    assert calibration.calls == 2


def test_default_freshness_does_not_treat_a_quiet_connected_account_as_stale():
    clock = FakeClock()
    checker = UserStreamFreshness()
    current = health(clock, event_at=False)
    clock.advance(10 * 60)
    assert checker.assess(current, now=clock.now()).fresh


def test_reconnect_requires_calibration_after_current_connection():
    clock = FakeClock()
    checker = UserStreamFreshness()
    old_calibration = clock.now()
    clock.advance(1)
    connected = clock.now()
    current = StreamHealth(
        True,
        connected,
        None,
        old_calibration,
        1,
        0,
        0,
        0,
    )
    assessment = checker.assess(current, now=clock.now())
    assert not assessment.fresh
    assert assessment.reason == "REST_CALIBRATION_PREDATES_CONNECTION"


def test_freshness_rejects_naive_timestamps_even_when_both_are_naive():
    checker = UserStreamFreshness()
    naive = datetime(2026, 7, 26, 12, 0, 0)
    current = StreamHealth(True, naive, naive, naive, 0, 0, 0, 0)

    with pytest.raises(ValueError, match="timezone-aware"):
        checker.assess(current, now=naive)
