import asyncio
from datetime import UTC, datetime

import pytest

from freqtrade.hedge.exchange.clock_sync import ClockSynchronizer
from freqtrade.hedge.exchange.rate_limit import (
    AdaptiveWeightLimiter,
    BinanceDataError,
    BinanceRateLimitError,
    RetryPolicy,
    parse_retry_after,
    run_with_retry,
)

from ._helpers import FakeClock


def test_clock_sync_midpoint_and_skew_guard():
    clock = FakeClock(datetime.fromtimestamp(1, tz=UTC))

    async def fetch():
        return 1100

    sync = ClockSynchronizer(
        clock=clock,
        sample_count=1,
        max_abs_skew_ms=200,
    )
    status = asyncio.run(sync.sync(fetch))
    assert round(status.offset_ms) == 100
    assert sync.timestamp_ms() == 1100

    bad = ClockSynchronizer(
        clock=clock,
        sample_count=1,
        max_abs_skew_ms=10,
    )
    with pytest.raises(BinanceDataError):
        asyncio.run(bad.sync(fetch))
    with pytest.raises(BinanceDataError):
        bad.timestamp_ms()


def test_retry_after_and_retry_execution():
    assert parse_retry_after({"Retry-After": "7"}) == 7
    clock = FakeClock()
    calls = []

    async def op():
        calls.append(1)
        if len(calls) < 3:
            raise BinanceRateLimitError(
                "slow",
                retry_after=2,
                retryable=True,
            )
        return "ok"

    result = asyncio.run(
        run_with_retry(
            op,
            policy=RetryPolicy(max_attempts=3, jitter_ratio=0),
            clock=clock,
        )
    )
    assert result == "ok"
    assert len(calls) == 3
    assert clock.mono == 104


def test_retry_after_is_honored_even_when_longer_than_backoff_cap():
    policy = RetryPolicy(max_delay_seconds=30, jitter_ratio=0)
    assert policy.delay_for(1, retry_after=3600) == 3600


def test_single_request_cannot_exceed_usable_weight_capacity():
    limiter = AdaptiveWeightLimiter(
        limit_per_minute=10,
        reserve_weight=2,
        clock=FakeClock(),
    )
    with pytest.raises(ValueError, match="exceeds"):
        asyncio.run(limiter.acquire(9))
