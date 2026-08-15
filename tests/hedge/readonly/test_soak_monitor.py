import asyncio
import json
import math
from dataclasses import replace
from decimal import Decimal

import pytest

from freqtrade.hedge.readonly.soak_monitor import (
    SoakMonitor,
    SoakObservation,
    SoakRunner,
)

from ._helpers import FakeClock


def observation(
    observed_at,
    *,
    funding="0",
    wallet="0",
    state="READY",
    fast=0,
    full=0,
    funding_period_observed=False,
):
    return SoakObservation(
        observed_at,
        state,
        1.0,
        fast,
        full,
        Decimal(wallet),
        Decimal(funding),
        Decimal("0"),
        Decimal("0"),
        Decimal("0"),
        funding_period_observed=funding_period_observed,
    )


def test_24_and_72_hour_zero_drift_funding_statistics(tmp_path):
    clock = FakeClock()
    monitor = SoakMonitor(
        tmp_path / "soak.jsonl",
        expected_interval_seconds=3600,
    )
    for hour in range(73):
        funding = "1" if hour == 8 else "0"
        monitor.append(
            observation(
                clock.now(),
                funding=funding,
                wallet=funding,
            )
        )
        if hour < 72:
            clock.advance(3600)

    observations = monitor.load()
    summary24 = monitor.summarize(observations[:25])
    assert summary24.meets_24h
    assert summary24.passed_24h

    summary72 = monitor.summarize(observations)
    assert summary72.meets_72h
    assert summary72.passed_72h
    assert summary72.funding_observation_count == 1


def test_sparse_observations_do_not_fake_a_continuous_24_hours(tmp_path):
    clock = FakeClock()
    monitor = SoakMonitor(tmp_path / "sparse.jsonl")
    monitor.append(observation(clock.now()))
    clock.advance(24 * 3600 + 1)
    monitor.append(observation(clock.now()))

    summary = monitor.summarize()
    assert summary.duration_hours >= 24
    assert not summary.continuous
    assert not summary.meets_24h
    assert not summary.passed_24h


def test_unexplained_drift_fails_observation(tmp_path):
    clock = FakeClock()
    monitor = SoakMonitor(
        tmp_path / "soak.jsonl",
        expected_interval_seconds=3600,
    )
    for hour in range(25):
        monitor.append(
            observation(
                clock.now(),
                wallet="1" if hour == 12 else "0",
            )
        )
        if hour < 24:
            clock.advance(3600)

    summary = monitor.summarize()
    assert summary.unexplained_drift_count == 1
    assert not summary.passed_24h


def test_non_ready_or_reconciliation_diff_fails_observation(tmp_path):
    clock = FakeClock()
    monitor = SoakMonitor(
        tmp_path / "state.jsonl",
        expected_interval_seconds=3600,
    )
    for hour in range(25):
        monitor.append(
            observation(
                clock.now(),
                state="DEGRADED" if hour == 2 else "READY",
                fast=1 if hour == 3 else 0,
            )
        )
        if hour < 24:
            clock.advance(3600)

    summary = monitor.summarize()
    assert summary.non_ready_observation_count == 1
    assert summary.reconciliation_diff_observation_count == 1
    assert not summary.passed_24h


def test_soak_runner_uses_current_run_and_fake_clock(tmp_path):
    clock = FakeClock()
    monitor = SoakMonitor(
        tmp_path / "runner.jsonl",
        expected_interval_seconds=3600,
    )
    monitor.append(observation(clock.now()))
    clock.advance(10 * 24 * 3600)

    class Provider:
        async def read_observation(self):
            return observation(clock.now())

    runner = SoakRunner(
        provider=Provider(),
        monitor=monitor,
        interval_seconds=3600,
        clock=clock,
    )
    summary = asyncio.run(runner.run(duration_hours=24.01))
    assert 25 <= summary.observation_count <= 26
    assert summary.meets_24h
    assert summary.continuous


def test_soak_rejects_nonfinite_runtime_values(tmp_path):
    clock = FakeClock()
    with pytest.raises(ValueError, match="stream_age"):
        SoakObservation(
            clock.now(),
            "ready",
            math.nan,
            0,
            0,
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
        )
    with pytest.raises(ValueError, match="expected_interval"):
        SoakMonitor(
            tmp_path / "x.jsonl",
            expected_interval_seconds=math.inf,
        )


def test_zero_value_funding_period_still_counts_as_observed(tmp_path):
    monitor = SoakMonitor(
        tmp_path / "funding.jsonl",
        expected_interval_seconds=3600,
        minimum_coverage_ratio=0.9,
        max_gap_multiplier=1.1,
    )
    clock = FakeClock()
    base = observation(clock.now())
    items = []
    for hour in range(73):
        items.append(
            replace(
                base,
                observed_at=clock.now(),
                funding_period_observed=hour == 8,
            )
        )
        clock.advance(3600)

    summary = monitor.summarize(items)
    assert summary.funding_observation_count == 1
    assert summary.passed_72h


def test_soak_rejects_non_boolean_funding_observation(tmp_path):
    clock = FakeClock()
    with pytest.raises(ValueError, match="funding_period_observed"):
        SoakObservation(
            clock.now(),
            "READY",
            0.0,
            0,
            0,
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            Decimal("0"),
            funding_period_observed=1,
        )

    path = tmp_path / "invalid_bool.jsonl"
    payload = {
        "observed_at": clock.now().isoformat(),
        "service_state": "READY",
        "stream_age_seconds": 0.0,
        "fast_diff_count": 0,
        "full_diff_count": 0,
        "wallet_delta": "0",
        "funding_income": "0",
        "realized_pnl": "0",
        "commissions": "0",
        "transfers": "0",
        "funding_period_observed": "false",
    }
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="funding_period_observed"):
        SoakMonitor(path).load()


def test_service_soak_provider_builds_deltas_from_runtime_and_repository():
    import asyncio
    from types import SimpleNamespace

    from freqtrade.hedge.exchange.base import CalibrationKind, ReadonlyState
    from freqtrade.hedge.readonly.soak_monitor import (
        ReadonlyServiceSoakProvider,
        SoakAccountingTotals,
    )

    class AccountingSource:
        def __init__(self):
            self.values = [
                SoakAccountingTotals(
                    Decimal("1000"),
                    Decimal("10"),
                    Decimal("20"),
                    Decimal("2"),
                    Decimal("5"),
                    1,
                ),
                SoakAccountingTotals(
                    Decimal("1014"),
                    Decimal("12"),
                    Decimal("30"),
                    Decimal("3"),
                    Decimal("8"),
                    2,
                ),
            ]

        async def read_soak_accounting_totals(self, account_id):
            return self.values.pop(0)

    class Service:
        status = SimpleNamespace(state=ReadonlyState.READY)

        def runtime_snapshot(self):
            return SimpleNamespace(
                status=self.status,
                freshness=SimpleNamespace(event_age_seconds=3.0),
            )

        def latest_calibration(self, kind):
            if kind is CalibrationKind.FAST:
                return SimpleNamespace(diff_count=0)
            if kind is CalibrationKind.FULL:
                return SimpleNamespace(diff_count=0)
            return None

    clock = FakeClock()
    provider = ReadonlyServiceSoakProvider(
        account_id="acct",
        service=Service(),
        accounting_source=AccountingSource(),
        clock=clock,
    )

    first = asyncio.run(provider.read_observation())
    clock.advance(60)
    second = asyncio.run(provider.read_observation())

    assert first.wallet_delta == Decimal("0")
    assert second.wallet_delta == Decimal("14")
    assert second.funding_income == Decimal("2")
    assert second.realized_pnl == Decimal("10")
    assert second.commissions == Decimal("1")
    assert second.transfers == Decimal("3")
    assert second.funding_period_observed
    assert second.unexplained_drift == Decimal("0")


def test_funding_period_is_counted_only_when_event_counter_advances(tmp_path):
    from types import SimpleNamespace

    from freqtrade.hedge.exchange.base import CalibrationKind, ReadonlyState
    from freqtrade.hedge.readonly.soak_monitor import (
        ReadonlyServiceSoakProvider,
        SoakAccountingTotals,
    )

    class AccountingSource:
        def __init__(self):
            self.values = [
                SoakAccountingTotals(
                    Decimal("1000"), Decimal("0"), Decimal("0"),
                    Decimal("0"), Decimal("0"), 0,
                ),
                SoakAccountingTotals(
                    Decimal("1001"), Decimal("1"), Decimal("0"),
                    Decimal("0"), Decimal("0"), 1,
                ),
                SoakAccountingTotals(
                    Decimal("1001"), Decimal("1"), Decimal("0"),
                    Decimal("0"), Decimal("0"), 1,
                ),
            ]

        async def read_soak_accounting_totals(self, account_id):
            return self.values.pop(0)

    class Service:
        status = SimpleNamespace(state=ReadonlyState.READY)

        def runtime_snapshot(self):
            return SimpleNamespace(
                status=self.status,
                freshness=SimpleNamespace(event_age_seconds=1.0),
            )

        def latest_calibration(self, kind):
            if kind in {CalibrationKind.FAST, CalibrationKind.FULL}:
                return SimpleNamespace(diff_count=0)
            return None

    baseline_path = tmp_path / "baseline.json"
    provider = ReadonlyServiceSoakProvider(
        account_id="acct",
        service=Service(),
        accounting_source=AccountingSource(),
        clock=FakeClock(),
        baseline_path=baseline_path,
        run_id="run-1",
    )

    first = asyncio.run(provider.read_observation())
    second = asyncio.run(provider.read_observation())
    third = asyncio.run(provider.read_observation())

    assert not first.funding_period_observed
    assert second.funding_period_observed
    assert second.funding_event_count_delta == 1
    assert not third.funding_period_observed
    assert third.funding_event_count_delta == 0
    assert baseline_path.exists()
    assert second.account_id == "acct"
    assert second.run_id == "run-1"


def test_soak_baseline_can_be_reloaded_after_process_restart(tmp_path):
    from types import SimpleNamespace

    from freqtrade.hedge.exchange.base import ReadonlyState
    from freqtrade.hedge.readonly.soak_monitor import (
        ReadonlyServiceSoakProvider,
        SoakAccountingTotals,
    )

    class AccountingSource:
        async def read_soak_accounting_totals(self, account_id):
            return SoakAccountingTotals(
                Decimal("1010"), Decimal("2"), Decimal("10"),
                Decimal("1"), Decimal("0"), 2,
            )

    class Service:
        status = SimpleNamespace(state=ReadonlyState.READY)

        def runtime_snapshot(self):
            return SimpleNamespace(
                status=self.status,
                freshness=SimpleNamespace(event_age_seconds=1.0),
            )

        def latest_calibration(self, kind):
            return None

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "wallet_balance": "1000",
                "funding_income": "1",
                "realized_pnl": "0",
                "commissions": "0",
                "transfers": "0",
                "funding_event_count": 1,
            }
        ),
        encoding="utf-8",
    )
    provider = ReadonlyServiceSoakProvider(
        account_id="acct",
        service=Service(),
        accounting_source=AccountingSource(),
        clock=FakeClock(),
        baseline_path=baseline_path,
    )

    observation_value = asyncio.run(provider.read_observation())

    assert observation_value.wallet_delta == Decimal("10")
    assert observation_value.funding_income == Decimal("1")
    assert observation_value.funding_event_count_delta == 1
