from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256

from freqtrade.hedge.production.binance_dryrun import BinanceDryRunPolicy, BinanceDryRunSafetyContext
from freqtrade.hedge.production.closed_loop import (
    ClosedLoopCycleJournal, ClosedLoopCycleRecord, ClosedLoopCycleStatus, ZERO_HASH,
)
from freqtrade.hedge.production.closed_loop_dryrun import evaluate_closed_loop_binance_dryrun
from freqtrade.hedge.telemetry.dryrun import DryRunCycleTelemetry, StrategyTelemetry

NOW = datetime(2026, 8, 15, 5, 0, tzinfo=UTC)
D = Decimal


def _record(index: int, previous: str) -> ClosedLoopCycleRecord:
    h = lambda text: sha256(text.encode()).hexdigest()
    return ClosedLoopCycleRecord(
        sequence=index + 1, cycle_id=f"cycle-{index}", observed_at=NOW + timedelta(minutes=index),
        source_release="release", model_id="model", symbol="BTCUSDT",
        projection_sequence=index + 1, projection_observed_at=NOW + timedelta(minutes=index),
        projection_source_sha256=h(f"s{index}"), projection_semantic_sha256=h(f"p{index}"),
        long_margin_ratio=D(".12"), short_margin_ratio=D(".05"),
        long_notional_ratio=D(".36"), short_notional_ratio=D(".15"), confidence=D("1"),
        projection_accepted=True, projection_reasons=(), projection_chain_sha256=h(f"c{index}"),
        planner_profile_sha256=h("planner"), input_state_sha256=h(f"i{index}"),
        planning_sha256=h(f"pl{index}"), execution_sha256=h(f"ex{index}"),
        reconciliation_digest=h("r"), evidence_digest=h("e"), safety_allows_reduce=True,
        safety_allows_new_risk=True, status=ClosedLoopCycleStatus.COMMITTED,
        writes_attempted=1, previous_record_sha256=previous,
    )


def _journal(count: int = 4) -> ClosedLoopCycleJournal:
    journal = ClosedLoopCycleJournal()
    previous = ZERO_HASH
    for index in range(count):
        record = _record(index, previous)
        journal.append(record)
        previous = record.record_sha256
    return journal


def _telemetry(count: int = 4):
    strategy = StrategyTelemetry(model_version="model", regime="HPRL")
    return tuple(DryRunCycleTelemetry(
        cycle_id=f"cycle-{i}", account_id="dryrun:binance", symbol="BTCUSDT",
        timestamp=NOW + timedelta(minutes=i), mark_price=D("100"), equity=D("1000"),
        available_balance=D("700"), gross_notional=D("300"), net_quantity=D("1"),
        long_quantity=D("2"), short_quantity=D("1"), long_target_quantity=D("2"),
        short_target_quantity=D("1"), strategy=strategy,
    ) for i in range(count))


def _safety():
    return BinanceDryRunSafetyContext(
        exchange="binance", operation_mode="dry_run", real_market_data=True,
        exchange_write_capability=False, simulated_execution=True,
        hedge_mode_semantics=True, cross_margin_semantics=True, source_release="release",
        account_namespace="dryrun",
    )


def test_dryrun_requires_telemetry_cycle_ids_to_match_closed_loop_journal() -> None:
    report = evaluate_closed_loop_binance_dryrun(
        _telemetry(), journal=_journal(), safety=_safety(),
        policy=BinanceDryRunPolicy(minimum_cycles=4, minimum_duration=timedelta(minutes=3),
            maximum_cycle_gap=timedelta(minutes=2)),
    )
    assert report.passed
    assert report.linked_cycle_count == 4


def test_unjournaled_dryrun_cycle_fails_closed() -> None:
    report = evaluate_closed_loop_binance_dryrun(
        _telemetry(4), journal=_journal(3), safety=_safety(),
        policy=BinanceDryRunPolicy(minimum_cycles=4, minimum_duration=timedelta(minutes=3),
            maximum_cycle_gap=timedelta(minutes=2)),
    )
    assert not report.passed
    assert "BINANCE_DRYRUN_UNJOURNALED_CYCLE" in report.reasons
