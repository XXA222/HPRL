from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest

from freqtrade.hedge.production.backtest_stability import (
    BacktestChunkEvidence,
    TwoYearBacktestPolicy,
    evaluate_two_year_backtest_stability,
)
from freqtrade.hedge.production.binance_runtime_dryrun import (
    acceptance_probe_targets,
    collect_binance_real_market_preflight,
    run_binance_real_market_dryrun,
)
from freqtrade.hedge.production.postgres_runtime_closure import (
    PostgresRestoreSnapshot,
    PostgresSnapshotTable,
    verify_postgres_restore,
)
from freqtrade.hedge.production.risk_behavior import (
    HprlBehaviorObservation,
    HprlBehaviorPolicy,
    analyze_hprl_position_behavior,
)
from freqtrade.hedge.production.runtime_closure import (
    EvidenceState,
    RuntimeClosureEvidence,
    evaluate_runtime_closure_acceptance, initialize_runtime_closure_evidence_registry,
    load_runtime_closure_evidence_registry, record_runtime_closure_evidence,
)
from freqtrade.hedge.production.runtime_fault_injection import run_focused_runtime_fault_campaign
from freqtrade.hedge.production.runtime_test_capability import (
    _IMPORT_BY_DISTRIBUTION,
    probe_runtime_test_capability,
    read_postgres_dependency_specs,
    read_test_dependency_specs,
)
from freqtrade.hedge.production.shadow import ShadowMetrics
from freqtrade.hedge.production.shadow_runtime import ShadowWindow
from freqtrade.hedge.production.shadow_soak_runtime import JsonlShadowWindowJournal

ROOT = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 8, 15, 8, 0, tzinfo=UTC)
D = Decimal


class _ReadonlyBinance:
    def __init__(self) -> None:
        self.clock_syncs = 0
        self.price_reads = 0

    async def synchronize_clock(self):
        self.clock_syncs += 1

    async def preflight_permissions(self, policy=None):
        return SimpleNamespace(strict_readonly_verified=True, runtime_readonly_enforced=True)

    async def fetch_bundle(self, include_fills=False):
        positions = (
            SimpleNamespace(symbol="BTCUSDT", position_side="LONG", quantity=D("0"), leverage=3),
            SimpleNamespace(symbol="BTCUSDT", position_side="SHORT", quantity=D("0"), leverage=3),
        )
        configuration = SimpleNamespace(
            hedge_mode=True,
            active_margin_modes=("cross",),
            leverage_by_symbol_side={"BTCUSDT:LONG": D("3"), "BTCUSDT:SHORT": D("3")},
        )
        account = SimpleNamespace(
            account_id="readonly-binance",
            total_margin_balance=D("1000"),
            total_available_balance=D("1000"),
        )
        return SimpleNamespace(
            positions=positions,
            configuration=configuration,
            account_snapshot=account,
            open_orders=(),
            collection_started_at=NOW,
            collection_completed_at=NOW + timedelta(milliseconds=10),
        )

    async def fetch_real_market_prices(self, symbol):
        self.price_reads += 1
        return D("99990"), D("100010"), D("100000")


class _UnsafeBinance(_ReadonlyBinance):
    def submit_order(self):  # pragma: no cover - never called; surface itself is forbidden
        raise AssertionError("must never be called")


def _hash(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _backtest_chunks():
    return (
        BacktestChunkEvidence(
            NOW, NOW + timedelta(days=350), 505_000, 505_000, 3000, 4 * 1024**3, 0,
            _hash("result-1"), _hash("data-1"),
        ),
        BacktestChunkEvidence(
            NOW + timedelta(days=350), NOW + timedelta(days=705), 511_000, 511_000, 3200,
            5 * 1024**3, 0, _hash("result-2"), _hash("data-2"),
        ),
    )


def test_runtime_test_dependencies_are_derived_from_pyproject() -> None:
    root = Path(__file__).resolve().parents[3]
    _, specs = read_test_dependency_specs(root)
    names = "\n".join(specs).lower()
    assert "pytest" in names
    assert "pytest-xdist" in names
    assert "pytest-timeout" in names
    assert "time-machine" in names


def test_runtime_test_capability_has_source_digest() -> None:
    root = Path(__file__).resolve().parents[3]
    report = probe_runtime_test_capability(root)
    assert len(report.source_tree_sha256) == 64
    assert report.python_executable


def test_postgres_restore_requires_isolated_database_and_exact_hashes() -> None:
    tables = (
        PostgresSnapshotTable("hedge_audit_events", 2, "a" * 64),
        PostgresSnapshotTable("hedge_execution_order_states", 1, "b" * 64),
    )
    source = PostgresRestoreSnapshot("prod", "16", tables, "c" * 64, NOW)
    restored = PostgresRestoreSnapshot("restore", "16", tables, "c" * 64, NOW + timedelta(minutes=1))
    assert verify_postgres_restore(source, restored).passed
    same_db = verify_postgres_restore(source, replace(restored, database_name="prod"))
    assert not same_db.passed
    assert "RESTORE_TARGET_NOT_ISOLATED" in same_db.reasons


def test_postgres_restore_rejects_hash_mismatch() -> None:
    source = PostgresRestoreSnapshot(
        "prod", "16", (PostgresSnapshotTable("hedge_audit_events", 2, "a" * 64),), "c" * 64, NOW
    )
    restored = PostgresRestoreSnapshot(
        "restore", "16", (PostgresSnapshotTable("hedge_audit_events", 2, "b" * 64),), "d" * 64, NOW
    )
    report = verify_postgres_restore(source, restored)
    assert not report.passed
    assert "RESTORE_TABLE_HASHES_MISMATCH" in report.reasons
    assert "RESTORE_SNAPSHOT_DIGEST_MISMATCH" in report.reasons


def test_binance_preflight_accepts_readonly_hedge_cross_client() -> None:
    client = _ReadonlyBinance()
    report = asyncio.run(collect_binance_real_market_preflight(client, symbol="BTC/USDT:USDT"))
    assert report.passed
    assert report.hedge_mode
    assert report.cross_margin
    assert report.leverage == D("3")
    assert client.clock_syncs == 1


def test_binance_preflight_refuses_exchange_write_surface() -> None:
    with pytest.raises(TypeError, match="exchange-write"):
        asyncio.run(collect_binance_real_market_preflight(_UnsafeBinance(), symbol="BTC/USDT:USDT"))


def test_binance_real_market_dryrun_uses_simulated_execution_only() -> None:
    client = _ReadonlyBinance()
    targets = acceptance_probe_targets("BTC/USDT:USDT", 5)
    report = asyncio.run(run_binance_real_market_dryrun(client, symbol="BTC/USDT:USDT", targets=targets))
    assert report.passed
    assert report.real_exchange_write_count == 0
    assert report.simulated_submit_count > 0
    assert report.simulated_fill_count > 0
    assert report.journal_valid
    assert len(report.cycles) == 5


def test_acceptance_probe_targets_include_dual_leg_state() -> None:
    targets = acceptance_probe_targets("BTC/USDT:USDT", 8)
    assert any(x.target_long_exposure > 0 and x.target_short_exposure > 0 for x in targets)
    assert all(x.metadata.get("source") == "acceptance-probe" for x in targets)


def test_focused_fault_campaign_converges_without_duplicate_writes() -> None:
    report = run_focused_runtime_fault_campaign()
    assert report.passed
    assert report.duplicate_write_free
    assert report.all_converged
    assert report.all_new_risk_fail_closed
    assert all(item.duplicate_writes == 0 for item in report.results)


def test_fault_campaign_contains_required_runtime_faults() -> None:
    names = {item.scenario.value for item in run_focused_runtime_fault_campaign().results}
    assert {
        "HTTP_TIMEOUT_AFTER_ACCEPT", "QUERY_TIMEOUT", "HTTP_429", "HTTP_5XX",
        "WS_DISCONNECT", "REST_STALE_SNAPSHOT", "PARTIAL_FILL",
        "PROCESS_CRASH_BEFORE_COMMIT", "PROCESS_CRASH_AFTER_COMMIT",
    } <= names


def _shadow_metrics() -> ShadowMetrics:
    return ShadowMetrics(
        duration=timedelta(hours=12),
        restart_recoveries=1,
        funding_cycles_observed=1,
        reconciliation_p99_seconds=0.2,
        loop_p99_ms=30,
        db_p99_ms=15,
        model_p99_ms=20,
        memory_growth_ratio=0.01,
        planner_churn_ratio=0.05,
        risk_reject_ratio=0.05,
    )


def test_shadow_journal_hash_chain_and_72h_qualification(tmp_path: Path) -> None:
    journal = JsonlShadowWindowJournal(tmp_path / "shadow.jsonl", source_release="r2")
    for index in range(6):
        window = ShadowWindow(
            NOW + timedelta(hours=12 * index),
            NOW + timedelta(hours=12 * (index + 1)),
            _shadow_metrics(),
            restart_boundary=index == 3,
            source_cursor_start=index * 100,
            source_cursor_end=index * 100 + 99,
        )
        journal.append(window, observed_at=window.ended_at)
    state = journal.load()
    assert state.valid
    assert len(state.records) == 6
    assert journal.qualify(target="24h").passed
    assert journal.qualify(target="72h").passed


def test_shadow_short_run_cannot_fake_24h(tmp_path: Path) -> None:
    journal = JsonlShadowWindowJournal(tmp_path / "short.jsonl", source_release="r2")
    window = ShadowWindow(NOW, NOW + timedelta(hours=12), _shadow_metrics(), source_cursor_end=99)
    journal.append(window, observed_at=window.ended_at)
    report = journal.qualify(target="24h")
    assert not report.passed
    assert "SOAK_DURATION_INSUFFICIENT" in report.reasons


def test_shadow_journal_detects_tamper(tmp_path: Path) -> None:
    journal = JsonlShadowWindowJournal(tmp_path / "shadow.jsonl", source_release="r2")
    window = ShadowWindow(NOW, NOW + timedelta(hours=12), _shadow_metrics(), source_cursor_end=99)
    journal.append(window, observed_at=window.ended_at)
    raw = (tmp_path / "shadow.jsonl").read_text(encoding="utf-8")
    (tmp_path / "shadow.jsonl").write_text(raw.replace('"source_release":"r2"', '"source_release":"bad"'), encoding="utf-8")
    assert not journal.load().valid


def test_two_year_backtest_requires_measured_repeat_digest() -> None:
    chunks = _backtest_chunks()
    first = evaluate_two_year_backtest_stability(chunks)
    assert not first.passed
    assert "TWO_YEAR_DETERMINISTIC_REPEAT_MISSING_OR_MISMATCH" in first.reasons
    second = evaluate_two_year_backtest_stability(chunks, repeat_result_sha256=first.aggregate_sha256)
    assert second.passed
    assert second.coverage >= timedelta(days=700)


def test_two_year_backtest_rejects_exit_137_and_rss_excess() -> None:
    chunks = _backtest_chunks()
    bad = (
        replace(chunks[0], exit_code=137, peak_rss_bytes=13 * 1024**3),
        chunks[1],
    )
    report = evaluate_two_year_backtest_stability(bad, repeat_result_sha256="0" * 64)
    assert not report.passed
    assert any(x.startswith("CHUNK_EXIT_NONZERO") for x in report.reasons)
    assert "TWO_YEAR_PEAK_RSS_EXCEEDED" in report.reasons


def test_two_year_policy_requires_long_coverage() -> None:
    policy = TwoYearBacktestPolicy()
    assert policy.minimum_coverage == timedelta(days=700)
    assert policy.minimum_bars == 700 * 24 * 60
    assert policy.maximum_peak_rss_bytes == 12 * 1024**3


def test_behavior_analysis_detects_upward_jump() -> None:
    rows = (
        HprlBehaviorObservation(NOW, D("0"), D("0"), 0, 0),
        HprlBehaviorObservation(NOW + timedelta(minutes=1), D("0.40"), D("0"), 0.01, 0),
    )
    report = analyze_hprl_position_behavior(
        rows, policy=HprlBehaviorPolicy(minimum_observations=2, minimum_distinct_joint_levels=1)
    )
    assert not report.passed
    assert report.upward_jump_violations == 1
    assert "BEHAVIOR_UPWARD_JUMP_VIOLATION" in report.reasons


def test_behavior_analysis_rejects_high_uncertainty_overrisk() -> None:
    rows = (
        HprlBehaviorObservation(NOW, D("0.25"), D("0.25"), 0, 0, 0.9),
        HprlBehaviorObservation(NOW + timedelta(minutes=1), D("0.25"), D("0.25"), 0, 0, 0.9),
    )
    report = analyze_hprl_position_behavior(
        rows, policy=HprlBehaviorPolicy(minimum_observations=2, minimum_distinct_joint_levels=1)
    )
    assert not report.passed
    assert report.high_uncertainty_overrisk > 0


def test_runtime_acceptance_is_pending_without_real_evidence(tmp_path: Path) -> None:
    report = evaluate_runtime_closure_acceptance({})
    assert report.state is EvidenceState.PENDING
    assert not report.passed
    assert len(report.pending_requirements) == 10
    registry = tmp_path / "runtime-evidence.json"
    initialize_runtime_closure_evidence_registry(registry)
    loaded = load_runtime_closure_evidence_registry(registry)
    assert len(loaded) == 10
    assert all(item.state is EvidenceState.PENDING for item in loaded.values())
    record_runtime_closure_evidence(
        registry, name="fault_campaign", state=EvidenceState.PASS,
        digest=_hash("fault-campaign-evidence"), detail="focused deterministic campaign",
    )
    assert load_runtime_closure_evidence_registry(registry)["fault_campaign"].state is EvidenceState.PASS


def test_runtime_acceptance_requires_all_evidence_pass() -> None:
    names = (
        "container_pytest", "postgres_core", "postgres_failover", "postgres_restore",
        "binance_real_market_dryrun", "fault_campaign", "shadow_24h", "shadow_72h",
        "two_year_backtest", "position_behavior",
    )
    evidence = {
        name: RuntimeClosureEvidence(name, EvidenceState.PASS, digest=_hash(name))
        for name in names
    }
    report = evaluate_runtime_closure_acceptance(evidence)
    assert report.passed
    evidence["postgres_failover"] = RuntimeClosureEvidence("postgres_failover", EvidenceState.FAIL)
    assert evaluate_runtime_closure_acceptance(evidence).state is EvidenceState.FAIL


def test_runtime_test_capability_uses_real_random_order_import_name() -> None:
    assert _IMPORT_BY_DISTRIBUTION["pytest-random-order"] == "random_order"


def test_runtime_closure_declares_dedicated_psycopg_binary_driver() -> None:
    specs = read_postgres_dependency_specs(ROOT)
    assert any(spec.startswith("psycopg[binary]") for spec in specs)
