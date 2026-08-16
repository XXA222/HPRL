from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
from freqtrade.hedge.production.backtest_real_environment import (
    JsonlBacktestEvidenceJournal,
    MeasuredBacktestCommand,
    MeasuredBacktestCommandReport,
    qualify_r3_two_year_backtest,
    run_measured_backtest_command,
)
from freqtrade.hedge.production.backtest_stability import BacktestChunkEvidence, TwoYearBacktestPolicy
from freqtrade.hedge.production.binance_real_environment import (
    inspect_binance_r3_safety_surface,
    run_binance_r3_real_market_acceptance,
)
from freqtrade.hedge.production.binance_runtime_dryrun import acceptance_probe_targets
from freqtrade.hedge.production import postgres_real_environment as pg_r3
from freqtrade.hedge.production.postgres_real_environment import (
    PostgresNodeIdentity,
    PostgresR3FailoverToken,
    verify_postgres_r3_failover_token,
)
from freqtrade.hedge.production.risk_behavior import HprlBehaviorObservation, HprlBehaviorPolicy
from freqtrade.hedge.production.risk_behavior_real_environment import (
    JsonlR3BehaviorJournal,
    R3BehaviorObservation,
    qualify_r3_behavior,
)
from freqtrade.hedge.production.shadow import ShadowMetrics
from freqtrade.hedge.production.shadow_real_environment import (
    JsonlR3ShadowJournal,
    MeasuredR3ShadowCommand,
    R3ShadowWindowEvidence,
    qualify_r3_shadow,
    run_measured_r3_shadow_command,
)
from freqtrade.hedge.production.shadow_runtime import ShadowWindow

NOW = datetime(2026, 8, 15, 8, 0, tzinfo=UTC)
D = Decimal


class _Transport:
    _ALLOWED = {
        ("GET", "/fapi/v1/time"),
        ("GET", "/fapi/v3/account"),
        ("GET", "/fapi/v3/positionRisk"),
        ("GET", "/fapi/v1/openOrders"),
        ("GET", "/fapi/v1/order"),
        ("GET", "/fapi/v1/positionSide/dual"),
        ("GET", "/fapi/v1/symbolConfig"),
        ("GET", "/fapi/v1/premiumIndex"),
        ("GET", "/fapi/v1/ticker/bookTicker"),
        ("POST", "/fapi/v1/listenKey"),
        ("PUT", "/fapi/v1/listenKey"),
        ("DELETE", "/fapi/v1/listenKey"),
    }

    def __init__(self) -> None:
        self.telemetry = SimpleNamespace(logical_request_count=0)


class _ReadonlyBinance:
    def __init__(self) -> None:
        self.transport = _Transport()

    async def synchronize_clock(self):
        self.transport.telemetry.logical_request_count += 1

    async def preflight_permissions(self, policy=None):
        self.transport.telemetry.logical_request_count += 1
        return SimpleNamespace(strict_readonly_verified=True, runtime_readonly_enforced=True, warnings=())

    async def fetch_bundle(self, include_fills=False):
        self.transport.telemetry.logical_request_count += 1
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
            account_id="r3-readonly",
            total_margin_balance=D("1000"),
            total_available_balance=D("1000"),
        )
        return SimpleNamespace(
            positions=positions,
            configuration=configuration,
            account_snapshot=account,
            open_orders=(),
            collection_started_at=NOW,
            collection_completed_at=NOW + timedelta(milliseconds=5),
        )

    async def fetch_real_market_prices(self, symbol):
        self.transport.telemetry.logical_request_count += 1
        return D("99990"), D("100010"), D("100000")


class _UnsafeTransport(_Transport):
    _ALLOWED = _Transport._ALLOWED | {("POST", "/fapi/v1/order")}


class _UnsafeClient(_ReadonlyBinance):
    def __init__(self) -> None:
        super().__init__()
        self.transport = _UnsafeTransport()


def _model_targets(count: int = 5) -> tuple[PlannedExecutionIntent, ...]:
    levels = ((0.05, 0.05), (0.12, 0.05), (0.12, 0.12), (0.05, 0.12), (0.05, 0.05))
    return tuple(
        PlannedExecutionIntent(
            symbol="BTC/USDT:USDT",
            target_long_exposure=levels[i % len(levels)][0],
            target_short_exposure=levels[i % len(levels)][1],
            confidence=1.0,
            model_id="hprl-r3-model",
            metadata={"source": "model-target-feed", "unit": "margin/equity", "uncertainty": "0.2"},
        )
        for i in range(count)
    )


def test_binance_r3_safety_surface_rejects_trade_write_route() -> None:
    assert inspect_binance_r3_safety_surface(_ReadonlyBinance()).passed
    report = inspect_binance_r3_safety_surface(_UnsafeClient())
    assert not report.passed
    assert "POST /fapi/v1/order" in report.forbidden_routes


def test_binance_r3_acceptance_probe_is_not_final_model_evidence() -> None:
    report = asyncio.run(run_binance_r3_real_market_acceptance(
        _ReadonlyBinance(), symbol="BTC/USDT:USDT",
        targets=acceptance_probe_targets("BTC/USDT:USDT", 5),
        require_model_target_feed=False,
    ))
    assert report.passed
    assert not report.production_evidence_eligible
    assert not report.model_target_feed
    assert report.real_trade_write_count == 0


def test_binance_r3_model_feed_is_eligible_and_emits_behavior() -> None:
    report = asyncio.run(run_binance_r3_real_market_acceptance(
        _ReadonlyBinance(), symbol="BTC/USDT:USDT", targets=_model_targets(5),
    ))
    assert report.passed
    assert report.production_evidence_eligible
    assert report.model_target_feed
    assert len(report.behavior_rows) == 5
    assert all(row.target_source == "model-target-feed" for row in report.behavior_rows)


def _hash(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _behavior_row(index: int, *, source: str = "model-target-feed") -> R3BehaviorObservation:
    return R3BehaviorObservation(
        cycle_id=f"cycle-{index}", model_id="hprl-r3", target_source=source,
        target_sha256=_hash(f"target-{index}"), market_evidence_sha256=_hash(f"market-{index}"),
        observation=HprlBehaviorObservation(
            timestamp=NOW + timedelta(minutes=index),
            long_margin_ratio=D("0.05") if index % 2 == 0 else D("0.12"),
            short_margin_ratio=D("0.05"), equity_return=0.001,
            drawdown=0.0, uncertainty=0.2,
        ),
    )


def test_behavior_journal_requires_model_target_feed(tmp_path: Path) -> None:
    journal = JsonlR3BehaviorJournal(tmp_path / "behavior.jsonl")
    journal.append((_behavior_row(0), _behavior_row(1)))
    state = journal.load()
    assert state.valid and len(state.records) == 2
    with pytest.raises(ValueError, match="model-target-feed"):
        journal.append((_behavior_row(2, source="acceptance-probe"),))


def test_behavior_journal_detects_tamper(tmp_path: Path) -> None:
    path = tmp_path / "behavior.jsonl"
    journal = JsonlR3BehaviorJournal(path)
    journal.append((_behavior_row(0),))
    path.write_text(path.read_text().replace("cycle-0", "cycle-X"), encoding="utf-8")
    assert not journal.load().valid


def test_behavior_qualification_consumes_durable_rows(tmp_path: Path) -> None:
    journal = JsonlR3BehaviorJournal(tmp_path / "behavior.jsonl")
    rows = tuple(_behavior_row(i) for i in range(8))
    journal.append(rows)
    report = qualify_r3_behavior(journal, policy=HprlBehaviorPolicy(
        minimum_observations=8,
        maximum_churn_ratio=1.0,
        minimum_distinct_joint_levels=2,
    ))
    assert report.passed
    assert report.market_bound and report.model_target_only


def _shadow_evidence(index: int, *, source: str = "model-target-feed") -> R3ShadowWindowEvidence:
    start = NOW + timedelta(hours=12 * index)
    end = start + timedelta(hours=12)
    return R3ShadowWindowEvidence(
        window=ShadowWindow(
            start, end,
            ShadowMetrics(
                duration=timedelta(hours=12), restart_recoveries=1 if index == 2 else 0,
                funding_cycles_observed=1, reconciliation_p99_seconds=0.2,
                loop_p99_ms=20, db_p99_ms=10, model_p99_ms=15,
                planner_churn_ratio=0.05, risk_reject_ratio=0.05,
            ),
            restart_boundary=index == 2,
            source_cursor_start=index * 100,
            source_cursor_end=(index + 1) * 100 - 1,
        ),
        source_release="freqtrade-hedge-hprl-v3-real-environment-r3",
        target_source=source,
        model_id="hprl-r3",
        model_observations=100,
        real_market_evidence_sha256=_hash(f"market-window-{index}"),
        behavior_chain_sha256=_hash(f"behavior-window-{index}"),
        process_rss_start_bytes=1024**3,
        process_rss_end_bytes=1024**3 + 1024,
        recorded_at=end,
    )


def test_r3_shadow_72h_requires_model_bound_windows(tmp_path: Path) -> None:
    journal = JsonlR3ShadowJournal(
        tmp_path / "shadow.jsonl", source_release="freqtrade-hedge-hprl-v3-real-environment-r3"
    )
    for i in range(6):
        journal.append(_shadow_evidence(i))
    report = qualify_r3_shadow(journal, target="72h")
    assert report.passed
    assert report.model_observations == 600
    assert report.model_target_only and report.real_market_bound


def test_r3_shadow_acceptance_probe_cannot_qualify(tmp_path: Path) -> None:
    journal = JsonlR3ShadowJournal(
        tmp_path / "shadow.jsonl", source_release="freqtrade-hedge-hprl-v3-real-environment-r3"
    )
    for i in range(2):
        journal.append(_shadow_evidence(i, source="acceptance-probe"))
    report = qualify_r3_shadow(journal, target="24h")
    assert not report.passed
    assert "SHADOW_R3_MODEL_TARGET_FEED_REQUIRED" in report.reasons


def test_backtest_measured_runner_collects_real_rss_and_hashes(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"; source.write_bytes(b"source")
    result = tmp_path / "result.json"; metrics = tmp_path / "metrics.json"
    script = tmp_path / "runner.py"
    script.write_text(
        "import json, pathlib\n"
        f"pathlib.Path({str(result)!r}).write_text(json.dumps({{'ok': True}}))\n"
        f"pathlib.Path({str(metrics)!r}).write_text(json.dumps({{'bars': 1000, 'events': 1000}}))\n",
        encoding="utf-8",
    )
    import sys
    command = MeasuredBacktestCommand(
        argv=(sys.executable, str(script)), cwd=str(tmp_path),
        started_at=NOW, ended_at=NOW + timedelta(days=1),
        source_data_path=str(source), result_path=str(result), metrics_path=str(metrics),
        timeout_seconds=30, poll_interval_seconds=0.01,
    )
    report = run_measured_backtest_command(command, output_dir=tmp_path / "out")
    assert report.passed and report.chunk is not None
    assert report.chunk.peak_rss_bytes > 0
    assert len(report.chunk.result_sha256) == 64
    assert len(report.chunk.source_data_sha256) == 64


def test_backtest_journal_hash_chain_and_qualification(tmp_path: Path) -> None:
    journal = JsonlBacktestEvidenceJournal(tmp_path / "backtest.jsonl")
    chunks = (
        BacktestChunkEvidence(NOW, NOW + timedelta(days=350), 505000, 505000, 1000, 4 * 1024**3, 0, _hash("r1"), _hash("s1")),
        BacktestChunkEvidence(NOW + timedelta(days=350), NOW + timedelta(days=705), 511000, 511000, 1100, 5 * 1024**3, 0, _hash("r2"), _hash("s2")),
    )
    for i, chunk in enumerate(chunks):
        journal.append(MeasuredBacktestCommandReport(chunk, _hash(f"cmd-{i}"), "", "", False, True, ()))
    first = journal.qualify(repeat_aggregate_sha256=None, policy=TwoYearBacktestPolicy())
    assert not first.passed
    assert "TWO_YEAR_DETERMINISTIC_REPEAT_MISSING_OR_MISMATCH" in first.reasons



def _append_backtest_chunks(journal: JsonlBacktestEvidenceJournal, chunks: tuple[BacktestChunkEvidence, ...], *, prefix: str) -> None:
    for i, chunk in enumerate(chunks):
        journal.append(MeasuredBacktestCommandReport(chunk, _hash(f"{prefix}-cmd-{i}"), "", "", False, True, ()))


def _two_year_chunks(*, second_result: str = "r2") -> tuple[BacktestChunkEvidence, ...]:
    return (
        BacktestChunkEvidence(NOW, NOW + timedelta(days=350), 505000, 505000, 1000, 4 * 1024**3, 0, _hash("r1"), _hash("s1")),
        BacktestChunkEvidence(NOW + timedelta(days=350), NOW + timedelta(days=705), 511000, 511000, 1100, 5 * 1024**3, 0, _hash(second_result), _hash("s2")),
    )


def test_two_year_repeat_requires_independent_journal(tmp_path: Path) -> None:
    journal = JsonlBacktestEvidenceJournal(tmp_path / "primary.jsonl")
    _append_backtest_chunks(journal, _two_year_chunks(), prefix="primary")
    report = qualify_r3_two_year_backtest(journal, journal, policy=TwoYearBacktestPolicy())
    assert not report.passed
    assert not report.independent_journals
    assert "TWO_YEAR_REPEAT_JOURNAL_MUST_BE_INDEPENDENT" in report.reasons


def test_two_year_repeat_rejects_semantic_result_mismatch(tmp_path: Path) -> None:
    primary = JsonlBacktestEvidenceJournal(tmp_path / "primary.jsonl")
    repeat = JsonlBacktestEvidenceJournal(tmp_path / "repeat.jsonl")
    _append_backtest_chunks(primary, _two_year_chunks(), prefix="primary")
    _append_backtest_chunks(repeat, _two_year_chunks(second_result="r2-different"), prefix="repeat")
    report = qualify_r3_two_year_backtest(primary, repeat, policy=TwoYearBacktestPolicy())
    assert not report.passed
    assert report.independent_journals
    assert not report.semantic_repeat_match
    assert "TWO_YEAR_DETERMINISTIC_REPEAT_SEMANTIC_MISMATCH" in report.reasons


def test_two_year_repeat_accepts_two_independently_measured_semantic_runs(tmp_path: Path) -> None:
    primary = JsonlBacktestEvidenceJournal(tmp_path / "primary.jsonl")
    repeat = JsonlBacktestEvidenceJournal(tmp_path / "repeat.jsonl")
    chunks = _two_year_chunks()
    _append_backtest_chunks(primary, chunks, prefix="primary")
    # Resource measurements may legitimately differ on the repeat run, while the
    # coverage, bars/events, source-data and result artifacts must be identical.
    repeat_chunks = tuple(
        replace(chunk, elapsed_seconds=chunk.elapsed_seconds + 7.0, peak_rss_bytes=chunk.peak_rss_bytes + 1024)
        for chunk in chunks
    )
    _append_backtest_chunks(repeat, repeat_chunks, prefix="repeat")
    report = qualify_r3_two_year_backtest(primary, repeat, policy=TwoYearBacktestPolicy())
    assert report.passed
    assert report.independent_journals
    assert report.same_chunk_count
    assert report.semantic_repeat_match
    assert report.primary_semantic_sha256 == report.repeat_semantic_sha256


def test_postgres_node_identity_primary_semantics() -> None:
    node = PostgresNodeIdentity(
        database_name="hedge", server_addr="10.0.0.1", server_port=5432,
        backend_pid=123, in_recovery=False, transaction_read_only=False,
        server_version="18", system_identifier="123456", wal_position="0/ABC", observed_at=NOW,
    )
    assert node.writable_primary
    assert node.endpoint == "10.0.0.1:5432"
    assert not replace(node, in_recovery=True).writable_primary


class _FakePgConnection:
    def __init__(self, role: str) -> None:
        self.role = role
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True


def _pg_identity(endpoint: str, *, recovery: bool = False, read_only: bool = False, system: str = "cluster-1") -> PostgresNodeIdentity:
    host, port = endpoint.split(":")
    return PostgresNodeIdentity(
        database_name="hedge",
        server_addr=host,
        server_port=int(port),
        backend_pid=101 if host.endswith("1") else 202,
        in_recovery=recovery,
        transaction_read_only=read_only,
        server_version="18.1",
        system_identifier=system,
        wal_position="0/ABC",
        observed_at=NOW,
    )


def _failover_token() -> PostgresR3FailoverToken:
    return PostgresR3FailoverToken(
        probe_id="probe-r3",
        payload_sha256="a" * 64,
        schema="freqtrade_hedge_probe",
        table="hprl_r3_failover_sentinel",
        primary=_pg_identity("10.0.0.1:5432"),
        writer_lock_key=12345,
        prepared_at=NOW,
    )


def _install_failover_stubs(monkeypatch: pytest.MonkeyPatch, *, routed: PostgresNodeIdentity, old: PostgresNodeIdentity, old_lock: bool) -> None:
    def fake_capture(connection, *, now):
        return routed if connection.role == "routed" else old

    def fake_scalar(connection, sql, params=()):
        if "payload_sha256" in sql:
            return "a" * 64
        if "pg_try_advisory_lock" in sql:
            return True if connection.role == "routed" else old_lock
        if "pg_advisory_unlock" in sql:
            return True
        return None

    monkeypatch.setattr(pg_r3, "capture_postgres_node_identity", fake_capture)
    monkeypatch.setattr(pg_r3, "_scalar", fake_scalar)
    monkeypatch.setattr(pg_r3, "_execute", lambda *args, **kwargs: None)


def test_postgres_r3_failover_rejects_same_routed_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _failover_token()
    _install_failover_stubs(
        monkeypatch,
        routed=_pg_identity("10.0.0.1:5432"),
        old=_pg_identity("10.0.0.1:5432", recovery=True, read_only=True),
        old_lock=False,
    )
    report = verify_postgres_r3_failover_token(
        token, lambda: _FakePgConnection("routed"),
        old_primary_factory=lambda: _FakePgConnection("old"), now=NOW,
    )
    assert not report.passed
    assert "FAILOVER_ROUTED_NODE_DID_NOT_CHANGE" in report.reasons


def test_postgres_r3_failover_rejects_cluster_identity_change(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _failover_token()
    _install_failover_stubs(
        monkeypatch,
        routed=_pg_identity("10.0.0.2:5432", system="cluster-2"),
        old=_pg_identity("10.0.0.1:5432", recovery=True, read_only=True),
        old_lock=False,
    )
    report = verify_postgres_r3_failover_token(
        token, lambda: _FakePgConnection("routed"),
        old_primary_factory=lambda: _FakePgConnection("old"), now=NOW,
    )
    assert not report.passed
    assert "FAILOVER_CLUSTER_IDENTITY_CHANGED" in report.reasons


def test_postgres_r3_failover_rejects_split_brain_old_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _failover_token()
    _install_failover_stubs(
        monkeypatch,
        routed=_pg_identity("10.0.0.2:5432"),
        old=_pg_identity("10.0.0.1:5432", recovery=False, read_only=False),
        old_lock=True,
    )
    report = verify_postgres_r3_failover_token(
        token, lambda: _FakePgConnection("routed"),
        old_primary_factory=lambda: _FakePgConnection("old"), now=NOW,
    )
    assert not report.passed
    assert "FAILOVER_OLD_PRIMARY_NOT_FENCED" in report.reasons


def test_postgres_r3_failover_accepts_changed_primary_and_fenced_old(monkeypatch: pytest.MonkeyPatch) -> None:
    token = _failover_token()
    _install_failover_stubs(
        monkeypatch,
        routed=_pg_identity("10.0.0.2:5432"),
        old=_pg_identity("10.0.0.1:5432", recovery=True, read_only=True),
        old_lock=False,
    )
    report = verify_postgres_r3_failover_token(
        token, lambda: _FakePgConnection("routed"),
        old_primary_factory=lambda: _FakePgConnection("old"), now=NOW,
    )
    assert report.passed
    assert report.routed_endpoint_changed
    assert report.cluster_identity_preserved
    assert report.old_primary_fenced


def test_behavior_qualification_rejects_mixed_model_ids(tmp_path: Path) -> None:
    journal = JsonlR3BehaviorJournal(tmp_path / "behavior.jsonl")
    first = _behavior_row(0)
    second = replace(_behavior_row(1), model_id="hprl-r3-other")
    journal.append((first, second))
    report = qualify_r3_behavior(journal, policy=HprlBehaviorPolicy(
        minimum_observations=2,
        maximum_churn_ratio=1.0,
        minimum_distinct_joint_levels=1,
    ))
    assert not report.passed
    assert "BEHAVIOR_R3_SINGLE_MODEL_REQUIRED" in report.reasons


def test_backtest_measured_runner_rejects_stale_output_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"; source.write_bytes(b"source")
    result = tmp_path / "result.json"; result.write_text("{}", encoding="utf-8")
    metrics = tmp_path / "metrics.json"; metrics.write_text('{"bars": 1}', encoding="utf-8")
    command = MeasuredBacktestCommand(
        argv=("python", "-c", "pass"), cwd=str(tmp_path),
        started_at=NOW, ended_at=NOW + timedelta(days=1),
        source_data_path=str(source), result_path=str(result), metrics_path=str(metrics),
        timeout_seconds=30,
    )
    report = run_measured_backtest_command(command, output_dir=tmp_path / "out")
    assert not report.passed
    assert "BACKTEST_RESULT_PREEXISTS" in report.reasons
    assert "BACKTEST_METRICS_PREEXISTS" in report.reasons



def test_measured_r3_shadow_binds_real_market_and_behavior(tmp_path: Path) -> None:
    metrics = tmp_path / "shadow-metrics.json"
    market = tmp_path / "real-market.json"
    behavior = tmp_path / "behavior.jsonl"
    shadow = tmp_path / "shadow.jsonl"
    script = tmp_path / "shadow_child.py"
    project_root = Path.cwd()
    script.write_text(
        "import json, pathlib, sys\n"
        f"sys.path.insert(0, {str(project_root)!r})\n"
        "from datetime import UTC, datetime\n"
        "from decimal import Decimal\n"
        "from hashlib import sha256\n"
        "from freqtrade.hedge.production.risk_behavior import HprlBehaviorObservation\n"
        "from freqtrade.hedge.production.risk_behavior_real_environment import JsonlR3BehaviorJournal, R3BehaviorObservation\n"
        f"behavior=pathlib.Path({str(behavior)!r})\n"
        "now=datetime.now(UTC)\n"
        "row=R3BehaviorObservation(cycle_id='shadow-cycle-1',model_id='hprl-r3-shadow',target_source='model-target-feed',target_sha256=sha256(b'target').hexdigest(),market_evidence_sha256=sha256(b'market').hexdigest(),observation=HprlBehaviorObservation(now,Decimal('0.05'),Decimal('0.05'),0.001,0.0,0.2))\n"
        "JsonlR3BehaviorJournal(behavior).append((row,))\n"
        f"pathlib.Path({str(metrics)!r}).write_text(json.dumps({{'funding_cycles_observed':1,'source_cursor_start':0,'source_cursor_end':0}}))\n"
        f"pathlib.Path({str(market)!r}).write_text(json.dumps({{'passed':True,'production_evidence_eligible':True,'model_target_feed':True,'real_trade_write_count':0}}))\n",
        encoding="utf-8",
    )
    import sys
    command = MeasuredR3ShadowCommand(
        argv=(sys.executable, str(script)), cwd=str(tmp_path),
        metrics_path=str(metrics), real_market_evidence_path=str(market),
        behavior_journal_path=str(behavior),
        source_release="freqtrade-hedge-hprl-v3-real-environment-r3",
        model_id="hprl-r3-shadow", timeout_seconds=30, poll_interval_seconds=0.01,
    )
    journal = JsonlR3ShadowJournal(shadow, source_release=command.source_release)
    report = run_measured_r3_shadow_command(command, shadow_journal=journal, output_dir=tmp_path / "out")
    assert report.passed
    assert report.evidence is not None
    assert report.evidence.model_observations == 1
    assert report.evidence.model_target_feed
    assert report.evidence.real_market_evidence_sha256 == sha256(market.read_bytes()).hexdigest()
    assert report.rss_samples > 0
    assert journal.load().valid
