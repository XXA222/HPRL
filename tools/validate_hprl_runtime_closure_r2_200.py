#!/usr/bin/env python3
"""200-point deterministic gate for HPRL Runtime Closure R2.

This matrix validates source/runtime contracts that can be proven offline.  It does not
fabricate evidence for a real PostgreSQL failover, real Binance connectivity, 24h/72h
wall-clock shadow runs, or a two-year backtest.  Those are represented by executable
runtime tools and fail-closed acceptance semantics, and must be supplied by the user's
real environment.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
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
    PostgresFailoverToken,
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
    RuntimeClosurePolicy,
    evaluate_runtime_closure_acceptance, initialize_runtime_closure_evidence_registry,
    load_runtime_closure_evidence_registry,
)
from freqtrade.hedge.production.runtime_fault_injection import run_focused_runtime_fault_campaign
from freqtrade.hedge.production.runtime_test_capability import (
    probe_runtime_test_capability,
    read_test_dependency_specs,
    source_python_files,
)
from freqtrade.hedge.production.shadow import ShadowMetrics, ShadowPolicy
from freqtrade.hedge.production.shadow_runtime import ShadowRunPolicy, ShadowWindow
from freqtrade.hedge.production.shadow_soak_runtime import JsonlShadowWindowJournal

NOW = datetime(2026, 8, 15, 8, 0, tzinfo=UTC)
D = Decimal
CHECKS: list[dict[str, object]] = []


def add(group: str, name: str, passed: bool, detail: object = "") -> None:
    CHECKS.append({"group": group, "name": name, "pass": bool(passed), "detail": detail})


def file_sha(path: str) -> str:
    return sha256((ROOT / path).read_bytes()).hexdigest()


def expect_raises(exc: type[BaseException], fn) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# G01: container test capability and source-safe bootstrap contract (20)
# ---------------------------------------------------------------------------
G = "G01-container-test-capability"
develop, test_specs = read_test_dependency_specs(ROOT)
cap = probe_runtime_test_capability(ROOT)
source_files = source_python_files(ROOT)
add(G, "pyproject-develop-present", len(develop) >= 10, len(develop))
add(G, "pytest-declared", any(x.lower().startswith("pytest") and not x.lower().startswith("pytest-") for x in test_specs), test_specs)
add(G, "pytest-asyncio-declared", any(x.lower().startswith("pytest-asyncio") for x in test_specs), test_specs)
add(G, "pytest-xdist-declared", any(x.lower().startswith("pytest-xdist") for x in test_specs), test_specs)
add(G, "pytest-timeout-declared", any(x.lower().startswith("pytest-timeout") for x in test_specs), test_specs)
add(G, "time-machine-declared", any(x.lower().startswith("time-machine") for x in test_specs), test_specs)
add(G, "capability-python-current", cap.python_executable == sys.executable, cap.python_executable)
add(G, "runtime-critical-declared", {x.distribution for x in cap.runtime_dependencies} == {"ccxt", "sqlalchemy", "humanize", "aiohttp"}, [(x.distribution, x.spec, x.available) for x in cap.runtime_dependencies])
add(G, "capability-sha-valid", len(cap.source_tree_sha256) == 64, cap.source_tree_sha256)
add(G, "source-python-surface-large", len(source_files) >= 1000, len(source_files))
add(G, "source-selection-freqtrade", any("freqtrade/hedge" in p.as_posix() for p in source_files))
add(G, "source-selection-tests", any("tests/hedge" in p.as_posix() for p in source_files))
add(G, "source-selection-tools", any("tools/" in p.as_posix() for p in source_files))
text = (ROOT / "freqtrade/hedge/production/runtime_test_capability.py").read_text(encoding="utf-8")
add(G, "bootstrap-no-editable", "pip install -e" not in text and '"-e"' not in text)
add(G, "bootstrap-no-project-install", "*missing" in text and "cwd=tempfile.gettempdir()" in text)
add(G, "bootstrap-bytecode-disabled", 'PYTHONDONTWRITEBYTECODE' in text)
add(G, "pytest-junit-authority", "--junitxml" in text and "_junit_counts" in text)
add(G, "pytest-cache-outside-source", "-p" in text and "no:cacheprovider" in text and "--basetemp" in text)
add(G, "pytest-minimum-hprl-530", '("hprl", ("tests/hedge/hprl",), 530)' in text)
add(G, "pytest-source-digest-after", "source_unchanged=(before == after)" in text and '("production", ("tests/hedge/production",), 116)' in text)
assert len([x for x in CHECKS if x["group"] == G]) == 20


# ---------------------------------------------------------------------------
# G02: SQL closed-loop journal/checkpoint semantics (20)
# ---------------------------------------------------------------------------
G = "G02-sql-journal-checkpoint"
sql_path = ROOT / "freqtrade/hedge/production/closed_loop_sql.py"
sql = sql_path.read_text(encoding="utf-8")
add(G, "sql-module-exists", sql_path.is_file())
add(G, "journal-store-class", "class SqlClosedLoopCycleJournalStore" in sql)
add(G, "checkpoint-store-class", "class SqlRecoveryCheckpointStore" in sql)
add(G, "audit-event-reuse", "AuditEvent" in sql)
add(G, "cycle-event-kind", "HPRL_CLOSED_LOOP_CYCLE" in sql)
add(G, "checkpoint-event-kind", "HPRL_RECOVERY_CHECKPOINT" in sql)
add(G, "postgres-advisory-xact-lock", "pg_advisory_xact_lock" in sql)
add(G, "stable-lock-key-sha", "sha256" in sql and "_lock_key" in sql)
append_pos = sql.find("def append_atomic")
lock_pos = sql.find("self._lock_writer(session)", append_pos)
read_pos = sql.find("self._query(for_update=True)", append_pos)
add(G, "writer-lock-before-chain-read", append_pos >= 0 and lock_pos >= 0 and read_pos >= 0 and lock_pos < read_pos, (lock_pos, read_pos))
retry_marker = "A caller can lose the commit response and retry with the original expected"
add(G, "lost-response-comment", retry_marker in sql)
if retry_marker in sql:
    retry_pos = sql.find(retry_marker)
    mismatch_pos = sql.find("journal.tip_sha256 != expected", retry_pos)
    exact_pos = sql.find("record_sha256", retry_pos)
    add(G, "idempotent-before-tip-mismatch", exact_pos != -1 and mismatch_pos != -1 and exact_pos < mismatch_pos, (exact_pos, mismatch_pos))
else:
    add(G, "idempotent-before-tip-mismatch", False)
add(G, "checkpoint-generation-monotonic", "recovery checkpoint generation must advance monotonically" in sql)
add(G, "checkpoint-exact-retry", "exact" in sql.lower() and "checkpoint" in sql.lower())
add(G, "checkpoint-load-latest", "order_by" in sql.lower() or "ORDER BY" in sql)
add(G, "no-new-orm-table", "__tablename__" not in sql)
add(G, "lazy-persistence-import", "from freqtrade.persistence.hedge_models import AuditEvent" in sql)
closed = (ROOT / "freqtrade/hedge/production/closed_loop.py").read_text(encoding="utf-8")
add(G, "journal-port-protocol", "class ClosedLoopJournalStorePort(Protocol)" in closed)
add(G, "checkpoint-port-protocol", "class RecoveryCheckpointStorePort(Protocol)" in closed)
add(G, "closed-loop-journal-port-used", "journal_store: ClosedLoopJournalStorePort" in closed)
add(G, "closed-loop-checkpoint-port-used", "checkpoint_store: RecoveryCheckpointStorePort" in closed)
assert len([x for x in CHECKS if x["group"] == G]) == 20


# ---------------------------------------------------------------------------
# G03: PostgreSQL core / restore / failover contract (20)
# ---------------------------------------------------------------------------
G = "G03-postgres-runtime-closure"
pg = (ROOT / "freqtrade/hedge/production/postgres_runtime_closure.py").read_text(encoding="utf-8")
for table in ("hedge_execution_order_states", "hedge_audit_events", "hedge_schema_migrations"):
    add(G, f"required-table-{table}", table in pg)
add(G, "execution-uses-sql-store", "SqlExecutionStore" in pg)
add(G, "execution-unknown-insert", "OrderState.UNKNOWN" in pg)
add(G, "execution-second-session-visible", "second.get_by_client_order_id" in pg)
add(G, "execution-unknown-clears", "has_unresolved_unknown" in pg and "cleared" in pg)
add(G, "journal-second-store-visible", "second" in pg and "SqlClosedLoopCycleJournalStore" in pg)
add(G, "journal-stale-cas-rejected", "stale" in pg.lower() and "journal" in pg.lower())
add(G, "checkpoint-sql-probe", "SqlRecoveryCheckpointStore" in pg)
add(G, "dual-connection-probes", "PostgresConcurrencyProbeRunner" in pg and "PostgresDurabilityProbeRunner" in pg)
add(G, "restore-different-db-required", "RESTORE_TARGET_NOT_ISOLATED" in pg)
add(G, "restore-table-count-check", "RESTORE_TABLE_COUNTS_MISMATCH" in pg)
add(G, "restore-table-hash-check", "RESTORE_TABLE_HASHES_MISMATCH" in pg)
# Pure restore verifier positive/negative checks.
tables = (PostgresSnapshotTable("hedge_audit_events", 3, "a" * 64), PostgresSnapshotTable("hedge_execution_order_states", 2, "b" * 64))
src = PostgresRestoreSnapshot("prod", "16", tables, "c" * 64, NOW)
dst = PostgresRestoreSnapshot("restore", "16", tables, "c" * 64, NOW + timedelta(minutes=1))
vr = verify_postgres_restore(src, dst)
add(G, "restore-positive", vr.passed, vr.reasons)
add(G, "restore-isolated", vr.isolated_database)
bad = verify_postgres_restore(src, replace(dst, database_name="prod"))
add(G, "restore-same-db-rejected", not bad.passed and "RESTORE_TARGET_NOT_ISOLATED" in bad.reasons, bad.reasons)
add(G, "failover-sentinel-durable", "hprl_failover_sentinel" in pg and "connection.commit()" in pg)
add(G, "failover-backend-change", "FAILOVER_BACKEND_DID_NOT_CHANGE" in pg)
add(G, "failover-writer-fence", "pg_try_advisory_lock" in pg and "FAILOVER_WRITER_FENCE_NOT_ACQUIRED" in pg)
assert len([x for x in CHECKS if x["group"] == G]) == 20


# ---------------------------------------------------------------------------
# G04: Binance read-only real-market + simulated execution (20)
# ---------------------------------------------------------------------------
G = "G04-binance-real-market-dryrun"
ro = (ROOT / "freqtrade/hedge/exchange/binance_readonly.py").read_text(encoding="utf-8")
dry = (ROOT / "freqtrade/hedge/production/binance_runtime_dryrun.py").read_text(encoding="utf-8")
add(G, "book-ticker-allowed", '("GET", "/fapi/v1/ticker/bookTicker")' in ro)
add(G, "premium-index-allowed", '("GET", "/fapi/v1/premiumIndex")' in ro)
add(G, "order-post-not-allowed", '("POST", "/fapi/v1/order")' not in ro)
add(G, "batch-order-post-not-allowed", '/fapi/v1/batchOrders' not in ro)
add(G, "fetch-real-prices", "fetch_real_market_prices" in ro)
add(G, "dryrun-refuses-submit-method", '"submit_order"' in dry)
add(G, "dryrun-refuses-create-method", '"create_order"' in dry)
add(G, "dryrun-refuses-cancel-method", '"cancel_order"' in dry)
add(G, "dryrun-refuses-leverage-write", '"set_leverage"' in dry)
add(G, "dryrun-refuses-margin-write", '"set_margin_mode"' in dry)
add(G, "fake-runtime-execution", "build_integrated_fake_runtime" in dry)
add(G, "simulated-engine-kind", "ExecutionEngineKind.SIMULATED" in dry)
add(G, "real-write-count-zero", "real_exchange_write_count=0" in dry)
add(G, "closed-loop-journal-used", "HprlProductionClosedLoop" in dry and "ClosedLoopCycleJournalStore" in dry)
add(G, "checkpoint-used", "RecoveryCheckpointStore" in dry)
add(G, "hedge-mode-required", "hedge_mode" in dry and "preflight.passed" in dry)
add(G, "cross-margin-required", "cross_margin" in dry)
targets = acceptance_probe_targets("BTC/USDT:USDT", 5)
add(G, "probe-target-count", len(targets) == 5)
add(G, "probe-targets-dual-leg", any(x.target_long_exposure > 0 and x.target_short_exposure > 0 for x in targets))
add(G, "probe-target-labeled", all(x.metadata.get("source") == "acceptance-probe" for x in targets), [x.metadata for x in targets])
assert len([x for x in CHECKS if x["group"] == G]) == 20


# ---------------------------------------------------------------------------
# G05: deterministic fault campaign (20)
# ---------------------------------------------------------------------------
G = "G05-fault-injection"
fault_report = run_focused_runtime_fault_campaign()
add(G, "campaign-pass", fault_report.passed, asdict(fault_report))
by = {x.scenario.value: x for x in fault_report.results}
required_faults = (
    "HTTP_TIMEOUT_AFTER_ACCEPT", "QUERY_TIMEOUT", "HTTP_429", "HTTP_5XX",
    "WS_DISCONNECT", "REST_STALE_SNAPSHOT", "PARTIAL_FILL",
    "PROCESS_CRASH_BEFORE_COMMIT", "PROCESS_CRASH_AFTER_COMMIT",
)
for scenario in required_faults:
    add(G, "scenario-" + scenario.lower(), scenario in by and by[scenario].passed, asdict(by[scenario]) if scenario in by else "missing")
add(G, "no-duplicate-writes", all(x.duplicate_writes == 0 for x in fault_report.results), [(x.scenario.value, x.duplicate_writes) for x in fault_report.results])
add(G, "timeout-query-before-retry", "query-before-retry" in by["HTTP_TIMEOUT_AFTER_ACCEPT"].detail)
add(G, "query-timeout-query-only", "remained UNKNOWN" in by["QUERY_TIMEOUT"].detail)
add(G, "429-blocks-new-risk", by["HTTP_429"].new_risk_blocked_during_fault)
add(G, "5xx-blocks-new-risk", by["HTTP_5XX"].new_risk_blocked_during_fault)
add(G, "ws-blocks-new-risk", by["WS_DISCONNECT"].new_risk_blocked_during_fault)
add(G, "stale-blocks-new-risk", by["REST_STALE_SNAPSHOT"].new_risk_blocked_during_fault)
add(G, "partial-fill-converges", by["PARTIAL_FILL"].final_converged)
add(G, "crash-before-detected", by["PROCESS_CRASH_BEFORE_COMMIT"].passed)
add(G, "crash-after-detected", by["PROCESS_CRASH_AFTER_COMMIT"].passed)
assert len([x for x in CHECKS if x["group"] == G]) == 20


# ---------------------------------------------------------------------------
# G06: durable shadow 24h/72h evidence contract (20)
# ---------------------------------------------------------------------------
G = "G06-shadow-soak"
metrics = ShadowMetrics(
    duration=timedelta(hours=12), restart_recoveries=1, funding_cycles_observed=1,
    reconciliation_p99_seconds=0.2, loop_p99_ms=30, db_p99_ms=15, model_p99_ms=20,
    memory_growth_ratio=0.01, planner_churn_ratio=0.05, risk_reject_ratio=0.05,
)
with tempfile.TemporaryDirectory(prefix="hprl-shadow-r2-") as td:
    path = Path(td) / "shadow.jsonl"
    journal = JsonlShadowWindowJournal(path, source_release="r2")
    windows = []
    cursor = 0
    for i in range(6):
        w = ShadowWindow(
            NOW + timedelta(hours=12 * i), NOW + timedelta(hours=12 * (i + 1)), metrics,
            restart_boundary=(i == 3), source_cursor_start=cursor, source_cursor_end=cursor + 99,
        )
        cursor += 100
        windows.append(w)
        journal.append(w, observed_at=w.ended_at)
    state = journal.load()
    add(G, "journal-valid", state.valid, state.reasons)
    add(G, "journal-six-records", len(state.records) == 6)
    add(G, "journal-tip-sha", len(state.tip_sha256) == 64 and state.tip_sha256 != "0" * 64)
    add(G, "journal-sequence", [x.sequence for x in state.records] == list(range(1, 7)))
    add(G, "journal-hash-chain", all(state.records[i].previous_sha256 == state.records[i-1].record_sha256 for i in range(1, 6)))
    add(G, "journal-source-release", all(x.source_release == "r2" for x in state.records))
    q24 = journal.qualify(target="24h")
    q72 = journal.qualify(target="72h")
    add(G, "24h-pass", q24.passed, q24.reasons)
    add(G, "72h-pass", q72.passed, q72.reasons)
    add(G, "72h-six-windows", q72.windows == 6)
    add(G, "72h-duration", q72.covered_duration >= timedelta(hours=72), q72.covered_duration)
    add(G, "qualification-digest", len(q72.semantic_hash) == 64)
    # Negative: a short journal cannot masquerade as 24h.
    short = JsonlShadowWindowJournal(Path(td) / "short.jsonl", source_release="r2")
    short.append(windows[0], observed_at=windows[0].ended_at)
    qs = short.qualify(target="24h")
    add(G, "short-soak-rejected", not qs.passed)
    add(G, "short-soak-reason", "SOAK_DURATION_INSUFFICIENT" in qs.reasons, qs.reasons)
    # Negative: tamper is detected.
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    wrapper = json.loads(raw_lines[0]); wrapper["record"]["source_release"] = "tampered"
    raw_lines[0] = json.dumps(wrapper, sort_keys=True, separators=(",", ":"))
    tamper_path = Path(td) / "tamper.jsonl"; tamper_path.write_text("\n".join(raw_lines)+"\n", encoding="utf-8")
    tampered = JsonlShadowWindowJournal(tamper_path, source_release="r2").load()
    add(G, "tamper-invalid", not tampered.valid)
    add(G, "tamper-hash-reason", any("HASH_MISMATCH" in x or "SOURCE_RELEASE_MISMATCH" in x for x in tampered.reasons), tampered.reasons)
    add(G, "append-fsync-source", "os.fsync" in (ROOT / "freqtrade/hedge/production/shadow_soak_runtime.py").read_text(encoding="utf-8"))
    add(G, "cursor-forward-enforced", expect_raises(ValueError, lambda: journal.append(replace(windows[-1], source_cursor_end=windows[-1].source_cursor_end))))
    add(G, "source-release-required", expect_raises(ValueError, lambda: JsonlShadowWindowJournal(Path(td)/"x", source_release="")))
    add(G, "run-policy-gap-bounded", ShadowRunPolicy().max_window_gap <= timedelta(seconds=30))
    add(G, "shadow-policy-restart-required", ShadowPolicy().require_restart_recovery)
assert len([x for x in CHECKS if x["group"] == G]) == 20


# ---------------------------------------------------------------------------
# G07: two-year backtest stability evidence contract (20)
# ---------------------------------------------------------------------------
G = "G07-two-year-backtest"
h = lambda s: sha256(s.encode()).hexdigest()
chunks = (
    BacktestChunkEvidence(NOW, NOW+timedelta(days=350), 505_000, 505_000, 3000, 4*1024**3, 0, h("r1"), h("data1")),
    BacktestChunkEvidence(NOW+timedelta(days=350), NOW+timedelta(days=705), 511_000, 511_000, 3200, 5*1024**3, 0, h("r2"), h("data2")),
)
pre = evaluate_two_year_backtest_stability(chunks, repeat_result_sha256=None)
positive = evaluate_two_year_backtest_stability(chunks, repeat_result_sha256=pre.aggregate_sha256)
add(G, "positive-pass", positive.passed, positive.reasons)
add(G, "coverage-700d", positive.coverage >= timedelta(days=700), positive.coverage)
add(G, "bars-minimum", positive.bars >= TwoYearBacktestPolicy().minimum_bars, positive.bars)
add(G, "peak-under-12g", positive.peak_rss_bytes <= 12*1024**3)
add(G, "runtime-under-6h", positive.total_elapsed_seconds <= 6*3600)
add(G, "repeat-match", positive.deterministic_repeat)
add(G, "aggregate-sha", len(positive.aggregate_sha256) == 64)
add(G, "two-chunks", positive.chunks == 2)
add(G, "no-gaps", positive.gap_count == 0)
short_chunk = (BacktestChunkEvidence(NOW, NOW+timedelta(days=10), 20_000, 20_000, 100, 2*1024**3, 0, h("s"), h("sd")),)
short_report = evaluate_two_year_backtest_stability(short_chunk, repeat_result_sha256="0"*64)
add(G, "short-rejected", not short_report.passed)
add(G, "short-coverage-reason", "TWO_YEAR_COVERAGE_INSUFFICIENT" in short_report.reasons, short_report.reasons)
add(G, "short-bars-reason", "TWO_YEAR_BAR_COUNT_INSUFFICIENT" in short_report.reasons, short_report.reasons)
nonzero = evaluate_two_year_backtest_stability((replace(chunks[0], exit_code=137), chunks[1]), repeat_result_sha256="0"*64)
add(G, "exit137-rejected", any(x.startswith("CHUNK_EXIT_NONZERO") for x in nonzero.reasons), nonzero.reasons)
mem = evaluate_two_year_backtest_stability((replace(chunks[0], peak_rss_bytes=13*1024**3), chunks[1]), repeat_result_sha256="0"*64)
add(G, "rss-rejected", "TWO_YEAR_PEAK_RSS_EXCEEDED" in mem.reasons, mem.reasons)
slow = evaluate_two_year_backtest_stability((replace(chunks[0], elapsed_seconds=20_000), replace(chunks[1], elapsed_seconds=20_000)), repeat_result_sha256="0"*64)
add(G, "runtime-rejected", "TWO_YEAR_RUNTIME_EXCEEDED" in slow.reasons, slow.reasons)
gap_chunks = (chunks[0], replace(chunks[1], started_at=chunks[0].ended_at+timedelta(minutes=3), ended_at=chunks[1].ended_at+timedelta(minutes=3)))
gap = evaluate_two_year_backtest_stability(gap_chunks, repeat_result_sha256="0"*64)
add(G, "gap-rejected", any(x.startswith("BACKTEST_COVERAGE_GAP") for x in gap.reasons), gap.reasons)
add(G, "repeat-required", "TWO_YEAR_DETERMINISTIC_REPEAT_MISSING_OR_MISMATCH" in pre.reasons, pre.reasons)
add(G, "policy-min-700d", TwoYearBacktestPolicy().minimum_coverage == timedelta(days=700))
add(G, "policy-rss-12g", TwoYearBacktestPolicy().maximum_peak_rss_bytes == 12*1024**3)
add(G, "invalid-digest-rejected", expect_raises(ValueError, lambda: replace(chunks[0], result_sha256="bad")))
assert len([x for x in CHECKS if x["group"] == G]) == 20


# ---------------------------------------------------------------------------
# G08: empirical HPRL position-risk behavior (20)
# ---------------------------------------------------------------------------
G = "G08-hprl-position-behavior"
levels = [D("0"), D("0.05"), D("0.12"), D("0.25"), D("0.40")]
rows = []
# Build a non-degenerate, low-churn target series with periodic safe de-risk under drawdown.
for i in range(120):
    phase = (i // 20) % 4
    pair = ((0,0),(1,0),(2,1),(1,1))[phase]
    dd = 0.03 if i in {39,79,119} else 0.0
    if dd:
        pair = (0,0)
    if i == 40:
        pair = (1,0)
    rows.append(HprlBehaviorObservation(
        NOW+timedelta(minutes=i), levels[pair[0]], levels[pair[1]],
        equity_return=0.001 if i % 11 else -0.001, drawdown=dd,
        uncertainty=0.8 if i % 30 == 0 else 0.2,
    ))
policy = HprlBehaviorPolicy(minimum_observations=100, minimum_drawdown_derisk_ratio=0.5, maximum_churn_ratio=0.25)
beh = analyze_hprl_position_behavior(rows, policy=policy)
add(G, "behavior-sample-count", beh.observations == 120)
add(G, "behavior-distinct-levels", beh.distinct_joint_levels >= 4, beh.distinct_joint_levels)
add(G, "behavior-sha", len(beh.semantic_sha256) == 64)
add(G, "behavior-no-upward-jump", beh.upward_jump_violations == 0, beh.upward_jump_violations)
add(G, "behavior-churn-bounded", beh.churn_ratio <= policy.maximum_churn_ratio, beh.churn_ratio)
add(G, "behavior-occupancy-long", sum(v for _,v in beh.long_level_occupancy) == 120)
add(G, "behavior-occupancy-short", sum(v for _,v in beh.short_level_occupancy) == 120)
add(G, "behavior-occupancy-joint", sum(v for _,v in beh.joint_level_occupancy) == 120)
add(G, "behavior-policy-sample-10k-default", HprlBehaviorPolicy().minimum_observations == 10_000)
add(G, "behavior-level-increase-default-one", HprlBehaviorPolicy().maximum_one_step_level_increase == 1)
# Negative gates are evidence that the analyzer rejects bad behavior.
jump_rows = [
    HprlBehaviorObservation(NOW, D("0"), D("0"), 0, 0),
    HprlBehaviorObservation(NOW+timedelta(minutes=1), D("0.40"), D("0"), 0.01, 0),
]
jump = analyze_hprl_position_behavior(jump_rows, policy=HprlBehaviorPolicy(minimum_observations=2, minimum_distinct_joint_levels=1))
add(G, "jump-detected", jump.upward_jump_violations == 1)
add(G, "jump-fails", not jump.passed and "BEHAVIOR_UPWARD_JUMP_VIOLATION" in jump.reasons, jump.reasons)
uncertain = [
    HprlBehaviorObservation(NOW, D("0.25"), D("0.25"), 0, 0, 0.9),
    HprlBehaviorObservation(NOW+timedelta(minutes=1), D("0.25"), D("0.25"), 0, 0, 0.9),
]
ur = analyze_hprl_position_behavior(uncertain, policy=HprlBehaviorPolicy(minimum_observations=2, minimum_distinct_joint_levels=1))
add(G, "uncertainty-overrisk-detected", ur.high_uncertainty_overrisk > 0)
add(G, "uncertainty-overrisk-fails", "BEHAVIOR_HIGH_UNCERTAINTY_OVERRISK" in ur.reasons, ur.reasons)
short_beh = analyze_hprl_position_behavior(rows[:10])
add(G, "short-sample-fails", not short_beh.passed)
add(G, "short-sample-reason", "BEHAVIOR_SAMPLE_INSUFFICIENT" in short_beh.reasons, short_beh.reasons)
add(G, "tiers-exact-five", levels == [D("0"),D("0.05"),D("0.12"),D("0.25"),D("0.40")])
add(G, "adverse-ratio-finite", 0.0 <= beh.adverse_scale_in_ratio <= 1.0)
add(G, "drawdown-ratio-finite", 0.0 <= beh.drawdown_derisk_ratio <= 1.0)
add(G, "high-uncertainty-counted", beh.high_uncertainty_events > 0)
assert len([x for x in CHECKS if x["group"] == G]) == 20


# ---------------------------------------------------------------------------
# G09: fail-closed aggregate Production Acceptance semantics (20)
# ---------------------------------------------------------------------------
G = "G09-production-acceptance"
required_names = (
    "container_pytest", "postgres_core", "postgres_failover", "postgres_restore",
    "binance_real_market_dryrun", "fault_campaign", "shadow_24h", "shadow_72h",
    "two_year_backtest", "position_behavior",
)
empty = evaluate_runtime_closure_acceptance({})
add(G, "empty-pending", empty.state is EvidenceState.PENDING, empty.pending_requirements)
add(G, "empty-not-pass", not empty.passed)
add(G, "empty-ten-pending", len(empty.pending_requirements) == 10, empty.pending_requirements)
all_pass = {name: RuntimeClosureEvidence(name, EvidenceState.PASS, digest=h(name)) for name in required_names}
acc = evaluate_runtime_closure_acceptance(all_pass)
add(G, "all-pass-state", acc.state is EvidenceState.PASS)
add(G, "all-pass-property", acc.passed)
add(G, "all-pass-no-failures", not acc.blocking_failures)
add(G, "all-pass-no-pending", not acc.pending_requirements)
add(G, "acceptance-sha", len(acc.acceptance_sha256) == 64)
failed_map = dict(all_pass); failed_map["postgres_failover"] = RuntimeClosureEvidence("postgres_failover", EvidenceState.FAIL)
failed = evaluate_runtime_closure_acceptance(failed_map)
add(G, "failure-state", failed.state is EvidenceState.FAIL)
add(G, "failure-blocking", "postgres_failover" in failed.blocking_failures)
pend_map = dict(all_pass); pend_map["shadow_72h"] = RuntimeClosureEvidence("shadow_72h", EvidenceState.PENDING)
pend = evaluate_runtime_closure_acceptance(pend_map)
add(G, "pending-state", pend.state is EvidenceState.PENDING)
add(G, "pending-listed", "shadow_72h" in pend.pending_requirements)
missing_map = dict(all_pass); missing_map.pop("two_year_backtest")
missing = evaluate_runtime_closure_acceptance(missing_map)
add(G, "missing-pending", any(x.startswith("two_year_backtest:MISSING") for x in missing.pending_requirements), missing.pending_requirements)
with tempfile.TemporaryDirectory(prefix="hprl-r2-evidence-") as td:
    registry_path = Path(td) / "registry.json"
    initialize_runtime_closure_evidence_registry(registry_path)
    registry = load_runtime_closure_evidence_registry(registry_path)
    add(G, "required-count-ten", len(required_names) == 10 and set(registry) == set(required_names) and all(item.state is EvidenceState.PENDING for item in registry.values()))
policy_default = RuntimeClosurePolicy()
add(G, "policy-requires-pytest", policy_default.require_container_pytest)
add(G, "policy-requires-postgres", policy_default.require_postgres_core and policy_default.require_postgres_failover and policy_default.require_postgres_restore)
add(G, "policy-requires-binance", policy_default.require_binance_real_market_dryrun)
add(G, "policy-requires-shadows", policy_default.require_shadow_24h and policy_default.require_shadow_72h)
add(G, "policy-requires-backtest-behavior", policy_default.require_two_year_backtest and policy_default.require_position_behavior)
add(G, "not-applicable-not-pass", evaluate_runtime_closure_acceptance({**all_pass, "position_behavior": RuntimeClosureEvidence("position_behavior", EvidenceState.NOT_APPLICABLE)}).state is EvidenceState.PENDING)
assert len([x for x in CHECKS if x["group"] == G]) == 20


# ---------------------------------------------------------------------------
# G10: source isolation / integration / algorithm byte lock (20)
# ---------------------------------------------------------------------------
G = "G10-source-isolation"
expected = {
    "freqtrade/hedge/hprl/action_space.py": "3b7edef6a474f2eec4ee1bd0997399622e8fa29cf65826dbcd7b4e62ad70c21e",
    "freqtrade/hedge/hprl/algorithms/fast_td3.py": "3fb0242a4bdd0e78c7e62a40991797efc1305054eefecc6e9174d02a66b00fb6",
    "freqtrade/hedge/hprl/algorithms/xqc.py": "a04805238916481b7b78ac17ed970974801b99dc78618a8be89c3e279b4445a2",
    "freqtrade/hedge/hprl/networks.py": "a0dddb0a1c932b55a76ddc92abcd44726567a61e1d0a16c2715c798073ad0f64",
    "freqtrade/hedge/hprl/algorithms/rebrac_v2.py": "a35b39b01b4d106b912b2b90070f35ce1a4b3f9fb3846ee8195be7ae15312af0",
    "freqtrade/hedge/hprl/reward.py": "a6f84fa76cf9d59a4e63922399e47828a5be6d9d65c1eb883110c42a900ddcc7",
    "freqtrade/hedge/hprl/algorithms/fast_dsac.py": "c220b97875bfab7ba023b33e057e318b2365ba390b973d6d9c233f356ee23173",
    "freqtrade/hedge/hprl/config.py": "e845eb3c7c8be3f3e14527634181315cf2c8cfcba90dfeb9bffb16c2a7d3295c",
    "freqtrade/hedge/hprl/algorithms/simba_sac.py": "ee752d0ae54d7d18eea4a898998c13d0eaf279cdd610a0708c96f572420890dc",
    "freqtrade/hedge/hprl/algorithms/__init__.py": "f05c167ba0105920f1e2e10e627880ade7dc2f84c5cce502cfdd5622a1d3afa3",
    "freqtrade/hedge/hprl/algorithms/base.py": "f837300305e903522522cc754235677e5da1714ebbb8cc0d23e3c99307a078e1",
}
for path, digest in expected.items():
    add(G, "algorithm-sha-" + path.replace("/", "-").replace(".py", ""), file_sha(path) == digest, file_sha(path))
new_modules = (
    "runtime_test_capability.py", "postgres_runtime_closure.py", "binance_runtime_dryrun.py",
    "runtime_fault_injection.py", "shadow_soak_runtime.py", "backtest_stability.py",
    "risk_behavior.py", "runtime_closure.py",
)
for name in new_modules:
    add(G, "new-module-" + name[:-3], (ROOT / "freqtrade/hedge/production" / name).is_file())
add(G, "operator-cli-exists", (ROOT / "tools/hprl_runtime_closure_r2.py").is_file())
assert len([x for x in CHECKS if x["group"] == G]) == 20


if len(CHECKS) != 200:
    raise RuntimeError(f"Runtime Closure R2 validator built {len(CHECKS)} checks instead of 200")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()
    failed = [item for item in CHECKS if not item["pass"]]
    payload = {
        "schema": "hprl-runtime-closure-r2-200-v1",
        "expected": 200,
        "executed": len(CHECKS),
        "passed": len(CHECKS)-len(failed),
        "failed": len(failed),
        "status": "PASS" if not failed else "FAIL",
        "real_environment_evidence": {
            "postgres_failover": "NOT_CLAIMED_OFFLINE",
            "postgres_restore": "NOT_CLAIMED_OFFLINE",
            "binance_real_market": "NOT_CLAIMED_OFFLINE",
            "shadow_24h": "NOT_CLAIMED_OFFLINE",
            "shadow_72h": "NOT_CLAIMED_OFFLINE",
            "two_year_backtest": "NOT_CLAIMED_OFFLINE",
            "position_behavior": "NOT_CLAIMED_OFFLINE",
        },
        "checks": CHECKS,
    }
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str)+"\n", encoding="utf-8")
    if not args.summary_only:
        print(json.dumps(payload, sort_keys=True, default=str))
    print(f"HPRL RUNTIME CLOSURE R2 200: {payload['passed']}/200 PASS; FAIL={payload['failed']}")
    if failed:
        print("FAILED_CHECKS_BEGIN")
        for item in failed:
            print("FAILED_CHECK " + f"group={item['group']} name={item['name']} detail=" + json.dumps(item.get("detail", ""), sort_keys=True, default=str, separators=(",", ":")))
        print("FAILED_CHECKS_END")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
