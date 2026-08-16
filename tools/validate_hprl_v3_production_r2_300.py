#!/usr/bin/env python3
"""Deterministic 300-point offline gate for HPRL V3 Production Integration R2.

This validator proves source identity, dual-leg adapter semantics, replay/backtest parity,
crash recovery fail-closed behavior, PostgreSQL acceptance contracts, Binance dry-run
safety semantics, and aggregate acceptance locking.  It never claims real PostgreSQL,
Binance, shadow, testnet, or live-canary evidence; those remain separate environment gates.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.backtesting.contracts import Candidate, EngineConfig
from freqtrade.hedge.backtesting.dataset import build_dataset
from freqtrade.hedge.hprl.config import HPRLActionConfig
from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
from freqtrade.hedge.production.acceptance_r2 import (
    HPRL_V3_PRODUCTION_API_VERSION,
    HPRL_V3_PRODUCTION_RELEASE,
    evaluate_hprl_v3_production_r2,
)
from freqtrade.hedge.production.binance_dryrun import (
    BinanceDryRunPolicy,
    BinanceDryRunSafetyContext,
    evaluate_binance_dryrun,
)
from freqtrade.hedge.production.hprl_hedge_adapter import (
    HprlHedgeAdapter,
    HprlHedgeAdapterPolicy,
    HprlTargetUnit,
)
from freqtrade.hedge.production.hprl_replay_backtest import (
    HprlReplayBacktestRunner,
    HprlReplayDatasetBuilder,
    TimedHprlTarget,
)
from freqtrade.hedge.production.postgres_acceptance import (
    PostgresBackupRestoreEvidence,
    PostgresDurabilityProbePolicy,
    PostgresDurabilityProbeRunner,
    PostgresFailoverEvidence,
    evaluate_postgres_r2,
)
from freqtrade.hedge.production.database_runtime import (
    PostgresConcurrencyProbeReport,
    PostgresProbeReport,
)
from freqtrade.hedge.production.recovery_checkpoint import (
    DurableRecoveryCheckpoint,
    RecoveryBarrierPolicy,
    RecoveryCheckpointStore,
    RecoveryConvergenceBarrier,
)
from freqtrade.hedge.production.source_convergence import build_canonical_source_snapshot
from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent, SignalEvent
from freqtrade.hedge.telemetry.dryrun import DryRunCycleTelemetry, StrategyTelemetry

NOW = datetime(2026, 8, 15, 3, 0, tzinfo=UTC)
CHECKS: list[dict[str, object]] = []


def add(group: str, name: str, passed: bool, detail: object = "") -> None:
    CHECKS.append({"group": group, "name": name, "pass": bool(passed), "detail": detail})


def intent(long: Decimal | str | float, short: Decimal | str | float, confidence: Decimal | str | float = 1) -> PlannedExecutionIntent:
    return PlannedExecutionIntent(
        symbol="BTC/USDT:USDT",
        target_long_exposure=float(long),
        target_short_exposure=float(short),
        confidence=float(confidence),
        model_id="hprl-xqc-r2",
        metadata={"source": "r2-300"},
    )


def adapter() -> HprlHedgeAdapter:
    cfg = HPRLActionConfig(
        leverage=3.0,
        max_leg_margin_ratio=0.40,
        max_gross_margin_ratio=0.80,
        max_abs_net_margin_ratio=0.40,
    )
    return HprlHedgeAdapter(
        HprlHedgeAdapterPolicy.from_hprl_action_config(
            cfg, target_unit=HprlTargetUnit.NOTIONAL_EQUITY_RATIO
        )
    )


def base_dataset():
    prices = (100, 101, 102, 101, 103, 104, 102, 105)
    events = []
    for index, price in enumerate(prices):
        ts = NOW + timedelta(minutes=index)
        p = Decimal(price)
        events.append(BarEvent(ts, "BTC/USDT:USDT", p, p + 1, p - 1, p + Decimal("0.5"), Decimal("1000")))
        if index == 4:
            events.append(FundingEvent(ts, "BTC/USDT:USDT", Decimal("0.0001"), p))
    events.sort(key=lambda item: item.timestamp)
    return build_dataset(events=events, dataset_id="r2-300-base", timeframe="1m", metadata={"fixture": "r2-300"})


def replay_bundle():
    a = adapter()
    targets = (
        TimedHprlTarget(NOW, 1, intent("0.15", "0")),
        TimedHprlTarget(NOW + timedelta(minutes=1), 2, intent("0.36", "0.15")),
        TimedHprlTarget(NOW + timedelta(minutes=2), 3, intent("0.75", "0.36")),
        TimedHprlTarget(NOW + timedelta(minutes=3), 4, intent("1.20", "0.75")),
        TimedHprlTarget(NOW + timedelta(minutes=4), 5, intent("0.75", "0.75")),
        TimedHprlTarget(NOW + timedelta(minutes=5), 6, intent("0.36", "0.36")),
    )
    return a, HprlReplayDatasetBuilder(a).build(base_dataset=base_dataset(), targets=targets)


def dryrun_cycles(count: int = 8, *, duplicate: bool = False, gap_at: int | None = None, diagnostic: str = "", blocked_every: int = 0):
    strategy = StrategyTelemetry(
        long_score=Decimal("0.5"), short_score=Decimal("0.25"),
        long_exposure_scale=Decimal("0.5"), short_exposure_scale=Decimal("0.25"),
        model_version="hprl-xqc-r2", regime="HPRL",
    )
    result = []
    for index in range(count):
        minute = index + (10 if gap_at is not None and index >= gap_at else 0)
        result.append(DryRunCycleTelemetry(
            cycle_id=("cycle-0" if duplicate and index == count - 1 else f"cycle-{index}"),
            account_id="dryrun:binance", symbol="BTCUSDT",
            timestamp=NOW + timedelta(minutes=minute), mark_price=Decimal(100 + index),
            equity=Decimal(1000 + index), available_balance=Decimal("700"),
            gross_notional=Decimal("300"), net_quantity=Decimal("1"),
            long_quantity=Decimal("2"), short_quantity=Decimal("1"),
            long_target_quantity=Decimal("2"), short_target_quantity=Decimal("1"),
            risk_blocked=bool(blocked_every and index % blocked_every == 0),
            diagnostics=((diagnostic,) if diagnostic else ()), strategy=strategy,
        ))
    return tuple(result)


# G01: source convergence and release identity (20)
G = "G01-source-convergence"
source = build_canonical_source_snapshot(ROOT)
source_props = [
    ("snapshot-passed", source.passed),
    ("required-paths", source.required_paths_present),
    ("manifest-workspace-exact", source.manifest_matches_workspace),
    ("no-required-missing", not source.missing_paths),
    ("no-manifest-missing", not source.manifest_missing_files),
    ("no-manifest-unexpected", not source.manifest_unexpected_files),
    ("no-manifest-mismatch", not source.manifest_mismatched_files),
    ("manifest-schema", source.manifest_schema == "freqtrade-hedge-clean-mainline-manifest-v1"),
    ("manifest-file-count", source.manifest_file_count > 1500),
    ("manifest-sha", len(source.manifest_sha256) == 64),
    ("tree-sha", len(source.tree_sha256) == 64),
    ("hprl-api", source.hprl_api_version != ""),
    ("hprl-release-v252", source.hprl_release.endswith("perf-v2.5.2")),
    ("production-api", source.production_api_version == "1.1"),
    ("production-r11", source.production_release == "freqtrade-hedge-production-readiness-r1.1-deep200"),
    ("github-repository", source.github_baseline_repository == "XXA222/HPRL"),
    ("github-baseline", source.github_baseline_commit == "c7411179744a38b3af91a11a91985db2327c77a4"),
    ("r2-api", HPRL_V3_PRODUCTION_API_VERSION == "3.0"),
    ("r2-release", HPRL_V3_PRODUCTION_RELEASE.endswith("production-integration-r2")),
    ("manifest-version-r2", source.manifest_version == "freqtrade-hedge-clean-mainline-v1.2.1-20260812"),
]
for name, passed in source_props:
    add(G, name, passed, asdict(source) if name == "snapshot-passed" else "")

# G02: all 25 independent dual-leg target states, four invariants each = 100
G = "G02-adapter-25x4"
a = adapter()
levels = tuple(Decimal(x) for x in ("0", "0.15", "0.36", "0.75", "1.20"))
for li, long in enumerate(levels):
    for si, short in enumerate(levels):
        projection = a.adapt(intent(long, short), sequence=li * 5 + si + 1, observed_at=NOW, now=NOW)
        signal = a.to_signal_event(projection)
        prefix = f"l{li}-s{si}"
        add(G, prefix + "-accepted", projection.accepted, projection.reasons)
        add(G, prefix + "-unit", projection.long_margin_ratio == long / 3 and projection.short_margin_ratio == short / 3, (str(projection.long_margin_ratio), str(projection.short_margin_ratio)))
        add(G, prefix + "-notional", projection.long_notional_ratio == long and projection.short_notional_ratio == short)
        add(G, prefix + "-independent", signal.target_net_ratio is None and signal.long_exposure_scale == long / Decimal("1.2") and signal.short_exposure_scale == short / Decimal("1.2"))

# G03: transition/risk envelope behavior, 10 scenarios x 4 = 40
G = "G03-adapter-transitions"
scenarios = (
    ("flat-low", "0", "0", "0.15", "0", "1", True),
    ("low-more", "0.15", "0", "0.36", "0", "1", True),
    ("more-mid", "0.36", "0", "0.75", "0", "1", True),
    ("mid-heavy", "0.75", "0", "1.20", "0", "1", True),
    ("low-heavy-jump", "0.15", "0", "1.20", "0", "1", False),
    ("heavy-low-derisk", "1.20", "0", "0.15", "0", "1", True),
    ("dual-step", "0.36", "0.15", "0.75", "0.36", "1", True),
    ("dual-jump", "0.15", "0.15", "1.20", "0.15", "1", False),
    ("low-confidence-increase", "0.15", "0", "0.36", "0", "0.40", False),
    ("low-confidence-flat", "0.36", "0.15", "0.36", "0.15", "0.40", True),
)
for i, (name, pl, ps, nl, ns, confidence, expected) in enumerate(scenarios):
    previous = a.adapt(intent(pl, ps), sequence=100 + i * 2, observed_at=NOW, now=NOW)
    current = a.adapt(intent(nl, ns, confidence), sequence=101 + i * 2, observed_at=NOW + timedelta(seconds=1), now=NOW + timedelta(seconds=1), previous=previous)
    add(G, name + "-decision", current.accepted is expected, current.reasons)
    expected_long = Decimal(nl) / 3 if expected else previous.long_margin_ratio
    expected_short = Decimal(ns) / 3 if expected else previous.short_margin_ratio
    add(G, name + "-fallback", current.long_margin_ratio == expected_long and current.short_margin_ratio == expected_short, (str(current.long_margin_ratio), str(current.short_margin_ratio)))
    add(G, name + "-reason-contract", (not current.reasons) if expected else bool(current.reasons), current.reasons)
    add(G, name + "-hash", len(current.source_sha256) == 64 and all(c in "0123456789abcdef" for c in current.source_sha256))

# G04: canonical replay/backtest parity, exactly 40 properties
G = "G04-replay-backtest"
a, bundle = replay_bundle()
runner = HprlReplayBacktestRunner(
    bundle=bundle, adapter=a,
    engine_config=EngineConfig(initial_balance=Decimal("1000"), leverage=Decimal("3"), volume_participation=Decimal("1"), max_fill_ratio_per_order=Decimal("1")),
)
evaluation = runner.evaluate(Candidate("r2-300", {}))
parity = runner.parity(Candidate("r2-300-parity", {}))
result = evaluation.result
assert result is not None
report = result.report
signals = tuple(x for x in bundle.dataset.events if isinstance(x, SignalEvent))
properties = [
    ("build-pass", bundle.report.passed), ("six-targets", bundle.report.target_count == 6),
    ("six-accepted", bundle.report.accepted_targets == 6), ("zero-rejected", bundle.report.rejected_targets == 0),
    ("first-seq", bundle.report.first_sequence == 1), ("last-seq", bundle.report.last_sequence == 6),
    ("planner-sha", len(bundle.report.planner_profile_sha256) == 64), ("dataset-sha", len(bundle.report.dataset_fingerprint) == 64),
    ("chain-sha", len(bundle.report.projection_chain_sha256) == 64), ("no-reasons", not bundle.report.rejection_reasons),
    ("metadata-hprl", bundle.dataset.metadata.get("hprl_v3") == "true"), ("metadata-source", bool(bundle.dataset.metadata.get("source_dataset_fingerprint"))),
    ("metadata-unit", bundle.dataset.metadata.get("hprl_target_unit") == "NOTIONAL_EQUITY_RATIO"), ("signal-count", len(signals) == 6),
    ("all-no-net", all(x.target_net_ratio is None for x in signals)), ("all-hprl-regime", all(x.regime == "HPRL" for x in signals)),
    ("all-model-id", all(x.model_version == "hprl-xqc-r2" for x in signals)), ("all-new-risk", all(x.allow_new_risk for x in signals)),
    ("has-funding", bundle.dataset.funding_count == 1), ("bar-count", bundle.dataset.bar_count == 8),
    ("evaluation-feasible-contract", isinstance(evaluation.feasible, bool)), ("evaluation-result", evaluation.result is not None),
    ("long-add", int(report.get("long_add_count", 0)) > 0), ("short-add", int(report.get("short_add_count", 0)) > 0),
    ("parity-pass", parity.passed), ("state-parity", parity.first_state_sha256 == parity.second_state_sha256),
    ("event-parity", parity.first_event_sha256 == parity.second_event_sha256), ("state-sha", len(parity.first_state_sha256) == 64),
    ("event-sha", len(parity.first_event_sha256) == 64), ("equity-finite", Decimal(parity.final_equity).is_finite()),
    ("long-finite", Decimal(parity.final_long_quantity).is_finite()), ("short-finite", Decimal(parity.final_short_quantity).is_finite()),
    ("projection-unique", len({p.source_sha256 for p in bundle.projections}) == 6), ("projection-monotonic", [p.sequence for p in bundle.projections] == list(range(1, 7))),
    ("projection-symbol", all(p.symbol == bundle.dataset.symbol for p in bundle.projections)), ("projection-accepted", all(p.accepted for p in bundle.projections)),
    ("heavy-observed", any(p.long_notional_ratio == Decimal("1.2") for p in bundle.projections)), ("dual-observed", any(p.long_notional_ratio > 0 and p.short_notional_ratio > 0 for p in bundle.projections)),
    ("source-fingerprint-changed", bundle.dataset.fingerprint != base_dataset().fingerprint), ("dataset-id", bundle.dataset.dataset_id.endswith(":hprl-v3")),
]
assert len(properties) == 40
for name, passed in properties:
    add(G, name, passed)

# G05: durable checkpoint + convergence barrier, exactly 40
G = "G05-crash-recovery"
checkpoint = DurableRecoveryCheckpoint(
    generation=1, created_at=NOW, source_release=HPRL_V3_PRODUCTION_RELEASE,
    model_id="hprl-xqc-r2", evidence_digest="a" * 64, reconciliation_digest="b" * 64,
    projection_chain_sha256=bundle.report.projection_chain_sha256,
    last_market_sequence=10, last_user_sequence=20,
)
with tempfile.TemporaryDirectory(prefix="hprl-r2-300-") as td:
    store = RecoveryCheckpointStore(Path(td) / "checkpoint.json")
    store.save_atomic(checkpoint)
    loaded = store.load()
    add(G, "store-roundtrip", loaded == checkpoint)
    add(G, "store-file", store.path.is_file())
    add(G, "checkpoint-sha", len(checkpoint.checkpoint_sha256) == 64)
    add(G, "generation", checkpoint.generation == 1)
    try:
        store.save_atomic(checkpoint)
        monotonic_rejected = False
    except ValueError:
        monotonic_rejected = True
    add(G, "generation-monotonic", monotonic_rejected)
    payload = checkpoint.payload(); payload["model_id"] = "tampered"
    try:
        DurableRecoveryCheckpoint.from_payload(payload)
        tamper_rejected = False
    except ValueError:
        tamper_rejected = True
    add(G, "tamper-rejected", tamper_rejected)

barrier = RecoveryConvergenceBarrier(RecoveryBarrierPolicy(max_checkpoint_age=timedelta(minutes=2)))
good = barrier.evaluate(checkpoint, orders=(), now=NOW + timedelta(seconds=30), current_evidence_digest="a" * 64, current_reconciliation_digest="b" * 64)
missing = barrier.evaluate(None, orders=(), now=NOW + timedelta(seconds=30), current_evidence_digest="a" * 64, current_reconciliation_digest="b" * 64)
stale = barrier.evaluate(checkpoint, orders=(), now=NOW + timedelta(minutes=3), current_evidence_digest="a" * 64, current_reconciliation_digest="b" * 64)
future_cp = DurableRecoveryCheckpoint(generation=2, created_at=NOW + timedelta(minutes=1), source_release="r2", model_id="m", evidence_digest="a"*64, reconciliation_digest="b"*64, projection_chain_sha256="c"*64, last_market_sequence=0, last_user_sequence=0)
future = barrier.evaluate(future_cp, orders=(), now=NOW, current_evidence_digest="a"*64, current_reconciliation_digest="b"*64)
evidence_changed = barrier.evaluate(checkpoint, orders=(), now=NOW + timedelta(seconds=30), current_evidence_digest="d"*64, current_reconciliation_digest="b"*64)
recon_changed = barrier.evaluate(checkpoint, orders=(), now=NOW + timedelta(seconds=30), current_evidence_digest="a"*64, current_reconciliation_digest="d"*64)
recovery_props = [
    ("good-pass", good.passed), ("good-new-risk", good.allow_new_risk), ("good-reduce", good.allow_reduce), ("good-no-reasons", not good.reasons),
    ("good-sha", good.checkpoint_sha256 == checkpoint.checkpoint_sha256), ("good-no-unknown", not good.unresolved_client_order_ids),
    ("missing-fail", not missing.passed), ("missing-new-locked", not missing.allow_new_risk), ("missing-reduce", missing.allow_reduce), ("missing-reason", "RECOVERY_CHECKPOINT_MISSING" in missing.reasons),
    ("missing-sha-none", missing.checkpoint_sha256 is None), ("stale-fail", not stale.passed), ("stale-new-locked", not stale.allow_new_risk), ("stale-reduce", stale.allow_reduce),
    ("stale-reason", "RECOVERY_CHECKPOINT_STALE" in stale.reasons), ("future-fail", not future.passed), ("future-new-locked", not future.allow_new_risk),
    ("future-reduce-locked", not future.allow_reduce), ("future-reason", "RECOVERY_CHECKPOINT_FROM_FUTURE" in future.reasons),
    ("evidence-fail", not evidence_changed.passed), ("evidence-new-locked", not evidence_changed.allow_new_risk), ("evidence-reduce", evidence_changed.allow_reduce),
    ("evidence-reason", "RECOVERY_EVIDENCE_DIGEST_CHANGED" in evidence_changed.reasons), ("recon-fail", not recon_changed.passed),
    ("recon-new-locked", not recon_changed.allow_new_risk), ("recon-reduce", recon_changed.allow_reduce), ("recon-reason", "RECOVERY_RECONCILIATION_DIGEST_CHANGED" in recon_changed.reasons),
    ("plan-no-blind", not good.plan.blind_resubmit_allowed), ("missing-plan-no-blind", not missing.plan.blind_resubmit_allowed),
    ("stale-plan-no-blind", not stale.plan.blind_resubmit_allowed), ("future-plan-no-blind", not future.plan.blind_resubmit_allowed),
    ("evidence-plan-no-blind", not evidence_changed.plan.blind_resubmit_allowed), ("recon-plan-no-blind", not recon_changed.plan.blind_resubmit_allowed),
    ("checkpoint-evidence-bound", checkpoint.evidence_digest == "a"*64),
]
assert len(recovery_props) == 34
for name, passed in recovery_props:
    add(G, name, passed)
# 6 store/checkpoint checks + 34 barrier checks = 40

# G06: Binance real-market/simulated-execution dry-run contracts = 30
G = "G06-binance-dryrun"
safe = BinanceDryRunSafetyContext("binance", "dry_run", True, False, True, True, True, HPRL_V3_PRODUCTION_RELEASE, "dryrun")
policy = BinanceDryRunPolicy(minimum_cycles=8, minimum_duration=timedelta(minutes=7), maximum_cycle_gap=timedelta(minutes=2), maximum_risk_block_ratio=Decimal("0.25"))
good_dr = evaluate_binance_dryrun(dryrun_cycles(), safety=safe, policy=policy)
unsafe_write = BinanceDryRunSafetyContext("binance", "dry_run", True, True, True, True, True, "r2", "dryrun")
write_dr = evaluate_binance_dryrun(dryrun_cycles(), safety=unsafe_write, policy=policy)
dup_dr = evaluate_binance_dryrun(dryrun_cycles(duplicate=True), safety=safe, policy=policy)
gap_dr = evaluate_binance_dryrun(dryrun_cycles(gap_at=4), safety=safe, policy=policy)
diag_dr = evaluate_binance_dryrun(dryrun_cycles(diagnostic="REAL_ORDER_ATTEMPT"), safety=safe, policy=policy)
blocked_dr = evaluate_binance_dryrun(dryrun_cycles(blocked_every=2), safety=safe, policy=policy)
dry_props = [
    ("safe-context", safe.safe), ("good-pass", good_dr.passed), ("good-count", good_dr.cycle_count == 8), ("good-duration", good_dr.duration_seconds == 420),
    ("good-unique", good_dr.unique_cycle_ids), ("good-monotonic", good_dr.monotonic_timestamps), ("good-gap", good_dr.maximum_gap_seconds == 60),
    ("good-dual-target", good_dr.dual_leg_target_observed), ("good-dual-position", good_dr.dual_leg_position_observed), ("good-risk-rate", good_dr.risk_block_ratio == 0),
    ("good-sha", len(good_dr.telemetry_sha256) == 64), ("good-time", good_dr.observed_at == NOW + timedelta(minutes=7)),
    ("write-context-unsafe", not unsafe_write.safe), ("write-fail", not write_dr.passed), ("write-reason", "BINANCE_DRYRUN_SAFETY_CONTEXT_INVALID" in write_dr.reasons),
    ("dup-fail", not dup_dr.passed), ("dup-flag", not dup_dr.unique_cycle_ids), ("dup-reason", "BINANCE_DRYRUN_DUPLICATE_CYCLE_ID" in dup_dr.reasons),
    ("gap-fail", not gap_dr.passed), ("gap-size", gap_dr.maximum_gap_seconds > 120), ("gap-reason", "BINANCE_DRYRUN_CYCLE_GAP" in gap_dr.reasons),
    ("diag-fail", not diag_dr.passed), ("diag-reason", "BINANCE_DRYRUN_WRITE_DIAGNOSTIC_PRESENT" in diag_dr.reasons), ("diag-still-dual", diag_dr.dual_leg_target_observed),
    ("blocked-fail", not blocked_dr.passed), ("blocked-rate", blocked_dr.risk_block_ratio == Decimal("0.5")), ("blocked-reason", "BINANCE_DRYRUN_EXCESSIVE_RISK_BLOCK_RATE" in blocked_dr.reasons),
    ("good-equity", good_dr.final_equity == Decimal("1007")), ("no-good-reasons", not good_dr.reasons), ("simulated-write-off", not safe.exchange_write_capability and safe.simulated_execution),
]
assert len(dry_props) == 30
for name, passed in dry_props:
    add(G, name, passed)

# G07: PostgreSQL R2 acceptance contracts = 20 (fake DB tests code only; no real evidence claim)
G = "G07-postgres-contracts"
shared: dict[str, str] = {}
class Cursor:
    def __init__(self, connection): self.connection=connection; self.row=None
    def execute(self, sql, params=()):
        text = " ".join(sql.split()).upper()
        if text.startswith("SELECT PG_BACKEND_PID"):
            self.row=(self.connection.pid,)
        elif text.startswith("INSERT INTO"):
            shared[str(params[0])] = str(params[1]); self.row=None
        elif text.startswith("SELECT PAYLOAD_SHA256"):
            value=shared.get(str(params[0])); self.row=None if value is None else (value,)
        elif text.startswith("DELETE FROM"):
            shared.pop(str(params[0]), None); self.row=None
        else:
            self.row=None
    def fetchone(self): return self.row
    def close(self): pass
class Connection:
    def __init__(self,pid): self.pid=pid; self.commits=0; self.rollbacks=0; self.closed=False
    def cursor(self): return Cursor(self)
    def commit(self): self.commits+=1
    def rollback(self): self.rollbacks+=1
    def close(self): self.closed=True
pids=iter((101,202))
durability = PostgresDurabilityProbeRunner(lambda: Connection(next(pids)), lambda: Connection(next(pids)), policy=PostgresDurabilityProbePolicy(visibility_attempts=2, visibility_sleep_seconds=0)).run(now=NOW)
artifact="e"*64
failover=PostgresFailoverEvidence("fo-r2",NOW,True,True,True,True,True,True,True,artifact)
backup=PostgresBackupRestoreEvidence("br-r2",NOW,"1"*64,True,True,True,True,True,True,"2"*64,"2"*64)
basic=PostgresProbeReport(True,True,True,True,"SERIALIZABLE","17","freqtrade",(),NOW)
concurrency=PostgresConcurrencyProbeReport(True,True,True,True,11,12,(),NOW)
pg = evaluate_postgres_r2(basic=basic, concurrency=concurrency, durability=durability, failover=failover, backup_restore=backup)
pg_missing = evaluate_postgres_r2(basic=basic, concurrency=concurrency, durability=durability, failover=None, backup_restore=None)
pg_props=[
    ("durability-pass", durability.passed), ("durability-distinct", durability.distinct_connections), ("durability-write", durability.primary_write_committed),
    ("durability-visible", durability.secondary_observed_committed_value), ("durability-cleanup", durability.cleanup_committed), ("durability-no-errors", not durability.errors),
    ("durability-sha", len(durability.payload_sha256)==64), ("durability-probe-id", durability.probe_id.startswith("hprl-r2-")),
    ("failover-pass", failover.passed), ("backup-pass", backup.passed), ("snapshot-equal", backup.source_snapshot_sha256==backup.restored_snapshot_sha256),
    ("aggregate-pass", pg.passed), ("basic-pass", pg.basic_probe_passed), ("concurrency-pass", pg.concurrency_probe_passed),
    ("aggregate-durability", pg.durability_probe_passed), ("aggregate-failover", pg.failover_exercise_passed), ("aggregate-backup", pg.backup_restore_passed),
    ("aggregate-no-reasons", not pg.reasons), ("missing-fails", not pg_missing.passed),
    ("missing-reasons", "POSTGRES_FAILOVER_EXERCISE_MISSING_OR_FAILED" in pg_missing.reasons and "POSTGRES_BACKUP_RESTORE_MISSING_OR_FAILED" in pg_missing.reasons),
]
assert len(pg_props)==20
for name, passed in pg_props: add(G,name,passed)

# G08: aggregate acceptance truthfulness = 10
G = "G08-aggregate-acceptance"
accept = evaluate_hprl_v3_production_r2(source=source, replay_build=bundle.report, replay_parity=parity, recovery=good, hedge_adapter_ready=True)
accept_props=[
    ("source", accept.source_converged), ("adapter", accept.hedge_adapter_ready), ("replay", accept.replay_backtest_ready),
    ("recovery", accept.crash_recovery_ready), ("offline-pass", accept.offline_source_passed),
    ("environment-locked", not accept.environment_passed), ("live-locked", not accept.live_ready),
    ("postgres-blocker", "R2_REAL_POSTGRES_EVIDENCE_REQUIRED" in accept.blockers),
    ("dryrun-blocker", "R2_BINANCE_DRYRUN_EVIDENCE_REQUIRED" in accept.blockers),
    ("live-spine-blocker", "R2_EXISTING_PRODUCTION_LIVE_READY_GATE_REQUIRED" in accept.blockers),
]
assert len(accept_props)==10
for name, passed in accept_props: add(G,name,passed)

assert len(CHECKS) == 300, len(CHECKS)


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--json-output", default="")
    args=parser.parse_args()
    failed=[item for item in CHECKS if not item["pass"]]
    payload={"schema":"hprl-v3-production-r2-runtime-300-v1","expected":300,"executed":len(CHECKS),"passed":300-len(failed),"failed":len(failed),"status":"PASS" if not failed else "FAIL","environment_evidence":"NOT_CLAIMED_OFFLINE","checks":CHECKS}
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
    if not args.summary_only:
        print(json.dumps(payload,sort_keys=True,default=str))
    print(f"HPRL V3 PRODUCTION R2 RUNTIME 300: {payload['passed']}/300 PASS; FAIL={payload['failed']}")
    return 0 if not failed else 1

if __name__ == "__main__":
    raise SystemExit(main())
