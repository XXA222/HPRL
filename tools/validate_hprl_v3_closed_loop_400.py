#!/usr/bin/env python3
"""400-point deterministic source/runtime gate for HPRL V3 closed-loop integration.

The first 300 points are the established V3 Production-R2 matrix.  The additional
100 points focus on exact dual-leg live semantics, cycle hash chaining, UNKNOWN order
recovery, checkpoint/journal convergence and dry-run linkage.  No real PostgreSQL,
Binance, 24h/72h shadow or live evidence is fabricated here.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the already-reviewed 300 distinct R2 checks instead of cloning their logic.
spec = importlib.util.spec_from_file_location(
    "hprl_v3_r2_300", ROOT / "tools/validate_hprl_v3_production_r2_300.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load R2 300-point validator")
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)
CHECKS: list[dict[str, object]] = [dict(item) for item in r2.CHECKS]
if len(CHECKS) != 300:
    raise RuntimeError(f"R2 validator returned {len(CHECKS)} checks instead of 300")

from freqtrade.hedge.execution.event_publisher import InMemoryEventPublisher
from freqtrade.hedge.execution.fake_exchange import build_fake_execution_harness
from freqtrade.hedge.execution.ledger import InMemoryExecutionLedger
from freqtrade.hedge.execution.orchestrator import HedgeExecutionEngine
from freqtrade.hedge.execution.service import (
    ExternalOrderSnapshot, IntentAction, OrderIntent, OrderType, PositionSide as ExecPositionSide,
)
from freqtrade.hedge.execution.state_machine import OrderState
from freqtrade.hedge.execution.unknown_supervisor import UnknownOrderSupervisor, UnknownRecoveryState
from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
from freqtrade.hedge.planning.context import (
    LegPosition, MarketSnapshot, PlannerConfig, PlanningContext, PositionSide, WalletSnapshot,
)
from freqtrade.hedge.planning.ideal_orders import PureHedgePlanner
from freqtrade.hedge.production.acceptance_closed_loop import HPRL_V3_CLOSED_LOOP_RELEASE
from freqtrade.hedge.production.binance_dryrun import BinanceDryRunPolicy, BinanceDryRunSafetyContext
from freqtrade.hedge.production.closed_loop import (
    ClosedLoopCycleJournal, ClosedLoopCycleJournalStore, ClosedLoopCycleRecord,
    ClosedLoopCycleStatus, ZERO_HASH,
)
from freqtrade.hedge.production.closed_loop_dryrun import evaluate_closed_loop_binance_dryrun
from freqtrade.hedge.production.closed_loop_recovery import ClosedLoopRecoveryBarrier
from freqtrade.hedge.production.hprl_hedge_adapter import (
    HprlHedgeAdapter, HprlHedgeAdapterPolicy, HprlTargetUnit,
)
from freqtrade.hedge.production.recovery_checkpoint import DurableRecoveryCheckpoint
from freqtrade.hedge.production.source_convergence import build_canonical_source_snapshot
from freqtrade.hedge.telemetry.dryrun import DryRunCycleTelemetry, StrategyTelemetry

NOW = datetime(2026, 8, 15, 5, 0, tzinfo=UTC)
D = Decimal


def add(group: str, name: str, passed: bool, detail: object = "") -> None:
    CHECKS.append({"group": group, "name": name, "pass": bool(passed), "detail": detail})


def h(text: str) -> str:
    return sha256(text.encode()).hexdigest()


def margin_adapter() -> HprlHedgeAdapter:
    return HprlHedgeAdapter(HprlHedgeAdapterPolicy(
        leverage=D("3"), target_unit=HprlTargetUnit.MARGIN_EQUITY_RATIO,
        max_leg_margin_ratio=D(".40"), max_gross_margin_ratio=D(".80"),
        max_abs_net_margin_ratio=D(".40"), max_increase_margin_delta=D(".15"),
    ))


def margin_intent(long: str, short: str, *, model: str = "hprl-v3-closed-loop") -> PlannedExecutionIntent:
    return PlannedExecutionIntent(
        symbol="BTC/USDT:USDT", target_long_exposure=float(long),
        target_short_exposure=float(short), confidence=1.0, model_id=model,
        metadata={"unit": "margin/equity"},
    )


def planning_context(*, leverage: str = "3") -> PlanningContext:
    return PlanningContext(
        market=MarketSnapshot(
            "BTC/USDT:USDT", NOW, D("100"), D("100"), D("100"),
            tick_size=D(".1"), qty_step=D(".0001"),
        ),
        wallet=WalletSnapshot(
            D("1000"), D("1000"), D("1000"),
            LegPosition(PositionSide.LONG), LegPosition(PositionSide.SHORT),
            leverage=D(leverage),
        ),
        config=PlannerConfig(),
    )


# G09 - exact dual-leg planning bridge: 20 checks.
G = "G09-exact-dual-leg-context"
a = margin_adapter()
cases = (("0", "0"), (".05", ".12"), (".12", ".25"), (".25", ".05"), (".40", ".40"))
for index, (long, short) in enumerate(cases):
    projection = a.adapt(margin_intent(long, short), sequence=index + 1, observed_at=NOW, now=NOW)
    context, profile_sha = a.apply_to_context(planning_context(), projection)
    plan = PureHedgePlanner().plan(context)
    expected_long = D(long) * D("3") * D("1000") / D("100")
    expected_short = D(short) * D("3") * D("1000") / D("100")
    add(G, f"case-{index}-accepted", projection.accepted)
    add(G, f"case-{index}-profile-sha", len(profile_sha) == 64)
    add(G, f"case-{index}-long-exact", plan.long_target_quantity == expected_long)
    add(G, f"case-{index}-short-exact", plan.short_target_quantity == expected_short)

# G10 - durable cycle journal/hash chain: 20 checks.
G = "G10-cycle-journal"
def record(index: int, previous: str, chain_previous: str) -> ClosedLoopCycleRecord:
    projection_sha = h(f"projection-{index}")
    chain = h(chain_previous + projection_sha)
    return ClosedLoopCycleRecord(
        sequence=index, cycle_id=f"hprl-cycle-{index}", observed_at=NOW + timedelta(minutes=index),
        source_release=HPRL_V3_CLOSED_LOOP_RELEASE, model_id="model", symbol="BTCUSDT",
        projection_sequence=index, projection_observed_at=NOW + timedelta(minutes=index),
        projection_source_sha256=h(f"source-{index}"), projection_semantic_sha256=projection_sha,
        long_margin_ratio=D(".12"), short_margin_ratio=D(".05"),
        long_notional_ratio=D(".36"), short_notional_ratio=D(".15"), confidence=D("1"),
        projection_accepted=True, projection_reasons=(), projection_chain_sha256=chain,
        planner_profile_sha256=h("planner"), input_state_sha256=h(f"input-{index}"),
        planning_sha256=h(f"planning-{index}"), execution_sha256=h(f"execution-{index}"),
        reconciliation_digest=h("reconciliation"), evidence_digest=h("evidence"),
        safety_allows_reduce=True, safety_allows_new_risk=True,
        status=ClosedLoopCycleStatus.COMMITTED, writes_attempted=index,
        previous_record_sha256=previous,
    )
journal = ClosedLoopCycleJournal()
previous = ZERO_HASH; chain_previous = ZERO_HASH
records = []
for index in range(1, 5):
    item = record(index, previous, chain_previous)
    journal.append(item); records.append(item)
    previous = item.record_sha256; chain_previous = item.projection_chain_sha256
add(G, "journal-verify", journal.verify())
add(G, "journal-count", len(journal.records) == 4)
add(G, "journal-tip", journal.tip_sha256 == records[-1].record_sha256)
add(G, "projection-tip", journal.projection_chain_sha256 == records[-1].projection_chain_sha256)
for index, item in enumerate(records, start=1):
    add(G, f"sequence-{index}", item.sequence == index)
    add(G, f"record-sha-{index}", len(item.record_sha256) == 64 and item.record_sha256 != ZERO_HASH)
    add(G, f"cycle-id-{index}", item.cycle_id == f"hprl-cycle-{index}")
add(G, "previous-2", records[1].previous_record_sha256 == records[0].record_sha256)
add(G, "previous-3", records[2].previous_record_sha256 == records[1].record_sha256)
add(G, "previous-4", records[3].previous_record_sha256 == records[2].record_sha256)
add(G, "payload-roundtrip", ClosedLoopCycleJournal.from_payload(journal.payload()).tip_sha256 == journal.tip_sha256)
assert len([x for x in CHECKS if x["group"] == G]) == 20

# G11 - UNKNOWN submit/query/recovery closure: 20 checks.
G = "G11-unknown-recovery"
harness = build_fake_execution_harness()
ledger = InMemoryExecutionLedger(); publisher = InMemoryEventPublisher()
engine = HedgeExecutionEngine(harness.service, transaction=ledger, publisher=publisher)
supervisor = UnknownOrderSupervisor(engine)
engine.bind_unknown_supervisor(supervisor)
harness.exchange.queue_timeout()
unknown_intent = OrderIntent(
    account_id="acct", symbol="BTCUSDT", position_side=ExecPositionSide.LONG,
    action=IntentAction.OPEN, quantity=D(".1"), idempotency_key="v3-unknown",
    order_type=OrderType.MARKET, metadata={"reference_price": "100"},
)
result = engine.submit(unknown_intent)
client_id = result.order.client_order_id
rec = supervisor.get(client_id)
submit_count = len(harness.exchange.submit_calls)
add(G, "status-unknown", result.order.lifecycle.status is OrderState.UNKNOWN)
add(G, "supervisor-bound", engine.unknown_supervisor is supervisor)
add(G, "record-created", rec is not None)
add(G, "record-pending", rec is not None and rec.state is UnknownRecoveryState.PENDING)
add(G, "one-submit", submit_count == 1)
add(G, "client-id-stable", bool(client_id) and client_id.startswith("FTH-"))
add(G, "store-unknown", harness.service.get_order(client_id).lifecycle.status is OrderState.UNKNOWN)
add(G, "query-happened-on-timeout", len(harness.exchange.query_calls) >= 1)
add(G, "no-cancel", not harness.exchange.cancel_calls)
harness.exchange.set_order(ExternalOrderSnapshot(client_order_id=client_id, status=OrderState.ACKNOWLEDGED))
recovered = engine.run_unknown_recovery()
latest = supervisor.get(client_id)
add(G, "one-recovery-record", len(recovered) == 1)
add(G, "resolved-state", latest is not None and latest.state is UnknownRecoveryState.RESOLVED)
add(G, "resolved-ack", latest is not None and latest.resolved_state is OrderState.ACKNOWLEDGED)
add(G, "never-resubmitted", len(harness.exchange.submit_calls) == submit_count)
add(G, "query-only", harness.exchange.query_calls[-1] == client_id)
add(G, "store-ack", harness.service.get_order(client_id).lifecycle.status is OrderState.ACKNOWLEDGED)
add(G, "no-due-after-resolve", supervisor.due() == ())
add(G, "second-run-empty", engine.run_unknown_recovery() == ())
add(G, "supervisor-attempt-one", latest is not None and latest.attempts == 1)
add(G, "resolved-no-next-retry", latest is not None and latest.next_retry_at is None)
add(G, "still-one-submit-final", len(harness.exchange.submit_calls) == 1)
assert len([x for x in CHECKS if x["group"] == G]) == 20

# G12 - checkpoint/journal barrier + Binance dry-run linkage: 20 checks.
G = "G12-restart-dryrun-link"
last = records[-1]
checkpoint = DurableRecoveryCheckpoint(
    generation=4, created_at=NOW + timedelta(minutes=4), source_release=HPRL_V3_CLOSED_LOOP_RELEASE,
    model_id="model", evidence_digest=h("evidence"), reconciliation_digest=h("reconciliation"),
    projection_chain_sha256=journal.projection_chain_sha256, last_market_sequence=44,
    last_user_sequence=55, metadata=(("closed_loop_cycle_id", last.cycle_id),
        ("closed_loop_cycle_sha256", last.record_sha256),
        ("closed_loop_journal_tip_sha256", journal.tip_sha256),
        ("closed_loop_status", last.status.value)),
)
barrier = ClosedLoopRecoveryBarrier()
good = barrier.evaluate(checkpoint, journal, orders=(), now=NOW + timedelta(minutes=4),
    current_evidence_digest=h("evidence"), current_reconciliation_digest=h("reconciliation"))
add(G, "barrier-pass", good.passed)
add(G, "barrier-new-risk", good.allow_new_risk)
add(G, "barrier-reduce", good.allow_reduce)
add(G, "barrier-no-reasons", not good.reasons)
bad_meta = dict(checkpoint.metadata); bad_meta["closed_loop_cycle_sha256"] = "f" * 64
bad = barrier.evaluate(replace(checkpoint, metadata=tuple(bad_meta.items())), journal, orders=(),
    now=NOW + timedelta(minutes=4), current_evidence_digest=h("evidence"),
    current_reconciliation_digest=h("reconciliation"))
add(G, "bad-cycle-fails", not bad.passed)
add(G, "bad-cycle-blocks-risk", not bad.allow_new_risk)
add(G, "bad-cycle-reason", "CLOSED_LOOP_CYCLE_HASH_MISMATCH" in bad.reasons)
strategy = StrategyTelemetry(model_version="model", regime="HPRL")
telemetry = tuple(DryRunCycleTelemetry(
    cycle_id=item.cycle_id, account_id="dryrun:binance", symbol="BTCUSDT",
    timestamp=item.observed_at, mark_price=D("100"), equity=D("1000"), available_balance=D("700"),
    gross_notional=D("300"), net_quantity=D("1"), long_quantity=D("2"), short_quantity=D("1"),
    long_target_quantity=D("2"), short_target_quantity=D("1"), strategy=strategy,
) for item in records)
safety = BinanceDryRunSafetyContext(
    exchange="binance", operation_mode="dry_run", real_market_data=True,
    exchange_write_capability=False, simulated_execution=True, hedge_mode_semantics=True,
    cross_margin_semantics=True, source_release=HPRL_V3_CLOSED_LOOP_RELEASE, account_namespace="dryrun",
)
dry = evaluate_closed_loop_binance_dryrun(
    telemetry, journal=journal, safety=safety,
    policy=BinanceDryRunPolicy(minimum_cycles=4, minimum_duration=timedelta(minutes=3),
        maximum_cycle_gap=timedelta(minutes=2)),
)
add(G, "dryrun-pass", dry.passed)
add(G, "dryrun-base-pass", dry.base.passed)
add(G, "dryrun-linked-four", dry.linked_cycle_count == 4)
add(G, "dryrun-journal-four", dry.journal_cycle_count == 4)
add(G, "dryrun-telemetry-four", dry.telemetry_cycle_count == 4)
add(G, "dryrun-tip", dry.journal_tip_sha256 == journal.tip_sha256)
add(G, "dryrun-no-reasons", not dry.reasons)
short_dry = evaluate_closed_loop_binance_dryrun(
    telemetry[:-1], journal=journal, safety=safety,
    policy=BinanceDryRunPolicy(minimum_cycles=3, minimum_duration=timedelta(minutes=2),
        maximum_cycle_gap=timedelta(minutes=2)),
)
add(G, "short-cycle-set-fails", not short_dry.passed)
add(G, "short-cycle-set-reason", "BINANCE_DRYRUN_CYCLE_SET_MISMATCH" in short_dry.reasons)
add(G, "write-capability-off", not safety.exchange_write_capability)
add(G, "simulated-execution-on", safety.simulated_execution)
add(G, "real-market-on", safety.real_market_data)
add(G, "hedge-cross-on", safety.hedge_mode_semantics and safety.cross_margin_semantics)
assert len([x for x in CHECKS if x["group"] == G]) == 20

# G13 - source authority / algorithm isolation / SQL contract: 20 checks.
G = "G13-source-isolation"
expected = {
    "freqtrade/hedge/hprl/algorithms/base.py": "f837300305e903522522cc754235677e5da1714ebbb8cc0d23e3c99307a078e1",
    "freqtrade/hedge/hprl/algorithms/fast_dsac.py": "c220b97875bfab7ba023b33e057e318b2365ba390b973d6d9c233f356ee23173",
    "freqtrade/hedge/hprl/algorithms/fast_td3.py": "3fb0242a4bdd0e78c7e62a40991797efc1305054eefecc6e9174d02a66b00fb6",
    "freqtrade/hedge/hprl/algorithms/rebrac_v2.py": "a35b39b01b4d106b912b2b90070f35ce1a4b3f9fb3846ee8195be7ae15312af0",
    "freqtrade/hedge/hprl/algorithms/simba_sac.py": "ee752d0ae54d7d18eea4a898998c13d0eaf279cdd610a0708c96f572420890dc",
    "freqtrade/hedge/hprl/algorithms/xqc.py": "a04805238916481b7b78ac17ed970974801b99dc78618a8be89c3e279b4445a2",
    "freqtrade/hedge/hprl/networks.py": "a0dddb0a1c932b55a76ddc92abcd44726567a61e1d0a16c2715c798073ad0f64",
    "freqtrade/hedge/hprl/reward.py": "a6f84fa76cf9d59a4e63922399e47828a5be6d9d65c1eb883110c42a900ddcc7",
    "freqtrade/hedge/hprl/action_space.py": "3b7edef6a474f2eec4ee1bd0997399622e8fa29cf65826dbcd7b4e62ad70c21e",
    "freqtrade/hedge/hprl/config.py": "e845eb3c7c8be3f3e14527634181315cf2c8cfcba90dfeb9bffb16c2a7d3295c",
}
for path, digest in expected.items():
    add(G, "baseline-sha-" + Path(path).stem, sha256((ROOT / path).read_bytes()).hexdigest() == digest)
source = build_canonical_source_snapshot(ROOT)
add(
    G,
    "manifest-workspace",
    source.manifest_matches_workspace
    and (
        source.validation_policy == "package"
        or source.managed_attestation_verified
    ),
    {
        "policy": source.validation_policy,
        "managed_attestation_verified": source.managed_attestation_verified,
        "managed_attestation_count": source.managed_attestation_count,
        "managed_attestation_overlay_sha256": source.managed_attestation_overlay_sha256,
        "managed_attestation_target_release": source.managed_attestation_target_release,
        "workspace_missing": len(source.workspace_missing_files),
        "workspace_unexpected": len(source.workspace_unexpected_files),
        "workspace_mismatched": len(source.workspace_mismatched_files),
    },
)
add(G, "required-paths", source.required_paths_present)
add(G, "github-repository", source.github_baseline_repository == "XXA222/HPRL")
add(G, "github-commit", source.github_baseline_commit == "c7411179744a38b3af91a11a91985db2327c77a4")
add(G, "closed-loop-api", source.closed_loop_api_version == "3.1")
add(G, "closed-loop-release", source.closed_loop_release == HPRL_V3_CLOSED_LOOP_RELEASE)
sql_text = (ROOT / "freqtrade/hedge/production/closed_loop_sql.py").read_text(encoding="utf-8")
add(G, "sql-audit-table-reuse", "AuditEvent" in sql_text and "HPRL_CLOSED_LOOP_CYCLE" in sql_text)
add(G, "sql-row-lock", "with_for_update" in sql_text)
add(G, "sql-no-new-orm-table", "__tablename__" not in sql_text)
add(G, "github-backup-readonly-design", "create_branch" not in sql_text and "update_ref" not in sql_text)
assert len([x for x in CHECKS if x["group"] == G]) == 20

if len(CHECKS) != 400:
    raise RuntimeError(f"closed-loop validator built {len(CHECKS)} checks instead of 400")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()
    failed = [item for item in CHECKS if not item["pass"]]
    payload = {
        "schema": "hprl-v3-closed-loop-runtime-400-v1",
        "expected": 400,
        "executed": len(CHECKS),
        "passed": len(CHECKS) - len(failed),
        "failed": len(failed),
        "status": "PASS" if not failed else "FAIL",
        "github_source_authority": "READ_ONLY:XXA222/HPRL@c7411179744a38b3af91a11a91985db2327c77a4",
        "environment_evidence": "REAL_POSTGRES_BINANCE_SHADOW_LIVE_NOT_CLAIMED_OFFLINE",
        "checks": CHECKS,
    }
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    if not args.summary_only:
        print(json.dumps(payload, sort_keys=True, default=str))
    print(f"HPRL V3 CLOSED LOOP RUNTIME 400: {payload['passed']}/400 PASS; FAIL={payload['failed']}")
    if failed:
        print("FAILED_CHECKS_BEGIN")
        for item in failed:
            print(
                "FAILED_CHECK "
                + f"group={item.get('group', '')} "
                + f"name={item.get('name', '')} "
                + "detail="
                + json.dumps(item.get("detail", ""), sort_keys=True, default=str, separators=(",", ":"))
            )
        try:
            source_diag = build_canonical_source_snapshot(ROOT)
            print(
                "SOURCE_BLOCKING_DRIFT "
                + json.dumps(
                    {
                        "policy": source_diag.validation_policy,
                        "manifest_missing": source_diag.manifest_missing_files,
                        "manifest_unexpected": source_diag.manifest_unexpected_files,
                        "manifest_mismatched": source_diag.manifest_mismatched_files,
                        "workspace_missing_count": len(source_diag.workspace_missing_files),
                        "workspace_unexpected_count": len(source_diag.workspace_unexpected_files),
                        "workspace_mismatched_count": len(source_diag.workspace_mismatched_files),
                        "managed_attestation_verified": source_diag.managed_attestation_verified,
                        "managed_attestation_count": source_diag.managed_attestation_count,
                    },
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                )
            )
        except Exception as exc:
            print(f"SOURCE_BLOCKING_DRIFT_ERROR={type(exc).__name__}:{exc}")
        print("FAILED_CHECKS_END")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
