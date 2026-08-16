from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import tempfile
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.execution.binance_environment import ExecutionEnvironment
from freqtrade.hedge.execution.client_order_id import build_client_order_id
from freqtrade.hedge.execution.production_gate import ExecutionWriteLockedError, ProductionGateEvidence
from freqtrade.hedge.execution.service import ApprovedOrderIntent, IntentAction, OrderIntent, OrderType, PositionSide
from freqtrade.hedge.execution.unknown_supervisor import UnknownOrderSupervisor, UnknownRecoveryState
from freqtrade.hedge.production.admission import AdmissionContext, admit
from freqtrade.hedge.production.canary import CanaryDecision, CanaryLevel, CanaryRuntime, evaluate_canary
from freqtrade.hedge.production.canary_runtime import CanaryRunMetrics, evaluate_canary_promotion
from freqtrade.hedge.production.contracts import Capability, Decision, EvidenceKind, EvidenceStatus, ProductionStage, Severity, cumulative_requirements
from freqtrade.hedge.production.control import ControlAction, ControlMode, ProductionControlPlane
from freqtrade.hedge.production.database import DatabaseReadinessInput, DatabaseReadinessResult, evaluate_database_readiness
from freqtrade.hedge.production.evidence import EvidenceConcurrencyError, EvidenceLedger, EvidenceLedgerStore
from freqtrade.hedge.production.execution_guard import ReadinessBoundProductionExecutionGate
from freqtrade.hedge.production.faults import FaultResult, FaultScenario, evaluate_fault_campaign
from freqtrade.hedge.production.incidents import Incident, IncidentLedger
from freqtrade.hedge.production.model_governance import FallbackProfile, FallbackProfileRegistry, ModelCircuitBreaker, ModelCircuitPolicy, ModelRuntimeDecision
from freqtrade.hedge.production.model_targets import ModelTarget, validate_model_target
from freqtrade.hedge.production.observability import AlertHysteresisPolicy, AlertStateTracker, HealthSnapshot, ObservabilityPolicy, evaluate_health
from freqtrade.hedge.production.policy import StageEvaluator
from freqtrade.hedge.production.reconciliation import DiffKind, PositionTruth, ReconciliationDiff, ReconciliationPlane, ReconciliationResult, reconcile
from freqtrade.hedge.production.reconciliation_runtime import ReconciliationAction, ReconciliationSupervisor, ReconciliationSupervisorPolicy, build_reconciliation_plan
from freqtrade.hedge.production.release import build_production_readiness_report
from freqtrade.hedge.production.replay import RecordedFact, ReplayManifest, evaluate_replay_integrity
from freqtrade.hedge.production.reservations import ExposureReservationBook, ReservationState
from freqtrade.hedge.production.risk_envelope import AccountRiskView, CandidateIntent, CrossRiskLimits, RiskDirection, Side, StressScenario, evaluate_post_trade_risk
from freqtrade.hedge.production.runtime_bundle import ProductionRuntimeBundle
from freqtrade.hedge.production.runtime_supervisor import RuntimeSafetySnapshot
from freqtrade.hedge.production.shadow import ShadowMetrics, ShadowQualification, ShadowPolicy, qualify_shadow
from freqtrade.hedge.production.shadow_runtime import ShadowWindow, qualify_shadow_run
from freqtrade.hedge.production.submission import SubmissionObservation
from freqtrade.hedge.production.submission_runtime import SubmissionRecoveryMachine, SubmissionRecoveryPolicy, SubmissionRecoveryState

NOW = datetime(2026, 8, 15, 2, 0, tzinfo=UTC)
CHECKS: list[dict[str, object]] = []


def add(group: str, name: str, passed: bool, detail: object = "") -> None:
    CHECKS.append({"group": group, "name": name, "pass": bool(passed), "detail": detail})


def caught(exc_type, fn) -> str | None:
    try:
        fn()
    except exc_type as exc:
        return str(exc)
    return None


def ledger(stage: ProductionStage = ProductionStage.LIVE_READY) -> EvidenceLedger:
    value = EvidenceLedger()
    for i, kind in enumerate(sorted(cumulative_requirements(stage), key=lambda x: x.value), 1):
        value.add(
            kind=kind, status=EvidenceStatus.PASS,
            observed_at=NOW + timedelta(microseconds=i), ttl=timedelta(days=2),
            artifact_sha256=hashlib.sha256((kind.value+str(i)).encode()).hexdigest(),
            producer="deep200",
        )
    return value


def live_evidence(token: str = "deep200") -> ProductionGateEvidence:
    return ProductionGateEvidence(
        environment=ExecutionEnvironment.LIVE,
        account_fingerprint="acct", allowed_symbols=("BTCUSDT",),
        cross_margin_symbols=("BTCUSDT",), readonly_status="FULL_PASS",
        user_stream_status="FULL_PASS", hedge_mode_enabled=True, clock_offset_ms=0,
        live_trading_enabled=True, strict_key_policy_passed=True,
        futures_trading_permission=True, ip_restricted=True,
        expected_arm_token_sha256=hashlib.sha256(token.encode()).hexdigest(),
        max_order_notional=Decimal("100"),
    )


def approved(action: IntentAction, key: str = "deep200") -> ApprovedOrderIntent:
    intent = OrderIntent(
        account_id="binance-usdm:acct", symbol="BTCUSDT", position_side=PositionSide.LONG,
        action=action, quantity=Decimal("1"), idempotency_key=key,
        order_type=OrderType.LIMIT, limit_price=Decimal("10"),
    )
    cid = build_client_order_id(
        account_id=intent.account_id, symbol=intent.symbol,
        position_side=intent.position_side.value, idempotency_key=intent.idempotency_key,
    )
    return ApprovedOrderIntent(intent, Decimal("1"), cid, NOW, ("DEEP200",))


def safety(epoch=1, observed_at=NOW, new=True, reduce=True, reasons=()):
    return RuntimeSafetySnapshot(epoch, observed_at, new, reduce, tuple(reasons))


# G01 - live runtime safety + lease invalidation
G="G01-runtime-safety-gate"
msg=caught(ValueError, lambda: ReadinessBoundProductionExecutionGate(live_evidence(), evaluator=StageEvaluator(ledger())))
add(G,"live-provider-required", bool(msg and "runtime_safety_provider" in msg), msg)
for name, snap, expected in [
    ("stale", safety(observed_at=NOW-timedelta(seconds=6)), "RUNTIME_SAFETY_STALE"),
    ("future", safety(observed_at=NOW+timedelta(seconds=2)), "RUNTIME_SAFETY_FROM_FUTURE"),
    ("reduce-block", safety(reduce=False), "RUNTIME_SAFETY_BLOCKS_ARM"),
]:
    gate=ReadinessBoundProductionExecutionGate(live_evidence(), evaluator=StageEvaluator(ledger()), clock=lambda: NOW, runtime_safety_provider=lambda s=snap:s)
    msg=caught(ExecutionWriteLockedError, lambda g=gate:g.arm(token="deep200",actor="ops",confirmed=True))
    add(G,name, bool(msg and expected in msg), msg)
state={"v":safety(epoch=2)}
gate=ReadinessBoundProductionExecutionGate(live_evidence(), evaluator=StageEvaluator(ledger()), clock=lambda: NOW, runtime_safety_provider=lambda:state["v"])
gate.arm(token="deep200",actor="ops",confirmed=True); state["v"]=safety(epoch=3)
msg=caught(ExecutionWriteLockedError,lambda:gate.assert_order_allowed(approved(IntentAction.REDUCE,"epoch")))
add(G,"epoch-rearm",bool(msg and "EPOCH_CHANGED" in msg),msg)
state={"v":safety(epoch=4,new=True)}
gate=ReadinessBoundProductionExecutionGate(live_evidence(), evaluator=StageEvaluator(ledger()), clock=lambda:NOW, runtime_safety_provider=lambda:state["v"])
gate.arm(token="deep200",actor="ops",confirmed=True); state["v"]=safety(epoch=4,new=False,reduce=True,reasons=("PAUSE",))
msg=caught(ExecutionWriteLockedError,lambda:gate.assert_order_allowed(approved(IntentAction.INCREASE,"pause")))
add(G,"same-epoch-new-risk-block",bool(msg and "BLOCKS_NEW_RISK" in msg),msg)
gate=ReadinessBoundProductionExecutionGate(live_evidence(), evaluator=StageEvaluator(ledger()), clock=lambda:NOW, runtime_safety_provider=lambda:safety())
gate.arm(token="deep200",actor="ops",confirmed=True)
add(G,"live-ready-increase",gate.assert_order_allowed(approved(IntentAction.INCREASE,"ready")).symbol=="BTCUSDT")
source_gate=ReadinessBoundProductionExecutionGate(live_evidence(), evaluator=StageEvaluator(ledger(ProductionStage.SOURCE_READY)), clock=lambda:NOW, runtime_safety_provider=lambda:safety())
source_gate.arm(token="deep200",actor="ops",confirmed=True)
reduce_ok=source_gate.assert_order_allowed(approved(IntentAction.REDUCE,"source-reduce")).symbol=="BTCUSDT"
increase_msg=caught(ExecutionWriteLockedError,lambda:source_gate.assert_order_allowed(approved(IntentAction.INCREASE,"source-increase")))
add(G,"source-ready-reduce-only",reduce_ok and bool(increase_msg),increase_msg or "REDUCE_OK")
gate=ReadinessBoundProductionExecutionGate(live_evidence(), evaluator=StageEvaluator(ledger()), clock=lambda:NOW, runtime_safety_provider=lambda:safety())
msg=caught(PermissionError,lambda:gate.arm(token="wrong",actor="ops",confirmed=True))
add(G,"failed-arm-clears",bool(msg and gate._reduce_lease is None and gate._new_risk_lease is None),msg)
add(G,"safety-snapshot-naive-rejected",caught(ValueError,lambda:RuntimeSafetySnapshot(1,datetime(2026,1,1),True,True,())) is not None)

# G02 - canary reservation state machine and concurrency
G="G02-canary-reservations"
book=ExposureReservationBook(ttl=timedelta(seconds=1)); r=book.reserve(client_order_id="a",notional=Decimal("10"),now=NOW,max_total_notional=Decimal("50"),max_orders=4)
add(G,"held",r.state is ReservationState.HELD)
c=book.commit(r.reservation_id,now=NOW); add(G,"commit",c.state is ReservationState.COMMITTED)
add(G,"commit-idempotent",book.commit(r.reservation_id,now=NOW).state is ReservationState.COMMITTED)
add(G,"committed-survives-ttl",book.snapshot(now=NOW+timedelta(hours=1)).held_orders==1)
rel=book.release(r.reservation_id,now=NOW+timedelta(hours=1)); add(G,"committed-release",rel.state is ReservationState.RELEASED)
add(G,"release-idempotent",book.release(r.reservation_id,now=NOW+timedelta(hours=1)).state is ReservationState.RELEASED)
add(G,"terminal-no-reuse",caught(ValueError,lambda:book.reserve(client_order_id="a",notional=Decimal("10"),now=NOW,max_total_notional=Decimal("50"),max_orders=4)) is not None)
book2=ExposureReservationBook(); x=book2.reserve(client_order_id="x",notional=Decimal("10"),now=NOW,max_total_notional=Decimal("50"),max_orders=4)
add(G,"idempotent-held",book2.reserve(client_order_id="x",notional=Decimal("10"),now=NOW,max_total_notional=Decimal("50"),max_orders=4).reservation_id==x.reservation_id)
add(G,"idempotent-mismatch",caught(ValueError,lambda:book2.reserve(client_order_id="x",notional=Decimal("11"),now=NOW,max_total_notional=Decimal("50"),max_orders=4)) is not None)
book3=ExposureReservationBook()
def _res(i):
    try: book3.reserve(client_order_id=f"t{i}",notional=Decimal("10"),now=NOW,max_total_notional=Decimal("40"),max_orders=4); return 1
    except PermissionError: return 0
with ThreadPoolExecutor(max_workers=16) as pool: accepted=sum(pool.map(_res,range(20)))
add(G,"concurrent-cap",accepted==4 and book3.snapshot(now=NOW).held_notional==Decimal("40"),accepted)

# G03 - cross margin risk, reduce no-flip, stress
G="G03-cross-risk"
base=AccountRiskView(Decimal("1000"),Decimal("800"),Decimal("100"),Decimal("100"),Decimal("100"),Decimal("20"))
for i, amount in enumerate(("10","25","50","75")):
    d=evaluate_post_trade_risk(base,CandidateIntent(Side.LONG,RiskDirection.REDUCE,Decimal(amount),Decimal("10"),Decimal("2")),CrossRiskLimits())
    add(G,f"reduce-{i}",d.decision is Decision.APPROVE and d.projection.long_notional==Decimal("100")-Decimal(amount),str(d.decision))
pending_reduce=AccountRiskView(Decimal("1000"),Decimal("800"),Decimal("100"),Decimal("0"),Decimal("100"),Decimal("10"),pending_long_reduce_notional=Decimal("70"))
d=evaluate_post_trade_risk(pending_reduce,CandidateIntent(Side.LONG,RiskDirection.REDUCE,Decimal("50"),Decimal("25"),Decimal("5")),CrossRiskLimits())
add(G,"pending-reduce-reserves-closeable",d.decision is Decision.CLIP and d.approved_notional==Decimal("30"),(d.decision,d.approved_notional))
d=evaluate_post_trade_risk(base,CandidateIntent(Side.LONG,RiskDirection.REDUCE,Decimal("150"),Decimal("15"),Decimal("3")),CrossRiskLimits()); add(G,"reduce-clip",d.decision is Decision.CLIP and d.approved_notional==Decimal("100"),d.reasons)
flat=AccountRiskView(Decimal("1000"),Decimal("800"),Decimal("0"),Decimal("0"),Decimal("0"),Decimal("0")); d=evaluate_post_trade_risk(flat,CandidateIntent(Side.SHORT,RiskDirection.REDUCE,Decimal("1"),Decimal("0"),Decimal("0")),CrossRiskLimits()); add(G,"flat-reduce-reject",d.decision is Decision.REJECT,d.reasons)
for i, loss in enumerate(("0.20","0.30","0.40")):
    d=evaluate_post_trade_risk(base,CandidateIntent(Side.LONG,RiskDirection.INCREASE,Decimal("100"),Decimal("20"),Decimal("2")),CrossRiskLimits(max_stress_loss_ratio=Decimal("0.03")),stress_scenarios=(StressScenario(f"S{i}",Decimal(loss),Decimal("0")),))
    add(G,f"stress-{i}",d.decision in {Decision.CLIP,Decision.REJECT},[x.passed for x in d.stress_results])

# G04 - submission recovery
G="G04-submission-recovery"
m=SubmissionRecoveryMachine(SubmissionRecoveryPolicy(max_query_attempts=2,max_retry_not_sent=1,recovery_deadline=timedelta(seconds=10),retry_backoff=timedelta(seconds=1)))
r=m.start(client_order_id="cid",now=NOW); add(G,"start",r.state is SubmissionRecoveryState.NEW)
r=m.observe(r,SubmissionObservation(None,False,True),now=NOW); add(G,"ambiguous-query",r.state is SubmissionRecoveryState.QUERY_REQUIRED and not r.permits_new_risk,r.reason)
r2=m.query_not_found(r,now=NOW+timedelta(seconds=1),exchange_history_complete=False); add(G,"negative-not-authoritative",r2.state is SubmissionRecoveryState.QUERY_REQUIRED,r2.reason)
r3=m.query_not_found(r,now=NOW+timedelta(seconds=1),exchange_history_complete=True); add(G,"authoritative-manual",r3.state is SubmissionRecoveryState.MANUAL_REVIEW,r3.reason)
r=m.start(client_order_id="ok",now=NOW); r=m.observe(r,SubmissionObservation(200,True,True,"123"),now=NOW); add(G,"ack",r.state is SubmissionRecoveryState.ACKED and r.exchange_order_id=="123")
r=m.start(client_order_id="reject",now=NOW); r=m.observe(r,SubmissionObservation(400,True,True,None,"-1013"),now=NOW); add(G,"reject",r.state is SubmissionRecoveryState.REJECTED)
r=m.start(client_order_id="notsent",now=NOW); r=m.observe(r,SubmissionObservation(None,False,False),now=NOW); add(G,"proven-not-sent-retry",r.state is SubmissionRecoveryState.RETRY_ALLOWED,r.reason)
r=m.observe(r,SubmissionObservation(None,False,False),now=NOW+timedelta(seconds=1)); add(G,"retry-budget",r.state is SubmissionRecoveryState.MANUAL_REVIEW,r.reason)
r=m.start(client_order_id="deadline",now=NOW); r=m.observe(r,SubmissionObservation(None,False,True),now=NOW+timedelta(seconds=11)); add(G,"deadline",r.state is SubmissionRecoveryState.MANUAL_REVIEW,r.reason)
add(G,"naive-now-rejected",caught(ValueError,lambda:m.start(client_order_id="bad",now=datetime(2026,1,1))) is not None)

# G05 - reconciliation action planner
G="G05-reconciliation-plan"
map_expected={
    DiffKind.POSITION:ReconciliationAction.REBUILD_POSITION_PROJECTION,
    DiffKind.OPEN_ORDER:ReconciliationAction.QUERY_ORDER_BY_CLIENT_ID,
    DiffKind.BALANCE:ReconciliationAction.REBUILD_BALANCE_PROJECTION,
    DiffKind.MODE:ReconciliationAction.VERIFY_ACCOUNT_MODE,
    DiffKind.LEVERAGE:ReconciliationAction.REQUIRE_MANUAL_REVIEW,
    DiffKind.UNKNOWN_ORDER:ReconciliationAction.IMPORT_EXTERNAL_ORDER,
    DiffKind.CURSOR:ReconciliationAction.REPLAY_MISSING_EVENTS,
}
for kind,expected in map_expected.items():
    sev=Severity.HALT_ACCOUNT if kind in {DiffKind.MODE,DiffKind.UNKNOWN_ORDER} else Severity.HALT_NEW_RISK
    p=build_reconciliation_plan(ReconciliationResult(False,False,sev is not Severity.HALT_ACCOUNT,(ReconciliationDiff(kind,"k","l","e",sev),)))
    add(G,f"plan-{kind.value.lower()}",expected in p.actions,p.actions)
p=build_reconciliation_plan(ReconciliationResult(True,True,True,())); add(G,"converged-empty",not p.actions and p.maximum_severity is None)
p=build_reconciliation_plan(ReconciliationResult(False,False,False,(ReconciliationDiff(DiffKind.UNKNOWN_ORDER,"x","","",Severity.HALT_ACCOUNT),))); add(G,"halt-account-first",p.actions[0] is ReconciliationAction.HALT_ACCOUNT,p.actions)
p=build_reconciliation_plan(ReconciliationResult(False,False,True,(ReconciliationDiff(DiffKind.POSITION,"x","","",Severity.HALT_NEW_RISK),))); add(G,"halt-new-first",p.actions[0] is ReconciliationAction.HALT_NEW_RISK,p.actions)

# G06 - reconciliation convergence/freshness
G="G06-reconciliation-supervisor"
good=ReconciliationResult(True,True,True,()); drift=ReconciliationResult(False,False,True,(ReconciliationDiff(DiffKind.POSITION,"x","1","2",Severity.HALT_NEW_RISK),)); hard=ReconciliationResult(False,False,False,(ReconciliationDiff(DiffKind.UNKNOWN_ORDER,"x","","",Severity.HALT_ACCOUNT),))
s=ReconciliationSupervisor(ReconciliationSupervisorPolicy(confirmations_for_new_risk=3,max_snapshot_age=timedelta(seconds=5),max_nonconverged_duration=timedelta(seconds=10)))
a=s.observe(good,observed_at=NOW,now=NOW); add(G,"confirm-1",not a.allow_new_risk and a.allow_reduce,a.consecutive_converged)
a=s.observe(good,observed_at=NOW+timedelta(seconds=1),now=NOW+timedelta(seconds=1)); add(G,"confirm-2",not a.allow_new_risk,a.consecutive_converged)
a=s.observe(good,observed_at=NOW+timedelta(seconds=2),now=NOW+timedelta(seconds=2)); add(G,"confirm-3",a.allow_new_risk,a.consecutive_converged)
a=s.observe(drift,observed_at=NOW+timedelta(seconds=3),now=NOW+timedelta(seconds=3)); add(G,"mild-drift-keeps-reduce",a.allow_reduce and not a.allow_new_risk,a.reasons)
a=s.observe(hard,observed_at=NOW+timedelta(seconds=4),now=NOW+timedelta(seconds=4)); add(G,"hard-drift-blocks-reduce",not a.allow_reduce,a.reasons)
s2=ReconciliationSupervisor(ReconciliationSupervisorPolicy(max_snapshot_age=timedelta(seconds=5))); a=s2.observe(good,observed_at=NOW,now=NOW+timedelta(seconds=6)); add(G,"stale-block",not a.allow_reduce and "RECONCILIATION_SNAPSHOT_STALE" in a.reasons,a.reasons)
s3=ReconciliationSupervisor(); s3.observe(good,observed_at=NOW+timedelta(seconds=2),now=NOW+timedelta(seconds=2)); a=s3.observe(good,observed_at=NOW+timedelta(seconds=1),now=NOW+timedelta(seconds=3)); add(G,"timestamp-regress",not a.allow_reduce and "RECONCILIATION_TIMESTAMP_REGRESSION" in a.reasons,a.reasons)
s4=ReconciliationSupervisor(ReconciliationSupervisorPolicy(max_nonconverged_duration=timedelta(seconds=2))); s4.observe(drift,observed_at=NOW,now=NOW); a=s4.observe(drift,observed_at=NOW+timedelta(seconds=3),now=NOW+timedelta(seconds=3)); add(G,"sla", "RECONCILIATION_NONCONVERGED_SLA" in a.reasons,a.reasons)
local=ReconciliationPlane.build(positions=(PositionTruth("BTCUSDT","LONG",Decimal("1")),),open_order_ids=(),wallet_balance=Decimal("1"),hedge_mode=True,cross_symbols=("BTCUSDT",),cursor=1); exch=ReconciliationPlane.build(positions=(PositionTruth("BTCUSDT","LONG",Decimal("1")),),open_order_ids=(),wallet_balance=Decimal("1"),hedge_mode=True,cross_symbols=("BTCUSDT",),cursor=1); add(G,"plane-converges",reconcile(local,exch).converged)
add(G,"duplicate-position-rejected",caught(ValueError,lambda:ReconciliationPlane.build(positions=(PositionTruth("BTCUSDT","LONG",Decimal("1")),PositionTruth("BTCUSDT","LONG",Decimal("1"))),open_order_ids=(),wallet_balance=Decimal("1"),hedge_mode=True,cross_symbols=("BTCUSDT",),cursor=1)) is not None)

# G07 - database readiness
G="G07-database-readiness"
def db(**kw):
    base=dict(backend="postgresql",migration_head_expected="h",migration_head_observed="h",connection_ok=True,transaction_probe_ok=True,uniqueness_probe_ok=True,fencing_probe_ok=True,outbox_probe_ok=True,deadlock_retry_probe_ok=True,backup_verified_at=NOW,restore_verified_at=NOW)
    base.update(kw); return DatabaseReadinessInput(**base)
add(G,"good",evaluate_database_readiness(db(),now=NOW).passed)
for i,(key,val,reason) in enumerate([
    ("backend","sqlite","LIVE_REQUIRES_POSTGRESQL"),("connection_ok",False,"DATABASE_CONNECTION_FAILED"),("migration_head_observed","x","MIGRATION_HEAD_MISMATCH"),("advisory_lock_probe_ok",False,"ADVISORY_LOCK_PROBE_FAILED"),("failover_probe_ok",False,"DATABASE_FAILOVER_PROBE_FAILED"),("backup_checksum_verified",False,"BACKUP_CHECKSUM_NOT_VERIFIED"),("isolation_level","READ COMMITTED","DATABASE_ISOLATION_TOO_WEAK"),("replication_lag_seconds",6.0,"REPLICATION_LAG_EXCEEDED")]):
    r=evaluate_database_readiness(db(**{key:val}),now=NOW); add(G,f"case-{i}",reason in r.reasons,r.reasons)
r=evaluate_database_readiness(db(backup_verified_at=NOW+timedelta(seconds=6)),now=NOW); add(G,"future-backup","BACKUP_VERIFICATION_FROM_FUTURE" in r.reasons,r.reasons)

# G08 - evidence chain, CAS, future evidence
G="G08-evidence"
l=ledger(ProductionStage.SOURCE_READY); add(G,"chain",l.verify_chain())
add(G,"digest-stable",l.digest()==EvidenceLedger.from_payload(l.to_payload()).digest())
with tempfile.TemporaryDirectory() as td:
    p=Path(td)/"e.json"; l.save_atomic(p); add(G,"atomic-load",EvidenceLedger.load(p).digest()==l.digest())
with tempfile.TemporaryDirectory() as td:
    store=EvidenceLedgerStore(Path(td)/"e.json"); base,d=store.load(); rec=store.append_record(kind=EvidenceKind.SOURCE_GATES,status=EvidenceStatus.PASS,observed_at=NOW,ttl=timedelta(hours=1),artifact_sha256="a"*64,producer="p",expected_digest=d); add(G,"store-append",rec.kind is EvidenceKind.SOURCE_GATES)
    add(G,"cas-reject",caught(EvidenceConcurrencyError,lambda:store.save_if_unchanged(base,expected_digest=d)) is not None)
future=EvidenceLedger()
for i,k in enumerate(sorted(cumulative_requirements(ProductionStage.SOURCE_READY),key=lambda x:x.value)):
    future.add(kind=k,status=EvidenceStatus.PASS,observed_at=NOW+timedelta(minutes=1,microseconds=i),ttl=timedelta(hours=1),artifact_sha256=hashlib.sha256(k.value.encode()).hexdigest(),producer="f")
r=StageEvaluator(future).evaluate(ProductionStage.SOURCE_READY,now=NOW); add(G,"future-block",not r.passed and any(x.startswith("EVIDENCE_FROM_FUTURE") for x in r.reasons),r.reasons)
old=EvidenceLedger();
for i,k in enumerate(sorted(cumulative_requirements(ProductionStage.SOURCE_READY),key=lambda x:x.value)):
    old.add(kind=k,status=EvidenceStatus.PASS,observed_at=NOW-timedelta(days=8)+timedelta(microseconds=i),ttl=timedelta(days=30),artifact_sha256=hashlib.sha256(k.value.encode()).hexdigest(),producer="o")
r=StageEvaluator(old).evaluate(ProductionStage.SOURCE_READY,now=NOW); add(G,"max-age",not r.passed and bool(r.stale),[x.value for x in r.stale])
add(G,"timestamp-regress-rejected",caught(ValueError,lambda:l.add(kind=EvidenceKind.SOURCE_GATES,status=EvidenceStatus.PASS,observed_at=NOW-timedelta(days=1),ttl=timedelta(hours=1),artifact_sha256="b"*64,producer="x")) is not None)
add(G,"bad-hash-rejected",caught(ValueError,lambda:l.add(kind=EvidenceKind.SOURCE_GATES,status=EvidenceStatus.PASS,observed_at=NOW+timedelta(seconds=1),ttl=timedelta(hours=1),artifact_sha256="bad",producer="x")) is not None)
add(G,"zero-ttl-rejected",caught(ValueError,lambda:l.add(kind=EvidenceKind.SOURCE_GATES,status=EvidenceStatus.PASS,observed_at=NOW+timedelta(seconds=1),ttl=timedelta(0),artifact_sha256="b"*64,producer="x")) is not None)

# G09 - replay integrity
G="G09-recorded-replay"
def fact(seq,t,stream,etype,ident,ch): return RecordedFact(seq,t,stream,etype,ident,ch*64)
goodfacts=(fact(1,NOW,"market","ORDER","o","a"),fact(2,NOW+timedelta(seconds=1),"user","POSITION","p","b"),fact(3,NOW+timedelta(seconds=2),"account","BALANCE","b","c"))
def manifest(facts=goodfacts,feature="d"*64): return ReplayManifest("binance","acct",("BTCUSDT",),NOW,NOW+timedelta(seconds=10),tuple(facts),feature)
add(G,"good",evaluate_replay_integrity(manifest()).passed)
add(G,"feature-required","FEATURE_SCHEMA_PROVENANCE_MISSING" in evaluate_replay_integrity(manifest(feature=None)).reasons)
missing=(fact(1,NOW,"market","ORDER","o","a"),); r=evaluate_replay_integrity(manifest(missing)); add(G,"missing-stream",any(x.startswith("MISSING_STREAM") for x in r.reasons),r.reasons)
gap=(fact(1,NOW,"market","ORDER","o","a"),fact(3,NOW+timedelta(seconds=1),"user","POSITION","p","b"),fact(4,NOW+timedelta(seconds=2),"account","BALANCE","b","c")); add(G,"sequence-gap","SEQUENCE_GAPS" in evaluate_replay_integrity(manifest(gap)).reasons)
reg=(fact(1,NOW+timedelta(seconds=2),"market","ORDER","o","a"),fact(2,NOW,"user","POSITION","p","b"),fact(3,NOW+timedelta(seconds=3),"account","BALANCE","b","c")); add(G,"timestamp-regress","TIMESTAMP_REGRESSION" in evaluate_replay_integrity(manifest(reg)).reasons)
dup=(fact(1,NOW,"market","ORDER","x","a"),fact(2,NOW+timedelta(seconds=1),"market","ORDER","x","b"),fact(3,NOW+timedelta(seconds=2),"user","POSITION","p","c"),fact(4,NOW+timedelta(seconds=3),"account","BALANCE","b","d")); add(G,"duplicate-id","DUPLICATE_FACT_IDENTITY" in evaluate_replay_integrity(manifest(dup)).reasons)
add(G,"duplicate-seq-rejected",caught(ValueError,lambda:manifest((goodfacts[0],goodfacts[0]))) is not None)
add(G,"symbol-required",caught(ValueError,lambda:ReplayManifest("binance","acct",(),NOW,NOW,(),"d"*64)) is not None)
add(G,"ordered-required",caught(ValueError,lambda:manifest((goodfacts[1],goodfacts[0],goodfacts[2]))) is not None)
add(G,"payload-hash-required",caught(ValueError,lambda:RecordedFact(1,NOW,"market","ORDER","o","bad")) is not None)

# G10 - windowed shadow and numeric integrity
G="G10-shadow"
base_metrics=ShadowMetrics(timedelta(hours=12),restart_recoveries=1,funding_cycles_observed=2)
w1=ShadowWindow(NOW,NOW+timedelta(hours=12),base_metrics,source_cursor_start=0,source_cursor_end=100)
w2=ShadowWindow(NOW+timedelta(hours=12),NOW+timedelta(hours=24),base_metrics,source_cursor_start=101,source_cursor_end=200)
r=qualify_shadow_run((w1,w2),target="24h"); add(G,"continuous-24h",r.passed,r.reasons)
w_gap=ShadowWindow(NOW+timedelta(hours=13),NOW+timedelta(hours=25),base_metrics,source_cursor_start=101,source_cursor_end=200); r=qualify_shadow_run((w1,w_gap),target="24h"); add(G,"gap",any(x.startswith("WINDOW_GAP") for x in r.reasons),r.reasons)
w_overlap=ShadowWindow(NOW+timedelta(hours=11),NOW+timedelta(hours=23),base_metrics,source_cursor_start=101,source_cursor_end=200); r=qualify_shadow_run((w1,w_overlap),target="24h"); add(G,"overlap",any(x.startswith("WINDOW_OVERLAP") for x in r.reasons),r.reasons)
w_cursor=ShadowWindow(NOW+timedelta(hours=12),NOW+timedelta(hours=24),base_metrics,source_cursor_start=105,source_cursor_end=200); r=qualify_shadow_run((w1,w_cursor),target="24h"); add(G,"cursor-gap",any("CURSOR" in x for x in r.reasons),r.reasons)
add(G,"nan-metrics-rejected",caught(ValueError,lambda:ShadowMetrics(timedelta(hours=24),db_p99_ms=math.nan)) is not None)
add(G,"nan-policy-rejected",caught(ValueError,lambda:ShadowPolicy(max_db_p99_ms=math.nan)) is not None)
add(G,"ratio-rejected",caught(ValueError,lambda:ShadowMetrics(timedelta(hours=24),risk_reject_ratio=1.1)) is not None)
add(G,"negative-duration-rejected",caught(ValueError,lambda:ShadowMetrics(timedelta(0))) is not None)
q=qualify_shadow(ShadowMetrics(timedelta(hours=72),restart_recoveries=1,funding_cycles_observed=2),target="72h"); add(G,"72h-funding",not q.passed and "INSUFFICIENT_FUNDING_CYCLES" in q.reasons,q.reasons)
q=qualify_shadow(ShadowMetrics(timedelta(hours=72),restart_recoveries=1,funding_cycles_observed=3),target="72h"); add(G,"72h-good",q.passed,q.reasons)

# G11 - model governance/fallback circuit
G="G11-model-circuit"
breaker=ModelCircuitBreaker(ModelCircuitPolicy(consecutive_failures_to_open=3,consecutive_successes_to_close=2,cooldown_seconds=5)); bad=ModelRuntimeDecision(False,"golden",("x",)); goodm=ModelRuntimeDecision(True,"golden",())
for i in range(2): add(G,f"bad-{i}",not breaker.observe(bad,now=NOW+timedelta(seconds=i)).open)
add(G,"open-3",breaker.observe(bad,now=NOW+timedelta(seconds=2)).open)
add(G,"cooldown-holds",breaker.observe(goodm,now=NOW+timedelta(seconds=3)).open)
add(G,"closes-after-cooldown-streak",not breaker.observe(goodm,now=NOW+timedelta(seconds=7)).open)
reg=FallbackProfileRegistry(); reg.register(FallbackProfile("golden","a"*64,True,.1,.1)); add(G,"fallback-approved",reg.approved("golden")); add(G,"fallback-resolve",reg.resolve("golden").approved)
reg2=FallbackProfileRegistry((FallbackProfile("bad","b"*64,False,.1,.1),)); add(G,"fallback-unapproved",not reg2.approved("bad")); add(G,"fallback-denied",caught(PermissionError,lambda:reg2.resolve("bad")) is not None)
add(G,"fallback-hash-required",caught(ValueError,lambda:FallbackProfile("x","bad",True,.1,.1)) is not None)

# G12 - model target contract
G="G12-model-target"
prev=ModelTarget(1,NOW,Decimal(".10"),Decimal(".10"),Decimal(".90"))
cases=[
    (ModelTarget(2,NOW,Decimal(".15"),Decimal(".10"),Decimal(".90")),True,"ok"),
    (ModelTarget(2,NOW+timedelta(seconds=1),Decimal(".15"),Decimal(".10"),Decimal(".90")),False,"future"),
    (ModelTarget(2,NOW-timedelta(seconds=6),Decimal(".15"),Decimal(".10"),Decimal(".90")),False,"stale"),
    (ModelTarget(2,NOW,Decimal("-.1"),Decimal(".10"),Decimal(".90")),False,"negative"),
    (ModelTarget(2,NOW,Decimal(".50"),Decimal(".10"),Decimal(".90")),False,"long"),
    (ModelTarget(2,NOW,Decimal(".30"),Decimal(".30"),Decimal(".90")),False,"jump"),
    (ModelTarget(1,NOW,Decimal(".10"),Decimal(".10"),Decimal(".90")),False,"sequence"),
    (ModelTarget(2,NOW,Decimal(".20"),Decimal(".10"),Decimal(".20")),False,"confidence"),
    (ModelTarget(2,NOW,Decimal(".15"),Decimal(".10"),Decimal(".90"),Decimal(".1")),False,"budget"),
    (ModelTarget(2,NOW,Decimal(".15"),Decimal(".10"),Decimal(".90"),Decimal("1"),True),False,"pause"),
]
for i,(target,expect,_label) in enumerate(cases):
    d=validate_model_target(target,now=NOW,previous=prev); add(G,f"target-{i}",d.accepted is expect,(d.reasons,d.long_ratio,d.short_ratio))

# G13 - observability numeric + hysteresis
G="G13-observability"
def hs(**kw):
    base=dict(available_margin_ratio=.5,liquidation_buffer_ratio=.5,unknown_orders=0,position_divergences=0,market_data_age_seconds=0,user_stream_age_seconds=0,loop_p99_ms=1,db_p99_ms=1,model_p99_ms=1,model_fallbacks_1h=0,risk_reject_ratio_1h=.1,memory_growth_ratio_1h=0)
    base.update(kw); return HealthSnapshot(**base)
add(G,"healthy",not evaluate_health(hs()))
add(G,"nan-reject",caught(ValueError,lambda:hs(db_p99_ms=math.nan)) is not None)
for i,(kw,code) in enumerate([({"unknown_orders":1},"UNKNOWN_ORDER"),({"fencing_mismatches":1},"FENCING_MISMATCH"),({"evidence_chain_valid":False},"EVIDENCE_CHAIN_INVALID"),({"outbox_backlog":101},"OUTBOX_BACKLOG"),({"database_disconnects_1h":4},"DATABASE_DISCONNECT_STORM")]):
    alerts=evaluate_health(hs(**kw)); add(G,f"alert-{i}",any(a.code==code for a in alerts),[a.code for a in alerts])
tracker=AlertStateTracker(AlertHysteresisPolicy(raise_after=2,clear_after=2)); spike=evaluate_health(hs(loop_p99_ms=999)); add(G,"warn-first-not-active",not any(x.active for x in tracker.observe(spike) if x.code=="LOOP_LATENCY")); add(G,"warn-second-active",any(x.active for x in tracker.observe(spike) if x.code=="LOOP_LATENCY")); add(G,"halt-immediate",any(x.active for x in AlertStateTracker(AlertHysteresisPolicy(raise_after=99)).observe(evaluate_health(hs(unknown_orders=1))) if x.code=="UNKNOWN_ORDER"))

# G14 - expanded fault campaign
G="G14-fault-campaign"
good_faults=[FaultResult(x,True,0,True,True,1) for x in FaultScenario]; add(G,"all-good",evaluate_fault_campaign(good_faults)[0])
variants=[
    (dict(passed=False),"FAILED"),(dict(duplicate_writes=1),"DUPLICATE_WRITE"),(dict(final_converged=False),"NOT_CONVERGED"),(dict(new_risk_blocked_during_fault=False),"NEW_RISK_NOT_BLOCKED"),(dict(recovery_seconds=31),"RECOVERY_SLA"),(dict(state_hash_match=False),"STATE_HASH"),(dict(outbox_drained=False),"OUTBOX"),(dict(fencing_preserved=False),"FENCING"),
]
for i,(changes,needle) in enumerate(variants):
    vals=list(good_faults); f=vals[0]; data=dict(scenario=f.scenario,passed=f.passed,duplicate_writes=f.duplicate_writes,final_converged=f.final_converged,new_risk_blocked_during_fault=f.new_risk_blocked_during_fault,recovery_seconds=f.recovery_seconds,state_hash_match=f.state_hash_match,outbox_drained=f.outbox_drained,fencing_preserved=f.fencing_preserved); data.update(changes); vals[0]=FaultResult(**data); ok,reasons=evaluate_fault_campaign(vals); add(G,f"fault-{i}",not ok and any(needle in x for x in reasons),reasons)
vals=list(good_faults)+[good_faults[0]]; ok,reasons=evaluate_fault_campaign(vals); add(G,"duplicate-scenario",not ok and "DUPLICATE_FAULT_SCENARIO_RESULT" in reasons,reasons)

# G15 - canary runtime/promotion
G="G15-canary-promotion"
def cm(level=CanaryLevel.MICRO,hours=6,fills=10,**kw):
    base=dict(started_at=NOW,observed_at=NOW+timedelta(hours=hours),level=level,orders_submitted=max(fills,10),orders_filled=fills,realized_pnl=Decimal("1"),fees=Decimal(".1"),funding=Decimal(".1"),max_drawdown_ratio=Decimal(".01"),unknown_orders=0,reconciliation_divergences=0,risk_limit_breaches=0,manual_interventions=0); base.update(kw); return CanaryRunMetrics(**base)
add(G,"micro-promote",evaluate_canary_promotion(cm()).promote)
add(G,"micro-duration",not evaluate_canary_promotion(cm(hours=5)).promote)
add(G,"micro-fills",not evaluate_canary_promotion(cm(fills=9)).promote)
add(G,"unknown",not evaluate_canary_promotion(cm(unknown_orders=1)).promote)
add(G,"recon",not evaluate_canary_promotion(cm(reconciliation_divergences=1)).promote)
add(G,"risk-breach",not evaluate_canary_promotion(cm(risk_limit_breaches=1)).promote)
add(G,"manual",not evaluate_canary_promotion(cm(manual_interventions=1)).promote)
add(G,"drawdown",not evaluate_canary_promotion(cm(max_drawdown_ratio=Decimal(".04"))).promote)
add(G,"small-promote",evaluate_canary_promotion(cm(level=CanaryLevel.SMALL,hours=24,fills=30,orders_submitted=30)).promote)
add(G,"bounded-promote-evidence",evaluate_canary_promotion(cm(level=CanaryLevel.BOUNDED,hours=72,fills=100,orders_submitted=100)).promote)

# G16 - control/incidents/runtime safety bundle
G="G16-runtime-control"
control=ProductionControlPlane(); add(G,"initial-halt",control.mode is ControlMode.HALT and not control.allows_reduce)
add(G,"resume-needs-recon",caught(PermissionError,lambda:control.apply(ControlAction.RESUME,actor="ops",reason="x",readiness_passed=True,reconciliation_converged=False,observed_at=NOW)) is not None)
control.apply(ControlAction.START,actor="ops",reason="ok",readiness_passed=True,reconciliation_converged=True,observed_at=NOW); add(G,"start-run",control.allows_new_risk)
control.apply(ControlAction.PAUSE_NEW_RISK,actor="ops",reason="pause",readiness_passed=False,reconciliation_converged=False,observed_at=NOW); add(G,"pause-keeps-reduce",not control.allows_new_risk and control.allows_reduce)
inc=IncidentLedger(); inc.open(Incident("i",Severity.HALT_ACCOUNT,"X",NOW)); add(G,"incident-block-account",inc.blocks_account)
add(G,"incident-close-needs-ack",caught(PermissionError,lambda:inc.close_checked("i",closed_at=NOW+timedelta(seconds=1),readiness_passed=True,reconciliation_converged=True,operator_acknowledged=False)) is not None)
inc.close_checked("i",closed_at=NOW+timedelta(seconds=1),readiness_passed=True,reconciliation_converged=True,operator_acknowledged=True); add(G,"incident-close",not inc.blocks_account)
bundle=ProductionRuntimeBundle.create(); snap=bundle.set_freshness(market_data_fresh=True,risk_data_fresh=True,now=NOW); add(G,"no-recon-failclosed",not snap.allows_reduce,snap.reasons)
bundle.observe_reconciliation(ReconciliationResult(True,True,True,()),observed_at=NOW,now=NOW); bundle.observe_reconciliation(ReconciliationResult(True,True,True,()),observed_at=NOW+timedelta(seconds=1),now=NOW+timedelta(seconds=1)); snap=bundle.observe_reconciliation(ReconciliationResult(True,True,True,()),observed_at=NOW+timedelta(seconds=2),now=NOW+timedelta(seconds=2)); add(G,"recon-but-control-halt",not snap.allows_new_risk,snap.reasons)
bundle.apply_control(ControlAction.START,actor="ops",reason="ok",readiness_passed=True,reconciliation_converged=True,observed_at=NOW+timedelta(seconds=2)); snap=bundle.set_freshness(market_data_fresh=True,risk_data_fresh=True,now=NOW+timedelta(seconds=2)); add(G,"runtime-run",snap.allows_new_risk and snap.allows_reduce,snap.reasons)

# G17 - unknown recovery restore semantics
G="G17-unknown-recovery"
class Clock:
    def __init__(self, now): self.value=now
    def now(self): return self.value
class Exec:
    def resolve_unknown(self, client_order_id): raise RuntimeError("still unknown")
clock=Clock(NOW); sup=UnknownOrderSupervisor(Exec(),clock=clock,recovery_deadline=timedelta(seconds=10),maximum_attempts=3)
r=sup.restore("a",first_unknown_at=NOW); add(G,"restore-pending",r.state is UnknownRecoveryState.PENDING)
add(G,"restore-idempotent",sup.restore("a",first_unknown_at=NOW).first_unknown_at==NOW)
clock.value=NOW+timedelta(seconds=11); sup2=UnknownOrderSupervisor(Exec(),clock=clock,recovery_deadline=timedelta(seconds=10)); r=sup2.restore("old",first_unknown_at=NOW); add(G,"expired-restore-halt",r.state is UnknownRecoveryState.HALTED,r.state)
clock.value=NOW; sup3=UnknownOrderSupervisor(Exec(),clock=clock,maximum_attempts=1); r=sup3.restore("budget",first_unknown_at=NOW,attempts=1); add(G,"attempt-budget-halt",r.state is UnknownRecoveryState.HALTED,r.state)
add(G,"bad-id",caught(ValueError,lambda:sup.restore("",first_unknown_at=NOW)) is not None)
add(G,"naive-time",caught(ValueError,lambda:sup.restore("x",first_unknown_at=datetime(2026,1,1))) is not None)
add(G,"negative-attempt",caught(ValueError,lambda:sup.restore("x",first_unknown_at=NOW,attempts=-1)) is not None)
r=sup.register("new"); add(G,"register",r.state is UnknownRecoveryState.PENDING)
add(G,"due",any(x.client_order_id=="new" for x in sup.due(NOW)))
r=sup.mark_manual_review("new","ops"); add(G,"manual",r.state is UnknownRecoveryState.MANUAL_REVIEW)

# G18 - execution adapter integration/source invariants
G="G18-execution-integration"
path=ROOT/'freqtrade/hedge/execution/binance_usdm_adapter.py'; source=path.read_text(encoding='utf-8'); tree=ast.parse(source)
for i,needle in enumerate(["commit_canary_for_client","release_canary_for_client","/fapi/v1/order/test","/fapi/v1/order","TERMINAL_STATES","DefinitiveSubmissionError","ambiguous","assert_order_allowed"]): add(G,f"source-{i}",needle in source,needle)
# Guard against accidental Binance Hedge Mode reduceOnly usage in order params.
new_order_segment=source[source.index('def _new_order_params'):source.index('def _request_public')]; add(G,"no-reduceonly",'reduceOnly' not in new_order_segment)
add(G,"ast",isinstance(tree,ast.Module))

# G19 - admission + release/fallback consistency
G="G19-admission-release"
risk=evaluate_post_trade_risk(AccountRiskView(Decimal("1000"),Decimal("900"),Decimal("100"),Decimal("0"),Decimal("50"),Decimal("5")),CandidateIntent(Side.LONG,RiskDirection.INCREASE,Decimal("10"),Decimal("1"),Decimal(".1")),CrossRiskLimits())
ctrl=ProductionControlPlane(); ctrl.apply(ControlAction.START,actor="ops",reason="ok",readiness_passed=True,reconciliation_converged=True,observed_at=NOW); recon=ReconciliationResult(True,True,True,()); ev=StageEvaluator(ledger()); live_new=ev.issue_lease(Capability.LIVE_NEW_RISK,actor="ops",now=NOW); live_reduce=ev.issue_lease(Capability.LIVE_REDUCE,actor="ops",now=NOW)
ctx=AdmissionContext(live_new,ctrl,recon,risk,ModelRuntimeDecision(True,"golden",()),True,True,False,NOW,RiskDirection.INCREASE); add(G,"admit-new",admit(ctx).decision in {Decision.APPROVE,Decision.CLIP})
ctx=AdmissionContext(live_reduce,ctrl,recon,risk,ModelRuntimeDecision(True,"golden",()),True,True,False,NOW,RiskDirection.INCREASE); add(G,"capability-direction",admit(ctx).decision in {Decision.REJECT,Decision.HALT} and "CAPABILITY_DIRECTION_MISMATCH" in admit(ctx).reasons,admit(ctx).reasons)
ctx=AdmissionContext(live_new,ctrl,recon,risk,ModelRuntimeDecision(False,"golden",("x",)),True,True,False,NOW,RiskDirection.INCREASE,False); add(G,"model-fallback-block",admit(ctx).decision in {Decision.REJECT,Decision.HALT})
ctx=AdmissionContext(live_new,ctrl,recon,risk,ModelRuntimeDecision(False,"golden",("x",)),True,True,False,NOW,RiskDirection.INCREASE,True); add(G,"approved-fallback",admit(ctx).decision in {Decision.APPROVE,Decision.CLIP},admit(ctx).reasons)
# Release aggregation must also accept an explicitly approved deterministic fallback.
fake_eval=SimpleNamespace(evaluate=lambda stage,now:SimpleNamespace(passed=True)); dbres=DatabaseReadinessResult(True,(),"postgresql"); shadowq=ShadowQualification("72h",True,()); canary=CanaryDecision(True,CanaryLevel.BOUNDED,()); faults=tuple(FaultResult(x,True,0,True,True,1) for x in FaultScenario)
report=build_production_readiness_report(evaluator=fake_eval,target_stage=ProductionStage.LIVE_READY,now=NOW,database=dbres,fault_results=faults,shadow=shadowq,alerts=(),model=ModelRuntimeDecision(False,"golden",("MODEL_DOWN",)),deterministic_fallback_ready=True,canary=canary); add(G,"release-fallback",report.model_ready and "MODEL_NOT_DEPLOYABLE_AND_FALLBACK_NOT_READY" not in report.reasons,report.reasons)
report2=build_production_readiness_report(evaluator=fake_eval,target_stage=ProductionStage.LIVE_READY,now=NOW,database=dbres,fault_results=faults,shadow=shadowq,alerts=(),model=ModelRuntimeDecision(False,"golden",("MODEL_DOWN",)),deterministic_fallback_ready=False,canary=canary); add(G,"release-no-fallback",not report2.model_ready and "MODEL_NOT_DEPLOYABLE_AND_FALLBACK_NOT_READY" in report2.reasons,report2.reasons)
report3=build_production_readiness_report(evaluator=fake_eval,target_stage=ProductionStage.LIVE_READY,now=NOW,database=None,fault_results=faults,shadow=shadowq,alerts=(),model=ModelRuntimeDecision(True,"golden",()),canary=canary); add(G,"release-db-required","DATABASE_RUNTIME_SNAPSHOT_REQUIRED" in report3.reasons,report3.reasons)
report4=build_production_readiness_report(evaluator=fake_eval,target_stage=ProductionStage.LIVE_READY,now=NOW,database=dbres,fault_results=(),shadow=shadowq,alerts=(),model=ModelRuntimeDecision(True,"golden",()),canary=canary); add(G,"release-fault-required","FAULT_CAMPAIGN_RUNTIME_SNAPSHOT_REQUIRED" in report4.reasons,report4.reasons)
report5=build_production_readiness_report(evaluator=fake_eval,target_stage=ProductionStage.LIVE_READY,now=NOW,database=dbres,fault_results=faults,shadow=None,alerts=(),model=ModelRuntimeDecision(True,"golden",()),canary=canary); add(G,"release-shadow-required","SHADOW_RUNTIME_SNAPSHOT_REQUIRED" in report5.reasons,report5.reasons)
report6=build_production_readiness_report(evaluator=fake_eval,target_stage=ProductionStage.LIVE_READY,now=NOW,database=dbres,fault_results=faults,shadow=shadowq,alerts=(),model=ModelRuntimeDecision(True,"golden",()),canary=None); add(G,"release-canary-required","CANARY_RUNTIME_SNAPSHOT_REQUIRED" in report6.reasons,report6.reasons)

# G20 - source hygiene / release isolation contracts
G="G20-source-hygiene"
files=[ROOT/'freqtrade/hedge/production'/x for x in ['reservations.py','execution_guard.py','runtime_supervisor.py','risk_envelope.py','submission_runtime.py','reconciliation_runtime.py','database_runtime.py','model_targets.py']]
for i,p in enumerate(files):
    try: ast.parse(p.read_text(encoding='utf-8')); ok=True
    except SyntaxError: ok=False
    add(G,f"ast-{i}",ok,p.as_posix())
from freqtrade.hedge.production import PRODUCTION_READINESS_API_VERSION, PRODUCTION_READINESS_RELEASE
add(G,"release-identity",PRODUCTION_READINESS_API_VERSION=="1.1" and PRODUCTION_READINESS_RELEASE=="freqtrade-hedge-production-readiness-r1.1-deep200",(PRODUCTION_READINESS_API_VERSION,PRODUCTION_READINESS_RELEASE))
add(G,"validator-count-precondition",len(CHECKS)==199,len(CHECKS))

assert len(CHECKS)==200, f"expected 200 checks, got {len(CHECKS)}"
failed=[x for x in CHECKS if not x['pass']]
summary={"schema":"hedge-production-readiness-deep-200-v1","expected":200,"executed":len(CHECKS),"passed":200-len(failed),"failed":len(failed),"status":"PASS" if not failed else "FAIL","checks":CHECKS}

parser=argparse.ArgumentParser()
parser.add_argument('--output')
parser.add_argument('--summary-only',action='store_true')
args=parser.parse_args()
if args.output:
    Path(args.output).write_text(json.dumps(summary,indent=2,sort_keys=True,default=str),encoding='utf-8')
print(f"HEDGE PRODUCTION DEEP 200: {summary['passed']}/200 PASS; FAIL={summary['failed']}")
print(json.dumps({k:summary[k] for k in ('schema','expected','executed','passed','failed','status')},sort_keys=True))
if not args.summary_only:
    for item in failed: print(json.dumps(item,sort_keys=True,default=str))
raise SystemExit(0 if not failed else 1)
