from __future__ import annotations

import hashlib
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from freqtrade.hedge.production.canary import CanaryLevel, CanaryRuntime, evaluate_canary
from freqtrade.hedge.production.contracts import Capability, Decision, EvidenceKind, EvidenceStatus, ProductionStage, cumulative_requirements
from freqtrade.hedge.production.control import ControlAction, ControlMode, ProductionControlPlane
from freqtrade.hedge.production.database import DatabaseReadinessInput, evaluate_database_readiness
from freqtrade.hedge.production.evidence import EvidenceLedger
from freqtrade.hedge.production.faults import FaultResult, FaultScenario, evaluate_fault_campaign
from freqtrade.hedge.production.execution_guard import ReadinessBoundProductionExecutionGate
from freqtrade.hedge.production.runtime_supervisor import RuntimeSafetySnapshot
from freqtrade.hedge.production.golden import GoldenSignal, deterministic_golden_target
from freqtrade.hedge.production.model_governance import ApprovalRecord, InferenceHealth, ModelIdentity, ModelStatus, decide_model_runtime
from freqtrade.hedge.production.policy import StageEvaluator
from freqtrade.hedge.production.reconciliation import PositionTruth, ReconciliationPlane, reconcile
from freqtrade.hedge.production.recovery import CrashPoint, RecoveryContext, build_recovery_plan
from freqtrade.hedge.production.risk_envelope import AccountRiskView, CandidateIntent, CrossRiskLimits, RiskDirection, Side, evaluate_post_trade_risk
from freqtrade.hedge.production.security import SecurityFacts, evaluate_security
from freqtrade.hedge.production.shadow import ShadowMetrics, qualify_shadow
from freqtrade.hedge.production.submission import SubmissionClass, SubmissionObservation, classify_submission
from freqtrade.hedge.execution.binance_environment import ExecutionEnvironment
from freqtrade.hedge.execution.client_order_id import build_client_order_id
from freqtrade.hedge.execution.production_gate import ExecutionWriteLockedError, ProductionGateEvidence
from freqtrade.hedge.execution.service import ApprovedOrderIntent, IntentAction, OrderIntent, OrderType, PositionSide

NOW=datetime(2026,8,15,tzinfo=UTC)

class ProductionReadinessR1Tests(unittest.TestCase):
    def ledger(self):
        l=EvidenceLedger()
        for kind in sorted(cumulative_requirements(ProductionStage.LIVE_READY),key=lambda x:x.value):
            l.add(kind=kind,status=EvidenceStatus.PASS,observed_at=NOW,ttl=timedelta(days=2),artifact_sha256='a'*64,producer='test')
        return l

    def test_all_stage_evidence_passes(self):
        ev=StageEvaluator(self.ledger())
        for stage in ProductionStage: self.assertTrue(ev.evaluate(stage,now=NOW).passed)

    def test_live_lease_is_short_lived(self):
        lease=StageEvaluator(self.ledger()).issue_lease(Capability.LIVE_NEW_RISK,actor='test',now=NOW)
        self.assertTrue(lease.valid_at(NOW)); self.assertFalse(lease.valid_at(NOW+timedelta(hours=1)))

    def test_ledger_persists_and_verifies(self):
        l=self.ledger()
        with tempfile.TemporaryDirectory() as d:
            path=l.save_atomic(Path(d)/'ledger.json'); loaded=EvidenceLedger.load(path)
            self.assertEqual(l.digest(),loaded.digest()); self.assertTrue(loaded.verify_chain())

    def test_sqlite_fails_live_db_gate(self):
        value=DatabaseReadinessInput('sqlite','h','h',True,True,True,True,True,True,NOW,NOW)
        self.assertFalse(evaluate_database_readiness(value,now=NOW).passed)

    def test_cross_risk_clips_or_rejects_stress(self):
        view=AccountRiskView(Decimal('1000'),Decimal('100'),Decimal('1200'),Decimal('200'),Decimal('600'),Decimal('200'))
        intent=CandidateIntent(Side.LONG,RiskDirection.INCREASE,Decimal('500'),Decimal('100'),Decimal('20'))
        self.assertIn(evaluate_post_trade_risk(view,intent,CrossRiskLimits()).decision,{Decision.CLIP,Decision.REJECT})

    def test_reduce_is_preferred(self):
        view=AccountRiskView(Decimal('1000'),Decimal('100'),Decimal('1200'),Decimal('200'),Decimal('600'),Decimal('200'))
        intent=CandidateIntent(Side.LONG,RiskDirection.REDUCE,Decimal('100'),Decimal('20'),Decimal('5'))
        self.assertEqual(evaluate_post_trade_risk(view,intent,CrossRiskLimits()).decision,Decision.APPROVE)

    def test_unknown_exchange_order_halts_account_reduce(self):
        local=ReconciliationPlane.build(positions=(),open_order_ids=(),wallet_balance=Decimal('1'),hedge_mode=True,cross_symbols=('BTCUSDT',),cursor=1)
        exchange=ReconciliationPlane.build(positions=(),open_order_ids=('external',),wallet_balance=Decimal('1'),hedge_mode=True,cross_symbols=('BTCUSDT',),cursor=1)
        self.assertFalse(reconcile(local,exchange).allow_reduce)

    def test_every_crash_plan_forbids_blind_resubmit(self):
        for p in CrashPoint:
            plan=build_recovery_plan(RecoveryContext(p,True,True,False,True,False,True)); self.assertFalse(plan.blind_resubmit_allowed)

    def test_ambiguous_submit_queries_before_retry(self):
        d=classify_submission(SubmissionObservation(None,False,True)); self.assertEqual(d.classification,SubmissionClass.AMBIGUOUS); self.assertFalse(d.direct_resubmit_allowed)

    def test_control_resume_requires_convergence(self):
        c=ProductionControlPlane()
        with self.assertRaises(PermissionError): c.apply(ControlAction.RESUME,actor='x',reason='x',readiness_passed=True,reconciliation_converged=False,observed_at=NOW)

    def test_control_pause_keeps_reduce(self):
        c=ProductionControlPlane(); c.apply(ControlAction.PAUSE_NEW_RISK,actor='x',reason='x',readiness_passed=False,reconciliation_converged=False,observed_at=NOW)
        self.assertFalse(c.allows_new_risk); self.assertTrue(c.allows_reduce)

    def test_shadow_72h_requires_three_funding_cycles(self):
        self.assertFalse(qualify_shadow(ShadowMetrics(timedelta(hours=72),restart_recoveries=1,funding_cycles_observed=2),target='72h').passed)
        self.assertTrue(qualify_shadow(ShadowMetrics(timedelta(hours=72),restart_recoveries=1,funding_cycles_observed=3),target='72h').passed)

    def test_fault_campaign_is_exhaustive(self):
        good=tuple(FaultResult(x,True,0,True,True,1) for x in FaultScenario); self.assertTrue(evaluate_fault_campaign(good)[0])
        self.assertFalse(evaluate_fault_campaign(good[:-1])[0])

    def test_model_requires_approval_and_exact_hashes(self):
        h='a'*64; identity=ModelIdentity('m','HPRL',h,h,h,h,'torch'); record=ApprovalRecord(identity,ModelStatus.APPROVED,NOW,'ops',True,True,True,'golden')
        health=InferenceHealth(1,True,h,h,.01); self.assertTrue(decide_model_runtime(record,health).use_model)
        self.assertFalse(decide_model_runtime(record,InferenceHealth(1,True,'b'*64,h,.01)).use_model)

    def test_security_rejects_withdrawal_permission(self):
        facts=SecurityFacts(True,True,True,True,True,True,True,True,True,True,True); self.assertFalse(evaluate_security(facts,live=True).passed)

    def test_canary_degrades_to_reduce_only(self):
        runtime=CanaryRuntime(Decimal('999'),Decimal('-100'),Decimal('.5'),100,1)
        self.assertEqual(evaluate_canary(CanaryLevel.MICRO,runtime).effective_level,CanaryLevel.REDUCE_ONLY)


    def safe_runtime(self):
        return RuntimeSafetySnapshot(
            safety_epoch=7,
            observed_at=NOW,
            allows_new_risk=True,
            allows_reduce=True,
            reasons=(),
        )

    def live_gate(self, evaluator, **kwargs):
        return ReadinessBoundProductionExecutionGate(
            self.live_evidence(),
            evaluator=evaluator,
            clock=lambda: NOW,
            runtime_safety_provider=self.safe_runtime,
            **kwargs,
        )

    def live_evidence(self, *, token: str = "production-test-token"):
        return ProductionGateEvidence(
            environment=ExecutionEnvironment.LIVE,
            account_fingerprint="acct",
            allowed_symbols=("BTCUSDT",),
            cross_margin_symbols=("BTCUSDT",),
            readonly_status="FULL_PASS",
            user_stream_status="FULL_PASS",
            hedge_mode_enabled=True,
            clock_offset_ms=0,
            live_trading_enabled=True,
            strict_key_policy_passed=True,
            futures_trading_permission=True,
            ip_restricted=True,
            expected_arm_token_sha256=hashlib.sha256(token.encode()).hexdigest(),
            max_order_notional=Decimal("100"),
        )

    def approved_intent(self, action: IntentAction):
        intent=OrderIntent(
            account_id="binance-usdm:acct",
            symbol="BTCUSDT",
            position_side=PositionSide.LONG,
            action=action,
            quantity=Decimal("1"),
            idempotency_key=f"guard-{action.value.lower()}",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("10"),
        )
        client_id=build_client_order_id(
            account_id=intent.account_id,
            symbol=intent.symbol,
            position_side=intent.position_side.value,
            idempotency_key=intent.idempotency_key,
        )
        return ApprovedOrderIntent(intent,Decimal("1"),client_id,NOW,("TEST",))

    def ledger_through(self, stage: ProductionStage):
        ledger=EvidenceLedger()
        for kind in sorted(cumulative_requirements(stage),key=lambda x:x.value):
            ledger.add(kind=kind,status=EvidenceStatus.PASS,observed_at=NOW,ttl=timedelta(days=2),artifact_sha256='a'*64,producer='test')
        return ledger

    def test_execution_guard_source_ready_is_emergency_reduce_only(self):
        gate=self.live_gate(
            StageEvaluator(self.ledger_through(ProductionStage.SOURCE_READY))
        )
        gate.arm(token="production-test-token",actor="ops",confirmed=True)
        permit=gate.assert_order_allowed(self.approved_intent(IntentAction.REDUCE))
        self.assertEqual(permit.symbol,"BTCUSDT")
        with self.assertRaises(ExecutionWriteLockedError):
            gate.assert_order_allowed(self.approved_intent(IntentAction.INCREASE))

    def test_execution_guard_live_candidate_is_reduce_only(self):
        gate=self.live_gate(
            StageEvaluator(self.ledger_through(ProductionStage.LIVE_CANDIDATE))
        )
        gate.arm(token="production-test-token",actor="ops",confirmed=True)
        permit=gate.assert_order_allowed(self.approved_intent(IntentAction.REDUCE))
        self.assertEqual(permit.symbol,"BTCUSDT")
        with self.assertRaises(ExecutionWriteLockedError):
            gate.assert_order_allowed(self.approved_intent(IntentAction.INCREASE))

    def test_execution_guard_candidate_micro_canary_allows_bounded_new_risk(self):
        gate=self.live_gate(
            StageEvaluator(self.ledger_through(ProductionStage.LIVE_CANDIDATE)),
            canary_level=CanaryLevel.MICRO,
            canary_runtime_provider=lambda: CanaryRuntime(Decimal("0"),Decimal("0"),Decimal("0"),0,0),
        )
        gate.arm(token="production-test-token",actor="ops",confirmed=True)
        permit=gate.assert_order_allowed(self.approved_intent(IntentAction.INCREASE))
        self.assertEqual(permit.notional,Decimal("10"))

    def test_execution_guard_failed_arm_clears_readiness_leases(self):
        gate=self.live_gate(StageEvaluator(self.ledger()))
        with self.assertRaises(PermissionError):
            gate.arm(token="wrong-token",actor="ops",confirmed=True)
        self.assertIsNone(gate._reduce_lease)
        self.assertIsNone(gate._canary_risk_lease)
        self.assertIsNone(gate._new_risk_lease)

    def test_execution_guard_live_ready_allows_new_risk(self):
        gate=self.live_gate(StageEvaluator(self.ledger()))
        gate.arm(token="production-test-token",actor="ops",confirmed=True)
        self.assertEqual(gate.assert_order_allowed(self.approved_intent(IntentAction.INCREASE)).symbol,"BTCUSDT")

    def test_execution_guard_evidence_digest_change_revokes_write(self):
        ledger=self.ledger()
        gate=self.live_gate(StageEvaluator(ledger))
        gate.arm(token="production-test-token",actor="ops",confirmed=True)
        ledger.add(
            kind=EvidenceKind.SOURCE_GATES,
            status=EvidenceStatus.PASS,
            observed_at=NOW+timedelta(seconds=1),
            ttl=timedelta(days=2),
            artifact_sha256='b'*64,
            producer='test-refresh',
        )
        with self.assertRaisesRegex(ExecutionWriteLockedError,"PRODUCTION_READINESS_EVIDENCE_CHANGED"):
            gate.assert_order_allowed(self.approved_intent(IntentAction.REDUCE))

    def test_golden_strategy_is_bounded(self):
        for a,b in [('2','1'),('1','2'),('1','1')]:
            t=deterministic_golden_target(GoldenSignal(Decimal(a),Decimal(b),Decimal('.1')))
            self.assertLessEqual(t.long_ratio,Decimal('.12')); self.assertLessEqual(t.short_ratio,Decimal('.12'))

if __name__=='__main__': unittest.main()
