from __future__ import annotations

import hashlib
import math
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from freqtrade.hedge.execution.binance_environment import ExecutionEnvironment
from freqtrade.hedge.execution.binance_usdm_adapter import (
    BinanceExecutionCredentials, BinanceUSDMExecutionAdapter, HttpResponse,
)
from freqtrade.hedge.execution.client_order_id import build_client_order_id
from freqtrade.hedge.execution.production_gate import ExecutionWriteLockedError, ProductionGateEvidence
from freqtrade.hedge.execution.service import (
    ApprovedOrderIntent, DefinitiveSubmissionError, ExecutionOrder, InMemoryExecutionStore,
    IntentAction, OrderIntent, OrderType, PositionSide,
)
from freqtrade.hedge.execution.state_machine import OrderLifecycle, OrderState
from freqtrade.hedge.production.canary import CanaryLevel, CanaryRuntime
from freqtrade.hedge.production.contracts import Capability, Decision, EvidenceStatus, ProductionStage, Severity, cumulative_requirements
from freqtrade.hedge.production.database import DatabaseReadinessInput, evaluate_database_readiness
from freqtrade.hedge.production.database_runtime import PostgresConcurrencyProbeRunner, PostgresProbeRunner
from freqtrade.hedge.production.evidence import EvidenceConcurrencyError, EvidenceKind, EvidenceLedger, EvidenceLedgerStore
from freqtrade.hedge.production.execution_guard import ReadinessBoundProductionExecutionGate
from freqtrade.hedge.production.faults import FaultResult, FaultScenario, evaluate_fault_campaign
from freqtrade.hedge.production.model_governance import FallbackProfile, FallbackProfileRegistry, ModelCircuitBreaker, ModelCircuitPolicy, ModelRuntimeDecision
from freqtrade.hedge.production.model_targets import ModelTarget, ModelTargetPolicy, validate_model_target
from freqtrade.hedge.production.observability import AlertHysteresisPolicy, AlertStateTracker, HealthSnapshot, evaluate_health
from freqtrade.hedge.production.policy import StageEvaluator
from freqtrade.hedge.production.reconciliation import DiffKind, ReconciliationDiff, ReconciliationResult
from freqtrade.hedge.production.reconciliation_runtime import ReconciliationAction, ReconciliationSupervisor, ReconciliationSupervisorPolicy, build_reconciliation_plan
from freqtrade.hedge.production.replay import RecordedFact, ReplayIntegrityPolicy, ReplayManifest, evaluate_replay_integrity
from freqtrade.hedge.production.reservations import ExposureReservationBook, ReservationState
from freqtrade.hedge.production.risk_envelope import AccountRiskView, CandidateIntent, CrossRiskLimits, RiskDirection, Side, StressScenario, evaluate_post_trade_risk
from freqtrade.hedge.production.runtime_supervisor import RuntimeSafetySnapshot
from freqtrade.hedge.production.shadow import ShadowMetrics, ShadowQualification
from freqtrade.hedge.production.shadow_runtime import ShadowWindow, qualify_shadow_run
from freqtrade.hedge.production.submission import SubmissionObservation
from freqtrade.hedge.production.submission_runtime import SubmissionRecoveryMachine, SubmissionRecoveryPolicy, SubmissionRecoveryState

NOW = datetime(2026, 8, 15, 1, 30, tzinfo=UTC)


def _ledger(stage: ProductionStage = ProductionStage.LIVE_READY) -> EvidenceLedger:
    ledger = EvidenceLedger()
    for index, kind in enumerate(sorted(cumulative_requirements(stage), key=lambda x: x.value), start=1):
        ledger.add(
            kind=kind,
            status=EvidenceStatus.PASS,
            observed_at=NOW + timedelta(microseconds=index),
            ttl=timedelta(days=2),
            artifact_sha256=hashlib.sha256(kind.value.encode()).hexdigest(),
            producer="deep-test",
        )
    return ledger


def _live_evidence(token: str = "deep-token") -> ProductionGateEvidence:
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


def _approved(action: IntentAction, key: str = "deep") -> ApprovedOrderIntent:
    intent = OrderIntent(
        account_id="binance-usdm:acct",
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        action=action,
        quantity=Decimal("1"),
        idempotency_key=key,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("10"),
    )
    cid = build_client_order_id(
        account_id=intent.account_id,
        symbol=intent.symbol,
        position_side=intent.position_side.value,
        idempotency_key=intent.idempotency_key,
    )
    return ApprovedOrderIntent(intent, Decimal("1"), cid, NOW, ("DEEP",))


def _safe(epoch: int = 1, *, observed_at: datetime = NOW, new: bool = True, reduce: bool = True):
    return RuntimeSafetySnapshot(epoch, observed_at, new, reduce, ())


class DeepProductionTests(unittest.TestCase):
    def test_live_gate_requires_runtime_safety_provider(self):
        with self.assertRaisesRegex(ValueError, "runtime_safety_provider"):
            ReadinessBoundProductionExecutionGate(_live_evidence(), evaluator=StageEvaluator(_ledger()))

    def test_source_ready_can_issue_reduce_without_new_risk(self):
        evaluator = StageEvaluator(_ledger(ProductionStage.SOURCE_READY))
        reduce_lease = evaluator.issue_lease(Capability.LIVE_REDUCE, actor="ops", now=NOW)
        self.assertEqual(reduce_lease.stage, ProductionStage.SOURCE_READY)
        with self.assertRaises(PermissionError):
            evaluator.issue_lease(Capability.LIVE_NEW_RISK, actor="ops", now=NOW)

    def test_runtime_safety_stale_blocks_arm(self):
        gate = ReadinessBoundProductionExecutionGate(
            _live_evidence(), evaluator=StageEvaluator(_ledger()), clock=lambda: NOW,
            runtime_safety_provider=lambda: _safe(observed_at=NOW-timedelta(seconds=6)),
        )
        with self.assertRaisesRegex(ExecutionWriteLockedError, "RUNTIME_SAFETY_STALE"):
            gate.arm(token="deep-token", actor="ops", confirmed=True)

    def test_runtime_safety_future_blocks_arm(self):
        gate = ReadinessBoundProductionExecutionGate(
            _live_evidence(), evaluator=StageEvaluator(_ledger()), clock=lambda: NOW,
            runtime_safety_provider=lambda: _safe(observed_at=NOW+timedelta(seconds=2)),
        )
        with self.assertRaisesRegex(ExecutionWriteLockedError, "RUNTIME_SAFETY_FROM_FUTURE"):
            gate.arm(token="deep-token", actor="ops", confirmed=True)

    def test_runtime_safety_epoch_revokes_armed_session(self):
        state = {"value": _safe(2)}
        gate = ReadinessBoundProductionExecutionGate(
            _live_evidence(), evaluator=StageEvaluator(_ledger()), clock=lambda: NOW,
            runtime_safety_provider=lambda: state["value"],
        )
        gate.arm(token="deep-token", actor="ops", confirmed=True)
        state["value"] = _safe(3)
        with self.assertRaisesRegex(ExecutionWriteLockedError, "EPOCH_CHANGED"):
            gate.assert_order_allowed(_approved(IntentAction.REDUCE))


    def _candidate_adapter(self, transport):
        book = ExposureReservationBook()
        gate = ReadinessBoundProductionExecutionGate(
            _live_evidence(),
            evaluator=StageEvaluator(_ledger(ProductionStage.LIVE_CANDIDATE)),
            clock=lambda: NOW,
            runtime_safety_provider=lambda: _safe(),
            canary_level=CanaryLevel.MICRO,
            canary_runtime_provider=lambda: CanaryRuntime(
                Decimal("0"), Decimal("0"), Decimal("0"), 0, 0
            ),
            canary_reservations=book,
        )
        gate.arm(token="deep-token", actor="ops", confirmed=True)
        store = InMemoryExecutionStore()
        adapter = BinanceUSDMExecutionAdapter(
            credentials=BinanceExecutionCredentials("k", "s"),
            gate=gate,
            store=store,
            transport=transport,
            base_url="https://fapi.binance.com",
            now_ms=lambda: 1_700_000_000_000,
            sleep=lambda _: None,
        )
        return adapter, gate, book, store

    def test_binance_test_order_releases_canary_reservation(self):
        class Transport:
            def request(self, method, url, headers, timeout):
                self.last = (method, url)
                return HttpResponse(200, {}, b"{}")
        adapter, _, book, _ = self._candidate_adapter(Transport())
        adapter.validate_order(_approved(IntentAction.INCREASE, "adapter-test"))
        self.assertEqual(book.snapshot(now=NOW).held_orders, 0)

    def test_binance_successful_submit_commits_until_terminal_fact(self):
        approved = _approved(IntentAction.INCREASE, "adapter-ack")
        class Transport:
            def request(self, method, url, headers, timeout):
                if method == "POST":
                    payload = {
                        "clientOrderId": approved.client_order_id, "status": "NEW",
                        "executedQty": "0", "orderId": 123, "updateTime": 1700000000000,
                    }
                else:
                    payload = {
                        "clientOrderId": approved.client_order_id, "status": "FILLED",
                        "executedQty": "1", "avgPrice": "10", "orderId": 123,
                        "updateTime": 1700000001000,
                    }
                import json
                return HttpResponse(200, {}, json.dumps(payload).encode())
        adapter, _, book, store = self._candidate_adapter(Transport())
        snap = adapter.submit_order(approved)
        self.assertEqual(snap.status, OrderState.ACKNOWLEDGED)
        self.assertEqual(book.snapshot(now=NOW).held_orders, 1)
        store.put(ExecutionOrder(
            approved.intent, approved.client_order_id, approved.approved_quantity,
            OrderLifecycle(status=OrderState.ACKNOWLEDGED, exchange_order_id="123", updated_at=NOW), NOW,
        ))
        terminal = adapter.query_order(client_order_id=approved.client_order_id)
        self.assertEqual(terminal.status, OrderState.FILLED)
        self.assertEqual(book.snapshot(now=NOW).held_orders, 0)

    def test_binance_definitive_submit_reject_releases_canary(self):
        class Transport:
            def request(self, method, url, headers, timeout):
                return HttpResponse(400, {}, b'{"code":-2010,"msg":"rejected"}')
        adapter, _, book, _ = self._candidate_adapter(Transport())
        with self.assertRaises(DefinitiveSubmissionError):
            adapter.submit_order(_approved(IntentAction.INCREASE, "adapter-reject"))
        self.assertEqual(book.snapshot(now=NOW).held_orders, 0)

    def test_binance_ambiguous_submit_holds_canary_until_reconciliation(self):
        approved = _approved(IntentAction.INCREASE, "adapter-unknown")
        class Transport:
            def request(self, method, url, headers, timeout):
                raise TimeoutError("network timeout")
        adapter, gate, book, _ = self._candidate_adapter(Transport())
        with self.assertRaises(TimeoutError):
            adapter.submit_order(approved)
        item = book.find_by_client_order(approved.client_order_id, now=NOW)
        self.assertIsNotNone(item)
        self.assertEqual(item.state, ReservationState.COMMITTED)
        self.assertEqual(book.snapshot(now=NOW + timedelta(hours=1)).held_orders, 1)
        gate.release_canary_for_client(approved.client_order_id)
        self.assertEqual(book.snapshot(now=NOW).held_orders, 0)

    def test_reservation_committed_can_release(self):
        book = ExposureReservationBook()
        item = book.reserve(client_order_id="cid", notional=Decimal("10"), now=NOW, max_total_notional=Decimal("50"), max_orders=4)
        committed = book.commit(item.reservation_id, now=NOW)
        self.assertEqual(committed.state, ReservationState.COMMITTED)
        released = book.release(item.reservation_id, now=NOW)
        self.assertEqual(released.state, ReservationState.RELEASED)
        self.assertEqual(book.snapshot(now=NOW).held_orders, 0)

    def test_reservation_committed_is_idempotent_by_client_id(self):
        book = ExposureReservationBook()
        item = book.reserve(client_order_id="cid", notional=Decimal("10"), now=NOW, max_total_notional=Decimal("50"), max_orders=4)
        committed = book.commit(item.reservation_id, now=NOW)
        again = book.reserve(client_order_id="cid", notional=Decimal("10"), now=NOW, max_total_notional=Decimal("50"), max_orders=4)
        self.assertEqual(again.reservation_id, committed.reservation_id)
        self.assertEqual(book.snapshot(now=NOW).held_orders, 1)

    def test_terminal_client_id_cannot_be_reused(self):
        book = ExposureReservationBook()
        item = book.reserve(client_order_id="cid", notional=Decimal("10"), now=NOW, max_total_notional=Decimal("50"), max_orders=4)
        book.release(item.reservation_id, now=NOW)
        with self.assertRaisesRegex(ValueError, "cannot be reused"):
            book.reserve(client_order_id="cid", notional=Decimal("10"), now=NOW, max_total_notional=Decimal("50"), max_orders=4)

    def test_committed_reservation_does_not_expire(self):
        book = ExposureReservationBook(ttl=timedelta(seconds=1))
        item = book.reserve(client_order_id="cid", notional=Decimal("10"), now=NOW, max_total_notional=Decimal("50"), max_orders=4)
        book.commit(item.reservation_id, now=NOW)
        book.expire(now=NOW+timedelta(hours=1))
        self.assertEqual(book.snapshot(now=NOW+timedelta(hours=1)).held_orders, 1)

    def test_reservation_concurrency_never_exceeds_order_limit(self):
        book = ExposureReservationBook()
        def reserve(i: int) -> bool:
            try:
                book.reserve(client_order_id=f"c{i}", notional=Decimal("10"), now=NOW, max_total_notional=Decimal("40"), max_orders=4)
                return True
            except PermissionError:
                return False
        with ThreadPoolExecutor(max_workers=16) as pool:
            accepted = list(pool.map(reserve, range(20)))
        self.assertEqual(sum(accepted), 4)
        self.assertEqual(book.snapshot(now=NOW).held_notional, Decimal("40"))

    def test_reduce_over_closeable_is_clipped_not_flipped(self):
        view = AccountRiskView(Decimal("1000"), Decimal("800"), Decimal("100"), Decimal("0"), Decimal("100"), Decimal("10"))
        intent = CandidateIntent(Side.LONG, RiskDirection.REDUCE, Decimal("150"), Decimal("75"), Decimal("10"))
        result = evaluate_post_trade_risk(view, intent, CrossRiskLimits())
        self.assertEqual(result.decision, Decision.CLIP)
        self.assertEqual(result.approved_notional, Decimal("100"))
        self.assertEqual(result.projection.long_notional, Decimal("0"))

    def test_pending_increase_does_not_expand_reduce_closeable(self):
        view = AccountRiskView(
            Decimal("1000"), Decimal("800"), Decimal("100"), Decimal("0"),
            Decimal("100"), Decimal("10"), pending_long_notional=Decimal("50"),
        )
        intent = CandidateIntent(
            Side.LONG, RiskDirection.REDUCE, Decimal("150"), Decimal("75"), Decimal("10")
        )
        result = evaluate_post_trade_risk(view, intent, CrossRiskLimits())
        self.assertEqual(result.decision, Decision.CLIP)
        self.assertEqual(result.approved_notional, Decimal("100"))

    def test_pending_reduce_reserves_closeable_capacity(self):
        view = AccountRiskView(
            Decimal("1000"), Decimal("800"), Decimal("100"), Decimal("0"),
            Decimal("100"), Decimal("10"),
            pending_long_reduce_notional=Decimal("70"),
        )
        intent = CandidateIntent(
            Side.LONG, RiskDirection.REDUCE, Decimal("50"), Decimal("25"), Decimal("5")
        )
        result = evaluate_post_trade_risk(view, intent, CrossRiskLimits())
        self.assertEqual(result.decision, Decision.CLIP)
        self.assertEqual(result.approved_notional, Decimal("30"))

    def test_pending_reduce_over_executed_position_invalidates_risk_snapshot(self):
        with self.assertRaisesRegex(ValueError, "pending_long_reduce_notional"):
            AccountRiskView(
                Decimal("1000"), Decimal("800"), Decimal("100"), Decimal("0"),
                Decimal("100"), Decimal("10"),
                pending_long_reduce_notional=Decimal("101"),
            )

    def test_reduce_flat_leg_rejected(self):
        view = AccountRiskView(Decimal("1000"), Decimal("800"), Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"))
        intent = CandidateIntent(Side.LONG, RiskDirection.REDUCE, Decimal("10"), Decimal("1"), Decimal("1"))
        result = evaluate_post_trade_risk(view, intent, CrossRiskLimits())
        self.assertEqual(result.decision, Decision.REJECT)
        self.assertIn("NO_CLOSEABLE_POSITION", result.reasons)

    def test_custom_stress_scenario_can_block_scale_in(self):
        view = AccountRiskView(Decimal("1000"), Decimal("700"), Decimal("200"), Decimal("0"), Decimal("100"), Decimal("10"))
        intent = CandidateIntent(Side.LONG, RiskDirection.INCREASE, Decimal("100"), Decimal("20"), Decimal("5"))
        result = evaluate_post_trade_risk(
            view, intent, CrossRiskLimits(max_stress_loss_ratio=Decimal("0.10")),
            stress_scenarios=(StressScenario("CRASH", Decimal("0.50"), Decimal("0")),),
        )
        self.assertIn(result.decision, {Decision.CLIP, Decision.REJECT})

    def test_ambiguous_submission_never_permits_new_risk(self):
        machine = SubmissionRecoveryMachine()
        record = machine.start(client_order_id="cid", now=NOW)
        record = machine.observe(record, SubmissionObservation(None, False, True), now=NOW)
        self.assertEqual(record.state, SubmissionRecoveryState.QUERY_REQUIRED)
        self.assertFalse(record.permits_new_risk)

    def test_authoritative_not_found_goes_manual_not_retry(self):
        machine = SubmissionRecoveryMachine()
        record = machine.start(client_order_id="cid", now=NOW)
        record = machine.observe(record, SubmissionObservation(None, False, True), now=NOW)
        record = machine.query_not_found(record, now=NOW+timedelta(seconds=1), exchange_history_complete=True)
        self.assertEqual(record.state, SubmissionRecoveryState.MANUAL_REVIEW)

    def test_ambiguous_query_budget_exhausts_to_manual(self):
        machine = SubmissionRecoveryMachine(SubmissionRecoveryPolicy(max_query_attempts=1))
        record = machine.start(client_order_id="cid", now=NOW)
        obs = SubmissionObservation(None, False, True)
        record = machine.observe(record, obs, now=NOW)
        record = machine.observe(record, obs, now=NOW+timedelta(seconds=1))
        self.assertEqual(record.state, SubmissionRecoveryState.MANUAL_REVIEW)

    def test_reconciliation_position_drift_keeps_reduce(self):
        diff = ReconciliationDiff(DiffKind.POSITION, "BTCUSDT:LONG", "2", "1", Severity.HALT_NEW_RISK)
        result = ReconciliationResult(False, False, True, (diff,))
        supervisor = ReconciliationSupervisor()
        snap = supervisor.observe(result, observed_at=NOW, now=NOW)
        self.assertTrue(snap.allow_reduce)
        self.assertFalse(snap.allow_new_risk)

    def test_reconciliation_needs_three_converged_observations_for_new_risk(self):
        result = ReconciliationResult(True, True, True, ())
        supervisor = ReconciliationSupervisor(ReconciliationSupervisorPolicy(confirmations_for_new_risk=3))
        self.assertFalse(supervisor.observe(result, observed_at=NOW, now=NOW).allow_new_risk)
        self.assertFalse(supervisor.observe(result, observed_at=NOW+timedelta(seconds=1), now=NOW+timedelta(seconds=1)).allow_new_risk)
        self.assertTrue(supervisor.observe(result, observed_at=NOW+timedelta(seconds=2), now=NOW+timedelta(seconds=2)).allow_new_risk)

    def test_unknown_order_plan_halts_account_and_manual_review(self):
        diff = ReconciliationDiff(DiffKind.UNKNOWN_ORDER, "x", "MISSING", "OPEN", Severity.HALT_ACCOUNT)
        plan = build_reconciliation_plan(ReconciliationResult(False, False, False, (diff,)))
        self.assertIn(ReconciliationAction.HALT_ACCOUNT, plan.actions)
        self.assertTrue(plan.requires_manual_review)

    def test_database_rejects_future_backup(self):
        value = DatabaseReadinessInput("postgresql", "h", "h", True, True, True, True, True, True, NOW+timedelta(minutes=1), NOW)
        result = evaluate_database_readiness(value, now=NOW)
        self.assertIn("BACKUP_VERIFICATION_FROM_FUTURE", result.reasons)

    def test_database_rejects_weak_isolation(self):
        value = DatabaseReadinessInput("postgresql", "h", "h", True, True, True, True, True, True, NOW, NOW, isolation_level="READ COMMITTED")
        self.assertIn("DATABASE_ISOLATION_TOO_WEAK", evaluate_database_readiness(value, now=NOW).reasons)

    def test_database_rejects_nonfinite_probe_metrics(self):
        with self.assertRaises(ValueError):
            DatabaseReadinessInput("postgresql", "h", "h", True, True, True, True, True, True, NOW, NOW, replication_lag_seconds=math.nan)

    def test_postgres_probe_commits_temp_fixture_before_rollback(self):
        class Cursor:
            def __init__(self, conn): self.conn=conn; self.row=None
            def execute(self, sql, params=()):
                if sql == "SELECT 1": self.row=(1,); return
                if sql.startswith("SHOW transaction_isolation"): self.row=("serializable",); return
                if sql.startswith("SHOW server_version"): self.row=("17",); return
                if sql.startswith("SELECT current_database"): self.row=("hedge",); return
                if sql.startswith("CREATE TEMP TABLE") or sql.startswith("DELETE FROM"):
                    self.row=None; return
                if sql.startswith("INSERT INTO hedge_pr_probe_tx"):
                    self.conn.tx_insert=True; return
                if sql.startswith("SELECT count(*) FROM hedge_pr_probe_tx"):
                    self.row=(0 if not self.conn.tx_insert else 1,); return
                if sql.startswith("INSERT INTO hedge_pr_probe_unique"):
                    if self.conn.unique_inserted: raise RuntimeError("duplicate key")
                    self.conn.unique_inserted=True; return
                if sql.startswith("SELECT pg_try_advisory_lock"):
                    self.row=(True,); return
                if sql.startswith("SELECT pg_advisory_unlock"):
                    self.row=(True,); return
                raise AssertionError(sql)
            def fetchone(self): return self.row
        class Conn:
            def __init__(self):
                self.commits=0; self.rollbacks=0; self.tx_insert=False; self.unique_inserted=False
            def cursor(self): return Cursor(self)
            def commit(self): self.commits+=1; self.tx_insert=False; self.unique_inserted=False
            def rollback(self): self.rollbacks+=1; self.tx_insert=False; self.unique_inserted=False
        conn=Conn()
        report=PostgresProbeRunner(conn).run(now=NOW)
        self.assertTrue(report.passed, report.errors)
        self.assertGreaterEqual(conn.commits, 2)
        self.assertGreaterEqual(conn.rollbacks, 2)

    def test_postgres_concurrency_probe_proves_single_writer(self):
        shared = {"holder": None}
        class Cursor:
            def __init__(self, conn): self.conn=conn; self.row=None
            def execute(self, sql, params=()):
                if "pg_backend_pid" in sql:
                    self.row=(self.conn.pid,)
                elif "pg_try_advisory_lock" in sql:
                    if shared["holder"] in {None, self.conn.pid}:
                        shared["holder"]=self.conn.pid; self.row=(True,)
                    else:
                        self.row=(False,)
                elif "pg_advisory_unlock" in sql:
                    ok=shared["holder"]==self.conn.pid
                    if ok: shared["holder"]=None
                    self.row=(ok,)
                else: raise AssertionError(sql)
            def fetchone(self): return self.row
        class Conn:
            def __init__(self,pid): self.pid=pid
            def cursor(self): return Cursor(self)
            def rollback(self): pass
            def close(self): pass
        pids=iter((101,202))
        report=PostgresConcurrencyProbeRunner(lambda: Conn(next(pids))).run(now=NOW)
        self.assertTrue(report.passed)
        self.assertTrue(report.secondary_blocked_while_primary_held)
        self.assertTrue(report.secondary_acquired_after_release)

    def test_health_nan_is_rejected(self):
        args = dict(available_margin_ratio=.5, liquidation_buffer_ratio=.5, unknown_orders=0, position_divergences=0, market_data_age_seconds=0, user_stream_age_seconds=0, loop_p99_ms=1, db_p99_ms=1, model_p99_ms=1, model_fallbacks_1h=0, risk_reject_ratio_1h=.1, memory_growth_ratio_1h=0)
        with self.assertRaises(ValueError):
            HealthSnapshot(**{**args, "db_p99_ms": math.nan})

    def test_shadow_nan_is_rejected(self):
        with self.assertRaises(ValueError):
            ShadowMetrics(timedelta(hours=24), reconciliation_p99_seconds=math.nan)

    def test_canary_nan_is_rejected(self):
        with self.assertRaises(ValueError):
            CanaryRuntime(Decimal("NaN"), Decimal("0"), Decimal("0"), 0, 0)

    def test_halt_account_alert_activates_immediately(self):
        health = HealthSnapshot(.5, .1, 0, 0, 0, 0, 1, 1, 1, 0, .1, 0)
        states = AlertStateTracker(AlertHysteresisPolicy(raise_after=3)).observe(evaluate_health(health))
        self.assertTrue(any(x.code == "LIQUIDATION_BUFFER_LOW" and x.active for x in states))

    def test_warning_hysteresis_does_not_fire_on_single_spike(self):
        health = HealthSnapshot(.5, .5, 0, 0, 0, 0, 999, 1, 1, 0, .1, 0)
        tracker = AlertStateTracker(AlertHysteresisPolicy(raise_after=2))
        first = tracker.observe(evaluate_health(health))
        self.assertFalse(any(x.code == "LOOP_LATENCY" and x.active for x in first))
        second = tracker.observe(evaluate_health(health))
        self.assertTrue(any(x.code == "LOOP_LATENCY" and x.active for x in second))

    def test_model_target_future_fails_closed(self):
        target = ModelTarget(1, NOW+timedelta(seconds=1), Decimal(".1"), Decimal("0"), Decimal(".9"))
        decision = validate_model_target(target, now=NOW)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.long_ratio, Decimal("0"))

    def test_model_target_low_confidence_scale_in_falls_back_previous(self):
        prev = ModelTarget(1, NOW, Decimal(".1"), Decimal("0"), Decimal(".9"))
        target = ModelTarget(2, NOW, Decimal(".2"), Decimal("0"), Decimal(".2"))
        decision = validate_model_target(target, now=NOW, previous=prev)
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.long_ratio, prev.long_ratio)

    def test_model_target_risk_budget_scales_only_after_validation(self):
        target = ModelTarget(1, NOW, Decimal(".2"), Decimal(".1"), Decimal(".9"), Decimal(".5"))
        decision = validate_model_target(target, now=NOW)
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.long_ratio, Decimal(".10"))
        self.assertEqual(decision.short_ratio, Decimal(".05"))

    def test_model_circuit_opens_and_needs_cooldown_successes(self):
        breaker = ModelCircuitBreaker(ModelCircuitPolicy(consecutive_failures_to_open=2, consecutive_successes_to_close=2, cooldown_seconds=10))
        bad = ModelRuntimeDecision(False, "golden", ("bad",))
        good = ModelRuntimeDecision(True, "golden", ())
        self.assertFalse(breaker.observe(bad, now=NOW).open)
        self.assertTrue(breaker.observe(bad, now=NOW+timedelta(seconds=1)).open)
        self.assertTrue(breaker.observe(good, now=NOW+timedelta(seconds=5)).open)
        self.assertFalse(breaker.observe(good, now=NOW+timedelta(seconds=11)).open)
        self.assertFalse(breaker.observe(good, now=NOW+timedelta(seconds=12)).open)

    def test_fallback_registry_requires_approved_profile(self):
        registry = FallbackProfileRegistry((FallbackProfile("golden", "a"*64, False, .1, .1),))
        self.assertFalse(registry.approved("golden"))
        with self.assertRaises(PermissionError):
            registry.resolve("golden")

    def _manifest(self, facts, feature=True):
        return ReplayManifest("binance", "acct", ("BTCUSDT",), NOW, NOW+timedelta(seconds=10), tuple(facts), "a"*64 if feature else None)

    def test_replay_integrity_requires_stream_and_event_coverage(self):
        facts = (RecordedFact(1, NOW, "market", "ORDER", "o1", "a"*64),)
        result = evaluate_replay_integrity(self._manifest(facts))
        self.assertFalse(result.passed)
        self.assertTrue(any(x.startswith("MISSING_STREAM") for x in result.reasons))

    def test_replay_timestamp_regression_is_detected(self):
        facts = (
            RecordedFact(1, NOW+timedelta(seconds=2), "market", "ORDER", "o", "a"*64),
            RecordedFact(2, NOW+timedelta(seconds=1), "user", "POSITION", "p", "b"*64),
            RecordedFact(3, NOW+timedelta(seconds=3), "account", "BALANCE", "b", "c"*64),
        )
        self.assertIn("TIMESTAMP_REGRESSION", evaluate_replay_integrity(self._manifest(facts)).reasons)

    def test_replay_duplicate_identity_is_detected(self):
        facts = (
            RecordedFact(1, NOW, "market", "ORDER", "same", "a"*64),
            RecordedFact(2, NOW+timedelta(seconds=1), "market", "ORDER", "same", "b"*64),
            RecordedFact(3, NOW+timedelta(seconds=2), "user", "POSITION", "p", "c"*64),
            RecordedFact(4, NOW+timedelta(seconds=3), "account", "BALANCE", "b", "d"*64),
        )
        self.assertIn("DUPLICATE_FACT_IDENTITY", evaluate_replay_integrity(self._manifest(facts)).reasons)

    def test_shadow_window_gap_fails_run(self):
        metrics = ShadowMetrics(timedelta(hours=12), restart_recoveries=1, funding_cycles_observed=1)
        windows = (
            ShadowWindow(NOW, NOW+timedelta(hours=12), metrics, source_cursor_start=0, source_cursor_end=100),
            ShadowWindow(NOW+timedelta(hours=13), NOW+timedelta(hours=25), metrics, source_cursor_start=101, source_cursor_end=200),
        )
        result = qualify_shadow_run(windows, target="24h")
        self.assertFalse(result.passed)
        self.assertTrue(any(x.startswith("WINDOW_GAP:") for x in result.reasons))

    def test_fault_campaign_detects_duplicate_scenario(self):
        good = [FaultResult(x, True, 0, True, True, 1) for x in FaultScenario]
        good.append(good[0])
        passed, reasons = evaluate_fault_campaign(good)
        self.assertFalse(passed)
        self.assertIn("DUPLICATE_FAULT_SCENARIO_RESULT", reasons)

    def test_fault_campaign_requires_state_hash_match(self):
        results = [FaultResult(x, True, 0, True, True, 1) for x in FaultScenario]
        results[0] = FaultResult(results[0].scenario, True, 0, True, True, 1, state_hash_match=False)
        passed, reasons = evaluate_fault_campaign(results)
        self.assertFalse(passed)
        self.assertTrue(any("STATE_HASH" in x for x in reasons))

    def test_evidence_store_compare_and_swap(self):
        with tempfile.TemporaryDirectory() as d:
            store = EvidenceLedgerStore(Path(d)/"evidence.json")
            ledger, digest = store.load()
            store.append_record(kind=EvidenceKind.SOURCE_GATES, status=EvidenceStatus.PASS, observed_at=NOW, ttl=timedelta(hours=1), artifact_sha256="a"*64, producer="a", expected_digest=digest)
            with self.assertRaises(EvidenceConcurrencyError):
                store.save_if_unchanged(ledger, expected_digest=digest)

    def test_future_evidence_blocks_stage(self):
        ledger = EvidenceLedger()
        for i, kind in enumerate(sorted(cumulative_requirements(ProductionStage.SOURCE_READY), key=lambda x:x.value)):
            ledger.add(kind=kind, status=EvidenceStatus.PASS, observed_at=NOW+timedelta(minutes=1, microseconds=i), ttl=timedelta(hours=1), artifact_sha256=hashlib.sha256(kind.value.encode()).hexdigest(), producer="future")
        result = StageEvaluator(ledger).evaluate(ProductionStage.SOURCE_READY, now=NOW)
        self.assertFalse(result.passed)
        self.assertTrue(any(x.startswith("EVIDENCE_FROM_FUTURE") for x in result.reasons))


if __name__ == "__main__":
    unittest.main()
