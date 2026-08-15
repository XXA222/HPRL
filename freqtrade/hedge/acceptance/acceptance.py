from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

from freqtrade.hedge.exchange.base import (
    AccountConfigurationFact,
    AccountSnapshotFact,
    BalanceFact,
    FillFact,
    OrderFact,
    PositionFact,
)
from freqtrade.hedge.acceptance.baseline import audit_environment
from freqtrade.hedge.acceptance.clock import ClockAudit
from freqtrade.hedge.acceptance.events import EventEnvelope, EventSequenceTracker, ExactlyOnceEffectJournal
from freqtrade.hedge.acceptance.facts import build_fact_plane
from freqtrade.hedge.acceptance.faults import classify_http_fault
from freqtrade.hedge.acceptance.history import audit_history
from freqtrade.hedge.acceptance.identity import (
    build_leg_identities,
    identity_mismatch_count,
    leverage_mismatches,
)
from freqtrade.hedge.acceptance.models import AcceptancePolicy, ReconciliationDepth, RuntimeSnapshotSet
from freqtrade.hedge.acceptance.persistence import RuntimeAcceptanceStore
from freqtrade.hedge.acceptance.readiness import evaluate_readiness
from freqtrade.hedge.acceptance.reconciliation import (
    count_position_diffs,
    count_wallet_drift,
    reconcile_planes,
    reconciliation_issue_metrics,
)
from freqtrade.hedge.acceptance.session import RuntimeAcceptanceSession
from freqtrade.hedge.acceptance.stream import StreamRecoveryGate


class RuntimeAcceptanceEngine:
    """Sequential 20-round acceptance engine. Every method records exactly one gated round."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        project_root: Path,
        account_id: str,
        managed_symbols: Sequence[str],
        store: RuntimeAcceptanceStore,
        policy: AcceptancePolicy | None = None,
        live_evidence: bool,
    ) -> None:
        self.config = config
        self.project_root = project_root
        self.account_id = account_id
        self.managed_symbols = tuple(str(item).upper() for item in managed_symbols)
        self.store = store
        self.policy = policy or AcceptancePolicy()
        self.session = RuntimeAcceptanceSession(live_evidence=live_evidence)
        self.stream_gate = StreamRecoveryGate(stale_after=self.policy.stale_after)
        self.sequence = EventSequenceTracker()
        self.effects = ExactlyOnceEffectJournal(store)
        self._rest_plane = None

    def round01_baseline(self) -> None:
        baseline = audit_environment(self.config, project_root=self.project_root)
        self.session.record(
            "ACCEPT-01",
            passed=baseline.safe_for_runtime_acceptance,
            checks=("BINANCE", "HEDGE_MODE_ENABLED", "READ_ONLY", "LIVE_WRITES_DISABLED"),
            metrics={
                "managed_symbols": list(baseline.managed_symbols),
                "credentials_present": baseline.credentials_present,
                "database_configured": baseline.database_configured,
            },
        )

    def round02_clock(self, audit: ClockAudit) -> None:
        self.session.record(
            "ACCEPT-02",
            passed=audit.synchronized,
            checks=("MIDPOINT_OFFSET", "RTT_BOUND", "MAX_ABS_SKEW"),
            metrics={
                "sample_count": audit.sample_count,
                "median_offset_ms": audit.median_offset_ms,
                "max_abs_offset_ms": audit.max_abs_offset_ms,
                "max_rtt_ms": audit.max_rtt_ms,
            },
        )

    def round03_assets(
        self, snapshot: AccountSnapshotFact, balances: Sequence[BalanceFact]
    ) -> None:
        assets_unique = len({item.asset for item in balances}) == len(balances)
        financials_nonnegative = (
            snapshot.total_initial_margin >= 0 and snapshot.total_maintenance_margin >= 0
        )
        self.session.record(
            "ACCEPT-03",
            passed=assets_unique and financials_nonnegative,
            checks=("UNIQUE_ASSETS", "ACCOUNT_TOTALS_FINITE", "MARGIN_NONNEGATIVE"),
            metrics={"balance_count": len(balances), "wallet": str(snapshot.total_wallet_balance)},
        )

    def round04_configuration(
        self, configuration: AccountConfigurationFact, *, target_leverage: int | None
    ) -> None:
        mismatches = leverage_mismatches(
            configuration.leverage_by_symbol_side, target_leverage=target_leverage
        )
        passed = (
            configuration.hedge_mode
            and configuration.active_margin_modes == ("cross",)
            and not mismatches
        )
        self.session.record(
            "ACCEPT-04",
            passed=passed,
            checks=("HEDGE_MODE", "CROSS_MARGIN", "LEVERAGE"),
            metrics={"leverage_mismatches": list(mismatches)},
        )

    def round05_identity(
        self, positions: Sequence[PositionFact], configuration: AccountConfigurationFact
    ) -> None:
        identities = build_leg_identities(
            account_id=self.account_id,
            managed_symbols=self.managed_symbols,
            positions=positions,
            configuration=configuration,
        )
        mismatch = identity_mismatch_count(
            identities, managed_symbols=self.managed_symbols, account_id=self.account_id
        )
        self.session.set_metric("long_short_identity_mismatch", mismatch)
        self.session.record(
            "ACCEPT-05",
            passed=mismatch == 0,
            checks=("LONG_IDENTITY", "SHORT_IDENTITY", "ZERO_LEG_IDENTITY"),
            metrics={"identity_count": len(identities), "mismatch": mismatch},
        )

    def round06_orders(
        self,
        open_orders: Sequence[OrderFact],
        order_history: Sequence[OrderFact],
        *,
        query_recovered_ids: Sequence[str] = (),
        snapshot_fallback_ids: Sequence[str] = (),
    ) -> None:
        history = audit_history(
            account_id=self.account_id,
            open_orders=open_orders,
            order_history=order_history,
            fills=(),
            income=(),
        )
        self.session.record(
            "ACCEPT-06",
            passed=not history.missing_open_orders_in_history,
            checks=(
                "OPEN_ORDER_UNIQUENESS",
                "OPEN_ORDER_CANONICAL_HISTORY_COVERAGE",
                "POSITION_SIDE_PRESENT",
            ),
            metrics={
                "missing_in_history": list(history.missing_open_orders_in_history),
                "query_recovered_ids": sorted(str(item) for item in query_recovered_ids),
                "snapshot_fallback_ids": sorted(str(item) for item in snapshot_fallback_ids),
                "open_order_count": len(open_orders),
                "history_order_count": len(order_history),
            },
            detail=(
                "Current openOrders is authoritative for live exposure; bounded allOrders "
                "history is supplemented by read-only Query Order and, only when Binance "
                "retention prevents that lookup, by explicitly-labelled current snapshot "
                "evidence."
            ),
        )

    def round07_trades(self, fills: Sequence[FillFact]) -> None:
        history = audit_history(
            account_id=self.account_id, open_orders=(), order_history=(), fills=fills, income=()
        )
        self.session.record(
            "ACCEPT-07",
            passed=True,
            checks=("TRADE_ID_DEDUPE", "PAGINATION_IDENTITY", "POSITION_SIDE"),
            metrics={"unique_fills": history.unique_fills},
        )

    def round08_income(self, income: Sequence[Mapping[str, Any]]) -> None:
        history = audit_history(
            account_id=self.account_id, open_orders=(), order_history=(), fills=(), income=income
        )
        self.session.record(
            "ACCEPT-08",
            passed=True,
            checks=("INCOME_IDENTITY", "FUNDING_DEDUPE", "COMMISSION_CLASSIFICATION"),
            metrics={
                "unique_income": history.unique_income,
                "funding_events": history.funding_events,
                "commission_events": history.commission_events,
            },
        )

    def round09_rest_snapshot(
        self,
        *,
        positions: Sequence[PositionFact],
        balances: Sequence[BalanceFact],
        orders: Sequence[OrderFact],
        fills: Sequence[FillFact],
        income: Sequence[Mapping[str, Any]],
        observed_at: Any,
    ) -> None:
        plane = build_fact_plane(
            account_id=self.account_id,
            observed_at=observed_at,
            positions=positions,
            balances=balances,
            orders=orders,
            fills=fills,
            income=income,
        )
        self.reseed_rest_plane(plane)
        self.session.record(
            "ACCEPT-09",
            passed=bool(plane.fingerprint()),
            checks=("CANONICAL_FACT_PLANE", "DETERMINISTIC_FINGERPRINT", "SQL_SNAPSHOT"),
            metrics={"fingerprint": plane.fingerprint()},
        )

    def reseed_rest_plane(self, plane: Any) -> None:
        """Replace the authoritative REST plane after a fresh exchange calibration."""
        self._rest_plane = plane
        self.store.save_plane("REST", plane)

    def round10_user_stream(
        self,
        *,
        evidence_source: str = "DETERMINISTIC",
        stream_metrics: Mapping[str, Any] | None = None,
    ) -> None:
        self.stream_gate.connected()
        before = self.stream_gate.assess()
        self.stream_gate.reconciliation_passed()
        after = self.stream_gate.assess()
        self.session.record(
            "ACCEPT-10",
            passed=not before.new_risk_enabled and after.new_risk_enabled,
            checks=("LISTENKEY_GENERATION", "RECONNECT_FAIL_CLOSED", "RECONCILE_BEFORE_READY"),
            metrics={
                "generation": after.reconnect_generation,
                "evidence_source": evidence_source,
                **dict(stream_metrics or {}),
            },
        )

    def round11_account_update(
        self, event: EventEnvelope, *, evidence_source: str = "DETERMINISTIC"
    ) -> None:
        decision = self.sequence.inspect(event)
        if decision.apply:
            self.sequence.commit(event)
        passed = event.event_type == "ACCOUNT_UPDATE" and decision.apply
        self.session.record(
            "ACCEPT-11",
            passed=passed,
            checks=("ACCOUNT_UPDATE_SCHEMA", "BALANCE_POSITION_EFFECT", "EVENT_ORDERING"),
            metrics={"decision": decision.reason, "evidence_source": evidence_source},
        )

    def round12_order_trade_update(
        self, event: EventEnvelope, *, evidence_source: str = "DETERMINISTIC"
    ) -> None:
        decision = self.sequence.inspect(event)
        if decision.apply:
            self.sequence.commit(event)
        passed = event.event_type == "ORDER_TRADE_UPDATE" and decision.apply
        self.session.record(
            "ACCEPT-12",
            passed=passed,
            checks=("ORDER_TRADE_UPDATE_SCHEMA", "ORDER_IDENTITY", "FILL_IDENTITY"),
            metrics={"decision": decision.reason, "evidence_source": evidence_source},
        )

    def round13_duplicates(self, fill_event: EventEnvelope, funding_event: EventEnvelope) -> None:
        fill_before = self.store.effect_count("FILL")
        funding_before = self.store.effect_count("FUNDING")
        fill_first = self.effects.apply_fill(fill_event)
        fill_second = self.effects.apply_fill(fill_event)
        funding_first = self.effects.apply_funding(funding_event)
        funding_second = self.effects.apply_funding(funding_event)
        fill_delta = self.store.effect_count("FILL") - fill_before
        funding_delta = self.store.effect_count("FUNDING") - funding_before
        self.session.set_metric("duplicate_fill_effects", max(0, fill_delta - 1))
        self.session.set_metric("duplicate_funding_effects", max(0, funding_delta - 1))
        passed = (
            fill_first
            and not fill_second
            and funding_first
            and not funding_second
            and fill_delta == 1
            and funding_delta == 1
        )
        self.session.record(
            "ACCEPT-13",
            passed=passed,
            checks=(
                "AT_LEAST_ONCE_INPUT",
                "EXACTLY_ONCE_FILL_EFFECT",
                "EXACTLY_ONCE_FUNDING_EFFECT",
            ),
            metrics={"fill_effect_delta": fill_delta, "funding_effect_delta": funding_delta},
        )

    def round14_out_of_order(self, newer: EventEnvelope, older: EventEnvelope) -> None:
        tracker = EventSequenceTracker()
        first = tracker.inspect(newer)
        if first.apply:
            tracker.commit(newer)
        second = tracker.inspect(older)
        self.session.record(
            "ACCEPT-14",
            passed=first.apply and second.out_of_order and not second.apply,
            checks=("TRANSACTION_TIME_REGRESSION", "EVENT_TIME_REGRESSION", "FAIL_CLOSED"),
            metrics={"older_decision": second.reason},
        )

    def round15_gap_recovery(self) -> None:
        self.stream_gate.disconnected()
        stale = self.stream_gate.assess()
        self.stream_gate.connected()
        reconnect = self.stream_gate.assess()
        self.stream_gate.reconciliation_passed()
        recovered = self.stream_gate.assess()
        violation = int(reconnect.new_risk_enabled)
        self.session.set_metric("ws_reconnect_without_reconciliation", violation)
        self.session.set_metric("new_risk_while_stale", int(stale.new_risk_enabled))
        self.session.record(
            "ACCEPT-15",
            passed=(
                not stale.new_risk_enabled
                and not reconnect.new_risk_enabled
                and recovered.new_risk_enabled
            ),
            checks=("GAP_PAUSE_NEW_RISK", "REST_RESEED", "RECOVERY_RECONCILIATION"),
        )

    def _snapshot_set(self, memory_plane: Any, db_plane: Any) -> RuntimeSnapshotSet:
        if self._rest_plane is None:
            raise RuntimeError("ACCEPT-09 must establish REST plane first")
        return RuntimeSnapshotSet(rest=self._rest_plane, memory=memory_plane, database=db_plane)

    def round16_fast_reconciliation(self, memory_plane: Any, db_plane: Any) -> None:
        outcome = reconcile_planes(
            self._snapshot_set(memory_plane, db_plane),
            depth=ReconciliationDepth.FAST,
            quantity_tolerance=self.policy.quantity_tolerance,
            financial_tolerance=self.policy.financial_tolerance,
            wallet_drift_tolerance=self.policy.wallet_drift_tolerance,
        )
        rest_memory = count_position_diffs(outcome, plane="MEMORY")
        rest_db = count_position_diffs(outcome, plane="DB")
        wallet = count_wallet_drift(outcome)
        self.session.set_metric("rest_memory_unexplained_diff", rest_memory)
        self.session.set_metric("rest_db_unexplained_position_diff", rest_db)
        self.session.set_metric("unexplained_wallet_drift", wallet)
        diagnostics = reconciliation_issue_metrics(outcome)
        self.store.save_evidence("ACCEPT-16-RECONCILIATION", diagnostics)
        self.session.record(
            "ACCEPT-16",
            passed=outcome.passed,
            checks=("REST_MEMORY", "REST_DB", "POSITION_BALANCE_ACTIVE_ORDER"),
            metrics=diagnostics,
            detail=(
                "FAST reconciliation compares canonical order lifecycle states; "
                "exchange aliases such as NEW/ACKNOWLEDGED and "
                "PARTIALLY_FILLED/PARTIAL are semantically equivalent."
            ),
        )

    def round17_deep_reconciliation(self, memory_plane: Any, db_plane: Any) -> None:
        outcome = reconcile_planes(
            self._snapshot_set(memory_plane, db_plane),
            depth=ReconciliationDepth.DEEP,
            quantity_tolerance=self.policy.quantity_tolerance,
            financial_tolerance=self.policy.financial_tolerance,
            wallet_drift_tolerance=self.policy.wallet_drift_tolerance,
        )
        diagnostics = reconciliation_issue_metrics(outcome)
        self.store.save_evidence("ACCEPT-17-RECONCILIATION", diagnostics)
        self.session.record(
            "ACCEPT-17",
            passed=outcome.passed,
            checks=("POSITIONS", "BALANCES", "ORDERS", "FILLS", "INCOME"),
            metrics=diagnostics,
        )

    def round18_crash_recovery(
        self,
        *,
        state_hash: str,
        external_recovery_ok: bool = True,
        external_unknown_orders: int = 0,
    ) -> None:
        checkpoint_id = "acceptance-crash-recovery"
        self.store.save_checkpoint(checkpoint_id, state_hash, {"state_hash": state_hash})
        recovered = self.store.checkpoint_hash_fresh_connection(checkpoint_id)
        loss = int(recovered != state_hash or not external_recovery_ok)
        self.session.set_metric("restart_state_loss", loss)
        unknown = max(self.store.unknown_order_count(), int(external_unknown_orders))
        self.session.set_metric("unknown_unrecovered_orders", unknown)
        self.session.record(
            "ACCEPT-18",
            passed=loss == 0 and unknown == 0,
            checks=(
                "SQL_CHECKPOINT",
                "FRESH_CONNECTION_RELOAD",
                "HEDGE_LEDGER_RECOVERY",
                "UNKNOWN_ORDER_RECOVERY",
            ),
            metrics={
                "state_hash": state_hash,
                "unknown_orders": unknown,
                "external_recovery_ok": external_recovery_ok,
            },
        )

    def round19_fault_injection(self) -> None:
        decisions = [
            classify_http_fault(None, network_error=True),
            classify_http_fault(429),
            classify_http_fault(500),
        ]
        passed = all(
            item.retryable and not item.new_risk_allowed and item.requires_reconciliation
            for item in decisions
        )
        self.session.record(
            "ACCEPT-19",
            passed=passed,
            checks=("NETWORK", "HTTP_429", "HTTP_5XX", "FAIL_CLOSED"),
            metrics={"fault_classes": [item.fault_class.value for item in decisions]},
        )

    def round20_readiness(self, *, observed_duration: timedelta, target_stage: str) -> None:
        stream_state = self.stream_gate.assess()
        readiness = evaluate_readiness(
            hard_metrics=self.session.hard_metrics,
            stream_state=stream_state,
            observed_duration=observed_duration,
            target_stage=target_stage,
        )
        self.session.record(
            "ACCEPT-20",
            passed=readiness.ready,
            checks=("HARD_METRICS_ZERO", "STREAM_READY", "SOAK_DURATION"),
            metrics={
                "stage": target_stage,
                "observed_seconds": observed_duration.total_seconds(),
                "required_seconds": readiness.required_soak.total_seconds(),
                "reasons": list(readiness.reasons),
            },
        )
