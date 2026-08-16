"""Canonical stateful runtime bundle for production safety decisions."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from .control import ControlAction, ProductionControlPlane
from .incidents import IncidentLedger
from .model_governance import ModelCircuitBreaker, ModelCircuitSnapshot, ModelRuntimeDecision
from .observability import AlertState, AlertStateTracker, HealthSnapshot, ObservabilityPolicy, evaluate_health
from .reconciliation import ReconciliationResult
from .reconciliation_runtime import ReconciliationSupervisor, ReconciliationSupervisorSnapshot
from .runtime_supervisor import ProductionRuntimeSafetySupervisor, RuntimeSafetyInput, RuntimeSafetySnapshot


@dataclass(slots=True)
class ProductionRuntimeBundle:
    control: ProductionControlPlane
    incidents: IncidentLedger
    reconciliation: ReconciliationSupervisor
    alerts: AlertStateTracker
    model_circuit: ModelCircuitBreaker
    safety: ProductionRuntimeSafetySupervisor
    _reconciliation_snapshot: ReconciliationSupervisorSnapshot | None = None
    _alert_states: tuple[AlertState, ...] = ()
    _model_snapshot: ModelCircuitSnapshot | None = None
    _market_data_fresh: bool = False
    _risk_data_fresh: bool = False
    _fallback_ready: bool = False

    @classmethod
    def create(cls) -> "ProductionRuntimeBundle":
        return cls(
            ProductionControlPlane(),
            IncidentLedger(),
            ReconciliationSupervisor(),
            AlertStateTracker(),
            ModelCircuitBreaker(),
            ProductionRuntimeSafetySupervisor(),
        )

    def observe_reconciliation(
        self,
        result: ReconciliationResult,
        *,
        observed_at: datetime,
        now: datetime,
    ) -> RuntimeSafetySnapshot:
        self._reconciliation_snapshot = self.reconciliation.observe(
            result, observed_at=observed_at, now=now
        )
        return self._refresh(now)

    def observe_health(
        self,
        health: HealthSnapshot,
        *,
        now: datetime,
        policy: ObservabilityPolicy | None = None,
    ) -> RuntimeSafetySnapshot:
        self._alert_states = self.alerts.observe(evaluate_health(health, policy))
        return self._refresh(now)

    def observe_model(
        self,
        decision: ModelRuntimeDecision,
        *,
        now: datetime,
        fallback_ready: bool = False,
    ) -> RuntimeSafetySnapshot:
        self._model_snapshot = self.model_circuit.observe(decision, now=now)
        self._fallback_ready = bool(fallback_ready)
        return self._refresh(now)

    def open_incident(self, incident, *, now: datetime) -> RuntimeSafetySnapshot:
        self.incidents.open(incident)
        return self._refresh(now)

    def close_incident(
        self,
        incident_id: str,
        *,
        now: datetime,
        readiness_passed: bool,
        reconciliation_converged: bool,
        operator_acknowledged: bool,
    ) -> RuntimeSafetySnapshot:
        self.incidents.close_checked(
            incident_id,
            closed_at=now,
            readiness_passed=readiness_passed,
            reconciliation_converged=reconciliation_converged,
            operator_acknowledged=operator_acknowledged,
        )
        return self._refresh(now)

    def set_freshness(
        self,
        *,
        market_data_fresh: bool,
        risk_data_fresh: bool,
        now: datetime,
    ) -> RuntimeSafetySnapshot:
        self._market_data_fresh = bool(market_data_fresh)
        self._risk_data_fresh = bool(risk_data_fresh)
        return self._refresh(now)

    def apply_control(
        self,
        action: ControlAction,
        *,
        actor: str,
        reason: str,
        readiness_passed: bool,
        reconciliation_converged: bool,
        observed_at: datetime,
    ) -> RuntimeSafetySnapshot:
        self.control.apply(
            action,
            actor=actor,
            reason=reason,
            readiness_passed=readiness_passed,
            reconciliation_converged=reconciliation_converged,
            observed_at=observed_at,
        )
        return self._refresh(observed_at)

    def safety_snapshot(self) -> RuntimeSafetySnapshot:
        return self.safety.snapshot()

    def _refresh(self, now: datetime) -> RuntimeSafetySnapshot:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        reconciliation = self._reconciliation_snapshot
        if reconciliation is None:
            reconciliation = ReconciliationSupervisorSnapshot(
                consecutive_converged=0,
                allow_new_risk=False,
                allow_reduce=False,
                first_nonconverged_at=now.astimezone(UTC),
                last_observed_at=None,
                reasons=("RECONCILIATION_NOT_OBSERVED",),
            )
        return self.safety.observe(
            RuntimeSafetyInput(
                control_mode=self.control.mode,
                reconciliation=reconciliation,
                active_alerts=self._alert_states,
                incident_blocks_new_risk=self.incidents.blocks_new_risk,
                incident_blocks_account=self.incidents.blocks_account,
                market_data_fresh=self._market_data_fresh,
                risk_data_fresh=self._risk_data_fresh,
                model_circuit=self._model_snapshot,
                deterministic_fallback_ready=self._fallback_ready,
            ),
            now=now,
        )
