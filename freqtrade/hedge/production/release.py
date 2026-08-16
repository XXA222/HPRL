"""Aggregate production readiness report and promotion decision."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .canary import CanaryDecision
from .contracts import Capability, GateResult, ProductionStage
from .database import DatabaseReadinessResult
from .faults import FaultResult, evaluate_fault_campaign
from .model_governance import ModelRuntimeDecision
from .observability import Alert
from .policy import StageEvaluator
from .shadow import ShadowQualification


@dataclass(frozen=True, slots=True)
class ProductionReadinessReport:
    target_stage: ProductionStage
    stage_gate: GateResult
    database_ready: bool
    fault_campaign_ready: bool
    shadow_ready: bool
    observability_ready: bool
    model_ready: bool
    canary_ready: bool
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.reasons


def build_production_readiness_report(
    *,
    evaluator: StageEvaluator,
    target_stage: ProductionStage,
    now: datetime,
    database: DatabaseReadinessResult | None = None,
    fault_results: tuple[FaultResult, ...] = (),
    shadow: ShadowQualification | None = None,
    alerts: tuple[Alert, ...] = (),
    model: ModelRuntimeDecision | None = None,
    deterministic_fallback_ready: bool = False,
    canary: CanaryDecision | None = None,
) -> ProductionReadinessReport:
    gate = evaluator.evaluate(target_stage, now=now)
    reasons: list[str] = []
    if not gate.passed: reasons.append("STAGE_EVIDENCE_GATE")
    database_required = target_stage in {
        ProductionStage.DATABASE_READY,
        ProductionStage.REPLAY_READY,
        ProductionStage.SHADOW_24H,
        ProductionStage.SHADOW_72H,
        ProductionStage.TESTNET_READY,
        ProductionStage.LIVE_CANDIDATE,
        ProductionStage.LIVE_READY,
    }
    db_ready = database is not None and database.passed if database_required else database is None or database.passed
    if database_required and database is None:
        reasons.append("DATABASE_RUNTIME_SNAPSHOT_REQUIRED")
    elif database is not None and not database.passed:
        reasons.append("DATABASE_NOT_READY")
    fault_required = target_stage in {
        ProductionStage.SHADOW_24H,
        ProductionStage.SHADOW_72H,
        ProductionStage.TESTNET_READY,
        ProductionStage.LIVE_CANDIDATE,
        ProductionStage.LIVE_READY,
    }
    fault_ready = not fault_required
    if fault_results:
        fault_ready, _ = evaluate_fault_campaign(fault_results)
        if not fault_ready: reasons.append("FAULT_CAMPAIGN_NOT_READY")
    elif fault_required:
        reasons.append("FAULT_CAMPAIGN_RUNTIME_SNAPSHOT_REQUIRED")
    shadow_required = target_stage in {
        ProductionStage.SHADOW_24H,
        ProductionStage.SHADOW_72H,
        ProductionStage.TESTNET_READY,
        ProductionStage.LIVE_CANDIDATE,
        ProductionStage.LIVE_READY,
    }
    shadow_ready = shadow is not None and shadow.passed if shadow_required else shadow is None or shadow.passed
    if shadow_required and shadow is None:
        reasons.append("SHADOW_RUNTIME_SNAPSHOT_REQUIRED")
    elif shadow is not None and not shadow.passed:
        reasons.append("SHADOW_NOT_READY")
    critical_alerts = [a for a in alerts if a.severity.value.startswith("HALT")]
    observability_ready = not critical_alerts
    if critical_alerts: reasons.append("CRITICAL_ALERTS_PRESENT")
    model_required = target_stage in {ProductionStage.LIVE_CANDIDATE, ProductionStage.LIVE_READY}
    model_ready = (
        model is not None and (model.use_model or deterministic_fallback_ready)
        if model_required
        else model is None or model.use_model or deterministic_fallback_ready
    )
    if model_required and model is None:
        reasons.append("MODEL_RUNTIME_SNAPSHOT_REQUIRED")
    elif model is not None and not model.use_model and not deterministic_fallback_ready:
        reasons.append("MODEL_NOT_DEPLOYABLE_AND_FALLBACK_NOT_READY")
    canary_required = target_stage is ProductionStage.LIVE_READY
    canary_ready = canary is not None and canary.allowed if canary_required else canary is None or canary.allowed
    if canary_required and canary is None:
        reasons.append("CANARY_RUNTIME_SNAPSHOT_REQUIRED")
    elif canary is not None and not canary.allowed:
        reasons.append("CANARY_NOT_READY")
    return ProductionReadinessReport(target_stage, gate, db_ready, fault_ready, shadow_ready, observability_ready, model_ready, canary_ready, tuple(reasons))
