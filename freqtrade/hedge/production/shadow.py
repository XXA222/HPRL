"""24/72 hour read-only/paper/shadow quantitative acceptance."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from math import isfinite


@dataclass(frozen=True, slots=True)
class ShadowMetrics:
    duration: timedelta
    rest_ws_position_divergences: int = 0
    unknown_orders_peak: int = 0
    unresolved_unknown_orders: int = 0
    sequence_gaps_unrecovered: int = 0
    candle_gaps_unrecovered: int = 0
    duplicate_effects: int = 0
    reconciliation_p99_seconds: float = 0.0
    loop_p99_ms: float = 0.0
    db_p99_ms: float = 0.0
    model_p99_ms: float = 0.0
    model_fallbacks: int = 0
    memory_growth_ratio: float = 0.0
    restart_recoveries: int = 0
    restart_recovery_failures: int = 0
    funding_cycles_observed: int = 0
    planner_churn_ratio: float = 0.0
    risk_reject_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.duration <= timedelta(0):
            raise ValueError("shadow duration must be positive")
        integer_fields = (
            "rest_ws_position_divergences", "unknown_orders_peak",
            "unresolved_unknown_orders", "sequence_gaps_unrecovered",
            "candle_gaps_unrecovered", "duplicate_effects", "model_fallbacks",
            "restart_recoveries", "restart_recovery_failures", "funding_cycles_observed",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        float_fields = (
            "reconciliation_p99_seconds", "loop_p99_ms", "db_p99_ms",
            "model_p99_ms", "memory_growth_ratio", "planner_churn_ratio",
            "risk_reject_ratio",
        )
        for name in float_fields:
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in ("reconciliation_p99_seconds", "loop_p99_ms", "db_p99_ms", "model_p99_ms"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("planner_churn_ratio", "risk_reject_ratio"):
            if getattr(self, name) < 0 or getattr(self, name) > 1:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass(frozen=True, slots=True)
class ShadowPolicy:
    max_reconciliation_p99_seconds: float = 5.0
    max_loop_p99_ms: float = 250.0
    max_db_p99_ms: float = 100.0
    max_model_p99_ms: float = 100.0
    max_memory_growth_ratio: float = 0.10
    max_planner_churn_ratio: float = 0.25
    max_risk_reject_ratio: float = 0.50
    require_restart_recovery: bool = True
    require_funding_cycle_for_24h: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_reconciliation_p99_seconds", "max_loop_p99_ms", "max_db_p99_ms",
            "max_model_p99_ms", "max_memory_growth_ratio",
            "max_planner_churn_ratio", "max_risk_reject_ratio",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        for name in ("max_planner_churn_ratio", "max_risk_reject_ratio"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1")


@dataclass(frozen=True, slots=True)
class ShadowQualification:
    target: str
    passed: bool
    reasons: tuple[str, ...]


def qualify_shadow(metrics: ShadowMetrics, *, target: str, policy: ShadowPolicy | None = None) -> ShadowQualification:
    policy = policy or ShadowPolicy()
    required = {"24h": timedelta(hours=24), "72h": timedelta(hours=72)}
    if target not in required:
        raise ValueError("target must be 24h or 72h")
    reasons: list[str] = []
    if metrics.duration < required[target]: reasons.append("SOAK_DURATION_INSUFFICIENT")
    zero_fields = {
        "REST_WS_POSITION_DIVERGENCE": metrics.rest_ws_position_divergences,
        "UNRESOLVED_UNKNOWN_ORDER": metrics.unresolved_unknown_orders,
        "UNRECOVERED_SEQUENCE_GAP": metrics.sequence_gaps_unrecovered,
        "UNRECOVERED_CANDLE_GAP": metrics.candle_gaps_unrecovered,
        "DUPLICATE_EFFECT": metrics.duplicate_effects,
        "RESTART_RECOVERY_FAILURE": metrics.restart_recovery_failures,
    }
    reasons.extend(name for name, value in zero_fields.items() if value != 0)
    if metrics.reconciliation_p99_seconds > policy.max_reconciliation_p99_seconds: reasons.append("RECONCILIATION_P99")
    if metrics.loop_p99_ms > policy.max_loop_p99_ms: reasons.append("LOOP_P99")
    if metrics.db_p99_ms > policy.max_db_p99_ms: reasons.append("DB_P99")
    if metrics.model_p99_ms > policy.max_model_p99_ms: reasons.append("MODEL_P99")
    if metrics.memory_growth_ratio > policy.max_memory_growth_ratio: reasons.append("MEMORY_GROWTH")
    if metrics.planner_churn_ratio > policy.max_planner_churn_ratio: reasons.append("PLANNER_CHURN")
    if metrics.risk_reject_ratio > policy.max_risk_reject_ratio: reasons.append("RISK_REJECT_RATE")
    if policy.require_restart_recovery and metrics.restart_recoveries < 1: reasons.append("NO_RESTART_RECOVERY_EVIDENCE")
    if target == "24h" and policy.require_funding_cycle_for_24h and metrics.funding_cycles_observed < 1: reasons.append("NO_FUNDING_CYCLE")
    if target == "72h" and metrics.funding_cycles_observed < 3: reasons.append("INSUFFICIENT_FUNDING_CYCLES")
    return ShadowQualification(target, not reasons, tuple(reasons))
