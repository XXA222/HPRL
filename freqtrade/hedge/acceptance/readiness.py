from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from freqtrade.hedge.acceptance.models import HardMetrics, RuntimeStage
from freqtrade.hedge.acceptance.stream import StreamAcceptanceState


@dataclass(frozen=True, slots=True)
class ProductionReadiness:
    ready: bool
    stage: str
    reasons: tuple[str, ...]
    required_soak: timedelta


def required_soak_for_stage(stage: str) -> timedelta:
    normalized = stage.strip().lower()
    values = {
        "smoke": timedelta(seconds=60),
        "1h": timedelta(hours=1),
        "6h": timedelta(hours=6),
        "24h": timedelta(hours=24),
        "72h": timedelta(hours=72),
    }
    if normalized not in values:
        raise ValueError("stage must be one of: smoke, 1h, 6h, 24h, 72h")
    return values[normalized]


def evaluate_readiness(
    *,
    hard_metrics: HardMetrics,
    stream_state: StreamAcceptanceState,
    observed_duration: timedelta,
    target_stage: str,
) -> ProductionReadiness:
    required = required_soak_for_stage(target_stage)
    reasons: list[str] = []
    if not hard_metrics.passed:
        reasons.extend(sorted(hard_metrics.failures()))
    if stream_state.stage is not RuntimeStage.READY or not stream_state.new_risk_enabled:
        reasons.append(f"STREAM_NOT_READY:{stream_state.reason}")
    if observed_duration < required:
        reasons.append("SOAK_DURATION_INSUFFICIENT")
    return ProductionReadiness(not reasons, target_stage.lower(), tuple(reasons), required)
