"""Out-of-sample evidence gate for HPRL risk/position-management learning.

A profitable equity curve is not accepted as proof that position management was learned.
The gate requires adaptive executed exposure, acceptable tail risk, and risk-adjusted edge over
simple baselines.  It also treats liquidation/account-bankruptcy evidence as a first-class failure
signal so environment autoreset cannot hide catastrophic risk.
"""

from __future__ import annotations

import heapq
import math
from array import array
from dataclasses import dataclass, fields
from statistics import fmean
from typing import Mapping, Sequence


_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class RiskLearningTrace:
    """Chronological executed-policy trace from one out-of-sample evaluation.

    Series are aligned by decision step. ``equity_return[t]`` is the net account return realized
    after executing the exposure represented by ``gross_margin[t]`` and ``level_index[t]``.
    For multi-symbol dual-leg runs, ``level_index`` should be the maximum executed tier across
    symbols/legs for the decision step. ``liquidations`` counts account-bankruptcy/autoreset events.
    """

    equity_return: Sequence[float]
    gross_margin: Sequence[float]
    drawdown: Sequence[float]
    level_index: Sequence[int]
    turnover: Sequence[float] = ()
    projected: Sequence[bool] = ()
    liquidations: int = 0


@dataclass(frozen=True, slots=True)
class BaselineRiskMetrics:
    name: str
    net_return: float
    max_drawdown: float
    cvar: float
    turnover: float = 0.0
    liquidations: int = 0


@dataclass(frozen=True, slots=True)
class RiskLearningAcceptanceConfig:
    level_count: int = 5
    min_steps: int = 512
    min_distinct_levels: int = 3
    max_single_level_fraction: float = 0.95
    min_active_fraction: float = 0.02
    max_projection_fraction: float = 0.20
    stress_drawdown_quantile: float = 0.80
    max_stress_to_calm_margin_ratio: float = 1.05
    min_post_loss_derisk_fraction: float = 0.45
    min_scale_in_success_rate: float = 0.50
    max_liquidations: int = 0
    drawdown_penalty: float = 1.0
    cvar_penalty: float = 5.0
    turnover_penalty: float = 0.001
    liquidation_penalty: float = 1.0
    min_baseline_utility_edge: float = 0.0
    require_baselines: bool = True

    def __post_init__(self) -> None:
        if self.level_count < 2:
            raise ValueError("level_count must be >= 2")
        if self.min_steps < 8:
            raise ValueError("min_steps must be >= 8")
        if not 1 <= self.min_distinct_levels <= self.level_count:
            raise ValueError("min_distinct_levels is outside the configured level range")
        if self.max_liquidations < 0:
            raise ValueError("max_liquidations cannot be negative")
        fractions = (
            self.max_single_level_fraction,
            self.min_active_fraction,
            self.max_projection_fraction,
            self.stress_drawdown_quantile,
            self.min_post_loss_derisk_fraction,
            self.min_scale_in_success_rate,
        )
        if any(not math.isfinite(value) for value in fractions):
            raise ValueError("acceptance fractions must be finite")
        if not 0 < self.max_single_level_fraction <= 1:
            raise ValueError("max_single_level_fraction must be in (0, 1]")
        if not 0 <= self.min_active_fraction <= 1:
            raise ValueError("min_active_fraction must be in [0, 1]")
        if not 0 <= self.max_projection_fraction <= 1:
            raise ValueError("max_projection_fraction must be in [0, 1]")
        if not 0 < self.stress_drawdown_quantile < 1:
            raise ValueError("stress_drawdown_quantile must be in (0, 1)")
        if not 0 <= self.min_post_loss_derisk_fraction <= 1:
            raise ValueError("min_post_loss_derisk_fraction must be in [0, 1]")
        if not 0 <= self.min_scale_in_success_rate <= 1:
            raise ValueError("min_scale_in_success_rate must be in [0, 1]")
        if self.max_stress_to_calm_margin_ratio <= 0:
            raise ValueError("max_stress_to_calm_margin_ratio must be positive")
        penalties = (
            self.drawdown_penalty,
            self.cvar_penalty,
            self.turnover_penalty,
            self.liquidation_penalty,
        )
        if any(not math.isfinite(value) or value < 0 for value in penalties):
            raise ValueError("utility penalties must be finite and non-negative")
        if not math.isfinite(self.min_baseline_utility_edge):
            raise ValueError("min_baseline_utility_edge must be finite")


@dataclass(frozen=True, slots=True)
class RiskLearningAcceptanceReport:
    schema: str
    verdict: str
    reasons: tuple[str, ...]
    steps: int
    distinct_levels: int
    level_occupancy: tuple[float, ...]
    normalized_level_entropy: float
    active_fraction: float
    projection_fraction: float | None
    stress_drawdown_threshold: float
    calm_mean_margin: float
    stress_mean_margin: float
    stress_to_calm_margin_ratio: float
    post_loss_derisk_fraction: float
    scale_in_success_rate: float
    realized_net_return: float
    realized_max_drawdown: float
    realized_cvar: float
    realized_turnover: float
    realized_liquidations: int
    policy_utility: float
    best_baseline_name: str | None
    best_baseline_utility: float | None
    baseline_utility_edge: float | None
    behavioral_pass: bool
    economic_pass: bool
    baseline_pass: bool | None


def _finite_float_series(
    name: str,
    values: Sequence[float],
    *,
    expected: int | None = None,
) -> array:
    """Validate into compact C doubles instead of duplicating million-row traces as tuples."""
    result = array("d", (float(value) for value in values))
    if expected is not None and len(result) != expected:
        raise ValueError(f"{name} length must be {expected}, got {len(result)}")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _level_series(values: Sequence[int], *, expected: int, level_count: int) -> array:
    typecode = "B" if level_count <= 256 else "I"
    result = array(typecode)
    for value in values:
        if isinstance(value, bool):
            raise ValueError("level_index must contain integer-valued levels")
        numeric = float(value)
        converted = int(numeric)
        if not math.isfinite(numeric) or numeric != converted:
            raise ValueError("level_index must contain integer-valued levels")
        if not 0 <= converted < level_count:
            raise ValueError("level_index is outside configured level range")
        result.append(converted)
    if len(result) != expected:
        raise ValueError(f"level_index length must be {expected}, got {len(result)}")
    return result


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _max_drawdown_from_returns(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    maximum = 0.0
    for value in returns:
        equity *= max(_EPS, 1.0 + float(value))
        peak = max(peak, equity)
        maximum = max(maximum, 1.0 - equity / peak)
    return maximum


def _net_return(returns: Sequence[float]) -> float:
    log_total = 0.0
    for value in returns:
        if value <= -1.0:
            return -1.0
        log_total += math.log1p(float(value))
    return math.expm1(max(-700.0, min(700.0, log_total)))


def _cvar(returns: Sequence[float], alpha: float = 0.05) -> float:
    if len(returns) == 0:
        return 0.0
    tail_count = max(1, math.ceil(len(returns) * alpha))
    tail = heapq.nsmallest(tail_count, returns)
    return max(0.0, -fmean(tail))


def _normalized_entropy(counts: Sequence[int]) -> float:
    total = sum(int(value) for value in counts)
    active_bins = sum(1 for value in counts if value > 0)
    if total <= 0 or active_bins <= 1:
        return 0.0
    entropy = 0.0
    for count in counts:
        if count <= 0:
            continue
        probability = count / total
        entropy -= probability * math.log(probability)
    return entropy / math.log(active_bins)


def _utility(metrics: BaselineRiskMetrics, config: RiskLearningAcceptanceConfig) -> float:
    return (
        float(metrics.net_return)
        - config.drawdown_penalty * float(metrics.max_drawdown)
        - config.cvar_penalty * float(metrics.cvar)
        - config.turnover_penalty * float(metrics.turnover)
        - config.liquidation_penalty * int(metrics.liquidations)
    )


def evaluate_risk_learning(
    trace: RiskLearningTrace,
    *,
    baselines: Sequence[BaselineRiskMetrics] = (),
    config: RiskLearningAcceptanceConfig | None = None,
) -> RiskLearningAcceptanceReport:
    """Return strict evidence that a policy learned risk/position management.

    ``PASS`` requires adaptive/non-degenerate behavior, positive risk-adjusted economics, no
    disallowed liquidation events, and baseline evidence when configured. Missing baseline evidence
    is ``INCONCLUSIVE`` rather than a false positive.
    """
    cfg = config or RiskLearningAcceptanceConfig()
    if not isinstance(trace.liquidations, int) or isinstance(trace.liquidations, bool):
        raise ValueError("liquidations must be an integer")
    if trace.liquidations < 0:
        raise ValueError("liquidations cannot be negative")

    returns = _finite_float_series("equity_return", trace.equity_return)
    steps = len(returns)
    if steps < cfg.min_steps:
        raise ValueError(f"risk-learning trace needs at least {cfg.min_steps} steps")
    margin = _finite_float_series("gross_margin", trace.gross_margin, expected=steps)
    drawdown = _finite_float_series("drawdown", trace.drawdown, expected=steps)
    levels = _level_series(trace.level_index, expected=steps, level_count=cfg.level_count)
    if any(value < 0 for value in margin):
        raise ValueError("gross_margin cannot be negative")
    if any(not 0 <= value <= 1.0 + 1e-9 for value in drawdown):
        raise ValueError("drawdown must be in [0, 1]")

    has_turnover = len(trace.turnover) > 0
    turnover_values = (
        _finite_float_series("turnover", trace.turnover, expected=steps)
        if has_turnover
        else None
    )
    if turnover_values is not None and any(value < 0 for value in turnover_values):
        raise ValueError("turnover cannot be negative")

    has_projection = len(trace.projected) > 0
    projected_values: array | None = None
    if has_projection:
        if len(trace.projected) != steps:
            raise ValueError("projected length must match the trace")
        projected_values = array("b", (1 if bool(value) else 0 for value in trace.projected))

    counts = [0] * cfg.level_count
    for value in levels:
        counts[value] += 1
    occupancy = tuple(value / steps for value in counts)
    distinct = sum(1 for value in counts if value > 0)
    entropy = _normalized_entropy(counts)
    active_fraction = sum(1 for value in margin if value > _EPS) / steps
    projection_fraction = (
        sum(projected_values) / steps if projected_values is not None else None
    )

    stress_threshold = _quantile(drawdown, cfg.stress_drawdown_quantile)
    stress_sum = 0.0
    stress_count = 0
    calm_sum = 0.0
    calm_count = 0
    for current_margin, current_drawdown in zip(margin, drawdown, strict=True):
        if current_drawdown >= stress_threshold:
            stress_sum += current_margin
            stress_count += 1
        else:
            calm_sum += current_margin
            calm_count += 1
    overall_mean = fmean(margin)
    stress_mean = stress_sum / stress_count if stress_count else overall_mean
    calm_mean = calm_sum / calm_count if calm_count else overall_mean
    stress_ratio = (
        stress_mean / calm_mean
        if calm_mean > _EPS
        else (0.0 if stress_mean <= _EPS else math.inf)
    )

    post_loss_events = 0
    post_loss_derisk = 0
    scale_in_events = 0
    scale_in_success = 0
    for index in range(steps - 1):
        current_margin = margin[index]
        next_margin = margin[index + 1]
        if returns[index] < 0.0 and current_margin > _EPS:
            post_loss_events += 1
            if next_margin <= current_margin + 1e-12:
                post_loss_derisk += 1
        if next_margin > current_margin + 1e-12:
            scale_in_events += 1
            if returns[index + 1] > 0.0:
                scale_in_success += 1
    derisk_fraction = post_loss_derisk / post_loss_events if post_loss_events else 1.0
    scale_success = scale_in_success / scale_in_events if scale_in_events else 0.0

    realized_return = _net_return(returns)
    realized_max_dd = _max_drawdown_from_returns(returns)
    realized_cvar = _cvar(returns)
    realized_turnover = sum(turnover_values) if turnover_values is not None else 0.0
    policy_metrics = BaselineRiskMetrics(
        name="hprl_policy",
        net_return=realized_return,
        max_drawdown=realized_max_dd,
        cvar=realized_cvar,
        turnover=realized_turnover,
        liquidations=trace.liquidations,
    )
    policy_utility = _utility(policy_metrics, cfg)

    reasons: list[str] = []
    if distinct < cfg.min_distinct_levels:
        reasons.append("insufficient_level_diversity")
    if max(occupancy) > cfg.max_single_level_fraction:
        reasons.append("single_level_collapse")
    if active_fraction < cfg.min_active_fraction:
        reasons.append("mostly_flat_policy")
    if projection_fraction is not None and projection_fraction > cfg.max_projection_fraction:
        reasons.append("excessive_risk_projection")
    stress_ok = stress_ratio <= cfg.max_stress_to_calm_margin_ratio
    derisk_ok = derisk_fraction >= cfg.min_post_loss_derisk_fraction
    if not (stress_ok or derisk_ok):
        reasons.append("no_stress_derisk_evidence")
    if scale_in_events == 0:
        reasons.append("no_scale_in_events")
    elif scale_success < cfg.min_scale_in_success_rate:
        reasons.append("scale_in_selectivity_below_gate")

    behavior_reasons = {
        "insufficient_level_diversity",
        "single_level_collapse",
        "mostly_flat_policy",
        "excessive_risk_projection",
        "no_stress_derisk_evidence",
        "no_scale_in_events",
        "scale_in_selectivity_below_gate",
    }
    behavioral_pass = not any(reason in behavior_reasons for reason in reasons)

    liquidation_ok = trace.liquidations <= cfg.max_liquidations
    if not liquidation_ok:
        reasons.append("liquidation_observed")
    if realized_return <= 0.0:
        reasons.append("nonpositive_out_of_sample_return")
    if policy_utility <= 0.0:
        reasons.append("nonpositive_risk_adjusted_utility")
    economic_pass = realized_return > 0.0 and policy_utility > 0.0 and liquidation_ok

    baseline_name: str | None = None
    best_baseline_utility: float | None = None
    baseline_edge: float | None = None
    baseline_pass: bool | None = None
    if len(baselines) > 0:
        baseline_pairs = [(item.name, _utility(item, cfg)) for item in baselines]
        baseline_name, best_baseline_utility = max(baseline_pairs, key=lambda item: item[1])
        baseline_edge = policy_utility - best_baseline_utility
        baseline_pass = baseline_edge >= cfg.min_baseline_utility_edge
        if not baseline_pass:
            reasons.append("no_risk_adjusted_edge_over_baselines")
    elif cfg.require_baselines:
        reasons.append("baseline_evidence_missing")

    if behavioral_pass and economic_pass and (baseline_pass is True or not cfg.require_baselines):
        verdict = "PASS"
    elif cfg.require_baselines and baseline_pass is None and behavioral_pass and economic_pass:
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"

    return RiskLearningAcceptanceReport(
        schema="hprl-risk-learning-acceptance-v2",
        verdict=verdict,
        reasons=tuple(dict.fromkeys(reasons)),
        steps=steps,
        distinct_levels=distinct,
        level_occupancy=occupancy,
        normalized_level_entropy=entropy,
        active_fraction=active_fraction,
        projection_fraction=projection_fraction,
        stress_drawdown_threshold=stress_threshold,
        calm_mean_margin=calm_mean,
        stress_mean_margin=stress_mean,
        stress_to_calm_margin_ratio=stress_ratio,
        post_loss_derisk_fraction=derisk_fraction,
        scale_in_success_rate=scale_success,
        realized_net_return=realized_return,
        realized_max_drawdown=realized_max_dd,
        realized_cvar=realized_cvar,
        realized_turnover=realized_turnover,
        realized_liquidations=trace.liquidations,
        policy_utility=policy_utility,
        best_baseline_name=baseline_name,
        best_baseline_utility=best_baseline_utility,
        baseline_utility_edge=baseline_edge,
        behavioral_pass=behavioral_pass,
        economic_pass=economic_pass,
        baseline_pass=baseline_pass,
    )


def report_as_dict(report: RiskLearningAcceptanceReport) -> Mapping[str, object]:
    """Stable JSON-serializable projection used by acceptance tooling."""
    return {field.name: getattr(report, field.name) for field in fields(report)}
