"""Out-of-sample evidence gate for HPRL risk/position-management learning.

The gate deliberately separates three questions:

1. Is the executed action stream valid and non-degenerate?
2. Does exposure adapt when the account is under stress instead of remaining static?
3. Does the learned policy beat simple static/random position baselines on a risk-adjusted utility?

A profitable equity curve alone is never accepted as proof that position management was learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean
from typing import Mapping, Sequence


_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class RiskLearningTrace:
    """Executed-policy trace from one chronological out-of-sample evaluation.

    All series are aligned by decision step. ``equity_return[t]`` is the net account return realized
    after executing the position represented by ``gross_margin[t]`` and ``level_index[t]``.
    ``drawdown[t]`` is measured after the same realization. For multi-symbol dual-leg runs,
    ``level_index`` should be the maximum executed tier across symbols/legs for that decision step.
    """

    equity_return: Sequence[float]
    gross_margin: Sequence[float]
    drawdown: Sequence[float]
    level_index: Sequence[int]
    turnover: Sequence[float] = ()
    projected: Sequence[bool] = ()


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
    policy_utility: float
    best_baseline_name: str | None
    best_baseline_utility: float | None
    baseline_utility_edge: float | None
    behavioral_pass: bool
    economic_pass: bool
    baseline_pass: bool | None


def _finite_float_series(
    name: str, values: Sequence[float], *, expected: int | None = None
) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if expected is not None and len(result) != expected:
        raise ValueError(f"{name} length must be {expected}, got {len(result)}")
    if any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _strict_int_series(
    values: Sequence[int], *, expected: int, level_count: int
) -> tuple[int, ...]:
    result: list[int] = []
    for value in values:
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError("level_index must contain integers")
        if not 0 <= value < level_count:
            raise ValueError("level_index is outside configured level range")
        result.append(value)
    if len(result) != expected:
        raise ValueError(f"level_index length must be {expected}, got {len(result)}")
    return tuple(result)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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
    if not returns:
        return 0.0
    tail_count = max(1, math.ceil(len(returns) * alpha))
    return max(0.0, -fmean(sorted(float(value) for value in returns)[:tail_count]))


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
    """Evaluate whether an HPRL policy has credible risk-management learning evidence.

    ``PASS`` requires both adaptive/non-degenerate behavior and economic evidence. When baseline
    evidence is required but missing, the result is ``INCONCLUSIVE`` rather than a false positive.
    """

    cfg = config or RiskLearningAcceptanceConfig()
    returns = _finite_float_series("equity_return", trace.equity_return)
    steps = len(returns)
    if steps < cfg.min_steps:
        raise ValueError(f"risk-learning trace needs at least {cfg.min_steps} steps")
    margin = _finite_float_series("gross_margin", trace.gross_margin, expected=steps)
    drawdown = _finite_float_series("drawdown", trace.drawdown, expected=steps)
    levels = _strict_int_series(trace.level_index, expected=steps, level_count=cfg.level_count)
    if any(value < 0 for value in margin):
        raise ValueError("gross_margin cannot be negative")
    if any(not 0 <= value <= 1.0 + 1e-9 for value in drawdown):
        raise ValueError("drawdown must be in [0, 1]")

    turnover_values = (
        _finite_float_series("turnover", trace.turnover, expected=steps)
        if trace.turnover
        else tuple(0.0 for _ in range(steps))
    )
    if any(value < 0 for value in turnover_values):
        raise ValueError("turnover cannot be negative")
    projected_values: tuple[bool, ...] | None = None
    if trace.projected:
        projected_values = tuple(bool(value) for value in trace.projected)
        if len(projected_values) != steps:
            raise ValueError("projected length must match the trace")

    counts = [0] * cfg.level_count
    for value in levels:
        counts[value] += 1
    occupancy = tuple(value / steps for value in counts)
    distinct = sum(1 for value in counts if value > 0)
    max_occupancy = max(occupancy)
    entropy = _normalized_entropy(counts)
    active_fraction = sum(1 for value in margin if value > _EPS) / steps
    projection_fraction = (
        sum(1 for value in projected_values if value) / steps
        if projected_values is not None
        else None
    )

    stress_threshold = _quantile(drawdown, cfg.stress_drawdown_quantile)
    stress_margin = [m for m, dd in zip(margin, drawdown, strict=True) if dd >= stress_threshold]
    calm_margin = [m for m, dd in zip(margin, drawdown, strict=True) if dd < stress_threshold]
    stress_mean = fmean(stress_margin) if stress_margin else fmean(margin)
    calm_mean = fmean(calm_margin) if calm_margin else fmean(margin)
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
    realized_turnover = sum(turnover_values)
    policy_metrics = BaselineRiskMetrics(
        name="hprl_policy",
        net_return=realized_return,
        max_drawdown=realized_max_dd,
        cvar=realized_cvar,
        turnover=realized_turnover,
        liquidations=0,
    )
    policy_utility = _utility(policy_metrics, cfg)

    reasons: list[str] = []
    if distinct < cfg.min_distinct_levels:
        reasons.append("insufficient_level_diversity")
    if max_occupancy > cfg.max_single_level_fraction:
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

    blocking_behavior_reasons = {
        "insufficient_level_diversity",
        "single_level_collapse",
        "mostly_flat_policy",
        "excessive_risk_projection",
        "no_stress_derisk_evidence",
        "no_scale_in_events",
        "scale_in_selectivity_below_gate",
    }
    behavioral_pass = not any(reason in blocking_behavior_reasons for reason in reasons)
    economic_pass = realized_return > 0.0 and policy_utility > 0.0
    if realized_return <= 0.0:
        reasons.append("nonpositive_out_of_sample_return")
    if policy_utility <= 0.0:
        reasons.append("nonpositive_risk_adjusted_utility")

    baseline_name: str | None = None
    best_baseline_utility: float | None = None
    baseline_edge: float | None = None
    baseline_pass: bool | None = None
    if baselines:
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
        schema="hprl-risk-learning-acceptance-v1",
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
    return {field: getattr(report, field) for field in report.__dataclass_fields__}
