"""Robust aggregation across walk-forward folds, regimes, and stress scenarios."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal


ZERO = Decimal(0)
DEFAULT_STABILITY_PENALTY = Decimal(1)


def _mean(values: Sequence[Decimal]) -> Decimal:
    return sum(values, ZERO) / Decimal(len(values))


def _median(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def _population_std(values: Sequence[Decimal]) -> Decimal:
    mean = _mean(values)
    variance = sum(((value - mean) ** 2 for value in values), ZERO) / Decimal(len(values))
    return variance.sqrt()


def aggregate_metric_sets(
    metric_sets: Sequence[Mapping[str, Decimal]],
    *,
    required_metrics: Sequence[str] = (),
) -> dict[str, Decimal]:
    if not metric_sets:
        raise ValueError("at least one metric set is required")
    common = set(metric_sets[0])
    for item in metric_sets[1:]:
        common &= set(item)
    missing = sorted(set(required_metrics) - common)
    if missing:
        raise ValueError(f"fold metrics are missing required fields: {', '.join(missing)}")
    output: dict[str, Decimal] = {}
    for name in sorted(common):
        values = tuple(item[name] for item in metric_sets)
        if any(not value.is_finite() for value in values):
            raise ValueError(f"fold metric {name} contains non-finite values")
        output[name] = _mean(values)
        output[f"{name}__median"] = _median(values)
        output[f"{name}__min"] = min(values)
        output[f"{name}__max"] = max(values)
        # Backward-compatible lower/upper aliases.  Risk constraints should use
        # explicit __max while return objectives commonly use __min.
        output[f"{name}__worst"] = min(values)
        output[f"{name}__best"] = max(values)
        output[f"{name}__std"] = _population_std(values)
        output[f"{name}__range"] = max(values) - min(values)
    output["fold_count"] = Decimal(len(metric_sets))
    return output


def robustness_score(
    aggregate: Mapping[str, Decimal],
    *,
    return_metric: str = "net_return",
    drawdown_metric: str = "max_drawdown",
    stability_penalty: Decimal = DEFAULT_STABILITY_PENALTY,
) -> Decimal:
    median = aggregate[f"{return_metric}__median"]
    worst = aggregate[f"{return_metric}__worst"]
    dispersion = aggregate[f"{return_metric}__std"]
    drawdown = aggregate.get(drawdown_metric, ZERO)
    return median + worst - stability_penalty * dispersion - abs(drawdown)
