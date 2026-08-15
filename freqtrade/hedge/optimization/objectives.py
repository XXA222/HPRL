"""Single- and multi-objective scoring contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal

from freqtrade.hedge.optimization.metrics import require_metrics
from freqtrade.hedge.optimization.types import ObjectiveDirection, ObjectiveSpec


def objective_values(
    metrics: Mapping[str, Decimal], objectives: Sequence[ObjectiveSpec]
) -> tuple[Decimal, ...]:
    if not objectives:
        raise ValueError("at least one objective is required")
    require_metrics(metrics, tuple(item.metric for item in objectives))
    return tuple(metrics[item.metric] for item in objectives)


def scalar_score(
    metrics: Mapping[str, Decimal], objectives: Sequence[ObjectiveSpec]
) -> Decimal:
    """Return a higher-is-better weighted score using exact decimal arithmetic."""

    values = objective_values(metrics, objectives)
    score = Decimal(0)
    for spec, value in zip(objectives, values, strict=True):
        oriented = value if spec.direction is ObjectiveDirection.MAXIMIZE else -value
        score += oriented * spec.weight
    if not score.is_finite():
        raise ValueError("scalar objective score is non-finite")
    return score


def lexicographic_key(
    values: Sequence[Decimal], objectives: Sequence[ObjectiveSpec]
) -> tuple[Decimal, ...]:
    if len(values) != len(objectives):
        raise ValueError("objective value count does not match objective specifications")
    return tuple(
        value if spec.direction is ObjectiveDirection.MAXIMIZE else -value
        for value, spec in zip(values, objectives, strict=True)
    )
