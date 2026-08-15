"""Fail-closed feasibility constraints for Hedge optimization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.optimization.metrics import require_metrics
from freqtrade.hedge.optimization.types import ConstraintSpec


@dataclass(frozen=True, slots=True)
class ConstraintEvaluation:
    feasible: bool
    violations: tuple[str, ...]
    total_violation: Decimal


def evaluate_constraints(
    metrics: Mapping[str, Decimal],
    constraints: Sequence[ConstraintSpec],
) -> ConstraintEvaluation:
    if not constraints:
        return ConstraintEvaluation(True, (), Decimal(0))
    require_metrics(metrics, tuple(item.metric for item in constraints))
    violations: list[str] = []
    total = Decimal(0)
    for constraint in constraints:
        value = metrics[constraint.metric]
        if constraint.minimum is not None and value < constraint.minimum:
            delta = constraint.minimum - value
            total += delta
            violations.append(
                f"{constraint.metric}={value} below minimum={constraint.minimum} by {delta}"
            )
        if constraint.maximum is not None and value > constraint.maximum:
            delta = value - constraint.maximum
            total += delta
            violations.append(
                f"{constraint.metric}={value} above maximum={constraint.maximum} by {delta}"
            )
    return ConstraintEvaluation(not violations, tuple(violations), total)
