from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal

from .contracts import BacktestEvaluation
from .decimal_utils import ZERO, to_decimal


def objective_vector(
    evaluation: BacktestEvaluation,
    objectives: Mapping[str, str],
) -> tuple[Decimal, ...]:
    vector: list[Decimal] = []
    for metric, direction in objectives.items():
        value = to_decimal(evaluation.metrics.get(metric, ZERO), field=metric)
        if direction == "maximize":
            vector.append(value)
        elif direction == "minimize":
            vector.append(-value)
        else:
            raise ValueError(f"invalid objective direction for {metric}: {direction}")
    return tuple(vector)


def dominates(left: tuple[Decimal, ...], right: tuple[Decimal, ...]) -> bool:
    if len(left) != len(right):
        raise ValueError("objective vectors must have equal length")
    return all(a >= b for a, b in zip(left, right, strict=True)) and any(
        a > b for a, b in zip(left, right, strict=True)
    )


def pareto_front(
    evaluations: Iterable[BacktestEvaluation],
    objectives: Mapping[str, str],
) -> tuple[BacktestEvaluation, ...]:
    feasible = tuple(item for item in evaluations if item.feasible)
    vectors = {item.candidate.candidate_id: objective_vector(item, objectives) for item in feasible}
    front = []
    for candidate in feasible:
        current = vectors[candidate.candidate.candidate_id]
        if not any(
            other is not candidate and dominates(vectors[other.candidate.candidate_id], current)
            for other in feasible
        ):
            front.append(candidate)
    return tuple(sorted(front, key=lambda item: item.candidate.candidate_id))


def crowding_distance(
    evaluations: Iterable[BacktestEvaluation],
    objectives: Mapping[str, str],
) -> dict[str, Decimal]:
    items = list(evaluations)
    if not items:
        return {}
    distances = {item.candidate.candidate_id: ZERO for item in items}
    for index, _ in enumerate(objectives):
        ordered = sorted(items, key=lambda item: objective_vector(item, objectives)[index])
        distances[ordered[0].candidate.candidate_id] = Decimal("Infinity")
        distances[ordered[-1].candidate.candidate_id] = Decimal("Infinity")
        low = objective_vector(ordered[0], objectives)[index]
        high = objective_vector(ordered[-1], objectives)[index]
        span = high - low
        if span == ZERO or len(ordered) < 3:
            continue
        for position in range(1, len(ordered) - 1):
            key = ordered[position].candidate.candidate_id
            if distances[key].is_infinite():
                continue
            previous = objective_vector(ordered[position - 1], objectives)[index]
            following = objective_vector(ordered[position + 1], objectives)[index]
            distances[key] += (following - previous) / span
    return distances
