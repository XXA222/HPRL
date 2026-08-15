"""Robust candidate ranking and multi-objective selection helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from math import log


ZERO = Decimal(0)
ONE = Decimal(1)
DEFAULT_TAIL_FRACTION = Decimal("0.25")


def _values(scores: Mapping[str, object]) -> dict[str, Decimal]:
    if not scores:
        raise ValueError("scenario scores cannot be empty")
    output = {name: Decimal(str(value)) for name, value in scores.items()}
    if any(not value.is_finite() for value in output.values()):
        raise ValueError("scenario scores must be finite")
    return output


def weighted_scenario_score(scores: Mapping[str, object], weights: Mapping[str, object]) -> Decimal:
    values = _values(scores)
    if set(values) != set(weights):
        raise ValueError("scenario score and weight keys must match")
    normalized = {name: Decimal(str(weight)) for name, weight in weights.items()}
    if (
        any(weight < ZERO for weight in normalized.values())
        or sum(normalized.values(), ZERO) <= ZERO
    ):
        raise ValueError("scenario weights must be non-negative with positive total")
    total_weight = sum(normalized.values(), ZERO)
    return sum((values[name] * normalized[name] for name in values), ZERO) / total_weight


def worst_case_score(scores: Mapping[str, object]) -> Decimal:
    return min(_values(scores).values())


def cvar_scenario_score(
    scores: Mapping[str, object],
    *,
    tail_fraction: Decimal = DEFAULT_TAIL_FRACTION,
) -> Decimal:
    values = sorted(_values(scores).values())
    if tail_fraction <= ZERO or tail_fraction > ONE:
        raise ValueError("tail_fraction must be in (0, 1]")
    count = max(1, int(len(values) * float(tail_fraction)))
    return sum(values[:count], ZERO) / Decimal(count)


def rank_candidates(scores: Mapping[str, object], *, maximize: bool = True) -> tuple[str, ...]:
    values = _values(scores)
    return tuple(
        name for name,
        _ in sorted(
            values.items(),
            key=lambda item: ((-item[1]) if maximize else item[1], item[0]),
        )
    )


def stability_penalty(fold_scores: Sequence[object], *, coefficient: object = ONE) -> Decimal:
    values = [Decimal(str(value)) for value in fold_scores]
    if not values:
        raise ValueError("fold_scores cannot be empty")
    coeff = Decimal(str(coefficient))
    if coeff < ZERO:
        raise ValueError("coefficient cannot be negative")
    mean = sum(values, ZERO) / Decimal(len(values))
    variance = sum(((value - mean) ** 2 for value in values), ZERO) / Decimal(len(values))
    return coeff * variance.sqrt()


def in_sample_out_of_sample_gap(in_sample: object, out_of_sample: object) -> Decimal:
    return Decimal(str(in_sample)) - Decimal(str(out_of_sample))


def spearman_rank_correlation(left: Sequence[object], right: Sequence[object]) -> Decimal:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("rank series must have equal length of at least two")
    if len(set(left)) != len(left) or len(set(right)) != len(right):
        raise ValueError("rank correlation does not accept ties")
    left_rank = {value: index for index, value in enumerate(sorted(left))}
    right_rank = {value: index for index, value in enumerate(sorted(right))}
    squared = sum((left_rank[a] - right_rank[b]) ** 2 for a, b in zip(left, right, strict=True))
    n = len(left)
    return Decimal(1) - Decimal(6 * squared) / Decimal(n * (n * n - 1))


def selection_entropy(selection_counts: Mapping[str, int]) -> Decimal:
    if not selection_counts or any(value < 0 for value in selection_counts.values()):
        raise ValueError("selection counts must be non-negative and non-empty")
    total = sum(selection_counts.values())
    if total <= 0:
        return ZERO
    entropy = -sum(
        (count / total) * log(count / total)
        for count in selection_counts.values()
        if count
    )
    maximum = log(len(selection_counts)) if len(selection_counts) > 1 else 0.0
    return Decimal(str(entropy / maximum if maximum else 0.0))


def knee_point(points: Sequence[tuple[object, object]]) -> int:
    if len(points) < 3:
        raise ValueError("at least three points are required")
    normalized = [(float(x), float(y)) for x, y in points]
    first, last = normalized[0], normalized[-1]
    dx, dy = last[0] - first[0], last[1] - first[1]
    denominator = (dx * dx + dy * dy) ** 0.5
    if denominator == 0:
        raise ValueError("knee endpoints cannot be identical")
    distances = [
        abs(dy * x - dx * y + last[0] * first[1] - last[1] * first[0]) / denominator
        for x, y in normalized
    ]
    return max(range(1, len(points) - 1), key=lambda index: (distances[index], -index))


def epsilon_pareto(
    points: Mapping[str, Sequence[object]],
    *,
    epsilons: Sequence[object],
    maximize: Sequence[bool],
) -> tuple[str, ...]:
    if not points or len(epsilons) != len(maximize):
        raise ValueError("invalid epsilon Pareto request")
    eps = tuple(Decimal(str(value)) for value in epsilons)
    vectors = {
        name: tuple(Decimal(str(value)) for value in vector)
        for name, vector in points.items()
    }
    width = len(eps)
    if (
        width == 0
        or any(len(vector) != width for vector in vectors.values())
        or any(value < ZERO for value in eps)
    ):
        raise ValueError("Pareto vector dimensions or epsilons are invalid")

    def dominates(left: tuple[Decimal, ...], right: tuple[Decimal, ...]) -> bool:
        no_worse = True
        strictly = False
        for lv, rv, epsilon, direction in zip(left, right, eps, maximize, strict=True):
            oriented_left, oriented_right = (lv, rv) if direction else (-lv, -rv)
            if oriented_left + epsilon < oriented_right:
                no_worse = False
                break
            if oriented_left > oriented_right + epsilon:
                strictly = True
        return no_worse and strictly

    front = []
    for name, vector in sorted(vectors.items()):
        if not any(
            other != name and dominates(other_vector, vector)
            for other, other_vector in vectors.items()
        ):
            front.append(name)
    return tuple(front)
