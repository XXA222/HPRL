"""Deterministic parameter-search utilities for Hedge research."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from random import Random
from statistics import fmean
from typing import Any


@dataclass(frozen=True, slots=True)
class SearchDimension:
    name: str
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.values:
            raise ValueError("search dimension requires name and values")
        if len(set(map(repr, self.values))) != len(self.values):
            raise ValueError("search dimension values must be unique")


@dataclass(frozen=True, slots=True)
class ObjectiveWeight:
    name: str
    weight: float
    maximize: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip() or not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("objective weight must be finite and positive")


def grid_candidates(
    dimensions: Sequence[SearchDimension],
    *,
    limit: int | None = None,
) -> tuple[dict[str, Any], ...]:
    if not dimensions:
        raise ValueError("at least one search dimension is required")
    if len({item.name for item in dimensions}) != len(dimensions):
        raise ValueError("search dimension names must be unique")
    if limit is not None and limit < 1:
        raise ValueError("candidate limit must be positive")
    keys = tuple(item.name for item in dimensions)
    products = itertools.product(*(item.values for item in dimensions))
    rows = (dict(zip(keys, values, strict=True)) for values in products)
    return tuple(itertools.islice(rows, limit)) if limit is not None else tuple(rows)


def _candidate_at_index(
    dimensions: Sequence[SearchDimension],
    index: int,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    remainder = index
    for dimension in reversed(dimensions):
        remainder, offset = divmod(remainder, len(dimension.values))
        values[dimension.name] = dimension.values[offset]
    return {dimension.name: values[dimension.name] for dimension in dimensions}


def random_candidates(
    dimensions: Sequence[SearchDimension],
    *,
    trials: int,
    seed: int,
) -> tuple[dict[str, Any], ...]:
    if trials < 1 or not dimensions:
        raise ValueError("trials and search dimensions must be positive")
    if len({item.name for item in dimensions}) != len(dimensions):
        raise ValueError("search dimension names must be unique")
    total = math.prod(len(item.values) for item in dimensions)
    if trials >= total:
        return grid_candidates(dimensions)
    rng = Random(seed)  # noqa: S311 - deterministic research sampling
    indices = sorted(rng.sample(range(total), trials))
    return tuple(_candidate_at_index(dimensions, index) for index in indices)


def scalar_score(metrics: Mapping[str, float], objectives: Sequence[ObjectiveWeight]) -> float:
    if not objectives:
        raise ValueError("at least one objective is required")
    score = 0.0
    total_weight = sum(item.weight for item in objectives)
    for objective in objectives:
        if objective.name not in metrics:
            raise ValueError(f"missing objective metric: {objective.name}")
        value = float(metrics[objective.name])
        if not math.isfinite(value):
            raise ValueError("objective metrics must be finite")
        score += (value if objective.maximize else -value) * objective.weight
    return score / total_weight


def constraints_satisfied(
    metrics: Mapping[str, float],
    constraints: Mapping[str, tuple[str, float]],
) -> tuple[bool, tuple[str, ...]]:
    violations: list[str] = []
    for name, (operator, threshold) in constraints.items():
        if name not in metrics:
            violations.append(f"missing:{name}")
            continue
        value = float(metrics[name])
        threshold_value = float(threshold)
        if not math.isfinite(value) or not math.isfinite(threshold_value):
            raise ValueError("constraint metrics and thresholds must be finite")
        if operator == "<=" and value > threshold_value:
            violations.append(f"{name}>{threshold_value}")
        elif operator == ">=" and value < threshold_value:
            violations.append(f"{name}<{threshold_value}")
        elif operator not in {"<=", ">="}:
            raise ValueError(f"unsupported constraint operator: {operator}")
    return not violations, tuple(violations)


def _dominates(
    left: Mapping[str, float],
    right: Mapping[str, float],
    objectives: Sequence[ObjectiveWeight],
) -> bool:
    comparisons = []
    for objective in objectives:
        if objective.name not in left or objective.name not in right:
            raise ValueError(f"missing pareto objective metric: {objective.name}")
        left_value = float(left[objective.name])
        right_value = float(right[objective.name])
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            raise ValueError("pareto objective metrics must be finite")
        comparisons.append(
            left_value >= right_value if objective.maximize else left_value <= right_value
        )
    strict = any(
        float(left[item.name]) != float(right[item.name])
        for item in objectives
    )
    return all(comparisons) and strict


def pareto_front(
    rows: Sequence[Mapping[str, float]],
    objectives: Sequence[ObjectiveWeight],
) -> tuple[int, ...]:
    if not rows or not objectives:
        raise ValueError("pareto front requires rows and objectives")
    selected: list[int] = []
    for index, row in enumerate(rows):
        if any(
            _dominates(other, row, objectives)
            for other_index, other in enumerate(rows)
            if other_index != index
        ):
            continue
        selected.append(index)
    return tuple(selected)


def rank_trials(
    rows: Sequence[Mapping[str, float]],
    objectives: Sequence[ObjectiveWeight],
) -> tuple[int, ...]:
    if not rows:
        raise ValueError("trial ranking requires rows")
    scored = [(scalar_score(row, objectives), index) for index, row in enumerate(rows)]
    return tuple(index for _, index in sorted(scored, reverse=True))


def top_k_diverse(
    candidates: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    k: int,
) -> tuple[int, ...]:
    if len(candidates) != len(scores) or k < 1 or not candidates:
        raise ValueError("candidate diversity inputs are invalid")
    numeric_scores = tuple(float(item) for item in scores)
    if any(not math.isfinite(item) for item in numeric_scores):
        raise ValueError("candidate diversity scores must be finite")
    ranked = sorted(range(len(scores)), key=numeric_scores.__getitem__, reverse=True)
    selected: list[int] = []
    signatures: set[tuple[tuple[str, str], ...]] = set()
    for index in ranked:
        signature = tuple(
            sorted((str(key), repr(value)) for key, value in candidates[index].items())
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        selected.append(index)
        if len(selected) == k:
            break
    return tuple(selected)


def early_stop(values: Sequence[float], *, patience: int, minimum_improvement: float = 0.0) -> bool:
    series = tuple(float(item) for item in values)
    if (
        patience < 1
        or not math.isfinite(float(minimum_improvement))
        or minimum_improvement < 0
        or any(not math.isfinite(item) for item in series)
    ):
        raise ValueError("early-stop inputs are invalid")
    if len(series) <= patience:
        return False
    best_before = max(series[:-patience])
    best_recent = max(series[-patience:])
    return best_recent <= best_before + minimum_improvement


def aggregate_fold_scores(values: Sequence[float], *, stability_penalty: float = 0.0) -> float:
    series = tuple(float(item) for item in values)
    if (
        not series
        or not math.isfinite(float(stability_penalty))
        or stability_penalty < 0
        or any(not math.isfinite(item) for item in series)
    ):
        raise ValueError("fold scores and non-negative finite penalty are required")
    mean = fmean(series)
    spread = max(series) - min(series)
    return mean - stability_penalty * spread
