"""Deterministic Pareto-front and non-dominated ranking utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from decimal import Decimal

from freqtrade.hedge.optimization.types import (
    ObjectiveDirection,
    ObjectiveSpec,
    TrialRecord,
    TrialStatus,
)


def dominates(
    left: Sequence[Decimal],
    right: Sequence[Decimal],
    objectives: Sequence[ObjectiveSpec],
) -> bool:
    if len(left) != len(right) or len(left) != len(objectives):
        raise ValueError("objective vector dimensions do not match")
    no_worse = True
    strictly_better = False
    for lhs, rhs, spec in zip(left, right, objectives, strict=True):
        if spec.direction is ObjectiveDirection.MAXIMIZE:
            no_worse &= lhs >= rhs
            strictly_better |= lhs > rhs
        else:
            no_worse &= lhs <= rhs
            strictly_better |= lhs < rhs
    return no_worse and strictly_better


def pareto_front(
    trials: Iterable[TrialRecord], objectives: Sequence[ObjectiveSpec]
) -> tuple[TrialRecord, ...]:
    candidates = tuple(
        item
        for item in trials
        if item.status is TrialStatus.COMPLETE
        and len(item.objective_values) == len(objectives)
    )
    result = []
    for candidate in candidates:
        if any(
            other.trial_id != candidate.trial_id
            and dominates(other.objective_values, candidate.objective_values, objectives)
            for other in candidates
        ):
            continue
        result.append(candidate)
    return tuple(sorted(result, key=lambda item: item.trial_id))


def non_dominated_ranks(
    trials: Iterable[TrialRecord], objectives: Sequence[ObjectiveSpec]
) -> dict[int, int]:
    remaining = {
        item.trial_id: item
        for item in trials
        if item.status is TrialStatus.COMPLETE
        and len(item.objective_values) == len(objectives)
    }
    ranks: dict[int, int] = {}
    rank = 0
    while remaining:
        front = pareto_front(remaining.values(), objectives)
        if not front:
            raise RuntimeError("could not construct Pareto rank for completed trials")
        for item in front:
            ranks[item.trial_id] = rank
            remaining.pop(item.trial_id)
        rank += 1
    return ranks
