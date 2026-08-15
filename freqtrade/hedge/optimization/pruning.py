"""Conservative median pruning for expensive walk-forward evaluations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.optimization.types import ObjectiveDirection


DEFAULT_MINIMUM_IMPROVEMENT = Decimal(0)


@dataclass(frozen=True, slots=True)
class MedianPruningPolicy:
    direction: ObjectiveDirection = ObjectiveDirection.MAXIMIZE
    warmup_steps: int = 1
    minimum_completed_trials: int = 5
    minimum_improvement: Decimal = DEFAULT_MINIMUM_IMPROVEMENT

    def __post_init__(self) -> None:
        if self.warmup_steps < 0:
            raise ValueError("pruner warmup steps cannot be negative")
        if self.minimum_completed_trials < 1:
            raise ValueError("pruner minimum completed trials must be positive")
        if not self.minimum_improvement.is_finite() or self.minimum_improvement < 0:
            raise ValueError("pruner minimum improvement must be finite and non-negative")


def _median(values: Sequence[Decimal]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def should_prune(
    *,
    step: int,
    current_value: Decimal,
    completed_histories: Mapping[int, Sequence[Decimal]],
    policy: MedianPruningPolicy,
) -> bool:
    """Compare a trial only with completed trials that reached the same step."""

    if step < policy.warmup_steps:
        return False
    peers = tuple(
        history[step]
        for history in completed_histories.values()
        if len(history) > step and history[step].is_finite()
    )
    if len(peers) < policy.minimum_completed_trials:
        return False
    median = _median(peers)
    if policy.direction is ObjectiveDirection.MAXIMIZE:
        return current_value + policy.minimum_improvement < median
    return current_value - policy.minimum_improvement > median
