from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from math import erf, log, sqrt

from .decimal_utils import ONE, ZERO, to_decimal


@dataclass(frozen=True, slots=True)
class BacktestOverfitResult:
    combinations: int
    below_median_count: int
    probability_of_backtest_overfitting: Decimal
    median_test_rank_percentile: Decimal
    median_logit: Decimal


@dataclass(frozen=True, slots=True)
class DeflatedSharpeResult:
    observed_sharpe: Decimal
    expected_max_sharpe: Decimal
    standard_error: Decimal
    probability_sharpe_is_genuine: Decimal


def _median(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return ZERO
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def probability_of_backtest_overfitting(
    *,
    train_scores: Mapping[str, Sequence[Decimal | int | str]],
    test_scores: Mapping[str, Sequence[Decimal | int | str]],
) -> BacktestOverfitResult:
    candidate_ids = tuple(sorted(train_scores))
    if candidate_ids != tuple(sorted(test_scores)):
        raise ValueError("train and test score matrices must contain the same candidates")
    if len(candidate_ids) < 2:
        raise ValueError("PBO requires at least two candidates")
    fold_count = len(train_scores[candidate_ids[0]])
    if fold_count < 1:
        raise ValueError("PBO requires at least one fold")
    if any(
        len(train_scores[item]) != fold_count or len(test_scores[item]) != fold_count
        for item in candidate_ids
    ):
        raise ValueError("all PBO score rows must have equal fold count")
    percentiles: list[Decimal] = []
    logits: list[Decimal] = []
    below = 0
    for fold in range(fold_count):
        selected = max(candidate_ids, key=lambda item: to_decimal(train_scores[item][fold]))
        ordered_test = sorted(
            candidate_ids,
            key=lambda item: (to_decimal(test_scores[item][fold]), item),
        )
        rank = ordered_test.index(selected) + 1
        percentile = Decimal(rank) / Decimal(len(candidate_ids))
        percentiles.append(percentile)
        if percentile <= Decimal("0.5"):
            below += 1
        clipped = min(max(percentile, Decimal("1e-12")), ONE - Decimal("1e-12"))
        logits.append(Decimal(str(log(float(clipped / (ONE - clipped))))))
    return BacktestOverfitResult(
        combinations=fold_count,
        below_median_count=below,
        probability_of_backtest_overfitting=Decimal(below) / Decimal(fold_count),
        median_test_rank_percentile=_median(percentiles),
        median_logit=_median(logits),
    )


def deflated_sharpe_probability(
    *,
    observed_sharpe: Decimal,
    trials: int,
    observations: int,
    skewness: Decimal = ZERO,
    excess_kurtosis: Decimal = ZERO,
) -> DeflatedSharpeResult:
    if trials < 1 or observations < 2:
        raise ValueError("deflated Sharpe requires trials >= 1 and observations >= 2")
    if any(not item.is_finite() for item in (observed_sharpe, skewness, excess_kurtosis)):
        raise ValueError("deflated Sharpe inputs must be finite")
    # Expected maximum of N approximately normal strategy Sharpes.
    euler_gamma = 0.5772156649015329
    if trials == 1:
        expected_max = ZERO
    else:
        z = sqrt(2.0 * log(float(trials)))
        correction = (log(log(float(trials))) + log(4.0 * 3.141592653589793)) / (2.0 * z)
        expected_max = Decimal(str(z - correction + euler_gamma / z))
    sr = float(observed_sharpe)
    variance_term = max(
        1e-18,
        1.0 - float(skewness) * sr + ((float(excess_kurtosis) + 2.0) / 4.0) * sr * sr,
    )
    standard_error = Decimal(str(sqrt(variance_term / (observations - 1))))
    z_score = (observed_sharpe - expected_max) / standard_error
    probability = Decimal(str(0.5 * (1.0 + erf(float(z_score) / sqrt(2.0)))))
    return DeflatedSharpeResult(
        observed_sharpe=observed_sharpe,
        expected_max_sharpe=expected_max,
        standard_error=standard_error,
        probability_sharpe_is_genuine=min(max(probability, ZERO), ONE),
    )


def selection_entropy(candidate_ids: Sequence[str]) -> Decimal:
    if not candidate_ids:
        return ZERO
    counts = Counter(candidate_ids)
    total = Decimal(len(candidate_ids))
    entropy = 0.0
    for candidate_id in sorted(counts):
        probability = float(Decimal(counts[candidate_id]) / total)
        entropy -= probability * log(probability)
    maximum = log(len(counts)) if len(counts) > 1 else 0.0
    return Decimal(str(entropy / maximum if maximum > 0 else 0.0))
