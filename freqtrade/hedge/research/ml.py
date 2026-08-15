"""ML experiment metrics, promotion contracts, and drift diagnostics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean


def _validate_ml_split_ratios(train_ratio: float, validation_ratio: float) -> None:
    if not math.isfinite(train_ratio) or not math.isfinite(validation_ratio):
        raise ValueError("ML split ratios must be finite")
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("ML split ratios must be within (0, 1)")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("ML split ratios must leave test data")


def _validate_ml_features(target: str, features: tuple[str, ...]) -> None:
    if not target.strip() or not features:
        raise ValueError("ML experiment requires target and features")
    if any(not item.strip() for item in features):
        raise ValueError("ML feature names cannot be empty")
    if len(set(features)) != len(features):
        raise ValueError("ML feature names must be unique")
    if target in features:
        raise ValueError("target cannot also be a feature")


@dataclass(frozen=True, slots=True)
class MLExperimentConfig:
    target: str
    features: tuple[str, ...]
    seed: int = 1
    model_family: str = "freqai"
    train_ratio: float = 0.70
    validation_ratio: float = 0.15

    def __post_init__(self) -> None:
        _validate_ml_features(self.target, self.features)
        if not self.model_family.strip():
            raise ValueError("ML model family cannot be empty")
        _validate_ml_split_ratios(self.train_ratio, self.validation_ratio)


def regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    truth, forecast = _pairs(actual, predicted)
    errors = tuple(right - left for left, right in zip(truth, forecast, strict=True))
    mae = fmean(abs(item) for item in errors)
    mse = fmean(item * item for item in errors)
    mean_truth = fmean(truth)
    denominator = sum((item - mean_truth) ** 2 for item in truth)
    r2 = 0.0 if denominator == 0 else 1.0 - sum(item * item for item in errors) / denominator
    directional_rows = tuple(
        1.0 if math.copysign(1.0, left) == math.copysign(1.0, right) else 0.0
        for left, right in zip(truth, forecast, strict=True)
        if left != 0 or right != 0
    )
    directional = 1.0 if not directional_rows else fmean(directional_rows)
    return {"mae": mae, "rmse": math.sqrt(mse), "r2": r2, "directional_accuracy": directional}


def _validated_binary(
    actual: Sequence[int | float],
    probabilities: Sequence[float],
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    raw_truth = tuple(float(item) for item in actual)
    probs = tuple(float(item) for item in probabilities)
    if (
        not raw_truth
        or len(raw_truth) != len(probs)
        or any(item not in {0.0, 1.0} for item in raw_truth)
    ):
        raise ValueError("binary labels/probabilities are invalid")
    truth = tuple(int(item) for item in raw_truth)
    if any(not 0 <= item <= 1 or not math.isfinite(item) for item in probs):
        raise ValueError("probabilities must be finite and within [0, 1]")
    return truth, probs


def _confusion_counts(truth: Sequence[int], predicted: Sequence[int]) -> tuple[int, int, int, int]:
    pairs = tuple(zip(truth, predicted, strict=True))
    tp = sum(left == right == 1 for left, right in pairs)
    tn = sum(left == right == 0 for left, right in pairs)
    fp = sum(left == 0 and right == 1 for left, right in pairs)
    fn = sum(left == 1 and right == 0 for left, right in pairs)
    return tp, tn, fp, fn


def binary_metrics(
    actual: Sequence[int | float],
    probabilities: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    if not math.isfinite(float(threshold)) or not 0 < threshold < 1:
        raise ValueError("classification threshold must be within (0, 1)")
    truth, probs = _validated_binary(actual, probabilities)
    predicted = tuple(1 if item >= threshold else 0 for item in probs)
    tp, tn, fp, fn = _confusion_counts(truth, predicted)
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    brier = fmean((prob - label) ** 2 for label, prob in zip(truth, probs, strict=True))
    return {
        "accuracy": (tp + tn) / len(truth),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier": brier,
    }


def _calibration_members(
    truth: Sequence[int],
    probabilities: Sequence[float],
    *,
    lower: float,
    upper: float,
    include_upper: bool,
) -> tuple[tuple[int, float], ...]:
    return tuple(
        (label, probability)
        for label, probability in zip(truth, probabilities, strict=True)
        if lower <= probability < upper or (include_upper and probability == upper)
    )


def calibration_bins(
    actual: Sequence[int | float],
    probabilities: Sequence[float],
    *,
    bins: int = 10,
) -> tuple[dict[str, float], ...]:
    if bins < 2:
        raise ValueError("calibration inputs are invalid")
    truth, probs = _validated_binary(actual, probabilities)
    rows: list[dict[str, float]] = []
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = _calibration_members(
            truth,
            probs,
            lower=lower,
            upper=upper,
            include_upper=index == bins - 1,
        )
        if members:
            rows.append(
                {
                    "lower": lower,
                    "upper": upper,
                    "count": float(len(members)),
                    "mean_probability": fmean(item[1] for item in members),
                    "positive_rate": fmean(item[0] for item in members),
                }
            )
    return tuple(rows)


def _bin_count(values: Sequence[float], low: float, high: float, *, include_high: bool) -> int:
    if include_high:
        return sum(low <= value <= high for value in values)
    return sum(low <= value < high for value in values)


def _psi_contribution(reference_ratio: float, current_ratio: float) -> float:
    epsilon = 1e-9
    left = max(reference_ratio, epsilon)
    right = max(current_ratio, epsilon)
    return (right - left) * math.log(right / left)


def population_stability_index(
    reference: Sequence[float],
    current: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    left = tuple(float(item) for item in reference)
    right = tuple(float(item) for item in current)
    if (
        not left
        or not right
        or bins < 2
        or any(not math.isfinite(item) for item in left + right)
    ):
        raise ValueError("PSI inputs are invalid")
    minimum = min(left + right)
    maximum = max(left + right)
    if minimum == maximum:
        return 0.0
    width = (maximum - minimum) / bins
    total = 0.0
    for index in range(bins):
        low = minimum + index * width
        high = maximum if index == bins - 1 else low + width
        include_high = index == bins - 1
        ref = _bin_count(left, low, high, include_high=include_high)
        cur = _bin_count(right, low, high, include_high=include_high)
        total += _psi_contribution(ref / len(left), cur / len(right))
    return total


def _promotion_value(
    metrics: dict[str, float],
    name: str,
    threshold: float,
) -> tuple[bool, float, float]:
    numeric_threshold = float(threshold)
    if not math.isfinite(numeric_threshold):
        raise ValueError("ML promotion thresholds must be finite")
    if name not in metrics:
        return False, 0.0, numeric_threshold
    value = float(metrics[name])
    if not math.isfinite(value):
        raise ValueError("ML promotion metrics must be finite")
    return True, value, numeric_threshold


def promotion_decision(
    metrics: dict[str, float],
    *,
    minimums: dict[str, float] | None = None,
    maximums: dict[str, float] | None = None,
) -> tuple[bool, tuple[str, ...]]:
    violations: list[str] = []
    for name, threshold in (minimums or {}).items():
        present, value, numeric_threshold = _promotion_value(metrics, name, threshold)
        if not present or value < numeric_threshold:
            violations.append(f"{name}<minimum")
    for name, threshold in (maximums or {}).items():
        present, value, numeric_threshold = _promotion_value(metrics, name, threshold)
        if not present or value > numeric_threshold:
            violations.append(f"{name}>maximum")
    return not violations, tuple(violations)


def _pairs(
    actual: Sequence[float],
    predicted: Sequence[float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    truth = tuple(float(item) for item in actual)
    forecast = tuple(float(item) for item in predicted)
    if not truth or len(truth) != len(forecast):
        raise ValueError("actual and predicted series must be non-empty and equal length")
    if any(not math.isfinite(item) for item in truth + forecast):
        raise ValueError("ML metrics require finite values")
    return truth, forecast
