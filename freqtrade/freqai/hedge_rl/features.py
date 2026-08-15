"""Feature contracts, leakage-safe normalization, and drift diagnostics (rounds 31-40)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256

import numpy as np
import numpy.typing as npt
import pandas as pd


FloatMatrix = npt.NDArray[np.float64]


# Round 31 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FeatureSchema:
    names: tuple[str, ...]
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        cleaned = tuple(str(name).strip() for name in self.names)
        if not cleaned or any(not name for name in cleaned):
            raise ValueError("feature names must be non-empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("feature names must be unique")
        object.__setattr__(self, "names", cleaned)
        if not self.version.strip():
            raise ValueError("feature schema version cannot be empty")

    @property
    def width(self) -> int:
        return len(self.names)

    @property
    def signature(self) -> str:
        payload = json.dumps(
            {"names": self.names, "version": self.version},
            separators=(",", ":"),
            sort_keys=True,
        )
        return sha256(payload.encode("utf-8")).hexdigest()

    def validate_frame(self, frame: pd.DataFrame) -> None:
        if tuple(str(column) for column in frame.columns) != self.names:
            raise ValueError("feature columns do not match the schema in exact order")


# Round 32 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class SanitizationReport:
    replaced_nonfinite: int
    clipped_values: int
    rows: int
    columns: int


def sanitize_feature_matrix(
    values: npt.ArrayLike,
    *,
    clip: float = 10.0,
    nonfinite_policy: str = "raise",
) -> tuple[FloatMatrix, SanitizationReport]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("feature matrix must be a non-empty 2D array")
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("clip must be finite and positive")
    invalid = ~np.isfinite(array)
    invalid_count = int(invalid.sum())
    if invalid_count and nonfinite_policy == "raise":
        raise ValueError("feature matrix contains non-finite values")
    if nonfinite_policy not in {"raise", "zero"}:
        raise ValueError("nonfinite_policy must be 'raise' or 'zero'")
    result = array.copy()
    if invalid_count:
        result[invalid] = 0.0
    clipped_count = int((np.abs(result) > clip).sum())
    np.clip(result, -clip, clip, out=result)
    return result, SanitizationReport(invalid_count, clipped_count, *result.shape)


# Round 33 -------------------------------------------------------------------------------
@dataclass(slots=True)
class RobustFeatureScaler:
    epsilon: float = 1e-9
    median_: npt.NDArray[np.float64] | None = field(init=False, default=None)
    iqr_: npt.NDArray[np.float64] | None = field(init=False, default=None)

    def fit(self, values: npt.ArrayLike) -> RobustFeatureScaler:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("fit values must be a finite 2D matrix")
        self.median_ = np.median(array, axis=0)
        q1, q3 = np.quantile(array, [0.25, 0.75], axis=0)
        self.iqr_ = np.maximum(q3 - q1, self.epsilon)
        return self

    def transform(self, values: npt.ArrayLike, *, clip: float | None = None) -> FloatMatrix:
        if self.median_ is None or self.iqr_ is None:
            raise RuntimeError("scaler must be fitted before transform")
        array = np.asarray(values, dtype=np.float64)
        if array.shape[-1] != self.median_.shape[0] or not np.isfinite(array).all():
            raise ValueError("transform values have incompatible width or non-finite values")
        result = (array - self.median_) / self.iqr_
        return np.clip(result, -clip, clip) if clip is not None else result

    def fit_transform(self, values: npt.ArrayLike, *, clip: float | None = None) -> FloatMatrix:
        return self.fit(values).transform(values, clip=clip)


# Round 34 -------------------------------------------------------------------------------
@dataclass(slots=True)
class StreamingMoments:
    width: int
    count: int = field(init=False, default=0)
    mean: npt.NDArray[np.float64] = field(init=False)
    m2: npt.NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("width must be positive")
        self.mean = np.zeros(self.width, dtype=np.float64)
        self.m2 = np.zeros(self.width, dtype=np.float64)

    def update(self, values: npt.ArrayLike) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        if not np.isfinite(array).all():
            raise ValueError("streaming values must be finite")
        for row in array:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (row - self.mean)

    def merge(self, other: StreamingMoments) -> None:
        if other.width != self.width:
            raise ValueError("cannot merge moments with different widths")
        if other.count == 0:
            return
        if self.count == 0:
            self.count = other.count
            self.mean = other.mean.copy()
            self.m2 = other.m2.copy()
            return
        total = self.count + other.count
        delta = other.mean - self.mean
        self.m2 += other.m2 + delta * delta * self.count * other.count / total
        self.mean += delta * other.count / total
        self.count = total

    @property
    def variance(self) -> npt.NDArray[np.float64]:
        return self.m2 / max(self.count - 1, 1)


# Round 35 -------------------------------------------------------------------------------
def causal_rolling_zscore(
    values: npt.ArrayLike,
    *,
    window: int,
    min_periods: int = 2,
    epsilon: float = 1e-9,
) -> FloatMatrix:
    """Normalize each row using observations strictly before that row."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("values must be a finite 2D matrix")
    if window < 2 or not 1 <= min_periods <= window:
        raise ValueError("invalid rolling window/min_periods")
    result = np.zeros_like(array)
    for index in range(len(array)):
        start = max(0, index - window)
        history = array[start:index]
        if len(history) < min_periods:
            result[index] = 0.0
            continue
        mean = history.mean(axis=0)
        std = history.std(axis=0, ddof=1)
        result[index] = (array[index] - mean) / np.maximum(std, epsilon)
    return result


# Round 36 -------------------------------------------------------------------------------
class MissingValuePolicy(StrEnum):
    REJECT = "REJECT"
    ZERO = "ZERO"
    FORWARD_FILL = "FORWARD_FILL"


def apply_missing_value_policy(
    frame: pd.DataFrame,
    *,
    policy: MissingValuePolicy,
    max_forward_fill: int = 1,
) -> pd.DataFrame:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    if policy is MissingValuePolicy.REJECT:
        if numeric.isna().any().any():
            raise ValueError("feature frame contains missing values")
        return numeric
    if policy is MissingValuePolicy.ZERO:
        return numeric.fillna(0.0)
    if max_forward_fill < 0:
        raise ValueError("max_forward_fill cannot be negative")
    result = numeric.ffill(limit=max_forward_fill)
    if result.isna().any().any():
        raise ValueError("missing values exceed the allowed forward-fill horizon")
    return result


# Round 37 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FeatureFreshness:
    age_steps: int
    max_age_steps: int
    fresh: bool
    reason: str


def evaluate_feature_freshness(
    *,
    produced_tick: int,
    decision_tick: int,
    max_age_steps: int,
) -> FeatureFreshness:
    if min(produced_tick, decision_tick, max_age_steps) < 0:
        raise ValueError("ticks and max_age_steps must be non-negative")
    if produced_tick > decision_tick:
        raise ValueError("feature timestamp cannot be in the future")
    age = decision_tick - produced_tick
    fresh = age <= max_age_steps
    return FeatureFreshness(age, max_age_steps, fresh, "FRESH" if fresh else "STALE_FEATURES")


# Round 38 -------------------------------------------------------------------------------
def population_stability_index(
    reference: npt.ArrayLike,
    current: npt.ArrayLike,
    *,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cur = np.asarray(current, dtype=np.float64).reshape(-1)
    if min(len(ref), len(cur)) < 2 or not np.isfinite(ref).all() or not np.isfinite(cur).all():
        raise ValueError("PSI samples must be finite and contain at least two values")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        span = max(abs(float(ref[0])) * 0.01, 1e-6)
        edges = np.array([ref[0] - span, ref[0], ref[0] + span])
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.maximum(ref_counts / len(ref), epsilon)
    cur_pct = np.maximum(cur_counts / len(cur), epsilon)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# Round 39 -------------------------------------------------------------------------------
def greedy_correlation_prune(frame: pd.DataFrame, *, threshold: float = 0.98) -> tuple[str, ...]:
    if not 0 < threshold < 1:
        raise ValueError("threshold must be within (0, 1)")
    numeric = frame.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("correlation input must be finite")
    corr = numeric.corr().abs().fillna(0.0)
    kept: list[str] = []
    for column in numeric.columns:
        if all(float(corr.loc[column, previous]) <= threshold for previous in kept):
            kept.append(str(column))
    return tuple(kept)


# Round 40 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FeatureManifest:
    schema_signature: str
    row_count: int
    minimums: tuple[float, ...]
    maximums: tuple[float, ...]
    means: tuple[float, ...]
    fingerprint: str

    @classmethod
    def build(cls, frame: pd.DataFrame, schema: FeatureSchema) -> FeatureManifest:
        schema.validate_frame(frame)
        values = frame.to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or len(values) < 1:
            raise ValueError("manifest requires a non-empty finite frame")
        digest = sha256()
        digest.update(schema.signature.encode("ascii"))
        digest.update(np.ascontiguousarray(values).tobytes())
        return cls(
            schema_signature=schema.signature,
            row_count=len(values),
            minimums=tuple(float(item) for item in values.min(axis=0)),
            maximums=tuple(float(item) for item in values.max(axis=0)),
            means=tuple(float(item) for item in values.mean(axis=0)),
            fingerprint=digest.hexdigest(),
        )
