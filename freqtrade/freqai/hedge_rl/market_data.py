"""Chronology, data-quality, split, sampling, and fingerprint tools (rounds 41-50)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise

import numpy as np
import numpy.typing as npt
import pandas as pd


# Round 41 -------------------------------------------------------------------------------
def validate_chronology(index: pd.Index) -> None:
    if len(index) < 2:
        raise ValueError("chronology requires at least two timestamps")
    if not index.is_monotonic_increasing:
        raise ValueError("timestamps must be monotonic increasing")
    if not index.is_unique:
        raise ValueError("timestamps must be unique")
    if isinstance(index, pd.DatetimeIndex) and index.tz is None:
        raise ValueError("datetime index must be timezone-aware")


# Round 42 -------------------------------------------------------------------------------
def duplicate_timestamps(index: pd.Index) -> tuple[object, ...]:
    duplicated = index[index.duplicated(keep=False)]
    unique = []
    seen: set[object] = set()
    for value in duplicated:
        if value not in seen:
            seen.add(value)
            unique.append(value)
    return tuple(unique)


# Round 43 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OHLCValidationReport:
    rows: int
    invalid_rows: tuple[int, ...]

    @property
    def valid(self) -> bool:
        return not self.invalid_rows


def validate_ohlc_consistency(
    frame: pd.DataFrame,
    *,
    raise_on_error: bool = True,
) -> OHLCValidationReport:
    required = ("open", "high", "low", "close")
    if missing := set(required).difference(frame.columns):
        raise ValueError(f"missing OHLC columns: {sorted(missing)}")
    values = frame.loc[:, required].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite_positive = np.isfinite(values).all(axis=1) & (values > 0).all(axis=1)
    open_, high, low, close = values.T
    ordered = (high >= np.maximum.reduce([open_, low, close])) & (
        low <= np.minimum.reduce([open_, high, close])
    )
    invalid = tuple(int(item) for item in np.flatnonzero(~(finite_positive & ordered)))
    report = OHLCValidationReport(len(frame), invalid)
    if invalid and raise_on_error:
        raise ValueError(f"invalid OHLC rows: {invalid[:20]}")
    return report


# Round 44 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MarketGap:
    previous: pd.Timestamp
    current: pd.Timestamp
    missing_intervals: int


def detect_market_gaps(
    index: pd.DatetimeIndex,
    *,
    expected_interval: pd.Timedelta,
) -> tuple[MarketGap, ...]:
    validate_chronology(index)
    if expected_interval <= pd.Timedelta(0):
        raise ValueError("expected_interval must be positive")
    gaps: list[MarketGap] = []
    for previous, current in pairwise(index):
        delta = current - previous
        ratio = delta / expected_interval
        if ratio > 1 + 1e-9:
            rounded = round(float(ratio))
            if not math.isclose(float(ratio), rounded, rel_tol=0, abs_tol=1e-9):
                raise ValueError(
                    "timestamp interval is not an integer multiple of expected_interval"
                )
            gaps.append(MarketGap(previous, current, rounded - 1))
        elif ratio < 1 - 1e-9:
            raise ValueError("timestamps are closer than expected_interval")
    return tuple(gaps)


# Round 45 -------------------------------------------------------------------------------
def align_funding_rates(
    candle_index: pd.DatetimeIndex,
    funding: pd.Series,
    *,
    default: float = 0.0,
) -> pd.Series:
    """Backward-align funding values so a candle never sees a future funding event."""

    validate_chronology(candle_index)
    if not isinstance(funding.index, pd.DatetimeIndex):
        raise TypeError("funding series requires a DatetimeIndex")
    if not funding.index.is_monotonic_increasing or not funding.index.is_unique:
        raise ValueError("funding timestamps must be sorted and unique")
    values = pd.to_numeric(funding, errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("funding rates must be finite")
    aligned = values.reindex(candle_index, method="ffill").fillna(float(default))
    aligned.name = "funding_rate"
    return aligned


# Round 46 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PurgedSplit:
    train: slice
    validation: slice
    test: slice
    embargo: int

    def indexes(self) -> tuple[set[int], set[int], set[int]]:
        return (
            set(range(self.train.start or 0, self.train.stop or 0)),
            set(range(self.validation.start or 0, self.validation.stop or 0)),
            set(range(self.test.start or 0, self.test.stop or 0)),
        )


def purged_chronological_split(
    rows: int,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    embargo: int = 1,
) -> PurgedSplit:
    if rows < 12 or embargo < 0:
        raise ValueError("insufficient rows or invalid embargo")
    if not 0 < validation_fraction < 0.5 or not 0 < test_fraction < 0.5:
        raise ValueError("split fractions must be within (0, 0.5)")
    validation_length = max(1, int(rows * validation_fraction))
    test_length = max(1, int(rows * test_fraction))
    train_stop = rows - validation_length - test_length - 2 * embargo
    validation_start = train_stop + embargo
    validation_stop = validation_start + validation_length
    test_start = validation_stop + embargo
    if train_stop < 2 or test_start >= rows:
        raise ValueError("split leaves insufficient training data")
    split = PurgedSplit(
        slice(0, train_stop),
        slice(validation_start, validation_stop),
        slice(test_start, rows),
        embargo,
    )
    train, validation, test = split.indexes()
    if train & validation or train & test or validation & test:
        raise RuntimeError("purged split contains overlapping rows")
    return split


# Round 47 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    train: slice
    evaluation: slice
    fold: int


def purged_walk_forward(
    rows: int,
    *,
    train_length: int,
    evaluation_length: int,
    embargo: int = 1,
    step: int | None = None,
    expanding: bool = False,
) -> tuple[WalkForwardFold, ...]:
    if min(rows, train_length, evaluation_length) < 1 or embargo < 0:
        raise ValueError("invalid walk-forward dimensions")
    stride = evaluation_length if step is None else step
    if stride < 1:
        raise ValueError("step must be positive")
    folds: list[WalkForwardFold] = []
    train_stop = train_length
    fold = 0
    while train_stop + embargo + evaluation_length <= rows:
        train_start = 0 if expanding else train_stop - train_length
        evaluation_start = train_stop + embargo
        folds.append(
            WalkForwardFold(
                slice(train_start, train_stop),
                slice(evaluation_start, evaluation_start + evaluation_length),
                fold,
            )
        )
        fold += 1
        train_stop += stride
    if not folds:
        raise ValueError("dataset is too short for a walk-forward fold")
    return tuple(folds)


# Round 48 -------------------------------------------------------------------------------
def deterministic_episode_starts(
    *,
    region_start: int,
    region_stop: int,
    window: int,
    episode_steps: int,
    count: int,
    seed: int,
    replace: bool = False,
) -> tuple[int, ...]:
    if min(region_start, window, episode_steps, count, seed) < 0 or region_stop <= region_start:
        raise ValueError("invalid episode sampling arguments")
    earliest = region_start + window - 1
    latest = region_stop - episode_steps - 1
    if latest < earliest:
        raise ValueError("region is too short")
    population = np.arange(earliest, latest + 1)
    if not replace and count > len(population):
        raise ValueError("count exceeds available unique episode starts")
    selected = np.random.default_rng(seed).choice(population, size=count, replace=replace)
    return tuple(int(item) for item in selected)


# Round 49 -------------------------------------------------------------------------------
def stationary_block_bootstrap_indices(
    rows: int,
    *,
    block_length: int,
    output_length: int | None = None,
    seed: int,
) -> npt.NDArray[np.int64]:
    """Circular fixed-block bootstrap preserving local temporal dependence."""

    length = rows if output_length is None else output_length
    if min(rows, block_length, length) < 1 or block_length > rows or seed < 0:
        raise ValueError("invalid block bootstrap arguments")
    rng = np.random.default_rng(seed)
    result: list[int] = []
    while len(result) < length:
        start = int(rng.integers(0, rows))
        result.extend((start + offset) % rows for offset in range(block_length))
    return np.asarray(result[:length], dtype=np.int64)


# Round 50 -------------------------------------------------------------------------------
def dataset_fingerprint(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    metadata: dict[str, object] | None = None,
) -> str:
    if len(features) != len(prices) or not features.index.equals(prices.index):
        raise ValueError("features and prices must align")
    digest = sha256()
    digest.update(json.dumps(tuple(map(str, features.columns))).encode("utf-8"))
    digest.update(json.dumps(tuple(map(str, prices.columns))).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(features, index=True).to_numpy().tobytes())
    digest.update(pd.util.hash_pandas_object(prices, index=True).to_numpy().tobytes())
    digest.update(json.dumps(metadata or {}, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()
