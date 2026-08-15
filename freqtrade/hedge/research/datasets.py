"""Dataset contracts and leakage guards shared by backtest, ML, and RL research."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from typing import Any


@dataclass(frozen=True, slots=True)
class TimeSplit:
    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    train: tuple[int, int]
    validation: tuple[int, int]
    test: tuple[int, int]


def fingerprint_rows(
    rows: Sequence[dict[str, Any]],
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    payload = {"rows": list(rows), "metadata": dict(metadata or {})}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_monotonic_timestamps(values: Sequence[datetime]) -> None:
    if not values:
        raise ValueError("dataset timestamps cannot be empty")
    if any(item.tzinfo is None for item in values):
        raise ValueError("dataset timestamps must be timezone-aware")
    if any(right <= left for left, right in pairwise(values)):
        raise ValueError("dataset timestamps must be strictly increasing")


def chronological_split(
    length: int,
    *,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    embargo: int = 0,
) -> TimeSplit:
    if length < 10:
        raise ValueError("dataset must contain at least 10 rows")
    if not math.isfinite(train_ratio) or not math.isfinite(validation_ratio):
        raise ValueError("split ratios must be finite")
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("split ratios must be within (0, 1)")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train + validation ratios must leave test data")
    if embargo < 0:
        raise ValueError("embargo cannot be negative")
    train_end = int(length * train_ratio)
    validation_start = train_end + embargo
    validation_end = validation_start + int(length * validation_ratio)
    test_start = validation_end + embargo
    if validation_start >= validation_end or test_start >= length:
        raise ValueError("embargo leaves insufficient split data")
    return TimeSplit(
        train=(0, train_end),
        validation=(validation_start, validation_end),
        test=(test_start, length),
    )


def walk_forward_folds(
    length: int,
    *,
    train: int,
    validation: int,
    test: int,
    step: int,
    embargo: int = 0,
) -> tuple[WalkForwardFold, ...]:
    values = (length, train, validation, test, step)
    if any(value < 1 for value in values) or embargo < 0:
        raise ValueError("walk-forward sizes must be positive and embargo non-negative")
    folds: list[WalkForwardFold] = []
    start = 0
    while True:
        train_end = start + train
        validation_start = train_end + embargo
        validation_end = validation_start + validation
        test_start = validation_end + embargo
        test_end = test_start + test
        if test_end > length:
            break
        folds.append(
            WalkForwardFold(
                train=(start, train_end),
                validation=(validation_start, validation_end),
                test=(test_start, test_end),
            )
        )
        start += step
    if not folds:
        raise ValueError("walk-forward configuration produces no folds")
    return tuple(folds)


def assert_no_overlap(split: TimeSplit | WalkForwardFold) -> None:
    train_end = split.train[1]
    validation_start, validation_end = split.validation
    test_start = split.test[0]
    if not train_end <= validation_start <= validation_end <= test_start:
        raise ValueError("dataset split overlaps or is out of order")


def validate_ohlcv_rows(rows: Sequence[dict[str, Any]]) -> None:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not rows:
        raise ValueError("OHLCV rows cannot be empty")
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"OHLCV row {index} missing: {', '.join(sorted(missing))}")
        open_ = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])
        close = float(row["close"])
        volume = float(row["volume"])
        numeric = (open_, high, low, close, volume)
        if any(not math.isfinite(item) for item in numeric):
            raise ValueError(f"OHLCV row {index} contains non-finite values")
        if (
            min(open_, high, low, close) <= 0
            or low > min(open_, close)
            or high < max(open_, close)
            or high < low
            or volume < 0
        ):
            raise ValueError(f"OHLCV row {index} is inconsistent")
