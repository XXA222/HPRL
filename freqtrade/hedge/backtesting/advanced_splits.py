"""Leakage-aware split and window diagnostics for time-series optimization."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class IndexWindow:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


def anchored_windows(
    count: int,
    *,
    minimum_train: int,
    validation_size: int,
    test_size: int,
    step: int | None = None,
) -> tuple[IndexWindow, ...]:
    if min(count, minimum_train, validation_size, test_size) <= 0:
        raise ValueError("window sizes must be positive")
    stride = test_size if step is None else step
    if stride <= 0:
        raise ValueError("step must be positive")
    output: list[IndexWindow] = []
    train_end = minimum_train
    while train_end + validation_size + test_size <= count:
        output.append(IndexWindow(
            tuple(range(0, train_end)),
            tuple(range(train_end, train_end + validation_size)),
            tuple(range(train_end + validation_size, train_end + validation_size + test_size)),
        ))
        train_end += stride
    if not output:
        raise ValueError("insufficient observations for anchored windows")
    return tuple(output)


def rolling_windows(
    count: int,
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    step: int | None = None,
) -> tuple[IndexWindow, ...]:
    if min(count, train_size, validation_size, test_size) <= 0:
        raise ValueError("window sizes must be positive")
    stride = test_size if step is None else step
    if stride <= 0:
        raise ValueError("step must be positive")
    width = train_size + validation_size + test_size
    output = []
    for start in range(0, count - width + 1, stride):
        output.append(IndexWindow(
            tuple(range(start, start + train_size)),
            tuple(range(start + train_size, start + train_size + validation_size)),
            tuple(range(start + train_size + validation_size, start + width)),
        ))
    if not output:
        raise ValueError("insufficient observations for rolling windows")
    return tuple(output)


def purged_kfold(
    count: int, *, folds: int, purge: int = 0, embargo: int = 0
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    if count <= 1 or folds < 2 or folds > count:
        raise ValueError("invalid count or fold count")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo cannot be negative")
    base, remainder = divmod(count, folds)
    output = []
    start = 0
    all_indices = set(range(count))
    for fold in range(folds):
        size = base + (1 if fold < remainder else 0)
        test = set(range(start, start + size))
        blocked_start = max(0, start - purge)
        blocked_end = min(count, start + size + embargo)
        blocked = set(range(blocked_start, blocked_end))
        train = tuple(sorted(all_indices - blocked))
        output.append((train, tuple(sorted(test))))
        start += size
    return tuple(output)


def embargo_indices(test_indices: Sequence[int], *, count: int, embargo: int) -> tuple[int, ...]:
    if not test_indices:
        return ()
    if count <= 0 or embargo < 0:
        raise ValueError("count must be positive and embargo non-negative")
    end = min(count, max(test_indices) + 1 + embargo)
    return tuple(range(max(test_indices) + 1, end))


def leakage_audit(
    train: Sequence[int], validation: Sequence[int], test: Sequence[int]
) -> tuple[str, ...]:
    train_set, validation_set, test_set = set(train), set(validation), set(test)
    issues: list[str] = []
    if train_set & validation_set:
        issues.append("train_validation_overlap")
    if train_set & test_set:
        issues.append("train_test_overlap")
    if validation_set & test_set:
        issues.append("validation_test_overlap")
    if any(index < 0 for index in train_set | validation_set | test_set):
        issues.append("negative_index")
    if train and validation and max(train) >= min(validation):
        issues.append("train_not_before_validation")
    if validation and test and max(validation) >= min(test):
        issues.append("validation_not_before_test")
    return tuple(issues)


def coverage_ratio(windows: Sequence[IndexWindow], *, count: int) -> float:
    if count <= 0:
        raise ValueError("count must be positive")
    covered = {
        index
        for window in windows
        for index in (*window.train, *window.validation, *window.test)
    }
    return len(covered) / count


def minimum_event_gate(window: IndexWindow, *, train: int, validation: int, test: int) -> bool:
    if min(train, validation, test) < 0:
        raise ValueError("minimum counts cannot be negative")
    return (
        len(window.train) >= train
        and len(window.validation) >= validation
        and len(window.test) >= test
    )


def regime_stratified_folds(regimes: Sequence[str], *, folds: int) -> tuple[tuple[int, ...], ...]:
    if folds < 2 or len(regimes) < folds:
        raise ValueError("invalid fold count")
    buckets: list[list[int]] = [[] for _ in range(folds)]
    counters: dict[str, int] = {}
    for index, regime in enumerate(regimes):
        if not regime:
            raise ValueError("regime labels cannot be empty")
        target = counters.get(regime, 0) % folds
        buckets[target].append(index)
        counters[regime] = counters.get(regime, 0) + 1
    return tuple(tuple(bucket) for bucket in buckets)


def window_fingerprint(window: IndexWindow) -> str:
    payload = {"train": window.train, "validation": window.validation, "test": window.test}
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def infer_timeframe_seconds(timestamps: Sequence[datetime]) -> int:
    if len(timestamps) < 2:
        raise ValueError("at least two timestamps are required")
    if any(item.tzinfo is None for item in timestamps):
        raise ValueError("timestamps must be timezone-aware")
    deltas = [right - left for left, right in pairwise(timestamps)]
    if any(delta <= timedelta(0) for delta in deltas):
        raise ValueError("timestamps must be strictly increasing")
    if len(set(deltas)) != 1:
        raise ValueError("timeframe cannot be inferred from irregular timestamps")
    delta = deltas[0]
    one_second = timedelta(seconds=1)
    if delta % one_second:
        raise ValueError("timeframe must resolve to a whole number of seconds")
    return delta // one_second
