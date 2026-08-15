"""Leakage-aware walk-forward window construction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class IndexRange:
    start: int
    stop: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop < self.start:
            raise ValueError("invalid half-open index range")

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True, slots=True)
class WalkForwardWindow:
    window_id: int
    train: IndexRange
    validation: IndexRange
    test: IndexRange
    train_start_time: datetime | None = None
    test_end_time: datetime | None = None


@dataclass(frozen=True, slots=True)
class WalkForwardSpec:
    train_size: int
    validation_size: int
    test_size: int
    step_size: int | None = None
    expanding: bool = False
    purge_size: int = 0
    embargo_size: int = 0
    minimum_windows: int = 1

    def __post_init__(self) -> None:
        sizes = (self.train_size, self.validation_size, self.test_size)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in sizes):
            raise TypeError("train/validation/test sizes must be integers")
        if self.step_size is not None and (
            isinstance(self.step_size, bool) or not isinstance(self.step_size, int)
        ):
            raise TypeError("step size must be an integer")
        if not isinstance(self.expanding, bool):
            raise TypeError("expanding must be a boolean")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.purge_size, self.embargo_size, self.minimum_windows)
        ):
            raise TypeError("purge, embargo and minimum_windows must be integers")
        if any(value <= 0 for value in sizes):
            raise ValueError("train/validation/test sizes must be positive")
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError("step size must be positive")
        if self.purge_size < 0 or self.embargo_size < 0:
            raise ValueError("purge and embargo sizes cannot be negative")
        if self.purge_size >= self.train_size:
            raise ValueError("purge size must be smaller than train size")
        if self.minimum_windows <= 0:
            raise ValueError("minimum windows must be positive")


def build_walk_forward_windows(
    item_count: int,
    spec: WalkForwardSpec,
    *,
    timestamps: Sequence[datetime] | None = None,
) -> tuple[WalkForwardWindow, ...]:
    if item_count <= 0:
        raise ValueError("walk-forward item count must be positive")
    if timestamps is not None:
        if len(timestamps) != item_count:
            raise ValueError("timestamp count must match item count")
        if any(right <= left for left, right in pairwise(timestamps)):
            raise ValueError("walk-forward timestamps must be strictly increasing")

    step = spec.step_size or spec.test_size
    output: list[WalkForwardWindow] = []
    anchor = 0
    window_id = 0
    while True:
        raw_train_start = 0 if spec.expanding else anchor
        raw_train_stop = anchor + spec.train_size
        train_stop = raw_train_stop - spec.purge_size
        validation_start = raw_train_stop + spec.embargo_size
        validation_stop = validation_start + spec.validation_size
        test_start = validation_stop + spec.embargo_size
        test_stop = test_start + spec.test_size
        if test_stop > item_count:
            break
        output.append(
            WalkForwardWindow(
                window_id=window_id,
                train=IndexRange(raw_train_start, train_stop),
                validation=IndexRange(validation_start, validation_stop),
                test=IndexRange(test_start, test_stop),
                train_start_time=None if timestamps is None else timestamps[raw_train_start],
                test_end_time=None if timestamps is None else timestamps[test_stop - 1],
            )
        )
        anchor += step
        window_id += 1
    if len(output) < spec.minimum_windows:
        required = (
            spec.train_size
            + spec.validation_size
            + spec.test_size
            + 2 * spec.embargo_size
        )
        raise ValueError(
            f"walk-forward data produced {len(output)} windows; minimum={spec.minimum_windows}; "
            f"at least {required} items are required for the first window"
        )
    return tuple(output)
