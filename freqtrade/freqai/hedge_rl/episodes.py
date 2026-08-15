"""Chronological episode and walk-forward split utilities without temporal leakage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class EpisodeSlice:
    start: int
    stop: int
    role: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("episode slice must satisfy 0 <= start < stop")
        if self.role not in {"train", "validation", "test"}:
            raise ValueError("unsupported episode role")

    @property
    def length(self) -> int:
        return self.stop - self.start

    def as_slice(self) -> slice:
        return slice(self.start, self.stop)


def chronological_split(
    rows: int,
    *,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    embargo: int = 0,
) -> tuple[EpisodeSlice, EpisodeSlice, EpisodeSlice]:
    if rows < 10:
        raise ValueError("at least 10 rows are required")
    if not 0 < validation_fraction < 0.5 or not 0 < test_fraction < 0.5:
        raise ValueError("validation_fraction and test_fraction must be within (0, 0.5)")
    if validation_fraction + test_fraction >= 0.8:
        raise ValueError("training split would be too small")
    if embargo < 0:
        raise ValueError("embargo cannot be negative")
    test_length = max(1, int(rows * test_fraction))
    validation_length = max(1, int(rows * validation_fraction))
    train_stop = rows - test_length - validation_length - 2 * embargo
    validation_start = train_stop + embargo
    validation_stop = validation_start + validation_length
    test_start = validation_stop + embargo
    if train_stop < 2 or test_start >= rows:
        raise ValueError("split and embargo leave insufficient data")
    return (
        EpisodeSlice(0, train_stop, "train"),
        EpisodeSlice(validation_start, validation_stop, "validation"),
        EpisodeSlice(test_start, rows, "test"),
    )


def walk_forward_slices(
    rows: int,
    *,
    train_length: int,
    evaluation_length: int,
    step: int | None = None,
    embargo: int = 0,
    expanding: bool = False,
) -> tuple[tuple[EpisodeSlice, EpisodeSlice], ...]:
    if min(rows, train_length, evaluation_length) <= 0:
        raise ValueError("row and window lengths must be positive")
    if embargo < 0:
        raise ValueError("embargo cannot be negative")
    stride = evaluation_length if step is None else step
    if stride < 1:
        raise ValueError("step must be positive")
    result: list[tuple[EpisodeSlice, EpisodeSlice]] = []
    anchor = train_length
    while anchor + embargo + evaluation_length <= rows:
        train_start = 0 if expanding else anchor - train_length
        train = EpisodeSlice(train_start, anchor, "train")
        validation = EpisodeSlice(
            anchor + embargo,
            anchor + embargo + evaluation_length,
            "validation",
        )
        result.append((train, validation))
        anchor += stride
    if not result:
        raise ValueError("dataset is too short for one walk-forward fold")
    return tuple(result)


def sample_episode_start(
    region: EpisodeSlice,
    *,
    window_size: int,
    episode_steps: int,
    seed: int,
) -> int:
    earliest = region.start + window_size - 1
    latest = region.stop - episode_steps - 1
    if latest < earliest:
        raise ValueError("region is too short for requested observation and episode")
    return int(np.random.default_rng(seed).integers(earliest, latest + 1))
