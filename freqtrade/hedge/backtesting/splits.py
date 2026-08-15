from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from freqtrade.hedge.simulation.exchange import BarEvent, SimulationInputEvent

from .contracts import BacktestDataset, SplitMode
from .dataset import build_dataset


@dataclass(frozen=True, slots=True)
class DatasetFold:
    fold_id: str
    mode: SplitMode
    train: BacktestDataset
    test: BacktestDataset
    train_bar_indices: tuple[int, ...]
    test_bar_indices: tuple[int, ...]
    purge_bars: int = 0
    embargo_bars: int = 0

    def __post_init__(self) -> None:
        if not self.train_bar_indices or not self.test_bar_indices:
            raise ValueError("fold train and test sets cannot be empty")
        if set(self.train_bar_indices).intersection(self.test_bar_indices):
            raise ValueError("fold train and test bar indices must be disjoint")


def _bar_times(dataset: BacktestDataset) -> tuple[datetime, ...]:
    return tuple(event.timestamp for event in dataset.events if isinstance(event, BarEvent))


def _dataset_from_indices(
    dataset: BacktestDataset,
    indices: Iterable[int],
    *,
    dataset_id: str,
) -> BacktestDataset:
    times = _bar_times(dataset)
    selected_indices = tuple(sorted(set(indices)))
    if not selected_indices:
        raise ValueError("dataset index selection cannot be empty")
    if selected_indices[0] < 0 or selected_indices[-1] >= len(times):
        raise IndexError("dataset bar index outside range")
    selected_times = {times[index] for index in selected_indices}
    events: list[SimulationInputEvent] = [
        event for event in dataset.events if event.timestamp in selected_times
    ]
    return build_dataset(
        events=events,
        dataset_id=dataset_id,
        timeframe=dataset.timeframe,
        metadata={**dataset.metadata, "parent_fingerprint": dataset.fingerprint},
    )


def walk_forward_splits(
    dataset: BacktestDataset,
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    gap_bars: int = 0,
    anchored: bool = False,
) -> tuple[DatasetFold, ...]:
    if min(train_bars, test_bars) < 1:
        raise ValueError("train_bars and test_bars must be positive")
    if gap_bars < 0:
        raise ValueError("gap_bars cannot be negative")
    step = step_bars or test_bars
    if step < 1:
        raise ValueError("step_bars must be positive")
    count = len(_bar_times(dataset))
    folds: list[DatasetFold] = []
    train_start = 0
    train_end = train_bars
    ordinal = 0
    while train_end + gap_bars + test_bars <= count:
        test_start = train_end + gap_bars
        test_end = test_start + test_bars
        effective_train_start = 0 if anchored else train_start
        train_indices = tuple(range(effective_train_start, train_end))
        test_indices = tuple(range(test_start, test_end))
        mode = SplitMode.ANCHORED if anchored else SplitMode.ROLLING
        fold_id = f"wf-{ordinal:03d}"
        folds.append(
            DatasetFold(
                fold_id=fold_id,
                mode=mode,
                train=_dataset_from_indices(
                    dataset,
                    train_indices,
                    dataset_id=f"{dataset.dataset_id}:{fold_id}:train",
                ),
                test=_dataset_from_indices(
                    dataset,
                    test_indices,
                    dataset_id=f"{dataset.dataset_id}:{fold_id}:test",
                ),
                train_bar_indices=train_indices,
                test_bar_indices=test_indices,
                purge_bars=gap_bars,
            )
        )
        ordinal += 1
        train_start += step
        train_end += step
    if not folds:
        raise ValueError("dataset is too short for requested walk-forward split")
    return tuple(folds)


def purged_kfold_splits(
    dataset: BacktestDataset,
    *,
    folds: int,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> tuple[DatasetFold, ...]:
    count = len(_bar_times(dataset))
    if folds < 2 or folds > count:
        raise ValueError("folds must be between 2 and the number of bars")
    if min(purge_bars, embargo_bars) < 0:
        raise ValueError("purge and embargo bars cannot be negative")
    base, remainder = divmod(count, folds)
    output: list[DatasetFold] = []
    start = 0
    for ordinal in range(folds):
        size = base + (1 if ordinal < remainder else 0)
        test_start = start
        test_end = start + size
        forbidden_start = max(0, test_start - purge_bars)
        forbidden_end = min(count, test_end + embargo_bars)
        test_indices = tuple(range(test_start, test_end))
        train_indices = tuple(
            index for index in range(count) if not forbidden_start <= index < forbidden_end
        )
        if not train_indices:
            raise ValueError("purge/embargo removed the complete training set")
        fold_id = f"pkf-{ordinal:03d}"
        output.append(
            DatasetFold(
                fold_id=fold_id,
                mode=SplitMode.PURGED_KFOLD,
                train=_dataset_from_indices(
                    dataset,
                    train_indices,
                    dataset_id=f"{dataset.dataset_id}:{fold_id}:train",
                ),
                test=_dataset_from_indices(
                    dataset,
                    test_indices,
                    dataset_id=f"{dataset.dataset_id}:{fold_id}:test",
                ),
                train_bar_indices=train_indices,
                test_bar_indices=test_indices,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
            )
        )
        start = test_end
    return tuple(output)
