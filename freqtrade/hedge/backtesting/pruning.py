from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from math import ceil

from freqtrade.hedge.simulation.exchange import BarEvent

from .contracts import (
    BacktestDataset,
    BacktestEvaluation,
    Candidate,
    EngineConfig,
    ObjectiveConfig,
)
from .dataset import slice_dataset
from .runner import HedgeBacktestRunner


DEFAULT_STAGE_FRACTIONS = (Decimal("0.25"), Decimal("0.5"), Decimal(1))
DEFAULT_KEEP_RATIO = Decimal("0.5")


@dataclass(frozen=True, slots=True)
class PruningStage:
    fraction: Decimal
    bar_count: int
    evaluations: tuple[BacktestEvaluation, ...]
    survivor_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SuccessiveHalvingResult:
    stages: tuple[PruningStage, ...]
    finalists: tuple[BacktestEvaluation, ...]
    pruned_candidate_ids: tuple[str, ...]


def _prefix(dataset: BacktestDataset, bar_count: int) -> BacktestDataset:
    bars = tuple(event for event in dataset.events if isinstance(event, BarEvent))
    if not 1 <= bar_count <= len(bars):
        raise ValueError("prefix bar_count outside dataset")
    return slice_dataset(
        dataset,
        start=bars[0].timestamp,
        end=bars[bar_count - 1].timestamp,
        dataset_id=f"{dataset.dataset_id}:prefix:{bar_count}",
    )


def successive_halving(
    *,
    dataset: BacktestDataset,
    candidates: Iterable[Candidate],
    stage_fractions: tuple[Decimal, ...] = DEFAULT_STAGE_FRACTIONS,
    keep_ratio: Decimal = DEFAULT_KEEP_RATIO,
    engine_config: EngineConfig | None = None,
    planner_config=None,
    objective_config: ObjectiveConfig | None = None,
) -> SuccessiveHalvingResult:
    materialized = tuple(candidates)
    if not materialized:
        raise ValueError("successive halving requires candidates")
    if any(not Decimal(0) < item <= Decimal(1) for item in stage_fractions):
        raise ValueError("stage fractions must be in (0, 1]")
    if (
        tuple(sorted(set(stage_fractions))) != stage_fractions
        or stage_fractions[-1] != Decimal(1)
    ):
        raise ValueError("stage fractions must be unique, increasing and end at 1")
    if not Decimal(0) < keep_ratio <= Decimal(1):
        raise ValueError("keep_ratio must be in (0, 1]")
    total_bars = dataset.bar_count
    survivors = list(materialized)
    stages: list[PruningStage] = []
    pruned: set[str] = set()
    for stage_index, fraction in enumerate(stage_fractions):
        count = max(2, min(total_bars, ceil(float(Decimal(total_bars) * fraction))))
        stage_dataset = _prefix(dataset, count)
        runner = HedgeBacktestRunner(
            dataset=stage_dataset,
            engine_config=engine_config,
            planner_config=planner_config,
            objective_config=objective_config,
        )
        evaluations = [runner.evaluate(candidate) for candidate in survivors]
        ranked = sorted(
            evaluations,
            key=lambda item: (item.feasible, item.objective_score, -item.candidate.ordinal),
            reverse=True,
        )
        is_final = stage_index == len(stage_fractions) - 1
        keep = len(ranked) if is_final else max(1, ceil(len(ranked) * float(keep_ratio)))
        survivor_ids = tuple(item.candidate.candidate_id for item in ranked[:keep])
        discarded = {item.candidate.candidate_id for item in ranked[keep:]}
        pruned.update(discarded)
        stages.append(
            PruningStage(
                fraction=fraction,
                bar_count=count,
                evaluations=tuple(evaluations),
                survivor_ids=survivor_ids,
            )
        )
        survivors = [item.candidate for item in ranked[:keep]]
    return SuccessiveHalvingResult(
        stages=tuple(stages),
        finalists=tuple(stages[-1].evaluations),
        pruned_candidate_ids=tuple(sorted(pruned)),
    )
