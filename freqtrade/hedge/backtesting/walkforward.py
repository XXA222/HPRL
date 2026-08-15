from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from statistics import fmean

from .contracts import (
    BacktestEvaluation,
    Candidate,
    EngineConfig,
    ObjectiveConfig,
    SearchMethod,
)
from .runner import HedgeBacktestRunner
from .search import SearchEngine
from .splits import DatasetFold


@dataclass(frozen=True, slots=True)
class WalkForwardFoldResult:
    fold_id: str
    selected_candidate: Candidate
    train_evaluation: BacktestEvaluation
    test_evaluation: BacktestEvaluation


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    folds: tuple[WalkForwardFoldResult, ...]
    started_at: datetime
    completed_at: datetime
    mean_test_score: Decimal
    profitable_fold_ratio: Decimal
    candidate_stability_ratio: Decimal


def run_walk_forward(
    *,
    folds: Iterable[DatasetFold],
    candidates: Iterable[Candidate],
    engine_config: EngineConfig | None = None,
    planner_config=None,
    objective_config: ObjectiveConfig | None = None,
    periods_per_year: int = 365 * 24 * 60,
) -> WalkForwardResult:
    started = datetime.now(UTC)
    materialized_candidates = tuple(candidates)
    if not materialized_candidates:
        raise ValueError("walk-forward requires at least one candidate")
    materialized_folds = tuple(folds)
    if not materialized_folds:
        raise ValueError("walk-forward requires at least one fold")
    results: list[WalkForwardFoldResult] = []
    for fold in materialized_folds:
        train_runner = HedgeBacktestRunner(
            dataset=fold.train,
            engine_config=engine_config,
            planner_config=planner_config,
            objective_config=objective_config,
            periods_per_year=periods_per_year,
        )
        summary = SearchEngine(runner=train_runner, run_id=fold.fold_id).run(
            materialized_candidates,
            method=SearchMethod.GRID,
        )
        if summary.best_candidate_id is None:
            raise ValueError(f"fold {fold.fold_id} has no feasible candidate")
        selected = next(
            item
            for item in materialized_candidates
            if item.candidate_id == summary.best_candidate_id
        )
        train_eval = next(
            item
            for item in summary.evaluations
            if item.candidate.candidate_id == selected.candidate_id
        )
        test_runner = HedgeBacktestRunner(
            dataset=fold.test,
            engine_config=engine_config,
            planner_config=planner_config,
            objective_config=objective_config,
            periods_per_year=periods_per_year,
        )
        test_eval = test_runner.evaluate(selected)
        results.append(
            WalkForwardFoldResult(
                fold_id=fold.fold_id,
                selected_candidate=selected,
                train_evaluation=train_eval,
                test_evaluation=test_eval,
            )
        )
    scores = [float(item.test_evaluation.objective_score) for item in results]
    profitable = sum(
        Decimal(str(item.test_evaluation.metrics.get("total_return_ratio", "0"))) > 0
        for item in results
    )
    selected_ids = [item.selected_candidate.candidate_id for item in results]
    mode_count = max(selected_ids.count(item) for item in set(selected_ids))
    return WalkForwardResult(
        folds=tuple(results),
        started_at=started,
        completed_at=datetime.now(UTC),
        mean_test_score=Decimal(str(fmean(scores))) if scores else Decimal(0),
        profitable_fold_ratio=Decimal(profitable) / Decimal(len(results)),
        candidate_stability_ratio=Decimal(mode_count) / Decimal(len(results)),
    )
