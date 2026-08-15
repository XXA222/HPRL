from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .contracts import BacktestEvaluation, Candidate
from .runner import HedgeBacktestRunner


@dataclass(frozen=True, slots=True)
class ParallelEvaluationResult:
    evaluations: tuple[BacktestEvaluation, ...]
    worker_count: int


def evaluate_parallel(
    *,
    runner: HedgeBacktestRunner,
    candidates: Iterable[Candidate],
    workers: int = 1,
) -> ParallelEvaluationResult:
    materialized = tuple(candidates)
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1 or len(materialized) < 2:
        return ParallelEvaluationResult(
            evaluations=tuple(runner.evaluate(item) for item in materialized),
            worker_count=1,
        )
    futures: dict[Future[BacktestEvaluation], Candidate] = {}
    results: list[BacktestEvaluation] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hedge-bt") as executor:
        for candidate in materialized:
            futures[executor.submit(runner.evaluate, candidate)] = candidate
        try:
            for future in as_completed(futures):
                results.append(future.result())
        except Exception:
            for future in futures:
                future.cancel()
            raise
    results.sort(key=lambda item: (item.candidate.ordinal, item.candidate.candidate_id))
    return ParallelEvaluationResult(evaluations=tuple(results), worker_count=workers)
