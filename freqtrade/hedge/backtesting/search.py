from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import sha256

from .cache import EvaluationCache
from .checkpoint import CheckpointStore
from .contracts import Candidate, OptimizationSummary, SearchMethod
from .decimal_utils import canonical_json
from .runner import HedgeBacktestRunner


class SearchEngine:
    def __init__(
        self,
        *,
        runner: HedgeBacktestRunner,
        cache: EvaluationCache | None = None,
        checkpoint: CheckpointStore | None = None,
        run_id: str = "hedge-search",
    ) -> None:
        self.runner = runner
        self.cache = cache
        self.checkpoint = checkpoint
        self.run_id = run_id
        self.engine_fingerprint = sha256(
            canonical_json(
                {
                    "engine": asdict(runner.engine_config),
                    "planner": asdict(runner.planner_config),
                    "objective": asdict(runner.objective_config),
                    "periods_per_year": runner.periods_per_year,
                }
            )
        ).hexdigest()

    def run(
        self,
        candidates: Iterable[Candidate],
        *,
        method: SearchMethod,
    ) -> OptimizationSummary:
        started = datetime.now(UTC)
        materialized = tuple(candidates)
        completed: set[str] = set()
        resumed = False
        checkpoint = self.checkpoint.load() if self.checkpoint else None
        if checkpoint is not None:
            if checkpoint.dataset_fingerprint != self.runner.dataset.fingerprint:
                raise ValueError("checkpoint dataset fingerprint mismatch")
            if checkpoint.engine_fingerprint != self.engine_fingerprint:
                raise ValueError("checkpoint engine fingerprint mismatch")
            completed.update(checkpoint.completed_candidate_ids)
            resumed = bool(completed)

        evaluations = []
        for candidate in materialized:
            cache_key = EvaluationCache.key(
                self.runner.dataset.fingerprint,
                candidate,
                self.engine_fingerprint,
            )
            cached = self.cache.get(cache_key) if self.cache else None
            if cached is not None:
                evaluation = cached.to_evaluation()
            else:
                evaluation = self.runner.evaluate(candidate)
                if self.cache:
                    self.cache.put(cache_key, evaluation)
            evaluations.append(evaluation)
            completed.add(candidate.candidate_id)
            if self.checkpoint:
                self.checkpoint.save(
                    run_id=self.run_id,
                    dataset_fingerprint=self.runner.dataset.fingerprint,
                    engine_fingerprint=self.engine_fingerprint,
                    completed_candidate_ids=completed,
                )
        feasible = [item for item in evaluations if item.feasible]
        best = max(
            feasible,
            key=lambda item: (item.objective_score, -item.candidate.ordinal),
            default=None,
        )
        return OptimizationSummary(
            method=method,
            evaluations=tuple(evaluations),
            best_candidate_id=best.candidate.candidate_id if best else None,
            started_at=started,
            completed_at=datetime.now(UTC),
            resumed=resumed,
        )
