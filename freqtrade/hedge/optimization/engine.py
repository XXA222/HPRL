"""Deterministic Hedge optimization engine with resume and fail-closed trials."""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from decimal import Decimal
from time import monotonic
from typing import Any, Protocol

from freqtrade.hedge.optimization.aggregation import aggregate_metric_sets
from freqtrade.hedge.optimization.quality import (
    validate_report_finite,
    validate_report_mapping,
)
from freqtrade.hedge.optimization.config import HedgeOptimizationConfig
from freqtrade.hedge.optimization.config_patch import apply_parameters
from freqtrade.hedge.optimization.constraints import evaluate_constraints
from freqtrade.hedge.optimization.fingerprint import (
    fingerprint,
    parameter_fingerprint,
    study_fingerprint,
)
from freqtrade.hedge.optimization.metrics import normalize_report
from freqtrade.hedge.optimization.objectives import objective_values, scalar_score
from freqtrade.hedge.optimization.pareto import pareto_front
from freqtrade.hedge.optimization.space import ParameterSpace
from freqtrade.hedge.optimization.splits import WalkForwardWindow, build_walk_forward_windows
from freqtrade.hedge.optimization.store import StudyStore
from freqtrade.hedge.optimization.stress import StressScenario, apply_stress_to_config
from freqtrade.hedge.optimization.types import OptimizationResult, TrialRecord, TrialStatus


class TrialEvaluator(Protocol):
    def __call__(
        self,
        config: Mapping[str, Any],
        context: EvaluationContext,
    ) -> Mapping[str, object]: ...


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    trial_id: int
    seed: int
    stress_scenario: StressScenario
    window: WalkForwardWindow | None
    evaluation_index: int


class OptimizationEngine:
    def __init__(
        self,
        *,
        base_config: Mapping[str, Any],
        optimization_config: HedgeOptimizationConfig,
        evaluator: TrialEvaluator,
        dataset_fingerprint: str,
        dataset_size: int | None = None,
        timestamps: Sequence[Any] | None = None,
        store: StudyStore | None = None,
    ) -> None:
        if not dataset_fingerprint.strip():
            raise ValueError("dataset fingerprint cannot be empty")
        self.base_config = dict(base_config)
        self.config = optimization_config
        self.evaluator = evaluator
        self.dataset_fingerprint = dataset_fingerprint
        self.space = ParameterSpace(self.config.parameters)
        self.store = store or StudyStore(self.config.storage_path)
        if self.config.walk_forward is None:
            self.windows: tuple[WalkForwardWindow | None, ...] = (None,)
        else:
            if dataset_size is None:
                raise ValueError("dataset_size is required for walk-forward optimization")
            self.windows = build_walk_forward_windows(
                dataset_size,
                self.config.walk_forward,
                timestamps=timestamps,
            )
        self.study_fingerprint = study_fingerprint(
            parameter_specs=self.config.parameters,
            objective_specs=self.config.objectives,
            constraint_specs=self.config.constraints,
            dataset_fingerprint=self.dataset_fingerprint,
            seed=self.config.seed,
            sampler=self.config.sampler,
            extra_definition={
                "engine_schema": "hedge-optimization-engine-v1",
                "stress_scenarios": self.config.stress_scenarios,
                "walk_forward": self.config.walk_forward,
            },
        )

    def _definition(self) -> dict[str, object]:
        return {
            "parameters": self.config.parameters,
            "objectives": self.config.objectives,
            "constraints": self.config.constraints,
            "sampler": self.config.sampler,
            "seed": self.config.seed,
            "stress_scenarios": self.config.stress_scenarios,
            "walk_forward": self.config.walk_forward,
            "study_fingerprint": self.study_fingerprint,
        }

    def _evaluate_one(self, trial_id: int, parameters: Mapping[str, object]) -> TrialRecord:
        started = monotonic()
        parameter_hash = parameter_fingerprint(parameters)
        worker = threading.current_thread().name
        try:
            patched = apply_parameters(self.base_config, self.config.parameters, parameters)
            config_hash = fingerprint(patched)
            metric_sets = []
            evaluation_index = 0
            for scenario in self.config.stress_scenarios:
                stressed = apply_stress_to_config(patched, scenario)
                for window in self.windows:
                    context = EvaluationContext(
                        trial_id=trial_id,
                        seed=self.config.seed + trial_id,
                        stress_scenario=scenario,
                        window=window,
                        evaluation_index=evaluation_index,
                    )
                    report = validate_report_mapping(self.evaluator(stressed, context))
                    validate_report_finite(report)
                    metric_sets.append(normalize_report(report))
                    evaluation_index += 1
            aggregate_suffixes = (
                "__median",
                "__min",
                "__max",
                "__worst",
                "__best",
                "__std",
                "__range",
            )

            def base_metric(name: str) -> str:
                for suffix in aggregate_suffixes:
                    if name.endswith(suffix):
                        return name[: -len(suffix)]
                return name

            required = tuple(
                dict.fromkeys(
                    [base_metric(item.metric) for item in self.config.objectives]
                    + [base_metric(item.metric) for item in self.config.constraints]
                )
            )
            metrics = aggregate_metric_sets(metric_sets, required_metrics=required)
            constraints = evaluate_constraints(metrics, self.config.constraints)
            values = objective_values(metrics, self.config.objectives)
            score = scalar_score(metrics, self.config.objectives)
            status = TrialStatus.COMPLETE if constraints.feasible else TrialStatus.INFEASIBLE
            return TrialRecord(
                trial_id=trial_id,
                parameter_hash=parameter_hash,
                parameters=dict(parameters),
                status=status,
                metrics=metrics,
                objective_values=values,
                scalar_score=score,
                constraint_violations=constraints.violations,
                duration_seconds=Decimal(str(monotonic() - started)),
                dataset_fingerprint=self.dataset_fingerprint,
                config_fingerprint=config_hash,
                worker=worker,
            )
        except Exception as exc:
            return TrialRecord(
                trial_id=trial_id,
                parameter_hash=parameter_hash,
                parameters=dict(parameters),
                status=TrialStatus.FAILED,
                error=f"{type(exc).__name__}: {exc}",
                duration_seconds=Decimal(str(monotonic() - started)),
                dataset_fingerprint=self.dataset_fingerprint,
                worker=worker,
            )

    def _run_pending(
        self,
        pending: Sequence[tuple[int, Mapping[str, object]]],
    ) -> tuple[TrialRecord, ...]:
        if not pending:
            return ()
        records: list[TrialRecord] = []
        failures = 0

        def accept(record: TrialRecord) -> None:
            nonlocal failures
            self.store.save_trial(self.config.study_name, record)
            records.append(record)
            if record.status is TrialStatus.FAILED:
                failures += 1
                if self.config.fail_fast:
                    raise RuntimeError(f"trial {record.trial_id} failed: {record.error}")
                if self.config.max_failures and failures > self.config.max_failures:
                    raise RuntimeError(
                        f"optimization exceeded max_failures={self.config.max_failures}"
                    )

        if self.config.workers == 1:
            for trial_id, parameters in pending:
                accept(self._evaluate_one(trial_id, parameters))
            return tuple(records)

        executor = ThreadPoolExecutor(
            max_workers=self.config.workers,
            thread_name_prefix="hedge-opt",
        )
        futures: dict[Future[TrialRecord], int] = {
            executor.submit(self._evaluate_one, trial_id, parameters): trial_id
            for trial_id, parameters in pending
        }
        try:
            for future in as_completed(futures):
                accept(future.result())
        except Exception:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        return tuple(records)

    def run(self) -> OptimizationResult:
        if not self.config.enabled:
            raise ValueError("hedge optimization is disabled")
        self.store.initialize_study(
            study_name=self.config.study_name,
            study_fingerprint=self.study_fingerprint,
            dataset_fingerprint=self.dataset_fingerprint,
            definition=self._definition(),
        )
        candidates = self.space.candidates(
            sampler=self.config.sampler,
            trials=self.config.trials,
            seed=self.config.seed,
            max_grid_candidates=self.config.max_grid_candidates,
        )
        resumed = self.store.completed_by_parameter_hash(self.config.study_name)
        records: list[TrialRecord] = []
        pending: list[tuple[int, Mapping[str, object]]] = []
        resumed_count = 0
        for trial_id, parameters in enumerate(candidates):
            parameter_hash = parameter_fingerprint(parameters)
            prior = resumed.get(parameter_hash)
            if prior is not None:
                records.append(prior)
                resumed_count += 1
            else:
                pending.append((trial_id, parameters))
        records.extend(self._run_pending(pending))
        records.sort(key=lambda item: item.trial_id)
        front = pareto_front(records, self.config.objectives)
        completed = [item for item in records if item.status is TrialStatus.COMPLETE]
        best = max(
            completed,
            key=lambda item: (
                item.scalar_score if item.scalar_score is not None else Decimal("-Infinity"),
                -item.trial_id,
            ),
            default=None,
        )
        return OptimizationResult(
            study_name=self.config.study_name,
            trials=tuple(records),
            pareto_trial_ids=tuple(item.trial_id for item in front),
            best_trial_id=None if best is None else best.trial_id,
            objective_specs=self.config.objectives,
            dataset_fingerprint=self.dataset_fingerprint,
            study_fingerprint=self.study_fingerprint,
            resumed_trials=resumed_count,
        )
