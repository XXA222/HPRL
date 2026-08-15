from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise

from .contracts import BacktestEvaluation, Candidate
from .decimal_utils import ZERO
from .runner import HedgeBacktestRunner
from .spaces import ParameterSpace


@dataclass(frozen=True, slots=True)
class ParameterSensitivity:
    parameter: str
    evaluations: tuple[BacktestEvaluation, ...]
    score_min: Decimal
    score_max: Decimal
    score_range: Decimal
    best_value: object
    worst_value: object
    monotonic_direction: str


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    baseline: BacktestEvaluation
    parameters: tuple[ParameterSensitivity, ...]


def _monotonic(scores: list[Decimal]) -> str:
    if len(scores) < 2:
        return "flat"
    nondecreasing = all(right >= left for left, right in pairwise(scores))
    nonincreasing = all(right <= left for left, right in pairwise(scores))
    if nondecreasing and nonincreasing:
        return "flat"
    if nondecreasing:
        return "increasing"
    if nonincreasing:
        return "decreasing"
    return "non_monotonic"


def one_at_a_time_sensitivity(
    *,
    runner: HedgeBacktestRunner,
    baseline: Candidate,
    space: ParameterSpace,
    max_values_per_parameter: int = 25,
) -> SensitivityResult:
    if max_values_per_parameter < 2:
        raise ValueError("max_values_per_parameter must be at least two")
    baseline_evaluation = runner.evaluate(baseline)
    output: list[ParameterSensitivity] = []
    for name in sorted(space):
        values = space[name].grid_values()
        if len(values) > max_values_per_parameter:
            raise ValueError(
                f"sensitivity parameter {name} has {len(values)} values; "
                f"limit={max_values_per_parameter}"
            )
        evaluations: list[BacktestEvaluation] = []
        for ordinal, value in enumerate(values):
            params = dict(baseline.parameters)
            params[name] = value
            candidate = Candidate(
                candidate_id=f"{baseline.candidate_id}:sensitivity:{name}:{ordinal}",
                parameters=params,
                ordinal=ordinal,
            )
            evaluations.append(runner.evaluate(candidate))
        finite = [item for item in evaluations if item.objective_score.is_finite()]
        ranked = finite or evaluations
        best = max(ranked, key=lambda item: item.objective_score)
        worst = min(ranked, key=lambda item: item.objective_score)
        scores = [item.objective_score for item in evaluations]
        finite_scores = [item for item in scores if item.is_finite()]
        score_min = min(finite_scores, default=Decimal("-Infinity"))
        score_max = max(finite_scores, default=Decimal("-Infinity"))
        output.append(
            ParameterSensitivity(
                parameter=name,
                evaluations=tuple(evaluations),
                score_min=score_min,
                score_max=score_max,
                score_range=(score_max - score_min if finite_scores else ZERO),
                best_value=best.candidate.parameters[name],
                worst_value=worst.candidate.parameters[name],
                monotonic_direction=_monotonic(scores),
            )
        )
    return SensitivityResult(baseline=baseline_evaluation, parameters=tuple(output))
