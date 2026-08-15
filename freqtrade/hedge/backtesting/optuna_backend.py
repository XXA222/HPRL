from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from .contracts import Candidate, OptimizationSummary, SearchMethod
from .runner import HedgeBacktestRunner
from .spaces import (
    BoolParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    ParameterSpace,
)


def _suggest(trial, name: str, spec):
    if isinstance(spec, BoolParameter):
        return trial.suggest_categorical(name, [False, True])
    if isinstance(spec, CategoricalParameter):
        return trial.suggest_categorical(name, list(spec.choices))
    if isinstance(spec, IntParameter):
        return trial.suggest_int(name, spec.low, spec.high, step=spec.step, log=spec.log)
    if isinstance(spec, DecimalParameter):
        if spec.step is not None:
            value = trial.suggest_float(
                name,
                float(spec.low),
                float(spec.high),
                step=float(spec.step),
                log=False,
            )
        else:
            value = trial.suggest_float(
                name,
                float(spec.low),
                float(spec.high),
                log=spec.log,
            )
        return Decimal(str(value))
    raise TypeError(f"unsupported parameter spec for {name}: {type(spec).__name__}")


def run_optuna_search(
    *,
    runner: HedgeBacktestRunner,
    space: ParameterSpace,
    trials: int,
    seed: int = 42,
    study_name: str = "freqtrade-hedge-bt20",
    storage: str | None = None,
    load_if_exists: bool = True,
    timeout_seconds: int | None = None,
) -> OptimizationSummary:
    """Run seeded Optuna TPE search; Optuna remains an optional hyperopt dependency."""
    if trials < 1:
        raise ValueError("Optuna trials must be positive")
    if timeout_seconds is not None and timeout_seconds < 1:
        raise ValueError("Optuna timeout must be positive")
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError(
            "Optuna is not installed; install requirements-hyperopt.txt or use grid/random search"
        ) from exc

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
    )
    evaluations = []
    started = datetime.now(UTC)

    def objective(trial):
        parameters = {name: _suggest(trial, name, space[name]) for name in sorted(space)}
        candidate = Candidate(
            candidate_id=f"optuna-{trial.number:06d}",
            parameters=parameters,
            ordinal=trial.number,
        )
        evaluation = runner.evaluate(candidate)
        evaluations.append(evaluation)
        trial.set_user_attr("candidate_id", candidate.candidate_id)
        trial.set_user_attr("feasible", evaluation.feasible)
        trial.set_user_attr("violations", list(evaluation.violations))
        for metric in (
            "total_return_ratio",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_ratio",
            "liquidation_count",
        ):
            if metric in evaluation.metrics:
                trial.set_user_attr(metric, str(evaluation.metrics[metric]))
        if not evaluation.feasible or not evaluation.objective_score.is_finite():
            return -1e300
        return float(evaluation.objective_score)

    study.optimize(objective, n_trials=trials, timeout=timeout_seconds, n_jobs=1)
    feasible = [item for item in evaluations if item.feasible]
    best = max(
        feasible,
        key=lambda item: (item.objective_score, -item.candidate.ordinal),
        default=None,
    )
    return OptimizationSummary(
        method=SearchMethod.OPTUNA,
        evaluations=tuple(sorted(evaluations, key=lambda item: item.candidate.ordinal)),
        best_candidate_id=best.candidate.candidate_id if best else None,
        started_at=started,
        completed_at=datetime.now(UTC),
        resumed=bool(storage and load_if_exists and len(study.trials) > len(evaluations)),
    )
