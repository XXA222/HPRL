"""Strict JSON configuration parser for ``hedge.optimization``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from freqtrade.hedge.optimization.quality import validate_optimization_definition
from freqtrade.hedge.optimization.splits import WalkForwardSpec
from freqtrade.hedge.optimization.stress import StressScenario
from freqtrade.hedge.optimization.types import (
    ConstraintSpec,
    ObjectiveDirection,
    ObjectiveSpec,
    ParameterKind,
    ParameterSpec,
)


@dataclass(frozen=True, slots=True)
class HedgeOptimizationConfig:
    enabled: bool
    study_name: str
    sampler: str
    trials: int
    seed: int
    workers: int
    storage_path: Path
    output_directory: Path
    fail_fast: bool
    max_failures: int
    parameters: tuple[ParameterSpec, ...]
    objectives: tuple[ObjectiveSpec, ...]
    constraints: tuple[ConstraintSpec, ...]
    stress_scenarios: tuple[StressScenario, ...]
    walk_forward: WalkForwardSpec | None = None
    max_grid_candidates: int = 100_000


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _strict_keys(item: Mapping[str, Any], allowed: set[str], *, field: str) -> None:
    unknown = sorted(str(key) for key in item if key not in allowed)
    if unknown:
        raise ValueError(f"{field} contains unsupported key(s): {', '.join(unknown)}")


def _integer(value: object, *, field: str, minimum: int, default: int) -> int:
    raw = default if value is None else value
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise TypeError(f"{field} must be an integer")
    if raw < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return raw


def _boolean(value: object, *, field: str, default: bool) -> bool:
    raw = default if value is None else value
    if not isinstance(raw, bool):
        raise TypeError(f"{field} must be a boolean")
    return raw


def _parameter(raw: object, index: int) -> ParameterSpec:
    item = _mapping(raw, field=f"hedge.optimization.parameters[{index}]")
    _strict_keys(
        item,
        {"name", "path", "kind", "low", "high", "step", "choices", "log"},
        field=f"hedge.optimization.parameters[{index}]",
    )
    try:
        kind = ParameterKind(str(item["kind"]).strip().lower())
        name = str(item["name"])
        path = str(item["path"])
    except KeyError as exc:
        raise ValueError(f"optimization parameter {index} is missing {exc.args[0]}") from exc
    choices_raw = item.get("choices", ())
    if isinstance(choices_raw, (str, bytes)) or not isinstance(choices_raw, Sequence):
        raise TypeError(f"optimization parameter {name}.choices must be an array")
    return ParameterSpec(
        name=name,
        path=path,
        kind=kind,
        low=item.get("low"),
        high=item.get("high"),
        step=item.get("step"),
        choices=tuple(choices_raw),
        log=_boolean(item.get("log"), field=f"parameter.{name}.log", default=False),
    )


def _objective(raw: object, index: int) -> ObjectiveSpec:
    item = _mapping(raw, field=f"hedge.optimization.objectives[{index}]")
    _strict_keys(
        item,
        {"metric", "direction", "weight"},
        field=f"hedge.optimization.objectives[{index}]",
    )
    try:
        metric = str(item["metric"])
        direction = ObjectiveDirection(str(item["direction"]).strip().lower())
    except KeyError as exc:
        raise ValueError(f"optimization objective {index} is missing {exc.args[0]}") from exc
    return ObjectiveSpec(metric, direction, item.get("weight", Decimal(1)))


def _constraint(raw: object, index: int) -> ConstraintSpec:
    item = _mapping(raw, field=f"hedge.optimization.constraints[{index}]")
    _strict_keys(
        item,
        {"metric", "minimum", "maximum"},
        field=f"hedge.optimization.constraints[{index}]",
    )
    if "metric" not in item:
        raise ValueError(f"optimization constraint {index} is missing metric")
    return ConstraintSpec(
        metric=str(item["metric"]),
        minimum=item.get("minimum"),
        maximum=item.get("maximum"),
    )


def _stress(raw: object, index: int) -> StressScenario:
    item = _mapping(raw, field=f"hedge.optimization.stress_scenarios[{index}]")
    _strict_keys(
        item,
        {
            "name",
            "maker_fee_multiplier",
            "taker_fee_multiplier",
            "slippage_bps_add",
            "volume_participation_multiplier",
            "funding_rate_multiplier",
        },
        field=f"hedge.optimization.stress_scenarios[{index}]",
    )
    if "name" not in item:
        raise ValueError(f"stress scenario {index} is missing name")
    return StressScenario(
        name=str(item["name"]),
        maker_fee_multiplier=item.get("maker_fee_multiplier", Decimal(1)),
        taker_fee_multiplier=item.get("taker_fee_multiplier", Decimal(1)),
        slippage_bps_add=item.get("slippage_bps_add", Decimal(0)),
        volume_participation_multiplier=item.get(
            "volume_participation_multiplier", Decimal(1)
        ),
        funding_rate_multiplier=item.get("funding_rate_multiplier", Decimal(1)),
    )


def _walk_forward(raw: object) -> WalkForwardSpec | None:
    if raw is None:
        return None
    item = _mapping(raw, field="hedge.optimization.walk_forward")
    _strict_keys(
        item,
        {
            "enabled",
            "train_size",
            "validation_size",
            "test_size",
            "step_size",
            "expanding",
            "purge_size",
            "embargo_size",
            "minimum_windows",
        },
        field="hedge.optimization.walk_forward",
    )
    if not _boolean(item.get("enabled"), field="walk_forward.enabled", default=True):
        return None
    required = ("train_size", "validation_size", "test_size")
    missing = [name for name in required if name not in item]
    if missing:
        raise ValueError(f"walk_forward is missing: {', '.join(missing)}")
    return WalkForwardSpec(
        train_size=_integer(
            item["train_size"], field="walk_forward.train_size", minimum=1, default=1
        ),
        validation_size=_integer(
            item["validation_size"],
            field="walk_forward.validation_size",
            minimum=1,
            default=1,
        ),
        test_size=_integer(item["test_size"], field="walk_forward.test_size", minimum=1, default=1),
        step_size=None if item.get("step_size") is None else _integer(
            item["step_size"], field="walk_forward.step_size", minimum=1, default=1
        ),
        expanding=_boolean(item.get("expanding"), field="walk_forward.expanding", default=False),
        purge_size=_integer(
            item.get("purge_size"), field="walk_forward.purge_size", minimum=0, default=0
        ),
        embargo_size=_integer(
            item.get("embargo_size"),
            field="walk_forward.embargo_size",
            minimum=0,
            default=0,
        ),
        minimum_windows=_integer(
            item.get("minimum_windows"), field="walk_forward.minimum_windows", minimum=1, default=1
        ),
    )


def parse_optimization_config(
    config: Mapping[str, Any],
    *,
    default_output_directory: Path | None = None,
) -> HedgeOptimizationConfig:
    hedge = _mapping(config.get("hedge", {}), field="hedge")
    raw = _mapping(hedge.get("optimization", {}), field="hedge.optimization")
    _strict_keys(
        raw,
        {
            "enabled",
            "study_name",
            "sampler",
            "trials",
            "seed",
            "workers",
            "storage_path",
            "output_directory",
            "fail_fast",
            "max_failures",
            "parameters",
            "objectives",
            "constraints",
            "stress_scenarios",
            "walk_forward",
            "max_grid_candidates",
        },
        field="hedge.optimization",
    )
    parameters_raw = raw.get("parameters", ())
    objectives_raw = raw.get("objectives", ())
    constraints_raw = raw.get("constraints", ())
    stresses_raw = raw.get("stress_scenarios", ({"name": "baseline"},))
    for value, field in (
        (parameters_raw, "parameters"),
        (objectives_raw, "objectives"),
        (constraints_raw, "constraints"),
        (stresses_raw, "stress_scenarios"),
    ):
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise TypeError(f"hedge.optimization.{field} must be an array")
    parameters = tuple(_parameter(item, index) for index, item in enumerate(parameters_raw))
    objectives = tuple(_objective(item, index) for index, item in enumerate(objectives_raw))
    constraints = tuple(_constraint(item, index) for index, item in enumerate(constraints_raw))
    stresses = tuple(_stress(item, index) for index, item in enumerate(stresses_raw))
    if not parameters:
        raise ValueError("hedge.optimization.parameters cannot be empty")
    if not objectives:
        raise ValueError("hedge.optimization.objectives cannot be empty")
    if len({item.name for item in stresses}) != len(stresses):
        raise ValueError("stress scenario names must be unique")
    sampler = str(raw.get("sampler", "random")).strip().lower()
    if sampler not in {"grid", "random"}:
        raise ValueError("hedge.optimization.sampler must be grid or random")
    study_name = str(raw.get("study_name", "hedge-backtest-optimization")).strip()
    if not study_name:
        raise ValueError("hedge.optimization.study_name cannot be empty")
    output_root = default_output_directory or Path("user_data") / "hyperopt_results"
    output = Path(str(raw.get("output_directory", output_root / study_name)))
    storage = Path(str(raw.get("storage_path", output / "study.sqlite")))
    parsed = HedgeOptimizationConfig(
        enabled=_boolean(raw.get("enabled"), field="hedge.optimization.enabled", default=True),
        study_name=study_name,
        sampler=sampler,
        trials=_integer(
            raw.get("trials"), field="hedge.optimization.trials", minimum=1, default=100
        ),
        seed=_integer(raw.get("seed"), field="hedge.optimization.seed", minimum=0, default=42),
        workers=_integer(
            raw.get("workers"), field="hedge.optimization.workers", minimum=1, default=1
        ),
        storage_path=storage,
        output_directory=output,
        fail_fast=_boolean(
            raw.get("fail_fast"), field="hedge.optimization.fail_fast", default=False
        ),
        max_failures=_integer(
            raw.get("max_failures"),
            field="hedge.optimization.max_failures",
            minimum=0,
            default=10,
        ),
        parameters=parameters,
        objectives=objectives,
        constraints=constraints,
        stress_scenarios=stresses,
        walk_forward=_walk_forward(raw.get("walk_forward")),
        max_grid_candidates=_integer(
            raw.get("max_grid_candidates"),
            field="hedge.optimization.max_grid_candidates",
            minimum=1,
            default=100_000,
        ),
    )
    validate_optimization_definition(parsed)
    return parsed
