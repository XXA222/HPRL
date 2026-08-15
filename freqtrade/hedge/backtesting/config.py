from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from decimal import Decimal
from pathlib import Path

from freqtrade.hedge.planning.context import PlannerConfig

from .contracts import EngineConfig, ObjectiveConfig, SearchMethod
from .decimal_utils import to_decimal
from .spaces import (
    BoolParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    ParameterSpace,
)


def _strict_keys(mapping: Mapping[str, object], allowed: set[str], *, section: str) -> None:
    extra = sorted(set(mapping) - allowed)
    if extra:
        raise ValueError(f"unknown {section} key(s): {', '.join(extra)}")


def _strict_bool(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field} must be bool")
    return value


def _strict_int(value: object, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    return value


def _convert_dataclass(cls, raw: Mapping[str, object]):
    allowed = {item.name for item in fields(cls)}
    _strict_keys(raw, allowed, section=cls.__name__)
    values: dict[str, object] = {}
    defaults = cls()
    for name, value in raw.items():
        current = getattr(defaults, name)
        if isinstance(current, Decimal):
            values[name] = to_decimal(value, field=name)
        elif isinstance(current, bool):
            values[name] = _strict_bool(value, field=name)
        elif isinstance(current, int):
            values[name] = _strict_int(value, field=name)
        else:
            values[name] = value
    return cls(**values)


def parse_parameter_space(raw: Mapping[str, object]) -> ParameterSpace:
    output = {}
    for name, spec_raw in raw.items():
        if not isinstance(spec_raw, Mapping):
            raise TypeError(f"parameter {name} spec must be an object")
        kind = str(spec_raw.get("type", ""))
        if kind == "decimal":
            _strict_keys(
                dict(spec_raw),
                {"type", "low", "high", "step", "log"},
                section=f"space.{name}",
            )
            output[name] = DecimalParameter(
                low=to_decimal(spec_raw["low"], field=f"{name}.low"),
                high=to_decimal(spec_raw["high"], field=f"{name}.high"),
                step=(
                    to_decimal(spec_raw["step"], field=f"{name}.step")
                    if spec_raw.get("step") is not None
                    else None
                ),
                log=_strict_bool(spec_raw.get("log", False), field=f"{name}.log"),
            )
        elif kind == "int":
            _strict_keys(
                dict(spec_raw),
                {"type", "low", "high", "step", "log"},
                section=f"space.{name}",
            )
            output[name] = IntParameter(
                low=_strict_int(spec_raw["low"], field=f"{name}.low"),
                high=_strict_int(spec_raw["high"], field=f"{name}.high"),
                step=_strict_int(spec_raw.get("step", 1), field=f"{name}.step", minimum=1),
                log=_strict_bool(spec_raw.get("log", False), field=f"{name}.log"),
            )
        elif kind == "categorical":
            _strict_keys(dict(spec_raw), {"type", "choices"}, section=f"space.{name}")
            choices = spec_raw.get("choices")
            if not isinstance(choices, list):
                raise TypeError(f"parameter {name}.choices must be a list")
            output[name] = CategoricalParameter(tuple(choices))
        elif kind == "bool":
            _strict_keys(dict(spec_raw), {"type"}, section=f"space.{name}")
            output[name] = BoolParameter()
        else:
            raise ValueError(f"parameter {name} has unsupported type {kind!r}")
    return output


def load_optimization_config(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("optimization config root must be an object")
    allowed = {
        "schema_version", "method", "engine", "planner", "objective", "space",
        "random_count", "seed", "max_candidates", "workers", "walk_forward",
    }
    _strict_keys(raw, allowed, section="optimization config")
    if raw.get("schema_version") != "hedge-optimization-config-v1":
        raise ValueError("unsupported optimization config schema")
    method = SearchMethod(str(raw.get("method", "grid")))
    engine_raw = raw.get("engine", {})
    planner_raw = raw.get("planner", {})
    objective_raw = raw.get("objective", {})
    if not all(isinstance(item, Mapping) for item in (engine_raw, planner_raw, objective_raw)):
        raise TypeError("engine, planner and objective sections must be objects")
    objective_allowed = {"weights", "minimums", "maximums", "reject_liquidation"}
    _strict_keys(objective_raw, objective_allowed, section="objective")
    def decimal_map(value: object, *, section: str) -> dict[str, Decimal]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{section} must be an object")
        return {str(key): to_decimal(item, field=str(key)) for key, item in value.items()}
    objective = ObjectiveConfig(
        weights=(
            decimal_map(objective_raw["weights"], section="objective.weights")
            if "weights" in objective_raw
            else ObjectiveConfig().weights
        ),
        minimums=decimal_map(objective_raw.get("minimums", {}), section="objective.minimums"),
        maximums=(
            decimal_map(objective_raw["maximums"], section="objective.maximums")
            if "maximums" in objective_raw
            else ObjectiveConfig().maximums
        ),
        reject_liquidation=_strict_bool(
            objective_raw.get("reject_liquidation", True),
            field="objective.reject_liquidation",
        ),
    )
    space_raw = raw.get("space", {})
    if not isinstance(space_raw, Mapping):
        raise TypeError("space must be an object")
    walk_raw = raw.get("walk_forward", {})
    if not isinstance(walk_raw, Mapping):
        raise TypeError("walk_forward must be an object")
    _strict_keys(
        walk_raw,
        {"train_bars", "test_bars", "step_bars", "gap_bars", "anchored"},
        section="walk_forward",
    )
    test_bars = _strict_int(
        walk_raw.get("test_bars", 250), field="walk_forward.test_bars", minimum=1
    )
    walk_forward = {
        "train_bars": _strict_int(
            walk_raw.get("train_bars", 1000), field="walk_forward.train_bars", minimum=1
        ),
        "test_bars": test_bars,
        "step_bars": _strict_int(
            walk_raw.get("step_bars", test_bars),
            field="walk_forward.step_bars",
            minimum=1,
        ),
        "gap_bars": _strict_int(
            walk_raw.get("gap_bars", 0), field="walk_forward.gap_bars", minimum=0
        ),
        "anchored": _strict_bool(
            walk_raw.get("anchored", False), field="walk_forward.anchored"
        ),
    }
    return {
        "method": method,
        "engine_config": _convert_dataclass(EngineConfig, engine_raw),
        "planner_config": _convert_dataclass(PlannerConfig, planner_raw),
        "objective_config": objective,
        "space": parse_parameter_space(space_raw),
        "random_count": _strict_int(raw.get("random_count", 100), field="random_count", minimum=1),
        "seed": _strict_int(raw.get("seed", 42), field="seed", minimum=0),
        "max_candidates": _strict_int(
            raw.get("max_candidates", 100_000), field="max_candidates", minimum=1
        ),
        "workers": _strict_int(raw.get("workers", 1), field="workers", minimum=1),
        "walk_forward": walk_forward,
    }
