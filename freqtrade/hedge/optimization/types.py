"""Strongly typed contracts for deterministic Hedge parameter optimization."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum


DEFAULT_OBJECTIVE_WEIGHT = Decimal(1)
DEFAULT_TRIAL_DURATION = Decimal(0)


class ParameterKind(StrEnum):
    DECIMAL = "decimal"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class TrialStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    PRUNED = "pruned"
    FAILED = "failed"
    INFEASIBLE = "infeasible"


def exact_decimal(value: object, *, field_name: str) -> Decimal:
    """Convert a JSON-compatible scalar to a finite :class:`Decimal`."""

    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be numeric, not boolean")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a valid decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """One optimization parameter mapped to a fail-closed configuration path."""

    name: str
    path: str
    kind: ParameterKind
    low: Decimal | int | None = None
    high: Decimal | int | None = None
    step: Decimal | int | None = None
    choices: tuple[object, ...] = ()
    log: bool = False

    def __post_init__(self) -> None:  # noqa: C901
        if not isinstance(self.log, bool):
            raise TypeError("parameter log flag must be boolean")
        if not self.name.strip():
            raise ValueError("parameter name cannot be empty")
        if not self.path.startswith("hedge."):
            raise ValueError("optimization parameter paths must stay below hedge.*")
        if ".." in self.path or self.path.endswith("."):
            raise ValueError("parameter path is malformed")
        if self.kind is ParameterKind.CATEGORICAL:
            if len(self.choices) < 2:
                raise ValueError("categorical parameters require at least two choices")
            if self.low is not None or self.high is not None or self.step is not None:
                raise ValueError("categorical parameters cannot define low/high/step")
            return
        if self.kind is ParameterKind.BOOLEAN:
            if (
                self.choices
                or self.low is not None
                or self.high is not None
                or self.step is not None
            ):
                raise ValueError("boolean parameters cannot define ranges or choices")
            return
        if self.low is None or self.high is None:
            raise ValueError("numeric parameters require low and high")
        if self.kind is ParameterKind.DECIMAL:
            low = exact_decimal(self.low, field_name=f"{self.name}.low")
            high = exact_decimal(self.high, field_name=f"{self.name}.high")
            step = None if self.step is None else exact_decimal(
                self.step, field_name=f"{self.name}.step"
            )
            object.__setattr__(self, "low", low)
            object.__setattr__(self, "high", high)
            object.__setattr__(self, "step", step)
        elif self.kind is ParameterKind.INTEGER:
            if isinstance(self.low, bool) or isinstance(self.high, bool):
                raise TypeError("integer bounds cannot be boolean")
            if not isinstance(self.low, int) or not isinstance(self.high, int):
                raise TypeError("integer bounds must be integers")
            if self.step is not None and (
                isinstance(self.step, bool) or not isinstance(self.step, int)
            ):
                raise TypeError("integer step must be an integer")
        if self.low > self.high:
            raise ValueError("parameter low cannot exceed high")
        if self.step is not None and self.step <= 0:
            raise ValueError("parameter step must be positive")
        if self.log and self.low <= 0:
            raise ValueError("log-scaled parameters require a positive lower bound")
        if self.log and self.step is not None:
            raise ValueError("log-scaled parameters cannot define a linear step")


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    metric: str
    direction: ObjectiveDirection
    weight: Decimal = DEFAULT_OBJECTIVE_WEIGHT

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("objective metric cannot be empty")
        weight = exact_decimal(self.weight, field_name=f"objective.{self.metric}.weight")
        if weight <= 0:
            raise ValueError("objective weight must be positive")
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True, slots=True)
class ConstraintSpec:
    metric: str
    minimum: Decimal | None = None
    maximum: Decimal | None = None

    def __post_init__(self) -> None:
        if not self.metric.strip():
            raise ValueError("constraint metric cannot be empty")
        minimum = self.minimum
        maximum = self.maximum
        if minimum is None and maximum is None:
            raise ValueError("constraint requires minimum and/or maximum")
        if minimum is not None:
            minimum = exact_decimal(minimum, field_name=f"constraint.{self.metric}.minimum")
            object.__setattr__(self, "minimum", minimum)
        if maximum is not None:
            maximum = exact_decimal(maximum, field_name=f"constraint.{self.metric}.maximum")
            object.__setattr__(self, "maximum", maximum)
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError("constraint minimum cannot exceed maximum")


@dataclass(frozen=True, slots=True)
class TrialRecord:
    trial_id: int
    parameter_hash: str
    parameters: Mapping[str, object]
    status: TrialStatus
    metrics: Mapping[str, Decimal] = field(default_factory=dict)
    objective_values: tuple[Decimal, ...] = ()
    scalar_score: Decimal | None = None
    constraint_violations: tuple[str, ...] = ()
    error: str | None = None
    duration_seconds: Decimal = DEFAULT_TRIAL_DURATION
    dataset_fingerprint: str = ""
    config_fingerprint: str = ""
    worker: str = ""


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    study_name: str
    trials: tuple[TrialRecord, ...]
    pareto_trial_ids: tuple[int, ...]
    best_trial_id: int | None
    objective_specs: tuple[ObjectiveSpec, ...]
    dataset_fingerprint: str
    study_fingerprint: str
    resumed_trials: int = 0

    @property
    def completed_trials(self) -> tuple[TrialRecord, ...]:
        return tuple(item for item in self.trials if item.status is TrialStatus.COMPLETE)
