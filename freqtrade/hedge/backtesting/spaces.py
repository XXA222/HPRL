from __future__ import annotations

import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from itertools import product
from typing import Protocol, runtime_checkable

from .contracts import Candidate
from .decimal_utils import canonical_json, to_decimal


@runtime_checkable
class ParameterSpec(Protocol):
    def grid_values(self) -> tuple[object, ...]: ...

    def sample(self, rng: random.Random) -> object: ...

    def validate(self, value: object) -> object: ...


@dataclass(frozen=True, slots=True)
class DecimalParameter:
    low: Decimal
    high: Decimal
    step: Decimal | None = None
    log: bool = False

    def __post_init__(self) -> None:
        if any(not item.is_finite() for item in (self.low, self.high)):
            raise ValueError("decimal parameter bounds must be finite")
        if self.high < self.low:
            raise ValueError("decimal parameter high must be >= low")
        if self.step is not None and (not self.step.is_finite() or self.step <= 0):
            raise ValueError("decimal parameter step must be positive")
        if self.log and self.low <= 0:
            raise ValueError("log decimal parameter low must be positive")
        if self.log and self.step is not None:
            raise ValueError("log decimal parameter cannot use linear step")

    def validate(self, value: object) -> Decimal:
        result = to_decimal(value)
        if not self.low <= result <= self.high:
            raise ValueError(f"decimal parameter {result} outside [{self.low}, {self.high}]")
        if self.step is not None:
            units = (result - self.low) / self.step
            if units != units.to_integral_value():
                raise ValueError(f"decimal parameter {result} is off step {self.step}")
        return result

    def grid_values(self) -> tuple[Decimal, ...]:
        if self.step is None:
            if self.low == self.high:
                return (self.low,)
            raise ValueError("grid search requires a step for ranged decimal parameters")
        count = int(((self.high - self.low) / self.step).to_integral_value())
        values = tuple(self.low + self.step * index for index in range(count + 1))
        if not values or values[-1] != self.high:
            raise ValueError("decimal range must be exactly divisible by step")
        return values

    def sample(self, rng: random.Random) -> Decimal:
        if self.low == self.high:
            return self.low
        if self.log:
            raw = math.exp(rng.uniform(math.log(float(self.low)), math.log(float(self.high))))
            return Decimal(str(raw))
        if self.step is not None:
            values = self.grid_values()
            return values[rng.randrange(len(values))]
        raw = rng.uniform(float(self.low), float(self.high))
        return Decimal(str(raw))


@dataclass(frozen=True, slots=True)
class IntParameter:
    low: int
    high: int
    step: int = 1
    log: bool = False

    def __post_init__(self) -> None:
        if self.high < self.low or self.step <= 0:
            raise ValueError("integer parameter bounds or step are invalid")
        if self.log and self.low <= 0:
            raise ValueError("log integer parameter low must be positive")
        if self.log and self.step != 1:
            raise ValueError("log integer parameter requires step=1")

    def validate(self, value: object) -> int:
        if isinstance(value, bool):
            raise TypeError("integer parameter cannot be bool")
        result = int(value)
        if result != value and not isinstance(value, str):
            raise ValueError("integer parameter must be integral")
        if not self.low <= result <= self.high or (result - self.low) % self.step:
            raise ValueError("integer parameter outside bounds or off step")
        return result

    def grid_values(self) -> tuple[int, ...]:
        return tuple(range(self.low, self.high + 1, self.step))

    def sample(self, rng: random.Random) -> int:
        values = self.grid_values()
        if not self.log:
            return values[rng.randrange(len(values))]
        raw = round(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        closest = min(values, key=lambda item: abs(item - raw))
        return closest


@dataclass(frozen=True, slots=True)
class CategoricalParameter:
    choices: tuple[object, ...]

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("categorical parameter requires choices")
        encoded = [canonical_json(item) for item in self.choices]
        if len(set(encoded)) != len(encoded):
            raise ValueError("categorical choices must be unique")

    def validate(self, value: object) -> object:
        if canonical_json(value) not in {canonical_json(item) for item in self.choices}:
            raise ValueError("value is not an allowed categorical choice")
        return value

    def grid_values(self) -> tuple[object, ...]:
        return self.choices

    def sample(self, rng: random.Random) -> object:
        return self.choices[rng.randrange(len(self.choices))]


@dataclass(frozen=True, slots=True)
class BoolParameter(CategoricalParameter):
    choices: tuple[object, ...] = (False, True)


ParameterSpace = Mapping[str, ParameterSpec]


def _candidate(parameters: Mapping[str, object], ordinal: int) -> Candidate:
    digest = sha256(canonical_json(parameters)).hexdigest()[:16]
    return Candidate(
        candidate_id=f"candidate-{digest}",
        parameters=dict(parameters),
        ordinal=ordinal,
    )


def grid_candidates(
    space: ParameterSpace,
    *,
    max_candidates: int = 100_000,
) -> tuple[Candidate, ...]:
    if not space:
        return (_candidate({}, 0),)
    names = tuple(sorted(space))
    values = tuple(space[name].grid_values() for name in names)
    total = math.prod(len(item) for item in values)
    if total > max_candidates:
        raise ValueError(f"grid contains {total} candidates; limit={max_candidates}")
    return tuple(
        _candidate(dict(zip(names, combination, strict=True)), ordinal)
        for ordinal, combination in enumerate(product(*values))
    )


def random_candidates(
    space: ParameterSpace,
    *,
    count: int,
    seed: int,
    max_attempt_factor: int = 100,
) -> tuple[Candidate, ...]:
    if count < 1:
        raise ValueError("random candidate count must be positive")
    rng = random.Random(seed)  # noqa: S311 - deterministic research sampling; not cryptographic
    names = tuple(sorted(space))
    seen: set[bytes] = set()
    output: list[Candidate] = []
    attempts = 0
    while len(output) < count and attempts < count * max_attempt_factor:
        attempts += 1
        params = {name: space[name].sample(rng) for name in names}
        key = canonical_json(params)
        if key in seen:
            continue
        seen.add(key)
        output.append(_candidate(params, len(output)))
    if len(output) < count:
        raise ValueError(
            "parameter space produced only "
            f"{len(output)} unique random candidates; requested={count}"
        )
    return tuple(output)


def validate_parameters(space: ParameterSpace, values: Mapping[str, object]) -> dict[str, object]:
    missing = sorted(set(space) - set(values))
    extra = sorted(set(values) - set(space))
    if missing or extra:
        raise ValueError(f"parameter keys mismatch; missing={missing}; extra={extra}")
    return {name: space[name].validate(values[name]) for name in sorted(space)}
