"""Deterministic parameter-space enumeration and sampling."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from decimal import Decimal
from itertools import product
from math import exp, log
from random import Random

from freqtrade.hedge.optimization.types import ParameterKind, ParameterSpec


def _integer_bounds(spec: ParameterSpec) -> tuple[int, int]:
    if not isinstance(spec.low, int) or isinstance(spec.low, bool):
        raise TypeError(f"integer parameter {spec.name} has a non-integer low bound")
    if not isinstance(spec.high, int) or isinstance(spec.high, bool):
        raise TypeError(f"integer parameter {spec.name} has a non-integer high bound")
    return spec.low, spec.high


def _integer_step(spec: ParameterSpec) -> int:
    step = 1 if spec.step is None else spec.step
    if not isinstance(step, int) or isinstance(step, bool):
        raise TypeError(f"integer parameter {spec.name} has a non-integer step")
    return step


def _decimal_bounds(spec: ParameterSpec) -> tuple[Decimal, Decimal]:
    if not isinstance(spec.low, Decimal) or not isinstance(spec.high, Decimal):
        raise TypeError(f"decimal parameter {spec.name} has non-decimal bounds")
    return spec.low, spec.high


def _decimal_step(spec: ParameterSpec) -> Decimal:
    if not isinstance(spec.step, Decimal):
        raise TypeError(f"decimal parameter {spec.name} has a non-decimal step")
    return spec.step


def _decimal_grid(spec: ParameterSpec) -> tuple[Decimal, ...]:
    low, high = _decimal_bounds(spec)
    if spec.step is None:
        if spec.low == spec.high:
            return (spec.low,)
        raise ValueError(f"grid search requires a step for decimal parameter {spec.name}")
    step = _decimal_step(spec)
    values: list[Decimal] = []
    current = low
    while current <= high:
        values.append(current)
        current += step
    if not values or values[-1] != high:
        raise ValueError(
            f"decimal range for {spec.name} is not exactly divisible by step {step}"
        )
    return tuple(values)


def _integer_grid(spec: ParameterSpec) -> tuple[int, ...]:
    low, high = _integer_bounds(spec)
    step = _integer_step(spec)
    values = tuple(range(low, high + 1, step))
    if not values or values[-1] != high:
        raise ValueError(
            f"integer range for {spec.name} is not exactly divisible by step {step}"
        )
    return values


def values_for_grid(spec: ParameterSpec) -> tuple[object, ...]:
    if spec.kind is ParameterKind.DECIMAL:
        return _decimal_grid(spec)
    if spec.kind is ParameterKind.INTEGER:
        return _integer_grid(spec)
    if spec.kind is ParameterKind.BOOLEAN:
        return (False, True)
    return spec.choices


class ParameterSpace:
    """Validated collection of uniquely named and uniquely mapped parameters."""

    def __init__(self, specs: Sequence[ParameterSpec]) -> None:
        self.specs = tuple(specs)
        names = [item.name for item in self.specs]
        paths = [item.path for item in self.specs]
        if not self.specs:
            raise ValueError("parameter space cannot be empty")
        if len(names) != len(set(names)):
            raise ValueError("parameter names must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("parameter paths must be unique")

    def grid_size(self) -> int:
        size = 1
        for spec in self.specs:
            size *= len(values_for_grid(spec))
        return size

    def iter_grid(self, *, max_candidates: int = 100_000) -> Iterator[dict[str, object]]:
        size = self.grid_size()
        if size > max_candidates:
            raise ValueError(
                f"grid contains {size} candidates, exceeding max_candidates={max_candidates}"
            )
        names = tuple(spec.name for spec in self.specs)
        dimensions = tuple(values_for_grid(spec) for spec in self.specs)
        for candidate in product(*dimensions):
            yield dict(zip(names, candidate, strict=True))

    def sample_random(self, count: int, *, seed: int) -> tuple[dict[str, object], ...]:
        if count <= 0:
            raise ValueError("random sample count must be positive")
        rng = Random(seed)  # noqa: S311 - deterministic research sampling; not cryptographic
        seen: set[tuple[tuple[str, object], ...]] = set()
        output: list[dict[str, object]] = []
        attempts = 0
        max_attempts = max(100, count * 100)
        while len(output) < count and attempts < max_attempts:
            attempts += 1
            candidate = {spec.name: self._sample_one(spec, rng) for spec in self.specs}
            key = tuple((spec.name, candidate[spec.name]) for spec in self.specs)
            if key in seen:
                continue
            seen.add(key)
            output.append(candidate)
        if len(output) != count:
            raise ValueError(
                f"parameter space yielded only {len(output)} unique samples; requested {count}"
            )
        return tuple(output)

    @staticmethod
    def _sample_one(spec: ParameterSpec, rng: Random) -> object:
        if spec.kind is ParameterKind.CATEGORICAL:
            return spec.choices[rng.randrange(len(spec.choices))]
        if spec.kind is ParameterKind.BOOLEAN:
            return bool(rng.getrandbits(1))
        if spec.kind is ParameterKind.INTEGER:
            values = _integer_grid(spec) if spec.step is not None else None
            if values is not None:
                return values[rng.randrange(len(values))]
            low, high = _integer_bounds(spec)
            return rng.randint(low, high)
        low, high = _decimal_bounds(spec)
        if spec.step is not None:
            values = _decimal_grid(spec)
            return values[rng.randrange(len(values))]
        if low == high:
            return low
        unit = Decimal(str(rng.random()))
        if spec.log:
            sampled_float = exp(
                log(float(low)) + float(unit) * (log(float(high)) - log(float(low)))
            )
            sampled = Decimal(str(sampled_float))
        else:
            sampled = low + (high - low) * unit
        exponent = min(low.as_tuple().exponent, high.as_tuple().exponent, -12)
        quantum = Decimal(1).scaleb(exponent)
        return sampled.quantize(quantum)

    def validate_candidate(self, candidate: Mapping[str, object]) -> None:
        expected = {item.name for item in self.specs}
        if set(candidate) != expected:
            raise ValueError("candidate keys do not exactly match parameter space")
        # Reuse deterministic grid/range semantics without mutating configuration.
        from freqtrade.hedge.optimization.config_patch import apply_parameters

        apply_parameters({}, self.specs, candidate)

    def candidates(
        self,
        *,
        sampler: str,
        trials: int,
        seed: int,
        max_grid_candidates: int = 100_000,
    ) -> tuple[dict[str, object], ...]:
        normalized = sampler.strip().lower()
        if normalized == "grid":
            values = tuple(self.iter_grid(max_candidates=max_grid_candidates))
            if trials > 0:
                values = values[:trials]
            return values
        if normalized == "random":
            return self.sample_random(trials, seed=seed)
        raise ValueError("sampler must be 'grid' or 'random'")
