"""Advanced deterministic sampling and candidate validation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from decimal import Decimal
from hashlib import sha256
from random import Random

from freqtrade.hedge.optimization.fingerprint import canonical_json
from freqtrade.hedge.optimization.types import ParameterKind, ParameterSpec


def conditional_parameter_active(
    candidate: Mapping[str, object],
    *,
    parent: str,
    allowed: Sequence[object],
) -> bool:
    if parent not in candidate:
        raise ValueError(f"missing conditional parent {parent}")
    return candidate[parent] in tuple(allowed)


def canonical_candidate(candidate: Mapping[str, object]) -> dict[str, object]:
    if not candidate:
        raise ValueError("candidate cannot be empty")
    return {str(key): candidate[key] for key in sorted(candidate)}


def candidate_hash(candidate: Mapping[str, object]) -> str:
    return sha256(canonical_json(canonical_candidate(candidate))).hexdigest()


def validate_forbidden_combinations(
    candidate: Mapping[str, object],
    forbidden: Sequence[Mapping[str, object]],
) -> None:
    for rule in forbidden:
        if rule and all(candidate.get(key) == value for key, value in rule.items()):
            raise ValueError("candidate matches a forbidden parameter combination")


def validate_monotonic_constraints(
    candidate: Mapping[str, object],
    constraints: Sequence[tuple[str, str, str]],
) -> None:
    for left_name, operator, right_name in constraints:
        if left_name not in candidate or right_name not in candidate:
            raise ValueError("monotonic constraint references a missing parameter")
        left = Decimal(str(candidate[left_name]))
        right = Decimal(str(candidate[right_name]))
        valid = {
            "<": left < right,
            "<=": left <= right,
            ">": left > right,
            ">=": left >= right,
        }.get(operator)
        if valid is None:
            raise ValueError(f"unsupported monotonic operator {operator}")
        if not valid:
            raise ValueError(f"monotonic constraint failed: {left_name} {operator} {right_name}")


def _numeric_bounds(spec: ParameterSpec) -> tuple[Decimal, Decimal]:
    if spec.kind not in (ParameterKind.DECIMAL, ParameterKind.INTEGER):
        raise ValueError("advanced numeric sampler requires numeric parameters")
    return Decimal(str(spec.low)), Decimal(str(spec.high))


def latin_hypercube(
    specs: Sequence[ParameterSpec],
    *,
    count: int,
    seed: int,
) -> tuple[dict[str, object], ...]:
    if count <= 0 or not specs:
        raise ValueError("count and specs must be positive/non-empty")
    rng = Random(seed)  # noqa: S311 - deterministic research sampling; not cryptographic
    dimensions: list[list[Decimal]] = []
    for spec in specs:
        low, high = _numeric_bounds(spec)
        if spec.kind is ParameterKind.INTEGER:
            cardinality = int(high - low) + 1
            if count > cardinality:
                raise ValueError(
                    f"Latin hypercube count exceeds integer cardinality for {spec.name}"
                )
            samples = [
                low + Decimal(int(((index + rng.random()) / count) * cardinality))
                for index in range(count)
            ]
        else:
            samples = [
                low + (high - low) * Decimal(str((index + rng.random()) / count))
                for index in range(count)
            ]
        rng.shuffle(samples)
        dimensions.append(samples)
    output = []
    for index in range(count):
        candidate: dict[str, object] = {}
        for spec, samples in zip(specs, dimensions, strict=True):
            value = samples[index]
            candidate[spec.name] = int(value) if spec.kind is ParameterKind.INTEGER else value
        output.append(candidate)
    return tuple(output)


def _halton(index: int, base: int) -> Decimal:
    result = Decimal(0)
    fraction = Decimal(1)
    current = index
    while current > 0:
        fraction /= Decimal(base)
        result += fraction * Decimal(current % base)
        current //= base
    return result


def halton_samples(
    specs: Sequence[ParameterSpec],
    *,
    count: int,
    start_index: int = 1,
) -> tuple[dict[str, object], ...]:
    primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    if count <= 0 or start_index <= 0 or not specs or len(specs) > len(primes):
        raise ValueError("invalid Halton sampler request")
    output = []
    for offset in range(count):
        candidate: dict[str, object] = {}
        for spec, base in zip(specs, primes, strict=False):
            low, high = _numeric_bounds(spec)
            value = low + (high - low) * _halton(start_index + offset, base)
            candidate[spec.name] = int(value) if spec.kind is ParameterKind.INTEGER else value
        output.append(candidate)
    return tuple(output)


def neighborhood_candidates(
    center: Mapping[str, object],
    specs: Sequence[ParameterSpec],
    *,
    radius_steps: int = 1,
) -> tuple[dict[str, object], ...]:
    if radius_steps <= 0:
        raise ValueError("radius_steps must be positive")
    output: list[dict[str, object]] = []
    for spec in specs:
        if spec.name not in center or spec.kind not in (
            ParameterKind.DECIMAL,
            ParameterKind.INTEGER,
        ):
            continue
        step = (
            spec.step
            if spec.step is not None
            else (1 if spec.kind is ParameterKind.INTEGER else None)
        )
        if step is None:
            continue
        for direction in (-radius_steps, radius_steps):
            candidate = dict(center)
            value = Decimal(str(center[spec.name])) + Decimal(str(step)) * direction
            low, high = _numeric_bounds(spec)
            if low <= value <= high:
                candidate[spec.name] = int(value) if spec.kind is ParameterKind.INTEGER else value
                output.append(candidate)
    return deduplicate_candidates(output)


def parameter_distance(
    left: Mapping[str, object],
    right: Mapping[str, object],
    specs: Sequence[ParameterSpec],
) -> Decimal:
    total = Decimal(0)
    for spec in specs:
        if spec.name not in left or spec.name not in right:
            raise ValueError("distance candidates do not cover the parameter space")
        if spec.kind in (ParameterKind.DECIMAL, ParameterKind.INTEGER):
            low, high = _numeric_bounds(spec)
            span = high - low
            delta = abs(Decimal(str(left[spec.name])) - Decimal(str(right[spec.name])))
            total += delta / span if span else Decimal(0)
        else:
            total += Decimal(left[spec.name] != right[spec.name])
    return total


def deduplicate_candidates(
    candidates: Iterable[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    seen: set[str] = set()
    output: list[dict[str, object]] = []
    for candidate in candidates:
        normalized = canonical_candidate(candidate)
        fingerprint = candidate_hash(normalized)
        if fingerprint not in seen:
            seen.add(fingerprint)
            output.append(normalized)
    return tuple(output)
