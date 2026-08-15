"""Hedge lookahead and recursive-analysis evidence tools."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from json import dumps
from typing import Any


_VOLATILE_KEYS = {
    "generated_at", "run_id", "elapsed", "elapsed_seconds", "process_id", "hostname"
}


def _canonical(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _VOLATILE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    raw = getattr(value, "value", None)
    if raw is not None:
        return raw
    iso = getattr(value, "isoformat", None)
    if callable(iso):
        return iso()
    return value


def evidence_hash(value: Any) -> str:
    canonical = dumps(_canonical(value), sort_keys=True, separators=(",", ":"), default=str)
    return sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class HedgeAnalysisDifference:
    path: str
    baseline: Any
    candidate: Any
    category: str


def deep_differences(
    baseline: Any,
    candidate: Any,
    *,
    path: str = "$",
    tolerance: Decimal = Decimal("0"),
) -> tuple[HedgeAnalysisDifference, ...]:
    """Return deterministic structural and numeric differences."""

    left = _canonical(baseline)
    right = _canonical(candidate)
    rows: list[HedgeAnalysisDifference] = []
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        keys = sorted(set(left) | set(right))
        for key in keys:
            child = f"{path}.{key}"
            if key not in left:
                rows.append(HedgeAnalysisDifference(child, None, right[key], "ADDED"))
            elif key not in right:
                rows.append(HedgeAnalysisDifference(child, left[key], None, "REMOVED"))
            else:
                rows.extend(deep_differences(left[key], right[key], path=child, tolerance=tolerance))
        return tuple(rows)
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)) and isinstance(right, Sequence) and not isinstance(right, (str, bytes)):
        for index in range(max(len(left), len(right))):
            child = f"{path}[{index}]"
            if index >= len(left):
                rows.append(HedgeAnalysisDifference(child, None, right[index], "ADDED"))
            elif index >= len(right):
                rows.append(HedgeAnalysisDifference(child, left[index], None, "REMOVED"))
            else:
                rows.extend(deep_differences(left[index], right[index], path=child, tolerance=tolerance))
        return tuple(rows)
    try:
        left_decimal = Decimal(str(left))
        right_decimal = Decimal(str(right))
    except Exception:
        if left != right:
            rows.append(HedgeAnalysisDifference(path, left, right, "VALUE"))
    else:
        if abs(left_decimal - right_decimal) > tolerance:
            rows.append(HedgeAnalysisDifference(path, left, right, "NUMERIC"))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class HedgeLookaheadProbe:
    cutoff: int
    baseline_hash: str
    truncated_hash: str
    differences: tuple[HedgeAnalysisDifference, ...]

    @property
    def leaked(self) -> bool:
        return bool(self.differences)


@dataclass(frozen=True, slots=True)
class HedgeLookaheadReport:
    probes: tuple[HedgeLookaheadProbe, ...]
    compared_fields: tuple[str, ...]
    schema: str = "hedge-lookahead-analysis-v1"

    @property
    def passed(self) -> bool:
        return all(not probe.leaked for probe in self.probes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "passed": self.passed,
            "compared_fields": list(self.compared_fields),
            "probes": [
                {
                    "cutoff": probe.cutoff,
                    "baseline_hash": probe.baseline_hash,
                    "truncated_hash": probe.truncated_hash,
                    "leaked": probe.leaked,
                    "differences": [
                        {
                            "path": item.path,
                            "baseline": item.baseline,
                            "candidate": item.candidate,
                            "category": item.category,
                        }
                        for item in probe.differences
                    ],
                }
                for probe in self.probes
            ],
        }


class HedgeLookaheadAnalyzer:
    """Compare prefix outputs from a full run and independently truncated runs."""

    def __init__(self, *, fields: Iterable[str] = ("signals", "orders", "events", "snapshots")) -> None:
        self.fields = tuple(fields)

    @staticmethod
    def _prefix(payload: Mapping[str, Any], cutoff: int, fields: Sequence[str]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field in fields:
            value = payload.get(field, ())
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                result[field] = list(value[:cutoff])
            else:
                result[field] = value
        return result

    def analyze(
        self,
        baseline: Mapping[str, Any],
        truncated_runs: Mapping[int, Mapping[str, Any]],
        *,
        tolerance: Decimal = Decimal("0"),
    ) -> HedgeLookaheadReport:
        probes: list[HedgeLookaheadProbe] = []
        for cutoff, candidate in sorted(truncated_runs.items()):
            if cutoff <= 0:
                raise ValueError("lookahead cutoff must be positive")
            expected = self._prefix(baseline, cutoff, self.fields)
            actual = self._prefix(candidate, cutoff, self.fields)
            differences = deep_differences(expected, actual, tolerance=tolerance)
            probes.append(
                HedgeLookaheadProbe(
                    cutoff,
                    evidence_hash(expected),
                    evidence_hash(actual),
                    differences,
                )
            )
        return HedgeLookaheadReport(tuple(probes), self.fields)


@dataclass(frozen=True, slots=True)
class HedgeRecursiveProbe:
    startup_candles: int
    output_hash: str
    differences: tuple[HedgeAnalysisDifference, ...]


@dataclass(frozen=True, slots=True)
class HedgeRecursiveReport:
    reference_startup_candles: int
    probes: tuple[HedgeRecursiveProbe, ...]
    schema: str = "hedge-recursive-analysis-v1"

    @property
    def passed(self) -> bool:
        return all(not item.differences for item in self.probes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "passed": self.passed,
            "reference_startup_candles": self.reference_startup_candles,
            "probes": [
                {
                    "startup_candles": item.startup_candles,
                    "output_hash": item.output_hash,
                    "differences": [
                        {
                            "path": diff.path,
                            "baseline": diff.baseline,
                            "candidate": diff.candidate,
                            "category": diff.category,
                        }
                        for diff in item.differences
                    ],
                }
                for item in self.probes
            ],
        }


class HedgeRecursiveAnalyzer:
    """Detect unstable terminal indicators/signals across startup window lengths."""

    def analyze(
        self,
        outputs: Mapping[int, Mapping[str, Any]],
        *,
        compare_tail: int = 1,
        tolerance: Decimal = Decimal("0"),
    ) -> HedgeRecursiveReport:
        if not outputs:
            raise ValueError("recursive analysis requires at least one run")
        if compare_tail <= 0:
            raise ValueError("compare_tail must be positive")
        reference_size = max(outputs)
        reference = self._tail(outputs[reference_size], compare_tail)
        probes: list[HedgeRecursiveProbe] = []
        for size, payload in sorted(outputs.items()):
            current = self._tail(payload, compare_tail)
            probes.append(
                HedgeRecursiveProbe(
                    size,
                    evidence_hash(current),
                    deep_differences(reference, current, tolerance=tolerance),
                )
            )
        return HedgeRecursiveReport(reference_size, tuple(probes))

    @staticmethod
    def _tail(payload: Mapping[str, Any], count: int) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                result[key] = list(value[-count:])
            else:
                result[key] = value
        return result
