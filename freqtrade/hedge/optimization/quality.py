"""Optimization preflight, resume, and result quality gates."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

from freqtrade.hedge.optimization.fingerprint import canonical_json, parameter_fingerprint
from freqtrade.hedge.optimization.splits import WalkForwardSpec
from freqtrade.hedge.optimization.stress import StressScenario
from freqtrade.hedge.optimization.types import (
    ConstraintSpec,
    ObjectiveSpec,
    OptimizationResult,
    ParameterKind,
    ParameterSpec,
    TrialRecord,
    TrialStatus,
)


_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_SEGMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SENSITIVE_TOKENS = frozenset(
    {"api_key", "apikey", "secret", "password", "token", "credential", "private_key"}
)
_ZERO = Decimal(0)
DEFAULT_MAX_OBJECTIVE_WEIGHT_TOTAL = Decimal(1000000)
DEFAULT_MAX_STRESS_MULTIPLIER = Decimal(100)
DEFAULT_MAX_STRESS_SLIPPAGE_BPS = Decimal(10000)


def _finite_decimal(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def _canonical_choice(value: object) -> bytes:
    return canonical_json(value)


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


def ensure_unique_parameter_names(specs: Sequence[ParameterSpec]) -> None:
    names = [item.name for item in specs]
    if len(names) != len(set(names)):
        raise ValueError("optimization parameter names must be unique")


def ensure_unique_parameter_paths(specs: Sequence[ParameterSpec]) -> None:
    paths = [item.path for item in specs]
    if len(paths) != len(set(paths)):
        raise ValueError("optimization parameter paths must be unique")


def ensure_unique_objective_metrics(specs: Sequence[ObjectiveSpec]) -> None:
    metrics = [item.metric for item in specs]
    if len(metrics) != len(set(metrics)):
        raise ValueError("optimization objective metrics must be unique")


def ensure_unique_constraint_metrics(specs: Sequence[ConstraintSpec]) -> None:
    metrics = [item.metric for item in specs]
    if len(metrics) != len(set(metrics)):
        raise ValueError("optimization constraint metrics must be unique")


def ensure_unique_stress_names(scenarios: Sequence[StressScenario]) -> None:
    names = [item.name for item in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("stress scenario names must be unique")


def validate_study_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("study name must be a string")
    if not _SAFE_NAME.fullmatch(name.strip()):
        raise ValueError("study name must be 1-128 safe filename characters")


def validate_worker_budget(*, workers: int, trials: int) -> None:
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or isinstance(trials, bool)
        or not isinstance(trials, int)
    ):
        raise TypeError("workers and trials must be integers")
    if workers < 1 or trials < 1:
        raise ValueError("workers and trials must be positive")
    if workers > trials:
        raise ValueError("workers cannot exceed trial count")


def validate_failure_budget(*, max_failures: int, trials: int, fail_fast: bool) -> None:
    if (
        isinstance(max_failures, bool)
        or not isinstance(max_failures, int)
        or isinstance(trials, bool)
        or not isinstance(trials, int)
        or not isinstance(fail_fast, bool)
    ):
        raise TypeError("failure budget inputs have invalid types")
    if max_failures < 0 or trials < 1:
        raise ValueError("failure budget is invalid")
    # Budgets larger than the trial count are harmless (effectively unlimited).
    # fail_fast intentionally dominates max_failures and therefore does not conflict with it.


def validate_grid_enumerability(specs: Sequence[ParameterSpec]) -> None:
    for spec in specs:
        if spec.kind is ParameterKind.DECIMAL and spec.step is None and spec.low != spec.high:
            raise ValueError(f"grid sampler requires an enumerable decimal step for {spec.name}")


def grid_cardinality(specs: Sequence[ParameterSpec]) -> int:
    total = 1
    for spec in specs:
        if spec.kind is ParameterKind.BOOLEAN:
            width = 2
        elif spec.kind is ParameterKind.CATEGORICAL:
            width = len(spec.choices)
        elif spec.kind is ParameterKind.INTEGER:
            low, high = _integer_bounds(spec)
            step = _integer_step(spec)
            width = ((high - low) // step) + 1
        else:
            low, high = _decimal_bounds(spec)
            if spec.step is None:
                if low != high:
                    raise ValueError(f"decimal parameter {spec.name} is not enumerable")
                width = 1
            else:
                step = _decimal_step(spec)
                width = int((high - low) / step) + 1
        total *= width
    return total


def validate_grid_cardinality(specs: Sequence[ParameterSpec], *, maximum: int) -> int:
    if maximum < 1:
        raise ValueError("grid candidate maximum must be positive")
    total = grid_cardinality(specs)
    if total > maximum:
        raise ValueError(f"grid contains {total} candidates; limit={maximum}")
    return total


def validate_categorical_choices(specs: Sequence[ParameterSpec]) -> None:
    for spec in specs:
        if spec.kind is not ParameterKind.CATEGORICAL:
            continue
        encoded = [_canonical_choice(item) for item in spec.choices]
        if len(encoded) != len(set(encoded)):
            raise ValueError(f"categorical choices for {spec.name} are not canonically unique")
        for item in spec.choices:
            if isinstance(item, (Mapping, list, set, frozenset)):
                raise TypeError(
                    f"categorical choice for {spec.name} must be an immutable scalar/tuple"
                )


def validate_parameter_segments(specs: Sequence[ParameterSpec]) -> None:
    for spec in specs:
        parts = spec.path.split(".")
        if any(not _SAFE_SEGMENT.fullmatch(part) for part in parts):
            raise ValueError(f"parameter path contains an unsafe segment: {spec.path}")


def validate_parameter_names(specs: Sequence[ParameterSpec]) -> None:
    for spec in specs:
        if not _SAFE_SEGMENT.fullmatch(spec.name):
            raise ValueError(f"parameter name is unsafe: {spec.name!r}")


def validate_no_sensitive_parameter_tokens(specs: Sequence[ParameterSpec]) -> None:
    for spec in specs:
        haystack = f"{spec.name}.{spec.path}".lower()
        if any(token in haystack for token in _SENSITIVE_TOKENS):
            raise ValueError(f"sensitive configuration surface cannot be optimized: {spec.path}")


def validate_numeric_variation(specs: Sequence[ParameterSpec]) -> None:
    for spec in specs:
        if spec.kind in {ParameterKind.DECIMAL, ParameterKind.INTEGER} and spec.low == spec.high:
            raise ValueError(f"numeric optimization parameter {spec.name} has no variation")


def validate_step_alignment(specs: Sequence[ParameterSpec]) -> None:
    for spec in specs:
        if spec.kind is ParameterKind.DECIMAL and spec.step is not None:
            low, high = _decimal_bounds(spec)
            step = _decimal_step(spec)
            quotient = (high - low) / step
            if quotient != quotient.to_integral_value():
                raise ValueError(f"decimal span for {spec.name} is not divisible by its step")
        elif spec.kind is ParameterKind.INTEGER and spec.step is not None:
            low, high = _integer_bounds(spec)
            step = _integer_step(spec)
            if (high - low) % step:
                raise ValueError(f"integer span for {spec.name} is not divisible by its step")


def validate_decimal_precision(specs: Sequence[ParameterSpec], *, maximum_places: int = 18) -> None:
    if maximum_places < 0:
        raise ValueError("maximum decimal places cannot be negative")
    for spec in specs:
        if spec.kind is not ParameterKind.DECIMAL:
            continue
        for name, value in (("low", spec.low), ("high", spec.high), ("step", spec.step)):
            if value is None:
                continue
            if not isinstance(value, Decimal):
                raise TypeError(f"decimal parameter {spec.name}.{name} is not Decimal")
            places = max(0, -value.as_tuple().exponent)
            if places > maximum_places:
                raise ValueError(f"{spec.name}.{name} exceeds {maximum_places} decimal places")


def validate_objective_weights(
    specs: Sequence[ObjectiveSpec],
    *,
    maximum_total: Decimal = DEFAULT_MAX_OBJECTIVE_WEIGHT_TOTAL,
) -> None:
    total = sum((item.weight for item in specs), _ZERO)
    if not total.is_finite() or total <= _ZERO or total > maximum_total:
        raise ValueError("objective weight total is outside the safe range")


def validate_constraint_consistency(specs: Sequence[ConstraintSpec]) -> None:
    for spec in specs:
        if spec.minimum is not None and spec.maximum is not None and spec.minimum > spec.maximum:
            raise ValueError(f"constraint bounds are inconsistent for {spec.metric}")


def validate_stress_bounds(
    scenarios: Sequence[StressScenario],
    *,
    maximum_multiplier: Decimal = DEFAULT_MAX_STRESS_MULTIPLIER,
    maximum_slippage_bps: Decimal = DEFAULT_MAX_STRESS_SLIPPAGE_BPS,
) -> None:
    if maximum_multiplier <= 0 or maximum_slippage_bps < 0:
        raise ValueError("stress validation bounds are invalid")
    for scenario in scenarios:
        multipliers = (
            scenario.maker_fee_multiplier,
            scenario.taker_fee_multiplier,
            scenario.volume_participation_multiplier,
            scenario.funding_rate_multiplier,
        )
        if any(item > maximum_multiplier for item in multipliers):
            raise ValueError(f"stress scenario {scenario.name} exceeds multiplier safety bound")
        if scenario.slippage_bps_add > maximum_slippage_bps:
            raise ValueError(f"stress scenario {scenario.name} exceeds slippage safety bound")


def validate_walk_forward_capacity(
    spec: WalkForwardSpec | None,
    *,
    dataset_size: int | None,
) -> None:
    if spec is None:
        return
    if dataset_size is None or dataset_size <= 0:
        raise ValueError("walk-forward requires a positive dataset size")
    required = spec.train_size + spec.validation_size + spec.test_size + (2 * spec.embargo_size)
    if dataset_size < required:
        raise ValueError(
            f"dataset has {dataset_size} items; first walk-forward window requires {required}"
        )


def validate_output_storage_paths(*, output_directory: Path, storage_path: Path) -> None:
    output = Path(output_directory).expanduser()
    storage = Path(storage_path).expanduser()
    if output.exists() and not output.is_dir():
        raise ValueError("optimization output_directory exists but is not a directory")
    if storage.exists() and storage.is_dir():
        raise ValueError("optimization storage_path points to a directory")
    if output.resolve() == storage.resolve():
        raise ValueError("optimization output directory and storage file cannot be the same path")


def validate_dataset_fingerprint(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("dataset fingerprint must be a string")
    if not _HEX64.fullmatch(value):
        raise ValueError("dataset fingerprint must be a 64-character SHA-256 hex string")


def validate_dataset_size(value: int | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("dataset_size must be an integer when supplied")
    if value <= 0:
        raise ValueError("dataset_size must be a positive integer when supplied")


def validate_timestamp_count(
    timestamps: Sequence[datetime] | None,
    *,
    dataset_size: int | None,
) -> None:
    if timestamps is not None and dataset_size is not None and len(timestamps) != dataset_size:
        raise ValueError("timestamp count must equal dataset_size")


def validate_timestamp_awareness(timestamps: Sequence[datetime] | None) -> None:
    if timestamps is not None and any(
        item.tzinfo is None or item.utcoffset() is None for item in timestamps
    ):
        raise ValueError("optimization timestamps must all be timezone-aware")


def validate_timestamp_order(timestamps: Sequence[datetime] | None) -> None:
    if timestamps is None:
        return
    if any(right <= left for left, right in pairwise(timestamps)):
        raise ValueError("optimization timestamps must be strictly increasing")


def stable_trial_seed(seed: int, trial_id: int) -> int:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or isinstance(trial_id, bool)
        or not isinstance(trial_id, int)
    ):
        raise TypeError("seed and trial_id must be integers")
    if seed < 0 or trial_id < 0:
        raise ValueError("seed and trial_id must be non-negative")
    payload = f"hedge-bt100-seed-v1:{seed}:{trial_id}".encode("ascii")
    return int.from_bytes(sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def validate_report_mapping(report: object) -> Mapping[str, object]:
    if not isinstance(report, Mapping):
        raise TypeError("trial evaluator must return a mapping")
    if not report:
        raise ValueError("trial evaluator returned an empty report")
    return report


def validate_report_finite(report: Mapping[str, object]) -> None:
    for key, value in report.items():
        if isinstance(value, bool) or isinstance(value, str) or value is None:
            continue
        try:
            number = Decimal(str(value))
        except Exception as exc:
            raise ValueError(f"report metric {key} is not a supported scalar") from exc
        if not number.is_finite():
            raise ValueError(f"report metric {key} must be finite")


def validate_trial_record_core(record: TrialRecord) -> None:
    if record.trial_id < 0:
        raise ValueError("trial_id cannot be negative")
    if not _HEX64.fullmatch(record.parameter_hash):
        raise ValueError("trial parameter_hash must be SHA-256 hex")
    if record.duration_seconds < 0 or not record.duration_seconds.is_finite():
        raise ValueError("trial duration must be finite and non-negative")
    if record.status is TrialStatus.FAILED and not record.error:
        raise ValueError("failed trial must record an error")
    if record.status is TrialStatus.COMPLETE and record.scalar_score is None:
        raise ValueError("completed trial must have a scalar score")


def validate_trial_parameter_hash(record: TrialRecord) -> None:
    expected = parameter_fingerprint(record.parameters)
    if record.parameter_hash != expected:
        raise ValueError("trial parameter payload does not match parameter_hash")


def validate_trial_dataset(record: TrialRecord, *, expected_fingerprint: str) -> None:
    validate_dataset_fingerprint(expected_fingerprint)
    if record.dataset_fingerprint != expected_fingerprint:
        raise ValueError("trial dataset fingerprint differs from current study")


def validate_trial_objective_width(record: TrialRecord, *, expected: int) -> None:
    if expected < 1:
        raise ValueError("expected objective width must be positive")
    if record.status is TrialStatus.COMPLETE and len(record.objective_values) != expected:
        raise ValueError("completed trial objective vector has the wrong width")


def validate_resume_records(
    records: Iterable[TrialRecord],
    *,
    dataset_fingerprint: str,
    objective_width: int,
) -> None:
    validate_dataset_fingerprint(dataset_fingerprint)
    seen_ids: set[int] = set()
    seen_hashes: set[str] = set()
    for record in records:
        validate_trial_record_core(record)
        validate_trial_parameter_hash(record)
        validate_trial_dataset(record, expected_fingerprint=dataset_fingerprint)
        validate_trial_objective_width(record, expected=objective_width)
        if record.trial_id in seen_ids or record.parameter_hash in seen_hashes:
            raise ValueError("resume records contain duplicate trial identity")
        seen_ids.add(record.trial_id)
        seen_hashes.add(record.parameter_hash)


def status_counts(records: Iterable[TrialRecord]) -> Mapping[str, int]:
    counts = Counter(item.status.value for item in records)
    return dict(sorted(counts.items()))


def validate_best_trial(result: OptimizationResult) -> None:
    by_id = {item.trial_id: item for item in result.trials}
    if result.best_trial_id is None:
        if any(item.status is TrialStatus.COMPLETE for item in result.trials):
            raise ValueError("optimization result omitted best_trial_id despite completed trials")
        return
    trial = by_id.get(result.best_trial_id)
    if trial is None or trial.status is not TrialStatus.COMPLETE:
        raise ValueError("best_trial_id must reference a completed trial")


def validate_pareto_ids(result: OptimizationResult) -> None:
    if len(result.pareto_trial_ids) != len(set(result.pareto_trial_ids)):
        raise ValueError("Pareto trial ids must be unique")
    by_id = {item.trial_id: item for item in result.trials}
    for trial_id in result.pareto_trial_ids:
        item = by_id.get(trial_id)
        if item is None or item.status is not TrialStatus.COMPLETE:
            raise ValueError("Pareto ids must reference completed trials")


def validate_optimization_result(result: OptimizationResult) -> None:
    validate_dataset_fingerprint(result.dataset_fingerprint)
    if not _HEX64.fullmatch(result.study_fingerprint):
        raise ValueError("study fingerprint must be SHA-256 hex")
    if len({item.trial_id for item in result.trials}) != len(result.trials):
        raise ValueError("optimization result contains duplicate trial ids")
    if result.resumed_trials < 0 or result.resumed_trials > len(result.trials):
        raise ValueError("resumed_trials is outside valid range")
    validate_best_trial(result)
    validate_pareto_ids(result)


def validate_optimization_definition(config: object) -> None:
    """Apply the non-controversial optimization preflight checks to a parsed config."""
    specs = config.parameters
    objectives = config.objectives
    constraints = config.constraints
    scenarios = config.stress_scenarios
    ensure_unique_parameter_names(specs)
    ensure_unique_parameter_paths(specs)
    ensure_unique_objective_metrics(objectives)
    ensure_unique_constraint_metrics(constraints)
    ensure_unique_stress_names(scenarios)
    validate_study_name(config.study_name)
    validate_worker_budget(workers=config.workers, trials=config.trials)
    validate_categorical_choices(specs)
    validate_parameter_segments(specs)
    validate_parameter_names(specs)
    validate_no_sensitive_parameter_tokens(specs)
    validate_step_alignment(specs)
    validate_objective_weights(objectives)
    validate_constraint_consistency(constraints)
    validate_output_storage_paths(
        output_directory=config.output_directory,
        storage_path=config.storage_path,
    )
    if config.sampler == "grid":
        validate_grid_enumerability(specs)
        validate_grid_cardinality(specs, maximum=config.max_grid_candidates)
