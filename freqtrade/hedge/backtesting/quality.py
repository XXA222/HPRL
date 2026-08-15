"""Backtest dataset, metrics, and result quality gates."""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from decimal import Decimal
from itertools import pairwise

from freqtrade.hedge.simulation.exchange import (
    AccountEvent,
    BarEvent,
    FillEvent,
    FundingEvent,
    SignalEvent,
)

from .contracts import (
    BacktestDataset,
    BacktestEvaluation,
    Candidate,
    EngineConfig,
    ObjectiveConfig,
    OptimizationSummary,
)


_HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")
_TIMEFRAME = re.compile(r"^(?P<count>[1-9][0-9]*)(?P<unit>[smhdwM])$")
_ZERO = Decimal(0)
_ONE = Decimal(1)
DEFAULT_MAX_FUNDING_RATE = Decimal("0.10")
DEFAULT_MAX_FUNDING_MARK_DEVIATION = Decimal("0.25")
DEFAULT_MAX_FEE_RATE = Decimal("0.10")
DEFAULT_MAX_SLIPPAGE_BPS = Decimal(10000)
DEFAULT_MAX_LEVERAGE = Decimal(125)


def _decimal(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def validate_timeframe_label(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("timeframe must be a string")
    if not _TIMEFRAME.fullmatch(value.strip()):
        raise ValueError("timeframe must look like 1m, 5m, 1h, 1d, 1w, or 1M")


def timeframe_to_seconds(value: str) -> int:
    validate_timeframe_label(value)
    match = _TIMEFRAME.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"unsupported timeframe label: {value!r}")
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800, "M": 2592000}
    return int(match.group("count")) * multipliers[match.group("unit")]


def validate_dataset_id(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("dataset_id must be a string")
    if not value.strip() or len(value) > 256:
        raise ValueError("dataset_id must be a non-empty string of at most 256 characters")
    if any(ord(ch) < 32 for ch in value):
        raise ValueError("dataset_id cannot contain control characters")


def validate_symbol_label(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("dataset symbol must be a string")
    if not value.strip() or len(value) > 128 or any(ord(ch) < 32 for ch in value):
        raise ValueError("dataset symbol is invalid")


def validate_metadata_strings(metadata: Mapping[str, str]) -> None:
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("dataset metadata keys and values must be strings")
        if not key.strip() or any(ord(ch) < 32 for ch in key):
            raise ValueError("dataset metadata key is invalid")


def validate_metadata_budget(
    metadata: Mapping[str, str],
    *,
    maximum_entries: int = 64,
    maximum_value_length: int = 4096,
) -> None:
    if len(metadata) > maximum_entries:
        raise ValueError("dataset metadata contains too many entries")
    if any(len(value) > maximum_value_length for value in metadata.values()):
        raise ValueError("dataset metadata value exceeds size budget")


def validate_dataset_fingerprint(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("dataset fingerprint must be a string")
    if not _HEX64.fullmatch(value):
        raise ValueError("dataset fingerprint must be SHA-256 hex")


def validate_dataset_counts(dataset: BacktestDataset) -> None:
    bars = sum(isinstance(item, BarEvent) for item in dataset.events)
    signals = sum(isinstance(item, SignalEvent) for item in dataset.events)
    fundings = sum(isinstance(item, FundingEvent) for item in dataset.events)
    if (bars, signals, fundings) != (
        dataset.bar_count,
        dataset.signal_count,
        dataset.funding_count,
    ):
        raise ValueError("dataset event counts do not match payload")


def validate_dataset_bounds(dataset: BacktestDataset) -> None:
    bars = [item for item in dataset.events if isinstance(item, BarEvent)]
    if not bars or dataset.start != bars[0].timestamp or dataset.end != bars[-1].timestamp:
        raise ValueError("dataset start/end must match first/last bar timestamp")


def validate_duplicate_signals(events: Iterable[object]) -> None:
    seen: set[tuple[str, datetime]] = set()
    for event in events:
        if not isinstance(event, SignalEvent):
            continue
        key = (event.symbol, event.timestamp)
        if key in seen:
            raise ValueError("dataset contains duplicate signal timestamp")
        seen.add(key)


def validate_duplicate_funding(events: Iterable[object]) -> None:
    seen: set[tuple[str, datetime]] = set()
    for event in events:
        if not isinstance(event, FundingEvent):
            continue
        key = (event.symbol, event.timestamp)
        if key in seen:
            raise ValueError("dataset contains duplicate funding timestamp")
        seen.add(key)


def validate_event_bar_coverage(events: Iterable[object]) -> None:
    materialized = tuple(events)
    bars = [item for item in materialized if isinstance(item, BarEvent)]
    if not bars:
        raise ValueError("event coverage requires bars")
    start, end = bars[0].timestamp, bars[-1].timestamp
    for event in materialized:
        timestamp = event.timestamp if hasattr(event, "timestamp") else None
        if timestamp is not None and not start <= timestamp <= end:
            raise ValueError("event timestamp falls outside bar coverage")


def validate_bar_spacing(
    events: Iterable[object], *, timeframe_seconds: int, allow_gaps: bool = False
) -> None:
    if timeframe_seconds <= 0:
        raise ValueError("timeframe_seconds must be positive")
    bars = [item for item in events if isinstance(item, BarEvent)]
    interval = timedelta(seconds=timeframe_seconds)
    for left, right in pairwise(bars):
        delta = right.timestamp - left.timestamp
        if delta < interval or (not allow_gaps and delta != interval) or delta % interval:
            raise ValueError("bar spacing is inconsistent with timeframe")


def validate_minimum_history(dataset: BacktestDataset, *, minimum_bars: int) -> None:
    if minimum_bars < 1:
        raise ValueError("minimum_bars must be positive")
    if dataset.bar_count < minimum_bars:
        raise ValueError(f"dataset has {dataset.bar_count} bars; requires at least {minimum_bars}")


def validate_slice_bounds(*, parent: BacktestDataset, start: datetime, end: datetime) -> None:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("slice timestamps must be timezone-aware")
    if end < start:
        raise ValueError("slice end cannot precede start")
    if end < parent.start or start > parent.end:
        raise ValueError("slice does not overlap the parent dataset")


def validate_slice_nonempty(events: Sequence[object]) -> None:
    if not events or not any(isinstance(item, BarEvent) for item in events):
        raise ValueError("dataset slice must contain at least one bar")


def validate_signal_model_consistency(
    events: Iterable[object], *, maximum_versions: int = 1
) -> None:
    versions = {item.model_version for item in events if isinstance(item, SignalEvent)}
    if len(versions) > maximum_versions:
        raise ValueError("dataset mixes more signal model versions than allowed")


def validate_regime_cardinality(events: Iterable[object], *, maximum_regimes: int = 64) -> None:
    regimes = {item.regime for item in events if isinstance(item, SignalEvent)}
    if len(regimes) > maximum_regimes:
        raise ValueError("signal regime cardinality exceeds configured bound")


def validate_funding_rate_limit(
    events: Iterable[object], *, maximum_absolute_rate: Decimal = DEFAULT_MAX_FUNDING_RATE
) -> None:
    if maximum_absolute_rate <= 0:
        raise ValueError("maximum funding rate must be positive")
    for event in events:
        if not isinstance(event, FundingEvent):
            continue
        if abs(event.rate) > maximum_absolute_rate:
            raise ValueError("funding rate exceeds sanity bound")


def validate_funding_mark_deviation(
    events: Iterable[object], *, maximum_ratio: Decimal = DEFAULT_MAX_FUNDING_MARK_DEVIATION
) -> None:
    if maximum_ratio < 0:
        raise ValueError("maximum funding mark deviation cannot be negative")
    bars = {item.timestamp: item for item in events if isinstance(item, BarEvent)}
    for event in events:
        if not isinstance(event, FundingEvent):
            continue
        bar = bars.get(event.timestamp)
        if bar is None:
            continue
        ratio = abs(event.mark_price / bar.close - _ONE)
        if ratio > maximum_ratio:
            raise ValueError("funding mark price deviates excessively from same-timestamp close")


def validate_fee_rate_ceiling(
    config: EngineConfig, *, maximum_rate: Decimal = DEFAULT_MAX_FEE_RATE
) -> None:
    if maximum_rate <= 0:
        raise ValueError("maximum fee rate must be positive")
    if config.maker_fee_rate > maximum_rate or config.taker_fee_rate > maximum_rate:
        raise ValueError("engine fee rate exceeds sanity bound")


def validate_slippage_limit(
    config: EngineConfig, *, maximum_bps: Decimal = DEFAULT_MAX_SLIPPAGE_BPS
) -> None:
    if maximum_bps < 0:
        raise ValueError("maximum slippage bps cannot be negative")
    if config.market_slippage_bps > maximum_bps:
        raise ValueError("engine slippage exceeds sanity bound")


def validate_leverage_limit(
    config: EngineConfig, *, maximum_leverage: Decimal = DEFAULT_MAX_LEVERAGE
) -> None:
    if maximum_leverage <= 0:
        raise ValueError("maximum leverage must be positive")
    if config.leverage > maximum_leverage:
        raise ValueError("engine leverage exceeds configured research bound")


def validate_fill_limits(config: EngineConfig) -> None:
    if config.max_fills_per_bar and config.max_fills_per_bar < max(
        config.max_entry_layers_per_bar, config.max_reduce_layers_per_bar
    ):
        raise ValueError("max_fills_per_bar cannot be below an enabled per-side layer limit")


def validate_objective_metric_names(config: ObjectiveConfig) -> None:
    keys = [*config.weights, *config.minimums, *config.maximums]
    if any(not isinstance(key, str) for key in keys):
        raise TypeError("objective metric names must be strings")
    if any(not key.strip() for key in keys):
        raise ValueError("objective metric names must be non-empty")


def validate_drawdown_bounds(config: ObjectiveConfig) -> None:
    for mapping in (config.minimums, config.maximums):
        if "max_drawdown_ratio" in mapping:
            value = mapping["max_drawdown_ratio"]
            if value < _ZERO or value > _ONE:
                raise ValueError("max_drawdown_ratio bound must be within [0, 1]")


def validate_evaluation_elapsed(evaluation: BacktestEvaluation) -> None:
    if not evaluation.elapsed_seconds.is_finite() or evaluation.elapsed_seconds < _ZERO:
        raise ValueError("backtest evaluation elapsed_seconds must be finite and non-negative")


def validate_evaluation_timestamp(evaluation: BacktestEvaluation) -> None:
    if evaluation.evaluated_at.tzinfo is None or evaluation.evaluated_at.utcoffset() is None:
        raise ValueError("backtest evaluation timestamp must be timezone-aware")


def validate_evaluation_score(evaluation: BacktestEvaluation) -> None:
    if not evaluation.objective_score.is_finite():
        raise ValueError("backtest objective score must be finite")


def validate_metric_values(metrics: Mapping[str, object]) -> None:
    for key, value in metrics.items():
        if isinstance(value, (str, bool)):
            continue
        number = _decimal(value, field=key)
        if not number.is_finite():
            raise ValueError(f"metric {key} is not finite")


def validate_fill_metric_consistency(metrics: Mapping[str, object]) -> None:
    if "fill_count" not in metrics:
        return
    fill_count = int(metrics["fill_count"])
    maker = int(metrics.get("maker_fill_count_derived", 0))
    taker = int(metrics.get("taker_fill_count_derived", 0))
    if min(fill_count, maker, taker) < 0 or maker + taker != fill_count:
        raise ValueError("derived maker/taker fill counts do not sum to fill_count")


def validate_snapshot_order(result: object) -> None:
    snapshots = tuple(result.snapshots) if hasattr(result, "snapshots") else ()
    timestamps = [item.timestamp for item in snapshots]
    if any(right < left for left, right in pairwise(timestamps)):
        raise ValueError("simulation snapshots are not chronological")


def validate_snapshot_equity_finite(result: object) -> None:
    for snapshot in (result.snapshots if hasattr(result, "snapshots") else ()):
        if not snapshot.equity.is_finite():
            raise ValueError("simulation snapshot equity must be finite")


def validate_unique_fill_ids(events: Iterable[object]) -> None:
    ids = [item.event_id for item in events if isinstance(item, FillEvent)]
    if len(ids) != len(set(ids)):
        raise ValueError("simulation contains duplicate fill event ids")


def validate_unique_account_event_ids(events: Iterable[object]) -> None:
    ids = [item.event_id for item in events if isinstance(item, AccountEvent)]
    if len(ids) != len(set(ids)):
        raise ValueError("simulation contains duplicate account event ids")


def validate_simulation_chronology(events: Iterable[object]) -> None:
    timestamps = [item.timestamp for item in events if hasattr(item, "timestamp")]
    if any(right < left for left, right in pairwise(timestamps)):
        raise ValueError("simulation event log is not chronological")


def validate_candidate_id(candidate: Candidate) -> None:
    if not re.fullmatch(r"candidate-[0-9a-f]{16}", candidate.candidate_id):
        raise ValueError("candidate_id does not use the deterministic candidate hash format")


def validate_candidate_parameters(candidate: Candidate) -> None:
    if not isinstance(candidate.parameters, Mapping):
        raise TypeError("candidate parameters must be a mapping")
    for key in candidate.parameters:
        if not isinstance(key, str):
            raise TypeError("candidate parameter names must be strings")
        if not key.strip():
            raise ValueError("candidate parameter names must be non-empty")


def validate_summary_timestamps(summary: OptimizationSummary) -> None:
    for value in (summary.started_at, summary.completed_at):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("optimization summary timestamps must be timezone-aware")
    if summary.completed_at < summary.started_at:
        raise ValueError("optimization summary completed_at cannot precede started_at")


def validate_summary_candidate_ids(summary: OptimizationSummary) -> None:
    ids = [item.candidate.candidate_id for item in summary.evaluations]
    if len(ids) != len(set(ids)):
        raise ValueError("optimization summary contains duplicate candidate ids")
    if summary.best_candidate_id is not None and summary.best_candidate_id not in set(ids):
        raise ValueError("best_candidate_id is not present in evaluations")


def validate_dataset_contract(dataset: BacktestDataset) -> None:
    validate_dataset_id(dataset.dataset_id)
    validate_symbol_label(dataset.symbol)
    validate_timeframe_label(dataset.timeframe)
    validate_metadata_strings(dataset.metadata)
    validate_metadata_budget(dataset.metadata)
    validate_dataset_fingerprint(dataset.fingerprint)
    validate_dataset_counts(dataset)
    validate_dataset_bounds(dataset)
