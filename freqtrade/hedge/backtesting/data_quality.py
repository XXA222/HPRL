"""Fail-closed data quality gates for Hedge backtests.

The helpers in this module deliberately accept small, standard-library data
structures so they can run before Freqtrade's heavier data stack is imported.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class Gap:
    left: datetime
    right: datetime
    missing_intervals: int


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    event_count: int
    duplicate_timestamps: tuple[datetime, ...]
    gaps: tuple[Gap, ...]
    invalid_rows: tuple[int, ...]
    aligned_rows: int
    score: Decimal


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _decimal(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a valid decimal") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def validate_strict_timestamps(timestamps: Sequence[datetime]) -> tuple[datetime, ...]:
    normalized = tuple(_utc(item) for item in timestamps)
    for left, right in pairwise(normalized):
        if right <= left:
            raise ValueError("timestamps must be strictly increasing")
    return normalized


def validate_ohlcv_bar(row: Mapping[str, object]) -> dict[str, Decimal]:
    required = ("open", "high", "low", "close", "volume")
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError("OHLCV row is missing: " + ", ".join(missing))
    values = {name: _decimal(row[name], field=name) for name in required}
    if values["volume"] < 0:
        raise ValueError("volume cannot be negative")
    if values["low"] > min(values["open"], values["close"], values["high"]):
        raise ValueError("low exceeds an OHLC component")
    if values["high"] < max(values["open"], values["close"], values["low"]):
        raise ValueError("high is below an OHLC component")
    if values["low"] < 0 or values["open"] < 0 or values["close"] < 0:
        raise ValueError("prices cannot be negative")
    return values


def detect_gaps(timestamps: Sequence[datetime], *, timeframe_seconds: int) -> tuple[Gap, ...]:
    if timeframe_seconds <= 0:
        raise ValueError("timeframe_seconds must be positive")
    normalized = validate_strict_timestamps(timestamps)
    gaps: list[Gap] = []
    interval = timedelta(seconds=timeframe_seconds)
    for left, right in pairwise(normalized):
        delta = right - left
        if delta % interval:
            raise ValueError("timestamp delta is not aligned to timeframe")
        intervals = delta // interval
        if intervals > 1:
            gaps.append(Gap(left, right, intervals - 1))
    return tuple(gaps)


def detect_duplicate_timestamps(timestamps: Iterable[datetime]) -> tuple[datetime, ...]:
    seen: set[datetime] = set()
    duplicates: set[datetime] = set()
    for item in timestamps:
        current = _utc(item)
        if current in seen:
            duplicates.add(current)
        seen.add(current)
    return tuple(sorted(duplicates))


def validate_timeframe_alignment(timestamp: datetime, *, timeframe_seconds: int) -> bool:
    if timeframe_seconds <= 0:
        raise ValueError("timeframe_seconds must be positive")
    current = _utc(timestamp)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    return (current - epoch) % timedelta(seconds=timeframe_seconds) == timedelta(0)


def align_funding_events(
    funding_timestamps: Sequence[datetime],
    bar_timestamps: Sequence[datetime],
) -> tuple[int, ...]:
    bars = validate_strict_timestamps(bar_timestamps)
    if not bars:
        raise ValueError("bar_timestamps cannot be empty")
    output: list[int] = []
    for funding in (_utc(item) for item in funding_timestamps):
        if funding < bars[0] or funding > bars[-1]:
            raise ValueError("funding timestamp falls outside bar range")
        index = 0
        lo, hi = 0, len(bars) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if bars[mid] <= funding:
                index = mid
                lo = mid + 1
            else:
                hi = mid - 1
        output.append(index)
    return tuple(output)


def validate_signal_causality(
    signal_timestamp: datetime,
    first_eligible_fill_timestamp: datetime,
) -> None:
    signal = _utc(signal_timestamp)
    fill = _utc(first_eligible_fill_timestamp)
    if fill <= signal:
        raise ValueError("first eligible fill must occur after signal generation")


def dataset_quality_score(
    *,
    event_count: int,
    duplicate_count: int,
    missing_interval_count: int,
    invalid_row_count: int,
) -> Decimal:
    if event_count <= 0:
        return Decimal(0)
    if min(duplicate_count, missing_interval_count, invalid_row_count) < 0:
        raise ValueError("quality counts cannot be negative")
    penalty = Decimal(duplicate_count + missing_interval_count + invalid_row_count)
    return max(Decimal(0), Decimal(1) - penalty / Decimal(event_count))


def build_data_quality_report(
    rows: Sequence[Mapping[str, object]],
    timestamps: Sequence[datetime],
    *,
    timeframe_seconds: int,
) -> DataQualityReport:
    if len(rows) != len(timestamps):
        raise ValueError("rows and timestamps must have identical length")
    duplicates = detect_duplicate_timestamps(timestamps)
    unique_sorted = tuple(sorted(set(_utc(item) for item in timestamps)))
    gaps = detect_gaps(unique_sorted, timeframe_seconds=timeframe_seconds) if unique_sorted else ()
    invalid: list[int] = []
    aligned = 0
    for index, (row, timestamp) in enumerate(zip(rows, timestamps, strict=True)):
        try:
            validate_ohlcv_bar(row)
        except ValueError:
            invalid.append(index)
        if validate_timeframe_alignment(timestamp, timeframe_seconds=timeframe_seconds):
            aligned += 1
    missing = sum(item.missing_intervals for item in gaps)
    score = dataset_quality_score(
        event_count=len(rows),
        duplicate_count=len(duplicates),
        missing_interval_count=missing,
        invalid_row_count=len(invalid),
    )
    return DataQualityReport(len(rows), duplicates, gaps, tuple(invalid), aligned, score)


def canonical_event_fingerprint(events: Iterable[Mapping[str, object]]) -> str:
    def convert(value: object) -> object:
        if isinstance(value, datetime):
            return _utc(value).isoformat()
        if isinstance(value, Decimal):
            return format(value, "f")
        if isinstance(value, Mapping):
            return {
                str(key): convert(item)
                for key, item in sorted(value.items(), key=lambda x: str(x[0]))
            }
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, float):
            result = Decimal(str(value))
            if not result.is_finite():
                raise ValueError("event values must be finite")
            return format(result, "f")
        return value

    payload = [convert(event) for event in events]
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()
