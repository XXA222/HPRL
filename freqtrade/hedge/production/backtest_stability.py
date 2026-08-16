"""Two-year HPRL backtest stability evidence contract.

The gate intentionally evaluates *measured* chunk/run evidence.  It does not synthesize a
2-year PASS from short smoke tests.  Long histories may be processed in contiguous chunks
as long as coverage, resource ceilings, deterministic result digests and successful exits
are all demonstrated.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from math import isfinite
from typing import Iterable


@dataclass(frozen=True, slots=True)
class BacktestChunkEvidence:
    started_at: datetime
    ended_at: datetime
    bars: int
    events: int
    elapsed_seconds: float
    peak_rss_bytes: int
    exit_code: int
    result_sha256: str
    source_data_sha256: str

    def __post_init__(self) -> None:
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None:
            raise ValueError("backtest timestamps must be timezone-aware")
        if self.ended_at <= self.started_at:
            raise ValueError("backtest chunk must cover positive time")
        object.__setattr__(self, "started_at", self.started_at.astimezone(UTC))
        object.__setattr__(self, "ended_at", self.ended_at.astimezone(UTC))
        if self.bars <= 0 or self.events <= 0 or self.peak_rss_bytes <= 0:
            raise ValueError("backtest chunk counters/resources must be positive")
        if not isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be finite and positive")
        for name in ("result_sha256", "source_data_sha256"):
            value = str(getattr(self, name)).lower()
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"{name} must be SHA-256")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class TwoYearBacktestPolicy:
    minimum_coverage: timedelta = timedelta(days=700)
    maximum_gap: timedelta = timedelta(minutes=2)
    maximum_peak_rss_bytes: int = 12 * 1024**3
    maximum_total_elapsed_seconds: float = 6 * 60 * 60
    minimum_bars: int = 700 * 24 * 60
    require_repeat_digest: bool = True

    def __post_init__(self) -> None:
        if self.minimum_coverage < timedelta(days=365):
            raise ValueError("two-year policy coverage is implausibly short")
        if self.maximum_gap < timedelta(0):
            raise ValueError("maximum_gap must be nonnegative")
        if self.maximum_peak_rss_bytes <= 0 or self.maximum_total_elapsed_seconds <= 0:
            raise ValueError("resource ceilings must be positive")
        if self.minimum_bars <= 0:
            raise ValueError("minimum_bars must be positive")


@dataclass(frozen=True, slots=True)
class TwoYearBacktestStabilityReport:
    passed: bool
    chunks: int
    coverage: timedelta
    bars: int
    events: int
    total_elapsed_seconds: float
    peak_rss_bytes: int
    gap_count: int
    deterministic_repeat: bool
    aggregate_sha256: str
    reasons: tuple[str, ...]


def evaluate_two_year_backtest_stability(
    chunks: Iterable[BacktestChunkEvidence],
    *,
    repeat_result_sha256: str | None = None,
    policy: TwoYearBacktestPolicy | None = None,
) -> TwoYearBacktestStabilityReport:
    effective = policy or TwoYearBacktestPolicy()
    rows = tuple(sorted(chunks, key=lambda item: item.started_at))
    reasons: list[str] = []
    if not rows:
        return TwoYearBacktestStabilityReport(
            False, 0, timedelta(0), 0, 0, 0.0, 0, 0, False,
            sha256(b"[]").hexdigest(), ("NO_BACKTEST_EVIDENCE",),
        )
    coverage = rows[-1].ended_at - rows[0].started_at
    gap_count = 0
    for index, row in enumerate(rows):
        if row.exit_code != 0:
            reasons.append(f"CHUNK_EXIT_NONZERO:{index}:{row.exit_code}")
        if index:
            gap = row.started_at - rows[index - 1].ended_at
            if gap > effective.maximum_gap:
                gap_count += 1
                reasons.append(f"BACKTEST_COVERAGE_GAP:{index}:{gap.total_seconds():.3f}")
            if gap < timedelta(0):
                reasons.append(f"BACKTEST_CHUNK_OVERLAP:{index}:{-gap.total_seconds():.3f}")
    bars = sum(item.bars for item in rows)
    events = sum(item.events for item in rows)
    elapsed = sum(item.elapsed_seconds for item in rows)
    peak = max(item.peak_rss_bytes for item in rows)
    if coverage < effective.minimum_coverage:
        reasons.append("TWO_YEAR_COVERAGE_INSUFFICIENT")
    if bars < effective.minimum_bars:
        reasons.append("TWO_YEAR_BAR_COUNT_INSUFFICIENT")
    if peak > effective.maximum_peak_rss_bytes:
        reasons.append("TWO_YEAR_PEAK_RSS_EXCEEDED")
    if elapsed > effective.maximum_total_elapsed_seconds:
        reasons.append("TWO_YEAR_RUNTIME_EXCEEDED")
    aggregate_payload = [
        {
            "start": x.started_at.isoformat(), "end": x.ended_at.isoformat(),
            "bars": x.bars, "events": x.events, "elapsed": x.elapsed_seconds,
            "peak_rss": x.peak_rss_bytes, "exit": x.exit_code,
            "result": x.result_sha256, "source": x.source_data_sha256,
        }
        for x in rows
    ]
    aggregate = sha256(json.dumps(aggregate_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    repeat = bool(repeat_result_sha256 and repeat_result_sha256.lower() == aggregate)
    if effective.require_repeat_digest and not repeat:
        reasons.append("TWO_YEAR_DETERMINISTIC_REPEAT_MISSING_OR_MISMATCH")
    return TwoYearBacktestStabilityReport(
        passed=not reasons,
        chunks=len(rows),
        coverage=coverage,
        bars=bars,
        events=events,
        total_elapsed_seconds=elapsed,
        peak_rss_bytes=peak,
        gap_count=gap_count,
        deterministic_repeat=repeat,
        aggregate_sha256=aggregate,
        reasons=tuple(dict.fromkeys(reasons)),
    )
