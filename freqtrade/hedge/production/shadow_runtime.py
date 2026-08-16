"""Windowed soak-run integrity for 24/72h production shadow evidence."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json

from .shadow import ShadowMetrics, ShadowPolicy, qualify_shadow


@dataclass(frozen=True, slots=True)
class ShadowWindow:
    started_at: datetime
    ended_at: datetime
    metrics: ShadowMetrics
    restart_boundary: bool = False
    source_cursor_start: int = 0
    source_cursor_end: int = 0

    def __post_init__(self) -> None:
        if self.started_at.tzinfo is None or self.ended_at.tzinfo is None:
            raise ValueError("shadow window timestamps must be timezone-aware")
        if self.ended_at <= self.started_at:
            raise ValueError("shadow window must have positive duration")
        if self.source_cursor_start < 0 or self.source_cursor_end < self.source_cursor_start:
            raise ValueError("invalid shadow cursor interval")
        object.__setattr__(self, "started_at", self.started_at.astimezone(UTC))
        object.__setattr__(self, "ended_at", self.ended_at.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class ShadowRunPolicy:
    max_window_gap: timedelta = timedelta(seconds=30)
    max_window_overlap: timedelta = timedelta(seconds=1)
    minimum_window_duration: timedelta = timedelta(minutes=5)
    require_cursor_continuity: bool = True

    def __post_init__(self) -> None:
        if self.max_window_gap < timedelta(0) or self.max_window_overlap < timedelta(0):
            raise ValueError("shadow gap/overlap limits must be nonnegative")
        if self.minimum_window_duration <= timedelta(0):
            raise ValueError("minimum_window_duration must be positive")


@dataclass(frozen=True, slots=True)
class ShadowRunQualification:
    passed: bool
    reasons: tuple[str, ...]
    covered_duration: timedelta
    windows: int
    semantic_hash: str


def _aggregate(windows: tuple[ShadowWindow, ...]) -> ShadowMetrics:
    # Latency/ratio fields are conservative maxima; event counters sum.
    return ShadowMetrics(
        duration=sum((x.ended_at - x.started_at for x in windows), timedelta(0)),
        rest_ws_position_divergences=sum(x.metrics.rest_ws_position_divergences for x in windows),
        unknown_orders_peak=max((x.metrics.unknown_orders_peak for x in windows), default=0),
        unresolved_unknown_orders=max((x.metrics.unresolved_unknown_orders for x in windows), default=0),
        sequence_gaps_unrecovered=sum(x.metrics.sequence_gaps_unrecovered for x in windows),
        candle_gaps_unrecovered=sum(x.metrics.candle_gaps_unrecovered for x in windows),
        duplicate_effects=sum(x.metrics.duplicate_effects for x in windows),
        reconciliation_p99_seconds=max((x.metrics.reconciliation_p99_seconds for x in windows), default=0.0),
        loop_p99_ms=max((x.metrics.loop_p99_ms for x in windows), default=0.0),
        db_p99_ms=max((x.metrics.db_p99_ms for x in windows), default=0.0),
        model_p99_ms=max((x.metrics.model_p99_ms for x in windows), default=0.0),
        model_fallbacks=sum(x.metrics.model_fallbacks for x in windows),
        memory_growth_ratio=max((x.metrics.memory_growth_ratio for x in windows), default=0.0),
        restart_recoveries=sum(x.metrics.restart_recoveries for x in windows),
        restart_recovery_failures=sum(x.metrics.restart_recovery_failures for x in windows),
        funding_cycles_observed=sum(x.metrics.funding_cycles_observed for x in windows),
        planner_churn_ratio=max((x.metrics.planner_churn_ratio for x in windows), default=0.0),
        risk_reject_ratio=max((x.metrics.risk_reject_ratio for x in windows), default=0.0),
    )


def qualify_shadow_run(
    windows: tuple[ShadowWindow, ...],
    *,
    target: str,
    run_policy: ShadowRunPolicy | None = None,
    shadow_policy: ShadowPolicy | None = None,
) -> ShadowRunQualification:
    policy = run_policy or ShadowRunPolicy()
    reasons: list[str] = []
    ordered = tuple(sorted(windows, key=lambda x: x.started_at))
    if not ordered:
        return ShadowRunQualification(False, ("NO_SHADOW_WINDOWS",), timedelta(0), 0, sha256(b"[]").hexdigest())
    for index, window in enumerate(ordered):
        if window.ended_at - window.started_at < policy.minimum_window_duration:
            reasons.append(f"WINDOW_TOO_SHORT:{index}")
        if index:
            previous = ordered[index - 1]
            gap = window.started_at - previous.ended_at
            if gap > policy.max_window_gap:
                reasons.append(f"WINDOW_GAP:{index}:{gap.total_seconds():.3f}")
            if gap < -policy.max_window_overlap:
                reasons.append(f"WINDOW_OVERLAP:{index}:{-gap.total_seconds():.3f}")
            if policy.require_cursor_continuity and not window.restart_boundary:
                if window.source_cursor_start > previous.source_cursor_end + 1:
                    reasons.append(f"CURSOR_GAP:{index}")
                if window.source_cursor_start < previous.source_cursor_end:
                    reasons.append(f"CURSOR_REGRESSION:{index}")
    aggregate = _aggregate(ordered)
    base = qualify_shadow(aggregate, target=target, policy=shadow_policy)
    reasons.extend(base.reasons)
    payload = [
        {
            "start": x.started_at.isoformat(),
            "end": x.ended_at.isoformat(),
            "restart": x.restart_boundary,
            "cursor_start": x.source_cursor_start,
            "cursor_end": x.source_cursor_end,
        }
        for x in ordered
    ]
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ShadowRunQualification(not reasons, tuple(dict.fromkeys(reasons)), aggregate.duration, len(ordered), digest)
