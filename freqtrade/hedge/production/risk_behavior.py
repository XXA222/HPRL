"""Empirical position-management behavior analysis for Risk-Level/HPRL outputs.

This evaluates observed target sequences rather than inferring skill from reward curves.
A model is considered to demonstrate useful position management only when its target
history exhibits bounded tier transitions, de-risking under drawdown, limited adverse
scale-ins and non-degenerate use of multiple risk levels.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
import json
from math import isfinite
from typing import Iterable

LEVELS = (Decimal("0"), Decimal("0.05"), Decimal("0.12"), Decimal("0.25"), Decimal("0.40"))


def _level(value: Decimal) -> int:
    if value not in LEVELS:
        raise ValueError(f"margin ratio {value} is not an exact HPRL tier")
    return LEVELS.index(value)


@dataclass(frozen=True, slots=True)
class HprlBehaviorObservation:
    timestamp: datetime
    long_margin_ratio: Decimal
    short_margin_ratio: Decimal
    equity_return: float
    drawdown: float
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("behavior timestamp must be timezone-aware")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(UTC))
        object.__setattr__(self, "long_margin_ratio", Decimal(self.long_margin_ratio))
        object.__setattr__(self, "short_margin_ratio", Decimal(self.short_margin_ratio))
        _level(self.long_margin_ratio); _level(self.short_margin_ratio)
        for name in ("equity_return", "drawdown", "uncertainty"):
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.drawdown < 0 or not 0 <= self.uncertainty <= 1:
            raise ValueError("drawdown must be nonnegative and uncertainty in [0,1]")


@dataclass(frozen=True, slots=True)
class HprlBehaviorPolicy:
    minimum_observations: int = 10_000
    maximum_one_step_level_increase: int = 1
    maximum_adverse_scale_in_ratio: float = 0.35
    minimum_drawdown_derisk_ratio: float = 0.60
    maximum_churn_ratio: float = 0.35
    minimum_distinct_joint_levels: int = 4
    drawdown_trigger: float = 0.02
    high_uncertainty_threshold: float = 0.70
    maximum_high_uncertainty_gross_margin: Decimal = Decimal("0.24")


@dataclass(frozen=True, slots=True)
class HprlBehaviorReport:
    passed: bool
    observations: int
    distinct_joint_levels: int
    scale_ins: int
    adverse_scale_ins: int
    adverse_scale_in_ratio: float
    drawdown_events: int
    drawdown_derisks: int
    drawdown_derisk_ratio: float
    level_changes: int
    churn_ratio: float
    upward_jump_violations: int
    high_uncertainty_events: int
    high_uncertainty_overrisk: int
    long_level_occupancy: tuple[tuple[int, int], ...]
    short_level_occupancy: tuple[tuple[int, int], ...]
    joint_level_occupancy: tuple[tuple[str, int], ...]
    semantic_sha256: str
    reasons: tuple[str, ...]


def analyze_hprl_position_behavior(
    observations: Iterable[HprlBehaviorObservation],
    *,
    policy: HprlBehaviorPolicy | None = None,
) -> HprlBehaviorReport:
    p = policy or HprlBehaviorPolicy()
    rows = tuple(sorted(observations, key=lambda x: x.timestamp))
    reasons: list[str] = []
    if len(rows) < p.minimum_observations:
        reasons.append("BEHAVIOR_SAMPLE_INSUFFICIENT")
    long_counts: Counter[int] = Counter()
    short_counts: Counter[int] = Counter()
    joint_counts: Counter[str] = Counter()
    for row in rows:
        ll, sl = _level(row.long_margin_ratio), _level(row.short_margin_ratio)
        long_counts[ll] += 1; short_counts[sl] += 1; joint_counts[f"{ll}:{sl}"] += 1
    scale_ins = adverse = drawdown_events = derisks = changes = jump_violations = 0
    uncertainty_events = uncertainty_overrisk = 0
    for previous, current in zip(rows, rows[1:], strict=False):
        prev_l, prev_s = _level(previous.long_margin_ratio), _level(previous.short_margin_ratio)
        cur_l, cur_s = _level(current.long_margin_ratio), _level(current.short_margin_ratio)
        if (cur_l, cur_s) != (prev_l, prev_s):
            changes += 1
        gross_before = previous.long_margin_ratio + previous.short_margin_ratio
        gross_after = current.long_margin_ratio + current.short_margin_ratio
        if cur_l - prev_l > p.maximum_one_step_level_increase or cur_s - prev_s > p.maximum_one_step_level_increase:
            jump_violations += 1
        if gross_after > gross_before:
            scale_ins += 1
            if current.equity_return < 0 or current.drawdown > previous.drawdown:
                adverse += 1
        if current.drawdown >= p.drawdown_trigger and current.drawdown > previous.drawdown:
            drawdown_events += 1
            if gross_after < gross_before:
                derisks += 1
        if current.uncertainty >= p.high_uncertainty_threshold:
            uncertainty_events += 1
            if gross_after > p.maximum_high_uncertainty_gross_margin:
                uncertainty_overrisk += 1
    adverse_ratio = adverse / scale_ins if scale_ins else 0.0
    derisk_ratio = derisks / drawdown_events if drawdown_events else 0.0
    churn = changes / max(1, len(rows) - 1)
    if jump_violations:
        reasons.append("BEHAVIOR_UPWARD_JUMP_VIOLATION")
    if adverse_ratio > p.maximum_adverse_scale_in_ratio:
        reasons.append("BEHAVIOR_ADVERSE_SCALE_IN_EXCESSIVE")
    if drawdown_events and derisk_ratio < p.minimum_drawdown_derisk_ratio:
        reasons.append("BEHAVIOR_DRAWDOWN_DERISK_INSUFFICIENT")
    if churn > p.maximum_churn_ratio:
        reasons.append("BEHAVIOR_LEVEL_CHURN_EXCESSIVE")
    if len(joint_counts) < p.minimum_distinct_joint_levels:
        reasons.append("BEHAVIOR_LEVEL_USAGE_DEGENERATE")
    if uncertainty_overrisk:
        reasons.append("BEHAVIOR_HIGH_UNCERTAINTY_OVERRISK")
    payload = [
        [row.timestamp.isoformat(), str(row.long_margin_ratio), str(row.short_margin_ratio),
         row.equity_return, row.drawdown, row.uncertainty]
        for row in rows
    ]
    digest = sha256(json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
    return HprlBehaviorReport(
        passed=not reasons,
        observations=len(rows),
        distinct_joint_levels=len(joint_counts),
        scale_ins=scale_ins,
        adverse_scale_ins=adverse,
        adverse_scale_in_ratio=adverse_ratio,
        drawdown_events=drawdown_events,
        drawdown_derisks=derisks,
        drawdown_derisk_ratio=derisk_ratio,
        level_changes=changes,
        churn_ratio=churn,
        upward_jump_violations=jump_violations,
        high_uncertainty_events=uncertainty_events,
        high_uncertainty_overrisk=uncertainty_overrisk,
        long_level_occupancy=tuple(sorted(long_counts.items())),
        short_level_occupancy=tuple(sorted(short_counts.items())),
        joint_level_occupancy=tuple(sorted(joint_counts.items())),
        semantic_sha256=digest,
        reasons=tuple(dict.fromkeys(reasons)),
    )
