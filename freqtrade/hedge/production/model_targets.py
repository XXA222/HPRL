"""Fail-closed validation for ML/RL high-level exposure targets.

Models may propose bounded account targets only.  They never carry exchange identifiers,
order types, prices or client-order ids; deterministic Planner/Risk/Execution remain the
only path to venue writes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal


ZERO = Decimal("0")
ONE = Decimal("1")


@dataclass(frozen=True, slots=True)
class ModelTarget:
    sequence: int
    observed_at: datetime
    long_ratio: Decimal
    short_ratio: Decimal
    confidence: Decimal
    risk_budget_multiplier: Decimal = ONE
    pause_entry: bool = False

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("model target sequence must be nonnegative")
        if self.observed_at.tzinfo is None:
            raise ValueError("model target observed_at must be timezone-aware")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))
        for field_name in ("long_ratio", "short_ratio", "confidence", "risk_budget_multiplier"):
            value = Decimal(getattr(self, field_name))
            if not value.is_finite():
                raise ValueError(f"{field_name} must be finite")
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class ModelTargetPolicy:
    max_age: timedelta = timedelta(seconds=5)
    max_long_ratio: Decimal = Decimal("0.40")
    max_short_ratio: Decimal = Decimal("0.40")
    max_gross_ratio: Decimal = Decimal("0.80")
    max_abs_net_ratio: Decimal = Decimal("0.40")
    max_step_delta_ratio: Decimal = Decimal("0.15")
    min_confidence_for_increase: Decimal = Decimal("0.50")
    min_risk_budget_multiplier: Decimal = Decimal("0.25")
    max_risk_budget_multiplier: Decimal = Decimal("1.00")

    def __post_init__(self) -> None:
        if self.max_age <= timedelta(0):
            raise ValueError("model target max_age must be positive")
        for name in (
            "max_long_ratio", "max_short_ratio", "max_gross_ratio", "max_abs_net_ratio",
            "max_step_delta_ratio", "min_confidence_for_increase",
            "min_risk_budget_multiplier", "max_risk_budget_multiplier",
        ):
            value = Decimal(getattr(self, name))
            if not value.is_finite() or value < ZERO:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.max_risk_budget_multiplier < self.min_risk_budget_multiplier:
            raise ValueError("risk budget multiplier bounds are inverted")


@dataclass(frozen=True, slots=True)
class ModelTargetDecision:
    accepted: bool
    long_ratio: Decimal
    short_ratio: Decimal
    reasons: tuple[str, ...]


def validate_model_target(
    target: ModelTarget,
    *,
    now: datetime,
    previous: ModelTarget | None = None,
    policy: ModelTargetPolicy | None = None,
) -> ModelTargetDecision:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    policy = policy or ModelTargetPolicy()
    now = now.astimezone(UTC)
    reasons: list[str] = []
    if target.observed_at > now:
        reasons.append("MODEL_TARGET_FROM_FUTURE")
    elif now - target.observed_at > policy.max_age:
        reasons.append("MODEL_TARGET_STALE")
    if target.long_ratio < ZERO or target.short_ratio < ZERO:
        reasons.append("MODEL_TARGET_NEGATIVE_EXPOSURE")
    if target.long_ratio > policy.max_long_ratio:
        reasons.append("MODEL_TARGET_LONG_LIMIT")
    if target.short_ratio > policy.max_short_ratio:
        reasons.append("MODEL_TARGET_SHORT_LIMIT")
    gross = target.long_ratio + target.short_ratio
    net = target.long_ratio - target.short_ratio
    if gross > policy.max_gross_ratio:
        reasons.append("MODEL_TARGET_GROSS_LIMIT")
    if abs(net) > policy.max_abs_net_ratio:
        reasons.append("MODEL_TARGET_NET_LIMIT")
    if target.confidence < ZERO or target.confidence > ONE:
        reasons.append("MODEL_TARGET_CONFIDENCE_RANGE")
    if not (policy.min_risk_budget_multiplier <= target.risk_budget_multiplier <= policy.max_risk_budget_multiplier):
        reasons.append("MODEL_TARGET_RISK_BUDGET_RANGE")
    if previous is not None:
        if target.sequence <= previous.sequence:
            reasons.append("MODEL_TARGET_SEQUENCE_NOT_MONOTONIC")
        if abs(target.long_ratio - previous.long_ratio) > policy.max_step_delta_ratio:
            reasons.append("MODEL_TARGET_LONG_JUMP")
        if abs(target.short_ratio - previous.short_ratio) > policy.max_step_delta_ratio:
            reasons.append("MODEL_TARGET_SHORT_JUMP")
        increasing = target.long_ratio > previous.long_ratio or target.short_ratio > previous.short_ratio
        if increasing and target.confidence < policy.min_confidence_for_increase:
            reasons.append("MODEL_TARGET_LOW_CONFIDENCE_SCALE_IN")
    if target.pause_entry and previous is not None:
        # pause_entry may preserve or reduce exposure but cannot increase either leg.
        if target.long_ratio > previous.long_ratio or target.short_ratio > previous.short_ratio:
            reasons.append("MODEL_TARGET_PAUSE_ENTRY_INCREASE")
    if reasons:
        # Fail closed to previous target when available; otherwise flat.
        return ModelTargetDecision(
            False,
            previous.long_ratio if previous is not None else ZERO,
            previous.short_ratio if previous is not None else ZERO,
            tuple(dict.fromkeys(reasons)),
        )
    scale = target.risk_budget_multiplier
    return ModelTargetDecision(
        True,
        target.long_ratio * scale,
        target.short_ratio * scale,
        (),
    )
