"""Progressive live-candidate canary envelope."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import IntEnum


class CanaryLevel(IntEnum):
    LOCKED = 0
    REDUCE_ONLY = 1
    MICRO = 2
    SMALL = 3
    BOUNDED = 4


@dataclass(frozen=True, slots=True)
class CanaryLimits:
    level: CanaryLevel
    max_order_notional: Decimal
    max_gross_notional: Decimal
    max_daily_loss: Decimal
    max_drawdown_ratio: Decimal
    max_open_orders: int


DEFAULT_CANARY_LIMITS = {
    CanaryLevel.LOCKED: CanaryLimits(CanaryLevel.LOCKED, Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), 0),
    CanaryLevel.REDUCE_ONLY: CanaryLimits(CanaryLevel.REDUCE_ONLY, Decimal("0"), Decimal("0"), Decimal("0"), Decimal("0"), 16),
    CanaryLevel.MICRO: CanaryLimits(CanaryLevel.MICRO, Decimal("10"), Decimal("50"), Decimal("5"), Decimal("0.02"), 4),
    CanaryLevel.SMALL: CanaryLimits(CanaryLevel.SMALL, Decimal("25"), Decimal("150"), Decimal("10"), Decimal("0.03"), 8),
    CanaryLevel.BOUNDED: CanaryLimits(CanaryLevel.BOUNDED, Decimal("100"), Decimal("500"), Decimal("25"), Decimal("0.05"), 16),
}


@dataclass(frozen=True, slots=True)
class CanaryRuntime:
    gross_notional: Decimal
    daily_realized_pnl: Decimal
    drawdown_ratio: Decimal
    open_orders: int
    unresolved_incidents: int

    def __post_init__(self) -> None:
        for name in ("gross_notional", "daily_realized_pnl", "drawdown_ratio"):
            value = Decimal(getattr(self, name))
            if not value.is_finite():
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if self.gross_notional < 0 or self.drawdown_ratio < 0:
            raise ValueError("gross_notional/drawdown_ratio must be nonnegative")
        for name in ("open_orders", "unresolved_incidents"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class CanaryDecision:
    allowed: bool
    effective_level: CanaryLevel
    reasons: tuple[str, ...]


def evaluate_canary(level: CanaryLevel, runtime: CanaryRuntime) -> CanaryDecision:
    limits = DEFAULT_CANARY_LIMITS[level]
    reasons: list[str] = []
    if runtime.unresolved_incidents > 0: reasons.append("UNRESOLVED_INCIDENT")
    if level is CanaryLevel.LOCKED: reasons.append("LIVE_LOCKED")
    if level >= CanaryLevel.MICRO:
        if runtime.gross_notional > limits.max_gross_notional: reasons.append("CANARY_GROSS_LIMIT")
        if runtime.daily_realized_pnl < -limits.max_daily_loss: reasons.append("CANARY_DAILY_LOSS")
        if runtime.drawdown_ratio > limits.max_drawdown_ratio: reasons.append("CANARY_DRAWDOWN")
        if runtime.open_orders > limits.max_open_orders: reasons.append("CANARY_OPEN_ORDER_LIMIT")
    effective = level if not reasons else CanaryLevel.REDUCE_ONLY
    return CanaryDecision(not reasons, effective, tuple(reasons))
