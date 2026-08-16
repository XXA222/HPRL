"""Deterministic golden-strategy contract for end-to-end production validation."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class GoldenSignal:
    fast_ma: Decimal
    slow_ma: Decimal
    volatility: Decimal


@dataclass(frozen=True, slots=True)
class GoldenTarget:
    long_ratio: Decimal
    short_ratio: Decimal
    reason: str


def deterministic_golden_target(signal: GoldenSignal) -> GoldenTarget:
    if signal.volatility < 0:
        raise ValueError("volatility must be nonnegative")
    # Deliberately bounded and interpretable; this strategy validates plumbing, not alpha.
    if signal.fast_ma > signal.slow_ma:
        return GoldenTarget(Decimal("0.12"), Decimal("0.03"), "TREND_UP_BOUNDED")
    if signal.fast_ma < signal.slow_ma:
        return GoldenTarget(Decimal("0.03"), Decimal("0.12"), "TREND_DOWN_BOUNDED")
    return GoldenTarget(Decimal("0.03"), Decimal("0.03"), "NEUTRAL_PROBE")
