"""Risk-sensitive reward components and explainability (rounds 71-80)."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


# Round 71 -------------------------------------------------------------------------------
def safe_log_equity_return(previous_equity: float, equity: float, *, floor: float = 1e-12) -> float:
    previous = float(previous_equity)
    current = float(equity)
    if not all(math.isfinite(item) for item in (previous, current, floor)) or floor <= 0:
        raise ValueError("equity values and floor must be finite; floor must be positive")
    return math.log(max(current, floor) / max(previous, floor))


# Round 72 -------------------------------------------------------------------------------
def downside_deviation(returns: npt.ArrayLike, *, target: float = 0.0) -> float:
    values = np.asarray(returns, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all() or not math.isfinite(target):
        raise ValueError("returns and target must be finite")
    downside = np.minimum(values - target, 0.0)
    return float(np.sqrt(np.mean(downside**2)))


# Round 73 -------------------------------------------------------------------------------
def drawdown_delta(
    *,
    previous_equity: float,
    previous_peak: float,
    equity: float,
    peak: float,
) -> float:
    values = tuple(float(item) for item in (previous_equity, previous_peak, equity, peak))
    if not all(math.isfinite(item) for item in values) or previous_peak <= 0 or peak <= 0:
        raise ValueError("equity and peak values must be finite with positive peaks")
    previous_drawdown = max(0.0, 1.0 - previous_equity / previous_peak)
    current_drawdown = max(0.0, 1.0 - equity / peak)
    return current_drawdown - previous_drawdown


# Round 74 -------------------------------------------------------------------------------
def turnover_penalty(*, traded_notional: float, equity: float, weight: float) -> float:
    if not all(math.isfinite(float(item)) for item in (traded_notional, equity, weight)):
        raise ValueError("turnover inputs must be finite")
    if traded_notional < 0 or weight < 0:
        raise ValueError("turnover and weight cannot be negative")
    return weight * traded_notional / max(abs(equity), 1e-12)


# Round 75 -------------------------------------------------------------------------------
def exposure_penalty(
    *,
    gross_exposure: float,
    net_exposure: float,
    gross_limit: float,
    net_limit: float,
    weight: float = 1.0,
) -> float:
    values = tuple(
        float(item)
        for item in (gross_exposure, net_exposure, gross_limit, net_limit, weight)
    )
    if (
        not all(math.isfinite(item) for item in values)
        or min(gross_limit, net_limit) <= 0
        or weight < 0
    ):
        raise ValueError("invalid exposure penalty inputs")
    gross_excess = max(0.0, gross_exposure / gross_limit - 1.0)
    net_excess = max(0.0, abs(net_exposure) / net_limit - 1.0)
    return weight * (gross_excess**2 + net_excess**2)


# Round 76 -------------------------------------------------------------------------------
def funding_cost_penalty(*, funding_cashflow: float, equity: float, weight: float = 1.0) -> float:
    if (
        not all(math.isfinite(float(item)) for item in (funding_cashflow, equity, weight))
        or weight < 0
    ):
        raise ValueError("invalid funding penalty inputs")
    return weight * max(0.0, -funding_cashflow) / max(abs(equity), 1e-12)


# Round 77 -------------------------------------------------------------------------------
def invalid_action_penalty(
    *,
    invalid: bool,
    consecutive_invalid: int,
    base: float = 1.0,
    escalation: float = 0.25,
) -> float:
    if consecutive_invalid < 0 or base < 0 or escalation < 0:
        raise ValueError("invalid-action penalty inputs cannot be negative")
    return 0.0 if not invalid else base * (1.0 + escalation * consecutive_invalid)


# Round 78 -------------------------------------------------------------------------------
def conditional_value_at_risk(returns: npt.ArrayLike, *, alpha: float = 0.05) -> float:
    values = np.asarray(returns, dtype=np.float64).reshape(-1)
    if not len(values) or not np.isfinite(values).all() or not 0 < alpha <= 1:
        raise ValueError("CVaR requires finite returns and alpha within (0, 1]")
    cutoff = np.quantile(values, alpha)
    tail = values[values <= cutoff]
    return float(tail.mean())


# Round 79 -------------------------------------------------------------------------------
@dataclass(slots=True)
class RewardNormalizer:
    clip: float = 10.0
    epsilon: float = 1e-8
    count: int = field(init=False, default=0)
    mean: float = field(init=False, default=0.0)
    m2: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.clip)
            or not math.isfinite(self.epsilon)
            or min(self.clip, self.epsilon) <= 0
        ):
            raise ValueError("clip and epsilon must be finite and positive")

    def normalize(self, reward: float, *, update: bool = True) -> float:
        value = float(reward)
        if not math.isfinite(value):
            raise ValueError("reward must be finite")
        if update:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (value - self.mean)
        variance = self.m2 / max(self.count - 1, 1)
        normalized = (value - self.mean) / math.sqrt(max(variance, self.epsilon))
        return max(-self.clip, min(self.clip, normalized))


# Round 80 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ExplainedReward:
    total: float
    contributions: dict[str, float]
    checksum: float


class RewardExplainer:
    def aggregate(
        self,
        positive: Mapping[str, float],
        penalties: Mapping[str, float],
        *,
        clip: float | None = None,
    ) -> ExplainedReward:
        contributions: dict[str, float] = {}
        for name, value in positive.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"positive reward component {name} is non-finite")
            contributions[str(name)] = numeric
        for name, value in penalties.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0:
                raise ValueError(f"penalty component {name} must be finite and non-negative")
            contributions[str(name)] = -numeric
        checksum = float(sum(contributions.values()))
        total = checksum
        if clip is not None:
            if not math.isfinite(clip) or clip <= 0:
                raise ValueError("clip must be finite and positive")
            total = max(-clip, min(clip, total))
        return ExplainedReward(total=total, contributions=contributions, checksum=checksum)
