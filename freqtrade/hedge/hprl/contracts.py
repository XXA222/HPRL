"""Stable, dependency-light contracts shared by every HPRL component."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence

from .errors import HPRLShapeError


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass(frozen=True, slots=True)
class TensorShape:
    """Named tensor dimensions used in validation and reports."""

    environments: int
    symbols: int
    features: int
    action_dims_per_leg: int = 1

    def __post_init__(self) -> None:
        values = (self.environments, self.symbols, self.features, self.action_dims_per_leg)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in values
        ):
            raise HPRLShapeError("all tensor dimensions must be positive integers")

    @property
    def action_width(self) -> int:
        return self.symbols * 2 * self.action_dims_per_leg


@dataclass(frozen=True, slots=True)
class TargetExposure:
    """Continuous LONG/SHORT target exposure for one symbol.

    Values are independent; a hedged state such as long=0.7 and short=0.2 is valid.
    """

    symbol: str
    long: float
    short: float

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol cannot be empty")
        if any(not math.isfinite(value) or value < 0 for value in (self.long, self.short)):
            raise ValueError("target exposures must be finite and non-negative")

    @property
    def gross(self) -> float:
        return self.long + self.short

    @property
    def net(self) -> float:
        return self.long - self.short


@dataclass(frozen=True, slots=True)
class ActionProjection:
    """Result of projecting a raw policy action through the hard risk envelope."""

    requested: tuple[float, ...]
    projected: tuple[float, ...]
    clipped: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.requested or len(self.requested) != len(self.projected):
            raise ValueError("action projection vectors must be non-empty and equally sized")
        vectors = (*self.requested, *self.projected)
        if any(not math.isfinite(float(value)) for value in vectors):
            raise ValueError("action projection vectors must be finite")
        if any(float(value) < 0 for value in self.projected):
            raise ValueError("projected action values cannot be negative")
        if not isinstance(self.clipped, bool):
            raise ValueError("action projection clipped flag must be boolean")
        if any(not isinstance(reason, str) or not reason.strip() for reason in self.reasons):
            raise ValueError("action projection reasons must be non-empty strings")


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    total: float
    components: Mapping[str, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.total):
            raise ValueError("reward total must be finite")
        if any(not math.isfinite(float(value)) for value in self.components.values()):
            raise ValueError("reward components must be finite")


@dataclass(frozen=True, slots=True)
class StepDiagnostics:
    equity: float
    gross_exposure: float
    net_exposure: float
    drawdown: float
    turnover: float
    fee_cost: float
    slippage_cost: float
    market_impact_cost: float
    funding_pnl: float
    projection_count: int

    def __post_init__(self) -> None:
        finite = (
            self.equity,
            self.gross_exposure,
            self.net_exposure,
            self.drawdown,
            self.turnover,
            self.fee_cost,
            self.slippage_cost,
            self.market_impact_cost,
            self.funding_pnl,
        )
        if any(not math.isfinite(float(value)) for value in finite):
            raise ValueError("step diagnostics must be finite")
        nonnegative = (
            self.equity,
            self.gross_exposure,
            self.drawdown,
            self.turnover,
            self.fee_cost,
            self.slippage_cost,
            self.market_impact_cost,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("step diagnostics contain an invalid negative value")
        if self.drawdown > 1:
            raise ValueError("drawdown must be within [0, 1]")
        if not isinstance(self.projection_count, int) or isinstance(self.projection_count, bool):
            raise ValueError("projection_count must be a non-negative integer")
        if self.projection_count < 0:
            raise ValueError("projection_count must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class PlannedExecutionIntent:
    """Read-only planning output; it is deliberately not an exchange order."""

    symbol: str
    target_long_exposure: float
    target_short_exposure: float
    confidence: float
    model_id: str
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.model_id.strip():
            raise ValueError("symbol and model_id are required")
        values = (self.target_long_exposure, self.target_short_exposure, self.confidence)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("execution intent values must be finite")
        if self.target_long_exposure < 0 or self.target_short_exposure < 0:
            raise ValueError("target exposure cannot be negative")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be within [0, 1]")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.metadata.items()
        ):
            raise ValueError("execution intent metadata must contain string keys and values")


@dataclass(frozen=True, slots=True)
class OfflineTransition:
    observation: tuple[float, ...]
    action: tuple[float, ...]
    reward: float
    next_observation: tuple[float, ...]
    done: bool

    def __post_init__(self) -> None:
        vectors: Sequence[Sequence[float]] = (
            self.observation,
            self.action,
            self.next_observation,
        )
        if any(not values for values in vectors):
            raise ValueError("offline transition vectors cannot be empty")
        if any(not math.isfinite(float(value)) for values in vectors for value in values):
            raise ValueError("offline transition vectors must be finite")
        if not math.isfinite(self.reward):
            raise ValueError("offline transition reward must be finite")
        if not isinstance(self.done, bool):
            raise ValueError("offline transition done must be a boolean")
