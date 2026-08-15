from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum

from freqtrade.hedge.simulation.exchange import SimulationInputEvent, SimulationResult

from .decimal_utils import ONE, ZERO


DEFAULT_INITIAL_BALANCE = Decimal(1000)
DEFAULT_LEVERAGE = Decimal(3)
DEFAULT_MAKER_FEE_RATE = Decimal("0.0002")
DEFAULT_TAKER_FEE_RATE = Decimal("0.0004")
DEFAULT_VOLUME_PARTICIPATION = Decimal("0.10")
DEFAULT_PRICE_TICK = Decimal("0.01")
DEFAULT_QTY_STEP = Decimal("0.0001")


class SearchMethod(StrEnum):
    GRID = "grid"
    RANDOM = "random"
    OPTUNA = "optuna"


class SplitMode(StrEnum):
    ROLLING = "rolling"
    ANCHORED = "anchored"
    PURGED_KFOLD = "purged_kfold"


@dataclass(frozen=True, slots=True)
class BacktestDataset:
    events: tuple[SimulationInputEvent, ...]
    dataset_id: str
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    fingerprint: str
    bar_count: int
    signal_count: int
    funding_count: int
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.events:
            raise ValueError("backtest dataset cannot be empty")
        if not self.dataset_id.strip() or not self.symbol.strip() or not self.timeframe.strip():
            raise ValueError("dataset id, symbol and timeframe are required")
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("dataset timestamps must be timezone-aware")
        if self.end < self.start:
            raise ValueError("dataset end cannot precede start")
        if self.bar_count < 1 or min(self.signal_count, self.funding_count) < 0:
            raise ValueError("dataset event counts are invalid")
        if len(self.fingerprint) != 64:
            raise ValueError("dataset fingerprint must be SHA-256 hex")
        try:
            int(self.fingerprint, 16)
        except ValueError as exc:
            raise ValueError("dataset fingerprint must be SHA-256 hex") from exc


@dataclass(frozen=True, slots=True)
class EngineConfig:
    initial_balance: Decimal = DEFAULT_INITIAL_BALANCE
    leverage: Decimal = DEFAULT_LEVERAGE
    maker_fee_rate: Decimal = DEFAULT_MAKER_FEE_RATE
    taker_fee_rate: Decimal = DEFAULT_TAKER_FEE_RATE
    market_slippage_bps: Decimal = ZERO
    volume_participation: Decimal = DEFAULT_VOLUME_PARTICIPATION
    max_fill_ratio_per_order: Decimal = ONE
    max_entry_layers_per_bar: int = 1
    max_reduce_layers_per_bar: int = 1
    max_fills_per_bar: int = 0
    price_tick: Decimal = DEFAULT_PRICE_TICK
    qty_step: Decimal = DEFAULT_QTY_STEP
    min_fill_qty: Decimal = ZERO
    min_fill_notional: Decimal = ZERO

    def __post_init__(self) -> None:
        positive = (self.initial_balance, self.leverage, self.price_tick, self.qty_step)
        if any(not value.is_finite() or value <= ZERO for value in positive):
            raise ValueError("balance, leverage, price tick and quantity step must be positive")
        non_negative = (
            self.maker_fee_rate,
            self.taker_fee_rate,
            self.market_slippage_bps,
            self.min_fill_qty,
            self.min_fill_notional,
        )
        if any(not value.is_finite() or value < ZERO for value in non_negative):
            raise ValueError("engine costs and minimums cannot be negative")
        if not ZERO < self.volume_participation <= ONE:
            raise ValueError("volume participation must be in (0, 1]")
        if not ZERO < self.max_fill_ratio_per_order <= ONE:
            raise ValueError("max fill ratio must be in (0, 1]")
        if min(
            self.max_entry_layers_per_bar,
            self.max_reduce_layers_per_bar,
            self.max_fills_per_bar,
        ) < 0:
            raise ValueError("per-bar limits cannot be negative")


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    weights: Mapping[str, Decimal] = field(
        default_factory=lambda: {
            "total_return_ratio": Decimal("1.0"),
            "sharpe_ratio": Decimal("0.25"),
            "sortino_ratio": Decimal("0.15"),
            "max_drawdown_ratio": Decimal("-0.75"),
            "liquidation_count": Decimal(-100),
        }
    )
    minimums: Mapping[str, Decimal] = field(default_factory=dict)
    maximums: Mapping[str, Decimal] = field(
        default_factory=lambda: {"max_drawdown_ratio": Decimal("0.35")}
    )
    reject_liquidation: bool = True

    def __post_init__(self) -> None:
        values = [*self.weights.values(), *self.minimums.values(), *self.maximums.values()]
        if any(not value.is_finite() for value in values):
            raise ValueError("objective values must be finite")
        if not self.weights:
            raise ValueError("at least one objective weight is required")


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    parameters: Mapping[str, object]
    ordinal: int = 0

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate id cannot be empty")
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int):
            raise TypeError("candidate ordinal must be int")
        if self.ordinal < 0:
            raise ValueError("candidate ordinal cannot be negative")


@dataclass(frozen=True, slots=True)
class BacktestEvaluation:
    candidate: Candidate
    dataset_fingerprint: str
    result: SimulationResult | None
    metrics: Mapping[str, Decimal | int | bool | str]
    objective_score: Decimal
    feasible: bool
    violations: tuple[str, ...] = ()
    elapsed_seconds: Decimal = ZERO
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class OptimizationSummary:
    method: SearchMethod
    evaluations: tuple[BacktestEvaluation, ...]
    best_candidate_id: str | None
    started_at: datetime
    completed_at: datetime
    resumed: bool = False

    @property
    def feasible_count(self) -> int:
        return sum(1 for item in self.evaluations if item.feasible)
