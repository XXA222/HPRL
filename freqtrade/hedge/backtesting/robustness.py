from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from decimal import Decimal
from statistics import fmean

from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent, SignalEvent

from .contracts import (
    BacktestDataset,
    BacktestEvaluation,
    Candidate,
    EngineConfig,
    ObjectiveConfig,
)
from .dataset import build_dataset
from .decimal_utils import ONE, ZERO
from .runner import HedgeBacktestRunner


@dataclass(frozen=True, slots=True)
class StressScenario:
    scenario_id: str
    maker_fee_multiplier: Decimal = ONE
    taker_fee_multiplier: Decimal = ONE
    slippage_bps_addition: Decimal = ZERO
    volume_multiplier: Decimal = ONE
    funding_multiplier: Decimal = ONE
    bar_range_multiplier: Decimal = ONE

    def __post_init__(self) -> None:
        values = (
            self.maker_fee_multiplier,
            self.taker_fee_multiplier,
            self.slippage_bps_addition,
            self.volume_multiplier,
            self.funding_multiplier,
            self.bar_range_multiplier,
        )
        if any(not value.is_finite() for value in values):
            raise ValueError("stress scenario values must be finite")
        if min(
            self.maker_fee_multiplier,
            self.taker_fee_multiplier,
            self.volume_multiplier,
            self.bar_range_multiplier,
        ) <= ZERO:
            raise ValueError("stress multipliers must be positive")
        if self.slippage_bps_addition < ZERO:
            raise ValueError("stress slippage addition cannot be negative")
        if not self.scenario_id.strip():
            raise ValueError("stress scenario id cannot be empty")


@dataclass(frozen=True, slots=True)
class RobustnessResult:
    candidate: Candidate
    evaluations: tuple[tuple[str, BacktestEvaluation], ...]
    worst_score: Decimal
    mean_score: Decimal
    feasible_scenario_ratio: Decimal


def stressed_dataset(dataset: BacktestDataset, scenario: StressScenario) -> BacktestDataset:
    transformed = []
    for event in dataset.events:
        if isinstance(event, BarEvent):
            body_high = max(event.open, event.close)
            body_low = min(event.open, event.close)
            high_extension = (event.high - body_high) * scenario.bar_range_multiplier
            low_extension = (body_low - event.low) * scenario.bar_range_multiplier
            transformed.append(
                BarEvent(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    open=event.open,
                    high=body_high + high_extension,
                    low=max(Decimal("1e-18"), body_low - low_extension),
                    close=event.close,
                    volume=(
                        event.volume * scenario.volume_multiplier
                        if event.volume is not None
                        else None
                    ),
                )
            )
        elif isinstance(event, FundingEvent):
            transformed.append(replace(event, rate=event.rate * scenario.funding_multiplier))
        elif isinstance(event, SignalEvent):
            transformed.append(event)
        else:
            transformed.append(event)
    return build_dataset(
        events=transformed,
        dataset_id=f"{dataset.dataset_id}:stress:{scenario.scenario_id}",
        timeframe=dataset.timeframe,
        metadata={**dataset.metadata, "stress_scenario": scenario.scenario_id},
    )


def run_robustness_matrix(
    *,
    dataset: BacktestDataset,
    candidate: Candidate,
    scenarios: Iterable[StressScenario],
    engine_config: EngineConfig | None = None,
    planner_config=None,
    objective_config: ObjectiveConfig | None = None,
) -> RobustnessResult:
    base_engine = engine_config or EngineConfig()
    evaluations: list[tuple[str, BacktestEvaluation]] = []
    for scenario in scenarios:
        stressed_engine = replace(
            base_engine,
            maker_fee_rate=base_engine.maker_fee_rate * scenario.maker_fee_multiplier,
            taker_fee_rate=base_engine.taker_fee_rate * scenario.taker_fee_multiplier,
            market_slippage_bps=base_engine.market_slippage_bps + scenario.slippage_bps_addition,
            volume_participation=min(
                ONE,
                base_engine.volume_participation * scenario.volume_multiplier,
            ),
        )
        runner = HedgeBacktestRunner(
            dataset=stressed_dataset(dataset, scenario),
            engine_config=stressed_engine,
            planner_config=planner_config,
            objective_config=objective_config,
        )
        evaluations.append((scenario.scenario_id, runner.evaluate(candidate)))
    if not evaluations:
        raise ValueError("robustness matrix requires at least one scenario")
    scores = [item.objective_score for _, item in evaluations]
    finite_scores = [score for score in scores if score.is_finite()]
    mean_score = (
        Decimal(str(fmean(float(score) for score in finite_scores)))
        if finite_scores
        else Decimal("-Infinity")
    )
    return RobustnessResult(
        candidate=candidate,
        evaluations=tuple(evaluations),
        worst_score=min(scores),
        mean_score=mean_score,
        feasible_scenario_ratio=Decimal(sum(item.feasible for _, item in evaluations))
        / Decimal(len(evaluations)),
    )
