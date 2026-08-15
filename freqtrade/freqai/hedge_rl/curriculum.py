"""Deterministic curriculum stages for progressively harder Hedge RL training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd


class MarketScenario(StrEnum):
    BASELINE = "BASELINE"
    TREND = "TREND"
    MEAN_REVERSION = "MEAN_REVERSION"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    FUNDING_SHOCK = "FUNDING_SHOCK"
    LIQUIDITY_STRESS = "LIQUIDITY_STRESS"


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    minimum_progress: float
    scenario: MarketScenario
    volatility_multiplier: float = 1.0
    drift_per_step: float = 0.0
    funding_rate: float = 0.0
    slippage_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_progress <= 1:
            raise ValueError("minimum_progress must be within [0, 1]")
        if self.volatility_multiplier <= 0 or self.slippage_multiplier <= 0:
            raise ValueError("multipliers must be positive")


DEFAULT_CURRICULUM = (
    CurriculumStage(0.00, MarketScenario.BASELINE),
    CurriculumStage(0.20, MarketScenario.TREND, drift_per_step=0.0002),
    CurriculumStage(0.40, MarketScenario.MEAN_REVERSION, volatility_multiplier=1.2),
    CurriculumStage(0.60, MarketScenario.HIGH_VOLATILITY, volatility_multiplier=2.0),
    CurriculumStage(0.80, MarketScenario.FUNDING_SHOCK, funding_rate=0.0005),
    CurriculumStage(
        0.92,
        MarketScenario.LIQUIDITY_STRESS,
        volatility_multiplier=2.5,
        funding_rate=0.001,
        slippage_multiplier=3.0,
    ),
)


class CurriculumScheduler:
    def __init__(self, stages: tuple[CurriculumStage, ...] = DEFAULT_CURRICULUM) -> None:
        if not stages or stages[0].minimum_progress != 0:
            raise ValueError("curriculum must start at progress 0")
        if tuple(sorted(stage.minimum_progress for stage in stages)) != tuple(
            stage.minimum_progress for stage in stages
        ):
            raise ValueError("curriculum stages must be sorted")
        self.stages = stages

    def stage(self, progress: float) -> CurriculumStage:
        value = max(0.0, min(1.0, float(progress)))
        selected = self.stages[0]
        for candidate in self.stages:
            if candidate.minimum_progress <= value:
                selected = candidate
            else:
                break
        return selected

    def transform_prices(
        self,
        prices: pd.DataFrame,
        *,
        progress: float,
        seed: int,
    ) -> pd.DataFrame:
        required = {"open", "high", "low", "close"}
        if missing := required.difference(prices.columns):
            raise ValueError(f"prices are missing {sorted(missing)}")
        stage = self.stage(progress)
        result = prices.copy()
        close = pd.to_numeric(result["close"], errors="raise").to_numpy(dtype=float)
        if (close <= 0).any() or not np.isfinite(close).all():
            raise ValueError("close prices must be finite and positive")
        returns = np.zeros_like(close)
        returns[1:] = np.diff(np.log(close))
        rng = np.random.default_rng(seed)
        transformed = returns.copy()
        if stage.scenario is MarketScenario.MEAN_REVERSION:
            transformed[1:] = (
                -0.35 * returns[:-1]
                + returns[1:] * stage.volatility_multiplier
                + rng.normal(0, np.std(returns[1:]) * 0.15 + 1e-8, len(returns) - 1)
            )
        else:
            transformed[1:] = returns[1:] * stage.volatility_multiplier + stage.drift_per_step
        rebuilt_close = close[0] * np.exp(np.cumsum(transformed))
        open_values = np.r_[rebuilt_close[0], rebuilt_close[:-1]]
        original_range = np.maximum(
            pd.to_numeric(result["high"]).to_numpy(dtype=float)
            - pd.to_numeric(result["low"]).to_numpy(dtype=float),
            rebuilt_close * 1e-6,
        )
        stressed_range = original_range * stage.volatility_multiplier
        result["open"] = open_values
        result["close"] = rebuilt_close
        result["high"] = np.maximum(open_values, rebuilt_close) + stressed_range / 2
        result["low"] = np.maximum(
            np.minimum(open_values, rebuilt_close) - stressed_range / 2,
            np.finfo(float).tiny,
        )
        result["funding_rate"] = stage.funding_rate
        result.attrs["hedge_rl_scenario"] = stage.scenario.value
        result.attrs["slippage_multiplier"] = stage.slippage_multiplier
        return result
