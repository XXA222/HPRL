from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, replace
from decimal import Decimal
from time import perf_counter

from freqtrade.hedge.planning.context import PlannerConfig
from freqtrade.hedge.simulation.exchange import MarketRules
from freqtrade.hedge.simulation.matcher import MatchConfig
from freqtrade.hedge.simulation.replay import EventReplayEngine

from .contracts import (
    BacktestDataset,
    BacktestEvaluation,
    Candidate,
    EngineConfig,
    ObjectiveConfig,
)
from .decimal_utils import ZERO, to_decimal
from .metrics import compute_metrics


_PLANNER_FIELDS = {item.name: item for item in fields(PlannerConfig)}


def planner_config_with_overrides(
    base: PlannerConfig,
    parameters: Mapping[str, object],
) -> PlannerConfig:
    unknown = sorted(set(parameters) - set(_PLANNER_FIELDS))
    if unknown:
        raise ValueError("unknown PlannerConfig parameter(s): " + ", ".join(unknown))
    converted: dict[str, object] = {}
    for name, value in parameters.items():
        current = getattr(base, name)
        if isinstance(current, Decimal):
            converted[name] = to_decimal(value, field=name)
        elif isinstance(current, bool):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
            converted[name] = value
        elif isinstance(current, int):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be int")
            converted[name] = value
        else:
            converted[name] = value
    return replace(base, **converted)


def objective_score(
    metrics: Mapping[str, Decimal | int | bool | str],
    config: ObjectiveConfig,
) -> tuple[Decimal, bool, tuple[str, ...]]:
    violations: list[str] = []
    for key, threshold in config.minimums.items():
        value = to_decimal(metrics.get(key, ZERO), field=key)
        if value < threshold:
            violations.append(f"{key}={value} below minimum {threshold}")
    for key, threshold in config.maximums.items():
        value = to_decimal(metrics.get(key, ZERO), field=key)
        if value > threshold:
            violations.append(f"{key}={value} above maximum {threshold}")
    liquidation_raw = metrics.get("liquidation_count", 0)
    if isinstance(liquidation_raw, bool) or not isinstance(liquidation_raw, int):
        raise TypeError("liquidation_count must be int")
    liquidation_count = liquidation_raw
    if config.reject_liquidation and liquidation_count > 0:
        violations.append(f"liquidation_count={liquidation_count}")
    score = sum(
        (
            to_decimal(metrics.get(key, ZERO), field=key) * weight
            for key, weight in config.weights.items()
        ),
        ZERO,
    )
    if not score.is_finite():
        raise ValueError("objective score must be finite")
    feasible = not violations
    return score, feasible, tuple(violations)


class HedgeBacktestRunner:
    """Fresh-engine deterministic candidate runner over an immutable event dataset."""

    def __init__(
        self,
        *,
        dataset: BacktestDataset,
        engine_config: EngineConfig | None = None,
        planner_config: PlannerConfig | None = None,
        objective_config: ObjectiveConfig | None = None,
        periods_per_year: int = 365 * 24 * 60,
    ) -> None:
        self.dataset = dataset
        self.engine_config = engine_config or EngineConfig()
        self.planner_config = planner_config or PlannerConfig()
        self.objective_config = objective_config or ObjectiveConfig()
        self.periods_per_year = periods_per_year
        if periods_per_year < 1:
            raise ValueError("periods_per_year must be positive")

    def _engine(self, candidate: Candidate) -> EventReplayEngine:
        cfg = self.engine_config
        planner = planner_config_with_overrides(self.planner_config, candidate.parameters)
        rules = MarketRules(
            tick_size=cfg.price_tick,
            qty_step=cfg.qty_step,
            min_qty=cfg.min_fill_qty,
            min_notional=cfg.min_fill_notional,
        )
        matcher = MatchConfig(
            maker_fee_rate=cfg.maker_fee_rate,
            taker_fee_rate=cfg.taker_fee_rate,
            volume_participation=cfg.volume_participation,
            market_slippage_bps=cfg.market_slippage_bps,
            price_tick=cfg.price_tick,
            qty_step=cfg.qty_step,
            min_fill_qty=cfg.min_fill_qty,
            min_fill_notional=cfg.min_fill_notional,
            max_entry_layers_per_bar=cfg.max_entry_layers_per_bar,
            max_reduce_layers_per_bar=cfg.max_reduce_layers_per_bar,
            max_fill_ratio_per_order=cfg.max_fill_ratio_per_order,
            max_fills_per_bar=cfg.max_fills_per_bar,
        )
        return EventReplayEngine(
            initial_balance=cfg.initial_balance,
            planner_config=planner,
            leverage=cfg.leverage,
            fee_rate=cfg.taker_fee_rate,
            market_rules=rules,
            match_config=matcher,
        )

    def evaluate(self, candidate: Candidate) -> BacktestEvaluation:
        started = perf_counter()
        result = self._engine(candidate).replay(self.dataset.events)
        metrics = compute_metrics(result, periods_per_year=self.periods_per_year)
        score, feasible, violations = objective_score(metrics, self.objective_config)
        elapsed = Decimal(str(perf_counter() - started))
        return BacktestEvaluation(
            candidate=candidate,
            dataset_fingerprint=self.dataset.fingerprint,
            result=result,
            metrics=metrics,
            objective_score=score,
            feasible=feasible,
            violations=violations,
            elapsed_seconds=elapsed,
        )
