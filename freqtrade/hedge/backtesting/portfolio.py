from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from math import sqrt
from statistics import fmean, pstdev

from freqtrade.hedge.simulation.exchange import SimulationResult

from .contracts import (
    BacktestDataset,
    BacktestEvaluation,
    Candidate,
    EngineConfig,
    ObjectiveConfig,
)
from .decimal_utils import ONE, ZERO
from .runner import HedgeBacktestRunner


@dataclass(frozen=True, slots=True)
class PortfolioPoint:
    timestamp: datetime
    equity: Decimal
    drawdown_ratio: Decimal


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    evaluations: Mapping[str, BacktestEvaluation]
    equity_curve: tuple[PortfolioPoint, ...]
    initial_equity: Decimal
    final_equity: Decimal
    total_return_ratio: Decimal
    max_drawdown_ratio: Decimal
    sharpe_ratio: Decimal
    liquidated_symbol_count: int


def aggregate_simulation_results(
    evaluations: Mapping[str, BacktestEvaluation],
    *,
    periods_per_year: int = 365 * 24 * 60,
) -> PortfolioResult:
    if not evaluations:
        raise ValueError("portfolio aggregation requires evaluations")
    results: dict[str, SimulationResult] = {}
    for symbol, evaluation in evaluations.items():
        if evaluation.result is None:
            raise ValueError(f"portfolio evaluation for {symbol} has no materialized result")
        if not evaluation.result.snapshots:
            raise ValueError(f"portfolio evaluation for {symbol} has no snapshots")
        results[symbol] = evaluation.result
    timestamps = sorted(
        {snapshot.timestamp for result in results.values() for snapshot in result.snapshots}
    )
    indices = {symbol: 0 for symbol in results}
    current = {
        symbol: result.snapshots[0].equity for symbol, result in results.items()
    }
    initial = sum(
        (
            Decimal(str(evaluations[symbol].metrics.get("initial_balance", current[symbol])))
            for symbol in results
        ),
        ZERO,
    )
    peak = initial
    points: list[PortfolioPoint] = []
    returns: list[float] = []
    previous = initial
    for timestamp in timestamps:
        for symbol, result in results.items():
            snapshots = result.snapshots
            index = indices[symbol]
            while index < len(snapshots) and snapshots[index].timestamp <= timestamp:
                current[symbol] = snapshots[index].equity
                index += 1
            indices[symbol] = index
        equity = sum(current.values(), ZERO)
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak > ZERO else ZERO
        points.append(PortfolioPoint(timestamp=timestamp, equity=equity, drawdown_ratio=drawdown))
        if previous > ZERO:
            returns.append(float(equity / previous - ONE))
        previous = equity
    final = points[-1].equity
    volatility = pstdev(returns) if len(returns) > 1 else 0.0
    sharpe = fmean(returns) / volatility * sqrt(periods_per_year) if volatility > 0 else 0.0
    return PortfolioResult(
        evaluations=dict(evaluations),
        equity_curve=tuple(points),
        initial_equity=initial,
        final_equity=final,
        total_return_ratio=(final / initial - ONE if initial > ZERO else ZERO),
        max_drawdown_ratio=max((item.drawdown_ratio for item in points), default=ZERO),
        sharpe_ratio=Decimal(str(sharpe)),
        liquidated_symbol_count=sum(
            int(item.metrics.get("liquidation_count", 0)) > 0 for item in evaluations.values()
        ),
    )


def run_portfolio(
    *,
    datasets: Mapping[str, BacktestDataset],
    candidates: Mapping[str, Candidate] | Candidate,
    engine_configs: Mapping[str, EngineConfig] | None = None,
    planner_config=None,
    objective_config: ObjectiveConfig | None = None,
) -> PortfolioResult:
    evaluations: dict[str, BacktestEvaluation] = {}
    for symbol in sorted(datasets):
        candidate = candidates[symbol] if isinstance(candidates, Mapping) else candidates
        engine = engine_configs[symbol] if engine_configs and symbol in engine_configs else None
        runner = HedgeBacktestRunner(
            dataset=datasets[symbol],
            engine_config=engine,
            planner_config=planner_config,
            objective_config=objective_config,
        )
        evaluations[symbol] = runner.evaluate(candidate)
    return aggregate_simulation_results(evaluations)
