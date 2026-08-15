"""Freqtrade-facing adapter for the Hedge BT20 optimization toolkit.

This module deliberately keeps the core optimizer independent from pandas and
Freqtrade's heavy runtime imports.  The analyzed-dataframe bridge imports the
existing Hedge backtest adapter lazily, while event datasets can be optimized in
minimal/offline environments.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from freqtrade.hedge.backtesting.artifacts import write_optimization_artifacts
from freqtrade.hedge.backtesting.config import load_optimization_config
from freqtrade.hedge.backtesting.contracts import BacktestDataset, OptimizationSummary, SearchMethod
from freqtrade.hedge.backtesting.dataset import build_dataset
from freqtrade.hedge.backtesting.parallel import evaluate_parallel
from freqtrade.hedge.backtesting.runner import HedgeBacktestRunner
from freqtrade.hedge.backtesting.spaces import grid_candidates, random_candidates


def optimize_hedge_event_dataset(
    *,
    dataset: BacktestDataset,
    optimization_config: Path,
    output_dir: Path | None = None,
) -> OptimizationSummary:
    config = load_optimization_config(optimization_config)
    method = config["method"]
    if method is SearchMethod.GRID:
        candidates = grid_candidates(
            config["space"],
            max_candidates=config["max_candidates"],
        )
    elif method is SearchMethod.RANDOM:
        candidates = random_candidates(
            config["space"],
            count=config["random_count"],
            seed=config["seed"],
        )
    elif method is SearchMethod.OPTUNA:
        from freqtrade.hedge.backtesting.optuna_backend import run_optuna_search

        runner = HedgeBacktestRunner(
            dataset=dataset,
            engine_config=config["engine_config"],
            planner_config=config["planner_config"],
            objective_config=config["objective_config"],
        )
        summary = run_optuna_search(
            runner=runner,
            space=config["space"],
            trials=config["random_count"],
            seed=config["seed"],
        )
        if output_dir is not None:
            write_optimization_artifacts(summary, output_dir=output_dir)
        return summary
    else:  # pragma: no cover - exhaustive enum guard
        raise ValueError(f"unsupported optimization method: {method}")

    runner = HedgeBacktestRunner(
        dataset=dataset,
        engine_config=config["engine_config"],
        planner_config=config["planner_config"],
        objective_config=config["objective_config"],
    )
    started = datetime.now(UTC)
    evaluations = evaluate_parallel(
        runner=runner,
        candidates=candidates,
        workers=config["workers"],
    ).evaluations
    feasible = [item for item in evaluations if item.feasible]
    best = max(
        feasible,
        key=lambda item: (item.objective_score, -item.candidate.ordinal),
        default=None,
    )
    summary = OptimizationSummary(
        method=method,
        evaluations=evaluations,
        best_candidate_id=best.candidate.candidate_id if best else None,
        started_at=started,
        completed_at=datetime.now(UTC),
    )
    if output_dir is not None:
        write_optimization_artifacts(summary, output_dir=output_dir)
    return summary


def dataset_from_analyzed_dataframe(
    *,
    pair: str,
    timeframe: str,
    frame: Any,
    funding_frame: Any | None = None,
    strategy_version: object = None,
    require_funding_data: bool = False,
    max_missing_candles: int = 0,
    dataset_id: str = "freqtrade-analyzed-data",
) -> BacktestDataset:
    """Convert the existing production Hedge dataframe adapter into a BT20 dataset."""
    from freqtrade.optimize.hedge_backtesting import events_from_analyzed_dataframe

    adapted = events_from_analyzed_dataframe(
        pair=pair,
        timeframe=timeframe,
        frame=frame,
        funding_frame=funding_frame,
        strategy_version=strategy_version,
        require_funding_data=require_funding_data,
        max_missing_candles=max_missing_candles,
    )
    from freqtrade.hedge.simulation.exchange import FundingEvent, SignalEvent

    def event_priority(event: object) -> int:
        if isinstance(event, SignalEvent):
            return 0
        if isinstance(event, FundingEvent):
            return 1
        return 2

    normalized_events = tuple(
        event
        for _, event in sorted(
            enumerate(adapted.events),
            key=lambda item: (item[1].timestamp, event_priority(item[1]), item[0]),
        )
    )
    return build_dataset(
        events=normalized_events,
        dataset_id=dataset_id,
        timeframe=timeframe,
        metadata={
            "adapter": "freqtrade.optimize.hedge_backtesting",
            "source_data_fingerprint": adapted.data_fingerprint,
        },
    )
