"""Freqtrade historical-data adapter for the generic Hedge optimization engine."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from freqtrade.hedge.optimization.artifacts import OptimizationArtifacts, export_optimization_result
from freqtrade.hedge.optimization.config import parse_optimization_config
from freqtrade.hedge.optimization.engine import EvaluationContext, OptimizationEngine
from freqtrade.hedge.optimization.types import OptimizationResult


@dataclass(frozen=True, slots=True)
class HedgeOptimizationRun:
    result: OptimizationResult
    artifacts: OptimizationArtifacts
    dataset_pair: str
    dataset_timeframe: str
    dataset_start: datetime
    dataset_end: datetime
    dataset_bar_count: int


def window_timerange(
    timestamps: Sequence[datetime],
    context: EvaluationContext,
) -> str | None:
    """Return a millisecond timerange for a walk-forward test slice."""

    window = context.window
    if window is None:
        return None
    if not timestamps or window.test.stop > len(timestamps):
        raise ValueError("walk-forward test window exceeds dataset timestamps")
    start = timestamps[window.test.start]
    if window.test.stop < len(timestamps):
        stop = timestamps[window.test.stop]
    elif len(timestamps) >= 2:
        stop = timestamps[-1] + (timestamps[-1] - timestamps[-2])
    else:
        stop = timestamps[-1] + timedelta(minutes=1)
    start_ms = int(start.timestamp() * 1000)
    stop_ms = int(stop.timestamp() * 1000)
    if stop_ms <= start_ms:
        raise ValueError("walk-forward timerange must be positive")
    return f"{start_ms}-{stop_ms}"


def run_freqtrade_hedge_optimization(  # noqa: C901
    config: dict[str, Any],
    *,
    backtest_runner: Callable[..., Any] | None = None,
) -> HedgeOptimizationRun:
    """Optimize Hedge planner/matcher parameters on downloaded Freqtrade data.

    The first deterministic probe binds the study to the actual analyzed signal,
    OHLCV, and funding-event fingerprint.  Every trial then uses the normal
    ``run_freqtrade_hedge_backtest`` path; no alternate fill engine is introduced.
    """

    if backtest_runner is None:
        from freqtrade.optimize.hedge_backtesting import run_freqtrade_hedge_backtest

        backtest_runner = run_freqtrade_hedge_backtest
    default_output = Path(str(config.get("user_data_dir", "user_data"))) / "hyperopt_results"
    optimization = parse_optimization_config(
        config,
        default_output_directory=default_output,
    )
    output = optimization.output_directory.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    probe_path = output / ".dataset-probe.json"
    try:
        probe = backtest_runner(
            deepcopy(config), export_path=probe_path, export_events=False
        )
    finally:
        for path in (probe_path, probe_path.with_suffix(probe_path.suffix + ".sha256")):
            if path.exists():
                path.unlink()

    from freqtrade.hedge.simulation.exchange import BarEvent

    timestamps = tuple(
        event.timestamp for event in probe.dataset.events if isinstance(event, BarEvent)
    )
    if len(timestamps) != probe.dataset.bar_count:
        raise ValueError("dataset bar count does not match replay BarEvent count")

    trial_directory = output / ".trial-artifacts"
    trial_directory.mkdir(parents=True, exist_ok=True)

    def evaluator(
        trial_config: Mapping[str, Any],
        context: EvaluationContext,
    ) -> Mapping[str, object]:
        candidate = deepcopy(dict(trial_config))
        timerange = window_timerange(timestamps, context)
        if timerange is not None:
            candidate["timerange"] = timerange
        artifact = trial_directory / (
            f"trial-{context.trial_id:06d}-eval-{context.evaluation_index:04d}.json"
        )
        try:
            run = backtest_runner(candidate, export_path=artifact, export_events=False)
            if run.dataset.data_fingerprint == "":
                raise ValueError("trial backtest returned an empty data fingerprint")
            if run.dataset.pair != probe.dataset.pair:
                raise ValueError("trial backtest changed the managed dataset pair")
            if run.dataset.timeframe != probe.dataset.timeframe:
                raise ValueError("trial backtest changed the dataset timeframe")
            expected_bars = (
                probe.dataset.bar_count
                if context.window is None
                else context.window.test.length
            )
            if run.dataset.bar_count != expected_bars:
                raise ValueError(
                    "trial backtest returned an unexpected number of bars: "
                    f"expected={expected_bars}; actual={run.dataset.bar_count}"
                )
            # A full-range baseline trial must be byte-for-byte bound to the
            # initial analyzed dataset.  Walk-forward slices and explicit stress
            # scenarios intentionally transform replay inputs (timerange, funding
            # multiplier), so their fingerprints are expected to differ.
            if (
                context.window is None
                and context.stress_scenario.name == "baseline"
                and run.dataset.data_fingerprint != probe.dataset.data_fingerprint
            ):
                raise ValueError(
                    "baseline trial dataset fingerprint drifted from the optimization probe; "
                    "optimization parameters must not mutate analyzed market/signal inputs"
                )
            return run.result.report
        finally:
            for path in (artifact, artifact.with_suffix(artifact.suffix + ".sha256")):
                if path.exists():
                    path.unlink()

    try:
        result = OptimizationEngine(
            base_config=config,
            optimization_config=optimization,
            evaluator=evaluator,
            dataset_fingerprint=probe.dataset.data_fingerprint,
            dataset_size=probe.dataset.bar_count,
            timestamps=timestamps,
        ).run()
        artifacts = export_optimization_result(result, output)
    finally:
        shutil.rmtree(trial_directory, ignore_errors=True)
    return HedgeOptimizationRun(
        result=result,
        artifacts=artifacts,
        dataset_pair=probe.dataset.pair,
        dataset_timeframe=probe.dataset.timeframe,
        dataset_start=probe.dataset.start,
        dataset_end=probe.dataset.end,
        dataset_bar_count=probe.dataset.bar_count,
    )
