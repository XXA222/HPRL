from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from freqtrade.hedge.simulation.exchange import BarEvent, SignalEvent

from .artifacts import write_optimization_artifacts
from .config import load_optimization_config
from .contracts import Candidate, OptimizationSummary, SearchMethod
from .dataset import build_dataset
from .dataset_io import load_dataset
from .decimal_utils import json_value
from .parallel import evaluate_parallel
from .runner import HedgeBacktestRunner
from .spaces import grid_candidates, random_candidates
from .splits import walk_forward_splits
from .walkforward import run_walk_forward


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hedge-backtest-opt",
        description="Hedge backtesting and parameter optimization",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Validate and fingerprint a dataset")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--timeframe")
    validate.add_argument("--symbol")
    optimize = sub.add_parser("optimize", help="Run deterministic grid/random optimization")
    optimize.add_argument("--dataset", type=Path, required=True)
    optimize.add_argument("--config", type=Path, required=True)
    optimize.add_argument("--output", type=Path, required=True)
    optimize.add_argument("--timeframe")
    optimize.add_argument("--symbol")
    walk = sub.add_parser("walk-forward", help="Run walk-forward optimization")
    walk.add_argument("--dataset", type=Path, required=True)
    walk.add_argument("--config", type=Path, required=True)
    walk.add_argument("--output", type=Path, required=True)
    walk.add_argument("--timeframe")
    walk.add_argument("--symbol")
    sub.add_parser("self-test", help="Run a dependency-free deterministic smoke test")
    return parser


def _candidates(config: dict[str, object]):
    method = config["method"]
    space = config["space"]
    if method is SearchMethod.GRID:
        return grid_candidates(space, max_candidates=config["max_candidates"])
    if method is SearchMethod.RANDOM:
        return random_candidates(space, count=config["random_count"], seed=config["seed"])
    raise ValueError(
        "CLI Optuna mode requires the optional bridge and is not used by the "
        "dependency-free CLI"
    )


def _summary_from_evaluations(method, evaluations):
    feasible = [item for item in evaluations if item.feasible]
    best = max(
        feasible,
        key=lambda item: (item.objective_score, -item.candidate.ordinal),
        default=None,
    )
    now = datetime.now(UTC)
    return OptimizationSummary(
        method=method,
        evaluations=tuple(evaluations),
        best_candidate_id=best.candidate.candidate_id if best else None,
        started_at=now,
        completed_at=datetime.now(UTC),
    )


def _self_test() -> dict[str, object]:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    events = []
    for index in range(24):
        timestamp = start + timedelta(minutes=index)
        price = Decimal(100) + Decimal(index % 6)
        events.extend(
            [
                SignalEvent(timestamp, "BTC/USDT:USDT", Decimal(1), Decimal(0)),
                BarEvent(
                    timestamp,
                    "BTC/USDT:USDT",
                    price,
                    price + 2,
                    price - 1,
                    price + 1,
                    Decimal(1000),
                ),
            ]
        )
    dataset = build_dataset(events=events, dataset_id="self-test", timeframe="1m")
    evaluation = HedgeBacktestRunner(dataset=dataset).evaluate(Candidate("self-test", {}))
    return {
        "status": "PASS",
        "dataset_fingerprint": dataset.fingerprint,
        "bar_count": dataset.bar_count,
        "objective_score": evaluation.objective_score,
        "total_return_ratio": evaluation.metrics["total_return_ratio"],
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "self-test":
            print(
                json.dumps(
                    json_value(_self_test()), ensure_ascii=False, indent=2, sort_keys=True
                )
            )
            return 0
        dataset = load_dataset(args.dataset, timeframe=args.timeframe, default_symbol=args.symbol)
        if args.command == "validate":
            print(
                json.dumps(
                    json_value(
                        {
                            "status": "PASS",
                            "dataset_id": dataset.dataset_id,
                            "symbol": dataset.symbol,
                            "timeframe": dataset.timeframe,
                            "bars": dataset.bar_count,
                            "signals": dataset.signal_count,
                            "funding": dataset.funding_count,
                            "fingerprint": dataset.fingerprint,
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        config = load_optimization_config(args.config)
        candidates = _candidates(config)
        if args.command == "optimize":
            runner = HedgeBacktestRunner(
                dataset=dataset,
                engine_config=config["engine_config"],
                planner_config=config["planner_config"],
                objective_config=config["objective_config"],
            )
            parallel = evaluate_parallel(
                runner=runner, candidates=candidates, workers=config["workers"]
            )
            summary = _summary_from_evaluations(config["method"], parallel.evaluations)
            paths = write_optimization_artifacts(summary, output_dir=args.output)
            print(
                json.dumps(
                    json_value(
                        {
                            "status": "PASS",
                            "best_candidate_id": summary.best_candidate_id,
                            "artifacts": paths,
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        walk_raw = config["walk_forward"]
        if not isinstance(walk_raw, dict):
            raise TypeError("walk_forward config must be an object")
        folds = walk_forward_splits(
            dataset,
            train_bars=walk_raw["train_bars"],
            test_bars=walk_raw["test_bars"],
            step_bars=walk_raw["step_bars"],
            gap_bars=walk_raw["gap_bars"],
            anchored=walk_raw["anchored"],
        )
        result = run_walk_forward(
            folds=folds,
            candidates=candidates,
            engine_config=config["engine_config"],
            planner_config=config["planner_config"],
            objective_config=config["objective_config"],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(json_value(result), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        print(
            json.dumps(
                json_value(
                    {"status": "PASS", "folds": len(result.folds), "output": args.output}
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
