import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from freqtrade.hedge.optimization.config import parse_optimization_config
from freqtrade.hedge.optimization.engine import EvaluationContext, OptimizationEngine
from freqtrade.hedge.optimization.types import TrialStatus


def config(directory: str) -> dict:
    return {
        "hedge": {
            "planner": {},
            "paper": {},
            "optimization": {
                "study_name": "engine-test",
                "sampler": "grid",
                "trials": 3,
                "storage_path": str(Path(directory) / "study.sqlite"),
                "output_directory": str(Path(directory) / "out"),
                "parameters": [
                    {
                        "name": "layers",
                        "path": "hedge.planner.max_grid_layers",
                        "kind": "integer",
                        "low": 1,
                        "high": 3,
                    }
                ],
                "objectives": [
                    {"metric": "net_return", "direction": "maximize"},
                    {"metric": "max_drawdown", "direction": "minimize"},
                ],
                "constraints": [{"metric": "liquidated", "maximum": "0"}],
            },
        }
    }


class OptimizationEngineTest(unittest.TestCase):
    def test_runs_scores_persists_and_resumes(self) -> None:
        with TemporaryDirectory() as directory:
            raw = config(directory)
            parsed = parse_optimization_config(raw)
            calls: list[int] = []

            def evaluator(candidate, context: EvaluationContext):
                layers = candidate["hedge"]["planner"]["max_grid_layers"]
                calls.append(layers)
                return {
                    "initial_balance": "1000",
                    "final_equity": str(1000 + layers * 10),
                    "max_drawdown": str(Decimal("0.1") / layers),
                    "liquidated": False,
                }

            first = OptimizationEngine(
                base_config=raw,
                optimization_config=parsed,
                evaluator=evaluator,
                dataset_fingerprint="dataset-1",
            ).run()
            self.assertEqual(len(first.trials), 3)
            self.assertEqual(first.best_trial_id, 2)
            self.assertEqual(calls, [1, 2, 3])
            self.assertTrue(all(item.status is TrialStatus.COMPLETE for item in first.trials))

            calls.clear()
            second = OptimizationEngine(
                base_config=raw,
                optimization_config=parsed,
                evaluator=evaluator,
                dataset_fingerprint="dataset-1",
            ).run()
            self.assertEqual(second.resumed_trials, 3)
            self.assertEqual(calls, [])

    def test_constraint_violation_marks_trial_infeasible(self) -> None:
        with TemporaryDirectory() as directory:
            raw = config(directory)
            parsed = parse_optimization_config(raw)

            def evaluator(candidate, context):
                return {
                    "initial_balance": "1000",
                    "final_equity": "1100",
                    "max_drawdown": "0.1",
                    "liquidated": True,
                }

            result = OptimizationEngine(
                base_config=raw,
                optimization_config=parsed,
                evaluator=evaluator,
                dataset_fingerprint="dataset",
            ).run()
            self.assertTrue(all(item.status is TrialStatus.INFEASIBLE for item in result.trials))
            self.assertIsNone(result.best_trial_id)

    def test_evaluator_exception_is_captured_as_failed_trial(self) -> None:
        with TemporaryDirectory() as directory:
            raw = config(directory)
            parsed = parse_optimization_config(raw)

            def evaluator(candidate, context):
                raise RuntimeError("simulated failure")

            result = OptimizationEngine(
                base_config=raw,
                optimization_config=parsed,
                evaluator=evaluator,
                dataset_fingerprint="dataset",
            ).run()
            self.assertEqual(result.trials[0].status, TrialStatus.FAILED)
            self.assertIn("simulated failure", result.trials[0].error)


if __name__ == "__main__":
    unittest.main()

class AggregateObjectiveEngineTest(unittest.TestCase):
    def test_aggregate_suffix_objectives_resolve_base_metrics(self) -> None:
        with TemporaryDirectory() as directory:
            raw = config(directory)
            raw["hedge"]["optimization"]["objectives"] = [
                {"metric": "net_return__median", "direction": "maximize"},
                {"metric": "max_drawdown__max", "direction": "minimize"},
            ]
            raw["hedge"]["optimization"]["stress_scenarios"] = [
                {"name": "baseline"},
                {"name": "cost", "taker_fee_multiplier": "2"},
            ]
            parsed = parse_optimization_config(raw)

            def evaluator(candidate, context):
                layers = candidate["hedge"]["planner"]["max_grid_layers"]
                return {
                    "initial_balance": "1000",
                    "final_equity": str(1000 + layers * 10 - context.evaluation_index),
                    "max_drawdown": str(Decimal("0.2") / layers),
                    "liquidated": False,
                }

            result = OptimizationEngine(
                base_config=raw,
                optimization_config=parsed,
                evaluator=evaluator,
                dataset_fingerprint="aggregate-data",
            ).run()
            self.assertEqual(result.best_trial_id, 2)
            self.assertIn("net_return__median", result.trials[0].metrics)
            self.assertIn("max_drawdown__max", result.trials[0].metrics)
