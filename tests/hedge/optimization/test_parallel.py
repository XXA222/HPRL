import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from time import sleep

from freqtrade.hedge.optimization.config import parse_optimization_config
from freqtrade.hedge.optimization.engine import OptimizationEngine


class ParallelOptimizationTest(unittest.TestCase):
    def test_parallel_workers_preserve_deterministic_trial_order(self) -> None:
        with TemporaryDirectory() as directory:
            raw = {
                "hedge": {
                    "planner": {},
                    "paper": {},
                    "optimization": {
                        "study_name": "parallel",
                        "sampler": "grid",
                        "workers": 3,
                        "storage_path": str(Path(directory) / "study.sqlite"),
                        "parameters": [
                            {
                                "name": "layers",
                                "path": "hedge.planner.max_grid_layers",
                                "kind": "integer",
                                "low": 1,
                                "high": 6,
                            }
                        ],
                        "objectives": [{"metric": "net_return", "direction": "maximize"}],
                    },
                }
            }

            def evaluator(candidate, context):
                layers = candidate["hedge"]["planner"]["max_grid_layers"]
                sleep(0.005 * (7 - layers))
                return {
                    "initial_balance": "1000",
                    "final_equity": str(1000 + layers),
                }

            result = OptimizationEngine(
                base_config=raw,
                optimization_config=parse_optimization_config(raw),
                evaluator=evaluator,
                dataset_fingerprint="parallel-data",
            ).run()
            self.assertEqual(tuple(item.trial_id for item in result.trials), tuple(range(6)))
            self.assertGreaterEqual(len({item.worker for item in result.trials}), 2)
            self.assertEqual(result.best_trial_id, 5)


if __name__ == "__main__":
    unittest.main()
