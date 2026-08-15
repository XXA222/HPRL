import unittest
from decimal import Decimal

from freqtrade.hedge.optimization.config import parse_optimization_config
from freqtrade.hedge.optimization.types import ObjectiveDirection, ParameterKind


class OptimizationConfigTest(unittest.TestCase):
    def config(self) -> dict:
        return {
            "hedge": {
                "optimization": {
                    "study_name": "demo",
                    "trials": 12,
                    "sampler": "grid",
                    "parameters": [
                        {
                            "name": "spacing",
                            "path": "hedge.planner.grid_spacing",
                            "kind": "decimal",
                            "low": "0.005",
                            "high": "0.01",
                            "step": "0.005",
                        }
                    ],
                    "objectives": [
                        {"metric": "net_return", "direction": "maximize", "weight": "2"}
                    ],
                    "constraints": [{"metric": "max_drawdown", "maximum": "0.25"}],
                    "stress_scenarios": [
                        {"name": "baseline"},
                        {"name": "fees", "taker_fee_multiplier": "2"},
                    ],
                    "walk_forward": {"train_size": 60, "validation_size": 15, "test_size": 15},
                }
            }
        }

    def test_parses_strict_typed_configuration(self) -> None:
        parsed = parse_optimization_config(self.config())
        self.assertEqual(parsed.study_name, "demo")
        self.assertEqual(parsed.parameters[0].kind, ParameterKind.DECIMAL)
        self.assertEqual(parsed.objectives[0].direction, ObjectiveDirection.MAXIMIZE)
        self.assertEqual(parsed.objectives[0].weight, Decimal(2))
        self.assertEqual(parsed.walk_forward.train_size, 60)
        self.assertEqual(len(parsed.stress_scenarios), 2)

    def test_empty_parameter_or_objective_space_is_rejected(self) -> None:
        config = self.config()
        config["hedge"]["optimization"]["parameters"] = []
        with self.assertRaisesRegex(ValueError, "parameters"):
            parse_optimization_config(config)
        config = self.config()
        config["hedge"]["optimization"]["objectives"] = []
        with self.assertRaisesRegex(ValueError, "objectives"):
            parse_optimization_config(config)

    def test_invalid_worker_type_and_duplicate_stress_names_fail(self) -> None:
        config = self.config()
        config["hedge"]["optimization"]["workers"] = True
        with self.assertRaisesRegex(TypeError, "integer"):
            parse_optimization_config(config)
        config = self.config()
        config["hedge"]["optimization"]["stress_scenarios"] = [{"name": "same"}, {"name": "same"}]
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_optimization_config(config)


if __name__ == "__main__":
    unittest.main()

class DefaultOutputIsolationTest(unittest.TestCase):
    def test_default_output_directory_is_scoped_by_study_name(self) -> None:
        config = OptimizationConfigTest().config()
        optimization = config["hedge"]["optimization"]
        optimization.pop("output_directory", None)
        optimization.pop("storage_path", None)
        parsed = parse_optimization_config(
            config,
            default_output_directory=__import__("pathlib").Path("root"),
        )
        self.assertEqual(parsed.output_directory, __import__("pathlib").Path("root") / "demo")
        self.assertEqual(
            parsed.storage_path,
            __import__("pathlib").Path("root") / "demo" / "study.sqlite",
        )
