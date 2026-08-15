import unittest
from decimal import Decimal

from freqtrade.hedge.optimization.types import (
    ConstraintSpec,
    ObjectiveDirection,
    ObjectiveSpec,
    ParameterKind,
    ParameterSpec,
)


class ParameterContractsTest(unittest.TestCase):
    def test_decimal_parameter_is_normalized_exactly(self) -> None:
        spec = ParameterSpec(
            name="grid",
            path="hedge.planner.grid_spacing",
            kind=ParameterKind.DECIMAL,
            low="0.001",
            high="0.01",
            step="0.001",
        )
        self.assertEqual(spec.low, Decimal("0.001"))
        self.assertEqual(spec.step, Decimal("0.001"))

    def test_paths_are_fail_closed_to_hedge_namespace(self) -> None:
        with self.assertRaisesRegex(ValueError, "hedge"):
            ParameterSpec(
                name="db",
                path="db_url",
                kind=ParameterKind.CATEGORICAL,
                choices=("a", "b"),
            )

    def test_log_and_step_are_mutually_exclusive(self) -> None:
        with self.assertRaisesRegex(ValueError, "log"):
            ParameterSpec(
                name="spacing",
                path="hedge.planner.grid_spacing",
                kind=ParameterKind.DECIMAL,
                low="0.001",
                high="0.1",
                step="0.001",
                log=True,
            )

    def test_objective_and_constraint_validation(self) -> None:
        objective = ObjectiveSpec("total_return_ratio", ObjectiveDirection.MAXIMIZE, "2")
        constraint = ConstraintSpec("max_drawdown", maximum="0.20")
        self.assertEqual(objective.weight, Decimal(2))
        self.assertEqual(constraint.maximum, Decimal("0.20"))


if __name__ == "__main__":
    unittest.main()
