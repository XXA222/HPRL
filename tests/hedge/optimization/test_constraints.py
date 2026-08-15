import unittest
from decimal import Decimal

from freqtrade.hedge.optimization.constraints import evaluate_constraints
from freqtrade.hedge.optimization.types import ConstraintSpec


class ConstraintEvaluationTest(unittest.TestCase):
    def test_feasible_trial_has_zero_violation(self) -> None:
        result = evaluate_constraints(
            {"max_drawdown": Decimal("0.1")},
            (ConstraintSpec("max_drawdown", maximum="0.2"),),
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.total_violation, Decimal(0))

    def test_multiple_violations_are_accumulated_deterministically(self) -> None:
        result = evaluate_constraints(
            {"return": Decimal("-0.1"), "drawdown": Decimal("0.4")},
            (
                ConstraintSpec("return", minimum="0"),
                ConstraintSpec("drawdown", maximum="0.25"),
            ),
        )
        self.assertFalse(result.feasible)
        self.assertEqual(result.total_violation, Decimal("0.25"))
        self.assertEqual(len(result.violations), 2)

    def test_missing_metric_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            evaluate_constraints({}, (ConstraintSpec("liquidated", maximum="0"),))


if __name__ == "__main__":
    unittest.main()
