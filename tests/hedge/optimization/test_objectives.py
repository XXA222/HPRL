import unittest
from decimal import Decimal

from freqtrade.hedge.optimization.objectives import (
    lexicographic_key,
    objective_values,
    scalar_score,
)
from freqtrade.hedge.optimization.types import ObjectiveDirection, ObjectiveSpec


class ObjectiveScoringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.objectives = (
            ObjectiveSpec("net_return", ObjectiveDirection.MAXIMIZE, "2"),
            ObjectiveSpec("max_drawdown", ObjectiveDirection.MINIMIZE, "1"),
        )

    def test_scalar_score_orients_minimize_metrics(self) -> None:
        score = scalar_score(
            {"net_return": Decimal("0.2"), "max_drawdown": Decimal("0.1")},
            self.objectives,
        )
        self.assertEqual(score, Decimal("0.3"))

    def test_vector_preserves_original_metric_values(self) -> None:
        values = objective_values(
            {"net_return": Decimal("0.2"), "max_drawdown": Decimal("0.1")},
            self.objectives,
        )
        self.assertEqual(values, (Decimal("0.2"), Decimal("0.1")))
        self.assertEqual(
            lexicographic_key(values, self.objectives),
            (Decimal("0.2"), Decimal("-0.1")),
        )

    def test_missing_objective_metric_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            scalar_score({"net_return": Decimal("0.1")}, self.objectives)


if __name__ == "__main__":
    unittest.main()
