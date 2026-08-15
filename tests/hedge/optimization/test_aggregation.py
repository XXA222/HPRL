import unittest
from decimal import Decimal

from freqtrade.hedge.optimization.aggregation import aggregate_metric_sets, robustness_score


class FoldAggregationTest(unittest.TestCase):
    def test_aggregates_mean_median_worst_and_dispersion(self) -> None:
        result = aggregate_metric_sets(
            (
                {"net_return": Decimal("0.1"), "max_drawdown": Decimal("0.2")},
                {"net_return": Decimal("0.3"), "max_drawdown": Decimal("0.1")},
                {"net_return": Decimal("-0.1"), "max_drawdown": Decimal("0.3")},
            ),
            required_metrics=("net_return", "max_drawdown"),
        )
        self.assertEqual(result["net_return"], Decimal("0.1"))
        self.assertEqual(result["net_return__median"], Decimal("0.1"))
        self.assertEqual(result["net_return__worst"], Decimal("-0.1"))
        self.assertEqual(result["net_return__range"], Decimal("0.4"))
        self.assertEqual(result["fold_count"], Decimal(3))

    def test_robustness_score_penalizes_unstable_trials(self) -> None:
        stable = aggregate_metric_sets(
            ({"net_return": Decimal("0.1"), "max_drawdown": Decimal("0.1")},) * 3
        )
        unstable = aggregate_metric_sets(
            (
                {"net_return": Decimal("0.4"), "max_drawdown": Decimal("0.1")},
                {"net_return": Decimal("-0.2"), "max_drawdown": Decimal("0.1")},
                {"net_return": Decimal("0.1"), "max_drawdown": Decimal("0.1")},
            )
        )
        self.assertGreater(robustness_score(stable), robustness_score(unstable))

    def test_required_metrics_must_exist_in_every_fold(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            aggregate_metric_sets(
                ({"a": Decimal(1)}, {"b": Decimal(2)}),
                required_metrics=("a",),
            )


if __name__ == "__main__":
    unittest.main()

class ExplicitAggregateExtremaTest(unittest.TestCase):
    def test_min_and_max_are_unambiguous_for_risk_metrics(self) -> None:
        result = aggregate_metric_sets(
            (
                {"max_drawdown": Decimal("0.1")},
                {"max_drawdown": Decimal("0.4")},
            )
        )
        self.assertEqual(result["max_drawdown__min"], Decimal("0.1"))
        self.assertEqual(result["max_drawdown__max"], Decimal("0.4"))
