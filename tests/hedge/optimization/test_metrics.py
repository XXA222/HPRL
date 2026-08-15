import unittest
from decimal import Decimal

from freqtrade.hedge.optimization.metrics import normalize_report, require_metrics


class HedgeMetricsTest(unittest.TestCase):
    def test_derives_cost_risk_and_leg_balance_metrics(self) -> None:
        metrics = normalize_report(
            {
                "initial_balance": "1000",
                "final_equity": "1100",
                "fees": "5",
                "funding": "-2",
                "long_pnl": "80",
                "short_pnl": "20",
                "gross_peak": "500",
                "max_drawdown": "-0.1",
                "add_count": 4,
                "reduce_count": 6,
                "final_long_quantity": "1.2",
                "final_short_quantity": "0.8",
                "liquidated": False,
            }
        )
        self.assertEqual(metrics["net_return"], Decimal("0.1"))
        self.assertEqual(metrics["cost_drag_ratio"], Decimal("0.007"))
        self.assertEqual(metrics["leg_pnl_imbalance"], Decimal("0.6"))
        self.assertEqual(metrics["profit_per_action"], Decimal(10))
        self.assertEqual(metrics["max_drawdown"], Decimal("0.1"))

    def test_non_numeric_text_is_excluded_but_invalid_numbers_fail(self) -> None:
        metrics = normalize_report({"initial_balance": "100", "status": "PASS"})
        self.assertNotIn("status", metrics)
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_report({"initial_balance": float("inf")})

    def test_required_metric_check_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            require_metrics({"a": Decimal(1)}, ("a", "b"))


if __name__ == "__main__":
    unittest.main()
