import unittest
from decimal import Decimal

from freqtrade.hedge.optimization.stress import StressScenario, apply_stress_to_config


class StressScenarioTest(unittest.TestCase):
    def test_stress_changes_only_backtest_cost_inputs(self) -> None:
        base = {
            "api_key": "secret",
            "hedge": {
                "live_trading_enabled": False,
                "paper": {
                    "maker_fee_rate": "0.0002",
                    "taker_fee_rate": "0.0004",
                    "market_slippage_bps": "1",
                    "volume_participation": "0.2",
                },
            },
        }
        stressed = apply_stress_to_config(
            base,
            StressScenario(
                "hard",
                maker_fee_multiplier="2",
                taker_fee_multiplier="3",
                slippage_bps_add="4",
                volume_participation_multiplier="0.5",
                funding_rate_multiplier="2",
            ),
        )
        paper = stressed["hedge"]["paper"]
        self.assertEqual(paper["maker_fee_rate"], Decimal("0.0004"))
        self.assertEqual(paper["taker_fee_rate"], Decimal("0.0012"))
        self.assertEqual(paper["market_slippage_bps"], Decimal(5))
        self.assertEqual(paper["volume_participation"], Decimal("0.10"))
        self.assertEqual(stressed["api_key"], "secret")
        self.assertEqual(base["hedge"]["paper"]["maker_fee_rate"], "0.0002")

    def test_invalid_liquidity_multiplier_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "participation"):
            StressScenario("bad", volume_participation_multiplier="0")


if __name__ == "__main__":
    unittest.main()

class StressRuntimeMetadataTest(unittest.TestCase):
    def test_fee_and_funding_multipliers_are_preserved_for_backtest_adapter(self) -> None:
        stressed = apply_stress_to_config(
            {"hedge": {"paper": {}}},
            StressScenario(
                "costs",
                maker_fee_multiplier="2",
                taker_fee_multiplier="3",
                funding_rate_multiplier="4",
            ),
        )
        runtime = stressed["hedge_optimization_runtime"]
        self.assertEqual(runtime["maker_fee_multiplier"], Decimal(2))
        self.assertEqual(runtime["taker_fee_multiplier"], Decimal(3))
        self.assertEqual(runtime["funding_rate_multiplier"], Decimal(4))
