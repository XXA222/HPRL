
import unittest

from freqtrade.enums import PositionSide
from freqtrade.hedge.position_book import PositionRecord
from freqtrade.hedge.reconciliation import reconcile_positions


class TestHedgeReconciliation(unittest.TestCase):
    def test_equal_dual_positions_are_consistent(self) -> None:
        local = [
            PositionRecord("ETH/USDT:USDT", PositionSide.LONG, "5"),
            PositionRecord("ETH/USDT:USDT", PositionSide.SHORT, "2"),
        ]
        remote = [
            PositionRecord("ETH/USDT:USDT", PositionSide.LONG, "5"),
            PositionRecord("ETH/USDT:USDT", PositionSide.SHORT, "2"),
        ]
        result = reconcile_positions(local, remote)
        self.assertTrue(result.consistent)
        self.assertEqual(result.issues, ())

    def test_amount_drift_is_reported_per_side(self) -> None:
        result = reconcile_positions(
            [PositionRecord("ETH/USDT:USDT", PositionSide.SHORT, "2")],
            [PositionRecord("ETH/USDT:USDT", PositionSide.SHORT, "2.1")],
        )
        self.assertFalse(result.consistent)
        self.assertEqual(result.issues[0].code, "AMOUNT_MISMATCH")
        self.assertEqual(result.issues[0].key, ("ETH/USDT:USDT", "SHORT"))

    def test_missing_leg_is_reported(self) -> None:
        result = reconcile_positions(
            [PositionRecord("ETH/USDT:USDT", PositionSide.LONG, "5")],
            [
                PositionRecord("ETH/USDT:USDT", PositionSide.LONG, "5"),
                PositionRecord("ETH/USDT:USDT", PositionSide.SHORT, "1"),
            ],
        )
        self.assertFalse(result.consistent)
        self.assertEqual(result.issues[0].code, "MISSING_LOCAL")


if __name__ == "__main__":
    unittest.main()
