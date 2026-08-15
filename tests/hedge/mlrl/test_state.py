import unittest

from freqtrade.freqai.hedge_rl.state import (
    HedgeAccountState,
    HedgeLegSide,
    HedgeLegState,
    MarketBar,
)


class TestRound03State(unittest.TestCase):
    def test_dual_leg_exposure_and_pnl(self):
        state = HedgeAccountState(
            cash_balance=1000,
            equity=1000,
            peak_equity=1100,
            long=HedgeLegState(HedgeLegSide.LONG, 2, 100),
            short=HedgeLegState(HedgeLegSide.SHORT, 1, 110),
        )
        self.assertEqual(state.long.unrealized_pnl(105), 10)
        self.assertEqual(state.short.unrealized_pnl(105), 5)
        self.assertAlmostEqual(state.gross_exposure(105), 0.315)
        self.assertAlmostEqual(state.net_exposure(105), 0.105)
        self.assertAlmostEqual(state.drawdown(), 1 - 1000 / 1100)

    def test_market_bar_invariants(self):
        MarketBar(100, 110, 90, 105, 10)
        with self.assertRaises(ValueError):
            MarketBar(100, 99, 90, 105)

    def test_flat_leg_cannot_keep_price(self):
        with self.assertRaises(ValueError):
            HedgeLegState(HedgeLegSide.LONG, 0, 100)


if __name__ == "__main__":
    unittest.main()
