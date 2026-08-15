import unittest

from freqtrade.freqai.hedge_rl.actions import DEFAULT_ACTION_CATALOG, HedgeActions
from freqtrade.freqai.hedge_rl.costs import ExecutionCostModel
from freqtrade.freqai.hedge_rl.portfolio import HedgePortfolioSimulator


class TestRound07Portfolio(unittest.TestCase):
    def setUp(self):
        self.sim = HedgePortfolioSimulator(1000, ExecutionCostModel(fee_rate=0, slippage_bps=0))

    def test_independent_dual_legs(self):
        self.sim.apply_action(
            DEFAULT_ACTION_CATALOG.decode(HedgeActions.LONG_OPEN_MEDIUM),
            reference_price=100,
        )
        self.sim.apply_action(
            DEFAULT_ACTION_CATALOG.decode(HedgeActions.SHORT_OPEN_SMALL),
            reference_price=110,
        )
        self.assertAlmostEqual(self.sim.state.long.quantity, 2.5)
        self.assertAlmostEqual(self.sim.state.long.average_price, 100)
        self.assertAlmostEqual(self.sim.state.short.quantity, 100 / 110)
        self.assertAlmostEqual(self.sim.state.short.average_price, 110)
        self.sim.mark_to_market(105)
        self.assertAlmostEqual(self.sim.state.equity, 1000 + 12.5 + 100 / 110 * 5)

    def test_reduce_realizes_only_selected_leg(self):
        self.sim.apply_action(
            DEFAULT_ACTION_CATALOG.decode(HedgeActions.LONG_OPEN_MEDIUM),
            reference_price=100,
        )
        transition = self.sim.apply_action(
            DEFAULT_ACTION_CATALOG.decode(HedgeActions.LONG_REDUCE_MEDIUM),
            reference_price=120,
        )
        self.assertAlmostEqual(transition.realized_pnl, 25)
        self.assertAlmostEqual(self.sim.state.long.quantity, 1.25)
        self.assertEqual(self.sim.state.short.quantity, 0)

    def test_positive_funding_offsets_hedged_legs(self):
        self.sim.apply_action(
            DEFAULT_ACTION_CATALOG.decode(HedgeActions.BOTH_OPEN_SMALL),
            reference_price=100,
        )
        transition = self.sim.apply_action(
            DEFAULT_ACTION_CATALOG.decode(HedgeActions.HOLD),
            reference_price=100,
            funding_rate=0.001,
        )
        self.assertAlmostEqual(transition.funding_cashflow, 0.0)


if __name__ == "__main__":
    unittest.main()
