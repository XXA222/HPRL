import unittest

from freqtrade.freqai.hedge_rl.actions import Urgency
from freqtrade.freqai.hedge_rl.costs import ExecutionCostModel
from freqtrade.freqai.hedge_rl.state import HedgeLegSide


class TestRound06Costs(unittest.TestCase):
    def test_buy_sell_slippage_and_fee(self):
        model = ExecutionCostModel(fee_rate=0.001, slippage_bps=10)
        buy = model.estimate(reference_price=100, quantity=2, is_buy=True)
        sell = model.estimate(reference_price=100, quantity=2, is_buy=False)
        self.assertAlmostEqual(buy.fill_price, 100.1)
        self.assertAlmostEqual(sell.fill_price, 99.9)
        self.assertAlmostEqual(buy.fee, buy.notional * 0.001)

    def test_urgent_is_more_conservative(self):
        model = ExecutionCostModel(slippage_bps=5)
        normal = model.estimate(reference_price=100, quantity=1, is_buy=True)
        urgent = model.estimate(
            reference_price=100,
            quantity=1,
            is_buy=True,
            urgency=Urgency.URGENT,
        )
        self.assertGreater(urgent.fill_price, normal.fill_price)

    def test_funding_sign(self):
        self.assertLess(
            ExecutionCostModel.funding_cashflow(
                side=HedgeLegSide.LONG, notional=1000, funding_rate=0.001
            ),
            0,
        )
        self.assertGreater(
            ExecutionCostModel.funding_cashflow(
                side=HedgeLegSide.SHORT, notional=1000, funding_rate=0.001
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
