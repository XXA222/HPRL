import unittest

import numpy as np
import pandas as pd

from freqtrade.freqai.hedge_rl.curriculum import CurriculumScheduler, MarketScenario


class TestRound13Curriculum(unittest.TestCase):
    def setUp(self):
        close = np.linspace(100, 105, 20)
        self.prices = pd.DataFrame(
            {
                "open": close - 0.1,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
            }
        )

    def test_stage_progression(self):
        scheduler = CurriculumScheduler()
        self.assertEqual(scheduler.stage(0).scenario, MarketScenario.BASELINE)
        self.assertEqual(scheduler.stage(0.65).scenario, MarketScenario.HIGH_VOLATILITY)
        self.assertEqual(scheduler.stage(1).scenario, MarketScenario.LIQUIDITY_STRESS)

    def test_transform_is_deterministic_and_valid(self):
        scheduler = CurriculumScheduler()
        a = scheduler.transform_prices(self.prices, progress=0.95, seed=9)
        b = scheduler.transform_prices(self.prices, progress=0.95, seed=9)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue((a["high"] >= a[["open", "low", "close"]].max(axis=1)).all())
        self.assertTrue((a["low"] <= a[["open", "high", "close"]].min(axis=1)).all())
        self.assertEqual(a.attrs["hedge_rl_scenario"], "LIQUIDITY_STRESS")
        self.assertTrue((a["funding_rate"] == 0.001).all())


if __name__ == "__main__":
    unittest.main()
