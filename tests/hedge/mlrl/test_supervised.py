import unittest

import numpy as np
import pandas as pd
import torch

from freqtrade.freqai.hedge_rl.supervised import (
    TARGET_NAMES,
    HedgeMultiTaskLoss,
    build_hedge_multitask_targets,
)


class TestRound15Supervised(unittest.TestCase):
    def test_target_shape_bounds_and_tail(self):
        close = pd.Series(np.linspace(100, 120, 50))
        targets = build_hedge_multitask_targets(pd.DataFrame({"close": close}), horizon=5)
        self.assertEqual(tuple(targets.columns), TARGET_NAMES)
        self.assertTrue(targets.iloc[-5:].isna().all().all())
        valid = targets.iloc[:-5]
        self.assertTrue(valid[TARGET_NAMES[0]].between(0, 1).all())
        self.assertTrue(valid[TARGET_NAMES[1]].between(0, 1).all())
        self.assertTrue(valid[TARGET_NAMES[2]].between(-1, 1).all())

    def test_target_uses_exact_horizon_not_later_row(self):
        prices = pd.DataFrame({"close": np.arange(1, 31, dtype=float) + 100})
        before = build_hedge_multitask_targets(prices, horizon=3)
        changed = prices.copy()
        changed.loc[4, "close"] = 999
        after = build_hedge_multitask_targets(changed, horizon=3)
        self.assertNotEqual(before.loc[1, TARGET_NAMES[3]], after.loc[1, TARGET_NAMES[3]])
        self.assertEqual(before.loc[0, TARGET_NAMES[3]], after.loc[0, TARGET_NAMES[3]])

    def test_multitask_loss_backpropagates(self):
        prediction = torch.randn(8, 5, requires_grad=True)
        target = torch.cat(
            [torch.rand(8, 2), torch.rand(8, 1) * 2 - 1, torch.randn(8, 1), torch.rand(8, 1)],
            dim=1,
        )
        loss = HedgeMultiTaskLoss()(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(prediction.grad)


if __name__ == "__main__":
    unittest.main()
