import unittest

import numpy as np
import pandas as pd

from freqtrade.freqai.hedge_rl.actions import HedgeActions
from freqtrade.freqai.hedge_rl.environment import HedgeTradingEnv


def make_env(random_start=False):
    features = pd.DataFrame({"ret": np.linspace(-0.1, 0.1, 10), "vol": np.ones(10)})
    opens = np.array([100, 101, 102, 200, 201, 202, 203, 204, 205, 206], dtype=float)
    prices = pd.DataFrame(
        {
            "open": opens,
            "high": opens + 2,
            "low": opens - 2,
            "close": opens + 1,
            "volume": 10,
        }
    )
    config = {
        "freqai": {
            "hedge_rl_config": {
                "observation_window": 3,
                "max_episode_steps": 5,
                "random_start": random_start,
                "fee_rate": 0,
                "slippage_bps": 0,
            }
        }
    }
    return HedgeTradingEnv(df=features, prices=prices, config=config)


class TestRound10Environment(unittest.TestCase):
    def test_gym_contract_and_next_bar_execution(self):
        env = make_env()
        obs, info = env.reset(seed=7)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertEqual(info["tick"], 2)
        obs, reward, terminated, truncated, info = env.step(HedgeActions.LONG_OPEN_SMALL)
        self.assertEqual(info["tick"], 3)
        self.assertAlmostEqual(env.simulator.state.long.average_price, 200)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(np.isfinite(reward))

    def test_invalid_action_becomes_hold_and_is_penalized(self):
        env = make_env()
        env.reset()
        _, reward, _, _, info = env.step(HedgeActions.LONG_ADD_SMALL)
        self.assertTrue(info["invalid_action"])
        self.assertEqual(info["executed_action"], HedgeActions.HOLD)
        self.assertLess(reward, 0)
        self.assertEqual(env.simulator.state.long.quantity, 0)

    def test_action_mask_reflects_positions(self):
        env = make_env()
        env.reset()
        self.assertFalse(env.action_masks()[HedgeActions.LONG_CLOSE])
        env.step(HedgeActions.LONG_OPEN_SMALL)
        self.assertTrue(env.action_masks()[HedgeActions.LONG_CLOSE])
        self.assertFalse(env.action_masks()[HedgeActions.LONG_OPEN_SMALL])


if __name__ == "__main__":
    unittest.main()
