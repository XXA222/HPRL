import unittest

from freqtrade.freqai.hedge_rl.config import HedgeRLConfig
from freqtrade.freqai.hedge_rl.portfolio import PortfolioTransition
from freqtrade.freqai.hedge_rl.rewards import HedgeRewardModel
from freqtrade.freqai.hedge_rl.state import HedgeAccountState


class TestRound09Rewards(unittest.TestCase):
    def test_profitable_transition_positive(self):
        cfg = HedgeRLConfig(reward_clip=100)
        account = HedgeAccountState.initial(1010)
        transition = PortfolioTransition(1000, 1010, 10, 0, 0, 0, 0, 0, 0)
        result = HedgeRewardModel(cfg).calculate(
            transition=transition, account=account, mark=100
        )
        self.assertGreater(result.reward, 0)
        self.assertAlmostEqual(result.equity_return, 1.0)

    def test_invalid_action_penalized(self):
        cfg = HedgeRLConfig(reward_clip=100)
        account = HedgeAccountState.initial(1000)
        transition = PortfolioTransition(1000, 1000, 0, 0, 0, 0, 0, 0, 0)
        valid = HedgeRewardModel(cfg).calculate(
            transition=transition, account=account, mark=100, invalid_action=False
        )
        invalid = HedgeRewardModel(cfg).calculate(
            transition=transition, account=account, mark=100, invalid_action=True
        )
        self.assertLess(invalid.reward, valid.reward)
        self.assertEqual(invalid.invalid_action_penalty, 1.0)

    def test_reward_clipped(self):
        cfg = HedgeRLConfig(reward_clip=2)
        account = HedgeAccountState.initial(2000)
        transition = PortfolioTransition(1000, 2000, 1000, 0, 0, 0, 0, 0, 0)
        result = HedgeRewardModel(cfg).calculate(
            transition=transition, account=account, mark=100
        )
        self.assertEqual(result.reward, 2)


if __name__ == "__main__":
    unittest.main()
