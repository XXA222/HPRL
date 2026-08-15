import unittest

import numpy as np
import pandas as pd

from freqtrade.freqai.hedge_rl.actions import HedgeActions
from freqtrade.freqai.hedge_rl.config import HedgeRLConfig
from freqtrade.freqai.hedge_rl.inference import HedgeInferenceGuard
from freqtrade.freqai.hedge_rl.planner_adapter import HedgeRLPlannerAdapter
from freqtrade.freqai.hedge_rl.state import HedgeAccountState
from freqtrade.freqai.hedge_rl.strategy import apply_hedge_rl_action_columns


class TestRound18PlannerAdapter(unittest.TestCase):
    def test_guarded_action_maps_to_planner_signal(self):
        cfg = HedgeRLConfig(confidence_threshold=0.5)
        logits = np.zeros(21)
        logits[HedgeActions.LONG_OPEN_MEDIUM] = 10
        decision = HedgeInferenceGuard(cfg).decide(
            logits, action_mask=np.ones(21), feature_age_steps=0
        )
        signal = HedgeRLPlannerAdapter(cfg).from_decision(
            decision, account=HedgeAccountState.initial(1000), mark=100
        )
        self.assertGreater(signal.long_score, 0)
        self.assertEqual(signal.short_score, 0)
        self.assertGreater(signal.target_net_ratio, 0)
        self.assertTrue(signal.allow_new_risk)
        self.assertGreater(signal.confidence, 0.9)
        self.assertFalse(signal.shielded)
        self.assertEqual(signal.strategy_columns()["hedge_rl_action"], 2)

    def test_stale_decision_disables_new_risk(self):
        cfg = HedgeRLConfig(max_feature_age_steps=0)
        logits = np.zeros(21)
        logits[HedgeActions.SHORT_OPEN_SMALL] = 20
        decision = HedgeInferenceGuard(cfg).decide(
            logits, action_mask=np.ones(21), feature_age_steps=1
        )
        signal = HedgeRLPlannerAdapter(cfg).from_decision(
            decision, account=HedgeAccountState.initial(1000), mark=100
        )
        self.assertFalse(signal.allow_new_risk)
        self.assertEqual(signal.action, HedgeActions.HOLD)

    def test_strategy_columns_decode_actions(self):
        frame = pd.DataFrame({"&-hedge_action": [1, 8, 15, np.nan]})
        result = apply_hedge_rl_action_columns(frame)
        self.assertGreater(result.loc[0, "hedge_long_score"], 0)
        self.assertGreater(result.loc[1, "hedge_short_score"], 0)
        self.assertEqual(result.loc[2, "hedge_target_net_ratio"], 0)
        self.assertEqual(result.loc[3, "hedge_rl_reason"], "RL:HOLD")


if __name__ == "__main__":
    unittest.main()
