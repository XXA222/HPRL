import unittest

import numpy as np

from freqtrade.freqai.hedge_rl.actions import HedgeActions
from freqtrade.freqai.hedge_rl.config import HedgeRLConfig
from freqtrade.freqai.hedge_rl.inference import HedgeInferenceGuard


class TestRound17Inference(unittest.TestCase):
    def setUp(self):
        self.guard = HedgeInferenceGuard(
            HedgeRLConfig(confidence_threshold=0.6, max_feature_age_steps=1)
        )

    def test_masked_action_falls_to_valid_choice(self):
        logits = np.zeros(21)
        logits[HedgeActions.LONG_ADD_MEDIUM] = 10
        logits[HedgeActions.LONG_OPEN_SMALL] = 9
        mask = np.ones(21, dtype=bool)
        mask[HedgeActions.LONG_ADD_MEDIUM] = False
        # The project-wide test policy is np.seterr(all="raise"). Keep the isolated
        # ML/RL regression equally strict so expected softmax tail underflow cannot
        # escape detection until FinalAudit48.
        with np.errstate(all="raise"):
            decision = self.guard.decide(logits, action_mask=mask, feature_age_steps=0)
            self.assertEqual(np.geterr()["under"], "raise")
        self.assertEqual(decision.requested_action, HedgeActions.LONG_ADD_MEDIUM)
        self.assertEqual(decision.executed_action, HedgeActions.LONG_OPEN_SMALL)
        self.assertIn("REQUESTED_ACTION_MASKED", decision.reasons)


    def test_nonfinite_logits_fail_closed(self):
        logits = np.zeros(21)
        logits[HedgeActions.LONG_OPEN_MEDIUM] = np.inf
        decision = self.guard.decide(
            logits, action_mask=np.ones(21), feature_age_steps=0
        )
        self.assertEqual(decision.executed_action, HedgeActions.HOLD)
        self.assertIn("NONFINITE_LOGITS", decision.reasons)

    def test_stale_features_force_hold(self):
        logits = np.zeros(21)
        logits[HedgeActions.SHORT_OPEN_SMALL] = 20
        decision = self.guard.decide(logits, action_mask=np.ones(21), feature_age_steps=2)
        self.assertEqual(decision.executed_action, HedgeActions.HOLD)
        self.assertIn("STALE_FEATURES", decision.reasons)

    def test_low_confidence_force_hold(self):
        decision = self.guard.decide(
            np.zeros(21), action_mask=np.ones(21), feature_age_steps=0
        )
        self.assertEqual(decision.executed_action, HedgeActions.HOLD)
        self.assertIn("LOW_CONFIDENCE", decision.reasons)
        self.assertGreater(decision.normalized_entropy, 0.99)


if __name__ == "__main__":
    unittest.main()
