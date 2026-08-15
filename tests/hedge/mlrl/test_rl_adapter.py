import ast
import unittest
from pathlib import Path

from freqtrade.freqai.hedge_rl.actions import HedgeActions, Urgency
from freqtrade.freqai.hedge_rl.config import HedgeRLConfig
from freqtrade.freqai.hedge_rl.freqai_bridge import HedgeFreqAIPolicyBridge, HedgePolicyContext
from freqtrade.freqai.hedge_rl.policy import decode_policy_action


class TestRound16RLAdapter(unittest.TestCase):
    def test_policy_action_decoding(self):
        rebalance = decode_policy_action(HedgeActions.REBALANCE_TO_LONG)
        self.assertGreater(rebalance.long_delta, 0)
        self.assertLess(rebalance.short_delta, 0)
        emergency = decode_policy_action(HedgeActions.CLOSE_BOTH)
        self.assertTrue(emergency.close_long)
        self.assertTrue(emergency.close_short)
        self.assertEqual(emergency.urgency, Urgency.URGENT)


    def test_freqai_bridge_reproduces_flat_shape_and_mask(self):
        class FakeModel:
            def predict(self, observation, **kwargs):
                self.observation = observation
                self.kwargs = kwargs
                return HedgeActions.LONG_OPEN_SMALL, None

        cfg = HedgeRLConfig(observation_window=4)
        bridge = HedgeFreqAIPolicyBridge(
            feature_names=("a", "b"), window_size=4, config=cfg
        )
        context = HedgePolicyContext.neutral(1000, mark=100)
        features = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
        observation = bridge.observation(features, tick=3, context=context)
        model = FakeModel()
        action = bridge.predict_action(
            model, observation, context=context, use_masking=True
        )
        self.assertEqual(action, HedgeActions.LONG_OPEN_SMALL)
        self.assertEqual(observation.shape, (4 * 2 + 12,))
        self.assertEqual(len(model.kwargs["action_masks"]), 21)

    def test_builtin_freqai_adapter_contract(self):
        path = Path("freqtrade/freqai/prediction_models/HedgeReinforcementLearner.py")
        tree = ast.parse(path.read_text())
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        self.assertEqual([node.name for node in classes], ["HedgeReinforcementLearner"])
        assignment_names = {
            target.id
            for node in classes[0].body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertIn("MyRLEnv", assignment_names)


if __name__ == "__main__":
    unittest.main()
