import unittest

import torch

from freqtrade.freqai.hedge_rl.networks import (
    HedgeActorCriticNetwork,
    HedgeMultiTaskMLP,
    HedgeMultiTaskNetwork,
    HedgeTemporalEncoder,
)


class TestRound14Networks(unittest.TestCase):
    def test_actor_critic_shapes_and_gradients(self):
        encoder = HedgeTemporalEncoder(market_width=5, window_size=8, hidden_dim=32)
        model = HedgeActorCriticNetwork(encoder, action_count=21)
        x = torch.randn(4, 8 * 5 + 12)
        logits, values = model(x)
        self.assertEqual(logits.shape, (4, 21))
        self.assertEqual(values.shape, (4,))
        loss = logits.square().mean() + values.square().mean()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_multitask_controls_are_bounded(self):
        encoder = HedgeTemporalEncoder(market_width=3, window_size=4, hidden_dim=24)
        model = HedgeMultiTaskNetwork(encoder)
        controls = model.controls(torch.randn(2, 4 * 3 + 12))
        self.assertTrue(torch.all((controls["long_score"] >= 0) & (controls["long_score"] <= 1)))
        self.assertTrue(
            torch.all(
                (controls["target_net_ratio"] >= -1)
                & (controls["target_net_ratio"] <= 1)
            )
        )
        self.assertTrue(torch.all(controls["future_volatility"] >= 0))

    def test_freqai_mlp_multioutput(self):
        model = HedgeMultiTaskMLP(10, output_dim=5, hidden_dim=16, n_layer=2)
        self.assertEqual(model(torch.randn(7, 10)).shape, (7, 5))


if __name__ == "__main__":
    unittest.main()
