from __future__ import annotations

import torch

from freqtrade.hedge.hprl.algorithms.rebrac_v2 import ConditionalFlowActor, ReBRACv2Agent
from freqtrade.hedge.hprl.config import HPRLTrainingConfig
from freqtrade.hedge.hprl.replay import ReplayBatch


def config() -> HPRLTrainingConfig:
    return HPRLTrainingConfig(
        device="cpu",
        batch_size=16,
        replay_capacity=64,
        warmup_steps=0,
        hidden_dim=32,
        hidden_depth=2,
        mixed_precision=False,
    )


def batch() -> ReplayBatch:
    return ReplayBatch(
        obs=torch.randn((16, 6)),
        action=torch.rand((16, 4)).clamp(0.05, 0.95),
        reward=torch.randn((16, 1)) * 0.01,
        next_obs=torch.randn((16, 6)),
        done=torch.zeros((16, 1)),
    )


def test_flow_actor_exact_log_prob_is_finite() -> None:
    actor = ConditionalFlowActor(6, 4, hidden_dim=32, flow_layers=4)
    obs = torch.randn((10, 6))
    action, log_prob, _ = actor.sample(obs)
    evaluated = actor.log_prob(obs, action)
    assert action.shape == (10, 4)
    assert log_prob.shape == (10, 1)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(evaluated).all()


def test_rebrac_multisample_action_selection() -> None:
    agent = ReBRACv2Agent(6, 4, config(), device="cpu")
    action = agent.act(torch.randn((3, 6)), samples=8)
    assert action.shape == (3, 4)
    assert torch.all((0 <= action) & (action <= 1))


def test_rebrac_staged_update_crosses_warmup() -> None:
    agent = ReBRACv2Agent(6, 4, config(), device="cpu")
    last = None
    for _ in range(agent.warmup_updates + 1):
        last = agent.update(batch()).values
    assert last is not None
    assert last["stage"] == 1.0
    assert torch.isfinite(torch.tensor(last["actor_loss"]))
