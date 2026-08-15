from __future__ import annotations

import pytest
import torch

from freqtrade.hedge.hprl.algorithms.fast_dsac import FastDSACAgent
from freqtrade.hedge.hprl.algorithms.fast_td3 import FastTD3Agent
from freqtrade.hedge.hprl.algorithms.simba_sac import SimbaSACAgent
from freqtrade.hedge.hprl.algorithms.xqc import XQCAgent
from freqtrade.hedge.hprl.config import HPRLTrainingConfig
from freqtrade.hedge.hprl.replay import ReplayBatch


def make_config() -> HPRLTrainingConfig:
    return HPRLTrainingConfig(
        device="cpu",
        batch_size=16,
        replay_capacity=64,
        warmup_steps=0,
        hidden_dim=32,
        hidden_depth=2,
        gradient_clip_norm=5.0,
        mixed_precision=False,
    )


def make_batch(obs_dim: int = 7, action_dim: int = 4, batch: int = 16) -> ReplayBatch:
    return ReplayBatch(
        obs=torch.randn((batch, obs_dim)),
        action=torch.rand((batch, action_dim)),
        reward=torch.randn((batch, 1)) * 0.02,
        next_obs=torch.randn((batch, obs_dim)),
        done=torch.zeros((batch, 1)),
    )


@pytest.mark.parametrize(
    "agent_cls",
    [FastTD3Agent, XQCAgent, SimbaSACAgent, FastDSACAgent],
)
def test_agent_action_shape_and_bounds(agent_cls) -> None:
    agent = agent_cls(7, 4, make_config(), device="cpu")
    action = agent.act(torch.randn((5, 7)), deterministic=True)
    assert action.shape == (5, 4)
    assert torch.all((0 <= action) & (action <= 1))


@pytest.mark.parametrize(
    "agent_cls",
    [FastTD3Agent, XQCAgent, SimbaSACAgent, FastDSACAgent],
)
def test_agent_update_is_finite(agent_cls) -> None:
    torch.manual_seed(123)
    agent = agent_cls(7, 4, make_config(), device="cpu")
    batch = make_batch()
    metrics = agent.update(batch).values
    assert metrics
    assert all(torch.isfinite(torch.tensor(float(value))) for value in metrics.values())


def test_fast_td3_delayed_actor_updates() -> None:
    agent = FastTD3Agent(7, 4, make_config(), device="cpu")
    first = agent.update(make_batch()).values
    second = agent.update(make_batch()).values
    assert first["actor_loss"] == 0.0
    assert second["actor_loss"] != 0.0


def test_xqc_critic_has_weight_norm_parameters() -> None:
    agent = XQCAgent(7, 4, make_config(), device="cpu")
    names = tuple(name for name, _ in agent.critic.named_parameters())
    assert any("parametrizations.weight" in name for name in names)


def test_fast_dsac_uses_per_dimension_temperature() -> None:
    agent = FastDSACAgent(7, 4, make_config(), device="cpu")
    assert tuple(agent.log_alpha.shape) == (4,)
