from __future__ import annotations

import pytest
import torch

from freqtrade.hedge.hprl.config import HPRLActionConfig, HPRLEnvironmentConfig
from freqtrade.hedge.hprl.data import TensorMarketDataset
from freqtrade.hedge.hprl.env import VectorizedHedgeEnv


def make_dataset() -> TensorMarketDataset:
    features = torch.zeros((8, 2, 3), dtype=torch.float32)
    returns = torch.tensor(
        [
            [0.01, -0.01],
            [0.02, 0.00],
            [-0.01, 0.01],
            [0.00, 0.01],
            [0.01, 0.01],
            [-0.02, -0.01],
            [0.01, 0.00],
            [0.00, 0.00],
        ],
        dtype=torch.float32,
    )
    funding = torch.zeros_like(returns)
    available = torch.full_like(returns, 100_000.0)
    return TensorMarketDataset(
        features,
        returns,
        funding,
        available,
        symbols=("BTC", "ETH"),
    ).validate()


def test_env_shapes_and_reset() -> None:
    cfg = HPRLEnvironmentConfig(parallel_envs=4)
    env = VectorizedHedgeEnv(make_dataset(), cfg)
    obs, info = env.reset()
    assert obs.shape == (4, env.observation_dim)
    assert env.action_dim == 4
    assert info["start_index"] == 0


def test_long_makes_money_on_positive_return_before_cost_effect() -> None:
    cfg = HPRLEnvironmentConfig(
        parallel_envs=1,
        action=HPRLActionConfig(max_step_change=1.0),
    )
    env = VectorizedHedgeEnv(make_dataset(), cfg)
    action = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
    step = env.step(action)
    assert step.info["equity"].item() > cfg.initial_equity
    assert step.reward.shape == (1,)


def test_short_makes_money_on_negative_return() -> None:
    cfg = HPRLEnvironmentConfig(
        parallel_envs=1,
        action=HPRLActionConfig(max_step_change=1.0),
    )
    env = VectorizedHedgeEnv(make_dataset(), cfg)
    # ETH forward return at t=0 is -1%, so a short leg should profit.
    action = torch.tensor([[0.0, 0.0, 0.0, 0.5]])
    step = env.step(action)
    assert step.info["equity"].item() > cfg.initial_equity


def test_hedged_long_short_can_coexist() -> None:
    cfg = HPRLEnvironmentConfig(
        parallel_envs=1,
        action=HPRLActionConfig(mode="continuous", max_step_change=1.0),
    )
    env = VectorizedHedgeEnv(make_dataset(), cfg, device="cpu")
    env.step(torch.tensor([[0.4, 0.2, 0.0, 0.0]]))
    position = env.position
    assert position[0, 0, 0] == pytest.approx(0.4)
    assert position[0, 0, 1] == pytest.approx(0.2)


def test_env_exhaustion_requires_reset() -> None:
    cfg = HPRLEnvironmentConfig(parallel_envs=1)
    env = VectorizedHedgeEnv(make_dataset(), cfg)
    action = torch.zeros((1, env.action_dim))
    for _ in range(7):
        step = env.step(action)
    assert bool(step.truncated.item())
    with pytest.raises(RuntimeError):
        env.step(action)


def test_dataset_shape_validation() -> None:
    bad = TensorMarketDataset(torch.zeros((3, 2, 1)), torch.zeros((3, 3)))
    with pytest.raises(ValueError):
        bad.validate()
