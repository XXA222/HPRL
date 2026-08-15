from __future__ import annotations

import pytest
import torch

from freqtrade.hedge.hprl.replay import TensorReplayBuffer


def test_replay_add_and_sample() -> None:
    buffer = TensorReplayBuffer(32, 5, 2)
    obs = torch.randn((8, 5))
    action = torch.rand((8, 2))
    reward = torch.randn(8)
    next_obs = torch.randn((8, 5))
    done = torch.zeros(8)
    buffer.add(obs, action, reward, next_obs, done)
    assert len(buffer) == 8
    batch = buffer.sample(4)
    assert batch.obs.shape == (4, 5)
    assert batch.action.shape == (4, 2)
    assert batch.reward.shape == (4, 1)


def test_replay_wraps_capacity() -> None:
    buffer = TensorReplayBuffer(5, 2, 1)
    for _ in range(3):
        buffer.add(
            torch.randn((4, 2)),
            torch.rand((4, 1)),
            torch.randn(4),
            torch.randn((4, 2)),
            torch.zeros(4),
        )
    assert len(buffer) == 5


def test_replay_rejects_undersized_sample() -> None:
    buffer = TensorReplayBuffer(10, 2, 1)
    with pytest.raises(ValueError):
        buffer.sample(1)
