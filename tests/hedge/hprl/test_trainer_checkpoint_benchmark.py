from __future__ import annotations

import torch

from freqtrade.hedge.hprl.algorithms.fast_td3 import FastTD3Agent
from freqtrade.hedge.hprl.benchmark import benchmark_environment
from freqtrade.hedge.hprl.checkpoint import load_checkpoint, save_checkpoint
from freqtrade.hedge.hprl.config import HPRLActionConfig, HPRLEnvironmentConfig, HPRLTrainingConfig
from freqtrade.hedge.hprl.data import TensorMarketDataset
from freqtrade.hedge.hprl.env import VectorizedHedgeEnv
from freqtrade.hedge.hprl.trainer import OnlineTrainer


def make_env(envs: int = 4) -> VectorizedHedgeEnv:
    torch.manual_seed(0)
    features = torch.randn((20, 1, 4)) * 0.01
    returns = torch.randn((20, 1)) * 0.002
    dataset = TensorMarketDataset(features, returns, symbols=("BTC",)).validate()
    config = HPRLEnvironmentConfig(
        parallel_envs=envs,
        action=HPRLActionConfig(max_step_change=1.0),
    )
    return VectorizedHedgeEnv(dataset, config, device="cpu")


def make_training() -> HPRLTrainingConfig:
    return HPRLTrainingConfig(
        device="cpu",
        batch_size=8,
        replay_capacity=128,
        warmup_steps=8,
        hidden_dim=32,
        hidden_depth=2,
        mixed_precision=False,
    )


def test_online_trainer_runs_updates() -> None:
    env = make_env()
    config = make_training()
    agent = FastTD3Agent(env.observation_dim, env.action_dim, config, device="cpu")
    summary = OnlineTrainer(env, agent, config).run(8)
    assert summary.environment_steps == 8
    assert summary.transitions == 32
    assert summary.updates > 0
    assert summary.final_equity_mean > 0


def test_checkpoint_roundtrip(tmp_path) -> None:
    env = make_env()
    config = make_training()
    agent = FastTD3Agent(env.observation_dim, env.action_dim, config, device="cpu")
    before = [value.detach().clone() for value in agent.actor.parameters()]
    path = save_checkpoint(tmp_path / "agent.pt", agent, {"algorithm": "fast_td3", "version": 1})
    for value in agent.actor.parameters():
        value.data.zero_()
    metadata = load_checkpoint(path, agent)
    after = list(agent.actor.parameters())
    assert metadata["algorithm"] == "fast_td3"
    assert all(torch.allclose(left, right) for left, right in zip(before, after, strict=True))


def test_environment_microbenchmark_reports_throughput() -> None:
    result = benchmark_environment(make_env(), steps=5)
    assert result.transitions == 20
    assert result.transitions_per_second > 0


class _FakeScaler:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.scale}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.scale = float(state["scale"])


def test_checkpoint_roundtrip_restores_amp_scaler_state(tmp_path) -> None:
    env = make_env()
    config = make_training()
    agent = FastTD3Agent(env.observation_dim, env.action_dim, config, device="cpu")
    agent.precision.scaler = _FakeScaler(65536.0)
    path = save_checkpoint(tmp_path / "amp-agent.pt", agent, {"algorithm": "fast_td3"})
    agent.precision.scaler.scale = 1.0
    load_checkpoint(path, agent)
    assert agent.precision.scaler.scale == 65536.0
