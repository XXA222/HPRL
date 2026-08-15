from __future__ import annotations

from pathlib import Path

import pytest
import torch

from freqtrade.hedge.hprl.action_space import (
    TieredHedgeActionCodec,
    gaussian_tier_probabilities,
)
from freqtrade.hedge.hprl.algorithms.base import soft_update
from freqtrade.hedge.hprl.algorithms.fast_dsac import FastDSACAgent
from freqtrade.hedge.hprl.algorithms.fast_td3 import FastTD3Agent
from freqtrade.hedge.hprl.algorithms.rebrac_v2 import ReBRACv2Agent
from freqtrade.hedge.hprl.algorithms.simba_sac import SimbaSACAgent
from freqtrade.hedge.hprl.algorithms.xqc import XQCAgent
from freqtrade.hedge.hprl.config import (
    HPRLActionConfig,
    HPRLConfig,
    HPRLEnvironmentConfig,
    HPRLTrainingConfig,
)
from freqtrade.hedge.hprl.data import TensorMarketDataset
from freqtrade.hedge.hprl.env import VectorizedHedgeEnv
from freqtrade.hedge.hprl.performance import (
    configure_training_runtime,
    make_adam,
    resolve_optimizer_backend,
    suggested_cpu_threads,
    timed_iterations,
)
from freqtrade.hedge.hprl.replay import ReplayBatch, TensorReplayBuffer


def _cfg(**values):
    defaults = dict(
        device="cpu",
        hidden_dim=16,
        hidden_depth=1,
        batch_size=8,
        replay_capacity=64,
        warmup_steps=0,
        runtime_checks=False,
        metrics_interval=1000,
        optimizer_backend="foreach",
    )
    defaults.update(values)
    return HPRLTrainingConfig(**defaults)


def _batch(obs_dim=7, action_dim=4, batch=8):
    return ReplayBatch(
        obs=torch.randn(batch, obs_dim),
        action=torch.rand(batch, action_dim),
        reward=torch.randn(batch, 1) * 0.01,
        next_obs=torch.randn(batch, obs_dim),
        done=torch.zeros(batch, 1),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("optimizer_backend", "auto"),
        ("optimizer_backend", "fused"),
        ("optimizer_backend", "foreach"),
        ("optimizer_backend", "for_loop"),
        ("compile_mode", "off"),
        ("compile_mode", "auto"),
        ("compile_mode", "default"),
        ("compile_mode", "reduce-overhead"),
        ("compile_mode", "max-autotune"),
        ("compile_mode", "max-autotune-no-cudagraphs"),
    ],
)
def test_perf_config_valid(field, value):
    cfg = _cfg(**{field: value})
    assert getattr(cfg, field) == value


@pytest.mark.parametrize(
    "values",
    [
        {"optimizer_backend": "bad"},
        {"compile_mode": "fastest"},
        {"cpu_threads": -1},
        {"cpu_interop_threads": -1},
        {"compile_dynamic": 1},
        {"compile_fullgraph": 1},
        {"polyak_backend": "bad"},
        {"grad_clip_backend": "bad"},
        {"replay_capacity": 4, "batch_size": 8},
        {"return_scan_backend": "bad"},
    ],
)
def test_perf_config_invalid(values):
    with pytest.raises(Exception):
        _cfg(**values)


@pytest.mark.parametrize("case", range(10))
def test_optimizer_backend_cpu(case):
    backend = "foreach" if case % 2 == 0 else "for_loop"
    layer = torch.nn.Linear(8, 4)
    opt = make_adam(layer.parameters(), lr=3e-4, device="cpu", backend=backend)
    assert getattr(opt, "_hprl_backend") == backend
    x = torch.randn(8, 8)
    layer(x).square().mean().backward()
    opt.step()


@pytest.mark.parametrize("width", range(4, 14))
def test_foreach_polyak_matches_formula(width):
    src = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.LayerNorm(width))
    dst = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.LayerNorm(width))
    before = [p.detach().clone() for p in dst.parameters()]
    source = [p.detach().clone() for p in src.parameters()]
    tau = 0.125
    soft_update(dst, src, tau)
    for old, sp, now in zip(before, source, dst.parameters(), strict=True):
        torch.testing.assert_close(now, old.lerp(sp, tau))


@pytest.mark.parametrize("capacity", range(11, 21))
def test_replay_contiguous_write(capacity):
    buf = TensorReplayBuffer(capacity, 3, 2, device="cpu", pin_memory=False)
    n = min(7, capacity)
    obs = torch.arange(n * 3, dtype=torch.float32).reshape(n, 3)
    act = torch.arange(n * 2, dtype=torch.float32).reshape(n, 2)
    rew = torch.arange(n, dtype=torch.float32)
    buf.add(obs, act, rew, obs + 1, torch.zeros(n))
    torch.testing.assert_close(buf.obs[:n], obs)
    torch.testing.assert_close(buf.action[:n], act)


@pytest.mark.parametrize("offset", range(10))
def test_replay_wrap_write(offset):
    capacity = 13
    buf = TensorReplayBuffer(capacity, 2, 1, device="cpu", pin_memory=False)
    first = 9 + (offset % 3)
    x = torch.arange(first * 2, dtype=torch.float32).reshape(first, 2)
    buf.add(x, x[:, :1], torch.zeros(first), x, torch.zeros(first))
    y = torch.full((7, 2), float(100 + offset))
    buf.add(y, y[:, :1], torch.zeros(7), y, torch.zeros(7))
    assert len(buf) == capacity
    assert int((buf.obs == float(100 + offset)).all(dim=1).sum()) == 7


@pytest.mark.parametrize("dim", range(1, 11))
def test_gaussian_actor_deterministic_fast_path(dim):
    from freqtrade.hedge.hprl.networks import GaussianActor

    actor = GaussianActor(5, dim, hidden_dim=16, depth=1)
    obs = torch.randn(6, 5)
    action1, logp1, mean1 = actor.sample(obs, deterministic=True, per_dim_log_prob=True)
    action2, logp2, mean2 = actor.sample(obs, deterministic=True, per_dim_log_prob=True)
    torch.testing.assert_close(action1, action2)
    torch.testing.assert_close(mean1, mean2)
    torch.testing.assert_close(logp1, logp2)


@pytest.mark.parametrize("dim", range(1, 11))
def test_gaussian_actor_stochastic_fast_path(dim):
    from freqtrade.hedge.hprl.networks import GaussianActor

    actor = GaussianActor(5, dim, hidden_dim=16, depth=1)
    compute_log_prob = dim % 2 == 0
    action, logp, mean = actor.sample(
        torch.randn(8, 5), per_dim_log_prob=True, compute_log_prob=compute_log_prob
    )
    assert action.shape == (8, dim)
    if compute_log_prob:
        assert logp.shape == (8, dim) and torch.isfinite(logp).all()
    else:
        assert logp is None
    assert torch.isfinite(action).all() and torch.isfinite(mean).all()


@pytest.mark.parametrize("levels", range(2, 12))
def test_tier_tensor_cache_and_probability(levels):
    mean = torch.zeros(4, 3)
    log_std = torch.zeros_like(mean)
    p1 = gaussian_tier_probabilities(mean, log_std, levels)
    p2 = gaussian_tier_probabilities(mean, log_std, levels)
    torch.testing.assert_close(p1, p2)
    torch.testing.assert_close(p1.sum(-1), torch.ones_like(p1.sum(-1)))


@pytest.mark.parametrize("envs", range(1, 11))
def test_env_reused_tail_buffers(envs):
    features = torch.randn(40, 2, 3)
    returns = torch.randn(40, 2) * 0.001
    market = TensorMarketDataset(features=features, forward_returns=returns)
    config = HPRLEnvironmentConfig(parallel_envs=envs, action=HPRLActionConfig())
    env = VectorizedHedgeEnv(market, config, device="cpu")
    obs, _ = env.reset()
    sortable_id = id(env._tail_sortable)
    for _ in range(3):
        step = env.step(torch.zeros(envs, 4))
        obs = step.observation
    assert id(env._tail_sortable) == sortable_id
    assert obs.shape[0] == envs


@pytest.mark.parametrize("case", range(10))
def test_runtime_performance_info(case):
    cfg = _cfg(
        optimizer_backend="auto",
        cpu_threads=0,
        cpu_interop_threads=max(1, torch.get_num_interop_threads()),
    )
    info = configure_training_runtime(cfg, "cpu")
    assert info.device == "cpu"
    assert info.optimizer_backend == "for_loop"
    assert info.cpu_threads >= 1
    assert case >= 0


@pytest.mark.parametrize("iterations", range(1, 11))
def test_timing_helper(iterations):
    x = torch.ones(16)
    result = timed_iterations(lambda: x.add_(1), warmup=1, iterations=iterations, device="cpu")
    assert result["iterations"] == iterations
    assert result["seconds"] >= 0
    assert result["iterations_per_second"] > 0


def _agent_update(cls, seed):
    torch.manual_seed(seed)
    cfg = _cfg()
    agent = cls(7, 4, cfg, device="cpu")
    metrics = agent.update(_batch(), collect_metrics=False)
    assert metrics.values == {}
    assert getattr(agent.actor_opt, "_hprl_backend") == "foreach"
    assert getattr(agent.critic_opt, "_hprl_backend") == "foreach"


@pytest.mark.parametrize("seed", range(10))
def test_fast_td3_perf_update(seed):
    _agent_update(FastTD3Agent, seed)


@pytest.mark.parametrize("seed", range(10))
def test_fast_dsac_perf_update(seed):
    _agent_update(FastDSACAgent, seed)


@pytest.mark.parametrize("seed", range(10))
def test_simba_perf_update(seed):
    _agent_update(SimbaSACAgent, seed)


@pytest.mark.parametrize("seed", range(10))
def test_xqc_perf_update(seed):
    _agent_update(XQCAgent, seed)


@pytest.mark.parametrize("seed", range(10))
def test_rebrac_perf_update(seed):
    _agent_update(ReBRACv2Agent, seed)


@pytest.mark.parametrize("batch", range(4, 14))
def test_replay_sampling_parity(batch):
    capacity = 64
    buf = TensorReplayBuffer(capacity, 3, 2, device="cpu", pin_memory=False)
    obs = torch.arange(capacity * 3, dtype=torch.float32).reshape(capacity, 3)
    buf.add(obs, obs[:, :2], torch.arange(capacity).float(), obs + 1, torch.zeros(capacity))
    sample = buf.sample_reusable(batch) if batch % 2 == 0 else buf.sample(batch)
    assert sample.obs.shape == (batch, 3)
    assert sample.action.shape == (batch, 2)
    assert torch.isfinite(sample.obs).all()
    if batch % 2 == 0:
        pointer = sample.obs.data_ptr()
        assert buf.sample_reusable(batch).obs.data_ptr() == pointer


@pytest.mark.parametrize("case", range(10))
def test_performance_source_invariants(case):
    root = Path(__file__).resolve().parents[3]
    base = (root / "freqtrade/hedge/hprl/algorithms/base.py").read_text()
    replay = (root / "freqtrade/hedge/hprl/replay.py").read_text()
    networks = (root / "freqtrade/hedge/hprl/networks.py").read_text()
    action = (root / "freqtrade/hedge/hprl/action_space.py").read_text()
    invariants = (
        "torch._foreach_lerp_",
        "torch.cat((obs, action, reward, next_obs, done)",
        "_reparameterized_sigmoid_gaussian",
        "TierBoundaryBuffers",
        "optimizer_backend",
        "compile_mode",
        "cpu_threads",
        "resolve_polyak_foreach",
        "pre_equity = self._equity",
        "self._tail_sortable.copy_",
    )
    corpus = "\n".join(
        (
            base,
            replay,
            networks,
            action,
            (root / "freqtrade/hedge/hprl/config.py").read_text(),
            (root / "freqtrade/hedge/hprl/device.py").read_text(),
            (root / "freqtrade/hedge/hprl/env.py").read_text(),
            (root / "freqtrade/hedge/hprl/performance.py").read_text(),
        )
    )
    assert invariants[case] in corpus


@pytest.mark.parametrize("case", range(10))
def test_config_mapping_performance_fields(case):
    mappings = [
        {"optimizer_backend": "foreach"},
        {"compile_mode": "off"},
        {"compile_dynamic": False},
        {"compile_fullgraph": False},
        {"cpu_threads": 0},
        {"cpu_interop_threads": 1},
        {"polyak_backend": "auto"},
        {"grad_clip_backend": "auto"},
        {"replay_reuse_sample_buffers": True},
        {"return_scan_backend": "auto"},
    ]
    cfg = HPRLConfig.from_mapping({"training": mappings[case]})
    key, value = next(iter(mappings[case].items()))
    assert getattr(cfg.training, key) == value

