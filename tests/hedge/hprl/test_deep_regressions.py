from __future__ import annotations

import json

import pytest
import torch

from freqtrade.hedge.hprl.adapters import ReadonlyTargetAdapter
from freqtrade.hedge.hprl.algorithms.fast_td3 import FastTD3Agent
from freqtrade.hedge.hprl.algorithms.rebrac_v2 import ReBRACv2Agent
from freqtrade.hedge.hprl.algorithms.xqc import XQCAgent
from freqtrade.hedge.hprl.checkpoint import load_checkpoint, save_checkpoint
from freqtrade.hedge.hprl.config import (
    HPRLActionConfig,
    HPRLCostConfig,
    HPRLEnvironmentConfig,
    HPRLRewardConfig,
    HPRLTrainingConfig,
)
from freqtrade.hedge.hprl.contracts import OfflineTransition
from freqtrade.hedge.hprl.costs import ExecutionCostModel
from freqtrade.hedge.hprl.data import OfflineTransitionDataset, TensorMarketDataset
from freqtrade.hedge.hprl.ensemble import GaussianStateBoundary, RiskAwareEnsembleRouter
from freqtrade.hedge.hprl.env import VectorizedHedgeEnv
from freqtrade.hedge.hprl.evaluation import evaluate_trading
from freqtrade.hedge.hprl.networks import CategoricalTwinCritic, HypersphericalLinear
from freqtrade.hedge.hprl.regime import AdaptiveRegimeDetector, RegimeSignature
from freqtrade.hedge.hprl.replay import ReplayBatch, TensorReplayBuffer
from freqtrade.hedge.hprl.reward import CompositeReward, RewardFactsTensor
from freqtrade.hedge.hprl.risk import HedgeActionProjector
from freqtrade.hedge.hprl.trainer import OfflineTrainer, OnlineTrainer


def small_training(**overrides) -> HPRLTrainingConfig:
    values = {
        "device": "cpu",
        "batch_size": 8,
        "replay_capacity": 64,
        "warmup_steps": 0,
        "hidden_dim": 32,
        "hidden_depth": 2,
        "mixed_precision": False,
    }
    values.update(overrides)
    return HPRLTrainingConfig(**values)


def random_batch(obs_dim: int = 6, action_dim: int = 4, batch: int = 8) -> ReplayBatch:
    return ReplayBatch(
        obs=torch.randn(batch, obs_dim),
        action=torch.rand(batch, action_dim),
        reward=torch.randn(batch, 1) * 0.01,
        next_obs=torch.randn(batch, obs_dim),
        done=torch.zeros(batch, 1),
    )


def market_dataset(length: int = 12, symbols: int = 1, *, dtype=torch.float32):
    features = torch.zeros((length, symbols, 3), dtype=dtype)
    returns = torch.full((length, symbols), 0.001, dtype=dtype)
    available = torch.full((length, symbols), 100_000.0, dtype=dtype)
    names = tuple(f"S{index}" for index in range(symbols))
    return TensorMarketDataset(features, returns, available_notional=available, symbols=names)


def test_training_defaults_to_auto_device() -> None:
    assert HPRLTrainingConfig().device == "auto"
    assert HPRLTrainingConfig().replay_device == "auto"
    assert HPRLTrainingConfig().mixed_precision is False
    assert HPRLTrainingConfig().allow_tf32 is True


def test_config_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        HPRLActionConfig(max_leg_exposure=float("nan"))
    with pytest.raises(ValueError):
        HPRLCostConfig(max_participation=float("inf"))
    with pytest.raises(ValueError):
        HPRLRewardConfig(equity=float("nan"))


def test_config_rejects_noninteger_dimensions() -> None:
    with pytest.raises(ValueError):
        HPRLEnvironmentConfig(parallel_envs=2.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        HPRLTrainingConfig(batch_size=8.5)  # type: ignore[arg-type]


def test_dataset_is_cast_to_float32_by_environment() -> None:
    env = VectorizedHedgeEnv(
        market_dataset(dtype=torch.float64),
        HPRLEnvironmentConfig(parallel_envs=2),
    )
    observation, _ = env.reset()
    assert env.dataset.features.dtype == torch.float32
    assert env.dataset.forward_returns.dtype == torch.float32
    assert observation.dtype == torch.float32


def test_dataset_rejects_duplicate_symbols() -> None:
    data = TensorMarketDataset(
        torch.zeros((3, 2, 1)),
        torch.zeros((3, 2)),
        symbols=("BTC", "BTC"),
    )
    with pytest.raises(ValueError):
        data.validate()


def test_dataset_rejects_zero_available_notional() -> None:
    data = TensorMarketDataset(
        torch.zeros((3, 1, 1)),
        torch.zeros((3, 1)),
        available_notional=torch.zeros((3, 1)),
    )
    with pytest.raises(ValueError):
        data.validate()


def test_jsonl_done_requires_real_boolean(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "observation": [0.0],
                "action": [0.0],
                "reward": 0.0,
                "next_observation": [0.0],
                "done": "false",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        OfflineTransitionDataset.from_jsonl(path)


def test_market_impact_keeps_increasing_above_participation_threshold() -> None:
    model = ExecutionCostModel(HPRLCostConfig(max_participation=0.05))
    low = model.evaluate(
        turnover_notional=torch.tensor([100.0]),
        equity=torch.tensor([1000.0]),
        available_notional=torch.tensor([1000.0]),
    )
    high = model.evaluate(
        turnover_notional=torch.tensor([500.0]),
        equity=torch.tensor([1000.0]),
        available_notional=torch.tensor([1000.0]),
    )
    assert high.market_impact.item() > low.market_impact.item()


def test_cost_model_rejects_invalid_inputs() -> None:
    model = ExecutionCostModel()
    with pytest.raises(ValueError):
        model.evaluate(turnover_notional=torch.tensor([-1.0]), equity=torch.tensor([1000.0]))
    with pytest.raises(ValueError):
        model.evaluate(turnover_notional=torch.tensor([1.0]), equity=torch.tensor([0.0]))


def test_multisymbol_net_projection_is_hard_bounded() -> None:
    config = HPRLActionConfig(
        max_leg_exposure=1.0,
        max_gross_exposure=3.0,
        max_abs_net_exposure=0.25,
        max_step_change=1.0,
    )
    projector = HedgeActionProjector(config)
    current = torch.zeros((2, 4, 2))
    raw = torch.zeros_like(current)
    raw[..., 0] = 1.0
    target = projector.project(raw, current).target
    gross = target.sum(dim=(-2, -1))
    net = (target[..., 0] - target[..., 1]).sum(dim=-1).abs()
    assert torch.all(gross <= config.max_gross_exposure + 1e-6)
    assert torch.all(net <= config.max_abs_net_exposure + 1e-6)


def test_projection_sanitizes_policy_nan_but_rejects_corrupt_current() -> None:
    projector = HedgeActionProjector(HPRLActionConfig(max_step_change=1.0))
    current = torch.zeros((1, 1, 2))
    result = projector.project(torch.tensor([[[float("nan"), float("inf")]]]), current)
    assert torch.isfinite(result.target).all()
    with pytest.raises(ValueError):
        projector.project(torch.zeros_like(current), torch.full_like(current, float("nan")))


def test_default_reward_does_not_double_count_execution_costs() -> None:
    model = CompositeReward()
    facts = RewardFactsTensor(
        equity_return=torch.tensor([0.01]),
        drawdown_increase=torch.tensor([0.0]),
        downside_return=torch.tensor([0.01]),
        cvar_loss=torch.tensor([0.0]),
        turnover_ratio=torch.tensor([0.0]),
        fee_ratio=torch.tensor([0.01]),
        slippage_ratio=torch.tensor([0.01]),
        impact_ratio=torch.tensor([0.01]),
        funding_ratio=torch.tensor([0.01]),
        projected=torch.tensor([False]),
    )
    _, components = model.evaluate_tensor(facts)
    assert components["fees"].item() == pytest.approx(0.0)
    assert components["slippage"].item() == pytest.approx(0.0)
    assert components["market_impact"].item() == pytest.approx(0.0)
    assert components["funding"].item() == pytest.approx(0.0)


def test_environment_reports_executed_projected_action() -> None:
    config = HPRLEnvironmentConfig(
        parallel_envs=1,
        action=HPRLActionConfig(
            max_gross_margin_ratio=0.20,
            max_abs_net_margin_ratio=0.20,
            max_increase_levels=4,
        ),
    )
    env = VectorizedHedgeEnv(market_dataset(), config)
    step = env.step(torch.ones((1, env.action_dim)))
    assert step.info["target_margin"].sum().item() <= 0.20 + 1e-6
    stored = step.info["executed_action"]
    assert torch.allclose(stored * 4.0, torch.round(stored * 4.0))


def test_replay_rejects_nonfinite_transition_and_zero_sample() -> None:
    buffer = TensorReplayBuffer(16, 2, 1)
    with pytest.raises(ValueError):
        buffer.add(
            torch.tensor([[float("nan"), 0.0]]),
            torch.zeros((1, 1)),
            torch.zeros(1),
            torch.zeros((1, 2)),
            torch.zeros(1),
        )
    with pytest.raises(ValueError):
        buffer.sample(0)


def test_hyperspherical_scale_is_numerically_bounded() -> None:
    layer = HypersphericalLinear(4, 3)
    layer.log_scale.data.fill_(1_000_000.0)
    output = layer(torch.randn((8, 4)))
    assert torch.isfinite(output).all()


def test_categorical_critic_validates_support_and_accepts_1d_target() -> None:
    with pytest.raises(ValueError):
        CategoricalTwinCritic(3, 2, bins=1)
    critic = CategoricalTwinCritic(3, 2, hidden_dim=16, depth=1)
    probs = critic.project_scalar(torch.tensor([0.1, -0.2]))
    assert probs.shape == (2, critic.bins)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2))


def test_rebrac_deterministic_action_really_is_deterministic() -> None:
    agent = ReBRACv2Agent(6, 4, small_training(), device="cpu")
    obs = torch.randn((3, 6))
    first = agent.act(obs, deterministic=True, samples=16)
    second = agent.act(obs, deterministic=True, samples=16)
    assert torch.allclose(first, second)


def test_rebrac_stochastic_multisample_action_can_change() -> None:
    agent = ReBRACv2Agent(6, 4, small_training(), device="cpu")
    obs = torch.randn((3, 6))
    first = agent.act(obs, deterministic=False, samples=8)
    second = agent.act(obs, deterministic=False, samples=8)
    assert not torch.allclose(first, second)


def test_checkpoint_restores_td3_targets_optimizer_and_counter(tmp_path) -> None:
    config = small_training()
    agent = FastTD3Agent(6, 4, config, device="cpu")
    agent.update(random_batch())
    agent.update(random_batch())
    expected_target = {key: value.clone() for key, value in agent.actor_target.state_dict().items()}
    expected_count = agent.update_count
    expected_opt_steps = len(agent.actor_opt.state_dict()["state"])
    path = save_checkpoint(tmp_path / "agent.pt", agent, {"algorithm": "fast_td3"})
    for value in agent.actor_target.parameters():
        value.data.zero_()
    agent.update_count = 999
    agent.actor_opt.state.clear()
    load_checkpoint(path, agent)
    assert agent.update_count == expected_count
    assert len(agent.actor_opt.state_dict()["state"]) == expected_opt_steps
    assert all(
        torch.allclose(value, expected_target[key])
        for key, value in agent.actor_target.state_dict().items()
    )


def test_checkpoint_restores_sac_temperature(tmp_path) -> None:
    agent = XQCAgent(6, 4, small_training(), device="cpu")
    agent.update(random_batch())
    expected = agent.log_alpha.detach().clone()
    path = save_checkpoint(tmp_path / "xqc.pt", agent, {"algorithm": "xqc"})
    agent.log_alpha.data.fill_(9.0)
    load_checkpoint(path, agent)
    assert torch.allclose(agent.log_alpha.detach(), expected)


def test_checkpoint_can_restore_torch_rng(tmp_path) -> None:
    agent = FastTD3Agent(6, 4, small_training(), device="cpu")
    torch.manual_seed(1234)
    path = save_checkpoint(tmp_path / "rng.pt", agent, {"algorithm": "fast_td3"})
    expected = torch.rand(8)
    _ = torch.rand(100)
    load_checkpoint(path, agent, restore_rng=True)
    actual = torch.rand(8)
    assert torch.allclose(actual, expected)


def test_checkpoint_rejects_non_json_metadata_before_write(tmp_path) -> None:
    agent = FastTD3Agent(6, 4, small_training(), device="cpu")
    path = tmp_path / "bad.pt"
    with pytest.raises(ValueError):
        save_checkpoint(path, agent, {"bad": object()})
    assert not path.exists()


def test_online_trainer_warmup_can_exceed_replay_capacity() -> None:
    env = VectorizedHedgeEnv(
        market_dataset(length=20),
        HPRLEnvironmentConfig(parallel_envs=4, action=HPRLActionConfig(max_step_change=1.0)),
        device="cpu",
    )
    config = small_training(replay_capacity=16, warmup_steps=40)
    agent = FastTD3Agent(env.observation_dim, env.action_dim, config, device="cpu")
    summary = OnlineTrainer(env, agent, config).run(12)
    assert summary.transitions == 48
    assert summary.updates > 0


class ConstantAgent:
    def __init__(self, action_dim: int, value: float | tuple[float, ...]) -> None:
        self.device = torch.device("cpu")
        self.action_dim = action_dim
        self.value = value

    def act(self, obs, *, deterministic: bool = False):
        if isinstance(self.value, tuple):
            row = torch.tensor(self.value, dtype=obs.dtype, device=obs.device)
            return row.reshape(1, -1).expand(obs.shape[0], -1).clone()
        return torch.full((obs.shape[0], self.action_dim), self.value, dtype=obs.dtype)

    def update(self, batch):  # pragma: no cover - this test deliberately avoids updates
        raise AssertionError("update should not run")


def test_online_summary_keeps_terminal_equity_after_group_reset() -> None:
    data = TensorMarketDataset(
        torch.zeros((3, 1, 2)),
        torch.full((3, 1), 0.01),
        symbols=("BTC",),
    )
    env = VectorizedHedgeEnv(
        data,
        HPRLEnvironmentConfig(
            parallel_envs=1,
            costs=HPRLCostConfig(
                maker_fee_bps=0.0,
                taker_fee_bps=0.0,
                base_slippage_bps=0.0,
                impact_coefficient_bps=0.0,
            ),
            action=HPRLActionConfig(max_step_change=1.0),
        ),
        device="cpu",
    )
    config = small_training(batch_size=16, replay_capacity=32)
    summary = OnlineTrainer(env, ConstantAgent(env.action_dim, (0.5, 0.0)), config).run(2)
    assert summary.group_resets == 1
    assert summary.final_equity_mean > env.config.initial_equity
    assert env.equity.item() == pytest.approx(env.config.initial_equity)


def test_mixed_precision_config_is_available_but_requires_cuda_runtime() -> None:
    from freqtrade.hedge.hprl.device import PrecisionManager

    config = small_training(mixed_precision=True)
    assert config.mixed_precision is True
    with pytest.raises(ValueError):
        PrecisionManager("cpu", enabled=True, dtype="auto")


def test_offline_trainer_runs_real_updates() -> None:
    rows = [
        OfflineTransition(
            observation=(float(index), 0.0),
            action=(0.2, 0.1),
            reward=0.001,
            next_observation=(float(index + 1), 0.0),
            done=index == 15,
        )
        for index in range(16)
    ]
    dataset = OfflineTransitionDataset(rows)
    config = small_training(batch_size=8, replay_capacity=32)
    agent = ReBRACv2Agent(2, 2, config, device="cpu")
    summary = OfflineTrainer(dataset, agent, config).run(3)
    assert summary.updates == 3
    assert summary.samples_seen == 24
    assert summary.last_metrics["critic_loss"] >= 0


def test_all_positive_returns_have_zero_cvar_loss() -> None:
    metrics = evaluate_trading([100.0, 101.0, 102.0, 103.0], periods_per_year=365)
    assert metrics.cvar == pytest.approx(0.0)


def test_calmar_uses_geometric_annualization() -> None:
    values = [100.0, 105.0, 103.0, 110.0]
    metrics = evaluate_trading(values, periods_per_year=3)
    expected_annualized = values[-1] / values[0] - 1.0
    assert metrics.calmar == pytest.approx(expected_annualized / metrics.max_drawdown)


def test_default_regime_high_vol_threshold_is_decimal_return_scale() -> None:
    detector = AdaptiveRegimeDetector()
    returns = torch.tensor([0.02, -0.02, 0.02, -0.02])
    assert detector.label(returns) == "high_vol"


def test_regime_signature_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        RegimeSignature(float("nan"), 0.1, 0.1)


class ConstantPolicy:
    def __init__(self, value: float) -> None:
        self.value = value

    def act(self, obs, *, deterministic: bool = True):
        return torch.full((obs.shape[0], 2), self.value, device=obs.device, dtype=obs.dtype)


def test_negative_profit_specialist_does_not_override_conservative_policy() -> None:
    torch.manual_seed(4)
    states = torch.randn((100, 3)) * 0.1
    boundary = GaussianStateBoundary(quantile=0.99).fit(states)
    router = RiskAwareEnsembleRouter(ConstantPolicy(0.1))
    router.register("bad", ConstantPolicy(0.9), boundary, profitability=-1.0)
    action = router.act(torch.zeros((2, 3)))
    assert torch.allclose(action, torch.full((2, 2), 0.1))


def test_adapter_rejects_duplicate_symbols() -> None:
    with pytest.raises(ValueError):
        ReadonlyTargetAdapter(("BTC", "BTC"), "model")


def test_resolve_device_supports_auto_cpu_and_cuda(monkeypatch) -> None:
    from types import SimpleNamespace

    from freqtrade.hedge.hprl.device import resolve_device
    from freqtrade.hedge.hprl.errors import HPRLDependencyError

    auto = resolve_device()
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert auto.resolved.startswith("cuda:") if expected == "cuda" else auto.resolved == "cpu"
    assert resolve_device("cpu").resolved == "cpu"
    if torch.cuda.is_available():
        assert resolve_device("cuda").resolved.startswith("cuda:")
    else:
        with pytest.raises(HPRLDependencyError):
            resolve_device("cuda")

    # Exercise CUDA selection logic without pretending this CPU-only CI host can execute kernels.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(
        name="Mock RTX", total_memory=8 * 1024**3
    ))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    if hasattr(torch.cuda, "is_tf32_supported"):
        monkeypatch.setattr(torch.cuda, "is_tf32_supported", lambda: True)
    mocked = resolve_device("auto")
    assert mocked.resolved == "cuda:0"
    assert mocked.device_name == "Mock RTX"
    assert mocked.total_memory_bytes == 8 * 1024**3


def test_negative_seed_is_rejected() -> None:
    with pytest.raises(ValueError):
        HPRLTrainingConfig(seed=-1)


def test_registry_agent_initialization_is_seed_reproducible() -> None:
    from freqtrade.hedge.hprl.registry import create_agent

    config = small_training(seed=123)
    first = create_agent("fast_td3", 6, 4, config, device="cpu")
    second = create_agent("fast_td3", 6, 4, config, device="cpu")
    assert all(
        torch.equal(a, b)
        for a, b in zip(first.actor.state_dict().values(), second.actor.state_dict().values())
    )


def test_cost_model_promotes_integer_turnover() -> None:
    result = ExecutionCostModel().evaluate(
        turnover_notional=torch.tensor([10], dtype=torch.int64),
        equity=torch.tensor([1000], dtype=torch.int64),
    )
    assert result.total.dtype == torch.float32
    assert torch.isfinite(result.total).all()


def test_soft_update_synchronizes_batchnorm_buffers() -> None:
    from freqtrade.hedge.hprl.algorithms.base import soft_update

    source = torch.nn.BatchNorm1d(3)
    target = torch.nn.BatchNorm1d(3)
    source.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0]))
    target.running_mean.zero_()
    source.num_batches_tracked.fill_(7)
    target.num_batches_tracked.zero_()
    soft_update(target, source, 0.5)
    assert torch.allclose(target.running_mean, torch.tensor([0.5, 1.0, 1.5]))
    assert target.num_batches_tracked.item() == 7


def test_xqc_target_batchnorm_does_not_drift_on_target_forward() -> None:
    agent = XQCAgent(6, 4, small_training(), device="cpu")
    assert agent.critic_target.training is False
    before = {
        name: value.clone()
        for name, value in agent.critic_target.named_buffers()
        if "running_" in name or "num_batches_tracked" in name
    }
    with torch.no_grad():
        agent.critic_target(torch.randn(8, 6), torch.rand(8, 4))
    after = dict(agent.critic_target.named_buffers())
    assert all(torch.equal(value, after[name]) for name, value in before.items())


def test_online_replay_stores_executed_not_raw_action() -> None:
    env = VectorizedHedgeEnv(
        market_dataset(length=8),
        HPRLEnvironmentConfig(
            parallel_envs=1,
            action=HPRLActionConfig(
                max_gross_margin_ratio=0.20,
                max_abs_net_margin_ratio=0.20,
                max_increase_levels=4,
            ),
        ),
        device="cpu",
    )
    config = small_training(batch_size=8, replay_capacity=16, warmup_steps=0)
    agent = ConstantAgent(env.action_dim, 1.0)
    trainer = OnlineTrainer(env, agent, config)
    trainer.run(1)
    stored = trainer.buffer.action[0]
    assert torch.allclose(stored * 4.0, torch.round(stored * 4.0))
    assert not torch.allclose(stored, torch.ones_like(stored))
    assert env.margin_position.sum().item() <= 0.20 + 1e-6


def test_warmup_actions_sample_canonical_tier_codes() -> None:
    env = VectorizedHedgeEnv(
        market_dataset(length=8, symbols=3),
        HPRLEnvironmentConfig(parallel_envs=64),
        device="cpu",
    )
    config = small_training(batch_size=8, replay_capacity=128, warmup_steps=1000)
    trainer = OnlineTrainer(env, ConstantAgent(env.action_dim, 0.0), config)
    torch.manual_seed(11)
    action = trainer._warmup_action()
    scaled = action * float(env.action_level_count - 1)
    assert torch.allclose(scaled, torch.round(scaled))
    assert action.min().item() == pytest.approx(0.0)
    assert action.max().item() == pytest.approx(1.0)


def test_walk_forward_default_purge_separates_sets() -> None:
    from freqtrade.hedge.hprl.evaluation import walk_forward_folds

    fold = walk_forward_folds(100, train=40, validation=10, test=10)[0]
    assert fold.validation_start - fold.train_end == 1
    assert fold.test_start - fold.validation_end == 1


def test_walk_forward_supports_label_horizon_purge() -> None:
    from freqtrade.hedge.hprl.evaluation import walk_forward_folds

    fold = walk_forward_folds(120, train=40, validation=10, test=10, purge=5)[0]
    assert fold.validation_start - fold.train_end == 5
    assert fold.test_start - fold.validation_end == 5


def test_reward_rejects_shape_broadcasting() -> None:
    model = CompositeReward()
    facts = RewardFactsTensor(
        equity_return=torch.zeros(2),
        drawdown_increase=torch.zeros(1),
        downside_return=torch.zeros(2),
        cvar_loss=torch.zeros(2),
        turnover_ratio=torch.zeros(2),
        fee_ratio=torch.zeros(2),
        slippage_ratio=torch.zeros(2),
        impact_ratio=torch.zeros(2),
        funding_ratio=torch.zeros(2),
        projected=torch.zeros(2, dtype=torch.bool),
    )
    with pytest.raises(ValueError):
        model.evaluate_tensor(facts)


def test_ood_score_rejects_nonfinite_state() -> None:
    boundary = GaussianStateBoundary().fit(torch.randn(32, 3))
    with pytest.raises(ValueError):
        boundary.score(torch.tensor([[float("nan"), 0.0, 0.0]]))


def test_regime_policy_temperature_must_be_finite() -> None:
    from freqtrade.hedge.hprl.regime import PolicyLibrary

    library = PolicyLibrary()
    library.register("p", ConstantPolicy(0.1), RegimeSignature(0.0, 0.01, 0.0))
    with pytest.raises(ValueError):
        library.weights(RegimeSignature(0.0, 0.01, 0.0), temperature=float("nan"))


def test_offline_transition_done_is_strict_boolean() -> None:
    with pytest.raises(ValueError):
        OfflineTransition(
            observation=(0.0,),
            action=(0.0,),
            reward=0.0,
            next_observation=(0.0,),
            done=1,  # type: ignore[arg-type]
        )


def test_exact_market_neutral_net_limit_is_valid() -> None:
    config = HPRLActionConfig(
        max_leg_exposure=1.0,
        max_gross_exposure=2.0,
        max_abs_net_exposure=0.0,
        max_step_change=1.0,
    )
    projector = HedgeActionProjector(config)
    current = torch.zeros((1, 2, 2))
    raw = torch.tensor([[[0.8, 0.1], [0.6, 0.2]]])
    target = projector.project(raw, current).target
    net = target[..., 0].sum() - target[..., 1].sum()
    assert net.item() == pytest.approx(0.0, abs=1e-6)


def test_mapping_unknown_key_raises_hprl_config_error() -> None:
    from freqtrade.hedge.hprl.config import HPRLConfig
    from freqtrade.hedge.hprl.errors import HPRLConfigError

    with pytest.raises(HPRLConfigError):
        HPRLConfig.from_mapping({"environment": {"unknown_key": 1}})


def test_dataset_requires_torch_tensors() -> None:
    import numpy as np

    data = TensorMarketDataset(
        np.zeros((3, 1, 1)),  # type: ignore[arg-type]
        np.zeros((3, 1)),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError):
        data.validate()


def test_environment_does_not_fabricate_liquidation_distance() -> None:
    env = VectorizedHedgeEnv(market_dataset(), HPRLEnvironmentConfig(parallel_envs=1))
    step = env.step(torch.zeros((1, env.action_dim)))
    assert step.info["liquidation_distance_modeled"] is False


def test_checkpoint_rejects_wrong_agent_class(tmp_path) -> None:
    from freqtrade.hedge.hprl.algorithms.simba_sac import SimbaSACAgent

    config = small_training()
    source = FastTD3Agent(6, 4, config, device="cpu")
    target = SimbaSACAgent(6, 4, config, device="cpu")
    path = save_checkpoint(tmp_path / "wrong.pt", source, {})
    with pytest.raises(ValueError, match="agent class mismatch"):
        load_checkpoint(path, target)


def test_rebrac_flow_sample_and_log_prob_are_consistent() -> None:
    from freqtrade.hedge.hprl.algorithms.rebrac_v2 import ConditionalFlowActor

    torch.manual_seed(42)
    actor = ConditionalFlowActor(5, 7, hidden_dim=32, flow_layers=4)
    obs = torch.randn(16, 5)
    action, sample_log_prob, expanded = actor.sample(obs, samples=3)
    evaluated_log_prob = actor.log_prob(expanded, action)
    assert torch.allclose(sample_log_prob, evaluated_log_prob, atol=1e-4, rtol=1e-4)


def test_rebrac_flow_rejects_zero_layers() -> None:
    from freqtrade.hedge.hprl.algorithms.rebrac_v2 import ConditionalFlowActor

    with pytest.raises(ValueError):
        ConditionalFlowActor(4, 2, hidden_dim=16, flow_layers=0)


def test_running_norm_rejects_empty_or_nonfinite_updates() -> None:
    from freqtrade.hedge.hprl.networks import RunningNorm

    norm = RunningNorm(3)
    with pytest.raises(ValueError):
        norm.update(torch.empty((0, 3)))
    with pytest.raises(ValueError):
        norm.update(torch.tensor([[0.0, float("nan"), 0.0]]))


def test_running_norm_updates_finite_statistics() -> None:
    from freqtrade.hedge.hprl.networks import RunningNorm

    norm = RunningNorm(3)
    values = torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    norm.update(values)
    assert torch.isfinite(norm.mean).all()
    assert torch.isfinite(norm.var).all()
    assert torch.all(norm.var >= 0)


def test_zero_impact_coefficient_stays_zero_for_large_turnover() -> None:
    model = ExecutionCostModel(HPRLCostConfig(impact_coefficient_bps=0.0))
    result = model.evaluate(
        turnover_notional=torch.tensor([1e30], dtype=torch.float32),
        equity=torch.tensor([1000.0]),
        available_notional=torch.tensor([1.0]),
    )
    assert result.market_impact.item() == pytest.approx(0.0)


def test_action_projection_contract_validates_payload() -> None:
    from freqtrade.hedge.hprl.contracts import ActionProjection

    with pytest.raises(ValueError):
        ActionProjection(requested=(0.1,), projected=(0.1, 0.2), clipped=False)
    with pytest.raises(ValueError):
        ActionProjection(requested=(0.1,), projected=(-0.1,), clipped=True)


def test_discounted_return_normalizer_stabilizes_replay_reward_scale() -> None:
    from freqtrade.hedge.hprl.trainer import DiscountedReturnNormalizer

    normalizer = DiscountedReturnNormalizer(4, 0.99, device="cpu")
    reward = torch.full((4,), -0.25)
    done = torch.zeros(4, dtype=torch.bool)
    first = normalizer.normalize(reward, done)
    second = normalizer.normalize(reward, done)
    assert torch.allclose(first, reward)
    assert normalizer.scale > 0.0
    assert torch.isfinite(second).all()
    assert second.abs().max().item() < 5.0


def test_discounted_return_normalizer_resets_terminal_accumulator() -> None:
    from freqtrade.hedge.hprl.trainer import DiscountedReturnNormalizer

    normalizer = DiscountedReturnNormalizer(1, 0.99, device="cpu")
    normalizer.normalize(torch.tensor([1.0]), torch.tensor([False]))
    normalizer.normalize(torch.tensor([2.0]), torch.tensor([True]))
    assert normalizer.discounted_return.item() == pytest.approx(2.0)


def test_online_categorical_agents_request_return_std_normalization() -> None:
    from freqtrade.hedge.hprl.algorithms.simba_sac import SimbaSACAgent

    config = small_training()
    xqc = XQCAgent(6, 4, config, device="cpu")
    simba = SimbaSACAgent(6, 4, config, device="cpu")
    td3 = FastTD3Agent(6, 4, config, device="cpu")
    assert xqc.reward_normalization == "return_std"
    assert simba.reward_normalization == "return_std"
    assert td3.reward_normalization == "return_std"


def test_xqc_uses_published_categorical_support() -> None:
    agent = XQCAgent(6, 4, small_training(), device="cpu")
    assert agent.critic.bins == 101
    assert agent.critic.value_min == pytest.approx(-5.0)
    assert agent.critic.value_max == pytest.approx(5.0)


def test_xqc_weightnorm_scales_are_projected_to_unit_sphere() -> None:
    agent = XQCAgent(6, 4, small_training(), device="cpu")
    agent.update(random_batch())
    scales = [
        parameter.detach()
        for name, parameter in agent.critic.named_parameters()
        if name.endswith("parametrizations.weight.original0")
    ]
    assert scales
    assert all(torch.allclose(scale, torch.ones_like(scale), atol=1e-6) for scale in scales)


def test_xqc_joined_bn_forward_counts_current_and_next_batch() -> None:
    agent = XQCAgent(6, 4, small_training(), device="cpu")
    batch = random_batch(batch=8)
    bn = next(
        module for module in agent.critic.modules() if isinstance(module, torch.nn.BatchNorm1d)
    )
    before = int(bn.num_batches_tracked.item())
    agent.update(batch)
    after = int(bn.num_batches_tracked.item())
    assert after == before + 1


def test_rebrac_offline_support_calibrates_to_dataset_returns() -> None:
    config = small_training(batch_size=2)
    agent = ReBRACv2Agent(2, 1, config, device="cpu")
    low, high = agent.calibrate_return_support(
        torch.tensor([[0.5], [1.0], [-0.25]]),
        torch.tensor([[0.0], [1.0], [1.0]]),
    )
    assert low < -0.25
    assert high > 1.0
    assert agent.critic.value_min == pytest.approx(low)
    assert agent.critic_target.value_max == pytest.approx(high)


def test_rebrac_dynamic_support_survives_checkpoint_roundtrip(tmp_path) -> None:
    config = small_training()
    source = ReBRACv2Agent(6, 4, config, device="cpu")
    source.calibrate_return_support(torch.tensor([0.2, -0.4]), torch.tensor([0.0, 1.0]))
    path = save_checkpoint(tmp_path / "rebrac-support.pt", source, {})
    target = ReBRACv2Agent(6, 4, config, device="cpu")
    load_checkpoint(path, target)
    assert torch.equal(source.critic.support, target.critic.support)
    assert source.critic.value_min == pytest.approx(target.critic.value_min)
    assert source.critic.value_max == pytest.approx(target.critic.value_max)


def test_evaluation_rejects_fractional_periods_or_liquidations() -> None:
    with pytest.raises(ValueError):
        evaluate_trading(
            [100.0, 101.0, 102.0],
            periods_per_year=365.5,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        evaluate_trading(
            [100.0, 101.0, 102.0],
            periods_per_year=365,
            liquidations=1.5,  # type: ignore[arg-type]
        )


def test_xqc_published_temperature_entropy_and_policy_delay() -> None:
    agent = XQCAgent(6, 4, small_training(), device="cpu")
    assert agent.alpha.item() == pytest.approx(0.01, rel=1e-6)
    assert agent.target_entropy == pytest.approx(-2.0)
    first = agent.update(random_batch()).values
    second = agent.update(random_batch()).values
    third = agent.update(random_batch()).values
    assert first["actor_loss"] == 0.0
    assert second["actor_loss"] == 0.0
    assert third["actor_loss"] != 0.0


def test_environment_tail_history_is_preallocated_bounded_ring() -> None:
    env = VectorizedHedgeEnv(
        market_dataset(length=140),
        HPRLEnvironmentConfig(parallel_envs=2),
    )
    for _ in range(130):
        env.step(torch.zeros((env.envs, env.action_dim)))
    assert env._return_history.shape == (128, env.envs)
    assert env._return_history_count == 128


def test_vector_env_autoresets_only_terminated_rows_without_group_reset() -> None:
    data = TensorMarketDataset(
        torch.zeros((3, 1, 2)),
        torch.full((3, 1), -1.0),
        symbols=("BTC",),
    )
    env = VectorizedHedgeEnv(
        data,
        HPRLEnvironmentConfig(
            parallel_envs=2,
            terminate_equity_ratio=0.25,
            costs=HPRLCostConfig(
                maker_fee_bps=0.0,
                taker_fee_bps=0.0,
                base_slippage_bps=0.0,
                impact_coefficient_bps=0.0,
            ),
            action=HPRLActionConfig(
                mode="continuous",
                max_leg_exposure=1.0,
                max_gross_exposure=1.0,
                max_abs_net_exposure=1.0,
                max_step_change=1.0,
            ),
        ),
        device="cpu",
    )
    env.reset()
    action = torch.tensor([[1.0, 0.0], [0.2, 0.0]])
    step = env.step(action)
    assert step.terminated.tolist() == [True, False]
    assert step.info["time_done"] is False
    assert step.info["equity"].tolist() == pytest.approx([0.0, 800.0])
    # The failed row is reset in-device for the next observation; the surviving row is preserved.
    assert env.equity.tolist() == pytest.approx([1000.0, 800.0])
    assert env.position[0].sum().item() == pytest.approx(0.0)
    assert env.position[1, 0, 0].item() == pytest.approx(0.2)


def test_agent_can_skip_host_metrics_materialization_on_hot_path() -> None:
    config = small_training()
    agent = FastTD3Agent(6, 2, config, device="cpu")
    metrics = agent.update(random_batch(obs_dim=6, action_dim=2), collect_metrics=False)
    assert metrics.values == {}
