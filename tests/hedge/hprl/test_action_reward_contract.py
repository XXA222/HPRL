from __future__ import annotations

import math

import pytest
import torch

from freqtrade.hedge.hprl.action_space import (
    TieredHedgeActionCodec,
    action_for_critic,
    canonicalize_offline_action_tensor,
    configure_agent_action_levels,
    gaussian_selected_tier_log_prob,
    gaussian_tier_entropy,
    gaussian_tier_probabilities,
    hard_quantize_unit_action,
    straight_through_quantize_unit_action,
)
from freqtrade.hedge.hprl.config import (
    HPRLActionConfig,
    HPRLConfig,
    HPRLEnvironmentConfig,
    HPRLRewardConfig,
    HPRLTrainingConfig,
)
from freqtrade.hedge.hprl.contracts import OfflineTransition
from freqtrade.hedge.hprl.data import OfflineTransitionDataset, TensorMarketDataset
from freqtrade.hedge.hprl.env import VectorizedHedgeEnv
from freqtrade.hedge.hprl.errors import HPRLConfigError
from freqtrade.hedge.hprl.reward import CompositeReward, RewardFactsTensor
from freqtrade.hedge.hprl.runtime import build_online_runtime


def _market(return_value: float = 0.0, *, length: int = 8, symbols: int = 1):
    return TensorMarketDataset(
        torch.zeros((length, symbols, 3), dtype=torch.float32),
        torch.full((length, symbols), float(return_value), dtype=torch.float32),
        symbols=tuple(f"S{i}" for i in range(symbols)),
    ).validate()


def _facts(**overrides):
    values = {
        "equity_return": torch.tensor([0.001]),
        "drawdown_increase": torch.tensor([0.0]),
        "downside_return": torch.tensor([0.001]),
        "cvar_loss": torch.tensor([0.0]),
        "turnover_ratio": torch.tensor([0.0]),
        "fee_ratio": torch.tensor([0.0]),
        "slippage_ratio": torch.tensor([0.0]),
        "impact_ratio": torch.tensor([0.0]),
        "funding_ratio": torch.tensor([0.0]),
        "quantization_distance": torch.tensor([0.0]),
        "constraint_distance": torch.tensor([0.0]),
        "gross_margin_ratio": torch.tensor([0.0]),
        "hedge_overlap_ratio": torch.tensor([0.0]),
        "opportunity_miss": torch.tensor([0.0]),
        "terminal": torch.tensor([False]),
    }
    values.update(overrides)
    return RewardFactsTensor(**values)


def test_default_tier_contract_is_5x5() -> None:
    cfg = HPRLActionConfig()
    assert cfg.mode == "tiered"
    assert cfg.position_levels == (0.0, 0.05, 0.12, 0.25, 0.40)
    assert cfg.level_count == 5
    assert cfg.joint_states_per_symbol == 25
    assert cfg.multi_discrete_nvec == (5, 5)


def test_levels_are_canonicalized_to_immutable_tuple() -> None:
    cfg = HPRLActionConfig(position_levels=[0, 0.1, 0.2])
    assert cfg.position_levels == (0.0, 0.1, 0.2)


def test_invalid_levels_rejected() -> None:
    invalid = [(), (0.0,), (0.01, 0.1), (0.0, 0.1, 0.1), (0.0, -0.1, 0.2)]
    for levels in invalid:
        with pytest.raises(HPRLConfigError):
            HPRLActionConfig(position_levels=levels)
    with pytest.raises(HPRLConfigError):
        HPRLTrainingConfig(tier_entropy_target_fraction=1.1)


def test_largest_level_must_fit_margin_envelope() -> None:
    with pytest.raises(HPRLConfigError):
        HPRLActionConfig(max_leg_margin_ratio=0.25)


def test_unlimited_derisk_is_valid_default() -> None:
    assert HPRLActionConfig().max_decrease_levels == -1


def test_policy_quantizer_uses_five_canonical_codes() -> None:
    action = torch.tensor([0.0, 0.12, 0.26, 0.50, 0.74, 0.88, 1.0])
    quantized = hard_quantize_unit_action(action, 5)
    allowed = {0.0, 0.25, 0.5, 0.75, 1.0}
    assert set(float(x) for x in quantized.tolist()) <= allowed


def test_straight_through_quantizer_keeps_identity_gradient() -> None:
    action = torch.tensor([0.31, 0.63], requires_grad=True)
    out = straight_through_quantize_unit_action(action, 5)
    out.sum().backward()
    assert out.tolist() == pytest.approx([0.25, 0.75])
    assert action.grad.tolist() == pytest.approx([1.0, 1.0])


def test_gaussian_tier_probabilities_form_exact_distribution() -> None:
    mean = torch.zeros((4, 3), requires_grad=True)
    log_std = torch.zeros((4, 3), requires_grad=True)
    probs = gaussian_tier_probabilities(mean, log_std, 5)
    assert probs.shape == (4, 3, 5)
    assert torch.allclose(probs.sum(dim=-1), torch.ones((4, 3)), atol=1e-6)
    probs.square().sum().backward()
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(log_std.grad).all()


def test_gaussian_tier_entropy_is_bounded_and_differentiable() -> None:
    mean = torch.zeros((3, 2), requires_grad=True)
    log_std = torch.zeros((3, 2), requires_grad=True)
    entropy = gaussian_tier_entropy(mean, log_std, 5)
    assert torch.all(entropy > 0)
    assert torch.all(entropy <= math.log(5.0) + 1e-6)
    entropy.mean().backward()
    assert torch.isfinite(mean.grad).all()
    assert torch.isfinite(log_std.grad).all()


def test_selected_tier_log_prob_matches_probability_mass() -> None:
    mean = torch.zeros((1, 2))
    log_std = torch.zeros((1, 2))
    action = torch.tensor([[0.50, 1.00]])
    probs = gaussian_tier_probabilities(mean, log_std, 5)
    selected = gaussian_selected_tier_log_prob(mean, log_std, action, 5)
    assert selected[0, 0].exp().item() == pytest.approx(probs[0, 0, 2].item())
    assert selected[0, 1].exp().item() == pytest.approx(probs[0, 1, 4].item())


def test_sac_family_reports_executed_tier_entropy() -> None:
    for algorithm in ("fast_dsac", "simba_sac"):
        config = HPRLConfig(
            environment=HPRLEnvironmentConfig(parallel_envs=4),
            training=HPRLTrainingConfig(
                algorithm=algorithm,
                device="cpu",
                batch_size=8,
                replay_capacity=64,
                warmup_steps=8,
                gradient_steps=1,
                hidden_dim=32,
                hidden_depth=1,
                metrics_interval=1,
            ),
        )
        runtime = build_online_runtime(_market(return_value=0.0005, length=16), config)
        summary = runtime.trainer.run(5)
        assert summary.last_metrics["tier_entropy_mean"] > 0
        assert math.isfinite(summary.last_metrics["tier_entropy_mean"])
        runtime.close()


def test_tier_decode_separates_margin_and_notional_by_leverage() -> None:
    cfg = HPRLActionConfig(leverage=3.0, max_increase_levels=4)
    codec = TieredHedgeActionCodec(cfg)
    current = torch.zeros((1, 1, 2))
    result = codec.decode(torch.tensor([[[1.0, 0.0]]]), current)
    assert result.target_margin[0, 0, 0].item() == pytest.approx(0.40)
    assert result.target_notional[0, 0, 0].item() == pytest.approx(1.20)


def test_tier_decode_limits_risk_on_margin_not_notional() -> None:
    cfg = HPRLActionConfig(
        leverage=5.0,
        max_gross_margin_ratio=0.40,
        max_abs_net_margin_ratio=0.40,
        max_increase_levels=4,
    )
    codec = TieredHedgeActionCodec(cfg)
    result = codec.decode(torch.ones((1, 1, 2)), torch.zeros((1, 1, 2)))
    assert result.target_margin.sum().item() <= 0.40 + 1e-6
    assert result.target_notional.sum().item() <= 2.0 + 1e-6


def test_increase_is_one_level_by_default() -> None:
    codec = TieredHedgeActionCodec(HPRLActionConfig())
    result = codec.decode(torch.tensor([[[1.0, 0.0]]]), torch.zeros((1, 1, 2)))
    assert result.executed_level_index[0, 0, 0].item() == 1
    assert result.target_margin[0, 0, 0].item() == pytest.approx(0.05)
    assert bool(result.transition_limited.item())


def test_derisk_can_drop_from_heavy_to_zero_immediately() -> None:
    codec = TieredHedgeActionCodec(HPRLActionConfig())
    current = torch.tensor([[[0.40, 0.0]]])
    result = codec.decode(torch.zeros((1, 1, 2)), current)
    assert result.executed_level_index[0, 0, 0].item() == 0
    assert result.target_margin.sum().item() == pytest.approx(0.0)


def test_global_gross_projection_remains_on_grid() -> None:
    cfg = HPRLActionConfig(
        max_gross_margin_ratio=0.30,
        max_abs_net_margin_ratio=0.30,
        max_increase_levels=4,
    )
    codec = TieredHedgeActionCodec(cfg)
    result = codec.decode(torch.ones((4, 2, 2)), torch.zeros((4, 2, 2)))
    assert torch.all(result.target_margin.sum(dim=(-2, -1)) <= 0.30 + 1e-6)
    levels = torch.tensor(cfg.position_levels)
    assert torch.isin(result.target_margin.flatten(), levels).all()


def test_positive_net_projection_reduces_long_only_as_needed() -> None:
    cfg = HPRLActionConfig(
        max_gross_margin_ratio=0.80,
        max_abs_net_margin_ratio=0.12,
        max_increase_levels=4,
    )
    codec = TieredHedgeActionCodec(cfg)
    raw = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    result = codec.decode(raw, torch.zeros_like(raw))
    net = (result.target_margin[..., 0] - result.target_margin[..., 1]).sum()
    assert net.item() <= 0.12 + 1e-6
    assert bool(result.risk_limited.item())


def test_negative_net_projection_reduces_short() -> None:
    cfg = HPRLActionConfig(
        max_gross_margin_ratio=0.80,
        max_abs_net_margin_ratio=0.12,
        max_increase_levels=4,
    )
    codec = TieredHedgeActionCodec(cfg)
    raw = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    result = codec.decode(raw, torch.zeros_like(raw))
    net = (result.target_margin[..., 0] - result.target_margin[..., 1]).sum()
    assert net.item() >= -0.12 - 1e-6


def test_uneven_custom_levels_do_not_break_net_envelope() -> None:
    cfg = HPRLActionConfig(
        position_levels=(0.0, 0.03, 0.17, 0.40),
        max_abs_net_margin_ratio=0.08,
        max_gross_margin_ratio=0.50,
        max_increase_levels=3,
    )
    result = TieredHedgeActionCodec(cfg).decode(
        torch.tensor([[[1.0, 0.3], [0.8, 0.0]]]),
        torch.zeros((1, 2, 2)),
    )
    net = (result.target_margin[..., 0] - result.target_margin[..., 1]).sum()
    assert abs(net.item()) <= 0.08 + 1e-6


def test_quantization_and_constraint_distances_are_separate() -> None:
    cfg = HPRLActionConfig(
        max_gross_margin_ratio=0.05,
        max_abs_net_margin_ratio=0.05,
        max_increase_levels=4,
    )
    result = TieredHedgeActionCodec(cfg).decode(
        torch.tensor([[[0.51, 0.0]]]), torch.zeros((1, 1, 2))
    )
    assert result.quantization_distance.item() > 0
    assert result.constraint_distance.item() > 0


def test_env_reports_margin_notional_and_policy_code_separately() -> None:
    cfg = HPRLEnvironmentConfig(
        parallel_envs=1,
        action=HPRLActionConfig(leverage=2.0, max_increase_levels=4),
    )
    env = VectorizedHedgeEnv(_market(0.001), cfg)
    step = env.step(torch.tensor([[1.0, 0.0]]))
    assert step.info["target_margin"][0, 0, 0].item() == pytest.approx(0.40)
    assert step.info["target_notional"][0, 0, 0].item() == pytest.approx(0.80)
    assert step.info["executed_action"][0, 0].item() == pytest.approx(1.0)
    assert step.info["joint_action_index"][0, 0].item() == 20


def test_leverage_changes_pnl_without_changing_margin_budget() -> None:
    base = HPRLActionConfig(leverage=1.0, max_increase_levels=4)
    levered = HPRLActionConfig(leverage=3.0, max_increase_levels=4)
    env1 = VectorizedHedgeEnv(_market(0.01), HPRLEnvironmentConfig(parallel_envs=1, action=base))
    env3 = VectorizedHedgeEnv(
        _market(0.01), HPRLEnvironmentConfig(parallel_envs=1, action=levered)
    )
    action = torch.tensor([[1.0, 0.0]])
    step1 = env1.step(action)
    step3 = env3.step(action)
    assert step1.info["target_margin"].sum().item() == pytest.approx(
        step3.info["target_margin"].sum().item()
    )
    assert step3.info["target_notional"].sum().item() == pytest.approx(
        3.0 * step1.info["target_notional"].sum().item()
    )
    assert step3.info["equity"].item() > step1.info["equity"].item()


def test_online_runtime_configures_agent_tier_count() -> None:
    config = HPRLConfig(
        environment=HPRLEnvironmentConfig(parallel_envs=2),
        training=HPRLTrainingConfig(
            algorithm="fast_td3",
            batch_size=4,
            replay_capacity=32,
            warmup_steps=4,
            hidden_dim=32,
            hidden_depth=1,
            device="cpu",
        ),
    )
    runtime = build_online_runtime(_market(length=10), config)
    assert runtime.agent.action_level_count == 5
    action = runtime.agent.act(torch.zeros((2, runtime.env.observation_dim)))
    assert torch.allclose(action * 4.0, torch.round(action * 4.0))
    runtime.close()


def test_action_for_critic_hard_and_ste_share_forward_grid() -> None:
    class Agent:
        pass

    agent = Agent()
    configure_agent_action_levels(agent, 5)
    x = torch.tensor([[0.31, 0.64]], requires_grad=True)
    hard = action_for_critic(agent, x, straight_through=False)
    ste = action_for_critic(agent, x, straight_through=True)
    assert torch.equal(hard, ste.detach())
    ste.sum().backward()
    assert x.grad.tolist()[0] == pytest.approx([1.0, 1.0])


def test_offline_margin_budget_conversion_is_exact() -> None:
    cfg = HPRLActionConfig()
    source = torch.tensor([[0.0, 0.05, 0.12, 0.25, 0.40]])
    encoded = canonicalize_offline_action_tensor(source, cfg, "margin_budget")
    assert encoded.tolist()[0] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_offline_notional_conversion_respects_leverage() -> None:
    cfg = HPRLActionConfig(leverage=2.0)
    source = torch.tensor([[0.10, 0.24, 0.50, 0.80]])
    encoded = canonicalize_offline_action_tensor(source, cfg, "notional_exposure")
    assert encoded.tolist()[0] == pytest.approx([0.25, 0.5, 0.75, 1.0])


def test_offline_unaligned_action_is_rejected() -> None:
    with pytest.raises(ValueError, match="not aligned"):
        canonicalize_offline_action_tensor(
            torch.tensor([[0.11]]), HPRLActionConfig(), "margin_budget"
        )


def test_offline_policy_code_must_already_be_on_grid() -> None:
    with pytest.raises(ValueError, match="tier grid"):
        canonicalize_offline_action_tensor(
            torch.tensor([[0.30]]), HPRLActionConfig(), "policy_code"
        )


def test_offline_dataset_declares_action_unit() -> None:
    row = OfflineTransition((0.0,), (0.05,), 0.0, (0.0,), False)
    dataset = OfflineTransitionDataset([row], action_unit="margin_budget")
    assert dataset.action_unit == "margin_budget"


def test_reward_positive_growth_is_positive() -> None:
    total, parts = CompositeReward().evaluate_tensor(_facts())
    assert total.item() > 0
    assert parts["equity"].item() > 0


def test_reward_loss_is_more_negative_with_downside_shaping() -> None:
    model = CompositeReward()
    positive, _ = model.evaluate_tensor(_facts(equity_return=torch.tensor([0.001])))
    negative, _ = model.evaluate_tensor(
        _facts(
            equity_return=torch.tensor([-0.001]),
            downside_return=torch.tensor([-0.001]),
        )
    )
    assert abs(negative.item()) > abs(positive.item())


def test_reward_drawdown_increment_penalized_but_recovery_not_penalized() -> None:
    model = CompositeReward()
    _, inc = model.evaluate_tensor(_facts(drawdown_increase=torch.tensor([0.01])))
    _, flat = model.evaluate_tensor(_facts(drawdown_increase=torch.tensor([-0.01])))
    assert inc["drawdown"].item() < 0
    assert flat["drawdown"].item() == pytest.approx(0.0)


def test_reward_cvar_penalty_is_monotonic() -> None:
    model = CompositeReward()
    low, _ = model.evaluate_tensor(_facts(cvar_loss=torch.tensor([0.001])))
    high, _ = model.evaluate_tensor(_facts(cvar_loss=torch.tensor([0.01])))
    assert high.item() < low.item()


def test_reward_turnover_regularizer_is_small_relative_to_old_design() -> None:
    model = CompositeReward()
    _, parts = model.evaluate_tensor(_facts(turnover_ratio=torch.tensor([0.10])))
    assert parts["turnover"].item() == pytest.approx(-0.01, abs=1e-6)


def test_reward_default_cost_shaping_does_not_double_count_net_equity() -> None:
    _, parts = CompositeReward().evaluate_tensor(
        _facts(
            fee_ratio=torch.tensor([0.01]),
            slippage_ratio=torch.tensor([0.01]),
            impact_ratio=torch.tensor([0.01]),
            funding_ratio=torch.tensor([0.01]),
        )
    )
    assert parts["fees"].item() == 0.0
    assert parts["slippage"].item() == 0.0
    assert parts["market_impact"].item() == 0.0
    assert parts["funding"].item() == 0.0


def test_reward_projection_penalty_uses_distance_not_binary_constant() -> None:
    model = CompositeReward()
    _, small = model.evaluate_tensor(_facts(constraint_distance=torch.tensor([0.10])))
    _, large = model.evaluate_tensor(_facts(constraint_distance=torch.tensor([0.50])))
    assert abs(large["risk_projection"].item()) > abs(small["risk_projection"].item())


def test_reward_quantization_alignment_is_independent_of_risk_projection() -> None:
    _, parts = CompositeReward().evaluate_tensor(
        _facts(
            quantization_distance=torch.tensor([0.10]),
            constraint_distance=torch.tensor([0.0]),
        )
    )
    assert parts["quantization_alignment"].item() < 0
    assert parts["risk_projection"].item() == pytest.approx(0.0)


def test_reward_gross_margin_penalty_only_above_soft_limit() -> None:
    model = CompositeReward()
    _, below = model.evaluate_tensor(_facts(gross_margin_ratio=torch.tensor([0.50])))
    _, above = model.evaluate_tensor(_facts(gross_margin_ratio=torch.tensor([0.80])))
    assert below["gross_margin_risk"].item() == pytest.approx(0.0)
    assert above["gross_margin_risk"].item() < 0


def test_reward_terminal_penalty_is_explicit() -> None:
    _, parts = CompositeReward().evaluate_tensor(_facts(terminal=torch.tensor([True])))
    assert parts["terminal_loss"].item() == pytest.approx(-2.0)


def test_reward_is_clipped_for_catastrophic_loss() -> None:
    total, _ = CompositeReward().evaluate_tensor(
        _facts(
            equity_return=torch.tensor([-1.0]),
            downside_return=torch.tensor([-1.0]),
            terminal=torch.tensor([True]),
        )
    )
    assert total.item() == pytest.approx(-5.0)


def test_reward_legacy_projected_mask_remains_compatible() -> None:
    facts = RewardFactsTensor(
        equity_return=torch.tensor([0.0]),
        drawdown_increase=torch.tensor([0.0]),
        downside_return=torch.tensor([0.0]),
        cvar_loss=torch.tensor([0.0]),
        turnover_ratio=torch.tensor([0.0]),
        fee_ratio=torch.tensor([0.0]),
        slippage_ratio=torch.tensor([0.0]),
        impact_ratio=torch.tensor([0.0]),
        funding_ratio=torch.tensor([0.0]),
        projected=torch.tensor([True]),
    )
    _, parts = CompositeReward().evaluate_tensor(facts)
    assert parts["risk_projection"].item() < 0


def test_reward_soft_limit_may_exceed_tighter_hard_envelope() -> None:
    cfg = HPRLConfig(
        environment=HPRLEnvironmentConfig(
            action=HPRLActionConfig(max_gross_margin_ratio=0.50),
            reward=HPRLRewardConfig(gross_margin_soft_limit=0.60),
        )
    )
    assert cfg.environment.action.max_gross_margin_ratio == pytest.approx(0.50)


def test_reward_log_growth_is_finite_near_total_loss() -> None:
    total, parts = CompositeReward().evaluate_tensor(
        _facts(equity_return=torch.tensor([-0.999999]))
    )
    assert math.isfinite(total.item())
    assert math.isfinite(parts["equity"].item())


def test_tier_hysteresis_holds_position_near_boundary() -> None:
    cfg = HPRLActionConfig(tier_hysteresis=0.03, max_increase_levels=4)
    codec = TieredHedgeActionCodec(cfg)
    # Current tier 1 has policy code .25. Boundary to tier 2 is .375; hysteresis raises it to .405.
    current = torch.tensor([[[0.05, 0.0]]])
    held = codec.decode(torch.tensor([[[0.39, 0.0]]]), current)
    crossed = codec.decode(torch.tensor([[[0.42, 0.0]]]), current)
    assert held.executed_level_index[0, 0, 0].item() == 1
    assert crossed.executed_level_index[0, 0, 0].item() == 2


def test_tier_hysteresis_still_allows_emergency_derisk() -> None:
    cfg = HPRLActionConfig(tier_hysteresis=0.03)
    codec = TieredHedgeActionCodec(cfg)
    current = torch.tensor([[[0.40, 0.0]]])
    result = codec.decode(torch.tensor([[[0.0, 0.0]]]), current)
    assert result.target_margin[0, 0, 0].item() == pytest.approx(0.0)


def test_hysteresis_cannot_consume_half_policy_step() -> None:
    with pytest.raises(HPRLConfigError):
        HPRLActionConfig(tier_hysteresis=0.125)


def test_cvar_history_is_independent_per_parallel_environment_after_reset() -> None:
    env = VectorizedHedgeEnv(
        _market(length=12),
        HPRLEnvironmentConfig(parallel_envs=2),
    )
    # Populate several negative-return observations for both rows.
    for _ in range(5):
        env._tail_loss(torch.tensor([-0.02, -0.01]))
    assert env._return_history_valid[:, 0].sum().item() == 5
    assert env._return_history_valid[:, 1].sum().item() == 5
    # Simulate the same row-local terminal cleanup performed by step().
    bankrupt = torch.tensor([True, False])
    env._return_history.mul_((~bankrupt).to(torch.float32).unsqueeze(0))
    env._return_history_valid &= (~bankrupt).unsqueeze(0)
    assert env._return_history_valid[:, 0].sum().item() == 0
    assert env._return_history_valid[:, 1].sum().item() == 5
    tail = env._tail_loss(torch.tensor([-0.001, -0.001]))
    assert torch.isfinite(tail).all()
    assert env._return_history_valid[:, 0].sum().item() == 1
    assert env._return_history_valid[:, 1].sum().item() == 6


def test_margin_budget_ratios_cannot_exceed_account_equity() -> None:
    with pytest.raises(HPRLConfigError):
        HPRLActionConfig(position_levels=(0.0, 0.5, 1.2), max_leg_margin_ratio=1.2)
    with pytest.raises(HPRLConfigError):
        HPRLActionConfig(max_gross_margin_ratio=1.1)


def test_environment_allows_tighter_hard_margin_than_reward_soft_limit() -> None:
    cfg = HPRLEnvironmentConfig(
        action=HPRLActionConfig(max_gross_margin_ratio=0.50),
        reward=HPRLRewardConfig(gross_margin_soft_limit=0.60),
    )
    assert cfg.action.max_gross_margin_ratio < cfg.reward.gross_margin_soft_limit


def test_from_mapping_accepts_json_position_level_list() -> None:
    cfg = HPRLConfig.from_mapping(
        {
            "environment": {
                "action": {
                    "mode": "tiered",
                    "position_levels": [0, 0.05, 0.12, 0.25, 0.40],
                }
            }
        }
    )
    assert cfg.environment.action.position_levels == (0.0, 0.05, 0.12, 0.25, 0.40)


@pytest.mark.parametrize("algorithm", ["fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2"])
def test_all_hprl_agents_emit_canonical_tier_codes(algorithm: str) -> None:
    config = HPRLConfig(
        environment=HPRLEnvironmentConfig(parallel_envs=2),
        training=HPRLTrainingConfig(
            algorithm=algorithm,
            device="cpu",
            batch_size=4,
            replay_capacity=32,
            warmup_steps=4,
            hidden_dim=32,
            hidden_depth=1,
        ),
    )
    runtime = build_online_runtime(_market(length=12), config)
    action = runtime.agent.act(torch.zeros((2, runtime.env.observation_dim)), deterministic=True)
    scaled = action * float(runtime.env.action_level_count - 1)
    assert torch.allclose(scaled, torch.round(scaled))
    runtime.close()


@pytest.mark.parametrize("algorithm", ["fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2"])
def test_all_hprl_algorithms_complete_tiered_gradient_update(algorithm: str) -> None:
    config = HPRLConfig(
        environment=HPRLEnvironmentConfig(parallel_envs=4),
        training=HPRLTrainingConfig(
            algorithm=algorithm,
            device="cpu",
            batch_size=8,
            replay_capacity=64,
            warmup_steps=8,
            gradient_steps=1,
            hidden_dim=32,
            hidden_depth=1,
            metrics_interval=1,
        ),
    )
    runtime = build_online_runtime(_market(return_value=0.0005, length=16), config)
    summary = runtime.trainer.run(5)
    assert summary.updates >= 1
    assert torch.isfinite(runtime.env.equity).all()
    runtime.close()


def test_reward_components_sum_to_total_when_not_clipped() -> None:
    total, parts = CompositeReward().evaluate_tensor(_facts(equity_return=torch.tensor([0.0001])))
    summed = torch.zeros_like(total)
    for value in parts.values():
        summed = summed + value
    assert total.item() == pytest.approx(summed.item(), abs=1e-7)


def test_one_percent_log_growth_is_about_one_reward_unit() -> None:
    total, parts = CompositeReward(
        HPRLRewardConfig(
            drawdown=0.0,
            downside=0.0,
            cvar=0.0,
            turnover=0.0,
            quantization_alignment=0.0,
            risk_projection=0.0,
            gross_margin_risk=0.0,
            terminal_loss=0.0,
        )
    ).evaluate_tensor(_facts(equity_return=torch.tensor([0.01])))
    assert parts["equity"].item() == pytest.approx(100.0 * math.log1p(0.01), rel=1e-6)
    assert total.item() == pytest.approx(parts["equity"].item())
