from __future__ import annotations

import pytest
import torch

from freqtrade.hedge.hprl.action_space import (
    gaussian_selected_tier_log_prob,
    gaussian_tier_probabilities,
)
from freqtrade.hedge.hprl.algorithms.rebrac_v2 import ConditionalFlowActor, ReBRACv2Agent
from freqtrade.hedge.hprl.config import HPRLTrainingConfig
from freqtrade.hedge.hprl.networks import (
    GaussianActor,
    HypersphericalLinear,
    SimbaGaussianActor,
    SimbaResidualBlock,
)
from freqtrade.hedge.hprl.performance import (
    auto_compile_policy,
    compile_break_even_updates,
    compile_policy_thresholds,
    estimate_compile_break_even_updates,
    estimate_compile_startup_seconds,
    host_interop_profile_info,
    resolve_compile_mode,
    resolve_host_interop_threads,
    resolve_rebrac_flow_precision,
)


@pytest.mark.parametrize("levels", range(2, 10))
def test_selected_tier_direct_matches_full_distribution(levels):
    torch.manual_seed(100 + levels)
    mean = torch.randn(256, 4).clamp(-2.5, 2.5)
    log_std = torch.randn(256, 4).clamp(-3.0, 1.5)
    index = torch.randint(levels, (256, 4))
    action = index.float() / float(levels - 1)
    full = gaussian_tier_probabilities(mean, log_std, levels)
    reference = torch.log(
        full.gather(-1, index.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8)
    )
    direct = gaussian_selected_tier_log_prob(mean, log_std, action, levels)
    torch.testing.assert_close(direct, reference, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("levels", range(2, 7))
def test_selected_tier_direct_has_finite_gradients(levels):
    torch.manual_seed(200 + levels)
    mean = torch.randn(64, 3, requires_grad=True)
    log_std = torch.randn(64, 3).clamp(-3.0, 1.0).requires_grad_()
    index = torch.randint(levels, (64, 3))
    action = index.float() / float(levels - 1)
    loss = -gaussian_selected_tier_log_prob(mean, log_std, action, levels).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert mean.grad is not None and torch.isfinite(mean.grad).all()
    assert log_std.grad is not None and torch.isfinite(log_std.grad).all()


@pytest.mark.parametrize(
    "algorithm,expected",
    [
        ("fast_td3", "reduce-overhead"),
        ("xqc", "reduce-overhead"),
        ("fast_dsac", "reduce-overhead"),
        ("simba_sac", "reduce-overhead"),
        ("rebrac_v2", "reduce-overhead"),
    ],
)
def test_algorithm_aware_auto_compile_policy(algorithm, expected):
    assert auto_compile_policy("rtx5070_laptop")[algorithm] == expected
    threshold = compile_break_even_updates(algorithm, "rtx5070_laptop")
    assert threshold is not None and threshold > 0
    assert resolve_compile_mode("auto", algorithm, "cpu", expected_updates=10**9) == "off"
    assert resolve_compile_mode(
        "auto", algorithm, "cuda:0", expected_updates=threshold - 1,
        hardware_profile="rtx5070_laptop",
    ) == "off"
    assert resolve_compile_mode(
        "auto", algorithm, "cuda:0", expected_updates=threshold,
        hardware_profile="rtx5070_laptop",
    ) == "reduce-overhead"


@pytest.mark.parametrize("mode", ["off", "default", "reduce-overhead", "max-autotune"])
def test_explicit_compile_mode_overrides_auto_policy(mode):
    assert resolve_compile_mode(mode, "fast_td3", "cpu") == mode


@pytest.mark.parametrize("value", ["auto", "fp32"])
def test_flow_likelihood_precision_cpu_config(value):
    cfg = HPRLTrainingConfig(
        device="cpu",
        batch_size=8,
        replay_capacity=32,
        hidden_dim=16,
        hidden_depth=1,
        flow_likelihood_precision=value,
    )
    agent = ReBRACv2Agent(5, 2, cfg, device="cpu")
    assert agent.flow_likelihood_precision == "fp32"


def test_flow_likelihood_mixed_requires_cuda_amp():
    cfg = HPRLTrainingConfig(
        device="cpu",
        batch_size=8,
        replay_capacity=32,
        hidden_dim=16,
        hidden_depth=1,
        flow_likelihood_precision="mixed",
    )
    with pytest.raises(ValueError, match="requires CUDA mixed_precision"):
        ReBRACv2Agent(5, 2, cfg, device="cpu")


@pytest.mark.parametrize("seed", range(5))
def test_flow_stable_fp32_likelihood_matches_default(seed):
    torch.manual_seed(seed)
    actor = ConditionalFlowActor(7, 4, hidden_dim=32, flow_layers=3)
    obs = torch.randn(64, 7)
    action = torch.rand(64, 4).mul(0.98).add(0.01)
    base = actor.log_prob(obs, action)
    stable = actor.log_prob(obs, action, stable_fp32=True)
    torch.testing.assert_close(stable, base, rtol=0.0, atol=0.0)
    (-stable.mean()).backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in actor.parameters()
    )


@pytest.mark.parametrize("actor_cls", [GaussianActor, SimbaGaussianActor])
@pytest.mark.parametrize("deterministic", [False, True])
def test_gaussian_sample_can_skip_unused_mean_action(actor_cls, deterministic):
    torch.manual_seed(77)
    actor = actor_cls(6, 4, hidden_dim=16, depth=1)
    action, log_prob, mean_action, mean, log_std = actor.sample(
        torch.randn(12, 6),
        deterministic=deterministic,
        return_params=True,
        compute_log_prob=False,
        compute_mean_action=False,
    )
    assert action.shape == (12, 4)
    assert log_prob is None
    assert mean_action is None
    assert mean.shape == log_std.shape == action.shape
    assert torch.isfinite(action).all()


def test_default_compile_policy_is_algorithm_aware_auto():
    cfg = HPRLTrainingConfig()
    assert cfg.compile_mode == "auto"
    assert cfg.flow_likelihood_precision == "auto"

@pytest.mark.parametrize("seed", range(5))
def test_flow_paired_surfaces_match_separate_surfaces(seed):
    torch.manual_seed(seed)
    actor = ConditionalFlowActor(7, 4, hidden_dim=32, flow_layers=3)
    obs = torch.randn(32, 7)
    data_action = torch.rand(32, 4).mul(0.98).add(0.01)

    torch.manual_seed(10_000 + seed)
    sampled, sample_log_prob, _ = actor.sample(obs, compute_log_prob=True)
    data_log_prob = actor.log_prob(obs, data_action)

    torch.manual_seed(10_000 + seed)
    paired_sampled, paired_sample_log_prob, paired_data_log_prob = (
        actor.sample_and_data_log_prob(obs, data_action)
    )
    torch.testing.assert_close(paired_sampled, sampled, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(paired_sample_log_prob, sample_log_prob, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(paired_data_log_prob, data_log_prob, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("algorithm", ["fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2"])
def test_horizon_aware_auto_compile_policy_on_cuda_device_object(algorithm):
    threshold = compile_break_even_updates(algorithm, "rtx5070_laptop")
    assert threshold is not None
    assert resolve_compile_mode(
        "auto", algorithm, torch.device("cuda:0"),
        expected_updates=threshold * 2, hardware_profile="rtx5070_laptop",
    ) == "reduce-overhead"
    assert resolve_compile_mode(
        "auto", algorithm, torch.device("cuda:0"),
        expected_updates=max(0, threshold - 1), hardware_profile="rtx5070_laptop",
    ) == "off"
    eager_expected = {
        "fast_td3": 1, "fast_dsac": 16, "simba_sac": 4, "xqc": 1, "rebrac_v2": 16
    }[algorithm]
    compiled_expected = {
        "fast_td3": 8, "fast_dsac": 1, "simba_sac": 32, "xqc": 1, "rebrac_v2": 16
    }[algorithm]
    assert resolve_host_interop_threads(
        0, algorithm, torch.device("cuda:0"), "rtx5070_laptop", compile_mode="off"
    ) == eager_expected
    assert resolve_host_interop_threads(
        0, algorithm, torch.device("cuda:0"), "rtx5070_laptop",
        compile_mode="reduce-overhead"
    ) == compiled_expected


@pytest.mark.parametrize("seed", range(5))
def test_hyperspherical_normalized_input_fastpath_matches_regular_forward(seed):
    torch.manual_seed(30_000 + seed)
    layer = HypersphericalLinear(16, 12)
    raw = torch.randn(64, 16)
    normalized = torch.nn.functional.normalize(raw, dim=-1)
    regular = layer(normalized)
    fast = layer.forward_normalized(normalized)
    torch.testing.assert_close(fast, regular, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("seed", range(5))
def test_simba_residual_normalized_fastpath_preserves_gradients(seed):
    import copy

    torch.manual_seed(40_000 + seed)
    fast_block = SimbaResidualBlock(16, expansion=2)
    old_block = copy.deepcopy(fast_block)
    raw_fast = torch.randn(32, 16, requires_grad=True)
    raw_old = raw_fast.detach().clone().requires_grad_()
    x_fast = torch.nn.functional.normalize(raw_fast, dim=-1)
    x_old = torch.nn.functional.normalize(raw_old, dim=-1)

    fast = fast_block(x_fast)
    old = torch.nn.functional.normalize(
        x_old + old_block.fc2(torch.nn.functional.silu(old_block.fc1(x_old))),
        dim=-1,
    )
    torch.testing.assert_close(fast, old, rtol=2e-6, atol=2e-6)
    fast.square().mean().backward()
    old.square().mean().backward()
    torch.testing.assert_close(raw_fast.grad, raw_old.grad, rtol=2e-5, atol=2e-6)
    for fast_parameter, old_parameter in zip(
        fast_block.parameters(), old_block.parameters(), strict=True
    ):
        torch.testing.assert_close(
            fast_parameter.grad, old_parameter.grad, rtol=2e-5, atol=2e-6
        )


def test_rtx5070_rebrac_auto_precision_uses_fp32():
    assert resolve_rebrac_flow_precision(
        "auto", "cuda:0", "rtx5070_laptop", mixed_precision_enabled=True
    ) == "fp32"
    assert resolve_rebrac_flow_precision(
        "auto", "cuda:0", "generic_cuda", mixed_precision_enabled=True
    ) == "mixed"
    assert resolve_rebrac_flow_precision(
        "fp32", "cuda:0", "rtx5070_laptop", mixed_precision_enabled=True
    ) == "fp32"


def test_rebrac_v20_cold_break_even_remains_3000_until_v22_cold_gate():
    # V2.0 cold-like acceptance remains the conservative production threshold.
    assert compile_break_even_updates("rebrac_v2", "rtx5070_laptop", "cold") == 3000


def test_v21_warm_cache_break_even_uses_corrected_startup_estimator():
    # V2.1 ReBRAC, same inter-op=16, 50 warmup updates. The corrected estimator
    # removes the 50 steady-state compiled updates from warmup_seconds first.
    startup = estimate_compile_startup_seconds(
        compiled_warmup_seconds=3.410115992000792,
        compiled_updates_per_second=126.42645499439992,
        warmup_iterations=50,
    )
    assert 2.9 < startup < 3.1
    threshold = estimate_compile_break_even_updates(
        eager_updates_per_second=44.47954766380794,
        compiled_updates_per_second=126.42645499439992,
        compiled_warmup_seconds=3.410115992000792,
        warmup_iterations=50,
        quantum=100,
    )
    assert threshold == 300
    # V2.2 true warm-cache hardware calibration supersedes the V2.1 estimate.
    assert compile_break_even_updates("rebrac_v2", "rtx5070_laptop", "warm") == 400


def test_compile_break_even_returns_none_when_compile_is_not_faster():
    assert estimate_compile_break_even_updates(
        eager_updates_per_second=100.0,
        compiled_updates_per_second=99.0,
        eager_warmup_seconds=1.0,
        compiled_warmup_seconds=10.0,
    ) is None


@pytest.mark.parametrize(
    "algorithm,cold,warm",
    [
        ("fast_td3", 7500, 900),
        ("fast_dsac", 6000, 500),
        ("simba_sac", 3000, 300),
        ("xqc", 6000, 1100),
        ("rebrac_v2", 3000, 400),
    ],
)
def test_rtx5070_dual_cache_compile_thresholds(algorithm, cold, warm):
    assert compile_policy_thresholds(algorithm, "rtx5070_laptop") == {"cold": cold, "warm": warm}
    assert resolve_compile_mode(
        "auto", algorithm, "cuda:0", expected_updates=warm,
        hardware_profile="rtx5070_laptop", compile_cache_state="warm",
    ) == "reduce-overhead"
    assert resolve_compile_mode(
        "auto", algorithm, "cuda:0", expected_updates=max(0, warm - 1),
        hardware_profile="rtx5070_laptop", compile_cache_state="warm",
    ) == "off"


def test_compile_cache_auto_is_conservative_cold():
    assert resolve_compile_mode(
        "auto", "fast_td3", "cuda:0", expected_updates=1000,
        hardware_profile="rtx5070_laptop", compile_cache_state="auto",
    ) == "off"
    assert resolve_compile_mode(
        "auto", "fast_td3", "cuda:0", expected_updates=1000,
        hardware_profile="rtx5070_laptop", compile_cache_state="warm",
    ) == "reduce-overhead"


def test_rtx5070_host_profile_metadata_is_exposed():
    info = host_interop_profile_info("xqc", "rtx5070_laptop", compile_mode="reduce-overhead")
    assert info is not None
    assert info["threads"] == 1
    assert info["confidence"] == "low"
    # XQC thread=1 is a low-confidence fallback; candidate margin is not attributed to it.
    assert float(info["margin_pct"]) == 0.0
