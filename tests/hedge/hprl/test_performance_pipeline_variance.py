from __future__ import annotations

import math

import pytest
import torch

from freqtrade.hedge.hprl.action_space import configure_agent_action_levels
from freqtrade.hedge.hprl.algorithms.base import FrozenModulePlan, PolyakUpdatePlan, soft_update
from freqtrade.hedge.hprl.calibration import (
    balanced_interleaved_orders,
    mad_inlier_mask,
    paired_bootstrap_superiority_probability,
    paired_speedup_summary,
    paired_winner_confidence,
    robust_distribution_summary,
)
from freqtrade.hedge.hprl.config import HPRLActionConfig, HPRLTrainingConfig
from freqtrade.hedge.hprl.performance import (
    agent_finite_state,
    compile_break_even_updates,
    condition_cuda_device,
    resolve_compile_scope,
    summarize_profile_operations,
)
from freqtrade.hedge.hprl.pipeline_benchmark import benchmark_training_pipeline
from freqtrade.hedge.hprl.registry import create_agent
from freqtrade.hedge.hprl.replay import ReplayBatch


def _config(algorithm: str) -> HPRLTrainingConfig:
    return HPRLTrainingConfig(
        algorithm=algorithm,
        device="cpu",
        replay_device="same",
        batch_size=16,
        replay_capacity=64,
        warmup_steps=0,
        hidden_dim=32,
        hidden_depth=1,
        compile_mode="off",
        metrics_interval=1000,
    )


def _batch() -> ReplayBatch:
    return ReplayBatch(
        obs=torch.randn(16, 8),
        action=torch.rand(16, 4),
        reward=torch.randn(16, 1) * 0.01,
        next_obs=torch.randn(16, 8),
        done=torch.zeros(16, 1),
    )


def test_mad_inlier_mask_rejects_single_large_outlier() -> None:
    mask = mad_inlier_mask([100.0, 101.0, 99.0, 100.5, 1000.0])
    assert mask == (True, True, True, True, False)


def test_mad_zero_keeps_small_non_identical_sample() -> None:
    assert mad_inlier_mask([1.0, 1.0, 2.0]) == (True, True, True)


def test_robust_distribution_reports_outlier_and_location() -> None:
    result = robust_distribution_summary([10, 10.1, 9.9, 10.05, 100])
    assert result["outliers"] == 1
    assert 9.9 <= result["robust_median"] <= 10.1
    assert result["winsorized_mean"] < 11


def test_paired_speedup_uses_matched_ratios() -> None:
    result = paired_speedup_summary([110, 220, 330], [100, 200, 300])
    assert result["count"] == 3
    assert result["median_speedup"] == pytest.approx(1.1)


def test_paired_bootstrap_is_deterministic() -> None:
    first = paired_bootstrap_superiority_probability([2, 3, 4], [1, 1, 1], samples=500, seed=7)
    second = paired_bootstrap_superiority_probability([2, 3, 4], [1, 1, 1], samples=500, seed=7)
    assert first == second == 1.0


def test_paired_confidence_detects_clear_winner() -> None:
    result = paired_winner_confidence(
        [120, 121, 119, 122, 120], [100, 101, 99, 100, 100], bootstrap_samples=1000
    )
    assert result["label"] in {"medium", "high"}
    assert result["paired_margin_pct"] > 15


@pytest.mark.parametrize("count", [2, 3, 4, 5])
def test_balanced_interleaved_orders_cover_each_candidate_per_round(count: int) -> None:
    candidates = list(range(1, count + 1))
    orders = balanced_interleaved_orders(candidates, 9, seed=123)
    assert len(orders) == 9
    assert all(sorted(order) == candidates for order in orders)
    assert len(set(orders)) > 1


@pytest.mark.parametrize(
    ("algorithm", "cold", "warm"),
    [
        ("fast_td3", 7500, 900),
        ("fast_dsac", 6000, 500),
        ("simba_sac", 3000, 300),
        ("xqc", 6000, 1100),
        ("rebrac_v2", 3000, 400),
    ],
)
def test_v22_rtx5070_thresholds_are_frozen(algorithm: str, cold: int, warm: int) -> None:
    assert compile_break_even_updates(algorithm, "rtx5070_laptop", "cold") == cold
    assert compile_break_even_updates(algorithm, "rtx5070_laptop", "warm") == warm


def test_compile_scope_validation() -> None:
    assert HPRLTrainingConfig(compile_scope="loss").compile_scope == "loss"
    with pytest.raises(Exception):
        HPRLTrainingConfig(compile_scope="bad")


def test_compile_scope_auto_remains_conservative_before_v23_gpu_gate() -> None:
    assert resolve_compile_scope("auto", "fast_dsac") == "module"
    assert resolve_compile_scope("loss", "fast_dsac") == "loss"


def test_polyak_update_plan_matches_reference() -> None:
    torch.manual_seed(1)
    source1 = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    target1 = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    source2 = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    target2 = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.BatchNorm1d(4))
    source2.load_state_dict(source1.state_dict())
    target2.load_state_dict(target1.state_dict())
    soft_update(target1, source1, 0.1, foreach=False)
    PolyakUpdatePlan(target2, source2, foreach=False).step(0.1)
    for left, right in zip(target1.state_dict().values(), target2.state_dict().values(), strict=True):
        assert torch.equal(left, right)


def test_frozen_module_plan_restores_state() -> None:
    module = torch.nn.Linear(3, 2)
    plan = FrozenModulePlan(module, eval_mode=True)
    assert module.training
    with plan.frozen():
        assert not module.training
        assert all(not p.requires_grad for p in module.parameters())
    assert module.training
    assert all(p.requires_grad for p in module.parameters())


@pytest.mark.parametrize("algorithm", ["fast_dsac", "simba_sac", "rebrac_v2"])
def test_target_agents_expose_cached_orchestration_plans(algorithm: str) -> None:
    agent = create_agent(algorithm, 8, 4, _config(algorithm), device="cpu")
    configure_agent_action_levels(agent, HPRLActionConfig().level_count)
    assert isinstance(agent._actor_params, tuple)
    assert isinstance(agent._critic_params, tuple)
    assert isinstance(agent._critic_polyak, PolyakUpdatePlan)
    metrics = agent.update(_batch(), collect_metrics=True)
    assert metrics.values
    assert all(math.isfinite(float(value)) for value in metrics.values.values())


def test_condition_cuda_device_is_noop_on_cpu() -> None:
    result = condition_cuda_device("cpu", milliseconds=100)
    assert result["enabled"] is False
    assert result["iterations"] == 0


def test_profile_operation_summary_counts_launch_surfaces() -> None:
    rows = [
        {"name": "cudaLaunchKernel", "count": 10},
        {"name": "cudaGraphLaunch", "count": 2},
        {"name": "aten::_foreach_lerp_", "count": 3},
        {"name": "aten::_to_copy", "count": 4},
        {"name": "Optimizer.step#Adam.step", "count": 5},
        {"name": "Torch-Compiled Region: 0/0", "count": 6},
    ]
    result = summarize_profile_operations(rows)["categories"]
    assert result["cuda_kernel_launches"] == 10
    assert result["cuda_graph_launches"] == 2
    assert result["foreach_ops"] == 3
    assert result["dtype_copy_ops"] == 4
    assert result["optimizer_steps"] == 5
    assert result["compiled_regions"] == 6


def test_cpu_pipeline_benchmark_covers_logging_and_checkpoint() -> None:
    cfg = _config("fast_td3")
    agent = create_agent("fast_td3", 8, 4, cfg, device="cpu")
    configure_agent_action_levels(agent, HPRLActionConfig().level_count)
    result = benchmark_training_pipeline(
        agent,
        obs_dim=8,
        action_dim=4,
        batch_size=16,
        iterations=3,
        warmup=1,
        replay_capacity=64,
        replay_device="same",
        metrics_interval=1,
        checkpoint_interval=2,
        diagnostic_iterations=1,
    )
    assert result.samples == 48
    assert result.metrics_events == 3
    assert result.checkpoints == 1
    assert result.checkpoint_bytes > 0
    assert result.samples_per_second > 0
    assert result.stage_diagnostics["checkpoint_bytes"] > 0


def test_agent_finite_state_detects_nonfinite_parameter() -> None:
    agent = create_agent("fast_td3", 8, 4, _config("fast_td3"), device="cpu")
    assert agent_finite_state(agent)["parameters_finite"] is True
    parameter = next(agent.actor.parameters())
    original = parameter.detach().clone()
    try:
        with torch.no_grad():
            parameter.reshape(-1)[0] = float("nan")
        result = agent_finite_state(agent)
        assert result["parameters_finite"] is False
        assert result["nonfinite"]
    finally:
        with torch.no_grad():
            parameter.copy_(original)


def test_rtx5070_auto_compile_scope_promotes_validated_loss_surfaces() -> None:
    for algorithm in ("fast_dsac", "simba_sac", "rebrac_v2"):
        assert resolve_compile_scope(
            "auto", algorithm, hardware_profile="rtx5070_laptop"
        ) == "loss"
    for algorithm in ("fast_td3", "xqc"):
        assert resolve_compile_scope(
            "auto", algorithm, hardware_profile="rtx5070_laptop"
        ) == "module"
    assert resolve_compile_scope(
        "auto", "fast_dsac", hardware_profile="generic_cuda"
    ) == "module"


def test_loss_post_scope_is_explicit_only() -> None:
    cfg = HPRLTrainingConfig(compile_scope="loss_post")
    assert cfg.compile_scope == "loss_post"
    assert resolve_compile_scope(
        "loss_post", "fast_dsac", hardware_profile="rtx5070_laptop"
    ) == "loss_post"


def test_optimizer_step_plan_matches_precision_backward_step() -> None:
    from freqtrade.hedge.hprl.algorithms.base import OptimizerStepPlan
    from freqtrade.hedge.hprl.device import PrecisionManager

    torch.manual_seed(91)
    left = torch.nn.Linear(4, 2)
    right = torch.nn.Linear(4, 2)
    right.load_state_dict(left.state_dict())
    left_opt = torch.optim.Adam(left.parameters(), lr=1e-3)
    right_opt = torch.optim.Adam(right.parameters(), lr=1e-3)
    precision_left = PrecisionManager("cpu")
    precision_right = PrecisionManager("cpu")
    x = torch.randn(8, 4)
    target = torch.randn(8, 2)
    left_loss = (left(x) - target).square().mean()
    right_loss = (right(x) - target).square().mean()
    left_norm = precision_left.backward_step(
        left_loss, left_opt, tuple(left.parameters()), 10.0
    )
    plan = OptimizerStepPlan(
        precision_right, right_opt, tuple(right.parameters()), 10.0
    )
    right_norm = plan.step(right_loss)
    assert torch.equal(left_norm, right_norm)
    for lp, rp in zip(left.parameters(), right.parameters(), strict=True):
        assert torch.equal(lp, rp)


def test_async_artifact_writer_checkpoint_is_snapshot_consistent(tmp_path) -> None:
    from freqtrade.hedge.hprl.async_io import AsyncArtifactWriter
    from freqtrade.hedge.hprl.checkpoint import load_checkpoint

    cfg = _config("fast_td3")
    source = create_agent("fast_td3", 8, 4, cfg, device="cpu")
    target = create_agent("fast_td3", 8, 4, cfg, device="cpu")
    expected = {k: v.detach().clone() for k, v in source.actor.state_dict().items()}
    path = tmp_path / "async.pt"
    with AsyncArtifactWriter(queue_size=2) as writer:
        writer.submit_metrics(tmp_path / "train.jsonl", {"loss": 1.25})
        writer.submit_checkpoint(path, source, {"kind": "async"})
        with torch.no_grad():
            for parameter in source.actor.parameters():
                parameter.add_(10.0)
    assert path.exists()
    metadata = load_checkpoint(path, target)
    assert metadata == {"kind": "async"}
    for key, value in target.actor.state_dict().items():
        assert torch.equal(value, expected[key])


def test_replay_staging_identity_is_stable_across_reuse() -> None:
    from freqtrade.hedge.hprl.replay import TensorReplayBuffer

    buffer = TensorReplayBuffer(64, 8, 4, device="cpu", pin_memory=False)
    for _ in range(4):
        buffer.add(
            torch.randn(16, 8), torch.rand(16, 4), torch.randn(16, 1),
            torch.randn(16, 8), torch.zeros(16, 1),
        )
    buffer.sample_reusable(16)
    before = buffer.staging_identity()
    for _ in range(10):
        buffer.sample_reusable(16)
    after = buffer.staging_identity()
    assert before == after
    buffer.release()


def test_sustained_cpu_benchmark_bounds_checkpoint_retention() -> None:
    from freqtrade.hedge.hprl.sustained_benchmark import benchmark_sustained_training

    cfg = _config("fast_td3")
    agent = create_agent("fast_td3", 8, 4, cfg, device="cpu")
    configure_agent_action_levels(agent, HPRLActionConfig().level_count)
    result = benchmark_sustained_training(
        agent, obs_dim=8, action_dim=4, batch_size=16, iterations=20, warmup=1,
        window_size=5, replay_capacity=64, replay_device="same", metrics_interval=5,
        checkpoint_interval=10, checkpoint_keep_last=1, artifact_queue_size=2,
    )
    assert result.parameters_finite
    assert result.replay_staging_stable
    assert result.checkpoints_submitted == 2
    assert result.checkpoints_retained == 1
    assert result.checkpoint_deletions == 1
    assert len(result.windows) == 4


def test_xqc_profiled_update_matches_production_update() -> None:
    from freqtrade.hedge.hprl.stage_profiling import StageRecorder

    cfg = _config("xqc")
    torch.manual_seed(701)
    left = create_agent("xqc", 8, 4, cfg, device="cpu")
    torch.manual_seed(701)
    right = create_agent("xqc", 8, 4, cfg, device="cpu")
    configure_agent_action_levels(left, HPRLActionConfig().level_count)
    configure_agent_action_levels(right, HPRLActionConfig().level_count)
    right.actor.load_state_dict(left.actor.state_dict())
    right.critic.load_state_dict(left.critic.state_dict())
    right.critic_target.load_state_dict(left.critic_target.state_dict())
    right.actor_opt.load_state_dict(left.actor_opt.state_dict())
    right.critic_opt.load_state_dict(left.critic_opt.state_dict())
    right.alpha_opt.load_state_dict(left.alpha_opt.state_dict())
    right.log_alpha.data.copy_(left.log_alpha.data)
    batch = _batch()
    torch.manual_seed(991)
    left_metrics = left.update(batch, collect_metrics=True)
    torch.manual_seed(991)
    right_metrics = right.profile_update_stages(
        batch, StageRecorder("cpu"), collect_metrics=True
    )
    assert left_metrics.values == right_metrics.values
    for name in ("actor", "critic", "critic_target"):
        lstate = getattr(left, name).state_dict()
        rstate = getattr(right, name).state_dict()
        assert lstate.keys() == rstate.keys()
        assert all(torch.equal(lstate[key], rstate[key]) for key in lstate)
    assert torch.equal(left.log_alpha, right.log_alpha)
