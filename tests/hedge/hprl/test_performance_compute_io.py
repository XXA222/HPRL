from __future__ import annotations

from pathlib import Path

import pytest

from freqtrade.hedge.hprl.artifact_policy import (
    ArtifactWorkload,
    resolve_artifact_io_mode,
)
from freqtrade.hedge.hprl.calibration import paired_scope_confidence_decision
from freqtrade.hedge.hprl.config import HPRLTrainingConfig
from freqtrade.hedge.hprl.performance import resolve_compile_scope


def test_paired_scope_requires_material_speedup_and_confidence():
    result = paired_scope_confidence_decision(
        [1.04, 1.05, 1.03, 1.06, 1.05, 1.04, 1.05],
        [1.0] * 7,
        min_speedup=1.03,
        bootstrap_threshold=0.95,
    )
    assert result["promote"] is True
    weak = paired_scope_confidence_decision(
        [1.01, 1.02, 1.00, 1.03, 1.01, 1.02, 1.01], [1.0] * 7,
        min_speedup=1.03,
    )
    assert weak["promote"] is False


def test_artifact_policy_is_workload_not_algorithm_based():
    heavy = ArtifactWorkload(32_000_000, 500, 4096, 20, 10_000, 0.0)
    light = ArtifactWorkload(4_000_000, 2000, 512, 100, 10_000, 0.0)
    assert resolve_artifact_io_mode("auto", heavy).resolved == "async"
    assert resolve_artifact_io_mode("auto", light).resolved == "sync"


def test_artifact_policy_backpressure_forces_sync():
    workload = ArtifactWorkload(64_000_000, 100, 4096, 10, 20_000, 0.01)
    decision = resolve_artifact_io_mode("auto", workload)
    assert decision.resolved == "sync"
    assert "backpressure" in decision.reason


def test_artifact_policy_explicit_override():
    workload = ArtifactWorkload(1, 0, 0, 100, 1, 0.0)
    assert resolve_artifact_io_mode("async", workload).resolved == "async"
    assert resolve_artifact_io_mode("sync", workload).resolved == "sync"


def test_xqc_fused_scope_is_explicit_only():
    cfg = HPRLTrainingConfig(algorithm="xqc", compile_scope="xqc_fused")
    assert cfg.compile_scope == "xqc_fused"
    assert resolve_compile_scope("auto", "xqc", hardware_profile="rtx5070_laptop") == "module"
    assert resolve_compile_scope("xqc_fused", "xqc", hardware_profile="rtx5070_laptop") == "xqc_fused"


def test_non_xqc_auto_scope_unchanged():
    assert resolve_compile_scope("auto", "rebrac_v2", hardware_profile="rtx5070_laptop") == "loss"
    assert resolve_compile_scope("auto", "simba_sac", hardware_profile="rtx5070_laptop") == "loss"
    assert resolve_compile_scope("auto", "fast_dsac", hardware_profile="rtx5070_laptop") == "loss"


def test_xqc_stacked_categorical_reductions_match_separate():
    import torch
    from freqtrade.hedge.hprl.networks import CategoricalTwinCritic
    torch.manual_seed(7)
    critic = CategoricalTwinCritic(5, 3, hidden_dim=16, depth=1, bins=17)
    obs = torch.randn(11, 5)
    action = torch.rand(11, 3)
    l1, l2 = critic.logits(obs, action)
    q1 = critic.expectation_from_logits(l1)
    q2 = critic.expectation_from_logits(l2)
    s1, s2 = critic.twin_expectation_stacked(l1, l2)
    target = torch.randn(11, 1)
    base = critic.twin_cross_entropy(l1, l2, target)
    fused = critic.twin_cross_entropy_stacked(l1, l2, target)
    assert torch.equal(q1, s1)
    assert torch.equal(q2, s2)
    assert torch.allclose(base, fused, rtol=1e-7, atol=1e-8)


def test_orchestration_profiler_accepts_xqc_fused_scope():
    from freqtrade.hedge.hprl.cli import build_parser

    args = build_parser().parse_args([
        "perf-orchestration-profile",
        "--algorithm", "xqc",
        "--compile-scope", "xqc_fused",
    ])
    assert args.compile_scope == "xqc_fused"


def test_v252_hardware_profiler_scope_capability_contract():
    import importlib.util
    from pathlib import Path

    tool = Path(__file__).resolve().parents[3] / "tools" / "run_hprl_performance_v25_rtx5070_gate.py"
    spec = importlib.util.spec_from_file_location("hprl_v252_gate", tool)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    result = module.profiler_scope_capability(("module", "loss", "loss_post", "xqc_fused"))
    assert result["pass"] is True
    assert all(result["results"].values())
    assert result["errors"] == []
