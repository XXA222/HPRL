from __future__ import annotations

import math

import pytest

from freqtrade.hedge.hprl.calibration import (
    bootstrap_superiority_probability,
    choose_threads_with_confidence,
    compile_cache_environment,
    distribution_summary,
    quantile,
    winner_confidence,
)
from freqtrade.hedge.hprl.config import HPRLTrainingConfig
from freqtrade.hedge.hprl.performance import (
    compile_policy_thresholds,
    estimate_compile_break_even_updates,
    estimate_compile_startup_seconds,
    normalize_compile_cache_state,
)


@pytest.mark.parametrize("q,expected", [(0.0, 1.0), (0.25, 2.0), (0.5, 3.0), (0.75, 4.0), (1.0, 5.0)])
def test_quantile_exact_grid(q, expected):
    assert quantile([1, 2, 3, 4, 5], q) == expected


def test_distribution_summary_reports_robust_statistics():
    summary = distribution_summary([10, 11, 12, 13, 30])
    assert summary["count"] == 5
    assert summary["median"] == 12
    assert summary["mad"] == 1
    assert summary["p10"] < summary["median"] < summary["p90"]
    assert summary["cv"] > 0


def test_bootstrap_superiority_is_deterministic():
    a = bootstrap_superiority_probability([10, 11, 12, 13, 14], [1, 2, 3, 4, 5], samples=500, seed=9)
    b = bootstrap_superiority_probability([10, 11, 12, 13, 14], [1, 2, 3, 4, 5], samples=500, seed=9)
    assert a == b and a > 0.99


def test_low_confidence_winner_falls_back_to_previous_threads():
    points = [
        {"status": "PASS", "cpu_interop_threads": 8, "median_updates_per_second": 101.0, "runs": [{"updates_per_second": x} for x in [97, 102, 101, 104, 96]]},
        {"status": "PASS", "cpu_interop_threads": 1, "median_updates_per_second": 100.0, "runs": [{"updates_per_second": x} for x in [98, 101, 100, 103, 97]]},
    ]
    choice = choose_threads_with_confidence(points, previous_threads=1, bootstrap_samples=500, seed=11)
    assert choice["confidence"] == "low"
    assert choice["fallback_used"] is True
    assert choice["recommended_threads"] == 1


def test_high_confidence_winner_is_adopted():
    points = [
        {"status": "PASS", "cpu_interop_threads": 8, "median_updates_per_second": 140.0, "runs": [{"updates_per_second": x} for x in [138, 139, 140, 141, 142]]},
        {"status": "PASS", "cpu_interop_threads": 1, "median_updates_per_second": 100.0, "runs": [{"updates_per_second": x} for x in [98, 99, 100, 101, 102]]},
    ]
    choice = choose_threads_with_confidence(points, previous_threads=1, bootstrap_samples=500, seed=12)
    assert choice["confidence"] == "high"
    assert choice["fallback_used"] is False
    assert choice["recommended_threads"] == 8


@pytest.mark.parametrize("state", ["cold", "warm"])
def test_compile_cache_environment_isolates_local_and_remote_caches(state):
    env = compile_cache_environment({"X": "1", "TORCHINDUCTOR_FORCE_DISABLE_CACHES": "1"}, cache_state=state, cache_dir="/tmp/hprl-v22-test")
    assert env["X"] == "1"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/hprl-v22-test"
    assert env["TRITON_CACHE_DIR"].endswith("/triton")
    assert env["TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"] == "0"
    if state == "cold":
        assert env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] == "1"
    else:
        assert "TORCHINDUCTOR_FORCE_DISABLE_CACHES" not in env


@pytest.mark.parametrize("value,expected", [("cold", "cold"), ("warm", "warm"), ("auto", "cold"), (None, "cold")])
def test_compile_cache_state_normalization_is_conservative(value, expected):
    assert normalize_compile_cache_state(value) == expected


def test_corrected_startup_estimator_removes_warmup_updates():
    total = 3.0 + 50 / 125.0
    assert math.isclose(
        estimate_compile_startup_seconds(compiled_warmup_seconds=total, compiled_updates_per_second=125.0, warmup_iterations=50),
        3.0,
        abs_tol=1e-12,
    )


def test_corrected_break_even_quantizes_after_safety_margin():
    threshold = estimate_compile_break_even_updates(
        eager_updates_per_second=50.0,
        compiled_updates_per_second=100.0,
        compiled_warmup_seconds=2.5 + 50 / 100.0,
        warmup_iterations=50,
        safety_margin=1.25,
        quantum=100,
    )
    assert threshold == 400


def test_rtx5070_warm_thresholds_are_distinct_not_collapsed_to_500():
    values = [compile_policy_thresholds(name, "rtx5070_laptop")["warm"] for name in ("fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2")]
    assert values == [900, 500, 300, 1100, 400]
    assert len(set(values)) == 5


def test_training_config_defaults_to_cold_cache_policy():
    cfg = HPRLTrainingConfig(batch_size=8, replay_capacity=32, hidden_dim=16, hidden_depth=1)
    assert cfg.compile_cache_state == "cold"


@pytest.mark.parametrize("bad", ["hot", "disk", "none"])
def test_training_config_rejects_unknown_cache_state(bad):
    with pytest.raises(Exception):
        HPRLTrainingConfig(batch_size=8, replay_capacity=32, hidden_dim=16, hidden_depth=1, compile_cache_state=bad)
