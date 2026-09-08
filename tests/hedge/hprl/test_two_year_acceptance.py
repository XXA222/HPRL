from freqtrade.hedge.hprl.two_year_acceptance import (
    GIB,
    RuntimeScaleGateConfig,
    RuntimeScaleSample,
    TwoYearCapacityConfig,
    assess_runtime_scale,
    build_two_year_capacity_plan,
    canonical_two_year_minute_bars,
)


def test_canonical_two_year_minute_bar_count():
    assert canonical_two_year_minute_bars() == 1_052_640


def test_capacity_plan_prefers_windowed_cpu_replay_and_fits_example_budget():
    plan = build_two_year_capacity_plan(
        TwoYearCapacityConfig(
            symbols=3,
            features_per_symbol=100,
            observation_dim=310,
            action_dim=6,
            replay_capacity=1_000_000,
            host_memory_bytes=32 * GIB,
            cuda_memory_bytes=8 * GIB,
        )
    )
    assert plan.bars == 1_052_640
    assert plan.market_dataset_bytes > plan.cuda_window_bytes
    assert plan.recommended_dataset_mode == "windowed"
    assert plan.recommended_replay_device == "cpu"
    assert plan.host_fit is True
    assert plan.cuda_fit is True


def test_complete_stable_history_passes_runtime_gate():
    samples = [
        RuntimeScaleSample(0, 0.1, rss_bytes=1_000_000_000, cuda_reserved_bytes=1_000_000_000),
        RuntimeScaleSample(250, 1.1, rss_bytes=1_010_000_000, cuda_reserved_bytes=1_010_000_000),
        RuntimeScaleSample(500, 2.1, rss_bytes=1_020_000_000, cuda_reserved_bytes=1_020_000_000),
        RuntimeScaleSample(750, 3.1, rss_bytes=1_030_000_000, cuda_reserved_bytes=1_030_000_000),
        RuntimeScaleSample(1000, 4.1, rss_bytes=1_040_000_000, cuda_reserved_bytes=1_040_000_000),
    ]
    report = assess_runtime_scale(
        samples,
        config=RuntimeScaleGateConfig(expected_steps=1000, max_projected_hours=1.0),
    )
    assert report.verdict == "PASS"
    assert report.full_history_observed is True
    assert report.throughput_stable is True
    assert report.memory_stable is True


def test_partial_run_is_only_provisional_and_degradation_fails():
    stable_partial = [
        RuntimeScaleSample(0, 0.1, rss_bytes=1_000_000_000, cuda_reserved_bytes=1_000_000_000),
        RuntimeScaleSample(200, 1.1, rss_bytes=1_010_000_000, cuda_reserved_bytes=1_010_000_000),
        RuntimeScaleSample(400, 2.1, rss_bytes=1_020_000_000, cuda_reserved_bytes=1_020_000_000),
    ]
    provisional = assess_runtime_scale(
        stable_partial,
        config=RuntimeScaleGateConfig(expected_steps=1000, max_projected_hours=1.0),
    )
    assert provisional.verdict == "PROVISIONAL"
    assert "full_history_not_observed" in provisional.reasons

    degraded = [
        RuntimeScaleSample(0, 0.1, rss_bytes=1_000_000_000, cuda_reserved_bytes=1_000_000_000),
        RuntimeScaleSample(250, 1.1, rss_bytes=1_010_000_000, cuda_reserved_bytes=1_010_000_000),
        RuntimeScaleSample(500, 2.1, rss_bytes=1_020_000_000, cuda_reserved_bytes=1_020_000_000),
        RuntimeScaleSample(625, 4.1, rss_bytes=1_030_000_000, cuda_reserved_bytes=1_030_000_000),
        RuntimeScaleSample(750, 6.1, rss_bytes=1_040_000_000, cuda_reserved_bytes=1_040_000_000),
    ]
    failed = assess_runtime_scale(
        degraded,
        config=RuntimeScaleGateConfig(expected_steps=750, max_projected_hours=1.0),
    )
    assert failed.verdict == "FAIL"
    assert "throughput_tail_degradation" in failed.reasons
