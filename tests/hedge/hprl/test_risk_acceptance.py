import pytest

from freqtrade.hedge.hprl.risk_acceptance import (
    BaselineRiskMetrics,
    RiskLearningAcceptanceConfig,
    RiskLearningTrace,
    evaluate_risk_learning,
)


def _adaptive_trace(repeats: int = 10, *, projected: bool = False):
    levels_pattern = [0, 1, 2, 3, 2, 1, 0, 1]
    margin_pattern = [0.0, 0.05, 0.12, 0.25, 0.12, 0.05, 0.0, 0.05]
    return_pattern = [0.0002, 0.0010, 0.0011, -0.0008, -0.0003, 0.0003, 0.0002, 0.0010]
    drawdown_pattern = [0.0, 0.0, 0.0, 0.0008, 0.0011, 0.0009, 0.0006, 0.0001]
    n = repeats * len(levels_pattern)
    return RiskLearningTrace(
        equity_return=return_pattern * repeats,
        gross_margin=margin_pattern * repeats,
        drawdown=drawdown_pattern * repeats,
        level_index=levels_pattern * repeats,
        turnover=[0.0] * n,
        projected=[projected] * n,
    )


def _config(**kwargs):
    defaults = dict(
        min_steps=32,
        min_distinct_levels=3,
        max_single_level_fraction=0.95,
        min_active_fraction=0.05,
        max_projection_fraction=0.20,
        stress_drawdown_quantile=0.75,
        max_stress_to_calm_margin_ratio=1.05,
        min_post_loss_derisk_fraction=0.45,
        min_scale_in_success_rate=0.50,
        drawdown_penalty=0.2,
        cvar_penalty=1.0,
        turnover_penalty=0.0,
        liquidation_penalty=1.0,
        min_baseline_utility_edge=0.0,
    )
    defaults.update(kwargs)
    return RiskLearningAcceptanceConfig(**defaults)


def test_profitable_static_heavy_policy_is_not_learning():
    n = 64
    trace = RiskLearningTrace(
        equity_return=[0.0002] * n,
        gross_margin=[0.40] * n,
        drawdown=[0.0] * n,
        level_index=[4] * n,
    )
    report = evaluate_risk_learning(
        trace,
        baselines=[BaselineRiskMetrics("flat", 0.0, 0.0, 0.0)],
        config=_config(),
    )
    assert report.verdict == "FAIL"
    assert "single_level_collapse" in report.reasons
    assert report.economic_pass is True
    assert report.behavioral_pass is False


def test_adaptive_policy_passes_with_weaker_baseline():
    report = evaluate_risk_learning(
        _adaptive_trace(),
        baselines=[BaselineRiskMetrics("static_light", 0.005, 0.01, 0.001)],
        config=_config(),
    )
    assert report.verdict == "PASS"
    assert report.behavioral_pass is True
    assert report.economic_pass is True
    assert report.baseline_pass is True
    assert report.distinct_levels >= 4
    assert report.scale_in_success_rate >= 0.5


def test_good_behavior_without_baseline_is_inconclusive():
    report = evaluate_risk_learning(_adaptive_trace(), config=_config())
    assert report.verdict == "INCONCLUSIVE"
    assert report.behavioral_pass is True
    assert report.economic_pass is True
    assert report.baseline_pass is None
    assert "baseline_evidence_missing" in report.reasons


def test_excessive_projection_fails_acceptance():
    report = evaluate_risk_learning(
        _adaptive_trace(projected=True),
        baselines=[BaselineRiskMetrics("flat", 0.0, 0.0, 0.0)],
        config=_config(),
    )
    assert report.verdict == "FAIL"
    assert "excessive_risk_projection" in report.reasons


def test_invalid_trace_length_is_rejected():
    trace = RiskLearningTrace(
        equity_return=[0.001] * 32,
        gross_margin=[0.05] * 31,
        drawdown=[0.0] * 32,
        level_index=[1] * 32,
    )
    with pytest.raises(ValueError, match="gross_margin length"):
        evaluate_risk_learning(trace, config=_config(require_baselines=False))
