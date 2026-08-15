from __future__ import annotations

import math

import numpy as np
import pytest

from freqtrade.freqai.hedge_rl.reward_extensions import (
    RewardExplainer,
    RewardNormalizer,
    conditional_value_at_risk,
    downside_deviation,
    drawdown_delta,
    exposure_penalty,
    funding_cost_penalty,
    invalid_action_penalty,
    safe_log_equity_return,
    turnover_penalty,
)


def test_round71_safe_log_equity_return_is_additive_and_finite():
    first = safe_log_equity_return(100, 110)
    second = safe_log_equity_return(110, 121)
    total = safe_log_equity_return(100, 121)
    assert first + second == pytest.approx(total)
    assert math.isfinite(safe_log_equity_return(0, 1))


def test_round72_downside_deviation_ignores_upside():
    result = downside_deviation([0.1, -0.1, -0.2, 0.3])
    expected = math.sqrt((0 + 0.01 + 0.04 + 0) / 4)
    assert result == pytest.approx(expected)


def test_round73_drawdown_delta_tracks_only_change_in_drawdown():
    assert drawdown_delta(
        previous_equity=100,
        previous_peak=100,
        equity=90,
        peak=100,
    ) == pytest.approx(0.1)
    assert drawdown_delta(
        previous_equity=90,
        previous_peak=100,
        equity=95,
        peak=100,
    ) == pytest.approx(-0.05)


def test_round74_turnover_penalty_scales_by_equity():
    assert turnover_penalty(traded_notional=100, equity=1000, weight=2) == pytest.approx(0.2)
    assert turnover_penalty(traded_notional=0, equity=1000, weight=2) == 0


def test_round75_exposure_penalty_applies_only_beyond_limits():
    assert exposure_penalty(gross_exposure=1, net_exposure=0.2, gross_limit=1.2, net_limit=0.6) == 0
    assert exposure_penalty(
        gross_exposure=1.8,
        net_exposure=0.9,
        gross_limit=1.2,
        net_limit=0.6,
    ) == pytest.approx(0.5)


def test_round76_funding_penalty_penalizes_payment_not_receipt():
    assert funding_cost_penalty(funding_cashflow=-10, equity=1000, weight=2) == pytest.approx(0.02)
    assert funding_cost_penalty(funding_cashflow=10, equity=1000, weight=2) == 0


def test_round77_invalid_action_penalty_escalates_repeated_errors():
    assert invalid_action_penalty(invalid=False, consecutive_invalid=99) == 0
    assert invalid_action_penalty(invalid=True, consecutive_invalid=0, base=1, escalation=0.5) == 1
    assert invalid_action_penalty(invalid=True, consecutive_invalid=4, base=1, escalation=0.5) == 3


def test_round78_conditional_value_at_risk_averages_left_tail():
    values = np.arange(-10, 10, dtype=float)
    result = conditional_value_at_risk(values, alpha=0.1)
    assert result == pytest.approx(np.mean([-10, -9]))


def test_round79_reward_normalizer_updates_online_and_clips():
    normalizer = RewardNormalizer(clip=1)
    assert normalizer.normalize(10) == 0
    second = normalizer.normalize(1000)
    assert -1 <= second <= 1
    assert normalizer.count == 2
    with pytest.raises(ValueError):
        normalizer.normalize(float("nan"))


def test_round80_reward_explainer_reconciles_components_and_clipping():
    result = RewardExplainer().aggregate(
        {"equity": 2.0, "realized": 0.5},
        {"drawdown": 1.0, "turnover": 0.25},
        clip=1.0,
    )
    assert result.checksum == pytest.approx(1.25)
    assert result.total == 1.0
    assert sum(result.contributions.values()) == pytest.approx(result.checksum)
