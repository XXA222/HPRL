from __future__ import annotations

from datetime import UTC, datetime

import pytest

from freqtrade.freqai.hedge_rl.execution_models import (
    CancelLatencyPolicy,
    ExecutionAuditTrail,
    ExecutionEventType,
    LatencyModel,
    RejectionModel,
    adverse_selection_cost,
    apply_liquidity_cap,
    estimate_maker_taker_probability,
    partial_fill_schedule,
    spread_impact_bps,
    volatility_adjusted_slippage_bps,
)


def test_round61_volatility_adjusted_slippage_scales_and_caps():
    assert (
        volatility_adjusted_slippage_bps(
            base_bps=1,
            realized_volatility=0.01,
            reference_volatility=0.01,
        )
        == 1
    )
    assert (
        volatility_adjusted_slippage_bps(
            base_bps=1,
            realized_volatility=0.03,
            reference_volatility=0.01,
            sensitivity=2,
        )
        == 5
    )
    assert (
        volatility_adjusted_slippage_bps(
            base_bps=100,
            realized_volatility=1,
            reference_volatility=0.01,
            maximum_bps=250,
        )
        == 250
    )


def test_round62_spread_impact_increases_with_participation():
    low = spread_impact_bps(bid=99, ask=101, participation_rate=0.01)
    high = spread_impact_bps(bid=99, ask=101, participation_rate=1.0)
    assert low > 0 and high > low
    with pytest.raises(ValueError):
        spread_impact_bps(bid=101, ask=99, participation_rate=0.1)


def test_round63_liquidity_cap_limits_candle_participation():
    result = apply_liquidity_cap(requested_quantity=100, candle_volume=200, max_participation=0.1)
    assert result.executable_quantity == 20
    assert result.capped and result.participation_rate == 0.1


def test_round64_latency_model_is_deterministic_and_nonnegative():
    model = LatencyModel(mean_milliseconds=10, jitter_milliseconds=50, minimum_milliseconds=2)
    assert model.sample(seed=602) == model.sample(seed=602)
    assert model.sample(seed=602) >= 2


def test_round65_maker_taker_probabilities_sum_to_one():
    passive = estimate_maker_taker_probability(distance_to_mid_bps=1, urgency=0, queue_pressure=1)
    urgent = estimate_maker_taker_probability(distance_to_mid_bps=1, urgency=1, queue_pressure=-1)
    assert passive.maker_probability + passive.taker_probability == pytest.approx(1)
    assert passive.maker_probability > urgent.maker_probability


def test_round66_cancel_latency_policy_blocks_churn():
    policy = CancelLatencyPolicy(minimum_age_ms=250, replacement_cooldown_ms=500)
    assert not policy.can_cancel(order_age_ms=249)
    assert policy.can_cancel(order_age_ms=250)
    assert not policy.can_replace(last_replace_age_ms=499)
    assert policy.can_replace(last_replace_age_ms=500)


def test_round67_rejection_model_probability_and_sampling_are_deterministic():
    model = RejectionModel(base_probability=0.2, stress_multiplier=2)
    assert model.probability(stress=0) == 0.2
    assert model.probability(stress=1) == pytest.approx(0.6)
    assert model.rejected(stress=0.5, seed=602) == model.rejected(stress=0.5, seed=602)


def test_round68_partial_fill_schedule_sums_exactly_and_front_loads():
    fills = partial_fill_schedule(10, parts=4, front_load=2)
    assert sum(fills) == pytest.approx(10)
    assert fills[0] > fills[1] > fills[2] > fills[3] > 0


def test_round69_adverse_selection_cost_has_correct_trade_direction():
    assert adverse_selection_cost(quantity=2, fill_price=101, post_fill_mark=100, is_buy=True) == 2
    assert adverse_selection_cost(quantity=2, fill_price=99, post_fill_mark=100, is_buy=False) == 2
    assert adverse_selection_cost(quantity=2, fill_price=99, post_fill_mark=100, is_buy=True) == -2


def test_round70_execution_audit_trail_hash_chain_verifies():
    trail = ExecutionAuditTrail()
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    first = trail.append(
        ExecutionEventType.PREPARED,
        order_id="o1",
        payload={"qty": 1},
        timestamp=timestamp,
    )
    second = trail.append(
        ExecutionEventType.SUBMITTED,
        order_id="o1",
        payload={"venue": "paper"},
        timestamp=timestamp,
    )
    assert first.sequence == 0 and second.previous_hash == first.event_hash
    assert trail.verify()
    assert len(trail.events()) == 2
