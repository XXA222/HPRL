from __future__ import annotations

import numpy as np
import pytest

from freqtrade.freqai.hedge_rl.actions import DEFAULT_ACTION_CATALOG, HedgeActions
from freqtrade.freqai.hedge_rl.env_extensions import (
    EpisodeMetrics,
    HedgeEnvSnapshot,
    ScenarioInjector,
    VectorObservationAdapter,
    assert_action_mask_invariants,
    audit_next_bar_execution,
    invalid_action_fallback,
    terminal_reason,
    truncation_reason,
    verify_seed_determinism,
)
from freqtrade.freqai.hedge_rl.state import HedgeAccountState, HedgeLegSide, HedgeLegState
from tests.hedge.mlrl.advanced_helpers import env_factory, synthetic_prices


def test_round81_environment_seed_determinism_replays_trajectory():
    assert verify_seed_determinism(
        lambda: env_factory(random_start=True),
        seed=602,
        actions=[HedgeActions.LONG_OPEN_SMALL, HedgeActions.HOLD, HedgeActions.LONG_REDUCE_SMALL],
    )


def test_round82_environment_snapshot_restore_replays_next_step():
    env = env_factory()
    env.reset(seed=602)
    snapshot = HedgeEnvSnapshot.capture(env)
    first = env.step(HedgeActions.LONG_OPEN_SMALL)
    snapshot.restore(env)
    second = env.step(HedgeActions.LONG_OPEN_SMALL)
    assert np.array_equal(first[0], second[0])
    assert first[1] == second[1]
    assert first[4]["equity"] == second[4]["equity"]


def test_round83_no_lookahead_audit_confirms_next_bar_execution():
    audit = audit_next_bar_execution(env_factory(), action=HedgeActions.HOLD)
    assert audit.valid
    assert audit.execution_tick == audit.decision_tick + 1


def test_round84_terminal_reason_precedence_is_explicit():
    account = HedgeAccountState(
        cash_balance=0,
        equity=-1,
        peak_equity=100,
        long=HedgeLegState(HedgeLegSide.LONG),
        short=HedgeLegState(HedgeLegSide.SHORT),
    )
    assert (
        terminal_reason(account, mark=100, maintenance_rate=0.05, drawdown_stop=0.35)
        == "NONPOSITIVE_EQUITY"
    )
    healthy = HedgeAccountState.initial(1000)
    assert terminal_reason(healthy, mark=100, maintenance_rate=0.05, drawdown_stop=0.35) is None


def test_round85_truncation_reason_distinguishes_dataset_and_step_limit():
    assert (
        truncation_reason(current_tick=9, end_tick=9, episode_steps=3, max_episode_steps=10)
        == "DATASET_END"
    )
    assert (
        truncation_reason(current_tick=5, end_tick=9, episode_steps=10, max_episode_steps=10)
        == "MAX_EPISODE_STEPS"
    )
    assert (
        truncation_reason(current_tick=5, end_tick=9, episode_steps=2, max_episode_steps=10)
        is None
    )


def test_round86_invalid_action_fallback_is_fail_closed_to_hold():
    mask = np.zeros(len(DEFAULT_ACTION_CATALOG), dtype=bool)
    mask[HedgeActions.HOLD] = True
    assert invalid_action_fallback(HedgeActions.LONG_OPEN_SMALL, mask) is HedgeActions.HOLD
    mask[HedgeActions.LONG_OPEN_SMALL] = True
    assert (
        invalid_action_fallback(HedgeActions.LONG_OPEN_SMALL, mask)
        is HedgeActions.LONG_OPEN_SMALL
    )


def test_round87_action_mask_invariants_require_hold_and_nonempty():
    mask = np.ones(len(DEFAULT_ACTION_CATALOG), dtype=bool)
    assert_action_mask_invariants(mask)
    mask[HedgeActions.HOLD] = False
    with pytest.raises(AssertionError):
        assert_action_mask_invariants(mask)


def test_round88_vector_observation_adapter_round_trips_batch():
    adapter = VectorObservationAdapter(3)
    batch = adapter.stack([[1, 2, 3], np.array([4, 5, 6])])
    assert batch.shape == (2, 3) and batch.dtype == np.float32
    rows = adapter.unstack(batch)
    assert np.array_equal(rows[0], np.array([1, 2, 3], dtype=np.float32))


def test_round89_episode_metrics_reports_drawdown_and_invalid_count():
    metrics = EpisodeMetrics.start(1000)
    metrics.update(equity=1100, reward=1, invalid_action=False, traded_notional=100)
    metrics.update(equity=990, reward=-2, invalid_action=True, traded_notional=50)
    summary = metrics.summary()
    assert summary["steps"] == 2
    assert summary["invalid_actions"] == 1
    assert summary["max_drawdown"] == pytest.approx(0.1)
    assert summary["turnover"] == 150


def test_round90_scenario_injector_is_deterministic_and_marks_faults():
    prices = synthetic_prices(20)
    injector = ScenarioInjector()
    first = injector.inject(
        prices,
        progress=0.7,
        seed=602,
        gap_at=5,
        price_shock_at=10,
        price_shock_fraction=-0.1,
    )
    second = injector.inject(
        prices,
        progress=0.7,
        seed=602,
        gap_at=5,
        price_shock_at=10,
        price_shock_fraction=-0.1,
    )
    assert first.equals(second)
    assert len(first) == len(prices) - 1
    assert first.attrs["injected_gap"] == 5
    assert first.attrs["injected_price_shock"] == -0.1
