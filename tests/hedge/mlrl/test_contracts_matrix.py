from __future__ import annotations

import numpy as np
import pytest

from freqtrade.freqai.hedge_rl.actions import DEFAULT_ACTION_CATALOG, HedgeActions
from freqtrade.freqai.hedge_rl.config import HedgeRLConfig
from freqtrade.freqai.hedge_rl.contracts import (
    ActionRiskTier,
    ConfigSchemaVersion,
    InvalidActionAudit,
    InvalidActionJournal,
    SeedLedger,
    apply_environment_overrides,
    build_mask_reason_matrix,
    canonical_action_signature,
    classify_action_risk,
    deserialize_action_spec,
    mirror_action,
    safe_downgrade,
    serialize_action_spec,
)
from freqtrade.freqai.hedge_rl.state import HedgeAccountState


def test_round21_seed_ledger_is_stable_and_label_scoped():
    ledger = SeedLedger(602, "test")
    assert ledger.child("environment") == ledger.child("environment")
    assert ledger.child("environment") != ledger.child("network")
    assert ledger.snapshot(("a", "b")) == SeedLedger(602, "test").snapshot(("a", "b"))
    with pytest.raises(ValueError):
        ledger.snapshot(("a", "a"))


def test_round22_config_schema_version_compatibility():
    runtime = ConfigSchemaVersion.parse("2.4.1")
    assert ConfigSchemaVersion.parse("2.3.9").compatible_with(runtime)
    assert not ConfigSchemaVersion.parse("3.0.0").compatible_with(runtime)
    assert not ConfigSchemaVersion.parse("2.5.0").compatible_with(runtime)
    with pytest.raises(ValueError):
        ConfigSchemaVersion.parse("2.4")


def test_round23_environment_overrides_only_known_fields():
    config = HedgeRLConfig()
    updated = apply_environment_overrides(
        config,
        environ={
            "FREQTRADE_HEDGE_RL_OBSERVATION_WINDOW": "64",
            "FREQTRADE_HEDGE_RL_RANDOM_START": "false",
            "FREQTRADE_HEDGE_RL_ACTION_SIZE_FRACTIONS": "[0.05, 0.2]",
            "FREQTRADE_HEDGE_RL_UNKNOWN": "ignored",
        },
    )
    assert updated.observation_window == 64
    assert updated.random_start is False
    assert updated.action_size_fractions == (0.05, 0.2)


def test_round24_action_signature_is_canonical_sha256():
    signature = canonical_action_signature()
    assert len(signature) == 64
    assert signature == canonical_action_signature(DEFAULT_ACTION_CATALOG)
    assert all(character in "0123456789abcdef" for character in signature)


def test_round25_action_risk_tiers_are_explicit():
    assert (
        classify_action_risk(DEFAULT_ACTION_CATALOG.decode(HedgeActions.HOLD))
        is ActionRiskTier.HOLD
    )
    assert (
        classify_action_risk(DEFAULT_ACTION_CATALOG.decode(HedgeActions.LONG_CLOSE))
        is ActionRiskTier.REDUCE
    )
    assert (
        classify_action_risk(DEFAULT_ACTION_CATALOG.decode(HedgeActions.LONG_OPEN_SMALL))
        is ActionRiskTier.OPEN_SMALL
    )
    assert (
        classify_action_risk(DEFAULT_ACTION_CATALOG.decode(HedgeActions.LONG_OPEN_MEDIUM))
        is ActionRiskTier.OPEN_MEDIUM
    )
    assert (
        classify_action_risk(DEFAULT_ACTION_CATALOG.decode(HedgeActions.REBALANCE_TO_LONG))
        is ActionRiskTier.REBALANCE
    )
    assert (
        classify_action_risk(DEFAULT_ACTION_CATALOG.decode(HedgeActions.CLOSE_BOTH))
        is ActionRiskTier.EMERGENCY
    )


def test_round26_action_serialization_round_trip_and_name_guard():
    spec = DEFAULT_ACTION_CATALOG.decode(HedgeActions.REBALANCE_TO_SHORT)
    payload = serialize_action_spec(spec)
    assert deserialize_action_spec(payload) == spec
    payload["name"] = "WRONG"
    with pytest.raises(ValueError):
        deserialize_action_spec(payload)


def test_round27_action_mirror_is_involutive():
    for action in HedgeActions:
        assert mirror_action(mirror_action(action)) is action
    assert mirror_action(HedgeActions.LONG_OPEN_MEDIUM) is HedgeActions.SHORT_OPEN_MEDIUM
    assert mirror_action(HedgeActions.CLOSE_BOTH) is HedgeActions.CLOSE_BOTH


def test_round28_safe_downgrade_prefers_smaller_risk():
    mask = np.zeros(len(DEFAULT_ACTION_CATALOG), dtype=bool)
    mask[HedgeActions.HOLD] = True
    mask[HedgeActions.LONG_OPEN_SMALL] = True
    assert safe_downgrade(HedgeActions.LONG_OPEN_MEDIUM, mask) is HedgeActions.LONG_OPEN_SMALL
    mask[HedgeActions.LONG_OPEN_SMALL] = False
    assert safe_downgrade(HedgeActions.LONG_OPEN_MEDIUM, mask) is HedgeActions.HOLD


def test_round29_invalid_action_journal_hashes_observation_and_bounds_capacity():
    journal = InvalidActionJournal(capacity=1)
    observation = np.arange(8, dtype=np.float32)
    signature = journal.observation_signature(observation)
    first = InvalidActionAudit(
        1,
        HedgeActions.LONG_ADD_SMALL,
        HedgeActions.HOLD,
        ("FLAT",),
        signature,
    )
    second = InvalidActionAudit(
        2,
        HedgeActions.SHORT_ADD_SMALL,
        HedgeActions.HOLD,
        ("FLAT",),
        signature,
    )
    journal.append(first)
    journal.append(second)
    assert journal.records() == (second,)


def test_round30_mask_reason_matrix_covers_all_actions_and_hold():
    rows = build_mask_reason_matrix(
        HedgeRLConfig(),
        account=HedgeAccountState.initial(1000.0),
        mark=100.0,
    )
    assert len(rows) == len(DEFAULT_ACTION_CATALOG)
    assert rows[HedgeActions.HOLD].allowed
    assert not rows[HedgeActions.LONG_ADD_SMALL].allowed
    assert "LONG_INCREASE_REQUIRES_POSITION" in rows[HedgeActions.LONG_ADD_SMALL].reasons
