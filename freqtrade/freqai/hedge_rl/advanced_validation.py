"""Installed-runtime smoke validation for the clean-mainline ML/RL subsystem."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch

from .accounting import (
    FillRecord,
    IdempotentFillLedger,
    PositionAccumulator,
    audit_account_invariants,
)
from .actions import DEFAULT_ACTION_CATALOG, HedgeActions
from .config import HedgeRLConfig
from .contracts import (
    ConfigSchemaVersion,
    SeedLedger,
    build_mask_reason_matrix,
    canonical_action_signature,
    mirror_action,
)
from .env_extensions import (
    EpisodeMetrics,
    HedgeEnvSnapshot,
    VectorObservationAdapter,
    assert_action_mask_invariants,
    verify_seed_determinism,
)
from .environment import HedgeTradingEnv
from .execution_models import (
    ExecutionAuditTrail,
    ExecutionEventType,
    apply_liquidity_cap,
    partial_fill_schedule,
)
from .features import (
    FeatureManifest,
    FeatureSchema,
    RobustFeatureScaler,
    causal_rolling_zscore,
    population_stability_index,
)
from .market_data import (
    dataset_fingerprint,
    detect_market_gaps,
    purged_chronological_split,
    stationary_block_bootstrap_indices,
)
from .reward_extensions import (
    RewardExplainer,
    RewardNormalizer,
    conditional_value_at_risk,
    safe_log_equity_return,
)
from .state import HedgeAccountState, HedgeLegSide
from .training_extensions import (
    AuxiliaryRiskHead,
    CheckpointCompatibility,
    DistributionalValueHead,
    fail_closed_policy_decision,
    mask_action_logits,
)


@dataclass(frozen=True, slots=True)
class AdvancedValidationReport:
    passed: bool
    checks: dict[str, bool]
    details: dict[str, int | float | str]


def _market(rows: int = 64) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(602)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, rows)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.maximum(close * 0.001, 0.02)
    index = pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC")
    prices = pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": np.linspace(1000, 1500, rows),
            "funding_rate": 0.0,
        },
        index=index,
    )
    features = pd.DataFrame(
        {
            "return": np.log(prices["close"] / prices["close"].shift(1)).fillna(0.0),
            "range": (prices["high"] - prices["low"]) / prices["close"],
        },
        index=index,
    )
    return features, prices


def _env_factory() -> HedgeTradingEnv:
    features, prices = _market()
    return HedgeTradingEnv(
        df=features,
        prices=prices,
        config={
            "freqai": {
                "hedge_rl_config": {
                    "observation_window": 8,
                    "max_episode_steps": 16,
                    "random_start": False,
                    "seed": 602,
                }
            }
        },
    )


def run_advanced_validation() -> AdvancedValidationReport:
    checks: dict[str, bool] = {}
    details: dict[str, int | float | str] = {
        "advanced_round_first": 21,
        "advanced_round_last": 100,
        "advanced_round_count": 80,
    }

    ledger = SeedLedger(602)
    checks["contracts_seed_version_action"] = bool(
        ledger.child("env") == SeedLedger(602).child("env")
        and ConfigSchemaVersion.parse("1.0.0").compatible_with(ConfigSchemaVersion.parse("1.1.0"))
        and len(canonical_action_signature()) == 64
        and all(mirror_action(mirror_action(action)) is action for action in HedgeActions)
    )
    matrix = build_mask_reason_matrix(
        HedgeRLConfig(), account=HedgeAccountState.initial(1000), mark=100
    )
    checks["contracts_mask_matrix"] = (
        len(matrix) == len(DEFAULT_ACTION_CATALOG) and matrix[0].allowed
    )

    features, prices = _market()
    schema = FeatureSchema(tuple(features.columns))
    scaler = RobustFeatureScaler().fit(features.to_numpy())
    normalized = scaler.transform(features.to_numpy())
    causal = causal_rolling_zscore(features.to_numpy(), window=8)
    manifest = FeatureManifest.build(features, schema)
    checks["features_schema_scaling_manifest"] = bool(
        np.isfinite(normalized).all()
        and np.isfinite(causal).all()
        and len(manifest.fingerprint) == 64
    )
    checks["features_drift"] = population_stability_index(
        features.iloc[:32, 0], features.iloc[:32, 0]
    ) < 1e-12

    split = purged_chronological_split(len(features), embargo=2)
    bootstrap = stationary_block_bootstrap_indices(len(features), block_length=4, seed=602)
    fingerprint = dataset_fingerprint(features, prices)
    checks["market_split_bootstrap_fingerprint"] = bool(
        split.validation.start - split.train.stop == 2
        and len(bootstrap) == len(features)
        and len(fingerprint) == 64
    )
    gapped = prices.index.delete(10)
    checks["market_gap_detection"] = len(
        detect_market_gaps(gapped, expected_interval=pd.Timedelta(minutes=1))
    ) == 1

    fill_ledger = IdempotentFillLedger()
    fill = FillRecord(
        "f1", "o1", HedgeLegSide.LONG, True, 1, 100, timestamp=datetime(2026, 1, 1, tzinfo=UTC)
    )
    accumulator = PositionAccumulator(HedgeLegSide.LONG).apply_fill(
        increasing=True, quantity=1, price=100
    )
    checks["accounting_fill_vwap"] = (
        fill_ledger.record(fill)
        and not fill_ledger.record(fill)
        and accumulator.average_price == 100
    )
    checks["accounting_invariants"] = audit_account_invariants(
        HedgeAccountState.initial(1000), mark=100
    ).valid

    cap = apply_liquidity_cap(requested_quantity=100, candle_volume=200, max_participation=0.1)
    schedule = partial_fill_schedule(10, parts=4)
    trail = ExecutionAuditTrail()
    trail.append(ExecutionEventType.PREPARED, order_id="o1")
    trail.append(ExecutionEventType.SUBMITTED, order_id="o1")
    checks["execution_liquidity_partial_fill"] = (
        cap.executable_quantity == 20 and math.isclose(sum(schedule), 10)
    )
    checks["execution_audit_chain"] = trail.verify()

    explainer = RewardExplainer().aggregate({"return": 1.0}, {"risk": 0.25})
    normalizer = RewardNormalizer()
    checks["reward_risk_metrics"] = bool(
        safe_log_equity_return(100, 101) > 0
        and conditional_value_at_risk([-2, -1, 1, 2], alpha=0.5) < 0
    )
    checks["reward_normalization_explain"] = (
        math.isfinite(normalizer.normalize(1.0)) and explainer.total == 0.75
    )

    checks["environment_determinism"] = verify_seed_determinism(
        _env_factory,
        seed=602,
        actions=(HedgeActions.LONG_OPEN_SMALL, HedgeActions.HOLD),
    )
    env = _env_factory()
    observation, info = env.reset(seed=602)
    snapshot = HedgeEnvSnapshot.capture(env)
    first = env.step(HedgeActions.HOLD)
    snapshot.restore(env)
    second = env.step(HedgeActions.HOLD)
    assert_action_mask_invariants(info["action_mask"])
    adapter = VectorObservationAdapter(len(observation))
    checks["environment_snapshot_vector"] = bool(
        np.array_equal(first[0], second[0])
        and adapter.stack([observation, observation]).shape[0] == 2
    )
    metrics = EpisodeMetrics.start(1000)
    metrics.update(equity=990, reward=-1, invalid_action=True, traded_notional=100)
    checks["environment_metrics"] = metrics.summary()["invalid_actions"] == 1

    logits = torch.zeros(1, len(DEFAULT_ACTION_CATALOG))
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[:, HedgeActions.HOLD] = True
    masked = mask_action_logits(logits, mask)
    distribution = DistributionalValueHead(4, atoms=11)
    dist_logits, value = distribution(torch.zeros(2, 4))
    risk = AuxiliaryRiskHead(4)(torch.zeros(2, 4))
    checks["training_mask_distribution_risk"] = bool(
        masked.argmax(-1).item() == HedgeActions.HOLD
        and dist_logits.shape == (2, 11)
        and value.shape == (2,)
        and len(risk) == 3
    )
    contract = CheckpointCompatibility("clean-mainline", "obs", "act", "gru")
    compatible, mismatches = contract.validate(
        {
            "source_version": "clean-mainline",
            "observation_signature": "obs",
            "action_signature": "act",
            "architecture": "gru",
        }
    )
    decision = fail_closed_policy_decision(
        np.zeros(len(DEFAULT_ACTION_CATALOG)),
        action_mask=np.ones(len(DEFAULT_ACTION_CATALOG), dtype=bool),
        feature_age_steps=0,
        config=HedgeRLConfig(confidence_threshold=0),
        model_compatible=False,
        account_projection_fresh=True,
    )
    checks["training_checkpoint_fail_closed"] = (
        compatible
        and not mismatches
        and decision.executed_action is HedgeActions.HOLD
    )

    details["checks_passed"] = sum(checks.values())
    details["checks_total"] = len(checks)
    return AdvancedValidationReport(all(checks.values()), checks, details)
