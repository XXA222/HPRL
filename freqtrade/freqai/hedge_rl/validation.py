"""End-to-end offline validation for the Hedge ML/RL subsystem."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .actions import HedgeActions
from .advanced_validation import run_advanced_validation
from .config import HedgeRLConfig
from .constraints import HedgeActionMasker
from .dataset import build_causal_market_features, validate_aligned_market_data
from .environment import HedgeTradingEnv
from .freqai_bridge import HedgeFreqAIPolicyBridge, HedgePolicyContext
from .inference import HedgeInferenceGuard
from .networks import HedgeActorCriticNetwork, HedgeMultiTaskMLP, HedgeTemporalEncoder
from .registry import HedgeModelRegistry, ModelManifest
from .state import HedgeAccountState, HedgeLegSide, HedgeLegState
from .supervised import build_hedge_multitask_targets


@dataclass(frozen=True, slots=True)
class ValidationReport:
    passed: bool
    checks: dict[str, bool]
    details: dict[str, int | float | str]


def _synthetic_market(rows: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(20260806)
    returns = rng.normal(0.00005, 0.002, rows)
    close = 100.0 * np.exp(np.cumsum(returns))
    open_values = np.r_[close[0], close[:-1]]
    spread = np.maximum(close * 0.001, 0.05)
    index = pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_values,
            "high": np.maximum(open_values, close) + spread,
            "low": np.minimum(open_values, close) - spread,
            "close": close,
            "volume": rng.lognormal(4, 0.25, rows),
            "funding_rate": np.where(np.arange(rows) % 32 == 0, 0.0001, 0.0),
        },
        index=index,
    )


def run_smoke_validation() -> ValidationReport:
    checks: dict[str, bool] = {}
    details: dict[str, int | float | str] = {}
    prices = _synthetic_market()
    features = build_causal_market_features(prices, volatility_window=12)
    validate_aligned_market_data(features, prices)
    checks["causal_dataset"] = True

    config = {
        "freqai": {
            "hedge_rl_config": {
                "observation_window": 16,
                "max_episode_steps": 64,
                "random_start": False,
                "fee_rate": 0.0004,
                "slippage_bps": 1.0,
            }
        }
    }
    env = HedgeTradingEnv(df=features, prices=prices, config=config)
    observation, info = env.reset(seed=11)
    checks["environment_reset"] = bool(
        observation.shape == env.observation_space.shape and np.isfinite(observation).all()
    )
    details["observation_size"] = int(observation.size)
    details["action_count"] = int(env.action_space.n)
    details["schema_signature"] = str(info["observation_schema"])

    prerequisites = {
        HedgeActions.LONG_ADD_SMALL: [HedgeActions.LONG_OPEN_SMALL],
        HedgeActions.LONG_ADD_MEDIUM: [HedgeActions.LONG_OPEN_SMALL],
        HedgeActions.LONG_REDUCE_SMALL: [HedgeActions.LONG_OPEN_MEDIUM],
        HedgeActions.LONG_REDUCE_MEDIUM: [HedgeActions.LONG_OPEN_MEDIUM],
        HedgeActions.LONG_CLOSE: [HedgeActions.LONG_OPEN_SMALL],
        HedgeActions.SHORT_ADD_SMALL: [HedgeActions.SHORT_OPEN_SMALL],
        HedgeActions.SHORT_ADD_MEDIUM: [HedgeActions.SHORT_OPEN_SMALL],
        HedgeActions.SHORT_REDUCE_SMALL: [HedgeActions.SHORT_OPEN_MEDIUM],
        HedgeActions.SHORT_REDUCE_MEDIUM: [HedgeActions.SHORT_OPEN_MEDIUM],
        HedgeActions.SHORT_CLOSE: [HedgeActions.SHORT_OPEN_SMALL],
        HedgeActions.BOTH_REDUCE_SMALL: [HedgeActions.BOTH_OPEN_SMALL],
        HedgeActions.REBALANCE_TO_LONG: [HedgeActions.BOTH_OPEN_SMALL],
        HedgeActions.REBALANCE_TO_SHORT: [HedgeActions.BOTH_OPEN_SMALL],
        HedgeActions.CLOSE_BOTH: [HedgeActions.BOTH_OPEN_SMALL],
        HedgeActions.EMERGENCY_REDUCE_BOTH: [HedgeActions.LONG_OPEN_SMALL],
    }
    finite_steps = True
    all_actions_executed = True
    for action in HedgeActions:
        env.reset(seed=int(action))
        for setup_action in prerequisites.get(action, []):
            env.step(setup_action)
        observation, reward, terminated, truncated, step_info = env.step(action)
        finite_steps = finite_steps and bool(
            np.isfinite(observation).all()
            and np.isfinite(reward)
            and np.isfinite(step_info["equity"])
            and isinstance(terminated, bool)
            and isinstance(truncated, bool)
        )
        all_actions_executed = all_actions_executed and (
            step_info["executed_action"] == int(action)
        )
    checks["all_actions_step"] = finite_steps
    checks["all_actions_executed"] = all_actions_executed

    trajectory = [
        HedgeActions.LONG_OPEN_SMALL,
        HedgeActions.SHORT_OPEN_SMALL,
        HedgeActions.LONG_ADD_SMALL,
        HedgeActions.SHORT_ADD_SMALL,
        HedgeActions.BOTH_REDUCE_SMALL,
        HedgeActions.CLOSE_BOTH,
    ]
    env_a = HedgeTradingEnv(df=features, prices=prices, config=config)
    env_b = HedgeTradingEnv(df=features, prices=prices, config=config)
    env_a.reset(seed=77)
    env_b.reset(seed=77)
    rewards_a: list[float] = []
    rewards_b: list[float] = []
    for action in trajectory:
        rewards_a.append(float(env_a.step(action)[1]))
        rewards_b.append(float(env_b.step(action)[1]))
    checks["deterministic_replay"] = bool(np.allclose(rewards_a, rewards_b))

    cfg = HedgeRLConfig.from_config(config)
    bridge = HedgeFreqAIPolicyBridge(
        feature_names=tuple(features.columns),
        window_size=cfg.observation_window,
        config=cfg,
    )
    neutral_context = HedgePolicyContext.neutral(cfg.starting_balance, mark=100.0)
    bridge_observation = bridge.observation(
        features.to_numpy(),
        tick=cfg.observation_window - 1,
        context=neutral_context,
    )
    checks["freqai_bridge_contract"] = bool(
        bridge_observation.shape == env.observation_space.shape
        and len(bridge.action_mask(neutral_context)) == env.action_space.n
    )

    one_leg = HedgeAccountState(
        cash_balance=1000.0,
        equity=1000.0,
        peak_equity=1000.0,
        long=HedgeLegState(HedgeLegSide.LONG, 1.0, 100.0),
        short=HedgeLegState(HedgeLegSide.SHORT),
    )
    checks["single_leg_emergency_reduce"] = HedgeActionMasker(cfg).evaluate(
        HedgeActions.EMERGENCY_REDUCE_BOTH, account=one_leg, mark=100.0
    ).allowed

    unsafe_logits = np.zeros(env.action_space.n)
    unsafe_logits[HedgeActions.LONG_OPEN_MEDIUM] = np.inf
    guarded = HedgeInferenceGuard(cfg).decide(
        unsafe_logits,
        action_mask=np.ones(env.action_space.n, dtype=bool),
        feature_age_steps=0,
    )
    checks["nonfinite_inference_fails_closed"] = (
        guarded.executed_action is HedgeActions.HOLD
        and "NONFINITE_LOGITS" in guarded.reasons
    )

    encoder = HedgeTemporalEncoder(
        market_width=features.shape[1],
        window_size=16,
        hidden_dim=32,
    )
    actor_critic = HedgeActorCriticNetwork(encoder, action_count=env.action_space.n)
    batch = torch.as_tensor(np.stack([observation, observation]), dtype=torch.float32)
    logits, values = actor_critic(batch)
    loss = logits.square().mean() + values.square().mean()
    loss.backward()
    checks["actor_critic_forward_backward"] = bool(
        logits.shape == (2, env.action_space.n)
        and values.shape == (2,)
        and torch.isfinite(loss)
    )

    targets = build_hedge_multitask_targets(prices, horizon=8, volatility_window=12)
    checks["multitask_targets"] = bool(
        targets.shape == (len(prices), 5)
        and targets.iloc[:-8].notna().all().all()
        and targets.iloc[-8:].isna().all().all()
    )

    with tempfile.TemporaryDirectory() as directory:
        model = HedgeMultiTaskMLP(features.shape[1], output_dim=5, hidden_dim=16, n_layer=1)
        registry = HedgeModelRegistry(directory)
        registry.save(
            "smoke-model",
            model,
            ModelManifest(
                model_version="smoke",
                model_kind="multitask",
                observation_schema_signature=env.schema.signature,
                source_version="clean-mainline",
            ),
        )
        loaded = registry.load_state(
            "smoke-model",
            expected_observation_schema=env.schema.signature,
        )
        checks["model_registry"] = "model_state_dict" in loaded

    advanced = run_advanced_validation()
    for name, passed_check in advanced.checks.items():
        checks[f"advanced_{name}"] = passed_check
    details["advanced_round_count"] = int(advanced.details["advanced_round_count"])
    details["advanced_checks_total"] = int(advanced.details["checks_total"])

    passed = all(checks.values())
    details["checks_passed"] = sum(checks.values())
    details["checks_total"] = len(checks)
    return ValidationReport(passed=passed, checks=checks, details=details)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = run_smoke_validation()
    payload = json.dumps(asdict(report), indent=2, sort_keys=True)
    print(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload + "\n", encoding="utf-8")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
