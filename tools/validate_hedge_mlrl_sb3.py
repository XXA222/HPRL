#!/usr/bin/env python3
"""Tiny real Gymnasium/SB3 compatibility smoke for the 21-action Hedge environment."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


def synthetic_market(rows: int = 96) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC")
    close = 100.0 + np.linspace(0.0, 2.0, rows) + np.sin(np.arange(rows) / 7.0)
    open_values = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_values,
            "high": np.maximum(open_values, close) + 0.2,
            "low": np.minimum(open_values, close) - 0.2,
            "close": close,
            "volume": np.full(rows, 1000.0),
            "funding_rate": np.zeros(rows),
        },
        index=index,
    )


def run(require_sb3: bool) -> dict[str, object]:
    try:
        import gymnasium
        import sb3_contrib
        import stable_baselines3
    except ImportError as exc:
        if require_sb3:
            raise RuntimeError("Gymnasium/SB3 dependencies are required for this gate") from exc
        return {
            "schema": "freqtrade-hedge-mlrl80-sb3-smoke-v1",
            "status": "SKIP_MISSING_DEPENDENCY",
            "missing": str(exc),
        }

    from freqtrade.freqai.hedge_rl.dataset import build_causal_market_features
    from freqtrade.freqai.hedge_rl.environment import HedgeTradingEnv
    from freqtrade.freqai.prediction_models.HedgePyTorchMultiTaskRegressor import (
        HedgePyTorchMultiTaskRegressor,
    )
    from freqtrade.freqai.prediction_models.HedgeReinforcementLearner import (
        HedgeReinforcementLearner,
    )

    if HedgeReinforcementLearner.MyRLEnv is not HedgeTradingEnv:
        raise AssertionError("HedgeReinforcementLearner must bind HedgeTradingEnv")
    if not HedgePyTorchMultiTaskRegressor.__name__.startswith("HedgePyTorch"):
        raise AssertionError("Hedge multitask FreqAI model import failed")

    prices = synthetic_market()
    features = build_causal_market_features(prices, volatility_window=8)
    config = {
        "freqai": {
            "hedge_rl_config": {
                "observation_window": 8,
                "max_episode_steps": 32,
                "random_start": False,
                "seed": 73,
            }
        }
    }
    env = HedgeTradingEnv(df=features, prices=prices, config=config)
    observation, info = env.reset(seed=73)
    if env.action_space.n != 21:
        raise AssertionError("Hedge action space must contain exactly 21 actions")
    if len(info["action_mask"]) != 21 or not info["action_mask"][0]:
        raise AssertionError("action mask must cover all actions and keep HOLD enabled")

    model = sb3_contrib.MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        learning_rate=1e-4,
        gamma=0.99,
        seed=73,
        verbose=0,
        device="cpu",
    )
    model.learn(total_timesteps=16)
    action, _ = model.predict(observation, deterministic=True, action_masks=env.action_masks())
    if not env.action_space.contains(int(action)):
        raise AssertionError("MaskablePPO returned an action outside the 21-action catalogue")
    return {
        "schema": "freqtrade-hedge-mlrl80-sb3-smoke-v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "action_count": 21,
        "observation_size": int(observation.size),
        "predicted_action": int(action),
        "gymnasium_version": gymnasium.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "sb3_contrib_version": sb3_contrib.__version__,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-sb3", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = run(args.require_sb3)
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] in {"PASS", "SKIP_MISSING_DEPENDENCY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
