"""Gymnasium-compatible causal environment for simultaneous LONG/SHORT Hedge legs."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .actions import DEFAULT_ACTION_CATALOG, HedgeActions
from .config import HedgeRLConfig
from .constraints import HedgeActionMasker
from .costs import ExecutionCostModel
from .gym_compat import gym, spaces
from .observation import HedgeObservationBuilder, ObservationSchema
from .portfolio import HedgePortfolioSimulator
from .rewards import HedgeRewardModel


class HedgeTradingEnv(gym.Env):
    """A discrete 21-action dual-leg environment suitable for PPO/MaskablePPO.

    Observations include features through the current closed candle.  An action is
    executed at the following candle's open and marked at that candle's close.
    This next-bar rule is deliberate and prevents same-candle lookahead.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        df,
        prices,
        reward_kwargs: dict[str, Any] | None = None,
        window_size: int | None = None,
        starting_point: bool = True,
        id: str = "hedge-rl-1",  # noqa: A002
        seed: int = 1,
        config: dict[str, Any] | None = None,
        live: bool = False,
        fee: float | None = None,
        can_short: bool = True,
        pair: str = "",
        df_raw=None,
    ) -> None:
        del reward_kwargs, starting_point, live, can_short, df_raw
        self.id = id
        self.pair = pair
        self.config_dict = config or {"freqai": {"hedge_rl_config": {}}}
        cfg = HedgeRLConfig.from_config(self.config_dict)
        if window_size is not None:
            cfg = replace(cfg, observation_window=int(window_size))
        if fee is not None:
            cfg = replace(cfg, fee_rate=float(fee))
        if seed != 1 or "seed" not in self.config_dict.get("freqai", {}).get("hedge_rl_config", {}):
            cfg = replace(cfg, seed=int(seed))
        self.rl_config = cfg
        self.seed_value = cfg.seed
        self._rng = np.random.default_rng(self.seed_value)

        self.feature_names, self.features, self.prices = self._prepare_market_data(
            df,
            prices,
            observation_window=cfg.observation_window,
        )

        self.schema = ObservationSchema(self.feature_names, cfg.observation_window)
        self.observation_builder = HedgeObservationBuilder(
            self.schema,
            feature_clip=cfg.feature_clip,
            normalize_market=False,
        )
        self.catalog = DEFAULT_ACTION_CATALOG
        self.action_space = spaces.Discrete(len(self.catalog))
        self.observation_space = spaces.Box(
            low=-cfg.feature_clip,
            high=cfg.feature_clip,
            shape=(self.schema.flat_size,),
            dtype=np.float32,
        )
        self.cost_model = ExecutionCostModel(cfg.fee_rate, cfg.slippage_bps)
        self.simulator = HedgePortfolioSimulator(cfg.starting_balance, self.cost_model)
        self.masker = HedgeActionMasker(cfg, self.catalog)
        self.reward_model = HedgeRewardModel(cfg)
        self._start_tick = cfg.observation_window - 1
        self._end_tick = len(self.prices) - 1
        self._current_tick = self._start_tick
        self._episode_steps = 0
        self._terminated = False
        self._truncated = False

    @staticmethod
    def _prepare_market_data(
        df,
        prices,
        *,
        observation_window: int,
    ) -> tuple[tuple[str, ...], np.ndarray, pd.DataFrame]:
        frame = pd.DataFrame(df).copy()
        price_frame = pd.DataFrame(prices).copy()
        if len(frame) != len(price_frame):
            raise ValueError("feature and price rows must have identical length")
        if len(frame) <= observation_window:
            raise ValueError("dataset must contain more rows than observation_window")

        feature_names = tuple(str(column) for column in frame.columns)
        features = frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if not np.isfinite(features).all():
            raise ValueError("environment features must be finite")

        required = {"open", "high", "low", "close"}
        missing = required.difference(price_frame.columns)
        if missing:
            raise ValueError(f"prices are missing required columns: {sorted(missing)}")
        for column in required | {"volume", "funding_rate"}:
            if column not in price_frame:
                price_frame[column] = 0.0
        price_frame = price_frame.reset_index(drop=True)
        HedgeTradingEnv._validate_price_frame(price_frame)
        return feature_names, features, price_frame

    @staticmethod
    def _validate_price_frame(price_frame: pd.DataFrame) -> None:
        values = price_frame[["open", "high", "low", "close"]].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError("OHLC prices must be finite and positive")

        volume = pd.to_numeric(price_frame["volume"], errors="coerce").to_numpy(dtype=float)
        funding = pd.to_numeric(price_frame["funding_rate"], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(volume).all() or (volume < 0).any():
            raise ValueError("volume must be finite and non-negative")
        if not np.isfinite(funding).all():
            raise ValueError("funding_rate must be finite")
        if (price_frame["high"] < price_frame[["open", "low", "close"]].max(axis=1)).any():
            raise ValueError("invalid high price")
        if (price_frame["low"] > price_frame[["open", "high", "close"]].min(axis=1)).any():
            raise ValueError("invalid low price")

    def _observation(self) -> np.ndarray:
        return self.observation_builder.build(
            self.features,
            tick=self._current_tick,
            account=self.simulator.state,
            mark=float(self.prices.iloc[self._current_tick]["close"]),
            maintenance_rate=self.rl_config.maintenance_margin_ratio,
            max_episode_steps=self.rl_config.max_episode_steps,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        try:
            super().reset(seed=seed)
        except TypeError:  # fallback Env
            super().reset(seed=seed, options=options)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        max_start = max(self._start_tick, self._end_tick - self.rl_config.max_episode_steps)
        if self.rl_config.random_start and max_start > self._start_tick:
            self._current_tick = int(self._rng.integers(self._start_tick, max_start + 1))
        else:
            self._current_tick = self._start_tick
        self._episode_steps = 0
        self._terminated = False
        self._truncated = False
        self.simulator = HedgePortfolioSimulator(self.rl_config.starting_balance, self.cost_model)
        info = {
            "tick": self._current_tick,
            "equity": self.simulator.state.equity,
            "action_mask": self.action_masks(),
            "observation_schema": self.schema.signature,
        }
        return self._observation(), info

    def action_masks(self) -> list[bool]:
        mark = float(self.prices.iloc[self._current_tick]["close"])
        return self.masker.mask(account=self.simulator.state, mark=mark).tolist()

    def get_actions(self):
        return HedgeActions

    def step(self, action: int):
        if self._terminated or self._truncated:
            raise RuntimeError("step() called after episode completion; call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"action {action!r} is outside the action space")
        current_mark = float(self.prices.iloc[self._current_tick]["close"])
        safety = self.masker.evaluate(int(action), account=self.simulator.state, mark=current_mark)
        invalid_action = not safety.allowed
        executed_action = HedgeActions.HOLD if invalid_action else HedgeActions(int(action))
        next_tick = self._current_tick + 1
        row = self.prices.iloc[next_tick]
        funding_rate = float(row.get("funding_rate", self.rl_config.funding_rate_per_step))
        if funding_rate == 0.0:
            funding_rate = self.rl_config.funding_rate_per_step
        transition = self.simulator.apply_action(
            self.catalog.decode(executed_action),
            reference_price=float(row["open"]),
            mark_price=float(row["close"]),
            funding_rate=funding_rate,
        )
        self._current_tick = next_tick
        self._episode_steps += 1
        account = self.simulator.state
        breakdown = self.reward_model.calculate(
            transition=transition,
            account=account,
            mark=float(row["close"]),
            invalid_action=invalid_action,
        )
        margin_ratio = account.maintenance_margin_ratio(
            float(row["close"]), self.rl_config.maintenance_margin_ratio
        )
        self._terminated = bool(
            account.equity <= 0
            or account.drawdown() >= self.rl_config.drawdown_stop
            or margin_ratio >= 1.0
        )
        self._truncated = bool(
            self._current_tick >= self._end_tick
            or self._episode_steps >= self.rl_config.max_episode_steps
        )
        info = {
            "tick": self._current_tick,
            "requested_action": int(action),
            "executed_action": int(executed_action),
            "invalid_action": invalid_action,
            "invalid_reasons": safety.reasons,
            "equity": account.equity,
            "cash_balance": account.cash_balance,
            "long_quantity": account.long.quantity,
            "short_quantity": account.short.quantity,
            "gross_exposure": account.gross_exposure(float(row["close"])),
            "net_exposure": account.net_exposure(float(row["close"])),
            "drawdown": account.drawdown(),
            "reward_components": breakdown.to_dict(),
            "action_mask": self.action_masks(),
        }
        return self._observation(), breakdown.reward, self._terminated, self._truncated, info
