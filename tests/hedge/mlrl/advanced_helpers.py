from __future__ import annotations

import numpy as np
import pandas as pd

from freqtrade.freqai.hedge_rl.environment import HedgeTradingEnv
from freqtrade.freqai.hedge_rl.state import HedgeAccountState, HedgeLegSide, HedgeLegState


def synthetic_prices(rows: int = 96) -> pd.DataFrame:
    rng = np.random.default_rng(602)
    returns = rng.normal(0.0001, 0.0015, rows)
    close = 100 * np.exp(np.cumsum(returns))
    open_ = np.r_[close[0] * 0.999, close[:-1]]
    spread = np.maximum(close * 0.001, 0.02)
    index = pd.date_range("2026-01-01", periods=rows, freq="min", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": np.linspace(1000, 2000, rows),
            "funding_rate": np.where(np.arange(rows) % 24 == 0, 0.0001, 0.0),
        },
        index=index,
    )


def synthetic_features(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    result = pd.DataFrame(
        {
            "return_1": np.log(close / close.shift(1)).fillna(0.0),
            "range": ((prices["high"] - prices["low"]) / close).fillna(0.0),
            "volume_log": np.log1p(prices["volume"]),
        },
        index=prices.index,
    )
    return result


def env_factory(*, random_start: bool = False, max_episode_steps: int = 32):
    prices = synthetic_prices()
    features = synthetic_features(prices)
    config = {
        "freqai": {
            "hedge_rl_config": {
                "observation_window": 8,
                "max_episode_steps": max_episode_steps,
                "random_start": random_start,
                "seed": 602,
            }
        }
    }
    return HedgeTradingEnv(df=features, prices=prices, config=config)


def account_with_both_legs() -> HedgeAccountState:
    return HedgeAccountState(
        cash_balance=1000.0,
        equity=1020.0,
        peak_equity=1050.0,
        long=HedgeLegState(HedgeLegSide.LONG, quantity=1.0, average_price=90.0),
        short=HedgeLegState(HedgeLegSide.SHORT, quantity=0.5, average_price=110.0),
        step=3,
        turnover=400.0,
    )
