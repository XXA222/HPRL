"""Reference strategy for local Hedge backtesting and durable Paper simulation.

The strategy produces continuous LONG/SHORT scores.  The Hedge engine converts
those scores into target positions and activates orders on the next candle, so
indicators from the current close never fill against the same candle path.
"""

from __future__ import annotations

import numpy as np
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class HedgeDualEmaExample(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "5m"
    process_only_new_candles = True
    startup_candle_count = 100

    minimal_roi = {"0": 10.0}
    stoploss = -0.99
    use_exit_signal = True

    def version(self) -> str:
        return "hedge-dual-ema-r2.1"

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        del metadata
        fast = dataframe["close"].ewm(span=12, adjust=False, min_periods=12).mean()
        slow = dataframe["close"].ewm(span=48, adjust=False, min_periods=48).mean()
        volatility = (
            dataframe["close"].pct_change().rolling(48, min_periods=24).std().clip(lower=0.0005)
        )
        normalized = ((fast - slow) / dataframe["close"] / volatility).clip(-3.0, 3.0)
        long_score = (0.5 + normalized / 6.0).clip(0.0, 1.0)
        short_score = (0.5 - normalized / 6.0).clip(0.0, 1.0)

        valid = fast.notna() & slow.notna() & volatility.notna()
        dataframe["hedge_long_score"] = np.where(valid, long_score, 0.0)
        dataframe["hedge_short_score"] = np.where(valid, short_score, 0.0)
        dataframe["hedge_target_net_ratio"] = np.where(valid, (normalized / 3.0).clip(-0.35, 0.35), 0.0)
        dataframe["hedge_model_version"] = self.version()
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        del metadata
        dataframe.loc[dataframe["hedge_long_score"] >= 0.60, "enter_long"] = 1
        dataframe.loc[dataframe["hedge_short_score"] >= 0.60, "enter_short"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        del metadata
        dataframe.loc[dataframe["hedge_long_score"] < 0.50, "exit_long"] = 1
        dataframe.loc[dataframe["hedge_short_score"] < 0.50, "exit_short"] = 1
        return dataframe
