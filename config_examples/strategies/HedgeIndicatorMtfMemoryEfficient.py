from __future__ import annotations

import os

import numpy as np
from pandas import DataFrame, Series

from freqtrade.strategy import IStrategy, merge_informative_pair


class HedgeIndicatorMtfMemoryEfficient(IStrategy):
    """Compact multi-timeframe indicator strategy for dual-leg Hedge testing.

    Exact exchange-native informative candles are consumed directly.  The strategy
    never resamples lower timeframes.  Higher-timeframe trend provides the core-leg
    bias while short-term RSI extremes can activate the opposite tactical leg, which
    naturally exercises simultaneous LONG/SHORT planning in volatile trends.
    """

    INTERFACE_VERSION = 3
    can_short = True
    process_only_new_candles = True

    timeframe = os.environ.get("HEDGE_TEST_BASE_TIMEFRAME", "1m").strip() or "1m"
    startup_candle_count = 80

    minimal_roi = {"0": 10.0}
    stoploss = -0.99
    use_exit_signal = True

    _SECONDS = {
        "1m": 60,
        "15m": 15 * 60,
        "4h": 4 * 60 * 60,
        "8h": 8 * 60 * 60,
        "1d": 24 * 60 * 60,
    }
    _TREND_WEIGHTS = {
        "1m": 0.05,
        "15m": 0.10,
        "4h": 0.25,
        "8h": 0.25,
        "1d": 0.35,
    }

    def version(self) -> str:
        return "hedge-indicator-mtf-memory-v1"

    @classmethod
    def _configured_informatives(cls) -> tuple[str, ...]:
        raw = os.environ.get("HEDGE_TEST_INFORMATIVE_TFS", "15m,4h,8h,1d")
        base_seconds = cls._SECONDS.get(cls.timeframe, 0)
        result: list[str] = []
        for value in raw.split(","):
            timeframe = value.strip()
            seconds = cls._SECONDS.get(timeframe)
            if (
                not timeframe
                or timeframe == cls.timeframe
                or seconds is None
                or seconds <= base_seconds
            ):
                continue
            if timeframe not in result:
                result.append(timeframe)
        return tuple(result)

    def informative_pairs(self):
        if self.dp is None:
            return []
        return [
            (pair, timeframe)
            for pair in self.dp.current_whitelist()
            for timeframe in self._configured_informatives()
        ]

    @staticmethod
    def _rsi(close: Series, period: int = 14) -> Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        ).mean()
        avg_loss = loss.ewm(
            alpha=1.0 / period,
            adjust=False,
            min_periods=period,
        ).mean()
        relative = avg_gain / avg_loss.replace(0.0, np.nan)
        return (100.0 - 100.0 / (1.0 + relative)).fillna(50.0)

    @classmethod
    def _compact_features(cls, dataframe: DataFrame) -> DataFrame:
        close = dataframe["close"].astype("float32", copy=False)
        ema_fast = close.ewm(span=20, adjust=False, min_periods=20).mean()
        ema_slow = close.ewm(span=55, adjust=False, min_periods=55).mean()
        trend = ((ema_fast / ema_slow) - 1.0).mul(25.0).clip(-1.0, 1.0)
        rsi = cls._rsi(close, 14)
        return DataFrame(
            {
                "date": dataframe["date"],
                "mtf_trend": trend.astype("float32"),
                "mtf_rsi": rsi.astype("float32"),
            },
            index=dataframe.index,
        )

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        base = self._compact_features(dataframe)
        base_trend = base["mtf_trend"]
        base_rsi = base["mtf_rsi"]
        active = self._configured_informatives()

        if self.dp is not None:
            for timeframe in active:
                informative = self.dp.get_pair_dataframe(
                    pair=metadata["pair"],
                    timeframe=timeframe,
                )
                if informative is None or informative.empty:
                    continue
                compact = self._compact_features(informative)
                dataframe = merge_informative_pair(
                    dataframe,
                    compact,
                    self.timeframe,
                    timeframe,
                    ffill=True,
                )

        weighted = base_trend.fillna(0.0) * self._TREND_WEIGHTS.get(
            self.timeframe,
            0.05,
        )
        weight = base_trend.notna().astype("float32") * self._TREND_WEIGHTS.get(
            self.timeframe,
            0.05,
        )

        configured_weight = self._TREND_WEIGHTS.get(self.timeframe, 0.05)
        rsi_15m = base_rsi
        temporary: list[str] = []
        for timeframe in active:
            trend_column = f"mtf_trend_{timeframe}"
            rsi_column = f"mtf_rsi_{timeframe}"
            configured_weight += self._TREND_WEIGHTS.get(timeframe, 0.10)
            if trend_column in dataframe.columns:
                series = dataframe[trend_column].astype("float32", copy=False)
                valid = series.notna().astype("float32")
                tf_weight = self._TREND_WEIGHTS.get(timeframe, 0.10)
                weighted = weighted + series.fillna(0.0) * tf_weight
                weight = weight + valid * tf_weight
                temporary.append(trend_column)
            if rsi_column in dataframe.columns:
                if timeframe == "15m":
                    rsi_15m = dataframe[rsi_column].astype("float32", copy=False)
                temporary.append(rsi_column)
            informative_date = f"date_{timeframe}"
            if informative_date in dataframe.columns:
                temporary.append(informative_date)

        direction = (weighted / weight.replace(0.0, np.nan)).fillna(0.0).clip(-1.0, 1.0)
        coverage = (weight / max(configured_weight, 1e-12)).clip(0.0, 1.0)

        long_core = direction.clip(lower=0.0)
        short_core = (-direction).clip(lower=0.0)
        oversold = np.maximum(
            ((38.0 - base_rsi) / 18.0).clip(0.0, 1.0),
            ((42.0 - rsi_15m) / 20.0).clip(0.0, 1.0),
        )
        overbought = np.maximum(
            ((base_rsi - 62.0) / 18.0).clip(0.0, 1.0),
            ((rsi_15m - 58.0) / 20.0).clip(0.0, 1.0),
        )

        # Core trend and tactical mean-reversion are additive rather than mutually
        # exclusive.  A bullish macro regime can therefore keep a LONG core while
        # an overbought 1m/15m condition opens a smaller SHORT tactical leg.
        long_score = (0.20 + 0.68 * long_core + 0.52 * oversold).clip(0.0, 1.0)
        short_score = (0.20 + 0.68 * short_core + 0.52 * overbought).clip(0.0, 1.0)
        enough_history = coverage >= 0.55

        long_score = long_score.where(enough_history, 0.0)
        short_score = short_score.where(enough_history, 0.0)
        confidence = (0.55 + 0.45 * direction.abs()).mul(coverage).clip(0.0, 1.0)
        risk_scale = (0.60 + 0.40 * coverage).clip(0.0, 1.0)

        dataframe["hedge_long_score"] = long_score.astype("float32")
        dataframe["hedge_short_score"] = short_score.astype("float32")
        dataframe["hedge_target_net_ratio"] = (direction * 0.18).astype("float32")
        dataframe["hedge_confidence"] = confidence.astype("float32")
        dataframe["hedge_risk_scale"] = risk_scale.astype("float32")
        dataframe["hedge_long_exposure_scale"] = (
            0.70 + 0.30 * np.maximum(long_core, oversold)
        ).clip(0.0, 1.0).astype("float32")
        dataframe["hedge_short_exposure_scale"] = (
            0.70 + 0.30 * np.maximum(short_core, overbought)
        ).clip(0.0, 1.0).astype("float32")
        dataframe["hedge_allow_new_risk"] = enough_history.astype(bool)

        # Remove all informative intermediates before the million-row dataframe is
        # handed to the Hedge adapter.  Constant object/string metadata columns are
        # intentionally omitted; ``version()`` supplies the model version instead.
        drop_columns = [name for name in dict.fromkeys(temporary) if name in dataframe.columns]
        if drop_columns:
            dataframe.drop(columns=drop_columns, inplace=True)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        del metadata
        dataframe["enter_long"] = (
            dataframe["hedge_allow_new_risk"]
            & (dataframe["hedge_long_score"] >= 0.58)
            & (dataframe["volume"] > 0)
        ).astype("int8")
        dataframe["enter_short"] = (
            dataframe["hedge_allow_new_risk"]
            & (dataframe["hedge_short_score"] >= 0.58)
            & (dataframe["volume"] > 0)
        ).astype("int8")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        del metadata
        dataframe["exit_long"] = (dataframe["hedge_long_score"] < 0.40).astype("int8")
        dataframe["exit_short"] = (dataframe["hedge_short_score"] < 0.40).astype("int8")
        return dataframe
