"""Causal dataset validation and window extraction for Hedge ML/RL."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


_FUTURE_NAME = re.compile(r"(?:future|lead|lookahead|t\+\d+|shift_?minus)", re.IGNORECASE)


def validate_aligned_market_data(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    reject_suspicious_feature_names: bool = True,
) -> None:
    if not isinstance(features, pd.DataFrame) or not isinstance(prices, pd.DataFrame):
        raise TypeError("features and prices must be pandas DataFrames")
    if len(features) != len(prices) or len(features) < 3:
        raise ValueError("features and prices must have equal length of at least 3")
    if not features.index.equals(prices.index):
        raise ValueError("feature and price indexes must align exactly")
    if not features.index.is_monotonic_increasing or not features.index.is_unique:
        raise ValueError("market index must be monotonic and unique")
    required = {"open", "high", "low", "close"}
    if missing := required.difference(prices.columns):
        raise ValueError(f"prices are missing {sorted(missing)}")
    feature_values = features.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    price_values = (
        prices[list(required)]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    if not np.isfinite(feature_values).all() or not np.isfinite(price_values).all():
        raise ValueError("features and OHLC prices must be finite")
    if (price_values <= 0).any():
        raise ValueError("OHLC prices must be positive")
    if reject_suspicious_feature_names:
        suspicious = [
            str(column)
            for column in features.columns
            if _FUTURE_NAME.search(str(column))
        ]
        if suspicious:
            raise ValueError(f"suspicious future-looking feature names: {suspicious}")


def build_causal_market_features(
    prices: pd.DataFrame,
    *,
    volatility_window: int = 20,
) -> pd.DataFrame:
    if volatility_window < 2:
        raise ValueError("volatility_window must be at least 2")
    required = {"open", "high", "low", "close"}
    if missing := required.difference(prices.columns):
        raise ValueError(f"prices are missing {sorted(missing)}")
    close = pd.to_numeric(prices["close"], errors="coerce")
    previous_close = close.shift(1)
    log_return = np.log(close / previous_close)
    true_range = (
        pd.concat(
            [
                prices["high"] - prices["low"],
                (prices["high"] - previous_close).abs(),
                (prices["low"] - previous_close).abs(),
            ],
            axis=1,
        )
        .max(axis=1)
        .div(previous_close)
    )
    close_location = (close - prices["low"]) / (prices["high"] - prices["low"]).replace(0, np.nan)
    volume = pd.to_numeric(
        prices.get("volume", pd.Series(0.0, index=prices.index)),
        errors="coerce",
    )
    volume_change = np.log1p(volume).diff()
    result = pd.DataFrame(
        {
            "log_return_1": log_return,
            "true_range": true_range,
            "close_location": close_location,
            "volume_log_change": volume_change,
            "realized_volatility": log_return.rolling(volatility_window, min_periods=2).std(),
        },
        index=prices.index,
    )
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass(frozen=True, slots=True)
class CausalWindowDataset:
    features: np.ndarray
    window_size: int

    def __post_init__(self) -> None:
        array = np.asarray(self.features, dtype=np.float32)
        if array.ndim != 2 or len(array) <= self.window_size:
            raise ValueError("features must be 2D and longer than window_size")
        if self.window_size < 2 or not np.isfinite(array).all():
            raise ValueError("window_size must be at least 2 and features finite")
        object.__setattr__(self, "features", array)

    def __len__(self) -> int:
        return len(self.features) - self.window_size

    def decision_tick(self, item: int) -> int:
        if item < 0 or item >= len(self):
            raise IndexError(item)
        return item + self.window_size - 1

    def __getitem__(self, item: int) -> tuple[np.ndarray, int]:
        tick = self.decision_tick(item)
        window = self.features[tick - self.window_size + 1 : tick + 1]
        return window.copy(), tick + 1
