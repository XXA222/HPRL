"""Supervised multitask labels and loss for Hedge neural-network models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn


TARGET_NAMES = (
    "&-hedge_long_score",
    "&-hedge_short_score",
    "&-hedge_target_net_ratio",
    "&-hedge_future_return",
    "&-hedge_future_volatility",
)


def build_hedge_multitask_targets(
    prices: pd.DataFrame,
    *,
    horizon: int = 12,
    volatility_window: int = 20,
) -> pd.DataFrame:
    """Create five future labels while preserving feature-side causality.

    The final ``horizon`` rows are NaN by design and must be removed by the
    training pipeline.  A target at row *t* uses prices no later than *t+horizon*.
    """

    if horizon < 1 or volatility_window < 2:
        raise ValueError("horizon must be positive and volatility_window at least 2")
    if "close" not in prices:
        raise ValueError("prices require a close column")
    close = pd.to_numeric(prices["close"], errors="coerce")
    if close.isna().any() or (close <= 0).any():
        raise ValueError("close prices must be finite and positive")
    future_close = close.shift(-horizon)
    future_return = future_close / close - 1.0
    log_returns = np.log(close / close.shift(1))
    trailing_volatility = log_returns.rolling(volatility_window, min_periods=2).std()
    future_volatility = trailing_volatility.shift(-horizon).clip(lower=0)
    scale = (future_volatility * np.sqrt(horizon)).replace(0, np.nan).fillna(1e-6)
    risk_adjusted = (future_return / scale).clip(-6, 6)
    long_score = 1.0 / (1.0 + np.exp(-risk_adjusted))
    short_score = 1.0 / (1.0 + np.exp(risk_adjusted))
    target_net = np.tanh(risk_adjusted / 2.0)
    result = pd.DataFrame(
        {
            TARGET_NAMES[0]: long_score,
            TARGET_NAMES[1]: short_score,
            TARGET_NAMES[2]: target_net,
            TARGET_NAMES[3]: future_return,
            TARGET_NAMES[4]: future_volatility,
        },
        index=prices.index,
    )
    result.iloc[-horizon:] = np.nan
    return result


@dataclass(frozen=True, slots=True)
class MultiTaskLossWeights:
    direction: float = 1.0
    target_net: float = 1.0
    future_return: float = 0.5
    future_volatility: float = 0.5


class HedgeMultiTaskLoss(nn.Module):
    def __init__(self, weights: MultiTaskLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or MultiTaskLossWeights()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if prediction.shape != target.shape or prediction.shape[-1] != 5:
            raise ValueError("prediction and target must share shape (..., 5)")
        long_loss = nn.functional.binary_cross_entropy_with_logits(
            prediction[..., 0], target[..., 0]
        )
        short_loss = nn.functional.binary_cross_entropy_with_logits(
            prediction[..., 1], target[..., 1]
        )
        net_loss = nn.functional.mse_loss(torch.tanh(prediction[..., 2]), target[..., 2])
        return_loss = nn.functional.smooth_l1_loss(prediction[..., 3], target[..., 3])
        volatility_loss = nn.functional.smooth_l1_loss(
            nn.functional.softplus(prediction[..., 4]), target[..., 4]
        )
        return (
            self.weights.direction * (long_loss + short_loss) / 2
            + self.weights.target_net * net_loss
            + self.weights.future_return * return_loss
            + self.weights.future_volatility * volatility_loss
        )
