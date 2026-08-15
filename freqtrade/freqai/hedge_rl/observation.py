"""Causal observation construction for dual-leg Hedge agents."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps

import numpy as np
import numpy.typing as npt

from .normalization import RunningNormalizer
from .state import HedgeAccountState


ACCOUNT_FEATURE_NAMES = (
    "long_exposure",
    "short_exposure",
    "long_unrealized_return",
    "short_unrealized_return",
    "gross_exposure",
    "net_exposure",
    "cash_to_equity",
    "drawdown",
    "maintenance_margin_ratio",
    "fees_to_equity",
    "funding_to_equity",
    "episode_progress",
)


@dataclass(frozen=True, slots=True)
class ObservationSchema:
    market_feature_names: tuple[str, ...]
    window_size: int
    account_feature_names: tuple[str, ...] = ACCOUNT_FEATURE_NAMES

    def __post_init__(self) -> None:
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not self.market_feature_names:
            raise ValueError("market_feature_names cannot be empty")
        all_names = self.market_feature_names + self.account_feature_names
        if len(set(all_names)) != len(all_names):
            raise ValueError("observation feature names must be unique")

    @property
    def market_width(self) -> int:
        return len(self.market_feature_names)

    @property
    def flat_size(self) -> int:
        return self.window_size * self.market_width + len(self.account_feature_names)

    @property
    def signature(self) -> str:
        payload = dumps(
            {
                "market": self.market_feature_names,
                "window": self.window_size,
                "account": self.account_feature_names,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return sha256(payload.encode()).hexdigest()


class HedgeObservationBuilder:
    """Build a fixed-width, finite, causal observation vector.

    The builder includes market rows only through ``tick``.  Environments should
    execute the decoded action against the following bar to avoid same-bar lookahead.
    """

    def __init__(
        self,
        schema: ObservationSchema,
        *,
        feature_clip: float = 10.0,
        normalize_market: bool = False,
    ) -> None:
        self.schema = schema
        self.feature_clip = float(feature_clip)
        if self.feature_clip <= 0:
            raise ValueError("feature_clip must be positive")
        self.normalizer = (
            RunningNormalizer(schema.market_width, clip=self.feature_clip)
            if normalize_market
            else None
        )

    def fit_market_normalizer(self, market_features: npt.ArrayLike) -> None:
        if self.normalizer is None:
            raise RuntimeError("market normalization is disabled")
        self.normalizer.update(market_features)

    def build(
        self,
        market_features: npt.ArrayLike,
        *,
        tick: int,
        account: HedgeAccountState,
        mark: float,
        maintenance_rate: float,
        max_episode_steps: int,
    ) -> npt.NDArray[np.float32]:
        features = np.asarray(market_features, dtype=np.float64)
        if features.ndim != 2 or features.shape[1] != self.schema.market_width:
            raise ValueError(
                f"market features must have shape (rows, {self.schema.market_width})"
            )
        start = tick - self.schema.window_size + 1
        if start < 0 or tick >= len(features):
            raise IndexError("tick does not have a complete causal observation window")
        window = features[start : tick + 1]
        if self.normalizer is not None:
            window = self.normalizer.normalize(window)
        else:
            window = np.nan_to_num(
                window,
                nan=0.0,
                posinf=self.feature_clip,
                neginf=-self.feature_clip,
            )
            window = np.clip(window, -self.feature_clip, self.feature_clip).astype(np.float32)

        equity = max(abs(account.equity), 1e-12)
        long_notional = account.long.notional(mark)
        short_notional = account.short.notional(mark)
        long_entry_notional = max(account.long.notional(account.long.average_price or mark), 1e-12)
        short_entry_notional = max(
            account.short.notional(account.short.average_price or mark),
            1e-12,
        )
        account_vector = np.asarray(
            [
                long_notional / equity,
                short_notional / equity,
                account.long.unrealized_pnl(mark) / long_entry_notional
                if account.long.quantity
                else 0.0,
                account.short.unrealized_pnl(mark) / short_entry_notional
                if account.short.quantity
                else 0.0,
                account.gross_exposure(mark),
                account.net_exposure(mark),
                account.cash_balance / equity,
                account.drawdown(),
                account.maintenance_margin_ratio(mark, maintenance_rate),
                (account.long.fees_paid + account.short.fees_paid) / equity,
                (account.long.funding_paid + account.short.funding_paid) / equity,
                min(1.0, account.step / max(1, max_episode_steps)),
            ],
            dtype=np.float64,
        )
        account_vector = np.nan_to_num(
            account_vector,
            nan=0.0,
            posinf=self.feature_clip,
            neginf=-self.feature_clip,
        )
        account_vector = np.clip(account_vector, -self.feature_clip, self.feature_clip)
        result = np.concatenate([window.reshape(-1), account_vector]).astype(np.float32)
        if result.shape != (self.schema.flat_size,) or not np.isfinite(result).all():
            raise RuntimeError("observation construction produced an invalid vector")
        return result
