"""Stable side-aware identities shared without importing the risk package.

This module intentionally lives at ``freqtrade.hedge`` package level so
Readiness can import identities without executing ``freqtrade.hedge.risk`` and
creating an import cycle through ``risk.actions -> readiness.gate``.
"""

from __future__ import annotations

from dataclasses import dataclass

from freqtrade.enums.hedge import PositionSide
from freqtrade.hedge.errors import HedgeConfigurationError
from freqtrade.hedge.symbols import canonicalize_symbol


def _nonempty(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise HedgeConfigurationError(f"{field} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise HedgeConfigurationError(f"{field} must not be empty.")
    if len(normalized) > 255:
        raise HedgeConfigurationError(f"{field} must not exceed 255 characters.")
    return normalized


@dataclass(frozen=True, order=True, slots=True)
class RiskPositionKey:
    """Complete identity for one exchange/account/symbol hedge leg."""

    exchange: str
    account_id: str
    symbol: str
    position_side: PositionSide

    def __post_init__(self) -> None:
        side = (
            self.position_side
            if isinstance(self.position_side, PositionSide)
            else PositionSide(str(self.position_side).upper())
        )
        if side is PositionSide.BOTH:
            raise HedgeConfigurationError("RiskPositionKey requires LONG or SHORT side.")
        object.__setattr__(self, "exchange", _nonempty(self.exchange, field="exchange").lower())
        object.__setattr__(self, "account_id", _nonempty(self.account_id, field="account_id"))
        object.__setattr__(self, "symbol", canonicalize_symbol(self.symbol))
        object.__setattr__(self, "position_side", side)

    @property
    def stable_id(self) -> str:
        return "|".join(
            (self.exchange, self.account_id, self.symbol, self.position_side.value)
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "exchange": self.exchange,
            "account_id": self.account_id,
            "symbol": self.symbol,
            "position_side": self.position_side.value,
            "stable_id": self.stable_id,
        }


@dataclass(frozen=True, slots=True)
class UnknownOrderRisk:
    """An unresolved venue order quarantining exactly one hedge leg."""

    position_key: RiskPositionKey
    client_order_id: str
    observed_at_ms: int
    venue_order_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "client_order_id",
            _nonempty(self.client_order_id, field="client_order_id"),
        )
        if self.venue_order_id is not None:
            object.__setattr__(
                self,
                "venue_order_id",
                _nonempty(self.venue_order_id, field="venue_order_id"),
            )
        if (
            isinstance(self.observed_at_ms, bool)
            or not isinstance(self.observed_at_ms, int)
            or self.observed_at_ms < 0
        ):
            raise HedgeConfigurationError("observed_at_ms must be a nonnegative integer.")

    def as_dict(self) -> dict[str, object]:
        return {
            "position_key": self.position_key.as_dict(),
            "client_order_id": self.client_order_id,
            "venue_order_id": self.venue_order_id,
            "observed_at_ms": self.observed_at_ms,
        }
