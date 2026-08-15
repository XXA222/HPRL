"""Numerically stable dual-leg portfolio state used by ML and RL components."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum


class HedgeLegSide(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def direction(self) -> float:
        return 1.0 if self is HedgeLegSide.LONG else -1.0


def _finite(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class MarketBar:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    funding_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("open", "high", "low", "close", "volume", "funding_rate"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if min(self.open, self.high, self.low, self.close) <= 0:
            raise ValueError("OHLC prices must be positive")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("high must be at least open, low, and close")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("low must be at most open, high, and close")
        if self.volume < 0:
            raise ValueError("volume cannot be negative")


@dataclass(frozen=True, slots=True)
class HedgeLegState:
    side: HedgeLegSide
    quantity: float = 0.0
    average_price: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    funding_paid: float = 0.0

    def __post_init__(self) -> None:
        for name in ("quantity", "average_price", "realized_pnl", "fees_paid", "funding_paid"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.quantity < 0:
            raise ValueError("quantity cannot be negative")
        if self.quantity == 0 and self.average_price != 0:
            raise ValueError("flat leg average_price must be zero")
        if self.quantity > 0 and self.average_price <= 0:
            raise ValueError("open leg average_price must be positive")
        if self.fees_paid < 0:
            raise ValueError("fees_paid cannot be negative")

    def notional(self, mark: float) -> float:
        return self.quantity * _finite("mark", mark)

    def unrealized_pnl(self, mark: float) -> float:
        mark_value = _finite("mark", mark)
        if self.quantity == 0:
            return 0.0
        return (mark_value - self.average_price) * self.quantity * self.side.direction

    def with_position(self, quantity: float, average_price: float) -> HedgeLegState:
        quantity = _finite("quantity", quantity)
        average_price = _finite("average_price", average_price)
        if quantity <= 1e-15:
            quantity, average_price = 0.0, 0.0
        return replace(self, quantity=quantity, average_price=average_price)


@dataclass(frozen=True, slots=True)
class HedgeAccountState:
    cash_balance: float
    equity: float
    peak_equity: float
    long: HedgeLegState
    short: HedgeLegState
    step: int = 0
    turnover: float = 0.0

    def __post_init__(self) -> None:
        for name in ("cash_balance", "equity", "peak_equity", "turnover"):
            object.__setattr__(self, name, _finite(name, getattr(self, name)))
        if self.long.side is not HedgeLegSide.LONG or self.short.side is not HedgeLegSide.SHORT:
            raise ValueError("account legs must be LONG and SHORT respectively")
        if self.peak_equity <= 0:
            raise ValueError("peak_equity must be positive")
        if self.step < 0 or self.turnover < 0:
            raise ValueError("step and turnover cannot be negative")

    @classmethod
    def initial(cls, balance: float) -> HedgeAccountState:
        balance = _finite("balance", balance)
        if balance <= 0:
            raise ValueError("balance must be positive")
        return cls(
            cash_balance=balance,
            equity=balance,
            peak_equity=balance,
            long=HedgeLegState(HedgeLegSide.LONG),
            short=HedgeLegState(HedgeLegSide.SHORT),
        )

    def gross_notional(self, mark: float) -> float:
        return self.long.notional(mark) + self.short.notional(mark)

    def net_notional(self, mark: float) -> float:
        return self.long.notional(mark) - self.short.notional(mark)

    def gross_exposure(self, mark: float) -> float:
        return self.gross_notional(mark) / max(abs(self.equity), 1e-12)

    def net_exposure(self, mark: float) -> float:
        return self.net_notional(mark) / max(abs(self.equity), 1e-12)

    def drawdown(self) -> float:
        return max(0.0, 1.0 - self.equity / self.peak_equity)

    def maintenance_margin_ratio(self, mark: float, maintenance_rate: float) -> float:
        maintenance = self.gross_notional(mark) * _finite("maintenance_rate", maintenance_rate)
        return maintenance / max(self.equity, 1e-12)
