"""Execution-cost, slippage, fee, and funding models for Hedge simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .actions import Urgency
from .state import HedgeLegSide


@dataclass(frozen=True, slots=True)
class ExecutionEstimate:
    fill_price: float
    notional: float
    fee: float
    slippage_cost: float


@dataclass(frozen=True, slots=True)
class ExecutionCostModel:
    fee_rate: float = 0.0004
    slippage_bps: float = 1.0
    passive_multiplier: float = 0.5
    urgent_multiplier: float = 2.0

    def __post_init__(self) -> None:
        for name in ("fee_rate", "slippage_bps", "passive_multiplier", "urgent_multiplier"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.fee_rate >= 0.1:
            raise ValueError("fee_rate must be below 10%")

    def _urgency_multiplier(self, urgency: Urgency) -> float:
        if urgency is Urgency.PASSIVE:
            return self.passive_multiplier
        if urgency is Urgency.URGENT:
            return self.urgent_multiplier
        return 1.0

    def estimate(
        self,
        *,
        reference_price: float,
        quantity: float,
        is_buy: bool,
        urgency: Urgency = Urgency.NORMAL,
    ) -> ExecutionEstimate:
        price = float(reference_price)
        qty = float(quantity)
        if not math.isfinite(price) or not math.isfinite(qty) or price <= 0 or qty < 0:
            raise ValueError("reference_price must be positive and quantity non-negative")
        slip_fraction = self.slippage_bps * self._urgency_multiplier(urgency) / 10_000.0
        fill_price = price * (1.0 + slip_fraction if is_buy else 1.0 - slip_fraction)
        notional = fill_price * qty
        fee = notional * self.fee_rate
        slippage_cost = abs(fill_price - price) * qty
        return ExecutionEstimate(fill_price, notional, fee, slippage_cost)

    @staticmethod
    def funding_cashflow(
        *,
        side: HedgeLegSide,
        notional: float,
        funding_rate: float,
    ) -> float:
        """Return wallet cashflow: negative means paid, positive means received."""

        value = float(notional)
        rate = float(funding_rate)
        if not math.isfinite(value) or not math.isfinite(rate) or value < 0:
            raise ValueError("notional must be non-negative and funding_rate finite")
        return -side.direction * value * rate
