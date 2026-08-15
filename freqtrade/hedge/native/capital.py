"""Translate upstream wallet/stake controls into an account-level Hedge budget."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping

from .models import (
    AdmissionCode,
    AdmissionDecision,
    CapitalSnapshot,
    NativeOrderIntent,
    ONE,
    ZERO,
    finite_decimal,
    utc_datetime,
)


def _ratio(value: object, *, field_name: str, default: Decimal) -> Decimal:
    if value is None:
        return default
    result = finite_decimal(value, field_name=field_name)
    if result < ZERO or result > ONE:
        raise ValueError(f"{field_name} must be between zero and one")
    return result


class FreqtradeCapitalPolicyAdapter:
    """Apply the stricter of official Freqtrade and Hedge capital limits."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        hedge_raw = config.get("hedge", {})
        self.hedge = hedge_raw if isinstance(hedge_raw, Mapping) else {}
        planner_raw = self.hedge.get("planner", {})
        self.planner = planner_raw if isinstance(planner_raw, Mapping) else {}

    def _official_limit(self, equity: Decimal, available: Decimal) -> Decimal:
        tradable_ratio = _ratio(
            self.config.get("tradable_balance_ratio", "0.99"),
            field_name="tradable_balance_ratio",
            default=Decimal("0.99"),
        )
        available_capital = self.config.get("available_capital")
        ratio_limit = max(equity * tradable_ratio, ZERO)
        if available_capital is None:
            return min(ratio_limit, available)
        configured = finite_decimal(available_capital, field_name="available_capital")
        if configured < ZERO:
            raise ValueError("available_capital cannot be negative")
        return min(ratio_limit, configured, available)

    def _hedge_limit(self, equity: Decimal) -> Decimal:
        ratio = self.planner.get(
            "max_gross_wallet_exposure",
            self.hedge.get("max_gross_exposure_ratio", "0.80"),
        )
        ratio_value = _ratio(
            ratio,
            field_name="max_gross_wallet_exposure",
            default=Decimal("0.80"),
        )
        notional = self.hedge.get("max_gross_notional")
        ratio_limit = equity * ratio_value
        if notional is None:
            return ratio_limit
        absolute = finite_decimal(notional, field_name="max_gross_notional")
        if absolute < ZERO:
            raise ValueError("max_gross_notional cannot be negative")
        return min(ratio_limit, absolute)

    def _single_order_limit(self, effective_limit: Decimal) -> Decimal:
        value = self.hedge.get(
            "max_single_order_notional",
            self.planner.get("max_single_order_notional"),
        )
        if value is None:
            return effective_limit
        limit = finite_decimal(value, field_name="max_single_order_notional")
        if limit < ZERO:
            raise ValueError("max_single_order_notional cannot be negative")
        return min(limit, effective_limit)

    def snapshot(
        self,
        *,
        equity: object,
        available_balance: object,
        current_gross_notional: object,
        at: datetime | None = None,
    ) -> CapitalSnapshot:
        equity_value = finite_decimal(equity, field_name="equity")
        available = finite_decimal(available_balance, field_name="available_balance")
        gross = finite_decimal(current_gross_notional, field_name="current_gross_notional")
        if min(equity_value, available, gross) < ZERO:
            raise ValueError("capital inputs cannot be negative")
        official = self._official_limit(equity_value, available)
        hedge = self._hedge_limit(equity_value)
        effective = min(official, hedge)
        remaining = max(effective - gross, ZERO)
        single = self._single_order_limit(effective)
        return CapitalSnapshot(
            equity=equity_value,
            available_balance=available,
            official_capital_limit=official,
            hedge_capital_limit=hedge,
            effective_capital_limit=effective,
            current_gross_notional=gross,
            remaining_notional=remaining,
            max_single_order_notional=single,
            observed_at=utc_datetime(at),
        )

    def admit(self, intent: NativeOrderIntent, snapshot: CapitalSnapshot) -> AdmissionDecision:
        if intent.reduce_only:
            return AdmissionDecision.allow(
                reason="REDUCE_ONLY_CAPITAL_EXEMPT",
                reduce_only_exempt=True,
            )
        if intent.notional > snapshot.max_single_order_notional:
            return AdmissionDecision.block(
                AdmissionCode.ORDER_TOO_LARGE,
                "order notional exceeds the effective single-order limit",
                metadata={
                    "notional": str(intent.notional),
                    "limit": str(snapshot.max_single_order_notional),
                },
            )
        if intent.notional > snapshot.remaining_notional:
            return AdmissionDecision.block(
                AdmissionCode.CAPITAL_EXHAUSTED,
                "order notional exceeds remaining effective Hedge capital",
                metadata={
                    "notional": str(intent.notional),
                    "remaining": str(snapshot.remaining_notional),
                },
            )
        return AdmissionDecision.allow(reason="CAPITAL_AVAILABLE")
