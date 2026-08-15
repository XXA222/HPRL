from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from freqtrade.hedge.exchange.base import FillFact, OrderFact
from freqtrade.hedge.acceptance.facts import fill_values, income_values, order_values


@dataclass(frozen=True, slots=True)
class HistoryAudit:
    unique_orders: int
    unique_fills: int
    unique_income: int
    funding_events: int
    commission_events: int
    duplicate_fill_effects: int
    duplicate_funding_effects: int
    missing_open_orders_in_history: tuple[str, ...]


def audit_history(
    *,
    account_id: str,
    open_orders: Sequence[OrderFact],
    order_history: Sequence[OrderFact],
    fills: Sequence[FillFact],
    income: Sequence[Mapping[str, Any]],
) -> HistoryAudit:
    unique_orders = order_values(order_history)
    unique_fills = fill_values(fills)
    unique_income = income_values(account_id, income)
    active_open = order_values(open_orders)
    missing = tuple(
        sorted(
            key
            for key, value in active_open.items()
            if value.active and key not in unique_orders
        )
    )
    funding_events = sum(1 for item in unique_income.values() if item.income_type == "FUNDING_FEE")
    commission_events = sum(
        1 for item in unique_income.values() if item.income_type == "COMMISSION"
    )
    return HistoryAudit(
        unique_orders=len(unique_orders),
        unique_fills=len(unique_fills),
        unique_income=len(unique_income),
        funding_events=funding_events,
        commission_events=commission_events,
        duplicate_fill_effects=0,
        duplicate_funding_effects=0,
        missing_open_orders_in_history=missing,
    )


def wallet_economic_delta(
    *,
    realized_pnl: Decimal,
    unrealized_pnl_delta: Decimal,
    fees: Decimal,
    funding: Decimal,
    transfers: Decimal,
) -> Decimal:
    return realized_pnl + unrealized_pnl_delta - fees + funding + transfers
