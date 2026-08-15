from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from typing import Any

from freqtrade.hedge.exchange.base import BalanceFact, FillFact, OrderFact, PositionFact
from freqtrade.hedge.acceptance.models import (
    BalanceValue,
    FactPlane,
    FillValue,
    IncomeValue,
    OrderValue,
    PositionValue,
)
from freqtrade.hedge.symbols import canonicalize_symbol


def position_values(items: Sequence[PositionFact]) -> dict[str, PositionValue]:
    result: dict[str, PositionValue] = {}
    for item in items:
        value = PositionValue(
            account_id=item.account_id,
            symbol=canonicalize_symbol(item.symbol),
            position_side=item.position_side,
            quantity=item.quantity,
            entry_price=item.entry_price,
            unrealized_pnl=item.unrealized_pnl,
        )
        if value.key in result and result[value.key] != value:
            raise ValueError(f"conflicting duplicate position: {value.key}")
        result[value.key] = value
    return result


def balance_values(items: Sequence[BalanceFact]) -> dict[str, BalanceValue]:
    result: dict[str, BalanceValue] = {}
    for item in items:
        value = BalanceValue(
            account_id=item.account_id,
            asset=item.asset,
            wallet_balance=item.wallet_balance,
            available_balance=item.available_balance,
            unrealized_pnl=item.unrealized_pnl,
        )
        if value.key in result and result[value.key] != value:
            raise ValueError(f"conflicting duplicate balance: {value.key}")
        result[value.key] = value
    return result


def order_values(items: Sequence[OrderFact]) -> dict[str, OrderValue]:
    result: dict[str, OrderValue] = {}
    for item in items:
        value = OrderValue(
            account_id=item.account_id,
            symbol=canonicalize_symbol(item.symbol),
            position_side=item.position_side,
            exchange_order_id=item.exchange_order_id,
            client_order_id=item.client_order_id,
            status=item.status,
            cumulative_filled_quantity=item.cumulative_filled_quantity,
            active=item.active,
        )
        previous = result.get(value.key)
        if previous is not None and previous != value:
            raise ValueError(f"conflicting duplicate order: {value.key}")
        result[value.key] = value
    return result


def fill_values(items: Sequence[FillFact]) -> dict[str, FillValue]:
    result: dict[str, FillValue] = {}
    for item in items:
        value = FillValue(
            account_id=item.account_id,
            symbol=canonicalize_symbol(item.symbol),
            position_side=item.position_side,
            exchange_trade_id=item.exchange_trade_id,
            exchange_order_id=item.exchange_order_id,
            quantity=item.quantity,
            price=item.price,
            commission=item.commission,
            realized_pnl=item.realized_pnl,
        )
        previous = result.get(value.key)
        if previous is not None and previous != value:
            raise ValueError(f"conflicting duplicate fill: {value.key}")
        result[value.key] = value
    return result


def income_values(
    account_id: str, items: Sequence[Mapping[str, Any]]
) -> dict[str, IncomeValue]:
    result: dict[str, IncomeValue] = {}
    for item in items:
        income_type = str(item.get("incomeType") or "").strip().upper()
        if not income_type:
            raise ValueError("incomeType is required")
        tran_id = str(item.get("tranId") or "").strip()
        event_time_ms = int(item.get("time") or 0)
        identity = tran_id or ":".join(
            (
                str(item.get("tradeId") or ""),
                str(event_time_ms),
                str(item.get("symbol") or ""),
                str(item.get("asset") or ""),
                str(item.get("income") or ""),
            )
        )
        value = IncomeValue(
            account_id=account_id,
            identity=identity,
            income_type=income_type,
            asset=str(item.get("asset") or "").strip().upper(),
            symbol=str(item.get("symbol") or "").strip().upper(),
            amount=Decimal(str(item.get("income") or "0")),
            event_time_ms=event_time_ms,
        )
        previous = result.get(value.key)
        if previous is not None and previous != value:
            raise ValueError(f"conflicting duplicate income: {value.key}")
        result[value.key] = value
    return result


def build_fact_plane(
    *,
    account_id: str,
    observed_at: datetime,
    positions: Sequence[PositionFact] = (),
    balances: Sequence[BalanceFact] = (),
    orders: Sequence[OrderFact] = (),
    fills: Sequence[FillFact] = (),
    income: Sequence[Mapping[str, Any]] = (),
) -> FactPlane:
    return FactPlane(
        account_id=account_id,
        observed_at=observed_at,
        positions=position_values(positions),
        balances=balance_values(balances),
        active_orders={key: value for key, value in order_values(orders).items() if value.active},
        fills=fill_values(fills),
        income=income_values(account_id, income),
    )
