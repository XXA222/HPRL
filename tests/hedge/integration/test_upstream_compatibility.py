from __future__ import annotations

from datetime import timedelta

from freqtrade.enums import PositionSide
from freqtrade.hedge.compatibility import (
    effective_trade_position_side,
    is_explicit_hedge_order,
    is_explicit_hedge_trade,
)
from freqtrade.persistence import Order, Trade
from freqtrade.util import dt_now


HEDGE_TRADE_FIELDS = {
    "account_id",
    "position_side",
    "open_slot_key",
    "hedge_version",
}
HEDGE_ORDER_FIELDS = {
    "position_side",
    "position_action",
    "action_group_id",
    "action",
    "client_order_id",
    "idempotency_key",
    "submit_state",
}


def _trade(**kwargs) -> Trade:
    values = {
        "pair": "ADA/USDT",
        "stake_amount": 10.0,
        "amount": 5.0,
        "amount_requested": 5.0,
        "fee_open": 0.0025,
        "fee_close": 0.0025,
        "open_date": dt_now() - timedelta(minutes=5),
        "open_rate": 2.0,
        "exchange": "binance",
        "is_open": True,
    }
    values.update(kwargs)
    return Trade(**values)


def _order(**kwargs) -> Order:
    values = {
        "ft_order_side": "buy",
        "order_id": "native-order",
        "ft_is_open": False,
        "ft_pair": "ADA/USDT",
        "ft_amount": 5.0,
        "ft_price": 2.0,
        "amount": 5.0,
        "filled": 5.0,
        "remaining": 0.0,
        "price": 2.0,
        "average": 2.0,
        "status": "closed",
        "order_type": "limit",
        "side": "buy",
    }
    values.update(kwargs)
    return Order(**values)


def test_plain_trade_keeps_native_serialization_schema() -> None:
    trade = _trade()
    assert not is_explicit_hedge_trade(trade)
    assert HEDGE_TRADE_FIELDS.isdisjoint(trade.to_json())


def test_explicit_hedge_trade_serializes_identity() -> None:
    trade = _trade(
        account_id="main",
        position_side=PositionSide.LONG.value,
    )
    result = trade.to_json()
    assert is_explicit_hedge_trade(trade)
    assert trade.hedge_version >= 2
    assert trade.open_slot_key == "main|ADA/USDT|LONG"
    assert result["account_id"] == "main"
    assert result["position_side"] == PositionSide.LONG.value
    assert result["hedge_version"] >= 2
    assert result["open_slot_key"] == "main|ADA/USDT|LONG"


def test_legacy_submit_state_does_not_pollute_native_order_schema() -> None:
    order = _order(submit_state="TERMINAL")
    assert not is_explicit_hedge_order(order)
    assert HEDGE_ORDER_FIELDS.isdisjoint(order.to_json("buy"))


def test_explicit_hedge_order_serializes_execution_identity() -> None:
    order = _order(
        position_side=PositionSide.SHORT.value,
        position_action="OPEN",
        action_group_id="group-1",
        action="OPEN_SHORT",
        client_order_id="hedge-client-order",
        idempotency_key="idem-1",
        submit_state="ACKNOWLEDGED",
    )
    result = order.to_json("sell")
    assert is_explicit_hedge_order(order)
    assert result["position_side"] == PositionSide.SHORT.value
    assert result["idempotency_key"] == "idem-1"
    assert result["submit_state"] == "ACKNOWLEDGED"


def test_both_side_falls_back_to_legacy_trade_direction() -> None:
    long_trade = _trade(position_side=PositionSide.BOTH.value, is_short=False)
    short_trade = _trade(position_side=PositionSide.BOTH.value, is_short=True)
    assert effective_trade_position_side(long_trade) == PositionSide.LONG.value
    assert effective_trade_position_side(short_trade) == PositionSide.SHORT.value
