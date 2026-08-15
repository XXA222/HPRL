from __future__ import annotations

from datetime import UTC, datetime

import pytest

from freqtrade.hedge.exchange.base import OrderOrigin
from freqtrade.hedge.exchange.binance_normalizer import normalize_order, normalize_symbol
from freqtrade.hedge.exchange.binance_readonly import BinanceReadonlyClient
from freqtrade.hedge.exchange.symbol_codec import (
    normalize_exchange_symbols,
    parse_canonical_pair,
    to_binance_symbol,
    to_canonical_pair,
)


class Transport:
    def set_timestamp_provider(self, provider):
        self.provider = provider


class Clock:
    def now(self):
        return datetime(2026, 7, 27, tzinfo=UTC)

    def monotonic(self):
        return 0.0

    async def sleep(self, seconds):
        return None


def order_payload(client_order_id: str) -> dict[str, object]:
    return {
        "symbol": "ETHUSDT",
        "positionSide": "LONG",
        "orderId": 1,
        "clientOrderId": client_order_id,
        "side": "BUY",
        "type": "LIMIT",
        "status": "NEW",
        "origQty": "1",
        "executedQty": "0",
        "avgPrice": "0",
        "reduceOnly": False,
        "updateTime": 1,
    }


def test_freqtrade_pair_and_binance_symbol_round_trip():
    parsed = parse_canonical_pair("eth/usdt:usdt")

    assert parsed.canonical == "ETH/USDT:USDT"
    assert parsed.exchange == "ETHUSDT"
    assert to_binance_symbol("ETH/USDT:USDT") == "ETHUSDT"
    assert to_canonical_pair("ETHUSDT") == "ETH/USDT:USDT"
    assert normalize_symbol("ETH/USDT:USDT") == "ETHUSDT"
    assert normalize_exchange_symbols(["ETHUSDT", "ETH/USDT:USDT"]) == (
        "ETHUSDT",
    )


def test_invalid_non_usdm_settlement_is_rejected():
    with pytest.raises(ValueError, match="settle"):
        to_binance_symbol("ETH/USDT:BTC")


def test_readonly_client_accepts_freqtrade_canonical_pair():
    client = BinanceReadonlyClient(
        transport=Transport(),
        account_id="acct",
        managed_symbols=["ETH/USDT:USDT"],
        clock=Clock(),
    )

    assert client.managed_symbols == ("ETHUSDT",)


def test_order_origin_and_quarantine_are_derived_from_client_order_id():
    system = normalize_order(
        order_payload("fthedge-long-open-1"),
        account_id="acct",
        system_client_order_prefixes=("fthedge-",),
    )
    external = normalize_order(
        order_payload("manual-order-1"),
        account_id="acct",
        system_client_order_prefixes=("fthedge-",),
    )

    assert system.origin is OrderOrigin.SYSTEM
    assert not system.quarantined
    assert external.origin is OrderOrigin.EXTERNAL
    assert external.quarantined
