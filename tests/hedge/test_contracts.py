from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from freqtrade.hedge.contracts import PositionKey, PositionRecord, ReadinessState
from freqtrade.hedge.contracts.errors import HedgeContractError
from freqtrade.hedge.contracts.events import AccountEvent, FillEvent
from freqtrade.hedge.contracts.version import HEDGE_EVENT_VERSION


def test_position_key_is_canonical_and_immutable() -> None:
    key = PositionKey("BINANCE", "main", "eth/usdt", "long")
    assert (key.exchange, key.canonical_symbol, key.position_side) == (
        "binance",
        "ETH/USDT:USDT",
        "LONG",
    )
    with pytest.raises(FrozenInstanceError):
        key.account_id = "other"  # type: ignore[misc]


def test_contract_quantities_are_finite_decimals() -> None:
    key = PositionKey("binance", "main", "ETH/USDT:USDT", "SHORT")
    record = PositionRecord(key, "1.25", Decimal("2000"), 1)
    assert record.quantity == Decimal("1.25")
    with pytest.raises(HedgeContractError):
        PositionRecord(key, "NaN", "1", 1)


def test_event_identity_time_and_version_are_public() -> None:
    event = AccountEvent("evt-1", "corr-1", "main", "ACCOUNT", 10, 11)
    fill = FillEvent(
        "fill-1", "corr-1", "main", "FILL", 10, 11, order_id="o", quantity="1", price="2"
    )
    assert event.hedge_event_version == HEDGE_EVENT_VERSION
    assert (fill.quantity, fill.price, ReadinessState.READY) == (
        Decimal("1"),
        Decimal("2"),
        "READY",
    )
