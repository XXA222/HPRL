from decimal import Decimal

import pytest

from freqtrade.hedge.contracts import (
    HEDGE_CONTRACT_VERSION,
    HEDGE_EVENT_VERSION,
    FillEvent,
    PositionKey,
    PositionRecord,
    ReadinessState,
)
from freqtrade.hedge.errors import HedgeDataError, HedgeSafetyError
from freqtrade.hedge.local_reduce_only import ReduceOnlyDecision, calculate_safe_reduce
from freqtrade.hedge.numeric import require_nonnegative_int, to_decimal
from freqtrade.hedge.symbols import canonicalize_symbol, symbols_equivalent


def test_frozen_contract_exports_remain_available() -> None:
    key = PositionKey(
        exchange="binance",
        account_id="hedge-main",
        canonical_symbol="ETH/USDT:USDT",
        position_side="LONG",
    )
    assert key.symbol == "ETHUSDT"
    assert key.canonical_symbol == "ETH/USDT:USDT"
    record = PositionRecord(key, Decimal("1"), Decimal("2000"), 1)
    assert record.quantity == Decimal("1")
    assert ReadinessState.READY.value == "READY"
    assert HEDGE_CONTRACT_VERSION == "1.0.0"
    assert HEDGE_EVENT_VERSION == 1


def test_numeric_symbol_and_reduce_compatibility() -> None:
    assert canonicalize_symbol("ETHUSDT", managed_pair="ETH/USDT:USDT") == "ETH/USDT:USDT"
    assert symbols_equivalent("ETHUSDT", "ETH/USDT:USDT")
    assert require_nonnegative_int("2", field="count") == 2
    with pytest.raises(HedgeDataError):
        to_decimal("NaN", field="amount")
    assert issubclass(HedgeSafetyError, Exception)
    result = calculate_safe_reduce(
        requested_quantity="2", confirmed_quantity="1", pending_reduce_quantity="0.25"
    )
    assert isinstance(result, ReduceOnlyDecision)
    assert result.available_to_reduce == Decimal("0.75")
    assert result.reason_code == "CLIPPED_TO_CONFIRMED_AVAILABLE"


def test_frozen_fill_event_constructor_remains_available() -> None:
    event = FillEvent(
        event_id="fill-1",
        correlation_id="corr-1",
        account_id="hedge-main",
        event_type="FILL",
        exchange_time_ms=1000,
        observed_time_ms=1001,
        order_id="order-1",
        quantity=Decimal("0.5"),
        price=Decimal("2000"),
    )
    assert event.order_id == "order-1"
    assert event.client_order_id == "order-1"
    assert event.quantity == Decimal("0.5")
    assert event.hedge_event_version == HEDGE_EVENT_VERSION
