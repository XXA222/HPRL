from decimal import Decimal

import pytest

from freqtrade.hedge.errors import HedgeDataError
from freqtrade.hedge.numeric import require_nonnegative, require_unit_interval, to_decimal
from freqtrade.hedge.symbols import canonicalize_symbol, symbols_equivalent


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "", True])
def test_nonfinite_or_invalid_decimal_is_rejected(value) -> None:
    with pytest.raises(HedgeDataError):
        to_decimal(value, field="value")


def test_negative_and_ratio_validation() -> None:
    with pytest.raises(HedgeDataError):
        require_nonnegative("-1", field="value")
    with pytest.raises(HedgeDataError):
        require_unit_interval("1.01", field="ratio")
    assert require_unit_interval("0.55", field="ratio") == Decimal("0.55")


def test_raw_and_unified_symbol_are_canonicalized() -> None:
    assert (
        canonicalize_symbol(
            "ETHUSDT",
            managed_pair="ETH/USDT:USDT",
        )
        == "ETH/USDT:USDT"
    )
    assert symbols_equivalent("ethusdt", "ETH/USDT:USDT")
