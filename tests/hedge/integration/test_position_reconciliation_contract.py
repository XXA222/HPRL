from decimal import Decimal

import pytest

from freqtrade.enums.hedge import PositionSide
from freqtrade.hedge.errors import HedgeDataError
from freqtrade.hedge.position_book import PositionRecord, SideAwarePositionBook
from freqtrade.hedge.reconciliation import reconcile_positions


def record(side, amount, version=0):
    return PositionRecord(
        exchange="binance",
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=side,
        amount=amount,
        version=version,
    )


def test_short_only_ratio_is_explicitly_undefined() -> None:
    book = SideAwarePositionBook([record(PositionSide.SHORT, "2")])
    assert (
        book.hedge_ratio(
            "ETH/USDT:USDT",
            account_id="main",
            exchange="binance",
        )
        is None
    )


def test_long_and_short_are_independent() -> None:
    book = SideAwarePositionBook(
        [
            record(PositionSide.LONG, "3"),
            record(PositionSide.SHORT, "2"),
        ]
    )
    assert book.gross_amount(
        "ETH/USDT:USDT",
        account_id="main",
        exchange="binance",
    ) == Decimal("5")
    assert book.net_amount(
        "ETH/USDT:USDT",
        account_id="main",
        exchange="binance",
    ) == Decimal("1")


@pytest.mark.parametrize("tolerance", ["-1", "NaN", "Infinity"])
def test_reconciliation_rejects_invalid_tolerance(tolerance) -> None:
    with pytest.raises(HedgeDataError):
        reconcile_positions([], [], amount_tolerance=tolerance)


def test_reconciliation_detects_side_specific_drift() -> None:
    result = reconcile_positions(
        [record(PositionSide.LONG, "3")],
        [record(PositionSide.LONG, "2.5")],
        amount_tolerance="0.1",
    )
    assert not result.consistent
    assert result.issues[0].key.position_side is PositionSide.LONG
