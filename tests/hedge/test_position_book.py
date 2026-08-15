
import unittest
from decimal import Decimal

from freqtrade.enums import PositionSide
from freqtrade.hedge.position_book import PositionRecord, SideAwarePositionBook


class TestSideAwarePositionBook(unittest.TestCase):
    def test_net_gross_and_hedge_ratio(self) -> None:
        book = SideAwarePositionBook(
            [
                PositionRecord("ETH/USDT:USDT", PositionSide.LONG, "5"),
                PositionRecord("ETH/USDT:USDT", PositionSide.SHORT, "2"),
            ]
        )
        self.assertEqual(book.net_amount("ETH/USDT:USDT"), Decimal("3"))
        self.assertEqual(book.gross_amount("ETH/USDT:USDT"), Decimal("7"))
        self.assertEqual(book.hedge_ratio("ETH/USDT:USDT"), Decimal("0.4"))

    def test_duplicate_key_is_rejected_during_initialization(self) -> None:
        records = [
            PositionRecord("ETH/USDT:USDT", PositionSide.LONG, "1"),
            PositionRecord("ETH/USDT:USDT", PositionSide.LONG, "2"),
        ]
        with self.assertRaises(ValueError):
            SideAwarePositionBook(records)


if __name__ == "__main__":
    unittest.main()
