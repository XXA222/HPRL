
import unittest
from datetime import UTC, datetime

from freqtrade.enums import PositionSide
from freqtrade.exceptions import OperationalException
from freqtrade.persistence.trade_model import LocalTrade


def make_trade(side: PositionSide, *, amount: float = 1.0) -> LocalTrade:
    return LocalTrade(
        pair="ETH/USDT:USDT",
        position_side=side.value,
        is_short=side is PositionSide.SHORT,
        open_date=datetime.now(UTC),
        amount=amount,
        stake_amount=100.0,
        open_rate=2000.0,
    )


class TestSideAwareTradeQuery(unittest.TestCase):
    def setUp(self) -> None:
        LocalTrade.reset_trades()

    def tearDown(self) -> None:
        LocalTrade.reset_trades()

    def test_same_pair_long_and_short_have_separate_indexes(self) -> None:
        long_trade = make_trade(PositionSide.LONG)
        short_trade = make_trade(PositionSide.SHORT)
        LocalTrade.add_bt_trade(long_trade)
        LocalTrade.add_bt_trade(short_trade)

        self.assertIs(
            LocalTrade.get_open_trade_for_pair_side(
                "ETH/USDT:USDT", PositionSide.LONG
            ),
            long_trade,
        )
        self.assertIs(
            LocalTrade.get_open_trade_for_pair_side(
                "ETH/USDT:USDT", PositionSide.SHORT
            ),
            short_trade,
        )
        self.assertEqual(len(LocalTrade.bt_trades_open_pp["ETH/USDT:USDT"]), 2)
        self.assertEqual(len(LocalTrade.bt_trades_open_pps), 2)

    def test_close_removes_only_target_side_index(self) -> None:
        long_trade = make_trade(PositionSide.LONG)
        short_trade = make_trade(PositionSide.SHORT)
        LocalTrade.add_bt_trade(long_trade)
        LocalTrade.add_bt_trade(short_trade)
        long_trade.close_profit_abs = 0.0
        LocalTrade.close_bt_trade(long_trade)

        self.assertIsNone(
            LocalTrade.get_open_trade_for_pair_side(
                "ETH/USDT:USDT", PositionSide.LONG
            )
        )
        self.assertIsNotNone(
            LocalTrade.get_open_trade_for_pair_side(
                "ETH/USDT:USDT", PositionSide.SHORT
            )
        )

    def test_duplicate_same_side_is_rejected_by_guard(self) -> None:
        LocalTrade.add_bt_trade(make_trade(PositionSide.LONG))
        LocalTrade.add_bt_trade(make_trade(PositionSide.LONG, amount=2.0))

        with self.assertRaises(OperationalException):
            LocalTrade.assert_single_open_trade_per_side("ETH/USDT:USDT")


if __name__ == "__main__":
    unittest.main()
