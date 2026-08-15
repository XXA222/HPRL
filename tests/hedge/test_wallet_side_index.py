
import unittest

from freqtrade.wallets import PositionWallet, Wallets, position_wallet_key


class TestWalletSideIndex(unittest.TestCase):
    def test_key_normalizes_long_and_short(self) -> None:
        self.assertEqual(
            position_wallet_key("ETH/USDT:USDT", "long"),
            ("ETH/USDT:USDT", "LONG"),
        )
        self.assertEqual(
            position_wallet_key("ETH/USDT:USDT", "short"),
            ("ETH/USDT:USDT", "SHORT"),
        )

    def test_side_getter_keeps_both_legs(self) -> None:
        wallets = object.__new__(Wallets)
        long_position = object()
        short_position = object()
        wallets._positions = {"ETH/USDT:USDT": short_position}
        wallets._positions_by_side = {
            ("ETH/USDT:USDT", "LONG"): long_position,
            ("ETH/USDT:USDT", "SHORT"): short_position,
        }

        self.assertIs(wallets.get_position("ETH/USDT:USDT", "LONG"), long_position)
        self.assertIs(wallets.get_position("ETH/USDT:USDT", "SHORT"), short_position)
        self.assertEqual(len(wallets.get_all_positions_by_side()), 2)

    def test_side_getter_projects_a_real_legacy_position_when_unambiguous(self) -> None:
        wallets = object.__new__(Wallets)
        legacy = PositionWallet(
            symbol="ETH/USDT:USDT",
            position=1.0,
            leverage=3.0,
            collateral=100.0,
            side="long",
            mark_price=2000.0,
        )
        wallets._positions = {"ETH/USDT:USDT": legacy}
        wallets._positions_by_side = {}

        positions = wallets.get_all_positions_by_side()

        self.assertEqual(positions, {("ETH/USDT:USDT", "LONG"): legacy})


if __name__ == "__main__":
    unittest.main()
