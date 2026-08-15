from freqtrade.wallets import (
    PositionWallet,
    legacy_position_projection,
    position_wallet_key,
)


def test_legacy_projection_is_removed_when_both_legs_exist() -> None:
    positions = {
        position_wallet_key("ETH/USDT:USDT", "LONG"): PositionWallet(
            "ETH/USDT:USDT", 2, 3, 100, "long"
        ),
        position_wallet_key("ETH/USDT:USDT", "SHORT"): PositionWallet(
            "ETH/USDT:USDT", 1, 3, 50, "short"
        ),
    }
    assert legacy_position_projection(positions) == {}


def test_legacy_projection_remains_for_one_leg() -> None:
    wallet = PositionWallet("ETH/USDT:USDT", 2, 3, 100, "long")
    positions = {position_wallet_key("ETH/USDT:USDT", "LONG"): wallet}
    assert legacy_position_projection(positions) == {"ETH/USDT:USDT": wallet}
