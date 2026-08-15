from decimal import Decimal

import pytest

from freqtrade.enums import TradingMode
from freqtrade.wallets import PositionWallet, Wallet, Wallets, position_wallet_key


def wallets() -> Wallets:
    result = Wallets.__new__(Wallets)
    result._config = {"trading_mode": TradingMode.FUTURES, "margin_mode": "cross"}
    result._stake_currency = "USDT"
    result._wallets = {"USDT": Wallet("USDT", free=850, used=150, total=1000)}
    result._positions = {}
    result._positions_by_side = {
        position_wallet_key("ETH/USDT:USDT", "LONG"): PositionWallet(
            "ETH/USDT:USDT", 2, 3, 100, "long", 2000
        ),
        position_wallet_key("ETH/USDT:USDT", "SHORT"): PositionWallet(
            "ETH/USDT:USDT", 1, 3, 50, "short", 2000
        ),
    }
    return result


def test_side_aware_exit_and_collateral() -> None:
    value = wallets()
    assert (
        value.get_exit_quantity(
            "ETH/USDT:USDT", "LONG", requested_quantity=2, pending_reduce_quantity=0.75
        )
        == 1.25
    )
    assert value.get_collateral() == 1000
    assert value.get_position_collateral("ETH/USDT:USDT", "SHORT") == 50


def test_gross_and_net_exposure() -> None:
    exposure = wallets().get_hedge_exposure("ETH/USDT:USDT", equity=Decimal("10000"))
    assert exposure.gross_total_notional == Decimal("6000")
    assert exposure.net_notional == Decimal("2000")
    assert exposure.gross_exposure_ratio == Decimal("0.6")


def test_nonfinite_values_fail_closed() -> None:
    value = wallets()
    with pytest.raises(ValueError, match="finite"):
        value.get_exit_quantity("ETH/USDT:USDT", "LONG", requested_quantity=float("nan"))
