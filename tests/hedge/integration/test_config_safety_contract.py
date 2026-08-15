from decimal import Decimal

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.hedge.config import validate_hedge_config


def base():
    return {
        "dry_run": True,
        "trading_mode": "futures",
        "margin_mode": "cross",
        "position_mode": "hedge",
        "hedge_mode_enabled": True,
        "managed_pair": "ETH/USDT:USDT",
        "exchange": {
            "name": "binance",
            "pair_whitelist": ["ETH/USDT:USDT"],
        },
        "hedge": {
            "exchange_adapter": "binance",
            "account_id": "main",
            "read_only": True,
            "live_trading_enabled": False,
            "max_gross_notional": "10000",
            "max_gross_exposure_ratio": "0.8",
        },
    }


def test_valid_p2_config() -> None:
    result = validate_hedge_config(base())
    assert result.account_id == "main"
    assert result.max_gross_notional == Decimal("10000")
    assert result.max_gross_exposure_ratio == Decimal("0.8")


def test_hedge_account_cannot_enter_legacy_engine() -> None:
    config = base()
    config["hedge_mode_enabled"] = False
    with pytest.raises(OperationalException, match="legacy engine"):
        validate_hedge_config(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("read_only", False),
        ("live_trading_enabled", True),
    ],
)
def test_p2_write_path_is_locked(field, value) -> None:
    config = base()
    config["hedge"][field] = value
    with pytest.raises(OperationalException, match="execution is locked"):
        validate_hedge_config(config)
