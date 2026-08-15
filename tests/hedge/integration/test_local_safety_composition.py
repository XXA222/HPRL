from __future__ import annotations

import subprocess
import sys

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.hedge.config import normalize_hedge_config
from freqtrade.hedge.contracts.adapters import assert_internal_contract_compatibility
from freqtrade.hedge.safety import install_native_exchange_write_barrier


class _CcxtSurface:
    def create_order(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("CCXT create_order must never run")

    def cancel_all_orders(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("CCXT cancel_all_orders must never run")

    def withdraw(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("CCXT withdraw must never run")


class _ExchangeSurface:
    def __init__(self) -> None:
        self._api = _CcxtSurface()
        self._api_async = _CcxtSurface()

    def create_order(self, *args, **kwargs):  # pragma: no cover - barrier must replace it
        raise AssertionError("native create_order must never run")

    def cancel_order(self, *args, **kwargs):  # pragma: no cover - barrier must replace it
        raise AssertionError("native cancel_order must never run")

    def set_margin_mode(self, *args, **kwargs):  # pragma: no cover
        raise AssertionError("native set_margin_mode must never run")


def _hedge_config() -> dict[str, object]:
    return {
        "hedge_mode_enabled": True,
        "position_mode": "hedge",
        "managed_pair": "ETH/USDT:USDT",
        "trading_mode": "futures",
        "margin_mode": "cross",
        "dry_run": True,
        "exchange": {"name": "binance"},
        "hedge": {
            "operation_mode": "paper",
            "read_only": True,
            "live_trading_enabled": False,
        },
    }


def test_combined_mode_is_temporarily_rejected() -> None:
    config = _hedge_config()
    config["hedge"]["operation_mode"] = "combined"  # type: ignore[index]
    with pytest.raises(OperationalException, match="removed|shadow"):
        normalize_hedge_config(config)


def test_native_exchange_write_barrier_blocks_indirect_calls() -> None:
    exchange = _ExchangeSurface()
    blocked = install_native_exchange_write_barrier(exchange, _hedge_config())
    assert {
        "EXCHANGE_METHOD:create_order",
        "EXCHANGE_METHOD:cancel_order",
        "EXCHANGE_METHOD:set_margin_mode",
        "CCXT_API:create_order",
        "CCXT_API:cancel_all_orders",
        "CCXT_API_ASYNC:withdraw",
    }.issubset(blocked)
    with pytest.raises(OperationalException, match="HEDGE_NATIVE_EXCHANGE_WRITE_BLOCKED"):
        exchange.create_order("ETH/USDT:USDT")
    with pytest.raises(OperationalException, match="HEDGE_NATIVE_EXCHANGE_WRITE_BLOCKED"):
        exchange.cancel_order("1", "ETH/USDT:USDT")
    with pytest.raises(OperationalException, match="HEDGE_NATIVE_EXCHANGE_WRITE_BLOCKED"):
        exchange._api.cancel_all_orders("ETH/USDT:USDT")
    with pytest.raises(OperationalException, match="HEDGE_NATIVE_EXCHANGE_WRITE_BLOCKED"):
        exchange._api_async.withdraw("ETH", 1, "address")


def test_contract_boundary_values_remain_compatible() -> None:
    assert_internal_contract_compatibility()


def test_readiness_and_risk_are_cold_importable_in_either_order() -> None:
    commands = (
        "import freqtrade.hedge.readiness; import freqtrade.hedge.risk",
        "import freqtrade.hedge.risk; import freqtrade.hedge.readiness",
    )
    for command in commands:
        result = subprocess.run(
            [sys.executable, "-c", command],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
