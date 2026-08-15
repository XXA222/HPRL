from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest

import importlib.util
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "freqtrade"
    / "commands"
    / "hedge_readonly_commands.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "hedge_readonly_commands_under_test",
    _MODULE_PATH,
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_assert_readonly_config = _MODULE._assert_readonly_config
_run_preflight = _MODULE._run_preflight
from freqtrade.exceptions import OperationalException
from freqtrade.hedge.exchange.base import (
    AccountConfigurationFact,
    AccountSnapshotFact,
    ApiPermissionReport,
    PositionFact,
    TransportTelemetry,
)
from freqtrade.hedge.exchange.clock_sync import ClockSyncStatus


def _valid_config():
    return {
        "hedge_mode_enabled": True,
        "exchange": {
            "name": "binance",
            "key": "key",
            "secret": "secret",
            "pair_whitelist": ["BTC/USDT:USDT"],
        },
        "hedge": {
            "read_only": True,
            "live_trading_enabled": False,
            "operation_mode": "readonly",
        },
    }


def test_readonly_preflight_config_fails_closed():
    config = _valid_config()
    _assert_readonly_config(config)

    config["hedge"]["live_trading_enabled"] = True
    with pytest.raises(OperationalException, match="live_trading_enabled=false"):
        _assert_readonly_config(config)


def test_readonly_preflight_sanitizes_proxy_and_halts_on_unmanaged(monkeypatch):
    now = datetime(2026, 8, 2, tzinfo=UTC)
    managed = PositionFact(
        account_id="acct",
        symbol="BTCUSDT",
        position_side="LONG",
        quantity=Decimal("1"),
        entry_price=Decimal("100"),
        mark_price=Decimal("101"),
        unrealized_pnl=Decimal("1"),
        liquidation_price=Decimal("50"),
        leverage=3,
        margin_mode="cross",
        update_time_ms=1,
        observed_at=now,
        source="TEST",
    )
    unmanaged = PositionFact(
        account_id="acct",
        symbol="ETHUSDT",
        position_side="SHORT",
        quantity=Decimal("2"),
        entry_price=Decimal("10"),
        mark_price=Decimal("9"),
        unrealized_pnl=Decimal("2"),
        liquidation_price=Decimal("20"),
        leverage=3,
        margin_mode="cross",
        update_time_ms=1,
        observed_at=now,
        source="TEST",
    )
    bundle = SimpleNamespace(
        account_snapshot=AccountSnapshotFact(
            account_id="acct",
            total_wallet_balance=Decimal("1000"),
            total_available_balance=Decimal("900"),
            total_margin_balance=Decimal("1002"),
            total_initial_margin=Decimal("100"),
            total_maintenance_margin=Decimal("10"),
            total_unrealized_pnl=Decimal("2"),
            observed_at=now,
            collection_started_at=now,
            collection_completed_at=now,
        ),
        balances=(),
        positions=(managed, unmanaged),
        open_orders=(),
        fills=(),
        configuration=AccountConfigurationFact(
            account_id="acct",
            hedge_mode=True,
            active_margin_modes=("cross",),
            leverage_by_symbol_side={"BTCUSDT:LONG": 3, "BTCUSDT:SHORT": 3},
            observed_at=now,
        ),
        collection_started_at=now,
        collection_completed_at=now,
    )

    class Client:
        def __init__(self):
            self.clock_sync = SimpleNamespace(
                status=ClockSyncStatus(True, 1.0, 2.0, 5, 1000.0)
            )

        async def synchronize_clock(self):
            return None

        async def preflight_permissions(self, policy):
            return ApiPermissionReport(
                read_enabled=True,
                futures_enabled=True,
                withdrawals_enabled=False,
                internal_transfer_enabled=False,
                universal_transfer_enabled=False,
                spot_margin_trading_enabled=False,
                strict_readonly_verified=True,
            )

        async def fetch_bundle(self, *, include_fills, fill_start_time_ms=None):
            assert include_fills is False
            assert fill_start_time_ms is None
            return bundle

    class Transport:
        telemetry = TransportTelemetry(1, 1, 0, 0, 0, 0, 1, 1.5, 200, None, now)

        async def close(self):
            return None

    runtime = SimpleNamespace(
        client=Client(),
        transport=Transport(),
        config=SimpleNamespace(
            account_id="acct",
            managed_symbols=("BTCUSDT",),
            permission_policy=object(),
            target_leverage=3,
            rest_proxy_url="http://127.0.0.1:7897",
            websocket_proxy_url="http://127.0.0.1:7897",
            trust_env_proxy=False,
            fill_lookback=__import__("datetime").timedelta(hours=72),
        ),
        clock=SimpleNamespace(now=lambda: now),
    )

    import freqtrade.hedge.readonly as readonly

    monkeypatch.setattr(
        readonly,
        "build_binance_readonly_runtime_from_freqtrade_config",
        lambda **kwargs: runtime,
    )

    import asyncio

    result = asyncio.run(_run_preflight(_valid_config(), include_history=False))

    assert result["status"] == "HALT"
    assert result["failure_reasons"] == ["UNMANAGED_POSITIONS"]
    assert result["proxy"] == {
        "rest_enabled": True,
        "websocket_enabled": True,
        "trust_env": False,
    }
    assert "7897" not in str(result)
    assert result["account"]["total_wallet_balance"] == "1000"
