from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock

from fastapi import FastAPI

from freqtrade.enums.hedge import PositionMode, PositionSide
from freqtrade.hedge.config import HedgeRuntimeConfig
from freqtrade.hedge.position_book import PositionRecord
from freqtrade.hedge.risk.models import AccountRiskSnapshot
from freqtrade.hedge.runtime import HedgeRuntime
from freqtrade.rpc.api_server.hedge_runtime import HedgeRuntimeQuery
from freqtrade.rpc.api_server.webserver import ApiServer
from tests.conftest import get_patched_freqtradebot


def runtime_config() -> HedgeRuntimeConfig:
    return HedgeRuntimeConfig(
        position_mode=PositionMode.HEDGE,
        enabled=True,
        managed_pair="ETH/USDT:USDT",
        account_id="main",
        exchange_adapter="binance",
    )


def test_runtime_defaults_fail_closed_and_queryable() -> None:
    runtime = HedgeRuntime(runtime_config())
    query = HedgeRuntimeQuery(lambda: runtime)

    readiness = query.readiness(account_id="main")
    risk = query.risk(account_id="main")

    assert readiness.ready is False
    assert readiness.kill_switch == "HALTED"
    assert risk.halted is True
    assert "RISK_DATA_INVALID" in risk.reasons


def test_runtime_publishes_atomic_ready_projection() -> None:
    runtime = HedgeRuntime(runtime_config())
    observed_at = datetime.now(UTC)
    position = PositionRecord(
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        amount=Decimal("1"),
        entry_price=Decimal("3000"),
        mark_price=Decimal("3100"),
        exchange="binance",
        account_id="main",
    )
    risk = AccountRiskSnapshot(
        account_id="main",
        equity=Decimal("10000"),
        wallet_balance=Decimal("10000"),
        available_balance=Decimal("8000"),
        initial_margin=Decimal("1000"),
        maintenance_margin=Decimal("100"),
        gross_long_notional=Decimal("3100"),
        gross_short_notional=Decimal("0"),
        net_notional=Decimal("3100"),
    )
    checks = {
        "readonly_service_bound": True,
        "rest_calibrated": True,
        "user_stream_fresh": True,
        "reconciliation_converged": True,
        "risk_snapshot_valid": True,
    }

    runtime.publish(
        positions=(position,),
        risk=risk,
        reconciliation_status="HEALTHY",
        reconciliation_at=observed_at,
        stream_state="CONNECTED",
        stream_last_event_at=observed_at,
        stream_reconnect_count=2,
        checks=checks,
    )
    query = HedgeRuntimeQuery(lambda: runtime)

    assert query.readiness(account_id="main").ready is True
    assert query.positions(account_id="main", symbol="ETHUSDT").legs[0].quantity == Decimal("1")
    assert query.user_stream(account_id="main").state == "CONNECTED"


def test_freqtradebot_hedge_process_never_enters_legacy_trading(default_conf, mocker) -> None:
    bot = get_patched_freqtradebot(mocker, default_conf)
    runtime = MagicMock()
    bot.hedge_runtime = runtime
    common_lifecycle = mocker.patch.object(bot, "_run_common_market_strategy_lifecycle")
    enter_positions = mocker.patch.object(bot, "enter_positions")
    exit_positions = mocker.patch.object(bot, "exit_positions")
    manage_open_orders = mocker.patch.object(bot, "manage_open_orders")
    mocker.patch.object(bot.rpc, "process_msg_queue")
    mocker.patch("freqtrade.freqtradebot.Trade.commit")

    bot.process()

    common_lifecycle.assert_called_once_with(hedge_mode=True)
    runtime.heartbeat.assert_called_once_with()
    runtime.halt.assert_called_once_with("HEDGE_COMPOSITION_NOT_BOUND")
    manage_open_orders.assert_not_called()
    enter_positions.assert_not_called()
    exit_positions.assert_not_called()


def test_hedge_common_lifecycle_refreshes_data_and_analyzes_strategy(
    default_conf, mocker
) -> None:
    bot = get_patched_freqtradebot(mocker, default_conf)
    bot.hedge_runtime = MagicMock()
    bot.hedge_runtime.config.managed_pair = "ETH/USDT:USDT"
    bot.active_pair_whitelist = []
    reload_markets = mocker.patch.object(bot.exchange, "reload_markets")
    mocker.patch.object(bot, "_refresh_active_whitelist", return_value=["BTC/USDT:USDT"])
    mocker.patch.object(
        bot.pairlists,
        "create_pair_list",
        return_value=[("BTC/USDT:USDT", "5m", "")],
    )
    refresh = mocker.patch.object(bot.dataprovider, "refresh")
    mocker.patch.object(bot.strategy, "gather_informative_pairs", return_value=[])
    bot_loop_start = mocker.patch.object(bot.strategy, "bot_loop_start")
    analyze = mocker.patch.object(bot.strategy, "analyze")

    bot._run_common_market_strategy_lifecycle(hedge_mode=True)

    reload_markets.assert_called_once_with()
    refresh.assert_called_once()
    bot_loop_start.assert_called_once()
    analyze.assert_called_once_with(["BTC/USDT:USDT", "ETH/USDT:USDT"])


def test_api_server_registers_hedge_control_plane_when_enabled(default_conf) -> None:
    default_conf["hedge_mode_enabled"] = True
    default_conf["api_server"] = {
        "username": "user",
        "password": "pass",
        "jwt_secret_key": "test-secret",
        "ws_token": "test-ws-token",
        "CORS_origins": [],
    }
    app = FastAPI()
    server = object.__new__(ApiServer)

    server.configure_app(app, default_conf)

    paths = app.openapi()["paths"]
    assert "get" in paths["/api/v1/hedge/readiness"]
    assert "post" in paths["/api/v1/hedge/orders"]
    assert hasattr(app.state, "hedge_event_hub")

    # FastAPI 0.137 keeps included routers as a lazy route tree, so direct
    # ``app.routes`` iteration no longer exposes every child path. Verify the
    # WebSocket route through the router matcher instead of private internals.
    from starlette.routing import Match

    scope = {
        "type": "websocket",
        "path": "/api/v1/hedge/ws",
        "root_path": "",
        "scheme": "ws",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
        "subprotocols": [],
    }
    assert any(route.matches(scope)[0] is Match.FULL for route in app.routes)
