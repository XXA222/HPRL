import asyncio

import pytest

from freqtrade.hedge.exchange.base import BinanceHttpResponse
from freqtrade.hedge.exchange.binance_readonly import (
    AiohttpBinanceRestTransport,
    BinanceReadonlyClient,
    PermissionPolicy,
)
from freqtrade.hedge.exchange.clock_sync import ClockSynchronizer
from freqtrade.hedge.exchange.rate_limit import (
    BinanceDataError,
    BinancePermissionError,
    BinanceTransportError,
)

from ._helpers import FakeClock


class Transport:
    def __init__(self):
        self.calls = []
        self.provider = None

    def set_timestamp_provider(self, provider):
        self.provider = provider

    async def request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        responses = {
            "/fapi/v3/positionRisk": [],
            "/fapi/v1/openOrders": [],
            "/fapi/v1/time": {"serverTime": 1000},
            "/sapi/v1/account/apiRestrictions": {
                "enableReading": True,
                "enableFutures": True,
                "enableWithdrawals": False,
                "enableInternalTransfer": False,
                "permitsUniversalTransfer": False,
                "enableSpotAndMarginTrading": False,
            },
        }
        return BinanceHttpResponse(responses[path], 200, {})


def readonly_client(
    transport,
    clock,
    *,
    symbols=("BTCUSDT",),
    clock_sync=None,
):
    return BinanceReadonlyClient(
        transport=transport,
        account_id="acct",
        managed_symbols=list(symbols),
        clock=clock,
        clock_sync=clock_sync,
    )


def test_fetch_positions_none_really_reads_full_account():
    transport = Transport()
    clock = FakeClock()
    sync = ClockSynchronizer(
        clock=clock,
        sample_count=1,
        max_abs_skew_ms=10_000_000_000_000,
    )
    client = readonly_client(
        transport,
        clock,
        clock_sync=sync,
    )

    asyncio.run(client.fetch_positions(None))

    assert transport.calls[-1][2]["params"] == {}



def test_fetch_positions_v3_uses_symbol_config_for_margin_and_leverage():
    class PositionV3Transport:
        def __init__(self):
            self.calls = []

        def set_timestamp_provider(self, provider):
            self.provider = provider

        async def request(self, method, path, **kwargs):
            self.calls.append((method, path, kwargs))
            if path == "/fapi/v3/positionRisk":
                return BinanceHttpResponse(
                    [
                        {
                            "symbol": "ETHUSDT",
                            "positionSide": "LONG",
                            "positionAmt": "1",
                            "entryPrice": "3000",
                            "markPrice": "3010",
                            "unRealizedProfit": "10",
                            "liquidationPrice": "0",
                            "isolatedMargin": "0",
                            "isolatedWallet": "0",
                            "updateTime": 1,
                        }
                    ],
                    200,
                    {},
                )
            if path == "/fapi/v1/symbolConfig":
                return BinanceHttpResponse(
                    [
                        {
                            "symbol": "ETHUSDT",
                            "marginType": "CROSSED",
                            "leverage": 7,
                        }
                    ],
                    200,
                    {},
                )
            raise AssertionError(path)

    clock = FakeClock()
    transport = PositionV3Transport()
    client = readonly_client(transport, clock, symbols=("ETHUSDT",))

    positions = asyncio.run(client.fetch_positions(None))

    assert len(positions) == 1
    assert positions[0].margin_mode == "cross"
    assert positions[0].leverage == 7
    assert "marginType" not in positions[0].raw
    assert "leverage" not in positions[0].raw
    assert [call[1] for call in transport.calls] == [
        "/fapi/v3/positionRisk",
        "/fapi/v1/symbolConfig",
    ]

def test_permission_preflight_requires_read_and_futures_but_forbids_money_movement():
    transport = Transport()
    clock = FakeClock()
    client = readonly_client(transport, clock)

    report = asyncio.run(client.preflight_permissions())

    assert report.strict_readonly_verified
    assert report.runtime_readonly_enforced
    assert any(
        "NOT_SEPARATELY_VERIFIABLE" in warning
        for warning in report.warnings
    )

    async def bad_request(method, path, **kwargs):
        return BinanceHttpResponse(
            {
                "enableReading": True,
                "enableFutures": True,
                "enableWithdrawals": True,
            },
            200,
            {},
        )

    transport.request = bad_request
    with pytest.raises(BinancePermissionError):
        asyncio.run(client.preflight_permissions())


def test_permission_preflight_accepts_usdm_read_when_enable_futures_is_false():
    transport = Transport()
    clock = FakeClock()
    client = readonly_client(transport, clock)

    async def readonly_request(method, path, **kwargs):
        transport.calls.append((method, path, kwargs))
        if path == "/sapi/v1/account/apiRestrictions":
            return BinanceHttpResponse(
                {
                    "enableReading": True,
                    "enableFutures": False,
                    "enableWithdrawals": False,
                    "enableInternalTransfer": False,
                    "permitsUniversalTransfer": False,
                    "enableSpotAndMarginTrading": False,
                },
                200,
                {},
            )
        if path == "/fapi/v3/account":
            return BinanceHttpResponse({"assets": [], "positions": []}, 200, {})
        raise AssertionError(path)

    transport.request = readonly_request
    report = asyncio.run(client.preflight_permissions())

    assert report.strict_readonly_verified
    assert report.futures_enabled is False
    assert any(
        "AUTHENTICATED_USDM_USER_DATA_READ_CONFIRMED" in warning
        for warning in report.warnings
    )
    assert any(call[1] == "/fapi/v3/account" for call in transport.calls)


def test_permission_preflight_rejects_when_metadata_and_usdm_read_are_unavailable():
    transport = Transport()
    clock = FakeClock()
    client = readonly_client(transport, clock)

    async def denied_request(method, path, **kwargs):
        transport.calls.append((method, path, kwargs))
        if path == "/sapi/v1/account/apiRestrictions":
            return BinanceHttpResponse(
                {
                    "enableReading": True,
                    "enableFutures": False,
                    "enableWithdrawals": False,
                },
                200,
                {},
            )
        if path == "/fapi/v3/account":
            raise BinancePermissionError("denied", status=401, code=-2015)
        raise AssertionError(path)

    transport.request = denied_request
    with pytest.raises(BinancePermissionError, match="FUTURES_PERMISSION_MISSING"):
        asyncio.run(client.preflight_permissions())


def test_transport_has_hard_write_endpoint_denylist():
    transport = AiohttpBinanceRestTransport(
        api_key="x",
        api_secret="y",
        timestamp_provider=lambda: 1,
    )
    with pytest.raises(BinancePermissionError):
        transport._ensure_allowed("POST", "/fapi/v1/order")


def fill_payload(trade_id):
    return {
        "symbol": "BTCUSDT",
        "positionSide": "LONG",
        "id": trade_id,
        "orderId": 10,
        "side": "BUY",
        "qty": "0.1",
        "price": "100",
        "commission": "0",
        "realizedPnl": "0",
        "time": trade_id,
    }


def order_payload(order_id):
    return {
        "symbol": "BTCUSDT",
        "positionSide": "LONG",
        "orderId": order_id,
        "clientOrderId": f"c-{order_id}",
        "side": "BUY",
        "type": "LIMIT",
        "status": "FILLED",
        "origQty": "1",
        "executedQty": "1",
        "avgPrice": "100",
        "reduceOnly": False,
        "updateTime": order_id,
    }


def open_order_payload(order_id):
    return {
        **order_payload(order_id),
        "status": "NEW",
        "executedQty": "0",
        "avgPrice": "0",
    }


def test_fills_and_order_history_paginate_without_gaps():
    class PagingTransport:
        def __init__(self):
            self.calls = []

        async def request(self, method, path, **kwargs):
            params = kwargs.get("params", {})
            self.calls.append((path, params))
            if path.endswith("userTrades"):
                data = (
                    [fill_payload(1), fill_payload(2)]
                    if "fromId" not in params
                    else [fill_payload(3)]
                )
            else:
                data = (
                    [order_payload(1), order_payload(2)]
                    if "orderId" not in params
                    else [order_payload(3)]
                )
            return BinanceHttpResponse(data, 200, {})

    transport = PagingTransport()
    clock = FakeClock()
    client = readonly_client(transport, clock)

    fills = asyncio.run(
        client.fetch_fills(
            ["BTCUSDT"],
            start_time_ms=0,
            end_time_ms=10,
            limit=2,
        )
    )
    orders = asyncio.run(
        client.fetch_all_orders(
            ["BTCUSDT"],
            start_time_ms=0,
            end_time_ms=10,
            limit=2,
        )
    )

    assert [item.exchange_trade_id for item in fills] == ["1", "2", "3"]
    assert [item.exchange_order_id for item in orders] == ["1", "2", "3"]
    assert any(
        params.get("fromId") == 3
        for path, params in transport.calls
        if path.endswith("userTrades")
    )
    assert any(
        params.get("orderId") == 3
        for path, params in transport.calls
        if path.endswith("allOrders")
    )


def test_fetch_bundle_recovers_open_order_missing_from_bounded_all_orders_history():
    current = open_order_payload(4242)

    class OpenOrderRecoveryTransport:
        def __init__(self):
            self.calls = []

        async def request(self, method, path, **kwargs):
            params = kwargs.get("params", {})
            self.calls.append((path, params))
            responses = {
                "/fapi/v3/account": account_payload(),
                "/fapi/v3/positionRisk": position_rows(),
                "/fapi/v1/openOrders": [current],
                "/fapi/v1/positionSide/dual": {"dualSidePosition": True},
                "/fapi/v1/symbolConfig": [
                    {"symbol": "BTCUSDT", "marginType": "CROSSED", "leverage": 5}
                ],
                "/fapi/v1/income": [],
                "/fapi/v1/allOrders": [],
                "/fapi/v1/userTrades": [],
                "/fapi/v1/order": current,
            }
            return BinanceHttpResponse(responses[path], 200, {})

    clock = FakeClock()
    transport = OpenOrderRecoveryTransport()
    client = readonly_client(transport, clock)
    bundle = asyncio.run(
        client.fetch_bundle(
            include_fills=True,
            fill_start_time_ms=int(clock.now().timestamp() * 1000) - 1000,
        )
    )

    assert [item.exchange_order_id for item in bundle.open_orders] == ["4242"]
    assert [item.exchange_order_id for item in bundle.order_history] == ["4242"]
    assert bundle.order_history[0].source == "BINANCE_REST_QUERY_ORDER"
    assert bundle.order_history_query_recovered_ids == ("4242",)
    assert bundle.order_history_snapshot_fallback_ids == ()
    query_call = next(params for path, params in transport.calls if path == "/fapi/v1/order")
    assert query_call == {"symbol": "BTCUSDT", "orderId": "4242"}


def test_open_order_retention_gap_uses_explicit_current_snapshot_fallback():
    current = open_order_payload(5252)

    class RetentionTransport:
        async def request(self, method, path, **kwargs):
            responses = {
                "/fapi/v3/account": account_payload(),
                "/fapi/v3/positionRisk": position_rows(),
                "/fapi/v1/openOrders": [current],
                "/fapi/v1/positionSide/dual": {"dualSidePosition": True},
                "/fapi/v1/symbolConfig": [
                    {"symbol": "BTCUSDT", "marginType": "CROSSED", "leverage": 5}
                ],
                "/fapi/v1/income": [],
                "/fapi/v1/allOrders": [],
                "/fapi/v1/userTrades": [],
            }
            if path == "/fapi/v1/order":
                raise BinanceTransportError(
                    "Order does not exist", status=400, code=-2013, retryable=False
                )
            return BinanceHttpResponse(responses[path], 200, {})

    clock = FakeClock()
    client = readonly_client(RetentionTransport(), clock)
    bundle = asyncio.run(
        client.fetch_bundle(
            include_fills=True,
            fill_start_time_ms=int(clock.now().timestamp() * 1000) - 1000,
        )
    )

    assert [item.exchange_order_id for item in bundle.order_history] == ["5252"]
    assert bundle.order_history[0].source == "BINANCE_REST_OPEN_ORDERS_RETENTION_FALLBACK"
    assert bundle.order_history_query_recovered_ids == ()
    assert bundle.order_history_snapshot_fallback_ids == ("5252",)


def test_income_history_uses_current_one_based_page_parameter():
    class PagingIncomeTransport:
        def __init__(self):
            self.params = []

        async def request(self, method, path, **kwargs):
            params = kwargs["params"]
            self.params.append(params)
            if params["page"] == 1:
                data = [
                    {
                        "tranId": 1,
                        "time": 1,
                        "incomeType": "FUNDING_FEE",
                        "income": "1",
                    },
                    {
                        "tranId": 1,
                        "time": 2,
                        "incomeType": "COMMISSION",
                        "income": "-1",
                    },
                ]
            elif params["page"] == 2:
                data = [
                    {
                        "tranId": 2,
                        "time": 3,
                        "incomeType": "FUNDING_FEE",
                        "income": "2",
                    }
                ]
            else:
                raise AssertionError(params)
            return BinanceHttpResponse(data, 200, {})

    clock = FakeClock()
    transport = PagingIncomeTransport()
    client = readonly_client(transport, clock)

    values = asyncio.run(
        client.fetch_income_history(
            start_time_ms=0,
            end_time_ms=3,
            limit=2,
        )
    )

    assert [
        (item["incomeType"], item["tranId"])
        for item in values
    ] == [
        ("FUNDING_FEE", 1),
        ("COMMISSION", 1),
        ("FUNDING_FEE", 2),
    ]
    assert [params["page"] for params in transport.params] == [1, 2]
    assert all(
        params["startTime"] == 0 and params["endTime"] == 3
        for params in transport.params
    )


def test_income_history_max_page_budget_fails_closed():
    class SaturatedTransport:
        async def request(self, method, path, **kwargs):
            page = kwargs["params"]["page"]
            return BinanceHttpResponse(
                [
                    {
                        "tranId": page,
                        "time": page,
                        "incomeType": "OTHER",
                        "income": "0",
                    }
                ],
                200,
                {},
            )

    clock = FakeClock()
    client = readonly_client(SaturatedTransport(), clock)
    with pytest.raises(BinanceDataError, match="exceeded max_pages"):
        asyncio.run(
            client.fetch_income_history(
                start_time_ms=0,
                end_time_ms=10,
                limit=1,
                max_pages=2,
            )
        )


def test_low_activity_history_walks_every_seven_day_window():
    seven_days = 7 * 24 * 60 * 60 * 1000

    class WindowTransport:
        def __init__(self):
            self.calls = []

        async def request(self, method, path, **kwargs):
            params = kwargs["params"]
            self.calls.append((path, params))
            event_time = params["startTime"]
            if path.endswith("userTrades"):
                data = [fill_payload(event_time + 1)]
                data[0]["time"] = event_time
            else:
                data = [order_payload(event_time + 1)]
                data[0]["updateTime"] = event_time
            return BinanceHttpResponse(data, 200, {})

    clock = FakeClock()
    transport = WindowTransport()
    client = readonly_client(transport, clock)
    end = 2 * seven_days + 10

    fills = asyncio.run(
        client.fetch_fills(
            ["BTCUSDT"],
            start_time_ms=0,
            end_time_ms=end,
            limit=1000,
        )
    )
    orders = asyncio.run(
        client.fetch_all_orders(
            ["BTCUSDT"],
            start_time_ms=0,
            end_time_ms=end,
            limit=1000,
        )
    )

    assert len(fills) == 3
    assert len(orders) == 3
    fill_windows = [
        (params["startTime"], params["endTime"])
        for path, params in transport.calls
        if path.endswith("userTrades")
    ]
    order_windows = [
        (params["startTime"], params["endTime"])
        for path, params in transport.calls
        if path.endswith("allOrders")
    ]
    assert len(fill_windows) == 3
    assert len(order_windows) == 3
    assert all(
        end_time - start_time < seven_days
        for start_time, end_time in fill_windows + order_windows
    )


def account_payload():
    return {
        "totalWalletBalance": "1000",
        "availableBalance": "900",
        "totalMarginBalance": "1000",
        "totalInitialMargin": "0",
        "totalMaintMargin": "0",
        "totalUnrealizedProfit": "0",
        "assets": [],
    }


def position_rows():
    base = {
        "symbol": "BTCUSDT",
        "positionAmt": "0",
        "entryPrice": "0",
        "markPrice": "60000",
        "unRealizedProfit": "0",
        "liquidationPrice": "0",
        "leverage": "5",
        "marginType": "cross",
        "updateTime": 1,
    }
    return [
        {**base, "positionSide": "LONG"},
        {**base, "positionSide": "SHORT"},
    ]


def test_fetch_bundle_reads_all_account_surfaces_and_history():
    class BundleTransport:
        def __init__(self):
            self.calls = []

        async def request(self, method, path, **kwargs):
            self.calls.append((path, kwargs.get("params", {})))
            responses = {
                "/fapi/v3/account": account_payload(),
                "/fapi/v3/positionRisk": position_rows(),
                "/fapi/v1/openOrders": [],
                "/fapi/v1/positionSide/dual": {
                    "dualSidePosition": True,
                },
                "/fapi/v1/symbolConfig": [
                    {
                        "symbol": "BTCUSDT",
                        "marginType": "CROSSED",
                        "leverage": 5,
                        "isAutoAddMargin": False,
                    }
                ],
                "/fapi/v1/allOrders": [],
                "/fapi/v1/userTrades": [],
                "/fapi/v1/income": [],
            }
            return BinanceHttpResponse(responses[path], 200, {})

    clock = FakeClock()
    transport = BundleTransport()
    client = readonly_client(transport, clock)
    bundle_value = asyncio.run(
        client.fetch_bundle(
            include_fills=True,
            fill_start_time_ms=int(clock.now().timestamp() * 1000) - 1000,
        )
    )

    assert bundle_value.configuration.hedge_mode
    assert bundle_value.order_history == ()
    position_call = next(
        params
        for path, params in transport.calls
        if path.endswith("positionRisk")
    )
    open_orders_call = next(
        params
        for path, params in transport.calls
        if path.endswith("openOrders")
    )
    symbol_config_call = next(
        params
        for path, params in transport.calls
        if path.endswith("symbolConfig")
    )
    assert position_call == {}
    assert open_orders_call == {}
    assert symbol_config_call == {}


def test_configuration_confirmation_uses_symbol_config_even_when_flat():
    clock = FakeClock()

    class ModeTransport:
        async def request(self, method, path, **kwargs):
            if path.endswith("positionSide/dual"):
                return BinanceHttpResponse(
                    {"dualSidePosition": True},
                    200,
                    {},
                )
            raise AssertionError(path)

    client = readonly_client(ModeTransport(), clock)
    configuration = asyncio.run(
        client.confirm_configuration(
            [
                {
                    "symbol": "BTCUSDT",
                    "marginType": "CROSSED",
                    "leverage": 5,
                }
            ]
        )
    )

    assert configuration.active_margin_modes == ("cross",)
    assert configuration.leverage_by_symbol_side == {
        "BTCUSDT:LONG": 5,
        "BTCUSDT:SHORT": 5,
    }




def test_bundle_tolerates_accountwide_delivery_symbol_config_metadata():
    class BundleTransport:
        def __init__(self):
            self.calls = []

        async def request(self, method, path, **kwargs):
            self.calls.append((path, kwargs.get("params", {})))
            if path == "/fapi/v3/account":
                return BinanceHttpResponse(account_payload(), 200, {})
            if path == "/fapi/v3/positionRisk":
                return BinanceHttpResponse(
                    [
                        {
                            "symbol": "ETHUSDT",
                            "positionSide": "LONG",
                            "positionAmt": "0",
                            "entryPrice": "0",
                            "markPrice": "3000",
                            "unRealizedProfit": "0",
                            "liquidationPrice": "0",
                            "isolatedMargin": "0",
                            "isolatedWallet": "0",
                            "updateTime": 1,
                        }
                    ],
                    200,
                    {},
                )
            if path == "/fapi/v1/openOrders":
                return BinanceHttpResponse([], 200, {})
            if path == "/fapi/v1/positionSide/dual":
                return BinanceHttpResponse({"dualSidePosition": True}, 200, {})
            if path == "/fapi/v1/symbolConfig":
                return BinanceHttpResponse(
                    [
                        {
                            "symbol": "ETHUSDT_261225",
                            "marginType": "CROSSED",
                            "leverage": 4,
                        },
                        {
                            "symbol": "ETHUSDT",
                            "marginType": "CROSSED",
                            "leverage": 7,
                        },
                    ],
                    200,
                    {},
                )
            raise AssertionError(path)

    clock = FakeClock()
    transport = BundleTransport()
    client = readonly_client(transport, clock, symbols=("ETHUSDT",))
    bundle_value = asyncio.run(client.fetch_bundle(include_fills=False))

    assert bundle_value.configuration.active_margin_modes == ("cross",)
    assert bundle_value.configuration.leverage_by_symbol_side == {
        "ETHUSDT:LONG": 7,
        "ETHUSDT:SHORT": 7,
    }
    assert tuple(item.symbol for item in bundle_value.positions) == ("ETHUSDT",)

def test_configuration_confirmation_ignores_delivery_contract_symbol_rows():
    clock = FakeClock()

    class ModeTransport:
        async def request(self, method, path, **kwargs):
            if path.endswith("positionSide/dual"):
                return BinanceHttpResponse({"dualSidePosition": True}, 200, {})
            raise AssertionError(path)

    client = readonly_client(ModeTransport(), clock, symbols=("ETHUSDT",))
    configuration = asyncio.run(
        client.confirm_configuration(
            [
                {
                    "symbol": "ETHUSDT_261225",
                    "marginType": "CROSSED",
                    "leverage": 4,
                },
                {
                    "symbol": "ETHUSDT",
                    "marginType": "CROSSED",
                    "leverage": 7,
                },
            ]
        )
    )

    assert configuration.active_margin_modes == ("cross",)
    assert configuration.leverage_by_symbol_side == {
        "ETHUSDT:LONG": 7,
        "ETHUSDT:SHORT": 7,
    }


def test_history_symbol_discovery_ignores_delivery_income_without_current_exposure():
    symbols = BinanceReadonlyClient._discovered_history_symbols(
        {"positions": []},
        (),
        (),
        (
            {"symbol": "ETHUSDT_261225"},
            {"symbol": "ETHUSDT"},
        ),
        ("BTCUSDT",),
    )

    assert symbols == ("BTCUSDT", "ETHUSDT")


def test_account_payload_delivery_exposure_fails_closed():
    payload = {
        "positions": [
            {
                "symbol": "ETHUSDT_261225",
                "positionAmt": "1",
                "openOrderInitialMargin": "0",
            }
        ]
    }

    with pytest.raises(BinanceDataError, match="non-perpetual account exposure"):
        BinanceReadonlyClient._account_payload_history_symbols(payload)

def test_configuration_confirmation_rejects_missing_noncross_and_zero_leverage():
    clock = FakeClock()

    class ModeTransport:
        async def request(self, method, path, **kwargs):
            return BinanceHttpResponse(
                {"dualSidePosition": True},
                200,
                {},
            )

    client = readonly_client(ModeTransport(), clock)
    with pytest.raises(BinancePermissionError, match="Missing managed"):
        asyncio.run(client.confirm_configuration([]))
    with pytest.raises(BinancePermissionError, match="Non-cross"):
        asyncio.run(
            client.confirm_configuration(
                [
                    {
                        "symbol": "BTCUSDT",
                        "marginType": "ISOLATED",
                        "leverage": 5,
                    }
                ]
            )
        )
    with pytest.raises(BinanceDataError, match="leverage must be positive"):
        asyncio.run(
            client.confirm_configuration(
                [
                    {
                        "symbol": "BTCUSDT",
                        "marginType": "CROSSED",
                        "leverage": 0,
                    }
                ]
            )
        )


def test_empty_symbol_filter_is_rejected_instead_of_becoming_full_account_read():
    transport = Transport()
    clock = FakeClock()
    client = readonly_client(transport, clock)

    with pytest.raises(ValueError):
        asyncio.run(client.fetch_positions([]))
    with pytest.raises(ValueError):
        asyncio.run(client.fetch_open_orders(""))


def test_same_numeric_ids_on_different_symbols_are_not_collapsed():
    class MultiSymbolTransport:
        async def request(self, method, path, **kwargs):
            symbol = kwargs["params"]["symbol"]
            if path.endswith("userTrades"):
                row = fill_payload(1)
                row["symbol"] = symbol
                row["time"] = 1
            else:
                row = order_payload(1)
                row["symbol"] = symbol
                row["updateTime"] = 1
            return BinanceHttpResponse([row], 200, {})

    clock = FakeClock()
    client = readonly_client(
        MultiSymbolTransport(),
        clock,
        symbols=("BTCUSDT", "ETHUSDT"),
    )
    fills = asyncio.run(
        client.fetch_fills(
            ["BTCUSDT", "ETHUSDT"],
            start_time_ms=0,
            end_time_ms=2,
        )
    )
    orders = asyncio.run(
        client.fetch_all_orders(
            ["BTCUSDT", "ETHUSDT"],
            start_time_ms=0,
            end_time_ms=2,
        )
    )

    assert {
        (item.symbol, item.exchange_trade_id)
        for item in fills
    } == {
        ("BTCUSDT", "1"),
        ("ETHUSDT", "1"),
    }
    assert {
        (item.symbol, item.exchange_order_id)
        for item in orders
    } == {
        ("BTCUSDT", "1"),
        ("ETHUSDT", "1"),
    }


def test_permission_policy_rejects_contradictory_futures_requirements():
    with pytest.raises(ValueError, match="contradictory"):
        PermissionPolicy(
            require_futures_enabled=True,
            forbid_futures_trading_permission=True,
        )


def test_transport_rejects_caller_supplied_signature():
    transport = AiohttpBinanceRestTransport(
        api_key="x",
        api_secret="y",
        timestamp_provider=lambda: 1,
    )
    with pytest.raises(BinanceDataError, match="Caller-supplied"):
        asyncio.run(
            transport.request(
                "GET",
                "/fapi/v1/time",
                params={"signature": "bad"},
            )
        )


def test_transport_owns_signed_timestamp_and_receive_window():
    transport = AiohttpBinanceRestTransport(
        api_key="x",
        api_secret="y",
        timestamp_provider=lambda: 123,
        recv_window_ms=4000,
    )
    signed = transport._request_parameters({"symbol": "BTCUSDT"}, signed=True)

    assert signed["timestamp"] == 123
    assert signed["recvWindow"] == 4000
    for protected in ("timestamp", "recvWindow", "signature"):
        with pytest.raises(BinanceDataError, match="not allowed"):
            transport._request_parameters({protected: 1}, signed=True)


def test_transport_rejects_cross_group_paths_and_unsafe_base_urls():
    transport = AiohttpBinanceRestTransport(
        api_key="x",
        api_secret="y",
        timestamp_provider=lambda: 1,
    )
    with pytest.raises(BinancePermissionError, match="belongs to futures"):
        asyncio.run(
            transport.request(
                "GET",
                "/fapi/v1/time",
                api_group="spot",
            )
        )
    with pytest.raises(ValueError, match="HTTPS"):
        AiohttpBinanceRestTransport(
            api_key="x",
            api_secret="y",
            futures_base_url="http://example.com",
        )


def test_transport_rejects_nonfinite_numeric_configuration():
    with pytest.raises(ValueError, match="finite positive"):
        AiohttpBinanceRestTransport(
            api_key="x",
            api_secret="y",
            timeout_seconds=float("nan"),
        )
    with pytest.raises(ValueError, match="60000"):
        AiohttpBinanceRestTransport(
            api_key="x",
            api_secret="y",
            recv_window_ms=60_001,
        )
    with pytest.raises(ValueError, match="finite positive"):
        BinanceReadonlyClient(
            transport=Transport(),
            account_id="acct",
            managed_symbols=["BTCUSDT"],
            max_collection_span_seconds=float("inf"),
        )


def test_fetch_bundle_collects_independent_rest_surfaces_concurrently():
    from freqtrade.hedge.exchange.base import AccountConfigurationFact

    class ConcurrentClient(BinanceReadonlyClient):
        def __init__(self, clock):
            super().__init__(
                transport=Transport(),
                account_id="acct",
                managed_symbols=["BTCUSDT"],
                clock=clock,
            )
            self.entered = 0
            self.all_entered = asyncio.Event()

        async def _barrier(self):
            self.entered += 1
            if self.entered == 4:
                self.all_entered.set()
            await asyncio.wait_for(self.all_entered.wait(), timeout=0.5)

        async def fetch_account_info(self):
            await self._barrier()
            return account_payload()

        async def fetch_positions(self, symbols=None):
            await self._barrier()
            return ()

        async def fetch_open_orders(self, symbol=None):
            await self._barrier()
            return ()

        async def confirm_configuration(self, symbol_configurations=None):
            await self._barrier()
            return AccountConfigurationFact(
                account_id="acct",
                hedge_mode=True,
                active_margin_modes=("cross",),
                leverage_by_symbol_side={
                    "BTCUSDT:LONG": 5,
                    "BTCUSDT:SHORT": 5,
                },
                observed_at=self.clock.now(),
                raw={},
            )

    clock = FakeClock()
    client = ConcurrentClient(clock)

    result = asyncio.run(client.fetch_bundle(include_fills=False))

    assert client.entered == 4
    assert result.positions == ()
    assert result.open_orders == ()


def test_history_symbol_discovery_includes_unmanaged_and_income_symbols():
    from decimal import Decimal

    from freqtrade.hedge.exchange.base import PositionFact

    clock = FakeClock()
    unmanaged = PositionFact(
        "acct",
        "ETHUSDT",
        "LONG",
        Decimal(1),
        Decimal(100),
        Decimal(101),
        Decimal(1),
        None,
        5,
        "cross",
        1,
        clock.now(),
        "TEST",
        {},
    )

    symbols = BinanceReadonlyClient._discovered_history_symbols(
        {"positions": []},
        (unmanaged,),
        (),
        ({"symbol": "SOLUSDT"},),
        ("BTCUSDT",),
    )

    assert symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT")


def test_bundle_collects_history_evidence_even_when_unmanaged_position_exists():
    class BundleTransport:
        def __init__(self):
            self.history_symbols = []

        async def request(self, method, path, **kwargs):
            params = kwargs.get("params", {})
            if path == "/fapi/v3/account":
                return BinanceHttpResponse(account_payload(), 200, {})
            if path == "/fapi/v3/positionRisk":
                eth = {
                    **position_rows()[0],
                    "symbol": "ETHUSDT",
                    "positionAmt": "1",
                    "entryPrice": "100",
                }
                return BinanceHttpResponse([*position_rows(), eth], 200, {})
            if path == "/fapi/v1/openOrders":
                return BinanceHttpResponse([], 200, {})
            if path == "/fapi/v1/positionSide/dual":
                return BinanceHttpResponse({"dualSidePosition": True}, 200, {})
            if path == "/fapi/v1/symbolConfig":
                return BinanceHttpResponse(
                    [
                        {
                            "symbol": "BTCUSDT",
                            "marginType": "CROSSED",
                            "leverage": 5,
                        }
                    ],
                    200,
                    {},
                )
            if path == "/fapi/v1/income":
                return BinanceHttpResponse(
                    [
                        {
                            "tranId": 1,
                            "time": 1,
                            "incomeType": "FUNDING_FEE",
                            "income": "0",
                            "asset": "USDT",
                            "symbol": "SOLUSDT",
                        }
                    ],
                    200,
                    {},
                )
            if path in {"/fapi/v1/allOrders", "/fapi/v1/userTrades"}:
                self.history_symbols.append(params["symbol"])
                return BinanceHttpResponse([], 200, {})
            raise AssertionError(path)

    clock = FakeClock()
    transport = BundleTransport()
    client = readonly_client(transport, clock)

    bundle_value = asyncio.run(
        client.fetch_bundle(
            include_fills=True,
            fill_start_time_ms=int(clock.now().timestamp() * 1000) - 1000,
        )
    )

    assert any(item.symbol == "ETHUSDT" and item.quantity > 0 for item in bundle_value.positions)
    assert set(transport.history_symbols) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def test_transport_telemetry_counts_retry_and_success():
    from freqtrade.hedge.exchange.rate_limit import BinanceTransportError, RetryPolicy

    class TelemetryTransport(AiohttpBinanceRestTransport):
        def __init__(self):
            super().__init__(
                api_key="x",
                api_secret="y",
                timestamp_provider=lambda: 1,
                futures_base_url="http://localhost:18000",
                retry_policy=RetryPolicy(
                    max_attempts=2,
                    base_delay_seconds=0,
                    max_delay_seconds=0,
                    jitter_ratio=0,
                ),
                clock=FakeClock(),
            )
            self.responses = 0

        async def _request_once(self, *args, **kwargs):
            self.responses += 1
            if self.responses == 1:
                raise BinanceTransportError("temporary", status=500, retryable=True)
            return BinanceHttpResponse({"serverTime": 1}, 200, {})

    transport = TelemetryTransport()

    response = asyncio.run(transport.request("GET", "/fapi/v1/time"))

    assert response.status == 200
    assert transport.telemetry.logical_request_count == 1
    assert transport.telemetry.attempt_count == 2
    assert transport.telemetry.retry_count == 1
    assert transport.telemetry.server_error_count == 1
    assert transport.telemetry.error_count == 0


def test_transport_applies_explicit_proxy_without_exposing_it_in_telemetry():
    class Response:
        status = 200
        headers = {}

        async def json(self, content_type=None):
            return {"serverTime": 123}

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class Session:
        def __init__(self):
            self.calls = []

        def request(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            return Response()

    session = Session()
    transport = AiohttpBinanceRestTransport(
        api_key="x",
        api_secret="y",
        session=session,
        proxy_url="http://127.0.0.1:7897/",
    )

    response = asyncio.run(transport.request("GET", "/fapi/v1/time"))

    assert response.status == 200
    assert session.calls[0][2]["proxy"] == "http://127.0.0.1:7897"
    assert "proxy" not in repr(transport.telemetry).lower()


def test_transport_rejects_unsafe_proxy_urls():
    with pytest.raises(ValueError, match=r"HTTP\(S\)"):
        AiohttpBinanceRestTransport(
            api_key="x",
            api_secret="y",
            proxy_url="socks5://127.0.0.1:7897",
        )
    with pytest.raises(ValueError, match="credentials"):
        AiohttpBinanceRestTransport(
            api_key="x",
            api_secret="y",
            proxy_url="http://user:secret@127.0.0.1:7897",
        )
