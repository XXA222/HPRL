from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from fastapi import FastAPI
from fastapi.testclient import TestClient

from freqtrade.hedge.execution.action_group import ActionGroupExecutor
from freqtrade.hedge.execution.fake_exchange import FakeExchangeExecutionPort
from freqtrade.hedge.execution.idempotency import InMemoryIdempotencyStore
from freqtrade.hedge.execution.kill_switch import KillSwitch
from freqtrade.hedge.execution.service import (
    AllowAllRiskApproval,
    ExecutionService,
    InMemoryExecutionStore,
)
from freqtrade.hedge.execution.unknown_resolver import UnknownOrderResolver
from freqtrade.rpc.api_server.hedge_auth import HedgePrincipal, HedgeRole
from freqtrade.rpc.api_server.hedge_readonly import (
    ExecutionReadonlyQueryAdapter,
    create_hedge_readonly_router,
)
from freqtrade.rpc.api_server.hedge_schemas import (
    DualLegPositionsResponse,
    ReadinessStatusSchema,
    ReconciliationStatusSchema,
    RiskStatusSchema,
    UserStreamStatusSchema,
)


class BaseQuery:
    def positions(self, *, account_id: str, symbol: str):
        return DualLegPositionsResponse(
            account_id=account_id,
            symbol=symbol,
            legs=(),
            as_of=datetime.now(UTC),
        )

    def risk(self, *, account_id: str):
        return RiskStatusSchema(
            account_id=account_id,
            equity=Decimal("1000"),
            gross_notional=Decimal("0"),
            gross_exposure_ratio=Decimal("0"),
            margin_utilization=Decimal("0"),
            liquidation_buffer_ratio=Decimal("1"),
        )

    def reconciliation(self, *, account_id: str):
        return ReconciliationStatusSchema(status="HEALTHY")

    def readiness(self, *, account_id: str):
        return ReadinessStatusSchema(ready=True, kill_switch="RUNNING")

    def user_stream(self, *, account_id: str):
        return UserStreamStatusSchema(state="CONNECTED")

    def audit(self, *, account_id: str, limit: int):
        return []


def principal() -> HedgePrincipal:
    return HedgePrincipal("tester", HedgeRole.VIEWER)


def make_service() -> tuple[ExecutionService, FakeExchangeExecutionPort]:
    exchange = FakeExchangeExecutionPort()
    service = ExecutionService(
        risk=AllowAllRiskApproval(),
        exchange=exchange,
        store=InMemoryExecutionStore(),
        idempotency=InMemoryIdempotencyStore(),
        unknown_resolver=UnknownOrderResolver(exchange),
        kill_switch=KillSwitch(),
    )
    return service, exchange


def test_execution_order_and_action_group_read_endpoints() -> None:
    service, exchange = make_service()
    report = ActionGroupExecutor(service).execute_close_both(
        account_id="main",
        symbol="ETHUSDT",
        long_quantity=Decimal("0.4"),
        short_quantity=Decimal("0.6"),
        idempotency_key="api-group",
    )
    first = report.results[0]
    snapshot = exchange.fill_order(
        first.order.client_order_id,
        quantity=first.order.approved_quantity,
        price="3000",
    )
    service.apply_exchange_event(snapshot)

    app = FastAPI()
    app.include_router(
        create_hedge_readonly_router(
            BaseQuery(),
            principal_dependency=principal,
            execution_query=ExecutionReadonlyQueryAdapter(service),
        )
    )
    client = TestClient(app)

    orders = client.get("/hedge/orders", params={"account_id": "main"})
    assert orders.status_code == 200
    assert orders.json()["count"] == 2
    assert {item["status"] for item in orders.json()["orders"]} == {
        "FILLED",
        "ACKNOWLEDGED",
    }

    single = client.get(f"/hedge/orders/{first.order.client_order_id}")
    assert single.status_code == 200
    assert single.json()["filled_quantity"] == "0.4"
    assert single.json()["remaining_quantity"] == "0.0"

    group = client.get(f"/hedge/action-groups/{report.action_group_id}")
    assert group.status_code == 200
    assert group.json()["status"] == "IN_PROGRESS"
    assert group.json()["filled_quantity"] == "0.4"


def test_execution_order_status_filter_and_not_found() -> None:
    service, _ = make_service()
    app = FastAPI()
    app.include_router(
        create_hedge_readonly_router(
            BaseQuery(),
            principal_dependency=principal,
            execution_query=ExecutionReadonlyQueryAdapter(service),
        )
    )
    client = TestClient(app)
    assert client.get("/hedge/orders", params={"status": "bad"}).status_code == 422
    assert client.get("/hedge/orders/missing-order").status_code == 404


def test_qq_wechat_execution_read_commands() -> None:
    from freqtrade.rpc.api_server.hedge_readonly import HedgeReadonlyCommandDispatcher
    from freqtrade.rpc.api_server.hedge_schemas import HedgeReadonlyCommandSchema

    service, _ = make_service()
    report = ActionGroupExecutor(service).execute_close_both(
        account_id="main",
        symbol="ETHUSDT",
        long_quantity=Decimal("0.2"),
        short_quantity=Decimal("0.3"),
        idempotency_key="command-group",
    )
    dispatcher = HedgeReadonlyCommandDispatcher(
        BaseQuery(),
        execution_query=ExecutionReadonlyQueryAdapter(service),
    )
    orders = dispatcher.dispatch(
        HedgeReadonlyCommandSchema(
            source="QQ",
            command="hedge.orders",
            request_id="orders-1",
            account_id="main",
        )
    )
    assert orders.ok
    assert orders.data["count"] == 2

    one = dispatcher.dispatch(
        HedgeReadonlyCommandSchema(
            source="WECHAT",
            command="hedge.order",
            request_id="order-1",
            client_order_id=report.results[0].order.client_order_id,
        )
    )
    assert one.data["client_order_id"] == report.results[0].order.client_order_id

    group = dispatcher.dispatch(
        HedgeReadonlyCommandSchema(
            source="QQ",
            command="hedge.action_group",
            request_id="group-1",
            action_group_id=str(report.action_group_id),
        )
    )
    assert group.ok
    assert len(group.data["orders"]) == 2


def test_execution_command_reports_unconfigured_read_model() -> None:
    from freqtrade.rpc.api_server.hedge_readonly import HedgeReadonlyCommandDispatcher
    from freqtrade.rpc.api_server.hedge_schemas import HedgeReadonlyCommandSchema

    response = HedgeReadonlyCommandDispatcher(BaseQuery()).dispatch(
        HedgeReadonlyCommandSchema(
            source="QQ",
            command="hedge.orders",
            request_id="orders-unconfigured",
        )
    )
    assert not response.ok
    assert "not configured" in response.message


def test_execution_command_returns_structured_not_found() -> None:
    from freqtrade.rpc.api_server.hedge_readonly import HedgeReadonlyCommandDispatcher
    from freqtrade.rpc.api_server.hedge_schemas import HedgeReadonlyCommandSchema

    service, _ = make_service()
    response = HedgeReadonlyCommandDispatcher(
        BaseQuery(),
        execution_query=ExecutionReadonlyQueryAdapter(service),
    ).dispatch(
        HedgeReadonlyCommandSchema(
            source="WECHAT",
            command="hedge.order",
            request_id="missing-1",
            client_order_id="missing-order",
        )
    )
    assert not response.ok
    assert response.data == {}
