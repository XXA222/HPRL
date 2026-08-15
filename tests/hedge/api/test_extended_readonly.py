from datetime import UTC, datetime
from decimal import Decimal

from fastapi import FastAPI
from fastapi.testclient import TestClient

from freqtrade.hedge.execution import (
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
    build_integrated_fake_runtime,
)
from freqtrade.rpc.api_server.hedge_auth import HedgePrincipal, HedgeRole
from freqtrade.rpc.api_server.hedge_readonly import (
    ExecutionLedgerReadonlyAdapter,
    create_hedge_readonly_router,
)
from freqtrade.rpc.api_server.hedge_schemas import (
    DualLegPositionsResponse,
    LegPositionSchema,
    OperationAuditSchema,
    ReadinessStatusSchema,
    ReconciliationStatusSchema,
    RiskStatusSchema,
    UserStreamStatusSchema,
)


class BaseQuery:
    def positions(self, *, account_id: str, symbol: str):
        return DualLegPositionsResponse(account_id=account_id, symbol=symbol, legs=(), as_of=datetime.now(UTC))
    def risk(self, *, account_id: str):
        return RiskStatusSchema(account_id=account_id, equity=Decimal("1000"), gross_notional=Decimal("0"), gross_exposure_ratio=Decimal("0"), margin_utilization=Decimal("0"), liquidation_buffer_ratio=Decimal("1"))
    def reconciliation(self, *, account_id: str):
        return ReconciliationStatusSchema(status="HEALTHY")
    def readiness(self, *, account_id: str):
        return ReadinessStatusSchema(state="READY", reasons=(), checked_at=datetime.now(UTC))
    def user_stream(self, *, account_id: str):
        return UserStreamStatusSchema(state="CONNECTED", last_event_at=datetime.now(UTC), reconnect_count=0, gap_detected=False)
    def audit(self, *, account_id: str, limit: int):
        return []


def principal() -> HedgePrincipal:
    return HedgePrincipal(subject="test", role=HedgeRole.VIEWER)


def test_pair_summary_and_events_routes() -> None:
    runtime = build_integrated_fake_runtime()
    submitted = runtime.engine.submit(
        OrderIntent(
            account_id="acct",
            symbol="ETHUSDT",
            position_side=PositionSide.LONG,
            action=IntentAction.OPEN,
            quantity=Decimal("1"),
            idempotency_key="open",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("100"),
        )
    )
    snapshot = runtime.exchange.fill_order(
        submitted.order.client_order_id, quantity="1", price="100", exchange_trade_id="t1"
    )
    runtime.engine.apply_exchange_event(snapshot)
    adapter = ExecutionLedgerReadonlyAdapter(runtime.core, runtime.ledger)
    app = FastAPI()
    app.include_router(
        create_hedge_readonly_router(
            BaseQuery(), principal_dependency=principal, execution_query=adapter
        )
    )
    client = TestClient(app)
    summary = client.get("/hedge/pair-summary/ETHUSDT?account_id=acct")
    assert summary.status_code == 200
    assert summary.json()["long_quantity"] == "1"
    events = client.get("/hedge/events?account_id=acct")
    assert events.status_code == 200
    assert events.json()["count"] >= 2
