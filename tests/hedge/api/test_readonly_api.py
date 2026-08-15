from datetime import UTC, datetime
from decimal import Decimal

from fastapi import FastAPI
from fastapi.testclient import TestClient

from freqtrade.rpc.api_server.hedge_auth import HedgePrincipal, HedgeRole
from freqtrade.rpc.api_server.hedge_readonly import (
    create_disabled_hedge_write_router,
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


class Query:
    def positions(self, *, account_id: str, symbol: str):
        return DualLegPositionsResponse(
            account_id=account_id,
            symbol=symbol,
            legs=(
                LegPositionSchema(
                    account_id=account_id,
                    symbol=symbol,
                    position_side="LONG",
                    quantity=Decimal("1"),
                    entry_price=Decimal("100"),
                ),
                LegPositionSchema(
                    account_id=account_id,
                    symbol=symbol,
                    position_side="SHORT",
                    quantity=Decimal("2"),
                    entry_price=Decimal("101"),
                ),
            ),
            as_of=datetime.now(UTC),
        )

    def risk(self, *, account_id: str):
        return RiskStatusSchema(
            account_id=account_id,
            equity=Decimal("1000"),
            gross_notional=Decimal("300"),
            gross_exposure_ratio=Decimal("0.3"),
            margin_utilization=Decimal("0.1"),
            liquidation_buffer_ratio=Decimal("0.9"),
        )

    def reconciliation(self, *, account_id: str):
        return ReconciliationStatusSchema(status="HEALTHY")

    def readiness(self, *, account_id: str):
        return ReadinessStatusSchema(
            ready=True,
            kill_switch="RUNNING",
            checks={"rest": True, "user_stream": True},
        )

    def user_stream(self, *, account_id: str):
        return UserStreamStatusSchema(state="CONNECTED")

    def audit(self, *, account_id: str, limit: int):
        return [
            OperationAuditSchema(
                audit_id="a1",
                actor="system",
                action="READ",
                outcome="OK",
                occurred_at=datetime.now(UTC),
            )
        ][:limit]


def principal() -> HedgePrincipal:
    return HedgePrincipal("tester", HedgeRole.VIEWER)


def test_default_router_is_read_only_and_exposes_dual_legs() -> None:
    app = FastAPI()
    router = create_hedge_readonly_router(Query(), principal_dependency=principal)
    app.include_router(router)
    assert all(route.methods <= {"GET", "HEAD"} for route in router.routes)

    response = TestClient(app).get("/hedge/positions/ETHUSDT")
    assert response.status_code == 200
    body = response.json()
    assert {leg["position_side"] for leg in body["legs"]} == {"LONG", "SHORT"}
    assert body["legs"][0]["quantity"] == "1"


def test_optional_write_routes_are_still_disabled() -> None:
    app = FastAPI()
    app.include_router(create_disabled_hedge_write_router())
    response = TestClient(app).post("/hedge/orders", json={})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "HEDGE_WRITE_DISABLED"
