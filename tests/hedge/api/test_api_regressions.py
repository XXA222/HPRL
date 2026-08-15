from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from freqtrade.hedge.telemetry.events import HedgeEventType, HedgeTelemetryEvent
from freqtrade.rpc.api_server.hedge_auth import (
    ConfirmationService,
    HedgePrincipal,
    HedgeRole,
    require_role,
)
from freqtrade.rpc.api_server.hedge_readonly import (
    HedgeReadonlyCommandDispatcher,
    create_hedge_readonly_router,
)
from freqtrade.rpc.api_server.hedge_schemas import (
    DualLegPositionsResponse,
    HedgeReadonlyCommandResponse,
    HedgeReadonlyCommandSchema,
    HedgeWsEventSchema,
    LegPositionSchema,
    OperationAuditSchema,
    ReadinessStatusSchema,
    ReconciliationStatusSchema,
    RiskStatusSchema,
    UserStreamStatusSchema,
)
from freqtrade.rpc.api_server.hedge_ws import (
    HedgeEventHub,
    create_hedge_ws_router,
)


class QueryPort:
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
                    entry_price=Decimal("3000"),
                ),
            ),
            as_of=datetime.now(UTC),
        )

    def risk(self, *, account_id: str):
        return RiskStatusSchema(
            account_id=account_id,
            equity=Decimal("1000"),
            gross_notional=Decimal("100"),
            gross_exposure_ratio=Decimal("0.1"),
            margin_utilization=Decimal("0.05"),
            liquidation_buffer_ratio=Decimal("0.9"),
        )

    def reconciliation(self, *, account_id: str):
        return ReconciliationStatusSchema(status="HEALTHY")

    def readiness(self, *, account_id: str):
        return ReadinessStatusSchema(
            ready=True,
            kill_switch="RUNNING",
            checks={"rest": True},
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


def viewer() -> HedgePrincipal:
    return HedgePrincipal("viewer", HedgeRole.VIEWER)


def test_position_schema_requires_aware_datetime() -> None:
    with pytest.raises(ValidationError):
        DualLegPositionsResponse(
            account_id="main",
            symbol="ETHUSDT",
            legs=(),
            as_of=datetime.now(),
        )


@pytest.mark.parametrize(
    ("quantity", "entry_price"),
    [(Decimal("0"), Decimal("1")), (Decimal("1"), Decimal("0"))],
)
def test_position_schema_rejects_inconsistent_entry_price(
    quantity: Decimal,
    entry_price: Decimal,
) -> None:
    with pytest.raises(ValidationError):
        LegPositionSchema(
            account_id="main",
            symbol="ETHUSDT",
            position_side="LONG",
            quantity=quantity,
            entry_price=entry_price,
        )


def test_schema_normalizes_settled_symbol_and_rejects_mismatch() -> None:
    leg = LegPositionSchema(
        account_id="main",
        symbol="eth/usdt:usdt",
        position_side="LONG",
        quantity=Decimal("1"),
        entry_price=Decimal("3000"),
    )
    assert leg.symbol == "ETHUSDT"
    with pytest.raises(ValidationError):
        LegPositionSchema(
            account_id="main",
            symbol="ETH/USDT:USDC",
            position_side="LONG",
            quantity=Decimal("1"),
            entry_price=Decimal("3000"),
        )


def test_readiness_cannot_claim_write_enabled() -> None:
    with pytest.raises(ValidationError, match="read-only"):
        ReadinessStatusSchema(
            ready=True,
            read_only=False,
            live_trading_enabled=True,
            kill_switch="RUNNING",
        )


def test_principal_rejects_control_characters_and_invalid_role() -> None:
    with pytest.raises(ValueError):
        HedgePrincipal("alice\nadmin", HedgeRole.ADMIN)
    with pytest.raises(ValueError):
        HedgePrincipal("alice", 999)


def test_confirmation_reissues_by_scope_and_revokes_previous() -> None:
    service = ConfirmationService(b"x" * 32)
    first = service.issue(subject="alice", action="HALT")
    second = service.issue(subject="alice", action="HALT")
    assert first != second
    assert not service.consume(token=first, subject="alice", action="HALT")
    assert service.consume(token=second, subject="alice", action="HALT")


def test_confirmation_capacity_is_bounded() -> None:
    service = ConfirmationService(b"x" * 32, max_pending=1)
    service.issue(subject="alice", action="HALT")
    with pytest.raises(RuntimeError, match="capacity"):
        service.issue(subject="bob", action="HALT")


def test_require_role_fails_closed_for_wrong_dependency_type() -> None:
    def broken():
        return {"subject": "alice", "role": "ADMIN"}

    app = FastAPI()
    dependency = require_role(HedgeRole.VIEWER, broken)

    @app.get("/protected")
    def protected(_=pytest.importorskip("fastapi").Depends(dependency)):
        return {"ok": True}

    response = TestClient(app).get("/protected")
    assert response.status_code == 401


def test_readonly_router_fails_closed_for_invalid_principal() -> None:
    app = FastAPI()
    app.include_router(
        create_hedge_readonly_router(
            QueryPort(),
            principal_dependency=lambda: "not-a-principal",
        )
    )
    response = TestClient(app).get("/hedge/risk")
    assert response.status_code == 401


def test_readonly_audit_limit_is_validated_not_silently_clamped() -> None:
    app = FastAPI()
    app.include_router(
        create_hedge_readonly_router(QueryPort(), principal_dependency=viewer)
    )
    response = TestClient(app).get("/hedge/audit?limit=501")
    assert response.status_code == 422


def test_readonly_router_rejects_control_character_account() -> None:
    app = FastAPI()
    app.include_router(
        create_hedge_readonly_router(QueryPort(), principal_dependency=viewer)
    )
    response = TestClient(app).get("/hedge/risk?account_id=bad%0Aactor")
    assert response.status_code == 422


def test_qq_wechat_dispatcher_is_read_only_and_returns_json_safe_data() -> None:
    dispatcher = HedgeReadonlyCommandDispatcher(QueryPort())
    command = HedgeReadonlyCommandSchema(
        source="QQ",
        command="hedge.positions",
        request_id="request-1",
        account_id="main",
        symbol="ETH/USDT:USDT",
    )
    response = dispatcher.dispatch(command)
    assert response.ok
    assert response.command == "hedge.positions"
    assert response.data["symbol"] == "ETHUSDT"
    assert response.data["legs"][0]["quantity"] == "1"


@pytest.mark.parametrize(
    "command",
    [
        "hedge.risk",
        "hedge.reconciliation",
        "hedge.readiness",
        "hedge.user_stream",
    ],
)
def test_dispatcher_supports_only_enumerated_read_commands(command: str) -> None:
    dispatcher = HedgeReadonlyCommandDispatcher(QueryPort())
    request = HedgeReadonlyCommandSchema(
        source="WECHAT",
        command=command,
        request_id=f"request-{command}",
        account_id="main",
    )
    assert dispatcher.dispatch(request).ok


def test_async_websocket_validator_is_awaited() -> None:
    async def validator(token):
        await asyncio.sleep(0)
        return token == "good"

    app = FastAPI()
    hub = HedgeEventHub()
    app.include_router(create_hedge_ws_router(hub, token_validator=validator))
    with TestClient(app).websocket_connect(
        "/hedge/ws",
        headers={"authorization": "Bearer good"},
    ) as websocket:
        asyncio.run(
            hub.publish(
                HedgeTelemetryEvent(HedgeEventType.ORDER, {"status": "ACK"})
            )
        )
        event = websocket.receive_json()
        assert event["event_type"] == "ORDER"


def test_non_boolean_websocket_validator_result_fails_closed() -> None:
    app = FastAPI()
    app.include_router(
        create_hedge_ws_router(
            HedgeEventHub(),
            token_validator=lambda token: "yes",
        )
    )
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with TestClient(app).websocket_connect(
            "/hedge/ws",
            headers={"authorization": "Bearer token"},
        ):
            pass
    assert exc_info.value.code == 1008


def test_websocket_rejects_non_bearer_authorization() -> None:
    app = FastAPI()
    app.include_router(
        create_hedge_ws_router(
            HedgeEventHub(),
            token_validator=lambda token: token == "token",
        )
    )
    with pytest.raises(WebSocketDisconnect):
        with TestClient(app).websocket_connect(
            "/hedge/ws",
            headers={"authorization": "Basic token"},
        ):
            pass


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf"), float("nan")])
def test_websocket_auth_timeout_must_be_positive_finite(timeout) -> None:
    with pytest.raises(ValueError):
        create_hedge_ws_router(
            HedgeEventHub(),
            token_validator=lambda token: True,
            auth_timeout_seconds=timeout,
        )


def test_websocket_subscription_is_removed_on_disconnect() -> None:
    hub = HedgeEventHub()
    app = FastAPI()
    app.include_router(
        create_hedge_ws_router(hub, token_validator=lambda token: True)
    )
    with TestClient(app).websocket_connect("/hedge/ws"):
        pass
    assert not hub._subscribers


def test_slow_sync_websocket_validator_is_bounded_by_timeout() -> None:
    import time

    def slow_validator(token):
        time.sleep(0.1)
        return True

    app = FastAPI()
    app.include_router(
        create_hedge_ws_router(
            HedgeEventHub(),
            token_validator=slow_validator,
            auth_timeout_seconds=0.01,
        )
    )
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with TestClient(app).websocket_connect("/hedge/ws"):
            pass
    assert exc_info.value.code == 1008



def test_audit_details_reject_nested_control_characters() -> None:
    with pytest.raises(ValidationError, match="invalid string"):
        OperationAuditSchema(
            audit_id="audit-1",
            actor="system",
            action="READ",
            outcome="OK",
            occurred_at=datetime.now(UTC),
            details={"nested": ["safe", "bad\nvalue"]},
        )


def test_ws_payload_rejects_nonfinite_numbers_and_deep_nesting() -> None:
    with pytest.raises(ValidationError, match="non-finite"):
        HedgeWsEventSchema(
            event_id="event-1",
            event_type="ORDER",
            occurred_at=datetime.now(UTC),
            account_id="main",
            payload={"ratio": float("inf")},
        )

    nested: object = "leaf"
    for _ in range(14):
        nested = {"level": nested}
    with pytest.raises(ValidationError, match="nesting is too deep"):
        HedgeWsEventSchema(
            event_id="event-2",
            event_type="ORDER",
            occurred_at=datetime.now(UTC),
            account_id="main",
            payload={"nested": nested},
        )


def test_ws_correlation_id_rejects_controls() -> None:
    with pytest.raises(ValidationError):
        HedgeWsEventSchema(
            event_id="event-3",
            event_type="AUDIT",
            occurred_at=datetime.now(UTC),
            account_id="main",
            correlation_id="corr\nforged",
            payload={},
        )


@pytest.mark.parametrize("bad_value", [1, 0, "true", None])
def test_readiness_checks_require_exact_booleans(bad_value) -> None:
    with pytest.raises(ValidationError, match="booleans"):
        ReadinessStatusSchema(
            ready=True,
            kill_switch="RUNNING",
            checks={"rest": bad_value},
        )


def test_readiness_names_and_locks_cannot_collide_after_normalization() -> None:
    with pytest.raises(ValidationError, match="collide"):
        ReadinessStatusSchema(
            ready=True,
            kill_switch="RUNNING",
            checks={"rest": True, " rest ": False},
        )
    with pytest.raises(ValidationError, match="unique"):
        ReadinessStatusSchema(
            ready=True,
            kill_switch="RUNNING",
            unknown_leg_locks=("main:ETHUSDT:LONG", "main:ETHUSDT:LONG"),
        )


def test_readonly_command_response_cannot_name_a_write_command() -> None:
    with pytest.raises(ValidationError):
        HedgeReadonlyCommandResponse(
            request_id="request-1",
            ok=False,
            command="hedge.orders.create",
            message="disabled",
            data={},
        )


def test_json_mapping_keys_cannot_collide_after_strip() -> None:
    with pytest.raises(ValidationError, match="collide"):
        OperationAuditSchema(
            audit_id="audit-2",
            actor="system",
            action="READ",
            outcome="OK",
            occurred_at=datetime.now(UTC),
            details={"key": 1, " key ": 2},
        )


def test_websocket_rejects_oversized_bearer_token_before_validator() -> None:
    calls: list[str | None] = []

    def validator(token):
        calls.append(token)
        return True

    app = FastAPI()
    app.include_router(
        create_hedge_ws_router(HedgeEventHub(), token_validator=validator)
    )
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with TestClient(app).websocket_connect(
            "/hedge/ws",
            headers={"authorization": f"Bearer {'x' * 4097}"},
        ):
            pass
    assert exc_info.value.code == 1008
    assert calls == []


def test_malformed_authorization_is_rejected_even_if_anonymous_is_allowed() -> None:
    calls: list[str | None] = []

    def validator(token):
        calls.append(token)
        return True

    app = FastAPI()
    app.include_router(
        create_hedge_ws_router(HedgeEventHub(), token_validator=validator)
    )
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with TestClient(app).websocket_connect(
            "/hedge/ws",
            headers={"authorization": "Basic malformed"},
        ):
            pass
    assert exc_info.value.code == 1008
    assert calls == []


def test_confirmation_rejects_oversized_or_control_token_without_consuming() -> None:
    service = ConfirmationService(b"x" * 32)
    valid = service.issue(subject="alice", action="HALT")
    assert not service.consume(
        token="x" * 513,
        subject="alice",
        action="HALT",
    )
    assert not service.consume(
        token="bad\nvalue",
        subject="alice",
        action="HALT",
    )
    assert service.consume(token=valid, subject="alice", action="HALT")
