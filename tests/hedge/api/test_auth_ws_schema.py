from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from freqtrade.rpc.api_server.hedge_auth import (
    ConfirmationService,
    HedgePrincipal,
    HedgeRole,
    require_role,
)
from freqtrade.rpc.api_server.hedge_schemas import (
    HedgeReadonlyCommandSchema,
    HedgeWsEventSchema,
)


def test_confirmation_is_one_time_and_scoped() -> None:
    service = ConfirmationService(b"x" * 32)
    token = service.issue(subject="alice", action="HALT")
    assert not service.consume(token=token, subject="alice", action="RESUME")

    token = service.issue(subject="alice", action="HALT")
    assert service.consume(token=token, subject="alice", action="HALT")
    assert not service.consume(token=token, subject="alice", action="HALT")


def test_rbac_rejects_viewer_for_admin_route() -> None:
    def viewer() -> HedgePrincipal:
        return HedgePrincipal("viewer", HedgeRole.VIEWER)

    app = FastAPI()
    admin_dep = require_role(HedgeRole.ADMIN, viewer)

    @app.get("/admin")
    def admin(_: HedgePrincipal = __import__("fastapi").Depends(admin_dep)):
        return {"ok": True}

    assert TestClient(app).get("/admin").status_code == 403


def test_ws_schema_is_versioned_and_strict() -> None:
    model = HedgeWsEventSchema(
        event_id="e1",
        event_type="LOCK",
        occurred_at=datetime.now(UTC),
        account_id="main",
        payload={"locked": True},
    )
    assert model.schema_version == 1


def test_qq_wechat_contract_is_read_only_only() -> None:
    allowed = {
        "hedge.positions",
        "hedge.risk",
        "hedge.reconciliation",
        "hedge.readiness",
        "hedge.user_stream",
        "hedge.orders",
        "hedge.order",
        "hedge.action_group",
        "hedge.pair_summary",
        "hedge.events",
    }
    model = HedgeReadonlyCommandSchema(
        source="QQ",
        command="hedge.positions",
        request_id="req-1",
        symbol="ETHUSDT",
    )
    assert model.command in allowed
    schema = HedgeReadonlyCommandSchema.model_json_schema()
    command_values = schema["properties"]["command"]["enum"]
    assert set(command_values) == allowed
    assert not any(
        action in value
        for value in command_values
        for action in ("submit", "cancel", "replace", "halt", "resume")
    )


def test_wrong_confirmation_scope_does_not_consume_valid_token() -> None:
    service = ConfirmationService(b"x" * 32)
    token = service.issue(subject="alice", action="HALT")
    assert not service.consume(token=token, subject="alice", action="RESUME")
    assert service.consume(token=token, subject="alice", action="HALT")


def test_confirmation_rejects_invalid_ttl_and_empty_scope() -> None:
    import pytest

    with pytest.raises(ValueError, match="positive"):
        ConfirmationService(b"x" * 32, ttl_seconds=0)
    service = ConfirmationService(b"x" * 32)
    with pytest.raises(ValueError, match="required"):
        service.issue(subject="", action="HALT")


def test_confirmation_binds_account_symbol_payload_and_idempotency() -> None:
    service = ConfirmationService(b"x" * 32)
    token = service.issue(
        subject="alice",
        action="REDUCE_ONLY",
        account_id="main",
        symbol="ETHUSDT",
        payload_hash="sha256:abc",
        idempotency_key="req-7",
    )
    assert not service.consume(
        token=token,
        subject="alice",
        action="REDUCE_ONLY",
        account_id="main",
        symbol="BTCUSDT",
        payload_hash="sha256:abc",
        idempotency_key="req-7",
    )
    assert service.consume(
        token=token,
        subject="alice",
        action="REDUCE_ONLY",
        account_id="main",
        symbol="ETHUSDT",
        payload_hash="sha256:abc",
        idempotency_key="req-7",
    )


def test_risk_manager_role_is_between_operator_and_admin() -> None:
    assert HedgeRole.VIEWER < HedgeRole.OPERATOR < HedgeRole.RISK_MANAGER < HedgeRole.ADMIN
