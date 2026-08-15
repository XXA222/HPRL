"""Optional Hedge REST/WebSocket route registration.

Kept outside :mod:`webserver` so future upstream upgrades need only preserve a
single stable hook before the UI catch-all router.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from ipaddress import ip_address
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException


def register_hedge_api_routes(app: FastAPI, config: dict[str, Any], api_server: object) -> None:
    """Register fail-closed Hedge routes when the feature gate is enabled."""

    if not config.get("hedge_mode_enabled", False):
        return

    api_config = config["api_server"]
    hedge_principal, validate_ws_token = _build_auth_dependencies(api_config)
    rpc, bot, event_hub = _register_core_routes(
        app,
        api_server=api_server,
        hedge_principal=hedge_principal,
        validate_ws_token=validate_ws_token,
    )
    _register_event_publishers(app, rpc=rpc, event_hub=event_hub)
    _register_native_routes(app, bot=bot, hedge_principal=hedge_principal)

    hedge_config = config.get("hedge", {})
    _register_dashboard_routes(
        app,
        config=config,
        api_config=api_config,
        hedge_config=hedge_config,
        hedge_principal=hedge_principal,
    )
    _register_hedge_research_routes(
        app=app,
        config=config,
        api_config=api_config,
        hedge_config=hedge_config,
        hedge_principal=hedge_principal,
    )


def _build_auth_dependencies(
    api_config: dict[str, Any],
) -> tuple[Callable[..., Any], Callable[[str | None], Any]]:
    from freqtrade.rpc.api_server.api_auth import get_user_from_token, http_basic_or_jwt_token
    from freqtrade.rpc.api_server.hedge_auth import HedgePrincipal, HedgeRole
    from freqtrade.rpc.api_server.hedge_ws import HedgeWsPrincipal

    raw_role_map = api_config.get("hedge_roles", {})
    role_map = raw_role_map if isinstance(raw_role_map, dict) else {}
    raw_ws_accounts = api_config.get("hedge_ws_accounts", {})
    ws_accounts = raw_ws_accounts if isinstance(raw_ws_accounts, dict) else {}
    hedge_principal = _build_http_principal_dependency(
        role_map=role_map,
        dependency=http_basic_or_jwt_token,
        principal_type=HedgePrincipal,
        role_type=HedgeRole,
    )
    ws_principal = _build_ws_principal_factory(
        api_config=api_config,
        role_map=role_map,
        ws_accounts=ws_accounts,
        principal_type=HedgeWsPrincipal,
        role_type=HedgeRole,
    )
    validate_ws_token = _build_ws_token_validator(
        api_config=api_config,
        get_user_from_token=get_user_from_token,
        ws_principal=ws_principal,
    )
    return hedge_principal, validate_ws_token


def _build_http_principal_dependency(
    *,
    role_map: dict[str, object],
    dependency: Callable[..., Any],
    principal_type: type[Any],
    role_type: type[Any],
) -> Callable[..., Any]:
    def hedge_principal(username: str = Depends(dependency)) -> Any:
        raw_role = str(role_map.get(username, "VIEWER")).strip().upper()
        try:
            role = role_type[raw_role]
        except KeyError as exc:
            raise HTTPException(status_code=403, detail="invalid configured hedge role") from exc
        return principal_type(subject=username, role=role)

    return hedge_principal


def _accounts_for_subject(
    ws_accounts: dict[str, object],
    subject: str,
) -> frozenset[str] | None:
    raw = ws_accounts.get(subject)
    if raw is None:
        return None
    if not isinstance(raw, list):
        return frozenset()
    return frozenset(str(item).strip() for item in raw if str(item).strip())


def _build_ws_principal_factory(
    *,
    api_config: dict[str, Any],
    role_map: dict[str, object],
    ws_accounts: dict[str, object],
    principal_type: type[Any],
    role_type: type[Any],
) -> Callable[[str], Any | None]:
    def ws_principal(subject: str) -> Any | None:
        raw_role = str(role_map.get(subject, "VIEWER")).strip().upper()
        try:
            role = role_type[raw_role]
        except KeyError:
            return None
        if role < role_type.VIEWER:
            return None
        accounts = _accounts_for_subject(ws_accounts, subject)
        if accounts == frozenset():
            return None
        return principal_type(
            subject=subject,
            role=role,
            account_ids=accounts,
            allow_sensitive=(
                role >= role_type.ADMIN
                and bool(api_config.get("hedge_ws_allow_sensitive_admin", False))
            ),
        )

    return ws_principal


def _build_ws_token_validator(
    *,
    api_config: dict[str, Any],
    get_user_from_token: Callable[..., Any],
    ws_principal: Callable[[str], Any | None],
) -> Callable[[str | None], Any]:
    def validate_ws_token(token: str | None) -> Any:
        if token is None:
            return False
        configured = api_config.get("ws_token")
        if _configured_ws_token_matches(configured, token):
            return ws_principal("ws-service") or False
        try:
            username = get_user_from_token(token, api_config["jwt_secret_key"])
        except HTTPException:
            return False
        return ws_principal(str(username)) or False

    return validate_ws_token


def _configured_ws_token_matches(configured: object, token: str) -> bool:
    if isinstance(configured, str):
        return secrets.compare_digest(configured, token)
    if not isinstance(configured, list):
        return False
    return any(
        isinstance(candidate, str) and secrets.compare_digest(candidate, token)
        for candidate in configured
    )


def _register_core_routes(
    app: FastAPI,
    *,
    api_server: object,
    hedge_principal: Callable[..., Any],
    validate_ws_token: Callable[[str | None], Any],
) -> tuple[object | None, object | None, object]:
    from freqtrade.rpc.api_server.hedge_readonly import (
        create_disabled_hedge_write_router,
        create_hedge_readonly_router,
    )
    from freqtrade.rpc.api_server.hedge_runtime import (
        HedgeExecutionRuntimeQuery,
        HedgeRuntimeQuery,
    )
    from freqtrade.rpc.api_server.hedge_ws import HedgeEventHub, create_hedge_ws_router

    event_hub = HedgeEventHub()
    app.state.hedge_event_hub = event_hub
    app.include_router(
        create_hedge_readonly_router(
            HedgeRuntimeQuery(),
            principal_dependency=hedge_principal,
            execution_query=HedgeExecutionRuntimeQuery(),
        ),
        prefix="/api/v1",
    )

    rpc = getattr(api_server, "_rpc", None)
    bot = getattr(rpc, "_freqtrade", None)
    composition = getattr(bot, "hedge_composition", None)
    control_service = getattr(composition, "control_service", None)
    if control_service is None:
        app.include_router(create_disabled_hedge_write_router(), prefix="/api/v1")
    else:
        from freqtrade.rpc.api_server.hedge_control import create_hedge_control_router

        app.include_router(
            create_hedge_control_router(
                control_service,
                principal_dependency=hedge_principal,
            ),
            prefix="/api/v1",
        )

    app.include_router(
        create_hedge_ws_router(event_hub, token_validator=validate_ws_token),
        prefix="/api/v1",
    )
    return rpc, bot, event_hub


def _register_event_publishers(app: FastAPI, *, rpc: object | None, event_hub: object) -> None:
    application = getattr(getattr(rpc, "_freqtrade", None), "hedge_application", None)
    publisher = getattr(getattr(application, "execution", None), "publisher", None)
    add_callback = getattr(publisher, "add_callback", None)
    set_callback = getattr(publisher, "set_callback", None)
    if not callable(add_callback) and not callable(set_callback):
        return

    from freqtrade.hedge.execution.event_publisher import HedgeEventHubPublisher

    hub_callback = HedgeEventHubPublisher(event_hub).publish
    if callable(add_callback):
        add_callback(hub_callback)
    elif callable(set_callback):
        set_callback(hub_callback)

    if rpc is None or not callable(add_callback):
        return

    from freqtrade.hedge.native.notifications import HedgeRpcEventBridge, hedge_event_from_outbox

    bridge = HedgeRpcEventBridge(rpc)
    app.state.hedge_rpc_event_bridge = bridge

    def publish_rpc_event(outbox_event: object) -> None:
        bridge.publish(hedge_event_from_outbox(outbox_event))

    add_callback(publish_rpc_event)


def _register_native_routes(
    app: FastAPI,
    *,
    bot: object | None,
    hedge_principal: Callable[..., Any],
) -> None:
    from freqtrade.rpc.api_server.hedge_auth import HedgePrincipal

    router = APIRouter(prefix="/hedge/native", tags=["hedge-native"])

    @router.get("/status")
    def hedge_native_status(
        _: HedgePrincipal = Depends(hedge_principal),
    ) -> dict[str, Any]:
        coordinator = getattr(bot, "hedge_native_convergence", None)
        return {
            "enabled": coordinator is not None,
            "convergence": None if coordinator is None else coordinator.status(),
            "rpc_event_bridge": (
                None
                if not hasattr(app.state, "hedge_rpc_event_bridge")
                else app.state.hedge_rpc_event_bridge.status()
            ),
        }

    @router.get("/universe")
    def hedge_native_universe(
        _: HedgePrincipal = Depends(hedge_principal),
    ) -> dict[str, Any]:
        manager = getattr(bot, "hedge_universe_manager", None)
        return {
            "enabled": manager is not None,
            "status": None if manager is None else manager.status(),
        }

    @router.get("/model")
    def hedge_native_model(
        _: HedgePrincipal = Depends(hedge_principal),
    ) -> dict[str, Any]:
        gate = getattr(bot, "hedge_model_gate", None)
        snapshot = None if gate is None else gate.snapshot()
        return {
            "enabled": gate is not None,
            "ready": None if snapshot is None else snapshot.ready,
            "reasons": [] if snapshot is None else list(snapshot.reasons),
            "model_version": "" if snapshot is None else snapshot.model_version,
            "feature_schema": "" if snapshot is None else snapshot.feature_schema,
            "expires_at": (
                None
                if snapshot is None or snapshot.expires_at is None
                else snapshot.expires_at.isoformat()
            ),
        }

    @router.get("/producer")
    def hedge_native_producer(
        _: HedgePrincipal = Depends(hedge_principal),
    ) -> dict[str, Any]:
        gate = getattr(bot, "hedge_producer_gate", None)
        return {
            "enabled": gate is not None,
            "status": None if gate is None else gate.status(),
        }

    app.include_router(router, prefix="/api/v1")


def _register_dashboard_routes(
    app: FastAPI,
    *,
    config: dict[str, Any],
    api_config: dict[str, Any],
    hedge_config: object,
    hedge_principal: Callable[..., Any],
) -> None:
    dashboard = hedge_config.get("dashboard", {}) if isinstance(hedge_config, dict) else {}
    if not isinstance(dashboard, dict) or not bool(dashboard.get("enabled", False)):
        return

    from freqtrade.rpc.api_server.hedge_dashboard import (
        HedgeDashboardQuery,
        create_hedge_dashboard_router,
        create_hedge_dashboard_ui_router,
    )

    _require_loopback_if_local_only(
        api_config,
        local_only=bool(dashboard.get("local_only", True)),
        invalid_message="Hedge dashboard listen address is invalid",
        non_loopback_message="Local Hedge dashboard requires a loopback API listen address",
    )
    account_id = str(hedge_config.get("account_id", "hedge-main"))
    symbol = str(config.get("managed_pair") or "")
    query = HedgeDashboardQuery(
        account_id=account_id,
        symbol=symbol,
        refresh_seconds=int(dashboard.get("refresh_seconds", 5)),
    )
    app.include_router(
        create_hedge_dashboard_router(query=query, principal_dependency=hedge_principal),
        prefix="/api/v1",
    )
    app.include_router(create_hedge_dashboard_ui_router())


def _register_hedge_research_routes(
    *,
    app: FastAPI,
    config: dict[str, Any],
    api_config: dict[str, Any],
    hedge_config: object,
    hedge_principal: Callable[..., Any],
) -> None:
    research = hedge_config.get("research", {}) if isinstance(hedge_config, dict) else {}
    if not isinstance(research, dict) or not bool(research.get("enabled", False)):
        return

    from freqtrade.hedge.research.config import build_research_service
    from freqtrade.rpc.api_server.hedge_research import (
        create_hedge_research_router,
        create_hedge_research_ui_router,
    )

    _require_loopback_if_local_only(
        api_config,
        local_only=bool(research.get("local_only", True)),
        invalid_message="Hedge research listen address is invalid",
        non_loopback_message="Hedge research dashboard requires a loopback API listen address",
    )
    research_service = build_research_service(config)
    app.state.hedge_research_service = research_service

    def stop_research_executor() -> None:
        research_service.stop_orchestration()
        executor = getattr(research_service, "executor", None)
        if executor is not None:
            executor.stop(terminate_running=True)

    app.add_event_handler("shutdown", stop_research_executor)
    app.include_router(
        create_hedge_research_router(
            service=research_service,
            principal_dependency=hedge_principal,
        ),
        prefix="/api/v1",
    )
    app.include_router(create_hedge_research_ui_router())


def _require_loopback_if_local_only(
    api_config: dict[str, Any],
    *,
    local_only: bool,
    invalid_message: str,
    non_loopback_message: str,
) -> None:
    if not local_only:
        return
    listen_ip = str(api_config.get("listen_ip_address", "127.0.0.1"))
    normalized = "127.0.0.1" if listen_ip.lower() == "localhost" else listen_ip
    try:
        is_loopback = ip_address(normalized).is_loopback
    except ValueError as exc:
        raise RuntimeError(invalid_message) from exc
    if not is_loopback:
        raise RuntimeError(non_loopback_message)
