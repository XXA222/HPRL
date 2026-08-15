"""Composition helper for the production control plane."""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Mapping

from freqtrade.exceptions import OperationalException
from freqtrade.hedge.control.config import hedge_control_plane_config_from_mapping
from freqtrade.hedge.control.service import HedgeControlService
from freqtrade.hedge.control.store import SqlControlOperationStore
from freqtrade.hedge.execution.action_group import ActionGroupExecutor
from freqtrade.hedge.control.auth import ConfirmationService
from freqtrade.hedge.integration.paper_control import PaperAccountViewProvider
from freqtrade.hedge.integration.production_main_loop import HedgeExecutionMode


def build_hedge_control_service(
    *,
    config: Mapping[str, Any],
    production_assembly: object | None,
    readonly_coordinator: object | None,
    paper_application: object | None = None,
    persistence_service: object,
) -> HedgeControlService | None:
    control_config = hedge_control_plane_config_from_mapping(config)
    if not control_config.enabled:
        return None
    if production_assembly is None:
        raise OperationalException(
            "hedge.control_plane requires hedge.main_loop.enabled=true"
        )
    api_config = config.get("api_server", {})
    if not isinstance(api_config, Mapping):
        raise OperationalException("api_server must be a JSON object")
    jwt_secret = str(api_config.get("jwt_secret_key", "")).strip()
    if len(jwt_secret) < 16:
        raise OperationalException(
            "hedge.control_plane requires api_server.jwt_secret_key"
        )
    loop = getattr(production_assembly, "loop", None)
    runtime = getattr(production_assembly, "execution_runtime", None)
    if loop is None or runtime is None:
        raise OperationalException("hedge control-plane dependencies are incomplete")
    if loop.mode is HedgeExecutionMode.HEDGE_SIMULATED:
        if paper_application is None:
            raise OperationalException(
                "simulated hedge control-plane requires the durable Paper application"
            )
        account_view = PaperAccountViewProvider(paper_application)
        if getattr(paper_application, "execution", None) is not runtime:
            raise OperationalException(
                "simulated hedge control-plane must share the Paper execution runtime"
            )
    else:
        if readonly_coordinator is None:
            raise OperationalException(
                "production hedge control-plane requires the Binance read-only coordinator"
            )
        account_view = getattr(
            getattr(readonly_coordinator, "readonly_runtime", None),
            "account_view",
            None,
        )
        if not callable(account_view):
            raise OperationalException("readonly account-view provider is unavailable")
    session_factory = getattr(persistence_service, "session_factory", None)
    if not callable(session_factory):
        raise OperationalException("hedge control-plane requires SQL persistence")
    secret = sha256(
        ("freqtrade-hedge-control-v1\x1f" + loop.account_id + "\x1f" + jwt_secret).encode(
            "utf-8"
        )
    ).digest()
    service = HedgeControlService(
        loop=loop,
        operation_store=SqlControlOperationStore(session_factory),
        confirmation_service=ConfirmationService(
            secret,
            ttl_seconds=control_config.confirmation_ttl_seconds,
            max_pending=control_config.max_pending_confirmations,
        ),
        account_view_provider=account_view,
        audit_recorder=getattr(persistence_service, "record_audit_event", None),
        exchange_write_surface=str(
            getattr(production_assembly, "exchange_write_surface", "NONE")
        ),
        action_group_executor=ActionGroupExecutor(
            loop.engine,
            getattr(runtime, "action_groups", None),
        ),
    )
    service.restore_state()
    return service
