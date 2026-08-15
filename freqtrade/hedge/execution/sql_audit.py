"""Durable execution-service audit adapter."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import uuid4

from freqtrade.persistence.hedge_models import AuditEvent


class SqlExecutionAuditLog:
    """Persist every ExecutionService audit transition immediately.

    The final order/fill/outbox transaction remains authoritative for external facts;
    this adapter preserves the service's fine-grained PREPARED/SUBMITTING/recovery
    diagnostics across process crashes.
    """

    def __init__(
        self,
        session_factory: object,
        *,
        account_id: str,
        exchange: str = "binance",
        actor: str = "hedge-authoritative-execution",
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory
        self._account_id = str(account_id).strip()
        self._exchange = str(exchange).strip().lower()
        self._actor = str(actor).strip()
        if not self._account_id or not self._exchange or not self._actor:
            raise ValueError("SQL execution audit identity is incomplete")

    def emit(self, event: str, payload: Mapping[str, Any]) -> None:
        event_name = str(event).strip().upper()
        if not event_name or len(event_name) > 128:
            raise ValueError("execution audit event is invalid")
        data = dict(payload)
        account_id = str(data.get("account_id", self._account_id)).strip()
        if account_id != self._account_id:
            raise PermissionError("execution audit account mismatch")
        entity_id = str(data.get("client_order_id", "")).strip() or None
        correlation_id = str(
            data.get("action_group_id") or data.get("intent_id") or uuid4().hex
        ).strip()[:128]
        encoded = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        with self._session_factory.begin() as session:  # type: ignore[operator]
            session.add(
                AuditEvent(
                    account_id=account_id,
                    exchange=self._exchange,
                    event_type=event_name,
                    entity_type="EXECUTION_SERVICE",
                    entity_id=entity_id,
                    severity="INFO",
                    reason_code=(
                        None
                        if data.get("reason") in (None, "")
                        else str(data.get("reason"))[:64]
                    ),
                    correlation_id=correlation_id,
                    actor=self._actor,
                    payload_json=encoded,
                    occurred_at=datetime.now(UTC).replace(tzinfo=None),
                )
            )
