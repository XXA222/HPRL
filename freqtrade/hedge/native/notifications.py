"""Unified Hedge event formatting for Telegram, Webhook, Discord, QQ and WeChat."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from typing import Any, Mapping

from .models import HedgeEvent


@dataclass(frozen=True, slots=True)
class NotificationPayload:
    channel: str
    title: str
    text: str
    structured: Mapping[str, Any]
    deduplication_key: str
    severity: str


class HedgeNotificationFormatter:
    IMPORTANT_TYPES = {
        "ORDER_REJECTED", "ORDER_UNKNOWN", "KILL_SWITCH", "READINESS_BLOCKED",
        "RECONCILIATION_DRIFT", "USER_STREAM_STALE", "LIQUIDATION_RISK",
    }

    @staticmethod
    def _title(event: HedgeEvent) -> str:
        side = "" if event.side is None else f" {event.side.value}"
        return f"Hedge {event.event_type}{side} {event.pair}".strip()

    @staticmethod
    def _details(event: HedgeEvent) -> str:
        preferred = (
            "action", "bucket", "quantity", "price", "filled_quantity", "pnl",
            "fee", "funding", "reason", "state", "risk_code", "correlation_id",
        )
        values: list[str] = []
        for key in preferred:
            value = event.payload.get(key)
            if value not in (None, ""):
                values.append(f"{key}={value}")
        if not values:
            values = [f"{key}={value}" for key, value in sorted(event.payload.items())[:8]]
        return ", ".join(values)

    def format(self, event: HedgeEvent, *, channel: str) -> NotificationPayload:
        channel = channel.lower().strip()
        if channel not in {"telegram", "webhook", "discord", "qq", "wechat", "log"}:
            raise ValueError(f"unsupported Hedge notification channel: {channel}")
        title = self._title(event)
        text = f"{title}\n{self._details(event)}\n{event.timestamp.isoformat()}"
        structured = {
            "schema": "hedge-notification-v1",
            "event_type": event.event_type,
            "pair": event.pair,
            "position_side": None if event.side is None else event.side.value,
            "severity": event.severity,
            "timestamp": event.timestamp.isoformat(),
            "correlation_id": event.correlation_id,
            "payload": dict(event.payload),
        }
        canonical = dumps(structured, sort_keys=True, separators=(",", ":"), default=str)
        return NotificationPayload(
            channel,
            title,
            text,
            structured,
            sha256(canonical.encode()).hexdigest(),
            "ERROR" if event.event_type in self.IMPORTANT_TYPES else event.severity,
        )

    def rpc_message(self, event: HedgeEvent) -> dict[str, Any]:
        """Create a generic RPC payload without importing Freqtrade RPC enums."""
        payload = self.format(event, channel="log")
        return {
            "type": "hedge_event",
            "status": payload.text,
            "hedge": dict(payload.structured),
            "deduplication_key": payload.deduplication_key,
        }


class HedgeRpcEventBridge:
    """Best-effort notifier with explicit failure accounting and no execution coupling."""

    def __init__(self, rpc_manager: Any, *, formatter: HedgeNotificationFormatter | None = None) -> None:
        self.rpc_manager = rpc_manager
        self.formatter = formatter or HedgeNotificationFormatter()
        self.sent = 0
        self.failed = 0
        self.last_error: str | None = None

    def publish(self, event: HedgeEvent) -> bool:
        sender = getattr(self.rpc_manager, "send_msg", None)
        if not callable(sender):
            self.failed += 1
            self.last_error = "RPC_MANAGER_SEND_MSG_UNAVAILABLE"
            return False
        try:
            sender(self.formatter.rpc_message(event))
        except Exception as exc:
            self.failed += 1
            self.last_error = f"{type(exc).__name__}:{exc}"
            return False
        self.sent += 1
        self.last_error = None
        return True

    def status(self) -> dict[str, Any]:
        return {"sent": self.sent, "failed": self.failed, "last_error": self.last_error}


def hedge_event_from_outbox(event: Any) -> HedgeEvent:
    """Translate the execution outbox contract into the unified notification event."""
    payload = dict(getattr(event, "payload", {}) or {})
    side = payload.get("position_side") or payload.get("side")
    from .models import HedgeSide

    parsed_side = None
    if side not in (None, ""):
        try:
            parsed_side = HedgeSide.parse(side)
        except ValueError:
            parsed_side = None
    return HedgeEvent(
        event_type=str(getattr(event, "event_type", "HEDGE_EVENT")),
        pair=str(payload.get("symbol", payload.get("pair", ""))),
        side=parsed_side,
        payload=payload,
        timestamp=getattr(event, "occurred_at", None),
        severity=str(payload.get("severity", "INFO")),
        correlation_id=str(getattr(event, "correlation_id", "")),
    )
