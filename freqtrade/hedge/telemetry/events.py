"""Versioned, immutable telemetry and WebSocket event envelope."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum, StrEnum
from types import MappingProxyType
from typing import Any, Mapping
from uuid import UUID, uuid4

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SYMBOL = re.compile(r"^[A-Za-z0-9/_:-]+$")


class HedgeEventType(StrEnum):
    INTENT = "INTENT"
    ORDER = "ORDER"
    FILL = "FILL"
    DRIFT = "DRIFT"
    HALT = "HALT"
    RECONNECT = "RECONNECT"
    LOCK = "LOCK"
    RECONCILIATION = "RECONCILIATION"
    READINESS = "READINESS"
    USER_STREAM = "USER_STREAM"
    AUDIT = "AUDIT"


def _text(
    value: object,
    *,
    field_name: str,
    limit: int,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        if optional:
            return None
        raise ValueError(f"{field_name} is required")
    if len(result) > limit or _CONTROL.search(result):
        raise ValueError(f"{field_name} is invalid")
    return result


def _freeze_json(
    value: object,
    *,
    depth: int = 0,
    budget: list[int] | None = None,
) -> object:
    if budget is None:
        budget = [10_000]
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("telemetry payload contains too many values")
    if depth > 16:
        raise ValueError("telemetry payload nesting is too deep")
    if isinstance(value, Enum):
        return _freeze_json(value.value, depth=depth + 1, budget=budget)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if len(value) > 65536 or _CONTROL.search(value):
            raise ValueError("telemetry payload string is invalid")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("telemetry payload float must be finite")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("telemetry payload decimal must be finite")
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("telemetry datetime must be timezone-aware")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Mapping):
        if len(value) > 1000:
            raise ValueError("telemetry mapping is too large")
        converted: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key).strip()
            if not key or len(key) > 256 or _CONTROL.search(key):
                raise ValueError("telemetry payload key is invalid")
            if key in converted:
                raise ValueError("telemetry payload keys collide after conversion")
            converted[key] = _freeze_json(item, depth=depth + 1, budget=budget)
        return MappingProxyType(converted)
    if isinstance(value, (list, tuple)):
        if len(value) > 1000:
            raise ValueError("telemetry sequence is too large")
        return tuple(_freeze_json(item, depth=depth + 1, budget=budget) for item in value)
    if isinstance(value, (set, frozenset)):
        if len(value) > 1000:
            raise ValueError("telemetry set is too large")
        frozen = [_freeze_json(item, depth=depth + 1, budget=budget) for item in value]
        return tuple(sorted(frozen, key=lambda item: f"{type(item).__name__}:{item!r}"))
    raise TypeError(
        "telemetry payload contains non-serializable type: "
        f"{type(value).__name__}"
    )


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_symbol(value: object) -> str | None:
    raw = _text(value, field_name="symbol", limit=64, optional=True)
    if raw is None:
        return None
    raw = raw.upper()
    if not _SYMBOL.fullmatch(raw):
        raise ValueError("symbol contains unsupported characters")
    parts = raw.split(":")
    if len(parts) > 2:
        raise ValueError("symbol contains multiple settlement suffixes")
    normalized = re.sub(r"[/_-]", "", parts[0])
    if not normalized or not normalized.isascii() or not normalized.isalnum():
        raise ValueError("symbol must contain ASCII alphanumeric characters")
    if len(parts) == 2 and not normalized.endswith(parts[1]):
        raise ValueError("settlement suffix must match quote asset")
    return normalized


@dataclass(frozen=True, slots=True)
class HedgeTelemetryEvent:
    event_type: HedgeEventType
    payload: Mapping[str, Any]
    account_id: str = "default"
    symbol: str | None = None
    correlation_id: str | None = None
    schema_version: int = 1
    payload_version: int = 1
    event_id: UUID = field(default_factory=uuid4)
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    exchange_time: datetime | None = None
    observed_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        try:
            event_type = (
                self.event_type
                if isinstance(self.event_type, HedgeEventType)
                else HedgeEventType(self.event_type)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("event_type is invalid") from exc
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
        ):
            raise TypeError("schema_version must be an integer")
        if not 1 <= self.schema_version <= 2_147_483_647:
            raise ValueError("schema_version is outside the supported range")
        if (
            not isinstance(self.payload_version, int)
            or isinstance(self.payload_version, bool)
            or not 1 <= self.payload_version <= 2_147_483_647
        ):
            raise ValueError("payload_version is outside the supported range")
        account_id = _text(
            self.account_id,
            field_name="account_id",
            limit=128,
        )
        correlation_id = _text(
            self.correlation_id,
            field_name="correlation_id",
            limit=256,
            optional=True,
        )
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        payload = _freeze_json(self.payload)
        if not isinstance(payload, Mapping):  # pragma: no cover
            raise TypeError("payload must be a mapping")
        event_id = self.event_id
        if not isinstance(event_id, UUID):
            try:
                event_id = UUID(str(event_id))
            except (TypeError, ValueError) as exc:
                raise ValueError("event_id must be a UUID") from exc
        if not isinstance(self.occurred_at, datetime):
            raise TypeError("occurred_at must be a datetime")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        if self.exchange_time is not None:
            if not isinstance(self.exchange_time, datetime):
                raise TypeError("exchange_time must be a datetime")
            if self.exchange_time.tzinfo is None or self.exchange_time.utcoffset() is None:
                raise ValueError("exchange_time must be timezone-aware")
        if not isinstance(self.observed_time, datetime):
            raise TypeError("observed_time must be a datetime")
        if self.observed_time.tzinfo is None or self.observed_time.utcoffset() is None:
            raise ValueError("observed_time must be timezone-aware")
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(self, "symbol", _canonical_symbol(self.symbol))
        object.__setattr__(self, "correlation_id", correlation_id)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "event_id", event_id)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "payload_version": self.payload_version,
            "event_id": str(self.event_id),
            "event_type": self.event_type.value,
            "occurred_at": self.occurred_at.isoformat(),
            "exchange_time": None if self.exchange_time is None else self.exchange_time.isoformat(),
            "observed_time": self.observed_time.isoformat(),
            "account_id": self.account_id,
            "symbol": self.symbol,
            "correlation_id": self.correlation_id,
            "payload": _thaw_json(self.payload),
        }
