"""Frozen persistence contracts for the hedge ledger.

This module deliberately contains no exchange I/O.  It normalizes identities,
order states, event metadata, and stable reason codes shared by persistence,
recovery, risk, and execution adapters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

CONTRACTS_VERSION = "hedge-contracts-v1"
EVENT_VERSION = 1
PAYLOAD_VERSION = 1
SCHEMA_VERSION = "h3-ledger-v2"

ORDER_STATUSES = frozenset(
    {
        "PLANNED",
        "APPROVED",
        "PREPARED",
        "SUBMITTING",
        "ACKNOWLEDGED",
        "PARTIAL",
        "FILLED",
        "CANCELED",
        "REJECTED",
        "UNKNOWN",
        "EXPIRED",
    }
)
TERMINAL_ORDER_STATUSES = frozenset({"FILLED", "CANCELED", "REJECTED", "EXPIRED"})
ORDER_STATUS_ALIASES = {
    "NEW": "ACKNOWLEDGED",
    "SUBMITTED": "ACKNOWLEDGED",
    "PARTIALLY_FILLED": "PARTIAL",
    "CANCELLED": "CANCELED",
    "FAILED": "UNKNOWN",
}

REASON_CODES = frozenset(
    {
        "DUPLICATE_FILL",
        "OUT_OF_ORDER_EVENT",
        "RECONCILIATION_DRIFT",
        "OUTBOX_BACKLOG",
        "UNKNOWN_ORDER",
        "IDEMPOTENCY_CONFLICT",
        "OVER_CLOSE_FILL",
        "PROJECTION_BLOCKED",
        "INVALID_RISK_FACT",
    }
)

_KNOWN_QUOTES = (
    "FDUSD",
    "USDT",
    "USDC",
    "BUSD",
    "TUSD",
    "BTC",
    "ETH",
    "BNB",
)
_SYMBOL_RE = re.compile(r"^[A-Z0-9._-]+$")


def normalize_order_status(value: str) -> str:
    normalized = value.strip().upper()
    normalized = ORDER_STATUS_ALIASES.get(normalized, normalized)
    if normalized not in ORDER_STATUSES:
        raise ValueError(f"Unsupported order status: {value!r}")
    return normalized


def canonical_symbol(value: str, *, settlement: str | None = None) -> str:
    """Normalize venue symbols to ``BASE/QUOTE:SETTLE``.

    The first project stage is Binance USDⓈ-M, but this parser also accepts
    already-canonical CCXT symbols.  Unknown compact symbols fail closed rather
    than silently creating a second position identity.
    """

    raw = value.strip().upper().replace(" ", "")
    if not raw:
        raise ValueError("Symbol must not be empty")
    if "/" in raw:
        base_quote, separator, settle = raw.partition(":")
        base, slash, quote = base_quote.partition("/")
        if not slash or not base or not quote:
            raise ValueError(f"Invalid canonical symbol: {value!r}")
        normalized_settle = settle or settlement or quote
        if not all(_SYMBOL_RE.fullmatch(part) for part in (base, quote, normalized_settle)):
            raise ValueError(f"Invalid symbol component: {value!r}")
        return f"{base}/{quote}:{normalized_settle}"

    compact = raw.replace("-", "").replace("_", "")
    for quote in _KNOWN_QUOTES:
        if compact.endswith(quote) and len(compact) > len(quote):
            base = compact[: -len(quote)]
            settle = settlement or quote
            if _SYMBOL_RE.fullmatch(base) and _SYMBOL_RE.fullmatch(settle):
                return f"{base}/{quote}:{settle}"
    raise ValueError(
        f"Cannot canonicalize compact symbol {value!r}; provide BASE/QUOTE:SETTLE"
    )


def stable_fact_key(kind: str, *parts: Any) -> str:
    material = "|".join((kind, *(str(part) for part in parts)))
    return sha256(material.encode("utf-8")).hexdigest()


def naive_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class EventMetadata:
    correlation_id: str
    causation_id: str | None = None
    payload_version: int = PAYLOAD_VERSION
    event_version: int = EVENT_VERSION
    contracts_version: str = CONTRACTS_VERSION
    schema_version: str = SCHEMA_VERSION
    exchange_time: datetime | None = None
    observed_time: datetime | None = None

    def headers(self) -> dict[str, Any]:
        return {
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "payload_version": self.payload_version,
            "event_version": self.event_version,
            "contracts_version": self.contracts_version,
            "schema_version": self.schema_version,
            "exchange_time": self.exchange_time,
            "observed_time": self.observed_time,
        }
