"""Structured JSON logging with recursive and inline secret redaction."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Mapping
from uuid import UUID

_REDACT_KEYS = {
    "apikey",
    "apisecret",
    "clientsecret",
    "privatesecret",
    "secret",
    "token",
    "accesstoken",
    "refreshtoken",
    "authorization",
    "jwt",
    "wstoken",
    "password",
    "passphrase",
    "cookie",
    "session",
}
_INLINE_AUTH = re.compile(
    r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?([\"']?)[^\s,;\"']+\2"
)
_INLINE_BEARER = re.compile(r"(?i)\bbearer\s+[^\s,;\"']+")
_INLINE_KEY_VALUE = re.compile(
    r"(?i)\b(api[_-]?key|api[_-]?secret|client[_-]?secret|private[_-]?secret|"
    r"access[_-]?token|refresh[_-]?token|ws[_-]?token|password|passphrase|jwt|"
    r"cookie|session)\s*[:=]\s*([\"']?)[^\s,;\"']+\2"
)


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _secret_key(value: object) -> bool:
    normalized = _normalized_key(value)
    return (
        normalized in _REDACT_KEYS
        or normalized.endswith(("token", "secret"))
        or normalized.startswith(("session", "cookie"))
    )


def _redact_text(value: str) -> str:
    result = _INLINE_AUTH.sub(r"\1***", value)
    result = _INLINE_BEARER.sub("Bearer ***", result)
    return _INLINE_KEY_VALUE.sub(lambda match: f"{match.group(1)}=***", result)


def _redact(
    value: Any,
    *,
    depth: int,
    budget: list[int],
    active: set[int],
) -> Any:
    budget[0] -= 1
    if budget[0] < 0:
        raise ValueError("structured log contains too many values")
    if depth > 16:
        raise ValueError("structured log nesting is too deep")
    if isinstance(value, Mapping):
        if len(value) > 1_000:
            raise ValueError("structured log mapping is too large")
        identity = id(value)
        if identity in active:
            raise ValueError("structured log contains a reference cycle")
        active.add(identity)
        try:
            result: dict[str, Any] = {}
            for raw_key, item in value.items():
                key = str(raw_key).strip()
                if (
                    not key
                    or len(key) > 256
                    or any(ord(ch) < 32 or ord(ch) == 127 for ch in key)
                ):
                    raise ValueError("structured log key is invalid")
                if key in result:
                    raise ValueError("structured log keys collide after conversion")
                result[key] = (
                    "***"
                    if _secret_key(raw_key)
                    else _redact(
                        item,
                        depth=depth + 1,
                        budget=budget,
                        active=active,
                    )
                )
            return result
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple, set, frozenset)):
        if len(value) > 1_000:
            raise ValueError("structured log sequence is too large")
        identity = id(value)
        if identity in active:
            raise ValueError("structured log contains a reference cycle")
        active.add(identity)
        try:
            items = [
                _redact(
                    item,
                    depth=depth + 1,
                    budget=budget,
                    active=active,
                )
                for item in value
            ]
            if isinstance(value, (set, frozenset)):
                items.sort(key=lambda item: f"{type(item).__name__}:{item!r}")
            return items
        finally:
            active.remove(identity)
    if isinstance(value, str):
        if len(value) > 65_536:
            raise ValueError("structured log string is too large")
        return _redact_text(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite Decimal cannot be logged")
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("naive datetime cannot be logged")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return _redact(
            value.value,
            depth=depth + 1,
            budget=budget,
            active=active,
        )
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float cannot be logged")
        return value
    raise TypeError(
        f"structured log contains unsupported type: {type(value).__name__}"
    )


def redact(value: Any) -> Any:
    return _redact(value, depth=0, budget=[10_000], active=set())



class HedgeJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _redact_text(record.getMessage()),
        }
        structured = getattr(record, "hedge", None)
        if structured is not None:
            payload["hedge"] = redact(structured)
        if record.exc_info:
            payload["exception"] = _redact_text(
                self.formatException(record.exc_info)
            )
        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )


def get_hedge_logger(name: str = "freqtrade.hedge") -> logging.Logger:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("logger name is required")
    return logging.getLogger(name.strip())


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    **fields: Any,
) -> None:
    if not isinstance(logger, logging.Logger):
        raise TypeError("logger must be a logging.Logger")
    logger.log(level, _redact_text(message), extra={"hedge": redact(fields)})
