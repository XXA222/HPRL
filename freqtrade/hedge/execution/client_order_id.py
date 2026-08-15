"""Deterministic and exchange-safe client order identifiers."""

from __future__ import annotations

import base64
import hashlib
import re

_ALLOWED = re.compile(r"^[A-Za-z0-9_-]{1,36}$")
_ASCII_ALNUM = re.compile(r"^[A-Za-z0-9]+$")
_SYMBOL_ALLOWED = re.compile(r"^[A-Za-z0-9/_:-]+$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_MAX_ATTEMPT = 36**4 - 1


def _clean_text(value: object, *, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{field_name} must not be empty")
    if len(result) > max_length or _CONTROL.search(result):
        raise ValueError(f"{field_name} is invalid")
    return result


def _normalize_symbol(value: object) -> str:
    raw = _clean_text(value, field_name="symbol", max_length=64).upper()
    if not _SYMBOL_ALLOWED.fullmatch(raw):
        raise ValueError("symbol contains unsupported characters")
    parts = raw.split(":")
    if len(parts) > 2:
        raise ValueError("symbol contains multiple settlement suffixes")
    base = parts[0]
    settle = parts[1] if len(parts) == 2 else None
    normalized = re.sub(r"[/_-]", "", base)
    if not normalized or not normalized.isascii() or not normalized.isalnum():
        raise ValueError("symbol must contain ASCII alphanumeric characters")
    if settle:
        if not settle.isascii() or not settle.isalnum() or not normalized.endswith(settle):
            raise ValueError("settlement suffix must match the normalized quote asset")
    return normalized


def _base36(value: int) -> str:
    alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if value == 0:
        return "0"
    encoded: list[str] = []
    while value:
        value, remainder = divmod(value, 36)
        encoded.append(alphabet[remainder])
    return "".join(reversed(encoded))


def build_client_order_id(
    *,
    account_id: str,
    symbol: str,
    position_side: str,
    idempotency_key: str,
    attempt: int = 0,
    prefix: str = "FTH",
) -> str:
    """Build a stable ID while preserving a hash suffix within the 36-char limit."""
    if not isinstance(attempt, int) or isinstance(attempt, bool):
        raise TypeError("attempt must be an integer")
    if attempt < 0 or attempt > _MAX_ATTEMPT:
        raise ValueError(f"attempt must be in [0, {_MAX_ATTEMPT}]")

    normalized_account = _clean_text(account_id, field_name="account_id", max_length=128)
    normalized_symbol = _normalize_symbol(symbol)
    normalized_key = _clean_text(
        idempotency_key, field_name="idempotency_key", max_length=256
    )
    raw_prefix = _clean_text(prefix, field_name="prefix", max_length=64).upper()
    if not _ASCII_ALNUM.fullmatch(raw_prefix):
        raise ValueError("prefix must contain only ASCII alphanumeric characters")
    normalized_prefix = raw_prefix[:4]
    side = _clean_text(position_side, field_name="position_side", max_length=8).upper()
    if side not in {"LONG", "SHORT"}:
        raise ValueError("position_side must be LONG or SHORT")

    material = "\x1f".join(
        (normalized_account, normalized_symbol, side, normalized_key, str(attempt))
    ).encode("utf-8")
    digest = base64.b32encode(hashlib.sha256(material).digest()).decode("ascii")
    digest = digest.rstrip("=")[:16]
    side_code = "L" if side == "LONG" else "S"
    client_id = (
        f"{normalized_prefix}-{normalized_symbol[:8]}-{side_code}{_base36(attempt)}-{digest}"
    )
    validate_client_order_id(client_id)
    return client_id


def validate_client_order_id(value: str) -> None:
    if not isinstance(value, str):
        raise TypeError("client order id must be a string")
    if not _ALLOWED.fullmatch(value):
        raise ValueError("client order id must match [A-Za-z0-9_-] and be <= 36 chars")
