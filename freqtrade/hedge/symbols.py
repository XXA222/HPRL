"""Canonical symbol helpers shared by all Hedge workstreams."""

from __future__ import annotations

import re

from freqtrade.hedge.errors import HedgeDataError
from freqtrade.hedge.exchange.symbol_codec import to_binance_symbol, to_canonical_pair

_SEPARATORS = re.compile(r"[/:\-_\s]")


def normalize_unified_symbol(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HedgeDataError("symbol must be a non-empty string.")
    normalized = value.strip().upper().replace(" ", "")
    if "/" not in normalized:
        return normalized
    pair, separator, settle = normalized.partition(":")
    base, slash, quote = pair.partition("/")
    if slash != "/" or not base or not quote:
        raise HedgeDataError(f"invalid unified symbol: {value!r}.")
    return f"{base}/{quote}{separator}{settle}" if separator else f"{base}/{quote}"


def canonicalize_symbol(value: str, *, managed_pair: str | None = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HedgeDataError("symbol must be a non-empty string.")
    raw = normalize_unified_symbol(value)
    if raw == "UNKNOWN":
        return raw
    try:
        canonical = to_canonical_pair(to_binance_symbol(raw)) if "/" in raw else to_canonical_pair(raw)
    except ValueError as exc:
        raise HedgeDataError(f"Unsupported hedge symbol: {value!r}.") from exc
    if managed_pair is None:
        return canonical
    managed = canonicalize_symbol(managed_pair)
    return managed if raw_symbol(canonical) == raw_symbol(managed) else canonical


def raw_symbol(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HedgeDataError("symbol must be a non-empty string.")
    try:
        return to_binance_symbol(canonicalize_symbol(value))
    except ValueError as exc:
        raise HedgeDataError(f"Unsupported hedge symbol: {value!r}.") from exc


def symbols_equivalent(left: str, right: str) -> bool:
    return raw_symbol(left) == raw_symbol(right)
