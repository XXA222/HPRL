from __future__ import annotations

import re
from dataclasses import dataclass

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{3,30}$")


@dataclass(frozen=True, slots=True)
class BinanceSymbol:
    """Canonical Freqtrade pair and Binance exchange symbol."""

    canonical: str
    exchange: str
    base: str
    quote: str
    settle: str


def _clean_token(value: str, *, field: str) -> str:
    token = str(value).strip().upper()
    if not token or not token.isalnum():
        raise ValueError(f"{field} must be a nonempty alphanumeric token")
    return token


def parse_canonical_pair(value: str) -> BinanceSymbol:
    """Parse ``ETH/USDT:USDT`` (or ``ETH/USDT``) for Binance USD-M."""

    raw = str(value).strip().upper()
    if not raw:
        raise ValueError("symbol must not be empty")
    if "/" not in raw:
        if not _SYMBOL_RE.fullmatch(raw):
            raise ValueError(f"invalid Binance exchange symbol: {value!r}")
        # Exchange-only symbols cannot be split safely without exchange metadata.
        return BinanceSymbol(raw, raw, "", "", "")

    market, separator, settle_value = raw.partition(":")
    base_value, slash, quote_value = market.partition("/")
    if slash != "/" or not base_value or not quote_value:
        raise ValueError(f"invalid canonical futures pair: {value!r}")
    base = _clean_token(base_value, field="base")
    quote = _clean_token(quote_value, field="quote")
    settle = _clean_token(settle_value if separator else quote, field="settle")
    if settle != quote:
        raise ValueError(
            "Binance USD-M canonical pair must settle in the quote asset"
        )
    exchange = f"{base}{quote}"
    return BinanceSymbol(
        canonical=f"{base}/{quote}:{settle}",
        exchange=exchange,
        base=base,
        quote=quote,
        settle=settle,
    )


def to_binance_symbol(value: str) -> str:
    return parse_canonical_pair(value).exchange


def to_canonical_pair(
    exchange_symbol: str,
    *,
    quote_assets: tuple[str, ...] = ("USDT", "USDC", "FDUSD", "BUSD"),
) -> str:
    """Convert a known USD-M exchange symbol to a Freqtrade canonical pair."""

    symbol = _clean_token(exchange_symbol, field="exchange_symbol")
    for quote in sorted(quote_assets, key=len, reverse=True):
        normalized_quote = _clean_token(quote, field="quote_asset")
        if symbol.endswith(normalized_quote) and len(symbol) > len(normalized_quote):
            base = symbol[: -len(normalized_quote)]
            return f"{base}/{normalized_quote}:{normalized_quote}"
    raise ValueError(
        f"cannot infer quote asset for Binance exchange symbol {exchange_symbol!r}"
    )


def normalize_exchange_symbols(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("symbols must be a sequence, not a string")
    normalized = {to_binance_symbol(value) for value in values}
    if not normalized:
        raise ValueError("symbols must not be empty")
    return tuple(sorted(normalized))
