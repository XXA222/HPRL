"""Frozen cross-direction hedge identity and order semantics contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from .errors import HedgeContractError

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_SYMBOL_ALLOWED = re.compile(r"^[A-Za-z0-9/_:-]+$")
_QUOTE_SUFFIXES = ("FDUSD", "USDT", "USDC", "BUSD", "TUSD", "USD", "BTC", "ETH", "EUR", "TRY")


class PositionSide(StrEnum):
    LONG = "LONG"
    SHORT = "SHORT"


class IntentAction(StrEnum):
    OPEN = "OPEN"
    INCREASE = "INCREASE"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"

    @property
    def reduces_risk(self) -> bool:
        return self in {IntentAction.REDUCE, IntentAction.CLOSE}


class OrderSide(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class TimeInForce(StrEnum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"
    GTX = "GTX"


def required_text(value: object, *, field_name: str, max_length: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > max_length or _CONTROL.search(normalized):
        raise ValueError(f"{field_name} is invalid")
    return normalized


def finite_decimal(value: Decimal | str | int | float, *, field_name: str) -> Decimal:
    if isinstance(value, (bool, float)):
        # Floats were accepted by the frozen v1 contract; stringify to keep exact text.
        if isinstance(value, bool):
            raise HedgeContractError(f"{field_name} must be a finite Decimal.")
    try:
        result = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise HedgeContractError(f"{field_name} must be a finite Decimal.") from exc
    if not result.is_finite():
        raise HedgeContractError(f"{field_name} must be a finite Decimal.")
    return result


def _canonical_pair(value: object) -> str:
    raw = required_text(value, field_name="symbol", max_length=64).upper().replace("-", "/").replace("_", "/")
    if not _SYMBOL_ALLOWED.fullmatch(raw):
        raise ValueError("symbol contains unsupported characters")
    market, separator, settle = raw.partition(":")
    if "/" in market:
        base, slash, quote = market.partition("/")
        if slash != "/" or not base or not quote or "/" in quote:
            raise ValueError("canonical_symbol must be a futures pair")
        settlement = settle or quote
        return f"{base}/{quote}:{settlement}"
    compact = re.sub(r"[/_-]", "", market)
    for quote in _QUOTE_SUFFIXES:
        if compact.endswith(quote) and len(compact) > len(quote):
            base = compact[:-len(quote)]
            return f"{base}/{quote}:{settle or quote}"
    if not compact or not compact.isascii() or not compact.isalnum():
        raise ValueError("symbol must contain ASCII alphanumeric characters")
    return compact


def canonical_symbol(value: object) -> str:
    """Return the frozen public Freqtrade futures-pair representation."""
    return _canonical_pair(value)


def _raw_symbol(value: object) -> str:
    canonical = _canonical_pair(value)
    market = canonical.split(":", 1)[0]
    return market.replace("/", "")


def expected_order_side(position_side: PositionSide, action: IntentAction) -> OrderSide:
    if position_side is PositionSide.LONG:
        return OrderSide.SELL if action.reduces_risk else OrderSide.BUY
    return OrderSide.BUY if action.reduces_risk else OrderSide.SELL


@dataclass(frozen=True, slots=True, order=True, init=False)
class PositionKey:
    exchange: str
    account_id: str
    symbol: str
    position_side: PositionSide
    _canonical_symbol: str = field(compare=False, repr=False)

    def __init__(
        self,
        exchange: str,
        account_id: str,
        symbol: str | None = None,
        position_side: PositionSide | str | None = None,
        *,
        canonical_symbol: str | None = None,
    ) -> None:
        if symbol is None:
            symbol = canonical_symbol
        elif canonical_symbol is not None and _raw_symbol(symbol) != _raw_symbol(canonical_symbol):
            raise ValueError("symbol and canonical_symbol identify different markets")
        if symbol is None:
            raise TypeError("symbol or canonical_symbol is required")
        exchange_value = required_text(exchange, field_name="exchange", max_length=64).lower()
        account_value = required_text(account_id, field_name="account_id", max_length=128)
        if position_side is None:
            raise TypeError("position_side is required")
        try:
            side = position_side if isinstance(position_side, PositionSide) else PositionSide(str(position_side).upper())
        except (TypeError, ValueError) as exc:
            raise ValueError("position_side is invalid") from exc
        object.__setattr__(self, "exchange", exchange_value)
        object.__setattr__(self, "account_id", account_value)
        object.__setattr__(self, "symbol", _raw_symbol(symbol))
        object.__setattr__(self, "position_side", side)
        object.__setattr__(self, "_canonical_symbol", _canonical_pair(symbol))

    @property
    def canonical_symbol(self) -> str:
        return self._canonical_symbol

    @property
    def lock_name(self) -> str:
        return f"{self.exchange}:{self.account_id}:{self.symbol}:{self.position_side.value}"


@dataclass(frozen=True, slots=True)
class PositionRecord:
    key: PositionKey
    quantity: Decimal
    entry_price: Decimal
    observed_time_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.key, PositionKey):
            raise HedgeContractError("key must be a PositionKey")
        object.__setattr__(self, "quantity", finite_decimal(self.quantity, field_name="quantity"))
        object.__setattr__(self, "entry_price", finite_decimal(self.entry_price, field_name="entry_price"))
        if isinstance(self.observed_time_ms, bool) or int(self.observed_time_ms) < 0:
            raise HedgeContractError("observed_time_ms must be nonnegative")


@dataclass(frozen=True, slots=True)
class TargetPosition:
    key: PositionKey
    target_quantity: Decimal
    correlation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_quantity", finite_decimal(self.target_quantity, field_name="target_quantity"))


@dataclass(frozen=True, slots=True)
class OrderIntent:
    intent_id: str
    key: PositionKey
    quantity: Decimal
    idempotency_key: str
    correlation_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantity", finite_decimal(self.quantity, field_name="quantity"))


@dataclass(frozen=True, slots=True)
class ApprovedOrderIntent:
    intent: OrderIntent
    approval_id: str
    approved_time_ms: int


@dataclass(frozen=True, slots=True)
class OrderSnapshot:
    order_id: str
    key: PositionKey
    status: str
    filled_quantity: Decimal
    exchange_time_ms: int
    observed_time_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "filled_quantity", finite_decimal(self.filled_quantity, field_name="filled_quantity"))


@dataclass(frozen=True, slots=True)
class AccountRiskSnapshot:
    account_id: str
    equity: Decimal
    margin_ratio: Decimal
    exchange_time_ms: int
    observed_time_ms: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "equity", finite_decimal(self.equity, field_name="equity"))
        object.__setattr__(self, "margin_ratio", finite_decimal(self.margin_ratio, field_name="margin_ratio"))


@dataclass(frozen=True, slots=True)
class ReconciliationDiff:
    key: PositionKey
    local_quantity: Decimal | None
    remote_quantity: Decimal | None
    reason: str

    def __post_init__(self) -> None:
        if self.local_quantity is not None:
            object.__setattr__(self, "local_quantity", finite_decimal(self.local_quantity, field_name="local_quantity"))
        if self.remote_quantity is not None:
            object.__setattr__(self, "remote_quantity", finite_decimal(self.remote_quantity, field_name="remote_quantity"))
