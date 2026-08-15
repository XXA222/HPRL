"""Strict market-rule discovery for Binance USD-M Testnet canaries.

The probe is intentionally public-data only.  It validates that a symbol is a
TRADING PERPETUAL contract, derives exact Decimal filters from ``exchangeInfo``
and chooses a passive post-only price inside Binance's percent-price envelope.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_FLOOR
from typing import Any, Mapping
from urllib.parse import urlencode

from .binance_environment import TESTNET_PROFILE
from .binance_usdm_adapter import HttpTransport, UrlLibHttpTransport
from .service import PositionSide
from .testnet import DEFAULT_TESTNET_MAX_NOTIONAL, TESTNET_ALLOWED_SYMBOLS


@dataclass(frozen=True, slots=True)
class TestnetSymbolRules:
    symbol: str
    mark_price: Decimal
    price_tick: Decimal
    quantity_step: Decimal
    minimum_quantity: Decimal
    maximum_quantity: Decimal
    minimum_notional: Decimal
    multiplier_up: Decimal
    multiplier_down: Decimal


@dataclass(frozen=True, slots=True)
class TestnetCanaryOrder:
    symbol: str
    position_side: PositionSide
    quantity: Decimal
    limit_price: Decimal
    notional: Decimal
    time_in_force: str
    mark_price: Decimal
    passive_offset_bps: int


class BinanceTestnetMarketProbe:
    """Fetch public Testnet market rules and construct a bounded GTX canary."""

    def __init__(
        self,
        *,
        transport: HttpTransport | None = None,
        proxy_url: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._transport = transport or UrlLibHttpTransport(proxy_url=proxy_url)
        self._timeout = float(timeout_seconds)

    def rules(self, symbol: str) -> TestnetSymbolRules:
        normalized = _symbol(symbol)
        exchange_info = self._get("/fapi/v1/exchangeInfo", {"symbol": normalized})
        premium = self._get("/fapi/v1/premiumIndex", {"symbol": normalized})
        if not isinstance(exchange_info, Mapping):
            raise RuntimeError("Testnet exchangeInfo returned invalid payload")
        rows = exchange_info.get("symbols")
        if not isinstance(rows, list):
            raise RuntimeError("Testnet exchangeInfo symbols are missing")
        row = next(
            (
                item
                for item in rows
                if isinstance(item, Mapping) and str(item.get("symbol", "")).upper() == normalized
            ),
            None,
        )
        if row is None:
            raise RuntimeError(f"Testnet symbol is unavailable: {normalized}")
        if str(row.get("contractType", "")).upper() != "PERPETUAL":
            raise RuntimeError(f"Testnet symbol is not PERPETUAL: {normalized}")
        if str(row.get("status", "")).upper() != "TRADING":
            raise RuntimeError(f"Testnet symbol is not TRADING: {normalized}")
        filters = {
            str(item.get("filterType", "")): item
            for item in row.get("filters", [])
            if isinstance(item, Mapping)
        }
        price_filter = filters.get("PRICE_FILTER", {})
        lot_filter = filters.get("LOT_SIZE", {})
        notional_filter = filters.get("MIN_NOTIONAL", {})
        percent_filter = filters.get("PERCENT_PRICE", {})
        if not isinstance(premium, Mapping):
            raise RuntimeError("Testnet premiumIndex returned invalid payload")
        return TestnetSymbolRules(
            symbol=normalized,
            mark_price=_positive_decimal(premium.get("markPrice"), "markPrice"),
            price_tick=_positive_decimal(price_filter.get("tickSize"), "tickSize"),
            quantity_step=_positive_decimal(lot_filter.get("stepSize"), "stepSize"),
            minimum_quantity=_positive_decimal(lot_filter.get("minQty"), "minQty"),
            maximum_quantity=_positive_decimal(lot_filter.get("maxQty"), "maxQty"),
            minimum_notional=_positive_decimal(
                notional_filter.get("notional", notional_filter.get("minNotional")),
                "minimum notional",
            ),
            multiplier_up=_positive_decimal(percent_filter.get("multiplierUp"), "multiplierUp"),
            multiplier_down=_positive_decimal(
                percent_filter.get("multiplierDown"), "multiplierDown"
            ),
        )

    def passive_order(
        self,
        *,
        symbol: str,
        position_side: PositionSide,
        target_notional: Decimal = Decimal("10"),
        max_notional: Decimal = DEFAULT_TESTNET_MAX_NOTIONAL,
        passive_offset_bps: int = 1000,
    ) -> TestnetCanaryOrder:
        rules = self.rules(symbol)
        side = (
            position_side
            if isinstance(position_side, PositionSide)
            else PositionSide(position_side)
        )
        target = _bounded_notional(target_notional, field_name="target_notional")
        maximum = _bounded_notional(max_notional, field_name="max_notional")
        if target > maximum:
            raise ValueError("target_notional exceeds max_notional")
        if not isinstance(passive_offset_bps, int) or isinstance(passive_offset_bps, bool):
            raise TypeError("passive_offset_bps must be an integer")
        if not 100 <= passive_offset_bps <= 3000:
            raise ValueError("passive_offset_bps must be in [100, 3000]")
        offset = Decimal(passive_offset_bps) / Decimal("10000")
        if side is PositionSide.LONG:
            raw_price = rules.mark_price * (Decimal("1") - offset)
            lower_guard = rules.mark_price * rules.multiplier_down * Decimal("1.001")
            raw_price = max(raw_price, lower_guard)
            price = _quantize_down(raw_price, rules.price_tick)
        else:
            raw_price = rules.mark_price * (Decimal("1") + offset)
            upper_guard = rules.mark_price * rules.multiplier_up * Decimal("0.999")
            raw_price = min(raw_price, upper_guard)
            price = _quantize_up(raw_price, rules.price_tick)
        if price <= 0:
            raise RuntimeError("derived Testnet canary price is invalid")
        required_notional = max(target, rules.minimum_notional * Decimal("1.01"))
        quantity = _quantize_up(required_notional / price, rules.quantity_step)
        quantity = max(quantity, rules.minimum_quantity)
        if quantity > rules.maximum_quantity:
            raise RuntimeError("derived Testnet canary quantity exceeds exchange maximum")
        notional = quantity * price
        if notional < rules.minimum_notional:
            quantity = _quantize_up(rules.minimum_notional / price, rules.quantity_step)
            notional = quantity * price
        if notional > maximum:
            raise RuntimeError(
                "Testnet minimum order notional cannot fit inside configured safety cap"
            )
        return TestnetCanaryOrder(
            symbol=rules.symbol,
            position_side=side,
            quantity=quantity,
            limit_price=price,
            notional=notional,
            time_in_force="GTX",
            mark_price=rules.mark_price,
            passive_offset_bps=passive_offset_bps,
        )

    def _get(self, path: str, params: Mapping[str, Any]) -> Any:
        allowed = {"/fapi/v1/exchangeInfo", "/fapi/v1/premiumIndex"}
        if path not in allowed:
            raise ValueError("unsupported Testnet public market endpoint")
        query = urlencode({key: str(value) for key, value in params.items()})
        url = f"{TESTNET_PROFILE.rest_base_url}{path}?{query}"
        response = self._transport.request("GET", url, {}, self._timeout)
        if response.status != 200:
            raise RuntimeError(f"Testnet public market request failed: HTTP {response.status}")
        try:
            return json.loads(response.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Testnet public market response is invalid JSON") from exc


for _public_type in (TestnetSymbolRules, TestnetCanaryOrder, BinanceTestnetMarketProbe):
    setattr(_public_type, "__test__", False)


def _symbol(value: str) -> str:
    result = str(value).strip().upper().replace("/", "").split(":", 1)[0]
    if result not in TESTNET_ALLOWED_SYMBOLS:
        raise ValueError("Testnet market probe only supports BTCUSDT and ETHUSDT")
    return result


def _positive_decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError(f"{field_name} must use exact decimal text")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is invalid") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return result


def _bounded_notional(value: Decimal, *, field_name: str) -> Decimal:
    result = _positive_decimal(value, field_name)
    if result > DEFAULT_TESTNET_MAX_NOTIONAL:
        raise ValueError(f"{field_name} must not exceed 25 USDT")
    return result


def _quantize_down(value: Decimal, step: Decimal) -> Decimal:
    units = (value / step).to_integral_value(rounding=ROUND_FLOOR)
    return units * step


def _quantize_up(value: Decimal, step: Decimal) -> Decimal:
    units = (value / step).to_integral_value(rounding=ROUND_CEILING)
    return units * step
