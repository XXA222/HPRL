"""Lightweight market snapshot builder for the production main loop."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Mapping, Protocol

from freqtrade.hedge.planning.context import MarketSnapshot


class CandleLike(Protocol):
    symbol: str
    close_time: object
    close: Decimal


def _positive(value: object, fallback: Decimal) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return fallback
    return parsed if parsed.is_finite() and parsed > 0 else fallback


def _nonnegative(value: object, fallback: Decimal = Decimal("0")) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return fallback
    return parsed if parsed.is_finite() and parsed >= 0 else fallback


def _precision_step(value: object, fallback: Decimal) -> Decimal:
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal("1").scaleb(-value)
    return _positive(value, fallback)


def build_production_market_snapshot(
    *,
    exchange: Any,
    pair: str,
    candle: CandleLike,
    ticker: Mapping[str, Any] | None,
    fallback: Mapping[str, Any],
) -> MarketSnapshot:
    mark = _positive(candle.close, Decimal("0"))
    if mark <= 0:
        raise RuntimeError("production candle close must be positive")
    ticker = ticker or {}
    bid = _positive(ticker.get("bid"), mark)
    ask = _positive(ticker.get("ask"), mark)
    if bid > ask:
        bid, ask = ask, bid

    fallback_tick = _positive(fallback.get("tick_size"), Decimal("0.01"))
    fallback_qty = _positive(fallback.get("qty_step"), Decimal("0.001"))
    fallback_min_qty = _nonnegative(fallback.get("min_qty"))
    fallback_min_notional = _nonnegative(fallback.get("min_notional"))
    market = getattr(exchange, "markets", {}).get(pair)
    if isinstance(market, Mapping):
        precision = market.get("precision") if isinstance(market.get("precision"), Mapping) else {}
        limits = market.get("limits") if isinstance(market.get("limits"), Mapping) else {}
        amount_limits = limits.get("amount") if isinstance(limits.get("amount"), Mapping) else {}
        cost_limits = limits.get("cost") if isinstance(limits.get("cost"), Mapping) else {}
        try:
            tick = Decimal(str(exchange.price_get_one_pip(pair, float(mark))))
        except (AttributeError, ArithmeticError, TypeError, ValueError):
            tick = _precision_step(precision.get("price"), fallback_tick)
        qty = _precision_step(precision.get("amount"), fallback_qty)
        min_qty = _nonnegative(amount_limits.get("min"), fallback_min_qty)
        min_notional = _nonnegative(cost_limits.get("min"), fallback_min_notional)
    else:
        tick = fallback_tick
        qty = fallback_qty
        min_qty = fallback_min_qty
        min_notional = fallback_min_notional

    return MarketSnapshot(
        symbol=pair,
        timestamp=candle.close_time,  # type: ignore[arg-type]
        bid=bid,
        ask=ask,
        mark=mark,
        tick_size=tick,
        qty_step=qty,
        min_qty=min_qty,
        min_notional=min_notional,
    )
