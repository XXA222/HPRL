"""Versioned market input backed by analyzed DataProvider OHLCV and exchange rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from json import dumps
import logging
from typing import Any, Mapping

from freqtrade.hedge.planning.context import MarketSnapshot
from freqtrade.hedge.simulation.exchange import BarEvent

from .signal_provider import AnalyzedCandle

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MarketRuleSnapshot:
    tick_size: Decimal
    qty_step: Decimal
    min_qty: Decimal
    min_notional: Decimal
    maker_fee_rate: Decimal
    taker_fee_rate: Decimal
    fee_source: str
    source: str
    version: str


@dataclass(frozen=True, slots=True)
class PaperMarketInput:
    market: MarketSnapshot
    bar: BarEvent
    rules: MarketRuleSnapshot
    source: str

    def __post_init__(self) -> None:
        if self.market.symbol != self.bar.symbol:
            raise ValueError("market and bar symbols must match")
        if self.market.timestamp != self.bar.timestamp:
            raise ValueError("market and bar timestamps must match")
        if self.market.mark != self.bar.close:
            raise ValueError("Paper mark must equal the analyzed candle close")


def _positive(value: object, fallback: Decimal) -> Decimal:
    try:
        result = Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return fallback
    return result if result.is_finite() and result > 0 else fallback


def _nonnegative(value: object, fallback: Decimal = Decimal("0")) -> Decimal:
    try:
        result = Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return fallback
    return result if result.is_finite() and result >= 0 else fallback


def _step_from_precision(value: object, fallback: Decimal) -> Decimal:
    if isinstance(value, int) and not isinstance(value, bool):
        return Decimal("1").scaleb(-value)
    return _positive(value, fallback)




def _fee_rate(value: object, fallback: Decimal) -> Decimal:
    result = _nonnegative(value, fallback)
    return result if result <= Decimal("1") else fallback


def _exchange_fee_rates(
    *,
    exchange: Any,
    pair: str,
    market: Mapping[str, Any] | None,
    fallback: Mapping[str, Any],
) -> tuple[Decimal, Decimal, str]:
    fallback_maker = _fee_rate(fallback.get("maker_fee_rate"), Decimal("0.0002"))
    fallback_taker = _fee_rate(fallback.get("taker_fee_rate"), Decimal("0.0004"))
    getter = getattr(exchange, "get_fee", None)
    if callable(getter):
        try:
            maker = _fee_rate(
                getter(pair, taker_or_maker="maker"),
                fallback_maker,
            )
            taker = _fee_rate(
                getter(pair, taker_or_maker="taker"),
                fallback_taker,
            )
            return maker, taker, "EXCHANGE_ACCOUNT_FEE"
        except Exception as exc:
            # Fee lookup is an exchange boundary.  Market metadata remains a
            # deterministic fallback and the selected source is made explicit.
            logger.warning("Paper exchange fee lookup failed: %s", type(exc).__name__)
    if market is not None:
        raw_maker = market.get("maker")
        raw_taker = market.get("taker")
        if raw_maker is not None and raw_taker is not None:
            maker = _fee_rate(raw_maker, fallback_maker)
            taker = _fee_rate(raw_taker, fallback_taker)
            return maker, taker, "EXCHANGE_MARKET_FEE"
    return fallback_maker, fallback_taker, "CONFIG_FALLBACK"

def exchange_market_rules(
    *,
    exchange: Any,
    pair: str,
    fallback: Mapping[str, Any],
) -> MarketRuleSnapshot:
    market = getattr(exchange, "markets", {}).get(pair)
    if not isinstance(market, Mapping):
        market = None

    maker_fee, taker_fee, fee_source = _exchange_fee_rates(
        exchange=exchange,
        pair=pair,
        market=market,
        fallback=fallback,
    )
    if market is None:
        payload = {
            "tick_size": str(fallback.get("tick_size", "0.01")),
            "qty_step": str(fallback.get("qty_step", "0.001")),
            "min_qty": str(fallback.get("min_qty", "0")),
            "min_notional": str(fallback.get("min_notional", "0")),
            "maker_fee_rate": str(maker_fee),
            "taker_fee_rate": str(taker_fee),
            "fee_source": fee_source,
        }
        version = sha256(dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        return MarketRuleSnapshot(
            tick_size=_positive(payload["tick_size"], Decimal("0.01")),
            qty_step=_positive(payload["qty_step"], Decimal("0.001")),
            min_qty=_nonnegative(payload["min_qty"]),
            min_notional=_nonnegative(payload["min_notional"]),
            maker_fee_rate=maker_fee,
            taker_fee_rate=taker_fee,
            fee_source=fee_source,
            source="CONFIG_FALLBACK",
            version=version,
        )

    precision = market.get("precision") if isinstance(market.get("precision"), Mapping) else {}
    limits = market.get("limits") if isinstance(market.get("limits"), Mapping) else {}
    amount_limits = limits.get("amount") if isinstance(limits.get("amount"), Mapping) else {}
    cost_limits = limits.get("cost") if isinstance(limits.get("cost"), Mapping) else {}

    fallback_tick = _positive(fallback.get("tick_size"), Decimal("0.01"))
    fallback_qty = _positive(fallback.get("qty_step"), Decimal("0.001"))
    fallback_min_qty = _nonnegative(fallback.get("min_qty"))
    fallback_min_notional = _nonnegative(fallback.get("min_notional"))
    try:
        tick = Decimal(str(exchange.price_get_one_pip(pair, float(market.get("last") or 1))))
    except (AttributeError, ArithmeticError, TypeError, ValueError):
        tick = _step_from_precision(precision.get("price"), fallback_tick)
    qty = _step_from_precision(precision.get("amount"), fallback_qty)
    min_qty = _nonnegative(amount_limits.get("min"), fallback_min_qty)
    min_notional = _nonnegative(cost_limits.get("min"), fallback_min_notional)

    payload = {
        "pair": pair,
        "tick": str(tick),
        "qty": str(qty),
        "min_qty": str(min_qty),
        "min_notional": str(min_notional),
        "market_id": str(market.get("id", "")),
        "maker_fee_rate": str(maker_fee),
        "taker_fee_rate": str(taker_fee),
        "fee_source": fee_source,
    }
    version = sha256(dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    return MarketRuleSnapshot(
        tick_size=_positive(tick, Decimal("0.01")),
        qty_step=_positive(qty, Decimal("0.001")),
        min_qty=min_qty,
        min_notional=min_notional,
        maker_fee_rate=maker_fee,
        taker_fee_rate=taker_fee,
        fee_source=fee_source,
        source="EXCHANGE_MARKETS",
        version=version,
    )


def build_dataprovider_market_input(
    *,
    exchange: Any,
    pair: str,
    candle: AnalyzedCandle,
    fallback: Mapping[str, Any],
    ticker: Mapping[str, Any] | None = None,
) -> PaperMarketInput:
    """Build Paper input from the exact analyzed OHLCV row.

    Ticker data may refine the spread only.  It never supplies or expands the
    OHLC path, and the planner mark remains the analyzed candle close.
    """

    if candle.symbol != pair:
        raise ValueError("analyzed candle pair does not match managed pair")
    close = candle.close
    ticker = ticker if isinstance(ticker, Mapping) else {}
    bid = _positive(ticker.get("bid"), close)
    ask = _positive(ticker.get("ask"), close)
    if bid > ask:
        bid, ask = ask, bid
    # A delayed ticker can be far outside the closed candle. Keep it for spread
    # observability, but do not use it to manufacture the matching path.
    rules = exchange_market_rules(exchange=exchange, pair=pair, fallback=fallback)
    bar = candle.to_bar_event()
    market = MarketSnapshot(
        symbol=pair,
        timestamp=bar.timestamp,
        bid=bid,
        ask=ask,
        mark=close,
        tick_size=rules.tick_size,
        qty_step=rules.qty_step,
        min_qty=rules.min_qty,
        min_notional=rules.min_notional,
    )
    return PaperMarketInput(
        market=market,
        bar=bar,
        rules=rules,
        source=candle.source,
    )


def build_ticker_compat_market_input(
    *,
    exchange: Any,
    pair: str,
    ticker: Mapping[str, Any],
    fallback: Mapping[str, Any],
    event_time: datetime | None = None,
) -> PaperMarketInput:
    """Legacy test-only market input used only with explicit ticker_compat."""

    raw_mark = ticker.get("last") or ticker.get("close") or ticker.get("bid") or ticker.get("ask")
    if raw_mark is None:
        raise RuntimeError(f"No usable ticker price for {pair}")
    mark = _positive(raw_mark, Decimal("0"))
    if mark <= 0:
        raise RuntimeError(f"No positive ticker price for {pair}")
    bid = _positive(ticker.get("bid"), mark)
    ask = _positive(ticker.get("ask"), mark)
    if bid > ask:
        bid, ask = ask, bid
    rules = exchange_market_rules(exchange=exchange, pair=pair, fallback=fallback)
    timestamp = event_time or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    bar = BarEvent(
        timestamp=timestamp,
        symbol=pair,
        open=mark,
        high=max(mark, bid, ask),
        low=min(mark, bid, ask),
        close=mark,
        volume=(
            None
            if fallback.get("bar_volume") is None
            else _nonnegative(fallback.get("bar_volume"))
        ),
    )
    return PaperMarketInput(
        market=MarketSnapshot(
            symbol=pair,
            timestamp=timestamp,
            bid=bid,
            ask=ask,
            mark=mark,
            tick_size=rules.tick_size,
            qty_step=rules.qty_step,
            min_qty=rules.min_qty,
            min_notional=rules.min_notional,
        ),
        bar=bar,
        rules=rules,
        source="TICKER_COMPAT_TEST_ONLY",
    )


# Compatibility alias for external callers. Production controller code uses the
# explicit DataProvider/ticker-compat builders above.
def build_market_snapshot(
    *,
    exchange: Any,
    pair: str,
    ticker: Mapping[str, Any],
    fallback: Mapping[str, Any],
    event_time: datetime | None = None,
) -> tuple[MarketSnapshot, MarketRuleSnapshot]:
    value = build_ticker_compat_market_input(
        exchange=exchange,
        pair=pair,
        ticker=ticker,
        fallback=fallback,
        event_time=event_time,
    )
    return value.market, value.rules
