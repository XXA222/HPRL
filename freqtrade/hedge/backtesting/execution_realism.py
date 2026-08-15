"""Deterministic execution-realism helpers for conservative Hedge replay."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, ROUND_UP, Decimal


ZERO = Decimal(0)
ONE = Decimal(1)
BPS = Decimal(10000)


def _d(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    result = Decimal(str(value))
    if not result.is_finite():
        raise ValueError(f"{name} must be finite")
    return result


def quantize_price(price: object, tick_size: object, *, round_up: bool = False) -> Decimal:
    price_d = _d(price, name="price")
    tick = _d(tick_size, name="tick_size")
    if price_d < 0 or tick <= 0:
        raise ValueError("price must be non-negative and tick_size positive")
    rounding = ROUND_UP if round_up else ROUND_DOWN
    return (price_d / tick).to_integral_value(rounding=rounding) * tick


def quantize_quantity(quantity: object, step_size: object) -> Decimal:
    qty = _d(quantity, name="quantity")
    step = _d(step_size, name="step_size")
    if qty < 0 or step <= 0:
        raise ValueError("quantity must be non-negative and step_size positive")
    return (qty / step).to_integral_value(rounding=ROUND_DOWN) * step


def validate_min_notional(quantity: object, price: object, minimum: object) -> bool:
    qty = _d(quantity, name="quantity")
    px = _d(price, name="price")
    threshold = _d(minimum, name="minimum")
    if min(qty, px, threshold) < 0:
        raise ValueError("notional inputs cannot be negative")
    return qty * px >= threshold


def fee_rate_by_volume(
    rolling_volume: object,
    tiers: Sequence[tuple[object, object]],
) -> Decimal:
    volume = _d(rolling_volume, name="rolling_volume")
    if volume < 0 or not tiers:
        raise ValueError("rolling volume must be non-negative and tiers non-empty")
    normalized = sorted(
        (
            (_d(limit, name="tier_limit"), _d(rate, name="tier_rate"))
            for limit, rate in tiers
        )
    )
    if normalized[0][0] != ZERO:
        raise ValueError("fee tiers must begin at zero")
    if any(rate < 0 for _, rate in normalized):
        raise ValueError("fee rates cannot be negative")
    selected = normalized[0][1]
    for limit, rate in normalized:
        if volume >= limit:
            selected = rate
        else:
            break
    return selected


def linear_slippage_bps(
    order_notional: object,
    market_notional: object,
    coefficient_bps: object,
) -> Decimal:
    order = _d(order_notional, name="order_notional")
    market = _d(market_notional, name="market_notional")
    coefficient = _d(coefficient_bps, name="coefficient_bps")
    if order < 0 or market <= 0 or coefficient < 0:
        raise ValueError("slippage inputs are outside valid range")
    return coefficient * order / market


def square_root_slippage_bps(
    order_notional: object,
    market_notional: object,
    coefficient_bps: object,
) -> Decimal:
    order = _d(order_notional, name="order_notional")
    market = _d(market_notional, name="market_notional")
    coefficient = _d(coefficient_bps, name="coefficient_bps")
    if order < 0 or market <= 0 or coefficient < 0:
        raise ValueError("slippage inputs are outside valid range")
    return coefficient * Decimal(str(math.sqrt(float(order / market))))


def participation_cap(
    bar_volume: object,
    participation: object,
    remaining_quantity: object,
) -> Decimal:
    volume = _d(bar_volume, name="bar_volume")
    ratio = _d(participation, name="participation")
    remaining = _d(remaining_quantity, name="remaining_quantity")
    if volume < 0 or remaining < 0 or ratio < 0 or ratio > ONE:
        raise ValueError("participation inputs are outside valid range")
    return min(remaining, volume * ratio)


def latency_shift(timestamp: datetime, latency_ms: int) -> datetime:
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    if latency_ms < 0:
        raise ValueError("latency_ms cannot be negative")
    return timestamp + timedelta(milliseconds=latency_ms)


def deterministic_partial_fill(
    remaining_quantity: object,
    executable_quantity: object,
    *,
    maximum_fill_ratio: object = ONE,
) -> Decimal:
    remaining = _d(remaining_quantity, name="remaining_quantity")
    executable = _d(executable_quantity, name="executable_quantity")
    ratio = _d(maximum_fill_ratio, name="maximum_fill_ratio")
    if remaining < 0 or executable < 0 or ratio < 0 or ratio > ONE:
        raise ValueError("partial-fill inputs are outside valid range")
    return min(remaining, executable, remaining * ratio)


def reduce_only_quantity(position_quantity: object, requested_quantity: object) -> Decimal:
    position = abs(_d(position_quantity, name="position_quantity"))
    requested = _d(requested_quantity, name="requested_quantity")
    if requested < 0:
        raise ValueError("requested quantity cannot be negative")
    return min(position, requested)


@dataclass(frozen=True, slots=True)
class QueueFill:
    filled: Decimal
    queue_ahead_after: Decimal


def queue_position_fill(
    traded_at_price: object,
    queue_ahead: object,
    remaining_quantity: object,
) -> QueueFill:
    traded = _d(traded_at_price, name="traded_at_price")
    ahead = _d(queue_ahead, name="queue_ahead")
    remaining = _d(remaining_quantity, name="remaining_quantity")
    if min(traded, ahead, remaining) < 0:
        raise ValueError("queue quantities cannot be negative")
    consumed_ahead = min(traded, ahead)
    leftover_trade = traded - consumed_ahead
    return QueueFill(min(remaining, leftover_trade), ahead - consumed_ahead)
