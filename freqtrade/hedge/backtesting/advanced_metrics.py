"""Additional risk and trade statistics for Hedge research reports."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from math import ceil, sqrt
from statistics import fmean


ZERO = Decimal(0)
ONE = Decimal(1)
DEFAULT_TAIL_QUANTILE = Decimal("0.95")
DEFAULT_CVAR_ALPHA = Decimal("0.95")


def _series(values: Iterable[object], *, name: str) -> tuple[Decimal, ...]:
    output: list[Decimal] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric")
        item = Decimal(str(value))
        if not item.is_finite():
            raise ValueError(f"{name} must contain finite values")
        output.append(item)
    return tuple(output)


def compound_annual_growth_rate(initial: object, final: object, years: object) -> Decimal:
    start = Decimal(str(initial))
    finish = Decimal(str(final))
    years_d = Decimal(str(years))
    if start <= 0 or finish < 0 or years_d <= 0:
        raise ValueError("CAGR inputs are outside valid range")
    return Decimal(str((float(finish / start) ** (1.0 / float(years_d))) - 1.0))


def omega_ratio(returns: Iterable[object], threshold: object = ZERO) -> Decimal:
    values = _series(returns, name="returns")
    target = Decimal(str(threshold))
    gains = sum((max(value - target, ZERO) for value in values), ZERO)
    losses = sum((max(target - value, ZERO) for value in values), ZERO)
    if losses == ZERO:
        return Decimal("Infinity") if gains > ZERO else ZERO
    return gains / losses


def ulcer_index(drawdowns: Iterable[object]) -> Decimal:
    values = _series(drawdowns, name="drawdowns")
    if not values:
        return ZERO
    if any(value < 0 for value in values):
        raise ValueError("drawdowns cannot be negative")
    return Decimal(str(sqrt(fmean(float(value * value) for value in values))))


def recovery_factor(net_profit: object, maximum_drawdown_amount: object) -> Decimal:
    profit = Decimal(str(net_profit))
    drawdown = Decimal(str(maximum_drawdown_amount))
    if drawdown < 0:
        raise ValueError("maximum drawdown amount cannot be negative")
    if drawdown == ZERO:
        return Decimal("Infinity") if profit > ZERO else ZERO
    return profit / drawdown


def profit_factor(trade_pnls: Iterable[object]) -> Decimal:
    values = _series(trade_pnls, name="trade_pnls")
    gains = sum((value for value in values if value > ZERO), ZERO)
    losses = -sum((value for value in values if value < ZERO), ZERO)
    if losses == ZERO:
        return Decimal("Infinity") if gains > ZERO else ZERO
    return gains / losses


def trade_expectancy(trade_pnls: Iterable[object]) -> Decimal:
    values = _series(trade_pnls, name="trade_pnls")
    return sum(values, ZERO) / Decimal(len(values)) if values else ZERO


def win_rate(trade_pnls: Iterable[object]) -> Decimal:
    values = _series(trade_pnls, name="trade_pnls")
    return Decimal(sum(value > ZERO for value in values)) / Decimal(len(values)) if values else ZERO


def tail_ratio(returns: Iterable[object], *, quantile: Decimal = DEFAULT_TAIL_QUANTILE) -> Decimal:
    values = sorted(_series(returns, name="returns"))
    if not values:
        return ZERO
    if quantile <= Decimal("0.5") or quantile >= ONE:
        raise ValueError("quantile must be between 0.5 and 1")
    upper_index = min(len(values) - 1, ceil((len(values) - 1) * float(quantile)))
    lower_index = max(0, int((len(values) - 1) * float(ONE - quantile)))
    upper = abs(values[upper_index])
    lower = abs(values[lower_index])
    if lower == ZERO:
        return Decimal("Infinity") if upper > ZERO else ZERO
    return upper / lower


def conditional_value_at_risk(
    returns: Iterable[object],
    *,
    alpha: Decimal = DEFAULT_CVAR_ALPHA,
) -> Decimal:
    values = sorted(_series(returns, name="returns"))
    if not values:
        return ZERO
    if alpha <= ZERO or alpha >= ONE:
        raise ValueError("alpha must be between zero and one")
    tail_count = max(1, int(len(values) * float(ONE - alpha)))
    return -sum(values[:tail_count], ZERO) / Decimal(tail_count)


def exposure_ratio(active_periods: int, total_periods: int) -> Decimal:
    if active_periods < 0 or total_periods <= 0 or active_periods > total_periods:
        raise ValueError("exposure period counts are outside valid range")
    return Decimal(active_periods) / Decimal(total_periods)
