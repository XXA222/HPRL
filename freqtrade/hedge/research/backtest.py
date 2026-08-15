"""Research-grade analytics layered over the existing Hedge backtest engine."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from statistics import fmean, pstdev


@dataclass(frozen=True, slots=True)
class EquitySummary:
    start: float
    end: float
    total_return: float
    max_drawdown: float
    volatility: float
    sharpe: float | None
    downside_deviation: float
    sortino: float | None
    win_rate: float
    profit_factor: float | None


def _finite(values: Iterable[float]) -> tuple[float, ...]:
    output = tuple(float(item) for item in values)
    if not output or any(not math.isfinite(item) for item in output):
        raise ValueError("research series must contain finite values")
    return output


def returns_from_equity(equity: Sequence[float]) -> tuple[float, ...]:
    values = _finite(equity)
    if any(item <= 0 for item in values):
        raise ValueError("equity must remain positive")
    return tuple(right / left - 1.0 for left, right in pairwise(values))


def maximum_drawdown(equity: Sequence[float]) -> float:
    values = _finite(equity)
    if any(item <= 0 for item in values):
        raise ValueError("equity must remain positive")
    peak = values[0]
    maximum = 0.0
    for value in values:
        peak = max(peak, value)
        maximum = max(maximum, 1.0 - value / peak)
    return maximum


def _risk_statistics(returns: Sequence[float], periods_per_year: int) -> dict[str, float | None]:
    average = fmean(returns)
    volatility = pstdev(returns) if len(returns) > 1 else 0.0
    annualizer = math.sqrt(periods_per_year)
    downside = tuple(min(0.0, item) for item in returns)
    downside_deviation = math.sqrt(fmean(item * item for item in downside))
    return {
        "volatility": volatility * annualizer,
        "sharpe": None if volatility == 0 else average / volatility * annualizer,
        "downside_deviation": downside_deviation * annualizer,
        "sortino": (
            None
            if downside_deviation == 0
            else average / downside_deviation * annualizer
        ),
    }


def _trade_statistics(returns: Sequence[float]) -> tuple[float, float | None]:
    wins = tuple(item for item in returns if item > 0)
    losses = tuple(item for item in returns if item < 0)
    gross_loss = -sum(losses)
    return (
        len(wins) / len(returns),
        None if gross_loss == 0 else sum(wins) / gross_loss,
    )


def summarize_equity(
    equity: Sequence[float],
    *,
    periods_per_year: int = 365 * 24 * 60,
) -> EquitySummary:
    values = _finite(equity)
    if len(values) < 2 or periods_per_year < 1:
        raise ValueError("equity summary requires at least two points and positive annualization")
    returns = returns_from_equity(values)
    risk = _risk_statistics(returns, periods_per_year)
    win_rate, profit_factor = _trade_statistics(returns)
    return EquitySummary(
        start=values[0],
        end=values[-1],
        total_return=values[-1] / values[0] - 1.0,
        max_drawdown=maximum_drawdown(values),
        volatility=float(risk["volatility"]),
        sharpe=risk["sharpe"],
        downside_deviation=float(risk["downside_deviation"]),
        sortino=risk["sortino"],
        win_rate=win_rate,
        profit_factor=profit_factor,
    )


def rolling_returns(equity: Sequence[float], *, window: int) -> tuple[float, ...]:
    values = _finite(equity)
    if window < 1 or window >= len(values):
        raise ValueError("rolling window must fit inside equity series")
    return tuple(
        values[index] / values[index - window] - 1.0
        for index in range(window, len(values))
    )


def benchmark_excess(strategy_equity: Sequence[float], benchmark_equity: Sequence[float]) -> float:
    strategy = _finite(strategy_equity)
    benchmark = _finite(benchmark_equity)
    if len(strategy) != len(benchmark):
        raise ValueError("strategy and benchmark lengths must match")
    if any(item <= 0 for item in strategy + benchmark):
        raise ValueError("strategy and benchmark equity must remain positive")
    return (strategy[-1] / strategy[0] - 1.0) - (benchmark[-1] / benchmark[0] - 1.0)


def fee_sensitivity(
    base_return: float,
    turnovers: Sequence[float],
    fee_bps: Sequence[float],
) -> dict[str, float]:
    if not math.isfinite(float(base_return)):
        raise ValueError("base return must be finite")
    turnover_values = _finite(turnovers)
    fees = _finite(fee_bps)
    if any(item < 0 for item in turnover_values + fees):
        raise ValueError("turnover and fee bps cannot be negative")
    total_turnover = sum(turnover_values)
    return {
        f"{fee:g}bps": float(base_return) - total_turnover * fee / 10_000.0
        for fee in fees
    }


def funding_sensitivity(
    base_return: float,
    exposure_steps: Sequence[float],
    rates: Sequence[float],
) -> dict[str, float]:
    if not math.isfinite(float(base_return)):
        raise ValueError("base return must be finite")
    exposure = _finite(exposure_steps)
    rate_values = _finite(rates)
    gross = sum(abs(item) for item in exposure)
    return {f"{rate:g}": float(base_return) - gross * rate for rate in rate_values}


def scenario_matrix(
    base_return: float,
    *,
    fee_multipliers: Sequence[float],
    slippage_multipliers: Sequence[float],
    cost_fraction: float,
) -> tuple[dict[str, float], ...]:
    if not math.isfinite(float(base_return)) or not math.isfinite(float(cost_fraction)):
        raise ValueError("scenario inputs must be finite")
    fees = _finite(fee_multipliers)
    slips = _finite(slippage_multipliers)
    if cost_fraction < 0 or any(item < 0 for item in fees + slips):
        raise ValueError("cost fraction and multipliers cannot be negative")
    return tuple(
        {
            "fee_multiplier": fee,
            "slippage_multiplier": slip,
            "stressed_return": float(base_return) - cost_fraction * (fee + slip) / 2.0,
        }
        for fee in fees
        for slip in slips
    )


def compare_runs(
    rows: Sequence[dict[str, float]],
    *,
    metric: str,
    descending: bool = True,
) -> tuple[dict[str, float], ...]:
    if not rows or any(metric not in row for row in rows):
        raise ValueError("comparison rows must contain requested metric")
    if any(not math.isfinite(float(row[metric])) for row in rows):
        raise ValueError("comparison metric must be finite")
    return tuple(
        sorted(
            (dict(row) for row in rows),
            key=lambda row: row[metric],
            reverse=descending,
        )
    )
