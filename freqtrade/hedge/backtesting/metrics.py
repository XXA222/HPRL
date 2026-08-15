from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal
from itertools import pairwise
from math import sqrt
from statistics import fmean, pstdev

from freqtrade.hedge.simulation.exchange import FillEvent, SimulationResult

from .advanced_metrics import (
    conditional_value_at_risk,
    exposure_ratio,
    omega_ratio,
    tail_ratio,
    ulcer_index,
)
from .decimal_utils import ONE, ZERO


def _safe_decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return Decimal(int(value))
    return Decimal(str(value))


def equity_returns(result: SimulationResult) -> tuple[Decimal, ...]:
    equities = [snapshot.equity for snapshot in result.snapshots]
    output: list[Decimal] = []
    for left, right in pairwise(equities):
        if left > ZERO:
            output.append((right / left) - ONE)
    return tuple(output)


def drawdown_series(result: SimulationResult) -> tuple[Decimal, ...]:
    peak = ZERO
    values: list[Decimal] = []
    for snapshot in result.snapshots:
        peak = max(peak, snapshot.equity)
        values.append((peak - snapshot.equity) / peak if peak > ZERO else ZERO)
    return tuple(values)


def compute_metrics(
    result: SimulationResult,
    *,
    periods_per_year: int = 365 * 24 * 60,
) -> dict[str, Decimal | int | bool | str]:
    report = dict(result.report)
    returns = equity_returns(result)
    float_returns = [float(item) for item in returns]
    mean = fmean(float_returns) if float_returns else 0.0
    volatility = pstdev(float_returns) if len(float_returns) > 1 else 0.0
    downside = [min(item, 0.0) for item in float_returns]
    downside_deviation = sqrt(fmean([item * item for item in downside])) if downside else 0.0
    annualizer = sqrt(max(periods_per_year, 1))
    sharpe = mean / volatility * annualizer if volatility > 0 else 0.0
    sortino = mean / downside_deviation * annualizer if downside_deviation > 0 else 0.0
    drawdowns = drawdown_series(result)
    max_drawdown_ratio = max(drawdowns, default=ZERO)
    total_return = _safe_decimal(report.get("total_return_ratio", ZERO))
    calmar = total_return / max_drawdown_ratio if max_drawdown_ratio > ZERO else ZERO
    fills = [event for event in result.events if isinstance(event, FillEvent)]
    notional = sum((event.quantity * event.price for event in fills), ZERO)
    maker_fills = sum(str(event.liquidity_role) == "MAKER" for event in fills)
    taker_fills = len(fills) - maker_fills
    metrics: dict[str, Decimal | int | bool | str] = {
        **report,
        "sharpe_ratio": Decimal(str(sharpe)),
        "sortino_ratio": Decimal(str(sortino)),
        "calmar_ratio": calmar,
        "max_drawdown_ratio": max_drawdown_ratio,
        "ulcer_index": ulcer_index(drawdowns),
        "cvar_95": conditional_value_at_risk(returns, alpha=Decimal("0.95")),
        "return_observation_count": len(returns),
        "fill_count": len(fills),
        "maker_fill_count_derived": maker_fills,
        "taker_fill_count_derived": taker_fills,
        "turnover_notional": notional,
    }
    omega = omega_ratio(returns)
    if omega.is_finite():
        metrics["omega_ratio"] = omega
    tail = tail_ratio(returns)
    if tail.is_finite():
        metrics["tail_ratio"] = tail
    active_periods = sum(snapshot.gross_notional > ZERO for snapshot in result.snapshots)
    if result.snapshots:
        metrics["exposure_ratio"] = exposure_ratio(active_periods, len(result.snapshots))
    else:
        metrics["exposure_ratio"] = ZERO
    if result.snapshots:
        initial = result.snapshots[0].equity
        metrics["turnover_ratio"] = notional / initial if initial > ZERO else ZERO
        metrics["ending_equity"] = result.snapshots[-1].equity
    else:
        metrics["turnover_ratio"] = ZERO
        metrics["ending_equity"] = _safe_decimal(report.get("final_equity", ZERO))
    return metrics


def aggregate_metric(
    metric_sets: Iterable[dict[str, Decimal | int | bool | str]],
    key: str,
) -> Decimal:
    values = [_safe_decimal(item[key]) for item in metric_sets if key in item]
    return sum(values, ZERO) / Decimal(len(values)) if values else ZERO
