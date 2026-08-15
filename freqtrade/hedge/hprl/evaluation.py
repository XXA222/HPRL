"""Trading-specific HPRL evaluation and walk-forward utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import fmean, pstdev
from typing import Sequence


@dataclass(frozen=True, slots=True)
class TradingMetrics:
    net_return: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    cvar: float
    turnover: float
    fees: float
    funding: float
    liquidations: int


def _max_drawdown(equity: Sequence[float]) -> float:
    peak = float(equity[0])
    maximum = 0.0
    for value in equity:
        peak = max(peak, float(value))
        if peak > 0:
            maximum = max(maximum, 1.0 - float(value) / peak)
    return maximum


def evaluate_trading(
    equity: Sequence[float],
    *,
    periods_per_year: int,
    alpha: float = 0.05,
    turnover: float = 0.0,
    fees: float = 0.0,
    funding: float = 0.0,
    liquidations: int = 0,
) -> TradingMetrics:
    values = tuple(float(value) for value in equity)
    if len(values) < 3 or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("equity curve must contain at least three positive finite values")
    if (
        not isinstance(periods_per_year, int)
        or isinstance(periods_per_year, bool)
        or periods_per_year < 1
        or not 0 < alpha <= 0.5
    ):
        raise ValueError("invalid evaluation annualization/CVaR parameters")
    if not isinstance(liquidations, int) or isinstance(liquidations, bool):
        raise ValueError("liquidations must be an integer")
    if not all(math.isfinite(value) for value in (turnover, fees, funding)):
        raise ValueError("evaluation cost and turnover inputs must be finite")
    if turnover < 0 or fees < 0 or liquidations < 0:
        raise ValueError("turnover, fees, and liquidations cannot be negative")
    returns = tuple(values[i] / values[i - 1] - 1.0 for i in range(1, len(values)))
    mean = fmean(returns)
    std = pstdev(returns)
    downside = tuple(min(value, 0.0) for value in returns)
    downside_std = math.sqrt(fmean(value * value for value in downside))
    scale = math.sqrt(periods_per_year)
    sharpe = mean / std * scale if std > 0 else 0.0
    sortino = mean / downside_std * scale if downside_std > 0 else 0.0
    max_dd = _max_drawdown(values)
    periods = len(returns)
    annualized_log = math.log(values[-1] / values[0]) * periods_per_year / periods
    annualized_log = max(-700.0, min(700.0, annualized_log))
    annualized = math.expm1(annualized_log)
    calmar = annualized / max_dd if max_dd > 0 else 0.0
    tail_count = max(1, math.ceil(len(returns) * alpha))
    cvar = max(0.0, -fmean(sorted(returns)[:tail_count]))
    return TradingMetrics(
        net_return=values[-1] / values[0] - 1.0,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        max_drawdown=max_dd,
        cvar=cvar,
        turnover=float(turnover),
        fees=float(fees),
        funding=float(funding),
        liquidations=int(liquidations),
    )


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int
    test_start: int
    test_end: int


def walk_forward_folds(
    length: int,
    *,
    train: int,
    validation: int,
    test: int,
    step: int | None = None,
    purge: int = 1,
) -> tuple[WalkForwardFold, ...]:
    """Build chronological train/validation/test folds with purge gaps.

    ``purge=1`` is the safe default for next-bar labels: the last training transition cannot use
    the first validation state as its forward realization, and the same separation is applied
    between validation and test. Increase purge for longer forward-label horizons.
    """
    integer_values = (length, train, validation, test, purge)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_values):
        raise ValueError("walk-forward dimensions and purge must be integers")
    if min(length, train, validation, test) < 1 or purge < 0:
        raise ValueError("walk-forward dimensions must be positive and purge non-negative")
    stride = test if step is None else step
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        raise ValueError("walk-forward step must be a positive integer")
    folds = []
    start = 0
    width = train + validation + test + 2 * purge
    while start + width <= length:
        train_end = start + train
        validation_start = train_end + purge
        validation_end = validation_start + validation
        test_start = validation_end + purge
        test_end = test_start + test
        folds.append(
            WalkForwardFold(
                train_start=start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
                test_start=test_start,
                test_end=test_end,
            )
        )
        start += stride
    if not folds:
        raise ValueError("dataset is too short for one walk-forward fold")
    return tuple(folds)
