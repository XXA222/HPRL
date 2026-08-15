from __future__ import annotations

import random
from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.simulation.exchange import SimulationResult

from .decimal_utils import ONE, ZERO
from .metrics import equity_returns


DEFAULT_RUIN_EQUITY_RATIO = Decimal("0.5")


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    paths: int
    horizon: int
    median_return: Decimal
    percentile_05_return: Decimal
    percentile_95_return: Decimal
    probability_of_loss: Decimal
    probability_of_ruin: Decimal
    median_max_drawdown: Decimal


def _percentile(values: list[Decimal], quantile: Decimal) -> Decimal:
    if not values:
        return ZERO
    ordered = sorted(values)
    position = int((Decimal(len(ordered) - 1) * quantile).to_integral_value())
    return ordered[position]


def _sample_blocks(
    returns: tuple[Decimal, ...],
    *,
    horizon: int,
    block_size: int,
    rng: random.Random,
) -> tuple[Decimal, ...]:
    output: list[Decimal] = []
    while len(output) < horizon:
        start = rng.randrange(len(returns))
        for offset in range(block_size):
            output.append(returns[(start + offset) % len(returns)])
            if len(output) == horizon:
                break
    return tuple(output)


def monte_carlo_equity(
    result: SimulationResult,
    *,
    paths: int = 1000,
    horizon: int | None = None,
    block_size: int = 5,
    ruin_equity_ratio: Decimal = DEFAULT_RUIN_EQUITY_RATIO,
    seed: int = 42,
) -> MonteCarloResult:
    returns = equity_returns(result)
    if not returns:
        raise ValueError("Monte Carlo requires at least one equity return")
    if paths < 1 or block_size < 1:
        raise ValueError("paths and block_size must be positive")
    actual_horizon = horizon or len(returns)
    if actual_horizon < 1:
        raise ValueError("horizon must be positive")
    if not ZERO < ruin_equity_ratio < ONE:
        raise ValueError("ruin_equity_ratio must be in (0, 1)")
    rng = random.Random(seed)  # noqa: S311 - deterministic research sampling; not cryptographic
    terminal_returns: list[Decimal] = []
    drawdowns: list[Decimal] = []
    loss_count = 0
    ruin_count = 0
    for _ in range(paths):
        equity = ONE
        peak = ONE
        max_drawdown = ZERO
        ruined = False
        for item in _sample_blocks(
            returns,
            horizon=actual_horizon,
            block_size=block_size,
            rng=rng,
        ):
            equity *= ONE + item
            peak = max(peak, equity)
            if peak > ZERO:
                max_drawdown = max(max_drawdown, (peak - equity) / peak)
            if equity <= ruin_equity_ratio:
                ruined = True
        terminal = equity - ONE
        terminal_returns.append(terminal)
        drawdowns.append(max_drawdown)
        loss_count += terminal < ZERO
        ruin_count += ruined
    return MonteCarloResult(
        paths=paths,
        horizon=actual_horizon,
        median_return=_percentile(terminal_returns, Decimal("0.50")),
        percentile_05_return=_percentile(terminal_returns, Decimal("0.05")),
        percentile_95_return=_percentile(terminal_returns, Decimal("0.95")),
        probability_of_loss=Decimal(loss_count) / Decimal(paths),
        probability_of_ruin=Decimal(ruin_count) / Decimal(paths),
        median_max_drawdown=_percentile(drawdowns, Decimal("0.50")),
    )
