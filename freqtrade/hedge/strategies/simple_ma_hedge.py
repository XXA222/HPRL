"""Simple dual-leg moving-average strategy for Hedge backtesting.

This strategy is intentionally small and deterministic.  It always keeps the
configured LONG and SHORT core allocations, while a mean-reversion signal tilts
only the tactical allocation.  Signals for a bar are calculated exclusively
from closes of earlier bars, so the helper does not introduce look-ahead bias.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.simulation.exchange import BarEvent, SignalEvent, SimulationInputEvent

ZERO = Decimal("0")
ONE = Decimal("1")


def _bounded(value: Decimal) -> Decimal:
    return max(ZERO, min(ONE, value))


@dataclass(frozen=True, slots=True)
class SimpleDualLegMaConfig:
    """Parameters for the tactical mean-reversion signal."""

    fast_window: int = 5
    slow_window: int = 20
    base_signal: Decimal = Decimal("0.20")
    sensitivity: Decimal = Decimal("18")

    def __post_init__(self) -> None:
        if self.fast_window <= 0:
            raise ValueError("fast_window must be positive")
        if self.slow_window <= self.fast_window:
            raise ValueError("slow_window must be greater than fast_window")
        if not self.base_signal.is_finite() or not self.sensitivity.is_finite():
            raise ValueError("strategy parameters must be finite")
        if not ZERO <= self.base_signal <= ONE:
            raise ValueError("base_signal must be between zero and one")
        if self.sensitivity < ZERO:
            raise ValueError("sensitivity cannot be negative")


class SimpleDualLegMaHedgeStrategy:
    """Generate independent LONG and SHORT tactical signals.

    The core allocations are controlled by :class:`PlannerConfig` and remain on
    both sides.  This class only changes tactical demand:

    * price below the slow average strengthens LONG tactical demand;
    * price above the slow average strengthens SHORT tactical demand;
    * both sides retain ``base_signal`` around the mean.
    """

    def __init__(self, config: SimpleDualLegMaConfig | None = None) -> None:
        self.config = config or SimpleDualLegMaConfig()

    def signal(self, prior_closes: Sequence[Decimal]) -> tuple[Decimal, Decimal]:
        """Return ``(long_signal, short_signal)`` from earlier closes only."""

        cfg = self.config
        if len(prior_closes) < cfg.slow_window:
            return cfg.base_signal, cfg.base_signal

        fast = sum(prior_closes[-cfg.fast_window :], ZERO) / Decimal(cfg.fast_window)
        slow = sum(prior_closes[-cfg.slow_window :], ZERO) / Decimal(cfg.slow_window)
        if slow <= ZERO:
            raise ValueError("slow moving average must be positive")

        deviation = (fast - slow) / slow
        long_signal = _bounded(cfg.base_signal - deviation * cfg.sensitivity)
        short_signal = _bounded(cfg.base_signal + deviation * cfg.sensitivity)
        return long_signal, short_signal

    def events(self, bars: Iterable[BarEvent]) -> Iterator[SimulationInputEvent]:
        """Yield a signal immediately before each bar without look-ahead."""

        prior_closes: list[Decimal] = []
        symbol: str | None = None
        previous_timestamp = None
        for bar in bars:
            if symbol is None:
                symbol = bar.symbol
            elif bar.symbol != symbol:
                raise ValueError("SimpleDualLegMaHedgeStrategy supports one symbol per run")
            if previous_timestamp is not None and bar.timestamp <= previous_timestamp:
                raise ValueError("bars must be strictly chronological")

            long_signal, short_signal = self.signal(prior_closes)
            yield SignalEvent(
                timestamp=bar.timestamp,
                symbol=bar.symbol,
                long_signal=long_signal,
                short_signal=short_signal,
            )
            yield bar
            prior_closes.append(bar.close)
            previous_timestamp = bar.timestamp
