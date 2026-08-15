"""Reference Hedge strategies."""

from freqtrade.hedge.strategies.simple_ma_hedge import (
    SimpleDualLegMaConfig,
    SimpleDualLegMaHedgeStrategy,
)

__all__ = ["SimpleDualLegMaConfig", "SimpleDualLegMaHedgeStrategy"]
