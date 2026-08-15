"""Canonical public surface for the Freqtrade Hedge clean mainline.

Versioned development packages and deprecated Binance compatibility facades are
intentionally not exported here.  Runtime code should import the canonical
subpackages directly when it needs exchange-, execution-, or persistence-
specific behavior.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "HedgeRuntimeConfig": ("freqtrade.hedge.config", "HedgeRuntimeConfig"),
    "normalize_hedge_config": ("freqtrade.hedge.config", "normalize_hedge_config"),
    "validate_hedge_config": ("freqtrade.hedge.config", "validate_hedge_config"),
    "HedgeAction": ("freqtrade.hedge.domain", "HedgeAction"),
    "HedgeActionPlan": ("freqtrade.hedge.domain", "HedgeActionPlan"),
    "PositionKey": ("freqtrade.hedge.domain", "PositionKey"),
    "PositionRecord": ("freqtrade.hedge.position_book", "PositionRecord"),
    "SideAwarePositionBook": ("freqtrade.hedge.position_book", "SideAwarePositionBook"),
    "ReconciliationIssue": ("freqtrade.hedge.reconciliation", "ReconciliationIssue"),
    "ReconciliationResult": ("freqtrade.hedge.reconciliation", "ReconciliationResult"),
    "reconcile_positions": ("freqtrade.hedge.reconciliation", "reconcile_positions"),
    "HedgeConfigurationError": ("freqtrade.hedge.errors", "HedgeConfigurationError"),
    "HedgeDataError": ("freqtrade.hedge.errors", "HedgeDataError"),
    "HedgeError": ("freqtrade.hedge.errors", "HedgeError"),
    "HedgeInvariantError": ("freqtrade.hedge.errors", "HedgeInvariantError"),
    "HedgeSafetyError": ("freqtrade.hedge.errors", "HedgeSafetyError"),
    "ReduceOnlyDecision": ("freqtrade.hedge.local_reduce_only", "ReduceOnlyDecision"),
    "calculate_safe_reduce": ("freqtrade.hedge.local_reduce_only", "calculate_safe_reduce"),
    "canonicalize_symbol": ("freqtrade.hedge.symbols", "canonicalize_symbol"),
    "raw_symbol": ("freqtrade.hedge.symbols", "raw_symbol"),
    "symbols_equivalent": ("freqtrade.hedge.symbols", "symbols_equivalent"),
    "AccountRiskSnapshot": ("freqtrade.hedge.risk", "AccountRiskSnapshot"),
    "HedgeRiskEngine": ("freqtrade.hedge.risk", "HedgeRiskEngine"),
    "RiskDecision": ("freqtrade.hedge.risk", "RiskDecision"),
    "RiskLimits": ("freqtrade.hedge.risk", "RiskLimits"),
    "StrategyContract": ("freqtrade.hedge.strategies.contract", "StrategyContract"),
    "StrategyDirective": ("freqtrade.hedge.strategies.contract", "StrategyDirective"),
    "HEDGE_SIGNAL_COLUMNS": ("freqtrade.hedge.strategies.contract", "HEDGE_SIGNAL_COLUMNS"),
    "SimpleDualLegMaConfig": (
        "freqtrade.hedge.strategies.simple_ma_hedge",
        "SimpleDualLegMaConfig",
    ),
    "SimpleDualLegMaHedgeStrategy": (
        "freqtrade.hedge.strategies.simple_ma_hedge",
        "SimpleDualLegMaHedgeStrategy",
    ),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
