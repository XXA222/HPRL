"""Public integration API with lazy imports.

Keeping the package root lazy prevents optional Freqtrade/persistence dependencies from
being imported merely to use the standalone planner, risk or Paper components.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "IntegratedFakeHedgeApplication": (".application", "IntegratedFakeHedgeApplication"),
    "PlanningExecutionResult": (".application", "PlanningExecutionResult"),
    "HedgeCompositionRoot": (".composition", "HedgeCompositionRoot"),
    "build_hedge_composition": (".composition", "build_hedge_composition"),
    "AsyncLoopThread": (".coordinator", "AsyncLoopThread"),
    "HedgeRuntimeCoordinator": (".coordinator", "HedgeRuntimeCoordinator"),
    "IntegratedPaperHedgeApplication": (".paper_runtime", "IntegratedPaperHedgeApplication"),
    "PaperCycleResult": (".paper_runtime", "PaperCycleResult"),
    "JsonPaperStateStore": (".paper_state", "JsonPaperStateStore"),
    "NullPaperStateStore": (".paper_state", "NullPaperStateStore"),
    "PaperStateStore": (".paper_state", "PaperStateStore"),
    "CentralRuntimeProjection": (".projection", "CentralRuntimeProjection"),
    "build_central_projection": (".projection", "build_central_projection"),
    "InMemoryReadonlyRepository": (".repository", "InMemoryReadonlyRepository"),
    "PersistenceMirroringReadonlyRepository": (
        ".repository",
        "PersistenceMirroringReadonlyRepository",
    ),
    "PortfolioRiskApprovalAdapter": (".risk_adapter", "PortfolioRiskApprovalAdapter"),
    "RuntimeRiskApprovalAdapter": (".risk_adapter", "RuntimeRiskApprovalAdapter"),
    "FreqtradeStrategySignalProvider": (
        ".signal_provider",
        "FreqtradeStrategySignalProvider",
    ),
    "HedgeSignalProviderPort": (".signal_provider", "HedgeSignalProviderPort"),
    "SignalSnapshot": (".signal_provider", "SignalSnapshot"),
    "MarketRuleSnapshot": (".market_data", "MarketRuleSnapshot"),
    "build_market_snapshot": (".market_data", "build_market_snapshot"),
    "HedgeController": (".controller", "HedgeController"),
    "HedgeControllerCycle": (".controller", "HedgeControllerCycle"),
    "ExecutionEngineKind": (".production_main_loop", "ExecutionEngineKind"),
    "HedgeExecutionMode": (".production_main_loop", "HedgeExecutionMode"),
    "HedgeMainLoopCycle": (".production_main_loop", "HedgeMainLoopCycle"),
    "ProductionEquivalentHedgeMainLoop": (
        ".production_main_loop",
        "ProductionEquivalentHedgeMainLoop",
    ),
    "RecoveryReport": (".production_main_loop", "RecoveryReport"),
    "EmergencyStopReport": (".production_main_loop", "EmergencyStopReport"),
    "ProductionMainLoopConfig": (".main_loop_config", "ProductionMainLoopConfig"),
    "production_main_loop_config_from_mapping": (
        ".main_loop_config",
        "production_main_loop_config_from_mapping",
    ),
    "ProductionMainLoopAssembly": (
        ".production_assembly",
        "ProductionMainLoopAssembly",
    ),
    "build_production_main_loop_assembly": (
        ".production_assembly",
        "build_production_main_loop_assembly",
    ),
    "PlanningContextEvidence": (".production_context", "PlanningContextEvidence"),
    "BuiltPlanningContext": (".production_context", "BuiltPlanningContext"),
    "ReadonlyPlanningContextBuilder": (
        ".production_context",
        "ReadonlyPlanningContextBuilder",
    ),
    "ProductionControllerCycle": (
        ".production_controller",
        "ProductionControllerCycle",
    ),
    "ProductionHedgeController": (
        ".production_controller",
        "ProductionHedgeController",
    ),
    "InMemoryStrategyStateStore": (".strategy_state", "InMemoryStrategyStateStore"),
    "JsonStrategyStateStore": (".strategy_state", "JsonStrategyStateStore"),
    "SqlStrategyStateStore": (".strategy_state", "SqlStrategyStateStore"),
    "StrategyStateStorePort": (".strategy_state", "StrategyStateStorePort"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
