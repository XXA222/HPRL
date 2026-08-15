"""Composition helper for the R3.2 production-equivalent main loop."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Mapping

from freqtrade.hedge.execution.integrated_fake import (
    IntegratedFakeRuntime,
    build_integrated_fake_runtime,
)
from freqtrade.hedge.execution.ownership import ExecutionOrderOwnershipRegistry
from freqtrade.hedge.planning.context import PlannerConfig
from freqtrade.hedge.integration.production_main_loop import (
    ExecutionEngineKind,
    HedgeExecutionMode,
    ProductionEquivalentHedgeMainLoop,
)
from freqtrade.hedge.integration.strategy_state import (
    InMemoryStrategyStateStore,
    JsonStrategyStateStore,
    SqlStrategyStateStore,
    StrategyStateStorePort,
)
from freqtrade.hedge.symbols import canonicalize_symbol

if TYPE_CHECKING:
    from .paper_runtime import IntegratedPaperHedgeApplication

from .main_loop_config import (
    ProductionMainLoopConfig,
    production_main_loop_config_from_mapping,
)
from .production_context import ReadonlyPlanningContextBuilder


@dataclass(frozen=True, slots=True)
class ProductionMainLoopAssembly:
    config: ProductionMainLoopConfig
    loop: ProductionEquivalentHedgeMainLoop
    context_builder: ReadonlyPlanningContextBuilder
    execution_runtime: object
    exchange_write_surface: str = "NONE"
    cycle_owner: str = "PRODUCTION_CONTROLLER"


def build_production_main_loop_assembly(
    *,
    config: Mapping[str, Any],
    session_factory: object,
    paper_application: "IntegratedPaperHedgeApplication | None" = None,
    production_runtime: object | None = None,
) -> ProductionMainLoopAssembly | None:
    loop_config = production_main_loop_config_from_mapping(config)
    if not loop_config.enabled:
        return None
    hedge = config.get("hedge", {})
    planner_values = hedge.get("planner", {}) if isinstance(hedge, Mapping) else {}
    if not isinstance(planner_values, Mapping):
        planner_values = {}
    planner_config = _planner_config_from_mapping(planner_values)
    pair = canonicalize_symbol(str(config.get("managed_pair", "")))
    account_id = str(hedge.get("account_id", "default")).strip() if isinstance(hedge, Mapping) else "default"
    strategy_id = "pure-hedge-planner"
    state_store = _state_store(
        loop_config,
        session_factory=session_factory,
        account_id=account_id,
        symbol=pair,
        strategy_id=strategy_id,
    )
    if loop_config.mode is HedgeExecutionMode.HEDGE_SIMULATED:
        if paper_application is None or paper_application.execution is None:
            raise ValueError(
                "HEDGE_SIMULATED requires the durable IntegratedPaperHedgeApplication"
            )
        if paper_application.account_id != account_id:
            raise ValueError("Paper account_id does not match hedge.main_loop account_id")
        runtime = paper_application.execution
        engine_kind = ExecutionEngineKind.SIMULATED
        exchange_write_surface = "SIMULATED"
    else:
        if production_runtime is not None:
            runtime = production_runtime
            if not all(hasattr(runtime, name) for name in ("engine", "store", "kill_switch")):
                raise TypeError(
                    "production_runtime must expose engine, store, and kill_switch"
                )
            exchange_write_surface = (
                "BINANCE_USDM_AUTHORITATIVE"
                if loop_config.mode is HedgeExecutionMode.HEDGE_PRODUCTION_ARMED
                else "LOCKED_AUTHORITATIVE"
            )
        else:
            if loop_config.mode is HedgeExecutionMode.HEDGE_PRODUCTION_ARMED:
                raise ValueError(
                    "HEDGE_PRODUCTION_ARMED requires the authoritative R5 runtime"
                )
            # Legacy locked planning-only compatibility. This graph can never write and
            # is not accepted for HEDGE_PRODUCTION_ARMED.
            runtime = build_integrated_fake_runtime()
            exchange_write_surface = "NONE"
        engine_kind = ExecutionEngineKind.PRODUCTION
    loop = ProductionEquivalentHedgeMainLoop(
        account_id=account_id,
        engine=runtime.engine,
        ownership=ExecutionOrderOwnershipRegistry(runtime.store),
        kill_switch=runtime.kill_switch,
        mode=loop_config.mode,
        engine_kind=engine_kind,
        strategy_id=strategy_id,
        allowed_symbols=loop_config.allowed_symbols,
        state_store=state_store,
        max_submissions_per_cycle=loop_config.max_submissions_per_cycle,
        max_cancellations_per_cycle=loop_config.max_cancellations_per_cycle,
        block_new_risk_on_external_side=loop_config.block_new_risk_on_external_side,
    )
    if paper_application is not None and loop_config.mode is HedgeExecutionMode.HEDGE_SIMULATED:
        paper_application.bind_new_risk_provider(lambda: loop.new_risk_enabled)
    return ProductionMainLoopAssembly(
        config=loop_config,
        loop=loop,
        context_builder=ReadonlyPlanningContextBuilder(
            planner_config=planner_config,
            allowed_symbols=loop_config.allowed_symbols,
        ),
        execution_runtime=runtime,
        exchange_write_surface=exchange_write_surface,
        cycle_owner=(
            "PAPER_APPLICATION"
            if loop_config.mode is HedgeExecutionMode.HEDGE_SIMULATED
            else "PRODUCTION_CONTROLLER"
        ),
    )


def _state_store(
    config: ProductionMainLoopConfig,
    *,
    session_factory: object,
    account_id: str,
    symbol: str,
    strategy_id: str,
) -> StrategyStateStorePort:
    if config.state_backend == "memory":
        return InMemoryStrategyStateStore()
    if config.state_backend == "json":
        path = Path(str(config.state_path)).expanduser()
        return JsonStrategyStateStore(path)
    return SqlStrategyStateStore(
        session_factory,
        exchange="binance",
        account_id=account_id,
        symbol=symbol,
        strategy_name=strategy_id,
    )


def _planner_config_from_mapping(values: Mapping[str, Any]) -> PlannerConfig:
    raw = dict(values)
    aliases = {
        "qty_scale": "grid_qty_growth",
        "grid_initial_distance": "trailing_trigger_distance",
    }
    for old_name, new_name in aliases.items():
        if old_name not in raw:
            continue
        if new_name in raw and raw[new_name] != raw[old_name]:
            raise ValueError(
                f"hedge.planner.{old_name} conflicts with hedge.planner.{new_name}"
            )
        raw[new_name] = raw.pop(old_name)
    fields = PlannerConfig.__dataclass_fields__
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError("unknown hedge.planner option(s): " + ", ".join(unknown))
    converted: dict[str, object] = {}
    for name, value in raw.items():
        default = fields[name].default
        if isinstance(default, Decimal):
            if isinstance(value, bool):
                raise ValueError(f"hedge.planner.{name} must be an exact decimal")
            converted[name] = Decimal(str(value))
        elif isinstance(default, int) and not isinstance(default, bool):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"hedge.planner.{name} must be an integer")
            converted[name] = value
        elif isinstance(default, bool):
            if not isinstance(value, bool):
                raise ValueError(f"hedge.planner.{name} must be a boolean")
            converted[name] = value
        else:
            converted[name] = value
    return PlannerConfig(**converted)
