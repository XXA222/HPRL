"""Lazy public facade for the hedge execution subsystem.

Keeping this package root import-light is a correctness requirement: canonical contract
adapters import the private execution DTO module, and eager package-wide imports would
re-enter the adapter through ``planner_adapter``.  Public names remain backward
compatible and are resolved only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ActionGroupExecutor": (".action_group", "ActionGroupExecutor"),
    "ActionGroupReport": (".action_group", "ActionGroupReport"),
    "build_close_both_intents": (".action_group", "build_close_both_intents"),
    "build_close_both_plan": (".action_group", "build_close_both_plan"),
    "ActionGroupMember": (".action_group_store", "ActionGroupMember"),
    "ActionGroupMemberState": (".action_group_store", "ActionGroupMemberState"),
    "ActionGroupRecord": (".action_group_store", "ActionGroupRecord"),
    "InMemoryActionGroupRepository": (
        ".action_group_store",
        "InMemoryActionGroupRepository",
    ),
    "CancelReplaceCoordinator": (".cancel_replace", "CancelReplaceCoordinator"),
    "CancelReplaceResult": (".cancel_replace", "CancelReplaceResult"),
    "HedgeEventHubPublisher": (".event_publisher", "HedgeEventHubPublisher"),
    "InMemoryEventPublisher": (".event_publisher", "InMemoryEventPublisher"),
    "FakeHedgeAccount": (".fake_account", "FakeHedgeAccount"),
    "FakeLegPosition": (".fake_account", "FakeLegPosition"),
    "PositionAwareFakeExchange": (".fake_account", "PositionAwareFakeExchange"),
    "BinanceExecutionAdapterStub": (".fake_exchange", "BinanceExecutionAdapterStub"),
    "FakeExchangeExecutionPort": (".fake_exchange", "FakeExchangeExecutionPort"),
    "FakeExecutionHarness": (".fake_exchange", "FakeExecutionHarness"),
    "build_fake_execution_harness": (".fake_exchange", "build_fake_execution_harness"),
    "InMemoryIdempotencyStore": (".idempotency", "InMemoryIdempotencyStore"),
    "IntegratedFakeRuntime": (".integrated_fake", "IntegratedFakeRuntime"),
    "build_integrated_fake_runtime": (
        ".integrated_fake",
        "build_integrated_fake_runtime",
    ),
    "KillSwitch": (".kill_switch", "KillSwitch"),
    "InMemoryExecutionLedger": (".ledger", "InMemoryExecutionLedger"),
    "LedgerAuditRecord": (".ledger", "LedgerAuditRecord"),
    "PositionProjection": (".ledger", "PositionProjection"),
    "HedgeExecutionEngine": (".orchestrator", "HedgeExecutionEngine"),
    "OutboxDispatcher": (".outbox_dispatcher", "OutboxDispatcher"),
    "OutboxDispatchReport": (".outbox_dispatcher", "OutboxDispatchReport"),
    "OutboxStorePort": (".outbox_dispatcher", "OutboxStorePort"),
    "adapt_planner_intent": (".planner_adapter", "adapt_planner_intent"),
    "adapt_planner_intents": (".planner_adapter", "adapt_planner_intents"),
    "AllowAllRiskApproval": (".service", "AllowAllRiskApproval"),
    "ApprovedOrderIntent": (".service", "ApprovedOrderIntent"),
    "DefinitiveCancellationError": (".service", "DefinitiveCancellationError"),
    "DefinitiveExchangeOperationError": (
        ".service",
        "DefinitiveExchangeOperationError",
    ),
    "DefinitiveSubmissionError": (".service", "DefinitiveSubmissionError"),
    "ExecutionBatchReport": (".service", "ExecutionBatchReport"),
    "ExecutionBlockedError": (".service", "ExecutionBlockedError"),
    "ExecutionOrder": (".service", "ExecutionOrder"),
    "ExecutionResult": (".service", "ExecutionResult"),
    "ExecutionService": (".service", "ExecutionService"),
    "ExternalOrderSnapshot": (".service", "ExternalOrderSnapshot"),
    "IdempotencyConflictError": (".service", "IdempotencyConflictError"),
    "InMemoryAuditLog": (".service", "InMemoryAuditLog"),
    "InMemoryExecutionStore": (".service", "InMemoryExecutionStore"),
    "IntentAction": (".service", "IntentAction"),
    "OrderIntent": (".service", "OrderIntent"),
    "OrderType": (".service", "OrderType"),
    "PositionSide": (".service", "PositionSide"),
    "RiskApproval": (".service", "RiskApproval"),
    "OrderLifecycle": (".state_machine", "OrderLifecycle"),
    "OrderState": (".state_machine", "OrderState"),
    "InMemoryUserStreamOrderCache": (
        ".unknown_resolver",
        "InMemoryUserStreamOrderCache",
    ),
    "UnknownOrderResolver": (".unknown_resolver", "UnknownOrderResolver"),
    "UserStreamOrderCachePort": (
        ".unknown_resolver",
        "UserStreamOrderCachePort",
    ),
    "UserStreamOrderCacheSinkPort": (
        ".unknown_resolver",
        "UserStreamOrderCacheSinkPort",
    ),
    "UnknownOrderSupervisor": (".unknown_supervisor", "UnknownOrderSupervisor"),
    "UnknownRecoveryRecord": (".unknown_supervisor", "UnknownRecoveryRecord"),
    "UnknownRecoveryState": (".unknown_supervisor", "UnknownRecoveryState"),
    "BinanceExecutionCredentials": (".binance_usdm_adapter", "BinanceExecutionCredentials"),
    "BinanceExecutionTelemetry": (".binance_usdm_adapter", "BinanceExecutionTelemetry"),
    "BinanceUSDMExecutionAdapter": (".binance_usdm_adapter", "BinanceUSDMExecutionAdapter"),
    "ExecutionEnvironment": (".production_gate", "ExecutionEnvironment"),
    "ExecutionPermit": (".production_gate", "ExecutionPermit"),
    "ExecutionWriteLockedError": (".production_gate", "ExecutionWriteLockedError"),
    "GateSnapshot": (".production_gate", "GateSnapshot"),
    "ProductionExecutionGate": (".production_gate", "ProductionExecutionGate"),
    "ProductionGateEvidence": (".production_gate", "ProductionGateEvidence"),
    "ProductionExecutionRuntime": (".production_runtime", "ProductionExecutionRuntime"),
    "build_production_execution_runtime": (
        ".production_runtime",
        "build_production_execution_runtime",
    ),
    "ExecutionOrderOwnershipRegistry": (".ownership", "ExecutionOrderOwnershipRegistry"),
    "OrderOwnership": (".ownership", "OrderOwnership"),
    "OwnershipDecision": (".ownership", "OwnershipDecision"),
    "ExecutionUserStreamBridge": (".user_stream_bridge", "ExecutionUserStreamBridge"),
    "order_trade_update_snapshot": (
        ".user_stream_bridge",
        "order_trade_update_snapshot",
    ),
    "BinanceTestnetCredentials": (".testnet", "BinanceTestnetCredentials"),
    "GuardedTestnetRuntime": (".testnet", "GuardedTestnetRuntime"),
    "TestnetReadonlyEvidence": (".testnet", "TestnetReadonlyEvidence"),
    "TestnetSubmitCancelReport": (".testnet", "TestnetSubmitCancelReport"),
    "TESTNET_ALLOWED_SYMBOLS": (".testnet", "TESTNET_ALLOWED_SYMBOLS"),
    "TESTNET_CREDENTIAL_MARKER": (".testnet", "TESTNET_CREDENTIAL_MARKER"),
    "load_binance_testnet_credentials": (".testnet", "load_binance_testnet_credentials"),
    "build_testnet_readonly_config": (".testnet", "build_testnet_readonly_config"),
    "build_testnet_readonly_runtime": (".testnet", "build_testnet_readonly_runtime"),
    "evidence_from_ready_testnet_readonly": (
        ".testnet",
        "evidence_from_ready_testnet_readonly",
    ),
    "build_guarded_testnet_runtime": (".testnet", "build_guarded_testnet_runtime"),
    "approve_testnet_intent": (".testnet", "approve_testnet_intent"),
    "run_submit_cancel_canary": (".testnet", "run_submit_cancel_canary"),
    "make_testnet_limit_intent": (".testnet", "make_testnet_limit_intent"),
    "TestnetScenarioMode": (".testnet_e2e", "TestnetScenarioMode"),
    "TestnetScenarioState": (".testnet_e2e", "TestnetScenarioState"),
    "TestnetE2EConfig": (".testnet_e2e", "TestnetE2EConfig"),
    "TestnetE2EReport": (".testnet_e2e", "TestnetE2EReport"),
    "TestnetRunJournal": (".testnet_e2e", "TestnetRunJournal"),
    "TestnetE2EOrchestrator": (".testnet_e2e", "TestnetE2EOrchestrator"),
    "TestnetSymbolRules": (".testnet_market", "TestnetSymbolRules"),
    "TestnetCanaryOrder": (".testnet_market", "TestnetCanaryOrder"),
    "BinanceTestnetMarketProbe": (".testnet_market", "BinanceTestnetMarketProbe"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
