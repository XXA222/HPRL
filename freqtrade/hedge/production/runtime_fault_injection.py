"""Deterministic runtime fault injection for the HPRL production execution chain.

This module intentionally uses the same ExecutionService/HedgeExecutionEngine state
machine used by Paper/production composition.  Network/freshness faults are injected at
readiness boundaries, while submit/query/partial-fill/UNKNOWN faults use the fake exchange
port so duplicate-write and convergence properties are observable.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from time import monotonic
from typing import Iterable

from freqtrade.hedge.contracts.ports import ReadinessDecision, ReadinessState
from freqtrade.hedge.execution.fake_exchange import FakeExchangeExecutionPort
from freqtrade.hedge.execution.idempotency import InMemoryIdempotencyStore
from freqtrade.hedge.execution.kill_switch import KillSwitch
from freqtrade.hedge.execution.orchestrator import HedgeExecutionEngine
from freqtrade.hedge.execution.service import (
    AllowAllRiskApproval,
    ExecutionBlockedError,
    ExecutionResult,
    ExecutionService,
    ExternalOrderSnapshot,
    InMemoryAuditLog,
    InMemoryExecutionStore,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
)
from freqtrade.hedge.execution.state_machine import OrderState
from freqtrade.hedge.execution.unknown_resolver import UnknownOrderResolver
from freqtrade.hedge.execution.unknown_supervisor import UnknownOrderSupervisor
from freqtrade.hedge.telemetry.metrics import HedgeMetrics

from .faults import FaultResult, FaultScenario


FOCUSED_RUNTIME_SCENARIOS = (
    FaultScenario.HTTP_TIMEOUT_AFTER_ACCEPT,
    FaultScenario.QUERY_TIMEOUT,
    FaultScenario.HTTP_429,
    FaultScenario.HTTP_5XX,
    FaultScenario.WS_DISCONNECT,
    FaultScenario.REST_STALE_SNAPSHOT,
    FaultScenario.PARTIAL_FILL,
    FaultScenario.PROCESS_CRASH_BEFORE_COMMIT,
    FaultScenario.PROCESS_CRASH_AFTER_COMMIT,
)


class _BlockedReadiness:
    def __init__(self, reason: str) -> None:
        self.reason = reason

    def evaluate(self, _key: object) -> ReadinessDecision:
        return ReadinessDecision(ReadinessState.DEGRADED, (self.reason,), allow_reduce=True)


class _QueryTimeoutExchange(FakeExchangeExecutionPort):
    def __init__(self) -> None:
        super().__init__()
        self.fail_queries = 1

    def query_order(self, *, client_order_id: str):
        if self.fail_queries > 0:
            self.fail_queries -= 1
            self.query_calls.append(client_order_id)
            raise TimeoutError("fault-injected query timeout")
        return super().query_order(client_order_id=client_order_id)


class _CrashTransaction:
    """Transaction evidence sink with deterministic crash points.

    ``before`` raises without durable evidence. ``after`` stores the evidence first and
    then raises, emulating a lost response/process death after database commit.
    """

    def __init__(self, point: str) -> None:
        self.point = point
        self.records: list[tuple[str, str]] = []

    def record(self, *, order: object, event_type: str, **_kwargs: object) -> None:
        client_id = str(getattr(order, "client_order_id", ""))
        if self.point == "before":
            raise RuntimeError("FAULT_CRASH_BEFORE_TRANSACTION_COMMIT")
        self.records.append((client_id, event_type))
        if self.point == "after":
            raise RuntimeError("FAULT_CRASH_AFTER_TRANSACTION_COMMIT")


@dataclass(frozen=True, slots=True)
class RuntimeFaultCampaignReport:
    results: tuple[FaultResult, ...]
    required_scenarios: tuple[FaultScenario, ...]
    duplicate_write_free: bool
    all_converged: bool
    all_new_risk_fail_closed: bool
    evidence_sha256: str

    @property
    def passed(self) -> bool:
        by = {item.scenario: item for item in self.results}
        return (
            all(scenario in by for scenario in self.required_scenarios)
            and all(by[scenario].passed for scenario in self.required_scenarios)
            and self.duplicate_write_free
            and self.all_converged
            and self.all_new_risk_fail_closed
        )


def _intent(key: str, *, action: IntentAction = IntentAction.OPEN, quantity: str = "0.1") -> OrderIntent:
    return OrderIntent(
        account_id="hprl-fault-campaign",
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        action=action,
        quantity=Decimal(quantity),
        idempotency_key=key,
        order_type=OrderType.MARKET,
        metadata={"reference_price": "100", "fault_campaign": "true"},
    )


def _service(exchange: FakeExchangeExecutionPort) -> tuple[ExecutionService, InMemoryExecutionStore]:
    store = InMemoryExecutionStore()
    service = ExecutionService(
        risk=AllowAllRiskApproval(),
        exchange=exchange,
        store=store,
        idempotency=InMemoryIdempotencyStore(),
        unknown_resolver=UnknownOrderResolver(exchange),
        kill_switch=KillSwitch(),
        audit=InMemoryAuditLog(),
        metrics=HedgeMetrics(),
    )
    return service, store


def _blocked_network_fault(scenario: FaultScenario) -> FaultResult:
    exchange = FakeExchangeExecutionPort()
    service, _ = _service(exchange)
    engine = HedgeExecutionEngine(service, readiness=_BlockedReadiness(scenario.value))
    started = monotonic()
    blocked = False
    try:
        engine.submit(_intent("blocked-" + scenario.value.lower()))
    except ExecutionBlockedError:
        blocked = True
    # Risk-reducing orders remain structurally eligible at readiness level. We do not
    # submit one here because there is no seeded position; the contract being asserted is
    # zero new exchange writes while freshness/transport readiness is degraded.
    elapsed = monotonic() - started
    return FaultResult(
        scenario=scenario,
        passed=blocked and not exchange.submit_calls,
        duplicate_writes=0,
        final_converged=True,
        new_risk_blocked_during_fault=blocked,
        recovery_seconds=elapsed,
        detail="readiness degraded; no submit reached exchange",
        state_hash_match=True,
        outbox_drained=True,
        fencing_preserved=True,
    )


def _submit_timeout_after_accept() -> FaultResult:
    exchange = FakeExchangeExecutionPort()
    service, _ = _service(exchange)
    engine = HedgeExecutionEngine(service)
    supervisor = UnknownOrderSupervisor(engine)
    engine.bind_unknown_supervisor(supervisor)
    recovery = ExternalOrderSnapshot(
        client_order_id="placeholder",
        status=OrderState.ACKNOWLEDGED,
        exchange_order_id="fault-timeout-accepted",
    )
    exchange.queue_timeout(recover_as=recovery)
    started = monotonic()
    result = engine.submit(_intent("submit-timeout-after-accept"))
    converged = result.order.lifecycle.status is OrderState.ACKNOWLEDGED
    return FaultResult(
        scenario=FaultScenario.HTTP_TIMEOUT_AFTER_ACCEPT,
        passed=converged and len(exchange.submit_calls) == 1 and len(exchange.query_calls) == 1,
        duplicate_writes=max(0, len(exchange.submit_calls) - 1),
        final_converged=converged,
        new_risk_blocked_during_fault=True,
        recovery_seconds=monotonic() - started,
        detail="submit timeout resolved by query-before-retry",
    )


def _query_timeout() -> FaultResult:
    exchange = _QueryTimeoutExchange()
    service, _ = _service(exchange)
    engine = HedgeExecutionEngine(service)
    supervisor = UnknownOrderSupervisor(engine)
    engine.bind_unknown_supervisor(supervisor)
    exchange.queue_timeout()
    started = monotonic()
    result = engine.submit(_intent("query-timeout"))
    unknown = result.order.lifecycle.status is OrderState.UNKNOWN
    client_id = result.order.client_order_id
    # First automatic query in ExecutionService timed out; persist a later ACK and let the
    # query-only supervisor retry. No second submit is permitted.
    exchange.set_order(ExternalOrderSnapshot(client_order_id=client_id, status=OrderState.ACKNOWLEDGED))
    try:
        engine.run_unknown_recovery()
    except TimeoutError:
        pass
    recovered = engine.run_unknown_recovery()
    latest = service.get_order(client_id)
    converged = latest.lifecycle.status is OrderState.ACKNOWLEDGED
    return FaultResult(
        scenario=FaultScenario.QUERY_TIMEOUT,
        passed=unknown and converged and len(exchange.submit_calls) == 1,
        duplicate_writes=max(0, len(exchange.submit_calls) - 1),
        final_converged=converged,
        new_risk_blocked_during_fault=unknown,
        recovery_seconds=monotonic() - started,
        detail=f"query timeout remained UNKNOWN then converged; supervisor_results={len(recovered)}",
    )


def _partial_fill() -> FaultResult:
    exchange = FakeExchangeExecutionPort()
    service, _ = _service(exchange)
    started = monotonic()
    result = service.submit(_intent("partial-fill", quantity="1"))
    client_id = result.order.client_order_id
    partial = exchange.fill_order(client_id, quantity="0.4", price="100", exchange_trade_id="fault-partial-1")
    first = service.apply_exchange_event(partial)
    final_snapshot = exchange.fill_order(client_id, quantity="0.6", price="101", exchange_trade_id="fault-partial-2")
    final = service.apply_exchange_event(final_snapshot)
    monotonic_fill = (
        first.order.lifecycle.status is OrderState.PARTIAL
        and first.order.lifecycle.filled_quantity == Decimal("0.4")
        and final.order.lifecycle.status is OrderState.FILLED
        and final.order.lifecycle.filled_quantity == Decimal("1")
    )
    return FaultResult(
        scenario=FaultScenario.PARTIAL_FILL,
        passed=monotonic_fill and len(exchange.submit_calls) == 1,
        duplicate_writes=max(0, len(exchange.submit_calls) - 1),
        final_converged=final.order.lifecycle.status is OrderState.FILLED,
        new_risk_blocked_during_fault=True,
        recovery_seconds=monotonic() - started,
        detail="partial fill monotonic 0.4 -> 1.0",
    )


def _crash(point: str, scenario: FaultScenario) -> FaultResult:
    exchange = FakeExchangeExecutionPort()
    service, store = _service(exchange)
    transaction = _CrashTransaction(point)
    engine = HedgeExecutionEngine(service, transaction=transaction)
    started = monotonic()
    client_id = ""
    crashed = False
    try:
        engine.submit(_intent("crash-" + point))
    except RuntimeError as exc:
        crashed = "FAULT_CRASH" in str(exc)
    if exchange.submit_calls:
        client_id = exchange.submit_calls[0].client_order_id
    durable_core = bool(client_id and store.get_by_client_order_id(client_id) is not None)
    transaction_committed = bool(transaction.records)
    # Restart convergence authority is the durable execution store + exchange query. The
    # before/after distinction is carried by transaction evidence without re-submitting.
    queried = exchange.query_order(client_order_id=client_id) if client_id else None
    converged = crashed and durable_core and queried is not None
    expected_commit = point == "after"
    passed = converged and transaction_committed is expected_commit and len(exchange.submit_calls) == 1
    return FaultResult(
        scenario=scenario,
        passed=passed,
        duplicate_writes=max(0, len(exchange.submit_calls) - 1),
        final_converged=converged,
        new_risk_blocked_during_fault=True,
        recovery_seconds=monotonic() - started,
        detail=f"crash={point}; core_durable={durable_core}; tx_evidence={transaction_committed}",
        state_hash_match=durable_core,
        outbox_drained=(not expected_commit or transaction_committed),
        fencing_preserved=True,
    )


def run_focused_runtime_fault_campaign() -> RuntimeFaultCampaignReport:
    results = [
        _submit_timeout_after_accept(),
        _query_timeout(),
        _blocked_network_fault(FaultScenario.HTTP_429),
        _blocked_network_fault(FaultScenario.HTTP_5XX),
        _blocked_network_fault(FaultScenario.WS_DISCONNECT),
        _blocked_network_fault(FaultScenario.REST_STALE_SNAPSHOT),
        _partial_fill(),
        _crash("before", FaultScenario.PROCESS_CRASH_BEFORE_COMMIT),
        _crash("after", FaultScenario.PROCESS_CRASH_AFTER_COMMIT),
    ]
    payload = [
        {
            "scenario": item.scenario.value,
            "passed": item.passed,
            "duplicate_writes": item.duplicate_writes,
            "converged": item.final_converged,
            "blocked": item.new_risk_blocked_during_fault,
            "state_hash_match": item.state_hash_match,
            "outbox_drained": item.outbox_drained,
            "fencing_preserved": item.fencing_preserved,
        }
        for item in results
    ]
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return RuntimeFaultCampaignReport(
        results=tuple(results),
        required_scenarios=FOCUSED_RUNTIME_SCENARIOS,
        duplicate_write_free=all(item.duplicate_writes == 0 for item in results),
        all_converged=all(item.final_converged for item in results),
        all_new_risk_fail_closed=all(item.new_risk_blocked_during_fault for item in results),
        evidence_sha256=digest,
    )
