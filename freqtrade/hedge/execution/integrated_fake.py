"""One-call integrated Fake runtime with execution gates, ledger and event publication."""

from __future__ import annotations

from dataclasses import dataclass

from freqtrade.hedge.contracts.ports import (
    AlwaysReadyGate,
    EventPublisherPort,
    ExecutionTransactionPort,
    InMemoryPositionLock,
    InMemorySingleWriter,
    MarketRulesPort,
    PositionLockPort,
    ReadinessGatePort,
    SingleWriterPort,
    StaticMarketRules,
)
from freqtrade.hedge.telemetry.metrics import HedgeMetrics

from .action_group_store import (
    ActionGroupRepository,
    InMemoryActionGroupRepository,
)
from .event_publisher import InMemoryEventPublisher
from .fake_account import FakeHedgeAccount, PositionAwareFakeExchange
from .idempotency import IdempotencyPort, InMemoryIdempotencyStore
from .kill_switch import KillSwitch
from .ledger import InMemoryExecutionLedger
from .orchestrator import HedgeExecutionEngine
from .service import (
    AllowAllRiskApproval,
    ExecutionResult,
    ExecutionService,
    ExecutionStorePort,
    InMemoryAuditLog,
    InMemoryExecutionStore,
    RiskApprovalPort,
)
from .unknown_resolver import InMemoryUserStreamOrderCache, UnknownOrderResolver


@dataclass(frozen=True, slots=True)
class IntegratedFakeRuntime:
    engine: HedgeExecutionEngine
    core: ExecutionService
    exchange: PositionAwareFakeExchange
    account: FakeHedgeAccount
    store: ExecutionStorePort
    idempotency: IdempotencyPort[ExecutionResult]
    ledger: ExecutionTransactionPort
    publisher: EventPublisherPort
    readiness: ReadinessGatePort
    single_writer: SingleWriterPort
    position_lock: PositionLockPort
    market_rules: MarketRulesPort
    action_groups: ActionGroupRepository
    kill_switch: KillSwitch
    audit: InMemoryAuditLog
    metrics: HedgeMetrics
    user_stream_cache: InMemoryUserStreamOrderCache


def build_integrated_fake_runtime(
    *,
    risk: RiskApprovalPort | None = None,
    publisher: EventPublisherPort | None = None,
    readiness: ReadinessGatePort | None = None,
    single_writer: SingleWriterPort | None = None,
    position_lock: PositionLockPort | None = None,
    market_rules: MarketRulesPort | None = None,
    transaction: ExecutionTransactionPort | None = None,
    action_groups: ActionGroupRepository | None = None,
    store: ExecutionStorePort | None = None,
    idempotency: IdempotencyPort[ExecutionResult] | None = None,
    fee_rate: object = None,
    strict_dependencies: bool = False,
) -> IntegratedFakeRuntime:
    from decimal import Decimal

    if strict_dependencies:
        required = {
            "risk": risk,
            "publisher": publisher,
            "readiness": readiness,
            "single_writer": single_writer,
            "position_lock": position_lock,
            "market_rules": market_rules,
            "transaction": transaction,
            "action_groups": action_groups,
            "store": store,
            "idempotency": idempotency,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError(
                "strict integrated Paper runtime requires explicit dependencies: "
                + ", ".join(missing)
            )
    normalized_fee = Decimal("0") if fee_rate is None else Decimal(str(fee_rate))
    account = FakeHedgeAccount(fee_rate=normalized_fee)
    exchange = PositionAwareFakeExchange(account)
    execution_store = store or InMemoryExecutionStore()
    idempotency_store = idempotency or InMemoryIdempotencyStore()
    user_stream_cache = InMemoryUserStreamOrderCache()
    resolver = UnknownOrderResolver(exchange, user_stream_cache=user_stream_cache)
    kill_switch = KillSwitch()
    audit = InMemoryAuditLog()
    metrics = HedgeMetrics()
    core = ExecutionService(
        risk=risk or AllowAllRiskApproval(),
        exchange=exchange,
        store=execution_store,
        idempotency=idempotency_store,
        unknown_resolver=resolver,
        kill_switch=kill_switch,
        audit=audit,
        metrics=metrics,
    )
    ledger = (
        transaction
        if isinstance(transaction, InMemoryExecutionLedger)
        else InMemoryExecutionLedger()
    )
    event_publisher = publisher or InMemoryEventPublisher()
    readiness_port = readiness or AlwaysReadyGate()
    single_writer_port = single_writer or InMemorySingleWriter()
    position_lock_port = position_lock or InMemoryPositionLock()
    market_rules_port = market_rules or StaticMarketRules()
    engine = HedgeExecutionEngine(
        core,
        readiness=readiness_port,
        single_writer=single_writer_port,
        position_lock=position_lock_port,
        market_rules=market_rules_port,
        transaction=transaction or ledger,
        publisher=event_publisher,
        user_stream_cache=user_stream_cache,
        strict_dependencies=strict_dependencies,
    )
    return IntegratedFakeRuntime(
        engine=engine,
        core=core,
        exchange=exchange,
        account=account,
        store=execution_store,
        idempotency=idempotency_store,
        ledger=transaction or ledger,
        publisher=event_publisher,
        readiness=readiness_port,
        single_writer=single_writer_port,
        position_lock=position_lock_port,
        market_rules=market_rules_port,
        action_groups=action_groups or InMemoryActionGroupRepository(),
        kill_switch=kill_switch,
        audit=audit,
        metrics=metrics,
        user_stream_cache=user_stream_cache,
    )
