"""Composition root for the production-capable hedge execution graph."""

from __future__ import annotations

from dataclasses import dataclass

from freqtrade.hedge.contracts.ports import (
    EventPublisherPort,
    ExecutionTransactionPort,
    MarketRulesPort,
    PositionLockPort,
    ReadinessGatePort,
    SingleWriterPort,
)
from freqtrade.hedge.telemetry.metrics import HedgeMetrics

from .action_group import ActionGroupExecutor
from .action_group_store import ActionGroupRepository
from .binance_usdm_adapter import (
    BinanceExecutionCredentials,
    BinanceUSDMExecutionAdapter,
    HttpTransport,
)
from .cancel_replace import CancelReplaceCoordinator
from .event_publisher import InMemoryEventPublisher
from .idempotency import IdempotencyPort
from .kill_switch import KillSwitch
from .orchestrator import HedgeExecutionEngine
from .production_gate import ProductionExecutionGate
from .service import (
    ExecutionResult,
    ExecutionService,
    ExecutionStorePort,
    AuditPort,
    InMemoryAuditLog,
    RiskApprovalPort,
)
from .unknown_resolver import InMemoryUserStreamOrderCache, UnknownOrderResolver
from .user_stream_bridge import ExecutionUserStreamBridge


@dataclass(frozen=True, slots=True)
class ProductionExecutionRuntime:
    engine: HedgeExecutionEngine
    core: ExecutionService
    exchange: BinanceUSDMExecutionAdapter
    gate: ProductionExecutionGate
    kill_switch: KillSwitch
    user_stream_bridge: ExecutionUserStreamBridge
    user_stream_cache: InMemoryUserStreamOrderCache
    cancel_replace: CancelReplaceCoordinator
    action_groups: ActionGroupExecutor
    audit: AuditPort
    metrics: HedgeMetrics

    def bind_user_stream(self, user_stream: object) -> None:
        setter = getattr(user_stream, "set_execution_order_event_callback", None)
        if not callable(setter):
            raise TypeError(
                "user_stream must expose set_execution_order_event_callback"
            )
        setter(self.user_stream_bridge.handle)


def build_production_execution_runtime(
    *,
    credentials: BinanceExecutionCredentials,
    gate: ProductionExecutionGate,
    risk: RiskApprovalPort,
    store: ExecutionStorePort,
    idempotency: IdempotencyPort[ExecutionResult],
    readiness: ReadinessGatePort,
    single_writer: SingleWriterPort,
    position_lock: PositionLockPort,
    market_rules: MarketRulesPort,
    transaction: ExecutionTransactionPort,
    publisher: EventPublisherPort | None,
    action_group_repository: ActionGroupRepository,
    proxy_url: str | None = None,
    base_url: str | None = None,
    transport: HttpTransport | None = None,
    user_stream: object | None = None,
    audit: AuditPort | None = None,
) -> ProductionExecutionRuntime:
    required = {
        "risk": risk,
        "store": store,
        "idempotency": idempotency,
        "readiness": readiness,
        "single_writer": single_writer,
        "position_lock": position_lock,
        "market_rules": market_rules,
        "transaction": transaction,
        "action_group_repository": action_group_repository,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    if missing:
        raise ValueError("production runtime requires explicit dependencies: " + ", ".join(missing))
    exchange = BinanceUSDMExecutionAdapter(
        credentials=credentials,
        gate=gate,
        store=store,
        proxy_url=proxy_url,
        base_url=base_url,
        transport=transport,
    )
    user_stream_cache = InMemoryUserStreamOrderCache()
    resolver = UnknownOrderResolver(exchange, user_stream_cache=user_stream_cache)
    kill_switch = KillSwitch(allow_risk_reduction_while_halted=True)
    audit_log: AuditPort = audit or InMemoryAuditLog()
    metrics = HedgeMetrics()
    core = ExecutionService(
        risk=risk,
        exchange=exchange,
        store=store,
        idempotency=idempotency,
        unknown_resolver=resolver,
        kill_switch=kill_switch,
        audit=audit_log,
        metrics=metrics,
    )
    event_publisher = publisher or InMemoryEventPublisher()
    engine = HedgeExecutionEngine(
        core,
        readiness=readiness,
        single_writer=single_writer,
        position_lock=position_lock,
        market_rules=market_rules,
        transaction=transaction,
        publisher=event_publisher,
        exchange="binance",
        user_stream_cache=user_stream_cache,
        strict_dependencies=True,
    )
    bridge = ExecutionUserStreamBridge(
        engine=engine,
        cache=user_stream_cache,
        allowed_symbols=gate.evidence.allowed_symbols,
    )
    runtime = ProductionExecutionRuntime(
        engine=engine,
        core=core,
        exchange=exchange,
        gate=gate,
        kill_switch=kill_switch,
        user_stream_bridge=bridge,
        user_stream_cache=user_stream_cache,
        cancel_replace=CancelReplaceCoordinator(core),
        action_groups=ActionGroupExecutor(engine, action_group_repository),
        audit=audit_log,
        metrics=metrics,
    )
    if user_stream is not None:
        runtime.bind_user_stream(user_stream)
    return runtime
