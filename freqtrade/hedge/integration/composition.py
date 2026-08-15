"""Authoritative composition root for the integrated hedge subsystem.

This module is the only production assembly point for Readiness, single-writer
fencing, side locks, direction-three risk, Paper simulation and Binance read-only
projection.  It intentionally rejects ``combined`` mode until real and simulated
account projections have separate public runtime views.
"""

from __future__ import annotations

import logging
import socket
from contextlib import AbstractContextManager
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping

from freqtrade.exceptions import OperationalException
from freqtrade.hedge.concurrency.database_lease import SqlAlchemyDatabaseLeaseStore
from freqtrade.hedge.concurrency.position_lock import PositionLockManager
from freqtrade.hedge.concurrency.single_writer import SingleWriterGuard
from freqtrade.hedge.control.assembly import build_hedge_control_service
from freqtrade.hedge.control.service import HedgeControlService
from freqtrade.hedge.contracts.ports import (
    MarketRules,
    PositionKey,
    ReadinessDecision,
    ReadinessState as ExecutionReadinessState,
    StaticMarketRules,
)
from freqtrade.hedge.identity import RiskPositionKey
from freqtrade.hedge.readiness.checks import ReadinessInputs
from freqtrade.hedge.readiness.state import ReadinessState
from freqtrade.hedge.risk.commit import SqlRiskApprovalCommitStore
from freqtrade.hedge.risk.limits import RiskLimits
from freqtrade.hedge.risk.runtime import HedgeRiskRuntime, build_hedge_risk_runtime
from freqtrade.hedge.runtime import HedgeRuntime
from freqtrade.hedge.symbols import raw_symbol
from freqtrade.hedge.safety import assert_supported_operation_mode
from freqtrade.persistence.hedge_execution_adapters import (
    SqlActionGroupRepository,
    SqlExecutionIdempotencyStore,
    SqlExecutionLedger,
    SqlExecutionStore,
)
from freqtrade.persistence.hedge_service import HedgePersistenceService

from .coordinator import HedgeRuntimeCoordinator
from .paper_events import SqlPaperAccountEventSink, SqlPaperExecutionRecovery
from .paper_runtime import IntegratedPaperHedgeApplication
from .production_assembly import (
    ProductionMainLoopAssembly,
    build_production_main_loop_assembly,
)
from .paper_state import (
    JsonPaperStateStore,
    NullPaperStateStore,
    PaperStateStore,
    SqlPaperStateStore,
)
from .repository import PersistenceMirroringReadonlyRepository
from .risk_adapter import RuntimeRiskApprovalAdapter

logger = logging.getLogger(__name__)


def _decimal(value: object, default: str) -> Decimal:
    result = Decimal(default) if value is None else Decimal(str(value))
    if not result.is_finite():
        raise OperationalException("hedge composition numeric values must be finite")
    return result


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _session_bind(session_factory: object) -> object | None:
    keyword_args = getattr(session_factory, "kw", None)
    if isinstance(keyword_args, Mapping) and keyword_args.get("bind") is not None:
        return keyword_args["bind"]
    bind = getattr(session_factory, "bind", None)
    if bind is not None:
        return bind
    try:
        session = session_factory()  # type: ignore[operator]
    except Exception:
        return None
    try:
        return session.get_bind()
    finally:
        session.close()


class RuntimeReadinessExecutionAdapter:
    def __init__(self, runtime: HedgeRiskRuntime) -> None:
        self._runtime = runtime

    def evaluate(self, position_key: PositionKey) -> ReadinessDecision:
        report = self._runtime.readiness.refresh()
        risk_key = RiskPositionKey(
            account_id=position_key.account_id,
            symbol=position_key.canonical_symbol,
            position_side=position_key.position_side.value,
            exchange=position_key.exchange,
        )
        if report.state is ReadinessState.READY:
            state = ExecutionReadinessState.READY
        elif report.state is ReadinessState.HALT:
            state = ExecutionReadinessState.HALT
        else:
            state = ExecutionReadinessState.DEGRADED
        return ReadinessDecision(
            state=state,
            reason_codes=tuple(code.value for code in report.reason_codes),
            allow_reduce=self._runtime.readiness.gate.allows_controlled_reduce(risk_key),
        )


class RuntimeSingleWriterExecutionAdapter:
    def __init__(self, writer: SingleWriterGuard) -> None:
        self._writer = writer

    def assert_leader(self, *, account_id: str, now: object) -> None:
        del account_id, now
        self._writer.assert_valid()

    def claim(self, *, account_id: str, owner_id: str) -> bool:
        del account_id, owner_id
        return self._writer.status().valid


class RuntimePositionLockExecutionAdapter:
    def __init__(self, locks: PositionLockManager) -> None:
        self._locks = locks

    def acquire(
        self,
        position_key: PositionKey | None = None,
        *,
        key: PositionKey | None = None,
        owner_id: str | None = None,
    ) -> AbstractContextManager[None] | bool:
        resolved = position_key if position_key is not None else key
        if resolved is None:
            raise TypeError("position_key or key is required")
        if owner_id is not None:
            # Direction-five legacy boolean lock acquisition is not used by the
            # integrated engine. Refuse it instead of creating a second owner model.
            return False
        return self._locks.lock(
            account_id=resolved.account_id,
            symbol=resolved.canonical_symbol,
            position_side=resolved.position_side.value,
            exchange=resolved.exchange,
        )


@dataclass(slots=True)
class HedgeCompositionRoot:
    central_runtime: HedgeRuntime
    operation_mode: str
    persistence_service: HedgePersistenceService
    writer: SingleWriterGuard | None = None
    risk_runtime: HedgeRiskRuntime | None = None
    paper_application: IntegratedPaperHedgeApplication | None = None
    readonly_coordinator: HedgeRuntimeCoordinator | None = None
    production_main_loop_assembly: ProductionMainLoopAssembly | None = None
    control_service: HedgeControlService | None = None
    _started: bool = False

    def start(self) -> None:
        if self._started:
            return
        try:
            if self.writer is not None and not self.writer.status().valid:
                self.writer.acquire()
            if self.risk_runtime is not None:
                self.risk_runtime.start()
            if self.readonly_coordinator is not None:
                self.readonly_coordinator.start()
        except Exception as exc:
            self.central_runtime.halt(f"HEDGE_COMPOSITION_START_FAILED:{type(exc).__name__}")
            self.stop()
            raise
        self._started = True

    def refresh(self) -> None:
        if not self._started:
            raise RuntimeError("hedge composition root has not been started")
        if self.writer is not None:
            status = self.writer.status()
            if not status.valid:
                self.central_runtime.halt(status.reason_code)
                raise RuntimeError("single-writer lease is not valid")
            self.writer.renew()
        if self.risk_runtime is not None:
            self.risk_runtime.refresh()
        if self.readonly_coordinator is not None:
            self.readonly_coordinator.refresh()

    def stop(self) -> None:
        if self.readonly_coordinator is not None:
            try:
                self.readonly_coordinator.stop()
            except Exception:
                logger.exception("Failed to stop readonly hedge coordinator")
        if self.risk_runtime is not None:
            try:
                self.risk_runtime.stop(release_lease=False)
            except Exception:
                logger.exception("Failed to stop hedge risk runtime")
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                logger.exception("Failed to release hedge writer lease")
        self._started = False


def _persistence_service(session_factory: object | None) -> HedgePersistenceService:
    if session_factory is None:
        raise OperationalException(
            "Hedge runtime requires an initialized durable SQLAlchemy session factory."
        )
    try:
        service = HedgePersistenceService(session_factory)  # type: ignore[arg-type]
        # Recovery is part of readiness: a runtime that cannot replay its durable
        # account state must not silently fall back to memory.
        service.recover_all()
        return service
    except Exception as exc:
        raise OperationalException(
            "Hedge persistence initialization or recovery failed; runtime is HALT."
        ) from exc


def _paper_state_store(
    *,
    config: Mapping[str, Any],
    paper: Mapping[str, Any],
    session_factory: object,
    account_id: str,
    symbol: str,
    source: str,
) -> PaperStateStore:
    ephemeral = paper.get("ephemeral", False)
    if not isinstance(ephemeral, bool):
        raise OperationalException("hedge.paper.ephemeral must be a boolean")
    if ephemeral:
        return NullPaperStateStore()

    backend = str(paper.get("state_backend", "sql")).strip().lower()
    if backend == "sql":
        return SqlPaperStateStore(
            session_factory,
            exchange=str(_mapping(config.get("exchange")).get("name", "binance")),
            account_id=account_id,
            symbol=symbol,
            source=source,
        )
    if backend == "json":
        raw_path = paper.get("state_path")
        if raw_path is None:
            user_data = Path(str(config.get("user_data_dir", "user_data")))
            raw_path = user_data / "hedge-paper-state.json"
        return JsonPaperStateStore(Path(str(raw_path)))
    raise OperationalException("hedge.paper.state_backend must be sql or json")


def _build_readonly_coordinator(
    *,
    config: Mapping[str, Any],
    central_runtime: HedgeRuntime,
    persistence: HedgePersistenceService,
) -> HedgeRuntimeCoordinator:
    exchange = _mapping(config.get("exchange"))
    if not str(exchange.get("key", "")).strip() or not str(
        exchange.get("secret", "")
    ).strip():
        raise OperationalException(
            "Readonly/shadow Hedge composition requires Binance API key and secret"
        )
    return HedgeRuntimeCoordinator(
        config=config,
        central_runtime=central_runtime,
        repository=PersistenceMirroringReadonlyRepository(persistence),
    )


def _build_paper_graph(
    *,
    config: Mapping[str, Any],
    central_runtime: HedgeRuntime,
    session_factory: object,
    persistence: HedgePersistenceService,
    lease_suffix: str,
) -> tuple[
    SingleWriterGuard,
    HedgeRiskRuntime,
    IntegratedPaperHedgeApplication,
]:
    hedge = _mapping(config.get("hedge"))
    paper = _mapping(hedge.get("paper"))
    account_id = central_runtime.config.account_id
    symbol = central_runtime.config.managed_pair or str(config.get("managed_pair", ""))
    if not symbol:
        raise OperationalException("Paper hedge composition requires managed_pair")
    leverage = _decimal(
        paper.get("leverage", hedge.get("target_leverage", "3")),
        "3",
    )
    if leverage < 1:
        raise OperationalException("Paper leverage must be greater than or equal to 1")

    bind = _session_bind(session_factory)
    if bind is None:
        raise OperationalException(
            "Paper hedge composition requires a durable SQLAlchemy lease binding."
        )
    try:
        lease_store = SqlAlchemyDatabaseLeaseStore(bind)
    except Exception as exc:
        raise OperationalException("Unable to initialize durable single-writer lease") from exc
    writer = SingleWriterGuard(
        lease_store,
        owner_id=f"{socket.gethostname()}:{account_id}:{lease_suffix}",
        lease_name=f"freqtrade-hedge:{account_id}:{lease_suffix}",
    )
    locks = PositionLockManager()
    limits = RiskLimits(
        max_margin_utilization=_decimal(hedge.get("max_margin_utilization"), "0.80"),
        min_liquidation_buffer_ratio=_decimal(
            hedge.get("min_liquidation_buffer_ratio"), "0.05"
        ),
        max_gross_notional=(
            None
            if hedge.get("max_gross_notional") is None
            else _decimal(hedge.get("max_gross_notional"), "0")
        ),
        max_gross_exposure_ratio=(
            None
            if hedge.get("max_gross_exposure_ratio") is None
            else _decimal(hedge.get("max_gross_exposure_ratio"), "0.80")
        ),
        max_single_order_notional=_decimal(
            hedge.get("max_single_order_notional", paper.get("initial_balance", "1000")),
            "1000",
        ),
    )
    # Direction-three currently exposes an exchange-oriented readiness input type.
    # These values are internal to the Paper risk transaction and are never
    # published as exchange health. The central projection uses paper.* checks.
    readiness_inputs = ReadinessInputs(
        database_migration_succeeded=True,
        database_migration_checksum_valid=True,
        single_writer_lease_valid=False,
        position_mode="hedge",
        margin_mode="cross",
        configured_leverage=leverage,
        observed_leverages=(leverage,),
        unmanaged_position_count=0,
        unmanaged_order_count=0,
        rest_snapshot_valid=True,
        user_stream_fresh=True,
        unknown_order_count=0,
        reconciliation_converged=True,
        risk_data_valid=True,
    )
    risk_runtime = build_hedge_risk_runtime(
        limits=limits,
        writer=writer,
        readiness_inputs=readiness_inputs,
        position_locks=locks,
        enable_lease_runner=False,
    )
    readiness_adapter = RuntimeReadinessExecutionAdapter(risk_runtime)
    writer_adapter = RuntimeSingleWriterExecutionAdapter(writer)
    lock_adapter = RuntimePositionLockExecutionAdapter(locks)
    market_rules = StaticMarketRules(
        MarketRules(
            quantity_step=_decimal(paper.get("qty_step"), "0.001"),
            price_tick=_decimal(paper.get("tick_size"), "0.01"),
            minimum_quantity=max(
                _decimal(paper.get("min_qty"), "0.001"),
                Decimal("0.00000001"),
            ),
            minimum_notional=max(
                _decimal(paper.get("min_notional"), "5"),
                Decimal("0.00000001"),
            ),
        )
    )
    commit_store = SqlRiskApprovalCommitStore(session_factory)
    action_groups = SqlActionGroupRepository(session_factory)
    execution_ledger = SqlExecutionLedger(session_factory)
    execution_store = SqlExecutionStore(session_factory)
    execution_idempotency = SqlExecutionIdempotencyStore(
        session_factory,
        execution_store,
        lease_seconds=central_runtime.config.paper.idempotency_lease_seconds
        if central_runtime.config.paper is not None
        else int(paper.get("idempotency_lease_seconds", 300)),
    )
    account_event_sink = SqlPaperAccountEventSink(
        persistence,
        account_id=account_id,
        exchange="paper",
        symbol=symbol,
        asset=str(config.get("stake_currency", "USDT")),
        venue_exchange=str(_mapping(config.get("exchange")).get("name", "binance")),
    )
    state_store = _paper_state_store(
        config=config,
        paper=paper,
        session_factory=session_factory,
        account_id=account_id,
        symbol=symbol,
        source="SHADOW" if lease_suffix == "shadow-paper" else "PAPER",
    )

    execution_recovery = SqlPaperExecutionRecovery(
        session_factory,
        account_id=account_id,
        symbol=raw_symbol(symbol),
    )
    application = IntegratedPaperHedgeApplication(
        config=config,
        account_id=account_id,
        symbol=symbol,
        publisher=None,
        build_execution=False,
        state_store=state_store,
        account_event_sink=account_event_sink,
        execution_recovery=execution_recovery,
    )
    risk_adapter = RuntimeRiskApprovalAdapter(
        runtime=risk_runtime,
        portfolio_provider=application.risk_portfolio,
        leverage=leverage,
        maintenance_margin_rate=application.planner_config.maintenance_margin_rate,
        commit_store=commit_store,
    )
    application.bind_execution(
        risk=risk_adapter,
        readiness=readiness_adapter,
        single_writer=writer_adapter,
        position_lock=lock_adapter,
        market_rules=market_rules,
        action_groups=action_groups,
        transaction=execution_ledger,
        store=execution_store,
        idempotency=execution_idempotency,
        account_event_sink=account_event_sink,
    )
    # H3 recovery is mandatory before the Paper graph is returned. JSON state is
    # an additional simulation checkpoint, never a replacement for the SQL fact store.
    persistence.recover_all()
    return writer, risk_runtime, application


def build_hedge_composition(
    *,
    config: Mapping[str, Any],
    central_runtime: HedgeRuntime,
    session_factory: object | None,
) -> HedgeCompositionRoot:
    """Build a mode-specific, fail-closed graph with source-separated projections."""

    hedge = _mapping(config.get("hedge"))
    operation_mode = assert_supported_operation_mode(
        hedge.get("operation_mode", central_runtime.config.operation_mode)
    )
    persistence = _persistence_service(session_factory)
    if session_factory is None:  # guarded by _persistence_service, retained for typing
        raise OperationalException("Hedge runtime requires a session factory")

    readonly = None
    writer = None
    risk_runtime = None
    application = None

    if operation_mode in {"readonly", "shadow"}:
        readonly = _build_readonly_coordinator(
            config=config,
            central_runtime=central_runtime,
            persistence=persistence,
        )

    if operation_mode in {"paper", "shadow"}:
        writer, risk_runtime, application = _build_paper_graph(
            config=config,
            central_runtime=central_runtime,
            session_factory=session_factory,
            persistence=persistence,
            lease_suffix="shadow-paper" if operation_mode == "shadow" else "paper",
        )

    if readonly is None and application is None:
        raise OperationalException(f"No Hedge composition was built for {operation_mode!r}")

    production_main_loop_assembly = build_production_main_loop_assembly(
        config=config,
        session_factory=session_factory,
        paper_application=application,
    )
    if (
        production_main_loop_assembly is not None
        and readonly is None
        and production_main_loop_assembly.cycle_owner != "PAPER_APPLICATION"
    ):
        raise OperationalException(
            "production-locked hedge.main_loop requires readonly or shadow operation mode."
        )
    control_service = build_hedge_control_service(
        config=config,
        production_assembly=production_main_loop_assembly,
        readonly_coordinator=readonly,
        paper_application=application,
        persistence_service=persistence,
    )

    return HedgeCompositionRoot(
        central_runtime=central_runtime,
        operation_mode=operation_mode,
        persistence_service=persistence,
        writer=writer,
        risk_runtime=risk_runtime,
        paper_application=application,
        readonly_coordinator=readonly,
        production_main_loop_assembly=production_main_loop_assembly,
        control_service=control_service,
    )
