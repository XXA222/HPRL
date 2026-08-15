"""High-level runtime facade for risk, readiness, locks and writer lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable, Iterable
import uuid

from freqtrade.enums.hedge import PositionAction, PositionSide
from freqtrade.hedge.concurrency.lease_runner import LeaseRunnerStatus, SingleWriterLeaseRunner
from freqtrade.hedge.concurrency.position_lock import PositionLockManager
from freqtrade.hedge.concurrency.single_writer import SingleWriterGuard
from freqtrade.hedge.readiness.checks import ReadinessInputs
from freqtrade.hedge.readiness.gate import ReadinessGate
from freqtrade.hedge.readiness.monitor import ReadinessMonitor
from freqtrade.hedge.numeric import require_positive
from freqtrade.hedge.readiness.state import (
    ReadinessReasonCode,
    ReadinessReport,
    ReadinessState,
)
from freqtrade.hedge.risk.actions import (
    RiskActionState,
    RiskActionStateMachine,
    RiskApprovalCoordinator,
    RiskEvent,
    RiskMode,
    UnifiedRiskApproval,
    UnifiedRiskBatchApproval,
)
from freqtrade.hedge.risk.engine import HedgeRiskEngine
from freqtrade.hedge.risk.facts import AccountRiskFacts
from freqtrade.hedge.risk.limits import RiskLimits
from freqtrade.hedge.risk.portfolio import RiskPortfolioSnapshot
from freqtrade.hedge.symbols import canonicalize_symbol


@dataclass(frozen=True, slots=True)
class OrderRiskIntent:
    symbol: str
    position_side: PositionSide
    action: PositionAction
    requested_quantity: Decimal
    reference_price: Decimal
    leverage: Decimal = Decimal("1")
    intent_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    idempotency_key: str = field(default_factory=lambda: uuid.uuid4().hex)
    correlation_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    expires_at_ms: int | None = None
    target_snapshot_version: int | None = None
    maintenance_margin_rate: Decimal = Decimal("0.01")

    def __post_init__(self) -> None:
        side = (
            self.position_side
            if isinstance(self.position_side, PositionSide)
            else PositionSide(str(self.position_side).upper())
        )
        action = (
            self.action
            if isinstance(self.action, PositionAction)
            else PositionAction(str(self.action).upper())
        )
        if side is PositionSide.BOTH:
            raise ValueError("Order risk intent side must be LONG or SHORT.")
        object.__setattr__(self, "symbol", canonicalize_symbol(self.symbol))
        object.__setattr__(self, "position_side", side)
        object.__setattr__(self, "action", action)
        object.__setattr__(
            self,
            "requested_quantity",
            require_positive(self.requested_quantity, field="requested_quantity"),
        )
        object.__setattr__(
            self,
            "reference_price",
            require_positive(self.reference_price, field="reference_price"),
        )
        leverage = require_positive(self.leverage, field="leverage")
        if leverage < 1:
            raise ValueError("leverage must be greater than or equal to 1.")
        object.__setattr__(self, "leverage", leverage)
        for field_name in ("intent_id", "idempotency_key", "correlation_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty.")
            object.__setattr__(self, field_name, value.strip())
        for field_name in ("expires_at_ms", "target_snapshot_version"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{field_name} must be a nonnegative integer.")
        rate = require_positive(
            self.maintenance_margin_rate,
            field="maintenance_margin_rate",
        )
        if rate > 1:
            raise ValueError("maintenance_margin_rate must not exceed 1.")
        object.__setattr__(self, "maintenance_margin_rate", rate)


@dataclass(frozen=True, slots=True)
class HedgeRiskRuntimeStatus:
    readiness: ReadinessReport
    risk_state: RiskActionState
    lease_runner: LeaseRunnerStatus | None
    reservations: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return {
            "readiness": self.readiness.as_dict(),
            "risk_state": {
                "mode": self.risk_state.mode.value,
                "reason_codes": list(self.risk_state.reason_codes),
                "version": self.risk_state.version,
            },
            "lease_runner": (
                None if self.lease_runner is None else self.lease_runner.as_dict()
            ),
            "reservations": self.reservations,
        }


class HedgeRiskRuntime:
    """Executable direction-three main path.

    The facade converts portfolio facts into requests, keeps readiness and the
    risk action state synchronized, and exposes single and atomic-batch order
    approvals without requiring callers to manipulate internal risk fields.
    """

    def __init__(
        self,
        *,
        coordinator: RiskApprovalCoordinator,
        readiness: ReadinessMonitor,
        state_machine: RiskActionStateMachine,
        lease_runner: SingleWriterLeaseRunner | None = None,
        synchronize_risk_state: bool = True,
    ) -> None:
        self._coordinator = coordinator
        self._readiness = readiness
        self._state_machine = state_machine
        self._lease_runner = lease_runner
        self._synchronize_risk_state = synchronize_risk_state

    @property
    def coordinator(self) -> RiskApprovalCoordinator:
        return self._coordinator

    @property
    def readiness(self) -> ReadinessMonitor:
        return self._readiness

    @property
    def state_machine(self) -> RiskActionStateMachine:
        return self._state_machine

    @property
    def lease_runner(self) -> SingleWriterLeaseRunner | None:
        return self._lease_runner

    def _sync_state(self, report: ReadinessReport) -> RiskActionState:
        if not self._synchronize_risk_state:
            return self._state_machine.state
        reasons = tuple(code.value for code in report.reason_codes)
        current = self._state_machine.state
        if report.state is ReadinessState.HALT:
            if current.mode is not RiskMode.HALT or current.reason_codes != reasons:
                return self._state_machine.transition(
                    RiskEvent.ENTER_HALT,
                    reason_codes=reasons or ("READINESS_HALT",),
                )
            return current
        only_position_unknown = bool(reasons) and all(
            reason is ReadinessReasonCode.UNKNOWN_ORDER_PRESENT
            for reason in report.reason_codes
        )
        if only_position_unknown:
            if current.mode is RiskMode.REDUCE_ONLY:
                return self._state_machine.transition(RiskEvent.RECOVERED)
            return current
        if not report.ready:
            desired_reasons = reasons or ("READINESS_NOT_READY",)
            if current.mode is RiskMode.NORMAL or (
                current.mode is RiskMode.REDUCE_ONLY
                and current.reason_codes != desired_reasons
            ):
                return self._state_machine.transition(
                    RiskEvent.ENTER_REDUCE_ONLY,
                    reason_codes=desired_reasons,
                )
            return current
        if current.mode is RiskMode.REDUCE_ONLY:
            return self._state_machine.transition(RiskEvent.RECOVERED)
        return current

    def refresh(self) -> HedgeRiskRuntimeStatus:
        report = self._readiness.refresh()
        self._sync_state(report)
        self._coordinator.prune_expired_reservations()
        return self.status()

    def start(self) -> HedgeRiskRuntimeStatus:
        if self._lease_runner is not None:
            self._lease_runner.start(require_initial_acquire=False)
        return self.refresh()

    def stop(self, *, release_lease: bool = True) -> HedgeRiskRuntimeStatus:
        self._coordinator.release_all_reservations()
        if self._lease_runner is not None:
            self._lease_runner.stop(release=release_lease)
        self._readiness.update(single_writer_lease_valid=False)
        return self.status()

    def _bind_portfolio_facts(self, portfolio: RiskPortfolioSnapshot) -> None:
        self._readiness.update(
            risk_data_valid=portfolio.account.effective_risk_data_valid,
            risk_snapshot_observed_at_ms=portfolio.account.observed_at_ms,
        )

    def bind_account_facts(self, facts: AccountRiskFacts) -> RiskPortfolioSnapshot:
        """Bind a reconciled direction-two snapshot to risk and Readiness."""

        if not isinstance(facts, AccountRiskFacts):
            raise TypeError("facts must be an AccountRiskFacts instance.")
        portfolio = facts.to_portfolio()
        self._readiness.update(
            reconciliation_converged=facts.reconciliation_converged,
            risk_data_valid=portfolio.account.effective_risk_data_valid,
            risk_snapshot_observed_at_ms=portfolio.account.observed_at_ms,
        )
        return portfolio

    def approve_order_from_facts(
        self,
        *,
        facts: AccountRiskFacts,
        intent: OrderRiskIntent,
        timeout_seconds: float | None = None,
    ) -> UnifiedRiskApproval:
        portfolio = self.bind_account_facts(facts)
        return self.approve_order(
            portfolio=portfolio,
            intent=intent,
            timeout_seconds=timeout_seconds,
        )

    def approve_order(
        self,
        *,
        portfolio: RiskPortfolioSnapshot,
        intent: OrderRiskIntent,
        timeout_seconds: float | None = None,
    ) -> UnifiedRiskApproval:
        self._bind_portfolio_facts(portfolio)
        self.refresh()
        request = portfolio.build_request(
            symbol=intent.symbol,
            position_side=intent.position_side,
            action=intent.action,
            requested_quantity=intent.requested_quantity,
            reference_price=intent.reference_price,
            leverage=intent.leverage,
            intent_id=intent.intent_id,
            idempotency_key=intent.idempotency_key,
            correlation_id=intent.correlation_id,
            expires_at_ms=intent.expires_at_ms,
            target_snapshot_version=intent.target_snapshot_version,
            maintenance_margin_rate=intent.maintenance_margin_rate,
        )
        return self._coordinator.approve(
            request=request,
            account=portfolio.account,
            timeout_seconds=timeout_seconds,
        )

    def approve_batch_from_facts(
        self,
        *,
        facts: AccountRiskFacts,
        intents: Iterable[OrderRiskIntent],
        timeout_seconds: float | None = None,
    ) -> UnifiedRiskBatchApproval:
        portfolio = self.bind_account_facts(facts)
        return self.approve_batch(
            portfolio=portfolio,
            intents=intents,
            timeout_seconds=timeout_seconds,
        )

    def approve_batch(
        self,
        *,
        portfolio: RiskPortfolioSnapshot,
        intents: Iterable[OrderRiskIntent],
        timeout_seconds: float | None = None,
    ) -> UnifiedRiskBatchApproval:
        self._bind_portfolio_facts(portfolio)
        self.refresh()
        requests = tuple(
            (
                portfolio.build_request(
                    symbol=intent.symbol,
                    position_side=intent.position_side,
                    action=intent.action,
                    requested_quantity=intent.requested_quantity,
                    reference_price=intent.reference_price,
                    leverage=intent.leverage,
                    intent_id=intent.intent_id,
                    idempotency_key=intent.idempotency_key,
                    correlation_id=intent.correlation_id,
                    expires_at_ms=intent.expires_at_ms,
                    target_snapshot_version=intent.target_snapshot_version,
                    maintenance_margin_rate=intent.maintenance_margin_rate,
                ),
                portfolio.account,
            )
            for intent in intents
        )
        return self._coordinator.approve_batch(
            requests,
            timeout_seconds=timeout_seconds,
        )

    def status(self) -> HedgeRiskRuntimeStatus:
        return HedgeRiskRuntimeStatus(
            readiness=self._readiness.report,
            risk_state=self._state_machine.state,
            lease_runner=(None if self._lease_runner is None else self._lease_runner.status()),
            reservations=self._coordinator.reservation_snapshot(),
        )


def build_hedge_risk_runtime(
    *,
    limits: RiskLimits,
    writer: SingleWriterGuard,
    readiness_inputs: ReadinessInputs,
    position_locks: PositionLockManager | None = None,
    state_machine: RiskActionStateMachine | None = None,
    lease_runner: SingleWriterLeaseRunner | None = None,
    enable_lease_runner: bool = True,
    lease_interval_seconds: float | None = None,
    lock_timeout_seconds: float = 5.0,
    reduce_reservation_ttl_seconds: float = 30.0,
    increase_reservation_ttl_ms: int = 30_000,
    readiness_clock_ms: Callable[[], int] | None = None,
    reservation_clock_ms: Callable[[], int] | None = None,
    lock_monotonic_clock: Callable[[], float] | None = None,
) -> HedgeRiskRuntime:
    """Build the complete direction-three runtime from integration inputs."""

    if not isinstance(enable_lease_runner, bool):
        raise ValueError("enable_lease_runner must be a boolean.")
    locks = position_locks or PositionLockManager(
        default_timeout_seconds=lock_timeout_seconds,
        reservation_ttl_seconds=reduce_reservation_ttl_seconds,
        monotonic_clock=lock_monotonic_clock,
    )
    state = state_machine or RiskActionStateMachine()
    gate = ReadinessGate(clock_ms=readiness_clock_ms)
    monitor = ReadinessMonitor(gate=gate, inputs=readiness_inputs, writer=writer)
    coordinator = RiskApprovalCoordinator(
        engine=HedgeRiskEngine(limits),
        locks=locks,
        writer=writer,
        readiness=gate,
        state_machine=state,
        reservation_ttl_ms=increase_reservation_ttl_ms,
        clock_ms=reservation_clock_ms,
    )
    runner = lease_runner
    if runner is not None and runner.guard is not writer:
        raise ValueError("lease_runner must use the supplied SingleWriterGuard.")
    if runner is None and enable_lease_runner:
        runner = SingleWriterLeaseRunner(
            writer,
            interval_seconds=lease_interval_seconds,
        )
    return HedgeRiskRuntime(
        coordinator=coordinator,
        readiness=monitor,
        state_machine=state,
        lease_runner=runner,
    )

