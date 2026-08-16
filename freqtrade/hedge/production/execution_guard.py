"""Production-readiness-bound execution gate and canonical runtime builder.

The existing Binance adapter remains unchanged.  This guard subclasses its gate so all
current adapter type checks continue to work, while arming/order admission also requires
short-lived Production Readiness capability leases.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Callable

from freqtrade.hedge.execution.binance_environment import ExecutionEnvironment
from freqtrade.hedge.execution.production_gate import (
    ExecutionWriteLockedError,
    ProductionExecutionGate,
    ProductionGateEvidence,
)
from freqtrade.hedge.execution.service import ApprovedOrderIntent

from .canary import CanaryLevel, CanaryRuntime, DEFAULT_CANARY_LIMITS, evaluate_canary
from .contracts import Capability, CapabilityLease
from .policy import StageEvaluator
from .reservations import ExposureReservation, ExposureReservationBook
from .runtime_supervisor import RuntimeSafetySnapshot


class ReadinessBoundProductionExecutionGate(ProductionExecutionGate):
    def __init__(
        self,
        evidence: ProductionGateEvidence,
        *,
        evaluator: StageEvaluator,
        clock: Callable[[], datetime] | None = None,
        canary_level: CanaryLevel = CanaryLevel.LOCKED,
        canary_runtime_provider: Callable[[], CanaryRuntime] | None = None,
        canary_reservations: ExposureReservationBook | None = None,
        runtime_safety_provider: Callable[[], RuntimeSafetySnapshot] | None = None,
        max_runtime_safety_age: timedelta = timedelta(seconds=5),
        max_runtime_future_skew: timedelta = timedelta(seconds=1),
    ) -> None:
        super().__init__(evidence)
        self._evaluator = evaluator
        self._clock = clock or (lambda: datetime.now(UTC))
        self._canary_level = CanaryLevel(canary_level)
        if self._canary_level > CanaryLevel.SMALL:
            raise ValueError("live candidate canary level must not exceed SMALL")
        self._canary_runtime_provider = canary_runtime_provider
        self._canary_reservations = canary_reservations or ExposureReservationBook()
        self._last_canary_reservation: ExposureReservation | None = None
        self._runtime_safety_provider = runtime_safety_provider
        if max_runtime_safety_age <= timedelta(0):
            raise ValueError("max_runtime_safety_age must be positive")
        if max_runtime_future_skew < timedelta(0):
            raise ValueError("max_runtime_future_skew must be nonnegative")
        self._max_runtime_safety_age = max_runtime_safety_age
        self._max_runtime_future_skew = max_runtime_future_skew
        if (
            self.evidence.environment is ExecutionEnvironment.LIVE
            and self._runtime_safety_provider is None
        ):
            raise ValueError("LIVE readiness-bound execution requires runtime_safety_provider")
        self._armed_safety_epoch: int | None = None
        self._reduce_lease: CapabilityLease | None = None
        self._canary_risk_lease: CapabilityLease | None = None
        self._new_risk_lease: CapabilityLease | None = None
        self._testnet_lease: CapabilityLease | None = None

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("production readiness gate clock must be timezone-aware")
        return value.astimezone(UTC)

    def _runtime_snapshot(self, now: datetime) -> RuntimeSafetySnapshot | None:
        provider = self._runtime_safety_provider
        if provider is None:
            return None
        snapshot = provider()
        if snapshot.observed_at > now + self._max_runtime_future_skew:
            raise ExecutionWriteLockedError("RUNTIME_SAFETY_FROM_FUTURE")
        if now - snapshot.observed_at > self._max_runtime_safety_age:
            raise ExecutionWriteLockedError("RUNTIME_SAFETY_STALE")
        return snapshot

    def arm(self, *, token: str, actor: str, confirmed: bool, ttl_seconds: int = 300):  # type: ignore[override]
        now = self._now()
        self._reduce_lease = None
        self._canary_risk_lease = None
        self._new_risk_lease = None
        self._testnet_lease = None
        try:
            if self._runtime_safety_provider is not None:
                runtime = self._runtime_snapshot(now)
                assert runtime is not None
                if self.evidence.environment is ExecutionEnvironment.LIVE and not runtime.allows_reduce:
                    raise ExecutionWriteLockedError("RUNTIME_SAFETY_BLOCKS_ARM")
                self._armed_safety_epoch = runtime.safety_epoch
            if self.evidence.environment is ExecutionEnvironment.LIVE:
                self._reduce_lease = self._evaluator.issue_lease(
                    Capability.LIVE_REDUCE, actor=actor, now=now
                )
                try:
                    self._canary_risk_lease = self._evaluator.issue_lease(
                        Capability.LIVE_CANARY_RISK, actor=actor, now=now
                    )
                except PermissionError:
                    self._canary_risk_lease = None
                try:
                    self._new_risk_lease = self._evaluator.issue_lease(
                        Capability.LIVE_NEW_RISK, actor=actor, now=now
                    )
                except PermissionError:
                    # Candidate live capability intentionally remains bounded until
                    # real LIVE_CANARY evidence promotes the ledger to LIVE_READY.
                    self._new_risk_lease = None
            elif self.evidence.environment is ExecutionEnvironment.TESTNET:
                self._testnet_lease = self._evaluator.issue_lease(
                    Capability.TESTNET_WRITE, actor=actor, now=now
                )
            return super().arm(
                token=token, actor=actor, confirmed=confirmed, ttl_seconds=ttl_seconds
            )
        except Exception:
            # Arming is transactional: a failed exchange gate/token check must not
            # leave readiness leases resident in this process.
            self._reduce_lease = None
            self._canary_risk_lease = None
            self._new_risk_lease = None
            self._testnet_lease = None
            self._armed_safety_epoch = None
            raise

    def disarm(self):  # type: ignore[override]
        self._reduce_lease = None
        self._canary_risk_lease = None
        self._new_risk_lease = None
        self._testnet_lease = None
        self._last_canary_reservation = None
        self._armed_safety_epoch = None
        return super().disarm()

    def assert_order_allowed(self, approved: ApprovedOrderIntent):  # type: ignore[override]
        now = self._now()
        if self._runtime_safety_provider is not None:
            runtime = self._runtime_snapshot(now)
            assert runtime is not None
            if self._armed_safety_epoch is None or runtime.safety_epoch != self._armed_safety_epoch:
                raise ExecutionWriteLockedError("RUNTIME_SAFETY_EPOCH_CHANGED_REARM_REQUIRED")
            if approved.intent.reduces_risk:
                if not runtime.allows_reduce:
                    raise ExecutionWriteLockedError("RUNTIME_SAFETY_BLOCKS_REDUCE")
            elif not runtime.allows_new_risk:
                raise ExecutionWriteLockedError("RUNTIME_SAFETY_BLOCKS_NEW_RISK")
        canary_candidate = False
        if self.evidence.environment is ExecutionEnvironment.LIVE:
            if approved.intent.reduces_risk:
                lease = self._reduce_lease
                capability = Capability.LIVE_REDUCE
            elif self._new_risk_lease is not None:
                lease = self._new_risk_lease
                capability = Capability.LIVE_NEW_RISK
            else:
                lease = self._canary_risk_lease
                capability = Capability.LIVE_CANARY_RISK
                canary_candidate = True
            self._assert_fresh_lease(lease, capability, now)
        elif self.evidence.environment is ExecutionEnvironment.TESTNET:
            self._assert_fresh_lease(
                self._testnet_lease, Capability.TESTNET_WRITE, now
            )

        permit = super().assert_order_allowed(approved)
        if canary_candidate:
            self._last_canary_reservation = self._assert_canary_order_allowed(
                approved.client_order_id, permit.notional
            )
        return permit

    def _assert_fresh_lease(
        self, lease: CapabilityLease | None, capability: Capability, now: datetime
    ) -> None:
        if lease is None or not lease.valid_at(now):
            raise ExecutionWriteLockedError(
                f"PRODUCTION_READINESS_{capability.value}_LEASE_REQUIRED"
            )
        current = self._evaluator.evaluate(lease.stage, now=now)
        if not current.passed or current.evidence_digest != lease.evidence_digest:
            raise ExecutionWriteLockedError("PRODUCTION_READINESS_EVIDENCE_CHANGED")

    @property
    def last_canary_reservation(self) -> ExposureReservation | None:
        return self._last_canary_reservation

    def commit_canary_reservation(self, reservation_id: str) -> ExposureReservation:
        return self._canary_reservations.commit(reservation_id, now=self._now())

    def release_canary_reservation(self, reservation_id: str) -> ExposureReservation:
        return self._canary_reservations.release(reservation_id, now=self._now())

    def commit_canary_for_client(self, client_order_id: str) -> ExposureReservation | None:
        item = self._canary_reservations.find_by_client_order(client_order_id, now=self._now())
        if item is None:
            return None
        if item.state.value == "COMMITTED":
            return item
        return self._canary_reservations.commit(item.reservation_id, now=self._now())

    def release_canary_for_client(self, client_order_id: str) -> ExposureReservation | None:
        item = self._canary_reservations.find_by_client_order(client_order_id, now=self._now())
        if item is None:
            return None
        if item.state.value in {"RELEASED", "EXPIRED"}:
            return item
        return self._canary_reservations.release(item.reservation_id, now=self._now())

    def _assert_canary_order_allowed(
        self, client_order_id: str, order_notional
    ) -> ExposureReservation:
        level = self._canary_level
        if level < CanaryLevel.MICRO:
            raise ExecutionWriteLockedError("LIVE_CANARY_MODE_NOT_ENABLED")
        provider = self._canary_runtime_provider
        if provider is None:
            raise ExecutionWriteLockedError("LIVE_CANARY_RUNTIME_REQUIRED")
        limits = DEFAULT_CANARY_LIMITS[level]
        if order_notional > limits.max_order_notional:
            raise ExecutionWriteLockedError("LIVE_CANARY_ORDER_NOTIONAL_LIMIT")
        runtime = provider()
        pending = self._canary_reservations.snapshot(now=self._now())
        projected = CanaryRuntime(
            gross_notional=runtime.gross_notional + pending.held_notional + order_notional,
            daily_realized_pnl=runtime.daily_realized_pnl,
            drawdown_ratio=runtime.drawdown_ratio,
            open_orders=runtime.open_orders + pending.held_orders + 1,
            unresolved_incidents=runtime.unresolved_incidents,
        )
        decision = evaluate_canary(level, projected)
        if not decision.allowed:
            raise ExecutionWriteLockedError(
                "LIVE_CANARY_BLOCKED:" + ",".join(decision.reasons)
            )
        remaining_gross = max(
            limits.max_gross_notional - runtime.gross_notional,
            0,
        )
        remaining_orders = max(limits.max_open_orders - runtime.open_orders, 0)
        return self._canary_reservations.reserve(
            client_order_id=client_order_id,
            notional=order_notional,
            now=self._now(),
            max_total_notional=remaining_gross,
            max_orders=remaining_orders,
        )


def build_readiness_bound_execution_gate(
    evidence: ProductionGateEvidence,
    *,
    evaluator: StageEvaluator,
    runtime_safety_provider: Callable[[], RuntimeSafetySnapshot] | None = None,
    max_runtime_safety_age: timedelta = timedelta(seconds=5),
    max_runtime_future_skew: timedelta = timedelta(seconds=1),
    clock: Callable[[], datetime] | None = None,
    canary_level: CanaryLevel = CanaryLevel.LOCKED,
    canary_runtime_provider: Callable[[], CanaryRuntime] | None = None,
    canary_reservations: ExposureReservationBook | None = None,
) -> ReadinessBoundProductionExecutionGate:
    """Canonical builder used by production composition roots."""
    return ReadinessBoundProductionExecutionGate(
        evidence,
        evaluator=evaluator,
        clock=clock,
        canary_level=canary_level,
        canary_runtime_provider=canary_runtime_provider,
        canary_reservations=canary_reservations,
        runtime_safety_provider=runtime_safety_provider,
        max_runtime_safety_age=max_runtime_safety_age,
        max_runtime_future_skew=max_runtime_future_skew,
    )
