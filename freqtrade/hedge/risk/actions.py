"""Risk action state machine and unified approval coordination."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
from threading import RLock
from typing import Callable, Iterable

from freqtrade.enums.hedge import PositionAction, PositionSide
from freqtrade.hedge.concurrency.database_lease import LeaseLost, LeaseRecord
from freqtrade.hedge.concurrency.lock_order import LockOrderViolation
from freqtrade.hedge.concurrency.position_lock import (
    DeadlockDetected,
    PositionLockManager,
    PositionLockTimeout,
    ReduceReservation,
)
from freqtrade.hedge.concurrency.single_writer import SingleWriterGuard
from freqtrade.hedge.readiness.gate import ReadinessGate
from freqtrade.hedge.risk.commit import ApprovalCommitRecord, RiskApprovalCommitPort
from freqtrade.hedge.risk.engine import HedgeRiskEngine
from freqtrade.hedge.risk.models import AccountRiskSnapshot, RiskRequest


class RiskMode(str, Enum):
    NORMAL = "NORMAL"
    REDUCE_ONLY = "REDUCE_ONLY"
    HALT = "HALT"


class RiskEvent(str, Enum):
    ENTER_REDUCE_ONLY = "ENTER_REDUCE_ONLY"
    ENTER_HALT = "ENTER_HALT"
    RECOVERED = "RECOVERED"
    OPERATOR_ACK = "OPERATOR_ACK"


@dataclass(frozen=True, slots=True)
class RiskActionState:
    mode: RiskMode
    reason_codes: tuple[str, ...]
    version: int

    def __post_init__(self) -> None:
        mode = self.mode if isinstance(self.mode, RiskMode) else RiskMode(str(self.mode).upper())
        object.__setattr__(self, "mode", mode)
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 0:
            raise ValueError("Risk action state version must be a nonnegative integer.")
        if not isinstance(self.reason_codes, tuple):
            raise ValueError("Risk action reason_codes must be a tuple.")
        normalized = tuple(
            reason.strip()
            for reason in self.reason_codes
            if isinstance(reason, str) and reason.strip()
        )
        if len(normalized) != len(self.reason_codes):
            raise ValueError("Risk action reason_codes must contain non-empty strings only.")
        object.__setattr__(self, "reason_codes", tuple(dict.fromkeys(normalized)))

    def allows(self, action: PositionAction, *, emergency: bool = False) -> bool:
        normalized_action = (
            action
            if isinstance(action, PositionAction)
            else PositionAction(str(action).upper())
        )
        if self.mode is RiskMode.NORMAL:
            return True
        if normalized_action in {PositionAction.REDUCE, PositionAction.CLOSE}:
            return self.mode is RiskMode.REDUCE_ONLY or emergency
        return False


class RiskActionStateMachine:
    """Thread-safe state machine with acknowledgement-gated HALT recovery."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._state = RiskActionState(RiskMode.NORMAL, (), 0)

    @property
    def state(self) -> RiskActionState:
        with self._lock:
            return self._state

    def transition(
        self,
        event: RiskEvent | str,
        *,
        reason_codes: tuple[str, ...] = (),
    ) -> RiskActionState:
        normalized_event = (
            event if isinstance(event, RiskEvent) else RiskEvent(str(event).upper())
        )
        if not isinstance(reason_codes, tuple):
            raise ValueError("reason_codes must be a tuple.")
        normalized_reasons = tuple(
            reason.strip()
            for reason in reason_codes
            if isinstance(reason, str) and reason.strip()
        )
        if len(normalized_reasons) != len(reason_codes):
            raise ValueError("reason_codes must contain non-empty strings only.")
        if (
            normalized_event in {RiskEvent.ENTER_HALT, RiskEvent.ENTER_REDUCE_ONLY}
            and not normalized_reasons
        ):
            raise ValueError("Entering a restrictive risk mode requires at least one reason code.")
        with self._lock:
            current = self._state
            if normalized_event is RiskEvent.ENTER_HALT:
                mode = RiskMode.HALT
            elif normalized_event is RiskEvent.ENTER_REDUCE_ONLY:
                mode = RiskMode.HALT if current.mode is RiskMode.HALT else RiskMode.REDUCE_ONLY
            elif normalized_event is RiskEvent.OPERATOR_ACK:
                mode = RiskMode.REDUCE_ONLY if current.mode is RiskMode.HALT else current.mode
            elif normalized_event is RiskEvent.RECOVERED:
                mode = RiskMode.NORMAL if current.mode is RiskMode.REDUCE_ONLY else current.mode
            else:  # pragma: no cover - normalization makes this unreachable.
                mode = current.mode
            if normalized_reasons:
                reasons = tuple(dict.fromkeys(normalized_reasons))
            else:
                reasons = () if mode is RiskMode.NORMAL else current.reason_codes
            self._state = RiskActionState(mode, reasons, current.version + 1)
            return self._state


@dataclass(slots=True)
class RiskApprovalReservation:
    """Local capacity reservation until a durable conditional commit succeeds."""

    coordinator: "RiskApprovalCoordinator"
    decision_id: str
    intent_id: str
    idempotency_key: str
    correlation_id: str
    risk_snapshot_id: str
    request_json: str
    risk_snapshot_json: str
    rules_version: str
    evaluated_at_ms: int
    approved_quantity: Decimal
    approved_notional: Decimal
    fencing_token: int
    target_snapshot_version: int | None = None
    intent_expires_at_ms: int | None = None
    token: str | None = None
    reduce_reservation: ReduceReservation | None = None
    expires_at_ms: int | None = None
    _released: bool = False
    _committed: bool = False
    _durable_commit_accepted: bool = False
    _commit_record: ApprovalCommitRecord | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def release(self) -> None:
        if self._released:
            return
        if self.reduce_reservation is not None:
            self.reduce_reservation.release()
        if self.token is not None:
            self.coordinator.release_increase_reservation(self.token)
        self._released = True

    def build_commit_record(self, *, durable_reference: str) -> ApprovalCommitRecord:
        if not isinstance(durable_reference, str) or not durable_reference.strip():
            raise ValueError("durable_reference must be a non-empty string.")
        normalized_reference = durable_reference.strip()
        if self._commit_record is not None:
            if self._commit_record.durable_reference != normalized_reference:
                raise RuntimeError(
                    "A risk reservation cannot be committed to two durable references."
                )
            if not self._durable_commit_accepted:
                if self.expired:
                    self.release()
                    raise RuntimeError("Cannot commit an expired risk reservation.")
                self.coordinator.assert_fencing_token(self.fencing_token)
            return self._commit_record
        if self._released and not self._committed:
            raise RuntimeError("Cannot commit a released risk reservation.")
        if self.expired:
            self.release()
            raise RuntimeError("Cannot commit an expired risk reservation.")
        self.coordinator.assert_fencing_token(self.fencing_token)
        record = ApprovalCommitRecord(
            decision_id=self.decision_id,
            intent_id=self.intent_id,
            idempotency_key=self.idempotency_key,
            correlation_id=self.correlation_id,
            risk_snapshot_id=self.risk_snapshot_id,
            request_json=self.request_json,
            risk_snapshot_json=self.risk_snapshot_json,
            rules_version=self.rules_version,
            fencing_token=self.fencing_token,
            approved_quantity=self.approved_quantity,
            approved_notional=self.approved_notional,
            evaluated_at_ms=self.evaluated_at_ms,
            committed_at_ms=self.coordinator.current_time_ms(),
            durable_reference=normalized_reference,
            target_snapshot_version=self.target_snapshot_version,
            intent_expires_at_ms=self.intent_expires_at_ms,
        )
        self._commit_record = record
        return record

    def mark_durable_commit_accepted(self) -> None:
        self._durable_commit_accepted = True

    def mark_committed(self) -> None:
        self._durable_commit_accepted = True
        self._committed = True
        self.release()

    def confirm(
        self,
        *,
        commit_port: RiskApprovalCommitPort,
        durable_reference: str,
    ) -> ApprovalCommitRecord:
        """Transfer ownership to the ledger before releasing local capacity."""

        record = self.build_commit_record(durable_reference=durable_reference)
        if not self._committed and not self._durable_commit_accepted:
            if not commit_port.commit_approval(record):
                raise RuntimeError("Durable risk approval commit was rejected.")
            self.mark_durable_commit_accepted()
        persisted = commit_port.read_commit(self.decision_id)
        if persisted != record:
            raise RuntimeError("Durable risk approval commit could not be verified.")
        if not self._committed:
            self.mark_committed()
        return record

    @property
    def expired(self) -> bool:
        if self.expires_at_ms is not None:
            return self.coordinator.current_time_ms() >= self.expires_at_ms
        if self.reduce_reservation is not None:
            return self.reduce_reservation.expired
        return False

    @property
    def committed(self) -> bool:
        return self._committed

    def __enter__(self) -> "RiskApprovalReservation":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._committed:
            self.release()


@dataclass(frozen=True, slots=True)
class UnifiedRiskApproval:
    allowed: bool
    approved_quantity: Decimal
    approved_notional: Decimal
    reason_codes: tuple[str, ...]
    reservation: RiskApprovalReservation | None = None
    fencing_token: int | None = None
    lease_expires_at_ms: int | None = None
    reservation_expires_at_ms: int | None = None
    decision_id: str | None = None
    intent_id: str | None = None
    idempotency_key: str | None = None
    risk_snapshot_id: str | None = None
    correlation_id: str | None = None
    rules_version: str | None = None
    target_snapshot_version: int | None = None
    evaluated_at_ms: int | None = None
    intent_expires_at_ms: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.allowed, bool):
            raise ValueError("allowed must be a boolean.")
        if not isinstance(self.reason_codes, tuple):
            raise ValueError("Approval reason_codes must be a tuple.")
        normalized_reasons = tuple(
            reason.strip()
            for reason in self.reason_codes
            if isinstance(reason, str) and reason.strip()
        )
        if len(normalized_reasons) != len(self.reason_codes) or not normalized_reasons:
            raise ValueError("Approval must contain non-empty reason codes.")
        object.__setattr__(self, "reason_codes", tuple(dict.fromkeys(normalized_reasons)))
        if self.allowed:
            if self.approved_quantity <= 0 or self.approved_notional <= 0:
                raise ValueError("Allowed approval must contain positive quantity and notional.")
            if self.reservation is None or self.fencing_token is None:
                raise ValueError("Allowed approval must carry reservation and fencing token.")
            if (
                isinstance(self.fencing_token, bool)
                or not isinstance(self.fencing_token, int)
                or self.fencing_token <= 0
            ):
                raise ValueError("fencing_token must be a positive integer.")
            for field_name in (
                "decision_id",
                "intent_id",
                "idempotency_key",
                "risk_snapshot_id",
                "correlation_id",
                "rules_version",
            ):
                value = getattr(self, field_name)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"Allowed approval must carry {field_name}.")
                object.__setattr__(self, field_name, value.strip())
            if (
                isinstance(self.lease_expires_at_ms, bool)
                or not isinstance(self.lease_expires_at_ms, int)
                or self.lease_expires_at_ms <= 0
            ):
                raise ValueError("Allowed approval must carry a positive lease expiry.")
            if (
                isinstance(self.evaluated_at_ms, bool)
                or not isinstance(self.evaluated_at_ms, int)
                or self.evaluated_at_ms < 0
            ):
                raise ValueError("Allowed approval must carry an evaluation timestamp.")
        elif self.approved_quantity != 0 or self.approved_notional != 0:
            raise ValueError("Denied approval must approve zero quantity and notional.")
        elif any(
            value is not None
            for value in (
                self.reservation,
                self.fencing_token,
                self.lease_expires_at_ms,
                self.reservation_expires_at_ms,
                self.decision_id,
                self.intent_id,
                self.idempotency_key,
                self.risk_snapshot_id,
                self.correlation_id,
                self.rules_version,
                self.target_snapshot_version,
                self.evaluated_at_ms,
                self.intent_expires_at_ms,
            )
        ):
            raise ValueError("Denied approval must not carry reservation or lease metadata.")

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "approved_quantity": str(self.approved_quantity),
            "approved_notional": str(self.approved_notional),
            "reason_codes": list(self.reason_codes),
            "fencing_token": self.fencing_token,
            "lease_expires_at_ms": self.lease_expires_at_ms,
            "reservation_expires_at_ms": self.reservation_expires_at_ms,
            "decision_id": self.decision_id,
            "intent_id": self.intent_id,
            "idempotency_key": self.idempotency_key,
            "risk_snapshot_id": self.risk_snapshot_id,
            "correlation_id": self.correlation_id,
            "rules_version": self.rules_version,
            "target_snapshot_version": self.target_snapshot_version,
            "evaluated_at_ms": self.evaluated_at_ms,
            "intent_expires_at_ms": self.intent_expires_at_ms,
        }


@dataclass(slots=True)
class UnifiedRiskBatchApproval:
    allowed: bool
    approvals: tuple[UnifiedRiskApproval, ...]
    reason_codes: tuple[str, ...]
    _released: bool = False

    def __post_init__(self) -> None:
        self.approvals = tuple(self.approvals)
        self.reason_codes = tuple(dict.fromkeys(self.reason_codes))
        if not self.reason_codes:
            raise ValueError("Batch approval must contain at least one reason code.")
        if self.allowed and (
            not self.approvals or not all(item.allowed for item in self.approvals)
        ):
            raise ValueError("Allowed batch must contain only allowed approvals.")
        if not self.allowed and any(item.allowed for item in self.approvals):
            raise ValueError("Denied batch must not expose usable allowed approvals.")

    def release(self) -> None:
        if self._released:
            return
        for approval in self.approvals:
            if approval.reservation is not None:
                approval.reservation.release()
        self._released = True

    def confirm(
        self,
        *,
        commit_port: RiskApprovalCommitPort,
        durable_references: tuple[str, ...],
    ) -> tuple[ApprovalCommitRecord, ...]:
        if len(durable_references) != len(self.approvals):
            raise ValueError("durable_references must match approval count.")
        reservations: list[RiskApprovalReservation] = []
        records: list[ApprovalCommitRecord] = []
        durable_commit_accepted = False
        try:
            for approval, durable_reference in zip(
                self.approvals,
                durable_references,
                strict=True,
            ):
                if approval.reservation is None:
                    raise RuntimeError("Allowed approval is missing reservation.")
                reservations.append(approval.reservation)
                records.append(
                    approval.reservation.build_commit_record(
                        durable_reference=durable_reference,
                    )
                )
            record_tuple = tuple(records)
            if not commit_port.commit_approval_batch(record_tuple):
                raise RuntimeError("Durable risk approval batch commit was rejected.")
            durable_commit_accepted = True
            for reservation in reservations:
                reservation.mark_durable_commit_accepted()
            if any(
                commit_port.read_commit(record.decision_id) != record
                for record in record_tuple
            ):
                raise RuntimeError("Durable risk approval batch commit verification failed.")
            for reservation in reservations:
                reservation.mark_committed()
        except Exception:
            # Before a durable commit is accepted, local reservations can be rolled
            # back safely.  After acceptance, verification can fail because the
            # database/read path is temporarily unavailable; retaining reservations
            # is the conservative choice and prevents pending risk from disappearing.
            if not durable_commit_accepted:
                self.release()
            raise
        self._released = True
        return tuple(records)

    def as_dict(self) -> dict[str, object]:
        return {
            "allowed": self.allowed,
            "reason_codes": list(self.reason_codes),
            "approvals": [item.as_dict() for item in self.approvals],
        }

    def __enter__(self) -> "UnifiedRiskBatchApproval":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


@dataclass(frozen=True, slots=True)
class _IncreaseReservationRecord:
    account_id: str
    symbol: str
    position_side: PositionSide
    notional: Decimal
    initial_margin: Decimal
    maintenance_margin: Decimal
    signed_net_notional_delta: Decimal
    created_at_ms: int
    expires_at_ms: int


class _ApprovalPreconditionFailed(RuntimeError):
    def __init__(self, *reason_codes: str) -> None:
        super().__init__(",".join(reason_codes))
        self.reason_codes = tuple(reason_codes)


class RiskApprovalCoordinator:
    """Single entry point for every order-risk approval.

    The coordinator revalidates the database lease on each approval, enforces
    ReadinessGate and the risk action state machine, serializes local increase
    capacity reservations, and performs reduce clipping under the side lock.
    Every allowed result carries the fencing token that the persistence layer
    must include in its conditional write.
    """

    def __init__(
        self,
        *,
        engine: HedgeRiskEngine,
        locks: PositionLockManager,
        writer: SingleWriterGuard,
        readiness: ReadinessGate,
        state_machine: RiskActionStateMachine,
        reservation_ttl_ms: int = 30_000,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if (
            isinstance(reservation_ttl_ms, bool)
            or not isinstance(reservation_ttl_ms, int)
            or reservation_ttl_ms <= 0
        ):
            raise ValueError("reservation_ttl_ms must be a positive integer.")
        self._engine = engine
        self._locks = locks
        self._writer = writer
        self._readiness = readiness
        self._state_machine = state_machine
        self._reservation_ttl_ms = reservation_ttl_ms
        self._clock_ms = clock_ms or (lambda: time.monotonic_ns() // 1_000_000)
        self._batch_lock = RLock()
        self._reservation_lock = RLock()
        self._increase_reservations: dict[str, _IncreaseReservationRecord] = {}

    def current_time_ms(self) -> int:
        value = self._clock_ms()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Reservation clock must return nonnegative integer milliseconds.")
        return value

    def _prune_expired_locked(self, now_ms: int | None = None) -> int:
        now = self.current_time_ms() if now_ms is None else now_ms
        expired = [
            token
            for token, item in self._increase_reservations.items()
            if item.expires_at_ms <= now
        ]
        for token in expired:
            self._increase_reservations.pop(token, None)
        return len(expired)

    def prune_expired_reservations(self) -> int:
        with self._reservation_lock:
            increases = self._prune_expired_locked()
        return increases + self._locks.prune_expired_reservations()

    @staticmethod
    def _deny(*reason_codes: str) -> UnifiedRiskApproval:
        return UnifiedRiskApproval(
            False,
            Decimal("0"),
            Decimal("0"),
            tuple(dict.fromkeys(reason_codes)) or ("RISK_APPROVAL_DENIED",),
        )

    def _assert_writer(self) -> tuple[LeaseRecord | None, str | None]:
        try:
            return self._writer.assert_valid(), None
        except LeaseLost:
            return None, "SINGLE_WRITER_LEASE_INVALID"

    def assert_fencing_token(self, fencing_token: int) -> LeaseRecord:
        lease = self._writer.assert_valid()
        if lease.fencing_token != fencing_token:
            raise LeaseLost("Risk approval fencing token is no longer current.")
        return lease

    def _increase_precondition_reasons(self, request: RiskRequest) -> tuple[str, ...]:
        report = self._readiness.report
        if not self._readiness.allows_new_risk(request.position_key):
            return (
                "READINESS_NOT_READY",
                *(code.value for code in report.reason_codes),
            )
        if not self._state_machine.state.allows(request.action):
            return ("RISK_STATE_FORBIDS_ACTION",)
        return ()

    def _reduce_precondition_reasons(self, request: RiskRequest) -> tuple[str, ...]:
        if not self._readiness.allows_controlled_reduce(request.position_key):
            report = self._readiness.report
            return (
                "CONTROLLED_REDUCE_NOT_READY",
                *(code.value for code in report.reason_codes),
            )
        if not self._state_machine.state.allows(request.action, emergency=True):
            return ("RISK_STATE_FORBIDS_ACTION",)
        if request.confirmed_quantity is None:
            return ("CONFIRMED_POSITION_REQUIRED",)
        return ()

    def approve(
        self,
        *,
        request: RiskRequest,
        account: AccountRiskSnapshot,
        timeout_seconds: float | None = None,
    ) -> UnifiedRiskApproval:
        if request.account_id != account.account_id:
            return self._deny("ACCOUNT_ID_MISMATCH")
        if not account.effective_risk_data_valid:
            return self._deny(*(account.risk_data_errors or ("RISK_DATA_INVALID",)))
        now_ms = self.current_time_ms()
        if request.expires_at_ms is not None and now_ms >= request.expires_at_ms:
            return self._deny("RISK_INTENT_EXPIRED")
        lease, writer_reason = self._assert_writer()
        if lease is None:
            return self._deny(writer_reason or "SINGLE_WRITER_LEASE_INVALID")

        if request.action in {PositionAction.REDUCE, PositionAction.CLOSE}:
            return self._approve_reduce(request, account, timeout_seconds=timeout_seconds)
        return self._approve_increase(request, account)

    def _approve_reduce(
        self,
        request: RiskRequest,
        account: AccountRiskSnapshot,
        *,
        timeout_seconds: float | None,
    ) -> UnifiedRiskApproval:
        reasons = self._reduce_precondition_reasons(request)
        if reasons:
            return self._deny(*reasons)
        if request.confirmed_quantity is None:
            return self._deny("CONFIRMED_POSITION_REQUIRED")

        validated_lease: LeaseRecord | None = None

        def validate_inside_lock() -> None:
            nonlocal validated_lease
            inner_reasons = self._reduce_precondition_reasons(request)
            if inner_reasons:
                raise _ApprovalPreconditionFailed(*inner_reasons)
            validated_lease = self._writer.assert_valid()

        try:
            reservation = self._locks.reserve_reduce(
                account_id=request.account_id,
                symbol=request.symbol,
                position_side=request.position_side,
                requested_quantity=request.requested_quantity,
                exchange=request.exchange,
                confirmed_quantity=request.confirmed_quantity,
                existing_pending_reduce_quantity=request.pending_reduce_quantity,
                timeout_seconds=timeout_seconds,
                pre_reservation_check=validate_inside_lock,
            )
        except LeaseLost:
            return self._deny("SINGLE_WRITER_LEASE_INVALID")
        except _ApprovalPreconditionFailed as exc:
            return self._deny(*exc.reason_codes)
        except PositionLockTimeout:
            return self._deny("POSITION_LOCK_TIMEOUT")
        except DeadlockDetected:
            return self._deny("POSITION_LOCK_DEADLOCK_DETECTED")
        except LockOrderViolation:
            return self._deny("POSITION_LOCK_ORDER_VIOLATION")

        approved = min(request.requested_quantity, reservation.allowed_quantity)
        if approved <= 0:
            reservation.release()
            return self._deny(reservation.reason_code)
        if validated_lease is None:
            reservation.release()
            return self._deny("SINGLE_WRITER_LEASE_INVALID")
        approved_notional = approved * request.reference_price
        decision_id = uuid.uuid4().hex
        risk_snapshot_id = account.snapshot_id
        evaluated_at_ms = self.current_time_ms()
        rules_version = "direction3-risk-v1.4"
        approval_reservation = RiskApprovalReservation(
            coordinator=self,
            decision_id=decision_id,
            intent_id=request.intent_id,
            idempotency_key=request.idempotency_key,
            correlation_id=request.correlation_id,
            risk_snapshot_id=risk_snapshot_id,
            request_json=json.dumps(
                request.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            risk_snapshot_json=json.dumps(
                account.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            rules_version=rules_version,
            evaluated_at_ms=evaluated_at_ms,
            approved_quantity=approved,
            approved_notional=approved_notional,
            fencing_token=validated_lease.fencing_token,
            target_snapshot_version=request.target_snapshot_version,
            intent_expires_at_ms=request.expires_at_ms,
            reduce_reservation=reservation,
        )
        return UnifiedRiskApproval(
            allowed=True,
            approved_quantity=approved,
            approved_notional=approved_notional,
            reason_codes=("CONTROLLED_REDUCE_APPROVED", reservation.reason_code),
            reservation=approval_reservation,
            fencing_token=validated_lease.fencing_token,
            lease_expires_at_ms=validated_lease.expires_at_ms,
            reservation_expires_at_ms=None,
            decision_id=decision_id,
            intent_id=request.intent_id,
            idempotency_key=request.idempotency_key,
            risk_snapshot_id=risk_snapshot_id,
            correlation_id=request.correlation_id,
            rules_version=rules_version,
            target_snapshot_version=request.target_snapshot_version,
            evaluated_at_ms=evaluated_at_ms,
            intent_expires_at_ms=request.expires_at_ms,
        )

    def _approve_increase(
        self,
        request: RiskRequest,
        account: AccountRiskSnapshot,
    ) -> UnifiedRiskApproval:
        reasons = self._increase_precondition_reasons(request)
        if reasons:
            return self._deny(*reasons)

        with self._reservation_lock:
            now_ms = self.current_time_ms()
            self._prune_expired_locked(now_ms)
            reasons = self._increase_precondition_reasons(request)
            if reasons:
                return self._deny(*reasons)
            lease, writer_reason = self._assert_writer()
            if lease is None:
                return self._deny(writer_reason or "SINGLE_WRITER_LEASE_INVALID")

            local_account = Decimal("0")
            local_initial_margin = Decimal("0")
            local_maintenance_margin = Decimal("0")
            local_net_delta = Decimal("0")
            local_symbol = Decimal("0")
            local_leg = Decimal("0")
            local_long = Decimal("0")
            local_short = Decimal("0")
            for item in self._increase_reservations.values():
                if item.account_id != request.account_id:
                    continue
                local_account += item.notional
                local_initial_margin += item.initial_margin
                local_maintenance_margin += item.maintenance_margin
                local_net_delta += item.signed_net_notional_delta
                if item.position_side is PositionSide.LONG:
                    local_long += item.notional
                else:
                    local_short += item.notional
                if item.symbol != request.symbol:
                    continue
                local_symbol += item.notional
                if item.position_side == request.position_side:
                    local_leg += item.notional

            account_with_pending = replace(
                account,
                available_balance=max(
                    account.available_balance - local_initial_margin,
                    Decimal("0"),
                ),
                pending_order_notional=account.pending_order_notional + local_account,
                pending_order_initial_margin=(
                    account.pending_order_initial_margin + local_initial_margin
                ),
                pending_order_maintenance_margin=(
                    account.pending_order_maintenance_margin + local_maintenance_margin
                ),
                pending_net_notional_delta=(
                    account.pending_net_notional_delta + local_net_delta
                ),
                pending_long_notional=account.pending_long_notional + local_long,
                pending_short_notional=account.pending_short_notional + local_short,
            )
            request_with_pending = replace(
                request,
                pending_leg_increase_notional=(
                    request.pending_leg_increase_notional + local_leg
                ),
                pending_symbol_increase_notional=(
                    request.pending_symbol_increase_notional + local_symbol
                ),
            )
            decision = self._engine.evaluate_request(
                request=request_with_pending,
                account=account_with_pending,
            )
            approved = min(request.requested_quantity, decision.approved_quantity)
            approved_notional = min(
                request.requested_notional,
                decision.approved_notional,
                approved * request.reference_price,
            )
            if not decision.allowed or approved <= 0 or approved_notional <= 0:
                return self._deny(*decision.reason_codes)

            # Revalidate all mutable gates immediately before reserving capacity.
            reasons = self._increase_precondition_reasons(request)
            if reasons:
                return self._deny(*reasons)
            lease, writer_reason = self._assert_writer()
            if lease is None:
                return self._deny(writer_reason or "SINGLE_WRITER_LEASE_INVALID")

            token = uuid.uuid4().hex
            initial_margin = approved_notional / request.leverage
            maintenance_margin = approved_notional * request.maintenance_margin_rate
            signed_delta = request.net_notional_sign * approved_notional
            reservation_created_at_ms = self.current_time_ms()
            reservation_expires_at_ms = (
                reservation_created_at_ms + self._reservation_ttl_ms
            )
            self._increase_reservations[token] = _IncreaseReservationRecord(
                request.account_id,
                request.symbol,
                request.position_side,
                approved_notional,
                initial_margin,
                maintenance_margin,
                signed_delta,
                reservation_created_at_ms,
                reservation_expires_at_ms,
            )
            decision_id = decision.decision_id
            risk_snapshot_id = decision.risk_snapshot_id or account.snapshot_id
            evaluated_at_ms = self.current_time_ms()
            approval_reservation = RiskApprovalReservation(
                coordinator=self,
                decision_id=decision_id,
                intent_id=request.intent_id,
                idempotency_key=request.idempotency_key,
                correlation_id=request.correlation_id,
                risk_snapshot_id=risk_snapshot_id,
                request_json=json.dumps(
                    request.as_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                risk_snapshot_json=json.dumps(
                    account.as_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
                rules_version=decision.rules_version,
                evaluated_at_ms=evaluated_at_ms,
                approved_quantity=approved,
                approved_notional=approved_notional,
                fencing_token=lease.fencing_token,
                target_snapshot_version=request.target_snapshot_version,
                intent_expires_at_ms=request.expires_at_ms,
                token=token,
                expires_at_ms=reservation_expires_at_ms,
            )
            return UnifiedRiskApproval(
                allowed=True,
                approved_quantity=approved,
                approved_notional=approved_notional,
                reason_codes=decision.reason_codes,
                reservation=approval_reservation,
                fencing_token=lease.fencing_token,
                lease_expires_at_ms=lease.expires_at_ms,
                reservation_expires_at_ms=reservation_expires_at_ms,
                decision_id=decision_id,
                intent_id=request.intent_id,
                idempotency_key=request.idempotency_key,
                risk_snapshot_id=risk_snapshot_id,
                correlation_id=request.correlation_id,
                rules_version=decision.rules_version,
                target_snapshot_version=request.target_snapshot_version,
                evaluated_at_ms=evaluated_at_ms,
                intent_expires_at_ms=request.expires_at_ms,
            )

    def release_increase_reservation(self, token: str) -> None:
        with self._reservation_lock:
            self._increase_reservations.pop(token, None)

    def pending_increase_notional(self, *, account_id: str) -> Decimal:
        if not isinstance(account_id, str):
            raise ValueError("account_id must be a string.")
        normalized_account = account_id.strip()
        if not normalized_account:
            raise ValueError("account_id must not be empty.")
        with self._reservation_lock:
            self._prune_expired_locked()
            return sum(
                (
                    item.notional
                    for item in self._increase_reservations.values()
                    if item.account_id == normalized_account
                ),
                Decimal("0"),
            )

    def release_all_reservations(self, *, account_id: str | None = None) -> int:
        normalized = None
        if account_id is not None:
            if not isinstance(account_id, str) or not account_id.strip():
                raise ValueError("account_id must be a non-empty string.")
            normalized = account_id.strip()
        with self._reservation_lock:
            self._prune_expired_locked()
            tokens = [
                token
                for token, item in self._increase_reservations.items()
                if normalized is None or item.account_id == normalized
            ]
            for token in tokens:
                self._increase_reservations.pop(token, None)
        return len(tokens) + self._locks.release_all_reservations(account_id=normalized)

    def reservation_snapshot(self) -> dict[str, object]:
        now = self.current_time_ms()
        with self._reservation_lock:
            self._prune_expired_locked(now)
            increases = [
                {
                    "token": token,
                    "account_id": item.account_id,
                    "symbol": item.symbol,
                    "position_side": item.position_side.value,
                    "notional": str(item.notional),
                    "initial_margin": str(item.initial_margin),
                    "maintenance_margin": str(item.maintenance_margin),
                    "expires_at_ms": item.expires_at_ms,
                    "expires_in_ms": max(item.expires_at_ms - now, 0),
                }
                for token, item in self._increase_reservations.items()
            ]
        return {
            "increase": sorted(
                increases,
                key=lambda item: (
                    str(item["account_id"]),
                    str(item["symbol"]),
                    str(item["position_side"]),
                    str(item["token"]),
                ),
            ),
            "reduce": list(self._locks.reservation_snapshot()),
        }

    def approve_batch(
        self,
        items: Iterable[tuple[RiskRequest, AccountRiskSnapshot]],
        *,
        timeout_seconds: float | None = None,
    ) -> UnifiedRiskBatchApproval:
        """Approve a sequence with rollback-atomic local reservations.

        Every successful item remains reserved until the batch is confirmed or
        released. Any denial rolls back earlier reservations and returns no
        usable approvals. Separate position sides remain independently operable.
        """

        with self._batch_lock:
            normalized_items = tuple(items)
            if not normalized_items:
                return UnifiedRiskBatchApproval(False, (), ("EMPTY_RISK_BATCH",))
            approvals: list[UnifiedRiskApproval] = []
            for index, item in enumerate(normalized_items):
                try:
                    request, account = item
                except (TypeError, ValueError) as exc:
                    for approval in approvals:
                        if approval.reservation is not None:
                            approval.reservation.release()
                    denied = tuple(
                        self._deny("BATCH_ROLLED_BACK") for _ in range(index)
                    ) + (self._deny("INVALID_BATCH_ITEM"),) + tuple(
                        self._deny("BATCH_NOT_EVALUATED")
                        for _ in range(len(normalized_items) - index - 1)
                    )
                    return UnifiedRiskBatchApproval(
                        False, denied, ("INVALID_BATCH_ITEM", type(exc).__name__)
                    )
                approval = self.approve(
                    request=request,
                    account=account,
                    timeout_seconds=timeout_seconds,
                )
                if not approval.allowed:
                    for previous in approvals:
                        if previous.reservation is not None:
                            previous.reservation.release()
                    denied = tuple(
                        self._deny("BATCH_ROLLED_BACK") for _ in approvals
                    ) + (approval,) + tuple(
                        self._deny("BATCH_NOT_EVALUATED")
                        for _ in range(len(normalized_items) - index - 1)
                    )
                    return UnifiedRiskBatchApproval(
                        False,
                        denied,
                        ("BATCH_APPROVAL_DENIED", *approval.reason_codes),
                    )
                approvals.append(approval)
            return UnifiedRiskBatchApproval(
                True,
                tuple(approvals),
                ("BATCH_APPROVED",),
            )
