"""Emergency Reduce-Only approval path."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

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
from freqtrade.hedge.risk.actions import RiskActionStateMachine


class _EmergencyPreconditionFailed(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EmergencyReduceApproval:
    allowed: bool
    approved_quantity: Decimal
    reason_codes: tuple[str, ...]
    reservation: ReduceReservation | None = None
    fencing_token: int | None = None
    lease_expires_at_ms: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.allowed, bool):
            raise ValueError("allowed must be a boolean.")
        if not isinstance(self.reason_codes, tuple):
            raise ValueError("Emergency approval reason_codes must be a tuple.")
        normalized = tuple(
            reason.strip()
            for reason in self.reason_codes
            if isinstance(reason, str) and reason.strip()
        )
        if len(normalized) != len(self.reason_codes) or not normalized:
            raise ValueError("Emergency approval must contain non-empty reason codes.")
        object.__setattr__(self, "reason_codes", tuple(dict.fromkeys(normalized)))
        if self.allowed:
            if self.approved_quantity <= 0:
                raise ValueError("Allowed emergency approval must approve positive quantity.")
            if self.reservation is None or self.fencing_token is None:
                raise ValueError("Allowed emergency approval must carry reservation and fencing.")
        elif self.approved_quantity != 0:
            raise ValueError("Denied emergency approval must approve zero quantity.")


class EmergencyReduceOnlyController:
    def __init__(
        self,
        *,
        locks: PositionLockManager,
        writer: SingleWriterGuard,
        state_machine: RiskActionStateMachine,
    ) -> None:
        self._locks = locks
        self._writer = writer
        self._state_machine = state_machine

    def approve(
        self,
        *,
        account_id: str,
        symbol: str,
        position_side: PositionSide | str,
        action: PositionAction | str,
        requested_quantity: Decimal,
        confirmed_quantity: Decimal,
        pending_reduce_quantity: Decimal = Decimal("0"),
        timeout_seconds: float | None = None,
    ) -> EmergencyReduceApproval:
        normalized_action = (
            action
            if isinstance(action, PositionAction)
            else PositionAction(str(action).upper())
        )
        if normalized_action not in {PositionAction.REDUCE, PositionAction.CLOSE}:
            return EmergencyReduceApproval(False, Decimal("0"), ("EMERGENCY_INCREASE_FORBIDDEN",))
        state = self._state_machine.state
        if not state.allows(normalized_action, emergency=True):
            return EmergencyReduceApproval(False, Decimal("0"), ("RISK_STATE_FORBIDS_ACTION",))

        validated_lease: LeaseRecord | None = None

        def validate_inside_lock() -> None:
            nonlocal validated_lease
            if not self._state_machine.state.allows(normalized_action, emergency=True):
                raise _EmergencyPreconditionFailed
            validated_lease = self._writer.assert_valid()

        try:
            reservation = self._locks.reserve_reduce(
                account_id=account_id,
                symbol=symbol,
                position_side=position_side,
                requested_quantity=requested_quantity,
                confirmed_quantity=confirmed_quantity,
                existing_pending_reduce_quantity=pending_reduce_quantity,
                timeout_seconds=timeout_seconds,
                pre_reservation_check=validate_inside_lock,
            )
        except LeaseLost:
            return EmergencyReduceApproval(
                False,
                Decimal("0"),
                ("SINGLE_WRITER_LEASE_INVALID",),
            )
        except PositionLockTimeout:
            return EmergencyReduceApproval(False, Decimal("0"), ("POSITION_LOCK_TIMEOUT",))
        except DeadlockDetected:
            return EmergencyReduceApproval(
                False,
                Decimal("0"),
                ("POSITION_LOCK_DEADLOCK_DETECTED",),
            )
        except LockOrderViolation:
            return EmergencyReduceApproval(
                False,
                Decimal("0"),
                ("POSITION_LOCK_ORDER_VIOLATION",),
            )
        except _EmergencyPreconditionFailed:
            return EmergencyReduceApproval(
                False,
                Decimal("0"),
                ("RISK_STATE_FORBIDS_ACTION",),
            )
        if reservation.allowed_quantity <= 0:
            reservation.release()
            return EmergencyReduceApproval(
                False,
                Decimal("0"),
                (reservation.reason_code,),
            )
        if validated_lease is None:
            reservation.release()
            return EmergencyReduceApproval(
                False,
                Decimal("0"),
                ("SINGLE_WRITER_LEASE_INVALID",),
            )
        return EmergencyReduceApproval(
            True,
            min(requested_quantity, reservation.allowed_quantity),
            ("EMERGENCY_REDUCE_ONLY", reservation.reason_code),
            reservation,
            validated_lease.fencing_token,
            validated_lease.expires_at_ms,
        )
