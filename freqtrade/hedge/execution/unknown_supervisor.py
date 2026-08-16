"""Retry scheduling and escalation for orders whose venue outcome is UNKNOWN."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import RLock
from typing import Protocol

from freqtrade.hedge.contracts.ports import ClockPort, SystemClock

from .service import ExecutionResult
from .state_machine import OrderState


class UnknownRecoveryState(StrEnum):
    PENDING = "PENDING"
    QUERYING = "QUERYING"
    RETRY_WAIT = "RETRY_WAIT"
    RESOLVED = "RESOLVED"
    HALTED = "HALTED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass(frozen=True, slots=True)
class UnknownRecoveryRecord:
    client_order_id: str
    state: UnknownRecoveryState
    first_unknown_at: datetime
    last_attempt_at: datetime | None = None
    next_retry_at: datetime | None = None
    attempts: int = 0
    last_message: str = ""
    resolved_state: OrderState | None = None


class UnknownRecoveryExecutionPort(Protocol):
    def resolve_unknown(self, client_order_id: str) -> ExecutionResult: ...


class UnknownOrderSupervisor:
    """Deterministic UNKNOWN retry coordinator.

    It never resubmits. Each due attempt delegates to ``resolve_unknown`` which must query
    venue/user-stream facts. A deadline or attempt exhaustion escalates to a durable HALTED
    record that can be surfaced by Readiness and the control plane.
    """

    def __init__(
        self,
        execution: UnknownRecoveryExecutionPort,
        *,
        clock: ClockPort | None = None,
        initial_backoff: timedelta = timedelta(seconds=2),
        maximum_backoff: timedelta = timedelta(minutes=2),
        maximum_attempts: int = 12,
        recovery_deadline: timedelta = timedelta(minutes=15),
    ) -> None:
        if initial_backoff.total_seconds() <= 0:
            raise ValueError("initial_backoff must be positive")
        if maximum_backoff < initial_backoff:
            raise ValueError("maximum_backoff cannot be below initial_backoff")
        if not isinstance(maximum_attempts, int) or maximum_attempts <= 0:
            raise ValueError("maximum_attempts must be positive")
        if recovery_deadline.total_seconds() <= 0:
            raise ValueError("recovery_deadline must be positive")
        self._execution = execution
        self._clock = clock or SystemClock()
        self._initial_backoff = initial_backoff
        self._maximum_backoff = maximum_backoff
        self._maximum_attempts = maximum_attempts
        self._recovery_deadline = recovery_deadline
        self._records: dict[str, UnknownRecoveryRecord] = {}
        self._lock = RLock()

    def register(self, client_order_id: str) -> UnknownRecoveryRecord:
        if not isinstance(client_order_id, str) or not client_order_id.strip():
            raise ValueError("client_order_id is required")
        key = client_order_id.strip()
        now = self._clock.now()
        with self._lock:
            current = self._records.get(key)
            if current is not None and current.state not in {
                UnknownRecoveryState.RESOLVED,
                UnknownRecoveryState.HALTED,
                UnknownRecoveryState.MANUAL_REVIEW,
            }:
                return current
            record = UnknownRecoveryRecord(
                client_order_id=key,
                state=UnknownRecoveryState.PENDING,
                first_unknown_at=now,
                next_retry_at=now,
            )
            self._records[key] = record
            return record

    def restore(
        self,
        client_order_id: str,
        *,
        first_unknown_at: datetime,
        attempts: int = 0,
        last_message: str = "RESTORED_FROM_DURABLE_UNKNOWN_ORDER",
    ) -> UnknownRecoveryRecord:
        """Restore scheduling state from a durable UNKNOWN execution order on restart."""
        if not isinstance(client_order_id, str) or not client_order_id.strip():
            raise ValueError("client_order_id is required")
        if first_unknown_at.tzinfo is None:
            raise ValueError("first_unknown_at must be timezone-aware")
        if attempts < 0:
            raise ValueError("attempts must be nonnegative")
        now = self._clock.now()
        key = client_order_id.strip()
        with self._lock:
            current = self._records.get(key)
            if current is not None:
                return current
            record = UnknownRecoveryRecord(
                client_order_id=key,
                state=UnknownRecoveryState.PENDING,
                first_unknown_at=first_unknown_at.astimezone(UTC),
                next_retry_at=now,
                attempts=attempts,
                last_message=last_message,
            )
            if self._expired(record, now):
                record = replace(
                    record,
                    state=UnknownRecoveryState.HALTED,
                    next_retry_at=None,
                    last_message="RESTORED_UNKNOWN_ALREADY_EXPIRED",
                )
            self._records[key] = record
            return record

    def get(self, client_order_id: str) -> UnknownRecoveryRecord | None:
        with self._lock:
            return self._records.get(client_order_id)

    def list_records(self) -> tuple[UnknownRecoveryRecord, ...]:
        with self._lock:
            return tuple(sorted(self._records.values(), key=lambda item: item.client_order_id))

    def due(self, now: datetime | None = None) -> tuple[UnknownRecoveryRecord, ...]:
        current_time = now or self._clock.now()
        with self._lock:
            return tuple(
                item
                for item in self._records.values()
                if item.state in {
                    UnknownRecoveryState.PENDING,
                    UnknownRecoveryState.RETRY_WAIT,
                }
                and (item.next_retry_at is None or item.next_retry_at <= current_time)
            )

    def run_due(self) -> tuple[UnknownRecoveryRecord, ...]:
        return tuple(self.attempt(item.client_order_id) for item in self.due())

    def attempt(self, client_order_id: str) -> UnknownRecoveryRecord:
        now = self._clock.now()
        with self._lock:
            current = self._records.get(client_order_id)
            if current is None:
                raise KeyError(client_order_id)
            if current.state in {
                UnknownRecoveryState.RESOLVED,
                UnknownRecoveryState.HALTED,
                UnknownRecoveryState.MANUAL_REVIEW,
            }:
                return current
            if current.next_retry_at is not None and current.next_retry_at > now:
                return current
            if self._expired(current, now):
                halted = replace(
                    current,
                    state=UnknownRecoveryState.HALTED,
                    last_attempt_at=now,
                    next_retry_at=None,
                    last_message="UNKNOWN_RECOVERY_DEADLINE_EXCEEDED",
                )
                self._records[client_order_id] = halted
                return halted
            querying = replace(
                current,
                state=UnknownRecoveryState.QUERYING,
                attempts=current.attempts + 1,
                last_attempt_at=now,
                next_retry_at=None,
            )
            self._records[client_order_id] = querying

        try:
            result = self._execution.resolve_unknown(client_order_id)
        except Exception as exc:
            message = f"{type(exc).__name__}:{exc}"
            status = OrderState.UNKNOWN
        else:
            message = result.message
            status = result.order.lifecycle.status

        completed_at = self._clock.now()
        with self._lock:
            latest = self._records[client_order_id]
            if status is not OrderState.UNKNOWN:
                resolved = replace(
                    latest,
                    state=UnknownRecoveryState.RESOLVED,
                    next_retry_at=None,
                    last_message=message,
                    resolved_state=status,
                )
                self._records[client_order_id] = resolved
                return resolved
            if self._expired(latest, completed_at):
                halted = replace(
                    latest,
                    state=UnknownRecoveryState.HALTED,
                    next_retry_at=None,
                    last_message=message or "UNKNOWN_RECOVERY_EXHAUSTED",
                )
                self._records[client_order_id] = halted
                return halted
            delay = min(
                self._initial_backoff * (2 ** max(latest.attempts - 1, 0)),
                self._maximum_backoff,
            )
            waiting = replace(
                latest,
                state=UnknownRecoveryState.RETRY_WAIT,
                next_retry_at=completed_at + delay,
                last_message=message or "UNKNOWN_REMAINS_UNRESOLVED",
            )
            self._records[client_order_id] = waiting
            return waiting

    def mark_manual_review(self, client_order_id: str, reason: str) -> UnknownRecoveryRecord:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("manual review reason is required")
        with self._lock:
            current = self._records.get(client_order_id)
            if current is None:
                raise KeyError(client_order_id)
            updated = replace(
                current,
                state=UnknownRecoveryState.MANUAL_REVIEW,
                next_retry_at=None,
                last_message=reason.strip(),
            )
            self._records[client_order_id] = updated
            return updated

    def _expired(self, record: UnknownRecoveryRecord, now: datetime) -> bool:
        return (
            record.attempts >= self._maximum_attempts
            or now - record.first_unknown_at >= self._recovery_deadline
        )
