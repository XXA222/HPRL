"""Durable submission-recovery state machine for ambiguous exchange writes.

This layer converts transport observations into a bounded query/retry protocol.  It never
permits a blind resubmit after a request may have reached the exchange.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from .submission import SubmissionClass, SubmissionObservation, classify_submission


class SubmissionRecoveryState(StrEnum):
    NEW = "NEW"
    ACKED = "ACKED"
    REJECTED = "REJECTED"
    QUERY_REQUIRED = "QUERY_REQUIRED"
    RETRY_ALLOWED = "RETRY_ALLOWED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


@dataclass(frozen=True, slots=True)
class SubmissionRecoveryPolicy:
    max_query_attempts: int = 6
    max_retry_not_sent: int = 3
    recovery_deadline: timedelta = timedelta(minutes=2)
    retry_backoff: timedelta = timedelta(seconds=2)

    def __post_init__(self) -> None:
        if self.max_query_attempts <= 0 or self.max_retry_not_sent < 0:
            raise ValueError("invalid submission recovery attempt limits")
        if self.recovery_deadline <= timedelta(0) or self.retry_backoff <= timedelta(0):
            raise ValueError("recovery durations must be positive")


@dataclass(frozen=True, slots=True)
class SubmissionRecoveryRecord:
    client_order_id: str
    first_seen_at: datetime
    updated_at: datetime
    state: SubmissionRecoveryState = SubmissionRecoveryState.NEW
    query_attempts: int = 0
    retry_attempts: int = 0
    exchange_order_id: str | None = None
    last_classification: SubmissionClass | None = None
    next_action_at: datetime | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.client_order_id.strip():
            raise ValueError("client_order_id is required")
        if self.first_seen_at.tzinfo is None or self.updated_at.tzinfo is None:
            raise ValueError("submission recovery timestamps must be timezone-aware")
        if self.updated_at < self.first_seen_at:
            raise ValueError("updated_at cannot precede first_seen_at")
        if self.query_attempts < 0 or self.retry_attempts < 0:
            raise ValueError("attempt counts must be nonnegative")
        object.__setattr__(self, "first_seen_at", self.first_seen_at.astimezone(UTC))
        object.__setattr__(self, "updated_at", self.updated_at.astimezone(UTC))
        if self.next_action_at is not None:
            if self.next_action_at.tzinfo is None:
                raise ValueError("next_action_at must be timezone-aware")
            object.__setattr__(self, "next_action_at", self.next_action_at.astimezone(UTC))

    @property
    def terminal(self) -> bool:
        return self.state in {
            SubmissionRecoveryState.ACKED,
            SubmissionRecoveryState.REJECTED,
            SubmissionRecoveryState.MANUAL_REVIEW,
        }

    @property
    def permits_new_risk(self) -> bool:
        return self.state in {SubmissionRecoveryState.ACKED, SubmissionRecoveryState.REJECTED}


class SubmissionRecoveryMachine:
    def __init__(self, policy: SubmissionRecoveryPolicy | None = None) -> None:
        self.policy = policy or SubmissionRecoveryPolicy()

    def start(self, *, client_order_id: str, now: datetime) -> SubmissionRecoveryRecord:
        return SubmissionRecoveryRecord(client_order_id, now, now)

    def observe(
        self,
        record: SubmissionRecoveryRecord,
        observation: SubmissionObservation,
        *,
        now: datetime,
    ) -> SubmissionRecoveryRecord:
        if record.terminal:
            return record
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now = now.astimezone(UTC)
        if now - record.first_seen_at > self.policy.recovery_deadline:
            return replace(
                record,
                state=SubmissionRecoveryState.MANUAL_REVIEW,
                updated_at=now,
                next_action_at=None,
                reason="RECOVERY_DEADLINE_EXCEEDED",
            )
        disposition = classify_submission(observation)
        if disposition.classification is SubmissionClass.DEFINITIVE_SUCCESS:
            return replace(
                record,
                state=SubmissionRecoveryState.ACKED,
                updated_at=now,
                exchange_order_id=observation.exchange_order_id,
                last_classification=disposition.classification,
                next_action_at=None,
                reason="ACK_CONFIRMED",
            )
        if disposition.classification is SubmissionClass.DEFINITIVE_REJECTION:
            return replace(
                record,
                state=SubmissionRecoveryState.REJECTED,
                updated_at=now,
                last_classification=disposition.classification,
                next_action_at=None,
                reason=observation.explicit_error_code or "DEFINITIVE_REJECTION",
            )
        if disposition.classification is SubmissionClass.AMBIGUOUS:
            attempts = record.query_attempts + 1
            if attempts > self.policy.max_query_attempts:
                return replace(
                    record,
                    state=SubmissionRecoveryState.MANUAL_REVIEW,
                    updated_at=now,
                    query_attempts=attempts,
                    last_classification=disposition.classification,
                    next_action_at=None,
                    reason="AMBIGUOUS_QUERY_BUDGET_EXHAUSTED",
                )
            return replace(
                record,
                state=SubmissionRecoveryState.QUERY_REQUIRED,
                updated_at=now,
                query_attempts=attempts,
                last_classification=disposition.classification,
                next_action_at=now + self.policy.retry_backoff,
                reason="QUERY_CLIENT_ORDER_ID_BEFORE_ANY_RETRY",
            )
        retries = record.retry_attempts + 1
        if retries > self.policy.max_retry_not_sent:
            return replace(
                record,
                state=SubmissionRecoveryState.MANUAL_REVIEW,
                updated_at=now,
                retry_attempts=retries,
                last_classification=disposition.classification,
                next_action_at=None,
                reason="NOT_SENT_RETRY_BUDGET_EXHAUSTED",
            )
        return replace(
            record,
            state=SubmissionRecoveryState.RETRY_ALLOWED,
            updated_at=now,
            retry_attempts=retries,
            last_classification=disposition.classification,
            next_action_at=now + self.policy.retry_backoff,
            reason="TRANSPORT_PROVED_NOT_SENT",
        )

    def query_not_found(
        self,
        record: SubmissionRecoveryRecord,
        *,
        now: datetime,
        exchange_history_complete: bool,
    ) -> SubmissionRecoveryRecord:
        """Handle client-order lookup returning no order.

        A negative lookup is not enough unless the caller proves the queried exchange
        history window is complete for the original submit time.
        """
        if record.state is not SubmissionRecoveryState.QUERY_REQUIRED:
            raise ValueError("query_not_found requires QUERY_REQUIRED state")
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now = now.astimezone(UTC)
        if not exchange_history_complete:
            return replace(
                record,
                updated_at=now,
                state=SubmissionRecoveryState.QUERY_REQUIRED,
                next_action_at=now + self.policy.retry_backoff,
                reason="NEGATIVE_LOOKUP_NOT_AUTHORITATIVE",
            )
        if record.query_attempts >= self.policy.max_query_attempts:
            return replace(
                record,
                updated_at=now,
                state=SubmissionRecoveryState.MANUAL_REVIEW,
                next_action_at=None,
                reason="AUTHORITATIVE_NOT_FOUND_BUT_QUERY_BUDGET_EXHAUSTED",
            )
        # Even an authoritative negative lookup does not turn an ambiguous write into a
        # blind retry.  It returns to NEW_RISK-blocked manual/query orchestration where a
        # higher layer may create a *new* intent only after reconciliation convergence.
        return replace(
            record,
            updated_at=now,
            state=SubmissionRecoveryState.MANUAL_REVIEW,
            next_action_at=None,
            reason="AUTHORITATIVE_NOT_FOUND_REQUIRES_RECONCILIATION_BEFORE_NEW_INTENT",
        )
