"""Submission outcome classification for exactly-once exchange intent handling."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SubmissionClass(StrEnum):
    DEFINITIVE_SUCCESS = "DEFINITIVE_SUCCESS"
    DEFINITIVE_REJECTION = "DEFINITIVE_REJECTION"
    AMBIGUOUS = "AMBIGUOUS"
    RETRYABLE_NOT_SENT = "RETRYABLE_NOT_SENT"


class SubmissionAction(StrEnum):
    PERSIST_ACK = "PERSIST_ACK"
    PERSIST_REJECTION = "PERSIST_REJECTION"
    QUERY_BY_CLIENT_ORDER_ID = "QUERY_BY_CLIENT_ORDER_ID"
    RETRY_SAME_INTENT = "RETRY_SAME_INTENT"
    HALT_NEW_RISK = "HALT_NEW_RISK"


@dataclass(frozen=True, slots=True)
class SubmissionObservation:
    http_status: int | None
    response_received: bool
    request_may_have_reached_exchange: bool
    exchange_order_id: str | None = None
    explicit_error_code: str | None = None


@dataclass(frozen=True, slots=True)
class SubmissionDisposition:
    classification: SubmissionClass
    actions: tuple[SubmissionAction, ...]
    direct_resubmit_allowed: bool


def classify_submission(value: SubmissionObservation) -> SubmissionDisposition:
    if value.response_received and value.exchange_order_id:
        return SubmissionDisposition(SubmissionClass.DEFINITIVE_SUCCESS, (SubmissionAction.PERSIST_ACK,), False)
    if value.response_received and value.http_status is not None and 400 <= value.http_status < 500 and value.http_status not in {408, 409, 418, 429}:
        return SubmissionDisposition(SubmissionClass.DEFINITIVE_REJECTION, (SubmissionAction.PERSIST_REJECTION,), False)
    if value.request_may_have_reached_exchange:
        return SubmissionDisposition(
            SubmissionClass.AMBIGUOUS,
            (SubmissionAction.HALT_NEW_RISK, SubmissionAction.QUERY_BY_CLIENT_ORDER_ID),
            False,
        )
    return SubmissionDisposition(SubmissionClass.RETRYABLE_NOT_SENT, (SubmissionAction.RETRY_SAME_INTENT,), True)
