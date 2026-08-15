"""Stable cross-direction reason codes and contract exceptions."""

from __future__ import annotations

from enum import StrEnum


class ReasonCode(StrEnum):
    HEDGE_FEATURE_DISABLED = "HEDGE_FEATURE_DISABLED"
    POSITION_MODE_MISMATCH = "POSITION_MODE_MISMATCH"
    MARGIN_MODE_MISMATCH = "MARGIN_MODE_MISMATCH"
    LEVERAGE_MISMATCH = "LEVERAGE_MISMATCH"
    UNMANAGED_POSITION = "UNMANAGED_POSITION"
    UNMANAGED_ORDER = "UNMANAGED_ORDER"
    STALE_REST_SNAPSHOT = "STALE_REST_SNAPSHOT"
    STALE_USER_STREAM = "STALE_USER_STREAM"
    CLOCK_SKEW_EXCEEDED = "CLOCK_SKEW_EXCEEDED"
    DUPLICATE_FILL = "DUPLICATE_FILL"
    OUT_OF_ORDER_EVENT = "OUT_OF_ORDER_EVENT"
    RECONCILIATION_DRIFT = "RECONCILIATION_DRIFT"
    OUTBOX_BACKLOG = "OUTBOX_BACKLOG"
    UNKNOWN_ORDER = "UNKNOWN_ORDER"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    MIN_NOTIONAL = "MIN_NOTIONAL"
    PRECISION_ZERO = "PRECISION_ZERO"
    SINGLE_WRITER_LOST = "SINGLE_WRITER_LOST"
    POSITION_LOCK_TIMEOUT = "POSITION_LOCK_TIMEOUT"
    DEADLOCK_DETECTED = "DEADLOCK_DETECTED"
    GROSS_NOTIONAL_LIMIT = "GROSS_NOTIONAL_LIMIT"
    GROSS_RATIO_LIMIT = "GROSS_RATIO_LIMIT"
    MARGIN_UTILIZATION_HIGH = "MARGIN_UTILIZATION_HIGH"
    LIQUIDATION_BUFFER_LOW = "LIQUIDATION_BUFFER_LOW"
    MAX_LAYERS_REACHED = "MAX_LAYERS_REACHED"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    CORE_FLOOR_PROTECTED = "CORE_FLOOR_PROTECTED"
    UNSTUCK_BUDGET_EXHAUSTED = "UNSTUCK_BUDGET_EXHAUSTED"
    INTENT_EXPIRED = "INTENT_EXPIRED"
    READINESS_NOT_READY = "READINESS_NOT_READY"


class HedgeContractError(ValueError):
    """A public contract invariant was violated.

    Both legacy free-form messages and v2 stable ReasonCode values are accepted.
    """

    def __init__(self, reason_code: ReasonCode | str, message: str = "") -> None:
        try:
            normalized: ReasonCode | str = ReasonCode(reason_code)
            text = message or normalized.value
        except (TypeError, ValueError):
            normalized = str(reason_code)
            text = message or normalized
        self.reason_code = normalized
        super().__init__(text)


class ContractVersionError(HedgeContractError):
    """A producer or consumer uses an unsupported frozen contract version."""
