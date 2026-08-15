"""Readiness states, reason policies and stable serialization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from freqtrade.hedge.identity import RiskPositionKey


class ReadinessState(str, Enum):
    STARTING = "STARTING"
    NOT_READY = "NOT_READY"  # compatibility for clock/initial failures
    DEGRADED = "DEGRADED"
    READY = "READY"
    HALT = "HALT"


class ReadinessSeverity(str, Enum):
    DEGRADED = "DEGRADED"
    HALT_WRITE = "HALT_WRITE"
    HALT_ACCOUNT = "HALT_ACCOUNT"


class ReadinessScope(str, Enum):
    ACCOUNT = "ACCOUNT"
    WRITE = "WRITE"
    POSITION = "POSITION"


class ReadinessReasonCode(str, Enum):
    DATABASE_MIGRATION_FAILED = "DATABASE_MIGRATION_FAILED"
    DATABASE_MIGRATION_CHECKSUM_INVALID = "DATABASE_MIGRATION_CHECKSUM_INVALID"
    SINGLE_WRITER_LEASE_INVALID = "SINGLE_WRITER_LEASE_INVALID"
    POSITION_MODE_NOT_HEDGE = "POSITION_MODE_NOT_HEDGE"
    MARGIN_MODE_NOT_CROSS = "MARGIN_MODE_NOT_CROSS"
    LEVERAGE_MISMATCH = "LEVERAGE_MISMATCH"
    UNMANAGED_POSITION_PRESENT = "UNMANAGED_POSITION_PRESENT"
    UNMANAGED_ORDER_PRESENT = "UNMANAGED_ORDER_PRESENT"
    REST_SNAPSHOT_INVALID = "REST_SNAPSHOT_INVALID"
    REST_SNAPSHOT_STALE = "REST_SNAPSHOT_STALE"
    USER_STREAM_STALE = "USER_STREAM_STALE"
    USER_STREAM_GAP = "USER_STREAM_GAP"
    UNKNOWN_ORDER_PRESENT = "UNKNOWN_ORDER_PRESENT"
    UNKNOWN_ORDER_SCOPE_INCOMPLETE = "UNKNOWN_ORDER_SCOPE_INCOMPLETE"
    RECONCILIATION_NOT_CONVERGED = "RECONCILIATION_NOT_CONVERGED"
    RISK_DATA_INVALID = "RISK_DATA_INVALID"
    RISK_SNAPSHOT_STALE = "RISK_SNAPSHOT_STALE"
    HALT_REASON_PRESENT = "HALT_REASON_PRESENT"
    READINESS_CLOCK_INVALID = "READINESS_CLOCK_INVALID"


@dataclass(frozen=True, slots=True)
class ReadinessReasonPolicy:
    severity: ReadinessSeverity
    scope: ReadinessScope
    controlled_reduce_allowed: bool


_REASON_POLICIES: dict[ReadinessReasonCode, ReadinessReasonPolicy] = {
    ReadinessReasonCode.DATABASE_MIGRATION_FAILED: ReadinessReasonPolicy(
        ReadinessSeverity.HALT_ACCOUNT, ReadinessScope.ACCOUNT, False
    ),
    ReadinessReasonCode.DATABASE_MIGRATION_CHECKSUM_INVALID: ReadinessReasonPolicy(
        ReadinessSeverity.HALT_ACCOUNT, ReadinessScope.ACCOUNT, False
    ),
    ReadinessReasonCode.SINGLE_WRITER_LEASE_INVALID: ReadinessReasonPolicy(
        ReadinessSeverity.HALT_WRITE, ReadinessScope.WRITE, False
    ),
    ReadinessReasonCode.POSITION_MODE_NOT_HEDGE: ReadinessReasonPolicy(
        ReadinessSeverity.HALT_ACCOUNT, ReadinessScope.ACCOUNT, False
    ),
    ReadinessReasonCode.MARGIN_MODE_NOT_CROSS: ReadinessReasonPolicy(
        ReadinessSeverity.HALT_ACCOUNT, ReadinessScope.ACCOUNT, False
    ),
    ReadinessReasonCode.LEVERAGE_MISMATCH: ReadinessReasonPolicy(
        ReadinessSeverity.DEGRADED, ReadinessScope.ACCOUNT, True
    ),
    ReadinessReasonCode.UNMANAGED_POSITION_PRESENT: ReadinessReasonPolicy(
        ReadinessSeverity.HALT_ACCOUNT, ReadinessScope.ACCOUNT, True
    ),
    ReadinessReasonCode.UNMANAGED_ORDER_PRESENT: ReadinessReasonPolicy(
        ReadinessSeverity.HALT_ACCOUNT, ReadinessScope.ACCOUNT, True
    ),
    ReadinessReasonCode.REST_SNAPSHOT_INVALID: ReadinessReasonPolicy(
        ReadinessSeverity.DEGRADED, ReadinessScope.ACCOUNT, False
    ),
    ReadinessReasonCode.REST_SNAPSHOT_STALE: ReadinessReasonPolicy(
        ReadinessSeverity.DEGRADED, ReadinessScope.ACCOUNT, False
    ),
    ReadinessReasonCode.USER_STREAM_STALE: ReadinessReasonPolicy(
        ReadinessSeverity.DEGRADED, ReadinessScope.ACCOUNT, True
    ),
    ReadinessReasonCode.USER_STREAM_GAP: ReadinessReasonPolicy(
        ReadinessSeverity.DEGRADED, ReadinessScope.ACCOUNT, True
    ),
    ReadinessReasonCode.UNKNOWN_ORDER_PRESENT: ReadinessReasonPolicy(
        ReadinessSeverity.DEGRADED, ReadinessScope.POSITION, False
    ),
    ReadinessReasonCode.UNKNOWN_ORDER_SCOPE_INCOMPLETE: ReadinessReasonPolicy(
        ReadinessSeverity.HALT_ACCOUNT, ReadinessScope.ACCOUNT, False
    ),
    ReadinessReasonCode.RECONCILIATION_NOT_CONVERGED: ReadinessReasonPolicy(
        ReadinessSeverity.DEGRADED, ReadinessScope.ACCOUNT, True
    ),
    ReadinessReasonCode.RISK_DATA_INVALID: ReadinessReasonPolicy(
        ReadinessSeverity.HALT_ACCOUNT, ReadinessScope.ACCOUNT, False
    ),
    ReadinessReasonCode.RISK_SNAPSHOT_STALE: ReadinessReasonPolicy(
        ReadinessSeverity.HALT_ACCOUNT, ReadinessScope.ACCOUNT, False
    ),
    ReadinessReasonCode.HALT_REASON_PRESENT: ReadinessReasonPolicy(
        ReadinessSeverity.HALT_ACCOUNT, ReadinessScope.ACCOUNT, False
    ),
    ReadinessReasonCode.READINESS_CLOCK_INVALID: ReadinessReasonPolicy(
        ReadinessSeverity.HALT_ACCOUNT, ReadinessScope.ACCOUNT, False
    ),
}


def reason_policy(code: ReadinessReasonCode) -> ReadinessReasonPolicy:
    return _REASON_POLICIES[code]


@dataclass(frozen=True, slots=True)
class ReadinessCheckResult:
    name: str
    passed: bool
    reason_code: ReadinessReasonCode | None = None
    detail: str = ""
    severity: ReadinessSeverity | None = None
    scope: ReadinessScope | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Readiness check name must not be empty.")
        if not isinstance(self.passed, bool):
            raise ValueError("Readiness check passed must be a boolean.")
        if self.passed and self.reason_code is not None:
            raise ValueError("Passing readiness check must not have a reason code.")
        if not self.passed and self.reason_code is None:
            raise ValueError("Failing readiness check must have a stable reason code.")
        if self.reason_code is not None:
            policy = reason_policy(self.reason_code)
            object.__setattr__(self, "severity", policy.severity)
            object.__setattr__(self, "scope", policy.scope)
        elif self.severity is not None or self.scope is not None:
            raise ValueError("Passing readiness check must not carry severity or scope.")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "detail", str(self.detail))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "reason_code": None if self.reason_code is None else self.reason_code.value,
            "detail": self.detail,
            "severity": None if self.severity is None else self.severity.value,
            "scope": None if self.scope is None else self.scope.value,
        }


@dataclass(frozen=True, slots=True)
class ReadinessReport:
    state: ReadinessState
    checks: tuple[ReadinessCheckResult, ...]
    reason_codes: tuple[ReadinessReasonCode, ...]
    evaluated_at_ms: int
    emergency_reduce_only_allowed: bool
    blocked_position_keys: tuple[RiskPositionKey, ...] = ()

    def __post_init__(self) -> None:
        state = self.state if isinstance(self.state, ReadinessState) else ReadinessState(self.state)
        object.__setattr__(self, "state", state)
        if isinstance(self.evaluated_at_ms, bool) or not isinstance(self.evaluated_at_ms, int):
            raise ValueError("evaluated_at_ms must be an integer.")
        if self.evaluated_at_ms < 0:
            raise ValueError("evaluated_at_ms must be nonnegative.")
        if not isinstance(self.emergency_reduce_only_allowed, bool):
            raise ValueError("emergency_reduce_only_allowed must be a boolean.")
        checks = tuple(self.checks)
        reasons = tuple(self.reason_codes)
        failed_reasons = tuple(
            check.reason_code
            for check in checks
            if not check.passed and check.reason_code is not None
        )
        if reasons != failed_reasons:
            raise ValueError("reason_codes must exactly match failing readiness checks.")
        if state is ReadinessState.READY and reasons:
            raise ValueError("READY report must not contain failure reasons.")
        if state is ReadinessState.STARTING and checks:
            raise ValueError("STARTING report must not contain evaluated checks.")
        blocked = tuple(dict.fromkeys(self.blocked_position_keys))
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "blocked_position_keys", blocked)

    @property
    def ready(self) -> bool:
        return self.state is ReadinessState.READY

    @property
    def max_severity(self) -> ReadinessSeverity | None:
        severities = [check.severity for check in self.checks if check.severity is not None]
        if not severities:
            return None
        order = {
            ReadinessSeverity.DEGRADED: 1,
            ReadinessSeverity.HALT_WRITE: 2,
            ReadinessSeverity.HALT_ACCOUNT: 3,
        }
        return max(severities, key=order.__getitem__)

    def is_position_blocked(self, position_key: RiskPositionKey) -> bool:
        return position_key in self.blocked_position_keys

    def as_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "ready": self.ready,
            "max_severity": None if self.max_severity is None else self.max_severity.value,
            "reason_codes": [code.value for code in self.reason_codes],
            "evaluated_at_ms": self.evaluated_at_ms,
            "emergency_reduce_only_allowed": self.emergency_reduce_only_allowed,
            "blocked_position_keys": [key.as_dict() for key in self.blocked_position_keys],
            "checks": [check.as_dict() for check in self.checks],
        }
