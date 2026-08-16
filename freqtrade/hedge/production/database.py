"""Production database readiness contracts.

Live Hedge execution requires PostgreSQL.  SQLite remains valid for unit tests, paper and
recorded replay but cannot satisfy the live capability gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import isfinite


@dataclass(frozen=True, slots=True)
class DatabaseReadinessInput:
    backend: str
    migration_head_expected: str
    migration_head_observed: str
    connection_ok: bool
    transaction_probe_ok: bool
    uniqueness_probe_ok: bool
    fencing_probe_ok: bool
    outbox_probe_ok: bool
    deadlock_retry_probe_ok: bool
    backup_verified_at: datetime | None
    restore_verified_at: datetime | None
    max_backup_age: timedelta = timedelta(hours=24)
    max_restore_age: timedelta = timedelta(days=7)
    isolation_level: str = "SERIALIZABLE"
    advisory_lock_probe_ok: bool = True
    failover_probe_ok: bool = True
    backup_checksum_verified: bool = True
    restore_rto_seconds: float = 0.0
    max_restore_rto_seconds: float = 300.0
    replication_lag_seconds: float = 0.0
    max_replication_lag_seconds: float = 5.0
    max_future_skew: timedelta = timedelta(seconds=5)

    def __post_init__(self) -> None:
        if self.max_backup_age <= timedelta(0) or self.max_restore_age <= timedelta(0):
            raise ValueError("database verification ages must be positive")
        if self.max_future_skew < timedelta(0) or self.max_future_skew > timedelta(minutes=5):
            raise ValueError("database max_future_skew must be in [0, 5m]")
        for name in (
            "restore_rto_seconds", "max_restore_rto_seconds",
            "replication_lag_seconds", "max_replication_lag_seconds",
        ):
            if not isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        for name in ("backup_verified_at", "restore_verified_at"):
            value = getattr(self, name)
            if value is not None and value.tzinfo is None:
                raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class DatabaseReadinessResult:
    passed: bool
    reasons: tuple[str, ...]
    backend: str


def evaluate_database_readiness(
    value: DatabaseReadinessInput, *, now: datetime
) -> DatabaseReadinessResult:
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(UTC)
    reasons: list[str] = []
    backend = value.backend.strip().lower()
    if backend not in {"postgresql", "postgres"}:
        reasons.append("LIVE_REQUIRES_POSTGRESQL")
    if not value.connection_ok:
        reasons.append("DATABASE_CONNECTION_FAILED")
    if not value.migration_head_expected or value.migration_head_observed != value.migration_head_expected:
        reasons.append("MIGRATION_HEAD_MISMATCH")
    probes = {
        "TRANSACTION_PROBE_FAILED": value.transaction_probe_ok,
        "UNIQUENESS_PROBE_FAILED": value.uniqueness_probe_ok,
        "FENCING_PROBE_FAILED": value.fencing_probe_ok,
        "OUTBOX_PROBE_FAILED": value.outbox_probe_ok,
        "DEADLOCK_RETRY_PROBE_FAILED": value.deadlock_retry_probe_ok,
        "ADVISORY_LOCK_PROBE_FAILED": value.advisory_lock_probe_ok,
        "DATABASE_FAILOVER_PROBE_FAILED": value.failover_probe_ok,
        "BACKUP_CHECKSUM_NOT_VERIFIED": value.backup_checksum_verified,
    }
    reasons.extend(name for name, passed in probes.items() if not passed)
    if value.backup_verified_at is None:
        reasons.append("BACKUP_NOT_VERIFIED")
    else:
        backup = value.backup_verified_at.astimezone(UTC)
        if backup > now + value.max_future_skew:
            reasons.append("BACKUP_VERIFICATION_FROM_FUTURE")
        elif now - backup > value.max_backup_age:
            reasons.append("BACKUP_VERIFICATION_STALE")
    if value.restore_verified_at is None:
        reasons.append("RESTORE_NOT_VERIFIED")
    else:
        restore = value.restore_verified_at.astimezone(UTC)
        if restore > now + value.max_future_skew:
            reasons.append("RESTORE_VERIFICATION_FROM_FUTURE")
        elif now - restore > value.max_restore_age:
            reasons.append("RESTORE_VERIFICATION_STALE")
    isolation = value.isolation_level.strip().upper().replace("_", " ")
    if isolation not in {"SERIALIZABLE", "REPEATABLE READ"}:
        reasons.append("DATABASE_ISOLATION_TOO_WEAK")
    if value.restore_rto_seconds < 0 or value.max_restore_rto_seconds <= 0:
        reasons.append("RESTORE_RTO_INVALID")
    elif value.restore_rto_seconds > value.max_restore_rto_seconds:
        reasons.append("RESTORE_RTO_EXCEEDED")
    if value.replication_lag_seconds < 0 or value.max_replication_lag_seconds < 0:
        reasons.append("REPLICATION_LAG_INVALID")
    elif value.replication_lag_seconds > value.max_replication_lag_seconds:
        reasons.append("REPLICATION_LAG_EXCEEDED")
    return DatabaseReadinessResult(not reasons, tuple(reasons), backend)
