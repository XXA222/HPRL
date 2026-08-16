"""Real-environment PostgreSQL durability/failover/backup acceptance contracts for R2.

The existing ``database_runtime`` module already proves basic PostgreSQL behavior and a
dual-connection advisory-lock handoff.  This module adds the missing *durability* probe
and typed evidence for destructive/operational exercises which cannot be honestly
fabricated in an offline source build (actual failover and pg_dump/pg_restore).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import re
import time
from typing import Any, Callable
from uuid import uuid4

from .database_runtime import PostgresConcurrencyProbeReport, PostgresProbeReport

_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _sha(value: object, *, field: str) -> str:
    text = str(value).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError(f"{field} must be SHA-256 hex")
    return text


@dataclass(frozen=True, slots=True)
class PostgresDurabilityProbeReport:
    distinct_connections: bool
    primary_write_committed: bool
    secondary_observed_committed_value: bool
    cleanup_committed: bool
    primary_backend_pid: int | None
    secondary_backend_pid: int | None
    probe_id: str
    payload_sha256: str
    visibility_attempts: int
    errors: tuple[str, ...]
    observed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _aware(self.observed_at))
        object.__setattr__(self, "payload_sha256", _sha(self.payload_sha256, field="payload_sha256"))

    @property
    def passed(self) -> bool:
        return (
            self.distinct_connections
            and self.primary_write_committed
            and self.secondary_observed_committed_value
            and self.cleanup_committed
            and not self.errors
        )


@dataclass(frozen=True, slots=True)
class PostgresDurabilityProbePolicy:
    schema: str = "freqtrade_hedge_probe"
    table: str = "hprl_r2_durability"
    visibility_attempts: int = 10
    visibility_sleep_seconds: float = 0.10

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.schema) or not _IDENTIFIER.fullmatch(self.table):
            raise ValueError("probe schema/table must be safe PostgreSQL identifiers")
        if self.visibility_attempts <= 0:
            raise ValueError("visibility_attempts must be positive")
        if self.visibility_sleep_seconds < 0 or self.visibility_sleep_seconds > 5:
            raise ValueError("visibility_sleep_seconds must be within [0,5]")


class PostgresDurabilityProbeRunner:
    """Commit a dedicated probe row on one connection and observe it on another.

    This performs bounded writes in a dedicated probe schema and always attempts cleanup.
    It is therefore intentionally opt-in and should run only against the target acceptance
    PostgreSQL cluster, never implicitly during import or source validation.
    """

    def __init__(
        self,
        primary_factory: Callable[[], Any],
        secondary_factory: Callable[[], Any],
        *,
        policy: PostgresDurabilityProbePolicy | None = None,
    ) -> None:
        if not callable(primary_factory) or not callable(secondary_factory):
            raise TypeError("PostgreSQL connection factories must be callable")
        self.primary_factory = primary_factory
        self.secondary_factory = secondary_factory
        self.policy = policy or PostgresDurabilityProbePolicy()

    @staticmethod
    def _scalar(connection: Any, sql: str, params: tuple[object, ...] = ()) -> object:
        cursor = connection.cursor()
        try:
            cursor.execute(sql, params)
            row = cursor.fetchone()
            return None if row is None else row[0]
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _execute(connection: Any, sql: str, params: tuple[object, ...] = ()) -> None:
        cursor = connection.cursor()
        try:
            cursor.execute(sql, params)
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()

    def run(self, *, now: datetime) -> PostgresDurabilityProbeReport:
        observed = _aware(now)
        probe_id = "hprl-r2-" + uuid4().hex
        payload = sha256(f"{probe_id}|{observed.isoformat()}".encode()).hexdigest()
        errors: list[str] = []
        primary = secondary = None
        pid1 = pid2 = None
        write_ok = visibility_ok = cleanup_ok = False
        attempts = 0
        table = f'"{self.policy.schema}"."{self.policy.table}"'
        try:
            primary = self.primary_factory()
            secondary = self.secondary_factory()
            pid1 = int(self._scalar(primary, "SELECT pg_backend_pid()"))
            pid2 = int(self._scalar(secondary, "SELECT pg_backend_pid()"))
            self._execute(primary, f'CREATE SCHEMA IF NOT EXISTS "{self.policy.schema}"')
            self._execute(
                primary,
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "probe_id text PRIMARY KEY, payload_sha256 text NOT NULL, created_at timestamptz NOT NULL)"
            )
            primary.commit()
            self._execute(
                primary,
                f"INSERT INTO {table} (probe_id,payload_sha256,created_at) VALUES (%s,%s,%s) "
                "ON CONFLICT (probe_id) DO UPDATE SET payload_sha256=EXCLUDED.payload_sha256, "
                "created_at=EXCLUDED.created_at",
                (probe_id, payload, observed),
            )
            primary.commit()
            write_ok = True
            for attempts in range(1, self.policy.visibility_attempts + 1):
                found = self._scalar(
                    secondary,
                    f"SELECT payload_sha256 FROM {table} WHERE probe_id=%s",
                    (probe_id,),
                )
                if found == payload:
                    visibility_ok = True
                    break
                rollback = getattr(secondary, "rollback", None)
                if callable(rollback):
                    rollback()
                if self.policy.visibility_sleep_seconds:
                    time.sleep(self.policy.visibility_sleep_seconds)
        except Exception as exc:
            errors.append(f"PROBE:{type(exc).__name__}:{exc}")
            for connection in (primary, secondary):
                try:
                    rollback = getattr(connection, "rollback", None)
                    if callable(rollback):
                        rollback()
                except Exception:
                    pass
        finally:
            if primary is not None and write_ok:
                try:
                    self._execute(primary, f"DELETE FROM {table} WHERE probe_id=%s", (probe_id,))
                    primary.commit()
                    cleanup_ok = True
                except Exception as exc:
                    errors.append(f"CLEANUP:{type(exc).__name__}:{exc}")
                    try:
                        primary.rollback()
                    except Exception:
                        pass
            for connection in (secondary, primary):
                if connection is None:
                    continue
                try:
                    close = getattr(connection, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass
        return PostgresDurabilityProbeReport(
            distinct_connections=(pid1 is not None and pid2 is not None and pid1 != pid2),
            primary_write_committed=write_ok,
            secondary_observed_committed_value=visibility_ok,
            cleanup_committed=cleanup_ok,
            primary_backend_pid=pid1,
            secondary_backend_pid=pid2,
            probe_id=probe_id,
            payload_sha256=payload,
            visibility_attempts=attempts,
            errors=tuple(errors),
            observed_at=observed,
        )


@dataclass(frozen=True, slots=True)
class PostgresFailoverEvidence:
    exercise_id: str
    observed_at: datetime
    primary_failure_observed: bool
    secondary_promoted_or_routed: bool
    durable_probe_visible_after_failover: bool
    secondary_writable_after_failover: bool
    writer_fence_reacquired: bool
    old_primary_fenced: bool
    execution_state_converged: bool
    artifact_sha256: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.exercise_id.strip():
            raise ValueError("failover exercise_id is required")
        object.__setattr__(self, "observed_at", _aware(self.observed_at))
        object.__setattr__(self, "artifact_sha256", _sha(self.artifact_sha256, field="artifact_sha256"))

    @property
    def passed(self) -> bool:
        return all((
            self.primary_failure_observed,
            self.secondary_promoted_or_routed,
            self.durable_probe_visible_after_failover,
            self.secondary_writable_after_failover,
            self.writer_fence_reacquired,
            self.old_primary_fenced,
            self.execution_state_converged,
        ))


@dataclass(frozen=True, slots=True)
class PostgresBackupRestoreEvidence:
    exercise_id: str
    observed_at: datetime
    backup_sha256: str
    isolated_restore_target: bool
    restore_completed: bool
    migration_head_match: bool
    execution_order_count_match: bool
    fill_event_count_match: bool
    reconciliation_converged: bool
    source_snapshot_sha256: str
    restored_snapshot_sha256: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.exercise_id.strip():
            raise ValueError("backup/restore exercise_id is required")
        object.__setattr__(self, "observed_at", _aware(self.observed_at))
        for name in ("backup_sha256", "source_snapshot_sha256", "restored_snapshot_sha256"):
            object.__setattr__(self, name, _sha(getattr(self, name), field=name))

    @property
    def passed(self) -> bool:
        return all((
            self.isolated_restore_target,
            self.restore_completed,
            self.migration_head_match,
            self.execution_order_count_match,
            self.fill_event_count_match,
            self.reconciliation_converged,
            self.source_snapshot_sha256 == self.restored_snapshot_sha256,
        ))


@dataclass(frozen=True, slots=True)
class PostgresR2AcceptanceReport:
    basic_probe_passed: bool
    concurrency_probe_passed: bool
    durability_probe_passed: bool
    failover_exercise_passed: bool
    backup_restore_passed: bool
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.reasons


def evaluate_postgres_r2(
    *,
    basic: PostgresProbeReport | None,
    concurrency: PostgresConcurrencyProbeReport | None,
    durability: PostgresDurabilityProbeReport | None,
    failover: PostgresFailoverEvidence | None,
    backup_restore: PostgresBackupRestoreEvidence | None,
) -> PostgresR2AcceptanceReport:
    reasons: list[str] = []
    checks = (
        ("POSTGRES_BASIC_PROBE_MISSING_OR_FAILED", bool(basic and basic.passed)),
        ("POSTGRES_CONCURRENCY_PROBE_MISSING_OR_FAILED", bool(concurrency and concurrency.passed)),
        ("POSTGRES_DURABILITY_PROBE_MISSING_OR_FAILED", bool(durability and durability.passed)),
        ("POSTGRES_FAILOVER_EXERCISE_MISSING_OR_FAILED", bool(failover and failover.passed)),
        ("POSTGRES_BACKUP_RESTORE_MISSING_OR_FAILED", bool(backup_restore and backup_restore.passed)),
    )
    reasons.extend(reason for reason, passed in checks if not passed)
    return PostgresR2AcceptanceReport(
        basic_probe_passed=checks[0][1],
        concurrency_probe_passed=checks[1][1],
        durability_probe_passed=checks[2][1],
        failover_exercise_passed=checks[3][1],
        backup_restore_passed=checks[4][1],
        reasons=tuple(reasons),
    )
