"""PostgreSQL DB-API production probes used to generate DATABASE_READY evidence.

The module intentionally depends only on Python DB-API semantics.  A caller may pass a
psycopg/psycopg2 connection without making either package a core runtime dependency.
All probes use temporary objects or advisory locks and leave no durable trading rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
from typing import Any


@dataclass(frozen=True, slots=True)
class PostgresProbeReport:
    connection_ok: bool
    transaction_rollback_ok: bool
    uniqueness_ok: bool
    advisory_lock_ok: bool
    isolation_level: str
    server_version: str
    database_name: str
    errors: tuple[str, ...]
    observed_at: datetime

    @property
    def passed(self) -> bool:
        return (
            self.connection_ok
            and self.transaction_rollback_ok
            and self.uniqueness_ok
            and self.advisory_lock_ok
            and self.isolation_level.upper() in {"SERIALIZABLE", "REPEATABLE READ"}
            and not self.errors
        )


class PostgresProbeRunner:
    """Execute non-destructive, rollback-contained production DB probes."""

    def __init__(self, connection: Any, *, advisory_lock_namespace: str = "freqtrade-hedge") -> None:
        self.connection = connection
        self.lock_key = int.from_bytes(
            hashlib.sha256(advisory_lock_namespace.encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=True,
        )

    def run(self, *, now: datetime) -> PostgresProbeReport:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        errors: list[str] = []
        connection_ok = False
        rollback_ok = False
        uniqueness_ok = False
        advisory_ok = False
        isolation = ""
        server_version = ""
        database_name = ""
        cursor = None
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            row = cursor.fetchone()
            connection_ok = bool(row and int(row[0]) == 1)
            cursor.execute("SHOW transaction_isolation")
            row = cursor.fetchone()
            isolation = str(row[0]).upper() if row else ""
            cursor.execute("SHOW server_version")
            row = cursor.fetchone()
            server_version = str(row[0]) if row else ""
            cursor.execute("SELECT current_database()")
            row = cursor.fetchone()
            database_name = str(row[0]) if row else ""
        except Exception as exc:
            errors.append(f"CONNECTION_PROBE:{type(exc).__name__}:{exc}")

        # Transaction rollback: create a temp table and roll back an inserted row.
        try:
            if hasattr(self.connection, "rollback"):
                self.connection.rollback()
            cursor = self.connection.cursor()
            cursor.execute("CREATE TEMP TABLE IF NOT EXISTS hedge_pr_probe_tx (id integer primary key) ON COMMIT PRESERVE ROWS")
            cursor.execute("DELETE FROM hedge_pr_probe_tx")
            # PostgreSQL DDL is transactional.  Commit only the session-local temp
            # fixture before testing rollback; otherwise the rollback would remove the
            # table itself and produce a false failure.
            commit = getattr(self.connection, "commit", None)
            if not callable(commit):
                raise TypeError("DB-API connection must implement commit for rollback probe")
            commit()
            cursor = self.connection.cursor()
            cursor.execute("INSERT INTO hedge_pr_probe_tx(id) VALUES (1)")
            self.connection.rollback()
            cursor = self.connection.cursor()
            cursor.execute("SELECT count(*) FROM hedge_pr_probe_tx")
            row = cursor.fetchone()
            rollback_ok = bool(row and int(row[0]) == 0)
        except Exception as exc:
            errors.append(f"TRANSACTION_ROLLBACK:{type(exc).__name__}:{exc}")
            try:
                self.connection.rollback()
            except Exception:
                pass

        # Uniqueness probe: duplicate PK must fail and transaction must remain recoverable.
        try:
            cursor = self.connection.cursor()
            cursor.execute("CREATE TEMP TABLE IF NOT EXISTS hedge_pr_probe_unique (id integer primary key) ON COMMIT PRESERVE ROWS")
            cursor.execute("DELETE FROM hedge_pr_probe_unique")
            commit = getattr(self.connection, "commit", None)
            if not callable(commit):
                raise TypeError("DB-API connection must implement commit for uniqueness probe")
            commit()
            cursor = self.connection.cursor()
            cursor.execute("INSERT INTO hedge_pr_probe_unique(id) VALUES (1)")
            duplicate_failed = False
            try:
                cursor.execute("INSERT INTO hedge_pr_probe_unique(id) VALUES (1)")
            except Exception:
                duplicate_failed = True
                self.connection.rollback()
            # A uniqueness violation aborts the transaction in PostgreSQL.  Prove the
            # rollback restored the session to a usable state rather than merely seeing
            # the expected exception.
            recoverable = False
            if duplicate_failed:
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                row = cursor.fetchone()
                recoverable = bool(row and int(row[0]) == 1)
            uniqueness_ok = duplicate_failed and recoverable
        except Exception as exc:
            errors.append(f"UNIQUENESS:{type(exc).__name__}:{exc}")
            try:
                self.connection.rollback()
            except Exception:
                pass

        # Advisory lock: proves a process can acquire/release the SingleWriter namespace.
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT pg_try_advisory_lock(%s)", (self.lock_key,))
            row = cursor.fetchone()
            acquired = bool(row and row[0])
            if acquired:
                cursor.execute("SELECT pg_advisory_unlock(%s)", (self.lock_key,))
                row2 = cursor.fetchone()
                advisory_ok = bool(row2 and row2[0])
            else:
                advisory_ok = False
        except Exception as exc:
            errors.append(f"ADVISORY_LOCK:{type(exc).__name__}:{exc}")
            try:
                self.connection.rollback()
            except Exception:
                pass

        return PostgresProbeReport(
            connection_ok,
            rollback_ok,
            uniqueness_ok,
            advisory_ok,
            isolation,
            server_version,
            database_name,
            tuple(errors),
            now.astimezone(UTC),
        )


@dataclass(frozen=True, slots=True)
class PostgresConcurrencyProbeReport:
    """Two-connection SingleWriter/advisory-lock exclusion evidence."""

    distinct_backends: bool
    primary_lock_acquired: bool
    secondary_blocked_while_primary_held: bool
    secondary_acquired_after_release: bool
    primary_backend_pid: int | None
    secondary_backend_pid: int | None
    errors: tuple[str, ...]
    observed_at: datetime

    def __post_init__(self) -> None:
        if self.observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))

    @property
    def passed(self) -> bool:
        return (
            self.distinct_backends
            and self.primary_lock_acquired
            and self.secondary_blocked_while_primary_held
            and self.secondary_acquired_after_release
            and not self.errors
        )


class PostgresConcurrencyProbeRunner:
    """Prove advisory SingleWriter exclusion with two independent DB connections.

    ``connection_factory`` must return a new DB-API connection each time.  No durable
    table is modified; the probe uses only backend identity and session advisory locks.
    """

    def __init__(
        self,
        connection_factory: Any,
        *,
        advisory_lock_namespace: str = "freqtrade-hedge",
    ) -> None:
        if not callable(connection_factory):
            raise TypeError("connection_factory must be callable")
        self.connection_factory = connection_factory
        self.lock_key = int.from_bytes(
            hashlib.sha256(advisory_lock_namespace.encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=True,
        )

    @staticmethod
    def _scalar(connection: Any, sql: str, params: tuple[object, ...] = ()) -> Any:
        cursor = connection.cursor()
        cursor.execute(sql, params)
        row = cursor.fetchone()
        return row[0] if row else None

    def run(self, *, now: datetime) -> PostgresConcurrencyProbeReport:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        errors: list[str] = []
        c1 = c2 = None
        pid1 = pid2 = None
        primary = False
        secondary_blocked = False
        secondary_after = False
        try:
            c1 = self.connection_factory()
            c2 = self.connection_factory()
            if c1 is c2:
                errors.append("CONNECTION_FACTORY_REUSED_OBJECT")
            pid1_raw = self._scalar(c1, "SELECT pg_backend_pid()")
            pid2_raw = self._scalar(c2, "SELECT pg_backend_pid()")
            pid1 = int(pid1_raw) if pid1_raw is not None else None
            pid2 = int(pid2_raw) if pid2_raw is not None else None
            if pid1 is None or pid2 is None:
                errors.append("BACKEND_PID_MISSING")
            primary = bool(
                self._scalar(c1, "SELECT pg_try_advisory_lock(%s)", (self.lock_key,))
            )
            if not primary:
                errors.append("PRIMARY_LOCK_NOT_ACQUIRED")
            if primary:
                second_during = bool(
                    self._scalar(c2, "SELECT pg_try_advisory_lock(%s)", (self.lock_key,))
                )
                secondary_blocked = not second_during
                if second_during:
                    # Defensive unlock if the backend unexpectedly acquired it.
                    self._scalar(c2, "SELECT pg_advisory_unlock(%s)", (self.lock_key,))
                    errors.append("SECONDARY_LOCK_NOT_EXCLUDED")
                released = bool(
                    self._scalar(c1, "SELECT pg_advisory_unlock(%s)", (self.lock_key,))
                )
                if not released:
                    errors.append("PRIMARY_LOCK_RELEASE_FAILED")
                if released:
                    secondary_after = bool(
                        self._scalar(c2, "SELECT pg_try_advisory_lock(%s)", (self.lock_key,))
                    )
                    if not secondary_after:
                        errors.append("SECONDARY_TAKEOVER_FAILED")
                    else:
                        self._scalar(c2, "SELECT pg_advisory_unlock(%s)", (self.lock_key,))
        except Exception as exc:
            errors.append(f"CONCURRENCY_PROBE:{type(exc).__name__}:{exc}")
        finally:
            for connection in (c2, c1):
                if connection is None:
                    continue
                try:
                    rollback = getattr(connection, "rollback", None)
                    if callable(rollback):
                        rollback()
                except Exception:
                    pass
                try:
                    close = getattr(connection, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass
        distinct = pid1 is not None and pid2 is not None and pid1 != pid2 and c1 is not c2
        return PostgresConcurrencyProbeReport(
            distinct,
            primary,
            secondary_blocked,
            secondary_after,
            pid1,
            pid2,
            tuple(errors),
            now.astimezone(UTC),
        )
