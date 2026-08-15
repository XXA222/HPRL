"""Idempotent and recoverable H3 hedge-ledger migrations.

The migration journal is persisted independently from each migration step.
A step left RUNNING or FAILED is safe to retry because all DDL/DML operations
are introspection guarded and each data step is deterministic.
"""

from __future__ import annotations

import hashlib
import inspect as python_inspect
import json
import logging
import os
import shutil
import socket
import sqlite3
import threading
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import Engine, bindparam, inspect, text

from freqtrade.exceptions import OperationalException

logger = logging.getLogger(__name__)

LEGACY_SCHEMA_VERSION = "h3-ledger-v1"
H3_SCHEMA_VERSION = "h3-ledger-v2"
H3_RELEASE = "2.0.0"
MIGRATION_TABLE = "hedge_schema_migrations"
FILL_TABLE = "hedge_fill_events"
LEGACY_REQUIRED_LEDGER_TABLES = frozenset(
    {
        "hedge_order_intents",
        "hedge_order_snapshots",
        "hedge_fill_events",
        "hedge_position_snapshots",
        "hedge_account_risk_snapshots",
        "hedge_account_events",
        "hedge_reconciliation_runs",
        "hedge_reconciliation_diffs",
        "hedge_strategy_side_states",
        "hedge_event_outbox",
        "hedge_schema_migrations",
    }
)
REQUIRED_LEDGER_TABLES = LEGACY_REQUIRED_LEDGER_TABLES | frozenset(
    {
        "hedge_current_orders",
        "hedge_target_positions",
        "hedge_core_position_states",
        "hedge_tactical_lots",
        "hedge_audit_events",
    }
)
R2_EXECUTION_AUTHORITY_TABLES = frozenset({
    "hedge_execution_order_states",
    "hedge_execution_idempotency",
})

_MEMORY_LOCK_GUARD = threading.Lock()
_MEMORY_LOCKS: dict[str, threading.Lock] = {}


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


def _migration_timestamp(engine: Engine, value: datetime | None = None) -> datetime | str:
    """Return a DBAPI-safe timestamp without deprecated SQLite adapters."""

    timestamp = value or _utcnow()
    if engine.dialect.name == "sqlite":
        return timestamp.isoformat(sep=" ", timespec="microseconds")
    return timestamp


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


class HedgeMigrationError(OperationalException):
    pass


class HedgeMigrationConflict(HedgeMigrationError):
    def __init__(self, message: str, conflicts: dict[str, Any]):
        super().__init__(message)
        self.conflicts = conflicts


@dataclass(frozen=True)
class MigrationStep:
    migration_id: str
    description: str
    apply: Callable[[Engine], dict[str, Any] | None]

    @property
    def checksum(self) -> str:
        try:
            number = int(self.migration_id.split("-", 2)[1])
        except (IndexError, ValueError):
            number = 999
        schema_version = LEGACY_SCHEMA_VERSION if number <= 15 else H3_SCHEMA_VERSION
        material = f"{schema_version}|{self.migration_id}|{self.description}"
        if number >= 16:
            try:
                source = python_inspect.getsource(self.apply)
            except (OSError, TypeError):
                source = repr(self.apply)
            material += "|" + hashlib.sha256(source.encode()).hexdigest()
        return hashlib.sha256(material.encode()).hexdigest()


@dataclass(frozen=True)
class HedgeMigrationReport:
    applied: tuple[str, ...] = field(default_factory=tuple)
    skipped: tuple[str, ...] = field(default_factory=tuple)
    backup_reference: str | None = None
    conflicts: dict[str, Any] = field(default_factory=dict)


CORE_COLUMNS: dict[str, tuple[tuple[str, str], ...]] = {
    "trades": (
        ("account_id", "VARCHAR(128)"),
        ("position_side", "VARCHAR(8)"),
        ("open_slot_key", "VARCHAR(512)"),
        ("hedge_version", "INTEGER"),
    ),
    "orders": (
        ("position_side", "VARCHAR(8)"),
        ("action", "VARCHAR(32)"),
        ("client_order_id", "VARCHAR(255)"),
        ("idempotency_key", "VARCHAR(255)"),
        ("submit_state", "VARCHAR(32)"),
    ),
}


def _table_exists(engine: Engine, table: str) -> bool:
    return table in inspect(engine).get_table_names()


def _column_names(engine: Engine, table: str) -> set[str]:
    if not _table_exists(engine, table):
        return set()
    return {column["name"] for column in inspect(engine).get_columns(table)}


def _index_names(engine: Engine, table: str) -> set[str]:
    if not _table_exists(engine, table):
        return set()
    return {index["name"] for index in inspect(engine).get_indexes(table) if index.get("name")}


def _unique_constraints(engine: Engine, table: str) -> list[dict[str, Any]]:
    if not _table_exists(engine, table):
        return []
    return [dict(item) for item in inspect(engine).get_unique_constraints(table)]


def _has_unique_columns(engine: Engine, table: str, columns: tuple[str, ...]) -> bool:
    expected = tuple(columns)
    return any(
        tuple(item.get("column_names") or ()) == expected
        for item in _unique_constraints(engine, table)
    )


def _quote(engine: Engine, identifier: str) -> str:
    return engine.dialect.identifier_preparer.quote(identifier)


def _memory_migration_lock(engine: Engine) -> threading.Lock:
    key = str(engine.url)
    with _MEMORY_LOCK_GUARD:
        return _MEMORY_LOCKS.setdefault(key, threading.Lock())


def _pid_is_alive(pid: int) -> bool:
    """Return whether *pid* currently identifies a live process.

    ``os.kill(pid, 0)`` is not a reliable existence probe on Windows: an
    impossible PID may raise ``OSError(87)`` (invalid parameter), which the
    previous implementation treated as "alive".  That left stale SQLite
    migration lock files in place until timeout.  Freqtrade already depends on
    psutil, so prefer its cross-platform PID implementation and retain a small
    stdlib fallback for minimal migration environments.
    """

    if isinstance(pid, bool) or pid <= 0:
        return False

    try:
        import psutil
    except ImportError:  # pragma: no cover - only for stripped migration tools
        psutil = None

    if psutil is not None:
        try:
            if not psutil.pid_exists(pid):
                return False
            process = psutil.Process(pid)
            if not process.is_running():
                return False
            zombie = getattr(psutil, "STATUS_ZOMBIE", None)
            return zombie is None or process.status() != zombie
        except psutil.NoSuchProcess:
            return False
        except psutil.ZombieProcess:
            return False
        except psutil.AccessDenied:
            return True
        except (OSError, ValueError):
            # A process that exists but cannot be inspected must be treated as
            # live so another runner cannot steal its migration lock.
            return True

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        # Windows uses winerror 87 / errno EINVAL for a non-existent or
        # impossible PID.  Other errors remain fail-closed.
        if getattr(exc, "winerror", None) == 87 or getattr(exc, "errno", None) == 22:
            return False
        return True
    return True


def _remove_stale_sqlite_lock(lock_path: Path) -> bool:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False

    now = datetime.now(UTC)
    expires_text = payload.get("expires_at")
    expired = False
    if expires_text:
        try:
            expires_at = datetime.fromisoformat(str(expires_text))
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=UTC)
            expired = expires_at <= now
        except ValueError:
            expired = False

    same_host = payload.get("hostname") == socket.gethostname()
    try:
        pid = int(payload.get("pid", -1))
    except (TypeError, ValueError):
        pid = -1
    dead_local_owner = same_host and not _pid_is_alive(pid)
    if not expired and not dead_local_owner:
        return False
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    return True


@contextmanager
def _migration_lock(engine: Engine, *, timeout_seconds: float = 60.0):
    """Serialize H3 migration runners across threads and host processes."""

    if engine.dialect.name == "postgresql":
        connection = engine.connect()
        acquired = False
        deadline = time.monotonic() + timeout_seconds
        try:
            while not acquired:
                acquired = bool(
                    connection.execute(
                        text(
                            "SELECT pg_try_advisory_lock("
                            "hashtextextended('freqtrade-hedge:h3-migration', 0))"
                        )
                    ).scalar_one()
                )
                if acquired:
                    break
                if time.monotonic() >= deadline:
                    raise HedgeMigrationError(
                        "Timed out waiting for PostgreSQL H3 migration lock"
                    )
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            yield
        finally:
            try:
                if acquired:
                    connection.execute(
                        text(
                            "SELECT pg_advisory_unlock("
                            "hashtextextended('freqtrade-hedge:h3-migration', 0))"
                        )
                    )
            finally:
                connection.close()
        return

    database = _sqlite_database_path(engine)
    if database is None:
        lock = _memory_migration_lock(engine)
        acquired = lock.acquire(timeout=timeout_seconds)
        if not acquired:
            raise HedgeMigrationError("Timed out waiting for in-memory H3 migration lock")
        try:
            yield
        finally:
            lock.release()
        return

    lock_path = database.with_suffix(database.suffix + ".h3-migration.lock")
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    created_at = datetime.now(UTC)
    payload = _canonical_json(
        {
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "created_at": created_at.isoformat(),
            "expires_at": (created_at + timedelta(seconds=timeout_seconds + 30)).isoformat(),
            "runner_id": uuid4().hex,
        }
    ).encode("utf-8")
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if _remove_stale_sqlite_lock(lock_path):
                continue
            if time.monotonic() >= deadline:
                raise HedgeMigrationError(
                    f"Timed out waiting for SQLite H3 migration lock: {lock_path}"
                )
            time.sleep(0.1)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        yield
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock_path.unlink(missing_ok=True)


def _add_column_if_missing(engine: Engine, table: str, column: str, sql_type: str) -> bool:
    if not _table_exists(engine, table) or column in _column_names(engine, table):
        return False
    with engine.begin() as connection:
        connection.execute(
            text(
                f"ALTER TABLE {_quote(engine, table)} "
                f"ADD COLUMN {_quote(engine, column)} {sql_type}"
            )
        )
    return True


def _ensure_migration_table(engine: Engine) -> None:
    from freqtrade.persistence.hedge_models import SchemaMigrationRecord

    SchemaMigrationRecord.__table__.create(engine, checkfirst=True)


def _load_record(engine: Engine, migration_id: str) -> dict[str, Any] | None:
    with engine.connect() as connection:
        row = connection.execute(
            text(
                "SELECT migration_id, checksum, state, attempt_count, backup_reference, "
                "conflict_report_json, error_message FROM hedge_schema_migrations "
                "WHERE migration_id = :migration_id"
            ),
            {"migration_id": migration_id},
        ).mappings().first()
        return dict(row) if row else None


def _upsert_running(
    engine: Engine,
    step: MigrationStep,
    *,
    backup_reference: str | None,
    runner_id: str,
) -> None:
    record = _load_record(engine, step.migration_id)
    now = _migration_timestamp(engine)
    with engine.begin() as connection:
        if record is None:
            connection.execute(
                text(
                    "INSERT INTO hedge_schema_migrations "
                    "(migration_id, checksum, state, attempt_count, started_at, "
                    "backup_reference, conflict_report_json, details_json, runner_id, "
                    "record_version) "
                    "VALUES (:migration_id, :checksum, 'RUNNING', 1, :started_at, "
                    ":backup_reference, '{}', '{}', :runner_id, 1)"
                ),
                {
                    "migration_id": step.migration_id,
                    "checksum": step.checksum,
                    "started_at": now,
                    "backup_reference": backup_reference,
                    "runner_id": runner_id,
                },
            )
        else:
            if record["checksum"] != step.checksum:
                raise HedgeMigrationError(
                    f"Checksum mismatch for {step.migration_id}; refusing silent migration drift."
                )
            connection.execute(
                text(
                    "UPDATE hedge_schema_migrations SET state='RUNNING', "
                    "attempt_count=attempt_count + 1, started_at=:started_at, applied_at=NULL, "
                    "failed_at=NULL, error_message=NULL, conflict_report_json='{}', "
                    "backup_reference=COALESCE(backup_reference, :backup_reference), "
                    "runner_id=:runner_id WHERE migration_id=:migration_id"
                ),
                {
                    "started_at": now,
                    "backup_reference": backup_reference,
                    "runner_id": runner_id,
                    "migration_id": step.migration_id,
                },
            )


def _mark_applied(engine: Engine, step: MigrationStep, details: dict[str, Any] | None) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE hedge_schema_migrations SET state='APPLIED', applied_at=:applied_at, "
                "failed_at=NULL, details_json=:details, error_message=NULL "
                "WHERE migration_id=:migration_id"
            ),
            {
                "applied_at": _migration_timestamp(engine),
                "details": _canonical_json(details or {}),
                "migration_id": step.migration_id,
            },
        )


def _mark_failed(
    engine: Engine,
    step: MigrationStep,
    error: Exception,
    *,
    conflicts: dict[str, Any] | None = None,
) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE hedge_schema_migrations SET state='FAILED', failed_at=:failed_at, "
                "error_message=:error_message, conflict_report_json=:conflicts "
                "WHERE migration_id=:migration_id"
            ),
            {
                "failed_at": _migration_timestamp(engine),
                "error_message": str(error)[:16000],
                "conflicts": _canonical_json(conflicts or {}),
                "migration_id": step.migration_id,
            },
        )


def _sqlite_database_path(engine: Engine) -> Path | None:
    if engine.dialect.name != "sqlite":
        return None
    database = engine.url.database
    if not database or database == ":memory:":
        return None
    path = Path(database).expanduser().resolve()
    return path if path.exists() else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _backup_sqlite(engine: Engine, backup_directory: Path | None) -> str:
    """Create and verify an immutable pre-migration SQLite snapshot.

    The backup is made through SQLite's online backup API, then independently
    reopened read-only and checked before its checksum is published.  This is
    deliberately more defensive on Windows where pooled SQLAlchemy handles and
    WAL sidecars can otherwise make a copy look complete while still being
    unusable for restore.
    """

    source_path = _sqlite_database_path(engine)
    if source_path is None:
        return "sqlite-memory:no-file-backup"

    directory = backup_directory or source_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = directory / f"{source_path.name}.pre-h3-{stamp}-{uuid4().hex[:8]}.sqlite"

    # Flush committed WAL content when possible.  A busy database may reject a
    # checkpoint; the online backup below still provides a consistent snapshot.
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql("PRAGMA wal_checkpoint(FULL)")
    except Exception:
        logger.debug("SQLite WAL checkpoint before H3 backup was not available", exc_info=True)

    source_uri = source_path.resolve().as_uri() + "?mode=ro"
    source = sqlite3.connect(source_uri, uri=True, timeout=30.0)
    destination = sqlite3.connect(str(target), timeout=30.0)
    try:
        source.backup(destination)
        destination.commit()
        integrity = destination.execute("PRAGMA quick_check").fetchone()
        if not integrity or integrity[0] != "ok":
            raise HedgeMigrationError(
                f"SQLite pre-migration backup integrity check failed: {integrity!r}"
            )
    finally:
        destination.close()
        source.close()

    # Reopen the finished file independently so the checksum is never emitted
    # for a snapshot that only worked through the original destination handle.
    verifier = sqlite3.connect(target.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        integrity = verifier.execute("PRAGMA quick_check").fetchone()
    finally:
        verifier.close()
    if not integrity or integrity[0] != "ok":
        target.unlink(missing_ok=True)
        raise HedgeMigrationError(
            f"SQLite pre-migration backup verification failed: {integrity!r}"
        )

    with target.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    checksum_path = target.with_suffix(target.suffix + ".sha256")
    checksum_path.write_text(
        f"{_file_sha256(target)}  {target.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    with checksum_path.open("rb+") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    return str(target)


def _postgresql_backup_tables(connection: Any, source_schema: str) -> list[str]:
    available = set(inspect(connection).get_table_names(schema=source_schema))
    selected = {
        table
        for table in available
        if table in {"trades", "orders"} or table.startswith("hedge_")
    }
    return sorted(selected)


def _backup_postgresql(engine: Engine) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    schema = f"freqtrade_h3_backup_{stamp}_{uuid4().hex[:6]}"
    quoted_schema = _quote(engine, schema)
    with engine.begin() as connection:
        connection.execute(text(f"CREATE SCHEMA {quoted_schema}"))
        source_schema = str(connection.execute(text("SELECT current_schema()" )).scalar_one())
        quoted_source_schema = _quote(engine, source_schema)
        connection.execute(
            text(
                f"CREATE TABLE {quoted_schema}.backup_manifest ("
                "table_name VARCHAR(255) PRIMARY KEY, row_count BIGINT NOT NULL, "
                "source_schema VARCHAR(255) NOT NULL, "
                "column_manifest_json TEXT NOT NULL, created_at TIMESTAMP NOT NULL)"
            )
        )
        for table in _postgresql_backup_tables(connection, source_schema):
            quoted_table = _quote(engine, table)
            connection.execute(
                text(
                    f"CREATE TABLE {quoted_schema}.{quoted_table} "
                    f"(LIKE {quoted_source_schema}.{quoted_table} INCLUDING ALL)"
                )
            )
            connection.execute(
                text(
                    f"INSERT INTO {quoted_schema}.{quoted_table} "
                    f"SELECT * FROM {quoted_source_schema}.{quoted_table}"
                )
            )
            row_count = int(
                connection.execute(
                    text(f"SELECT COUNT(*) FROM {quoted_schema}.{quoted_table}")
                ).scalar_one()
            )
            columns = [
                {
                    "name": column["name"],
                    "type": str(column["type"]),
                    "nullable": bool(column.get("nullable", True)),
                }
                for column in inspect(connection).get_columns(table, schema=source_schema)
            ]
            connection.execute(
                text(
                    f"INSERT INTO {quoted_schema}.backup_manifest "
                    "(table_name, row_count, source_schema, column_manifest_json, created_at) "
                    "VALUES (:table_name, :row_count, :source_schema, :columns, :created_at)"
                ),
                {
                    "table_name": table,
                    "row_count": row_count,
                    "source_schema": source_schema,
                    "columns": _canonical_json(columns),
                    "created_at": _migration_timestamp(engine),
                },
            )
    return f"postgresql-schema:{schema}"


def _postgresql_restore_table_order(
    connection: Any,
    target_schema: str,
    tables: Sequence[str],
) -> list[str]:
    """Return parent-before-child restore order for selected PostgreSQL tables.

    Only foreign keys inside the selected backup set participate.  Self references
    are safe under deferred constraints and therefore do not create graph edges.
    Cycles across multiple tables are rejected instead of silently relying on
    database-specific constraint timing.
    """

    selected = set(tables)
    children: dict[str, set[str]] = {table: set() for table in selected}
    indegree: dict[str, int] = {table: 0 for table in selected}
    inspector = inspect(connection)
    for child in sorted(selected):
        for foreign_key in inspector.get_foreign_keys(child, schema=target_schema):
            parent = str(foreign_key.get("referred_table") or "")
            referred_schema = foreign_key.get("referred_schema")
            if referred_schema not in (None, target_schema) or parent not in selected:
                continue
            if parent == child or child in children[parent]:
                continue
            children[parent].add(child)
            indegree[child] += 1

    ready = sorted(table for table, degree in indegree.items() if degree == 0)
    ordered: list[str] = []
    while ready:
        table = ready.pop(0)
        ordered.append(table)
        for child in sorted(children[table]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()

    if len(ordered) != len(selected):
        cyclic = sorted(table for table, degree in indegree.items() if degree > 0)
        raise HedgeMigrationError(
            "PostgreSQL restore dependency cycle cannot be restored safely: "
            f"{cyclic}"
        )
    return ordered


def _synchronize_postgresql_sequences(
    connection: Any,
    target_schema: str,
    tables: Sequence[str],
) -> None:
    """Advance serial/identity sequences to the restored maximum values."""

    quoted_target_schema = _quote(connection.engine, target_schema)
    inspector = inspect(connection)
    for table in tables:
        quoted_table = _quote(connection.engine, table)
        for column in inspector.get_columns(table, schema=target_schema):
            column_name = str(column["name"])
            sequence_name = connection.execute(
                text("SELECT pg_get_serial_sequence(:qualified_table, :column_name)"),
                {
                    "qualified_table": f"{target_schema}.{table}",
                    "column_name": column_name,
                },
            ).scalar_one_or_none()
            if not sequence_name:
                continue
            quoted_column = _quote(connection.engine, column_name)
            maximum = connection.execute(
                text(
                    f"SELECT MAX({quoted_column}) FROM "
                    f"{quoted_target_schema}.{quoted_table}"
                )
            ).scalar_one_or_none()
            if maximum is None:
                connection.execute(
                    text("SELECT setval(CAST(:sequence_name AS regclass), 1, false)"),
                    {"sequence_name": str(sequence_name)},
                )
            else:
                connection.execute(
                    text("SELECT setval(CAST(:sequence_name AS regclass), :value, true)"),
                    {"sequence_name": str(sequence_name), "value": int(maximum)},
                )


def restore_postgresql_backup(
    engine: Engine,
    backup_reference: str,
    *,
    drop_backup_schema: bool = False,
) -> dict[str, int]:
    """Restore all core and hedge tables from a PostgreSQL backup schema.

    The restore is transactional and verifies public/backup column identity
    before truncating any table.  This is intentionally fail-closed.
    """

    if engine.dialect.name != "postgresql":
        raise HedgeMigrationError("PostgreSQL backup restore requires PostgreSQL")
    prefix = "postgresql-schema:"
    if not backup_reference.startswith(prefix):
        raise HedgeMigrationError(f"Invalid PostgreSQL backup reference: {backup_reference}")
    schema = backup_reference[len(prefix) :]
    if not schema.startswith("freqtrade_h3_backup_"):
        raise HedgeMigrationError(f"Unsafe PostgreSQL backup schema: {schema}")
    quoted_schema = _quote(engine, schema)
    restored: dict[str, int] = {}
    with engine.begin() as connection:
        schemas = set(inspect(connection).get_schema_names())
        if schema not in schemas:
            raise HedgeMigrationError(f"PostgreSQL backup schema not found: {schema}")
        manifest_rows = connection.execute(
            text(
                f"SELECT table_name, row_count, source_schema, column_manifest_json "
                f"FROM {quoted_schema}.backup_manifest ORDER BY table_name"
            )
        ).mappings().all()
        if not manifest_rows:
            raise HedgeMigrationError(f"PostgreSQL backup manifest is empty: {schema}")
        target_schema = str(connection.execute(text("SELECT current_schema()" )).scalar_one())
        quoted_target_schema = _quote(engine, target_schema)
        available_public = set(inspect(connection).get_table_names(schema=target_schema))
        source_schemas = {str(row["source_schema"]) for row in manifest_rows}
        if len(source_schemas) != 1:
            raise HedgeMigrationError(
                f"PostgreSQL backup has inconsistent source schemas: {source_schemas}"
            )
        source_schema = next(iter(source_schemas))
        if source_schema != target_schema:
            raise HedgeMigrationError(
                "PostgreSQL backup source schema does not match the active restore schema: "
                f"{source_schema} != {target_schema}"
            )
        for row in manifest_rows:
            table = str(row["table_name"])
            if table not in available_public:
                raise HedgeMigrationError(
                    f"Restore target table is missing: {target_schema}.{table}"
                )
            public_columns = [
                column["name"]
                for column in inspect(connection).get_columns(table, schema=target_schema)
            ]
            backup_columns = [
                column["name"]
                for column in inspect(connection).get_columns(table, schema=schema)
            ]
            if public_columns != backup_columns:
                raise HedgeMigrationError(
                    f"Restore column mismatch for {table}: {public_columns} != {backup_columns}"
                )

        manifest_by_table = {str(row["table_name"]): row for row in manifest_rows}
        restore_order = _postgresql_restore_table_order(
            connection, target_schema, tuple(manifest_by_table)
        )
        connection.execute(text("SET CONSTRAINTS ALL DEFERRED"))
        truncate_targets = ", ".join(
            f"{quoted_target_schema}.{_quote(engine, table)}"
            for table in reversed(restore_order)
        )
        try:
            connection.execute(text(f"TRUNCATE TABLE {truncate_targets}"))
        except Exception as exc:
            raise HedgeMigrationError(
                "PostgreSQL restore refused to truncate the protected table set. "
                "A foreign-key dependent table may be missing from the backup manifest."
            ) from exc
        for table in restore_order:
            row = manifest_by_table[table]
            quoted_table = _quote(engine, table)
            connection.execute(
                text(
                    f"INSERT INTO {quoted_target_schema}.{quoted_table} "
                    f"SELECT * FROM {quoted_schema}.{quoted_table}"
                )
            )
            actual = int(
                connection.execute(
                    text(
                        f"SELECT COUNT(*) FROM {quoted_target_schema}.{quoted_table}"
                    )
                ).scalar_one()
            )
            expected = int(row["row_count"])
            if actual != expected:
                raise HedgeMigrationError(
                    f"PostgreSQL restore row count mismatch for {table}: {actual} != {expected}"
                )
            restored[table] = actual
        _synchronize_postgresql_sequences(connection, target_schema, restore_order)
        if drop_backup_schema:
            connection.execute(text(f"DROP SCHEMA {quoted_schema} CASCADE"))
    return restored


def create_pre_migration_backup(
    engine: Engine,
    backup_directory: str | Path | None = None,
) -> str:
    directory = Path(backup_directory) if backup_directory else None
    if engine.dialect.name == "sqlite":
        return _backup_sqlite(engine, directory)
    if engine.dialect.name == "postgresql":
        return _backup_postgresql(engine)
    raise HedgeMigrationError(
        f"H3 migrations support SQLite and PostgreSQL only, got {engine.dialect.name!r}."
    )


def restore_sqlite_backup(
    backup_path: str | Path,
    database_path: str | Path,
    *,
    require_checksum: bool = True,
) -> None:
    """Atomically restore a file-backed SQLite database from a pre-H3 backup."""

    backup = Path(backup_path).resolve()
    database = Path(database_path).resolve()
    if not backup.is_file():
        raise FileNotFoundError(backup)
    checksum_path = backup.with_suffix(backup.suffix + ".sha256")
    if require_checksum and not checksum_path.is_file():
        raise HedgeMigrationError(f"SQLite backup checksum file is missing: {checksum_path}")
    if checksum_path.is_file():
        expected = checksum_path.read_text(encoding="utf-8").split()[0]
        actual = _file_sha256(backup)
        if actual != expected:
            raise HedgeMigrationError(
                f"SQLite backup checksum mismatch for {backup}: {actual} != {expected}"
            )
    source = sqlite3.connect(f"file:{backup.as_posix()}?mode=ro", uri=True)
    try:
        integrity = source.execute("PRAGMA quick_check").fetchone()
    finally:
        source.close()
    if not integrity or integrity[0] != "ok":
        raise HedgeMigrationError(f"SQLite backup integrity check failed: {integrity!r}")
    database.parent.mkdir(parents=True, exist_ok=True)
    temporary = database.with_suffix(database.suffix + ".h3-restore.tmp")
    shutil.copy2(backup, temporary)
    # Windows rejects FlushFileBuffers/fsync on a read-only descriptor with
    # OSError(EBADF).  Open the copied database read/write even though no bytes
    # are changed, flush Python buffers, then fsync before the atomic replace.
    with temporary.open("r+b") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(database)
    for suffix in ("-wal", "-shm"):
        Path(str(database) + suffix).unlink(missing_ok=True)


def _step_add_core_columns(engine: Engine) -> dict[str, Any]:
    added: list[str] = []
    for table, columns in CORE_COLUMNS.items():
        for column, sql_type in columns:
            if _add_column_if_missing(engine, table, column, sql_type):
                added.append(f"{table}.{column}")
    return {"added_columns": added}


def _step_backfill_core(engine: Engine) -> dict[str, Any]:
    if not _table_exists(engine, "trades"):
        return {"trades": 0, "orders": 0}
    orders_exists = _table_exists(engine, "orders")
    order_columns = _column_names(engine, "orders") if orders_exists else set()
    true_literal = "1" if engine.dialect.name == "sqlite" else "TRUE"
    false_literal = "0" if engine.dialect.name == "sqlite" else "FALSE"
    concat = (
        "account_id || '|' || pair || '|' || position_side"
        if engine.dialect.name in {"sqlite", "postgresql"}
        else "CONCAT(account_id, '|', pair, '|', position_side)"
    )
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE trades SET account_id='default' "
                "WHERE account_id IS NULL OR account_id=''"
            )
        )
        connection.execute(
            text(
                "UPDATE trades SET position_side='BOTH' "
                "WHERE position_side IS NULL OR position_side NOT IN ('LONG','SHORT','BOTH')"
            )
        )
        connection.execute(
            text(
                "UPDATE trades SET hedge_version=0 "
                "WHERE hedge_version IS NULL OR hedge_version < 0"
            )
        )
        connection.execute(
            text(
                f"UPDATE trades SET open_slot_key={concat} WHERE is_open={true_literal} "
                "AND hedge_version >= 2 AND position_side IN ('LONG','SHORT') "
                "AND (open_slot_key IS NULL OR open_slot_key='')"
            )
        )
        connection.execute(
            text(
                "UPDATE trades SET open_slot_key=NULL "
                "WHERE hedge_version < 2 OR position_side='BOTH'"
            )
        )
        connection.execute(
            text(
                f"UPDATE trades SET open_slot_key=NULL WHERE is_open={false_literal} "
                "AND open_slot_key IS NOT NULL"
            )
        )
        orders_updated = 0
        order_backfill_mode = "orders-absent"
        if orders_exists:
            result = connection.execute(
                text(
                    "UPDATE orders SET idempotency_key=NULL "
                    "WHERE idempotency_key=''"
                )
            )
            orders_updated += int(result.rowcount or 0)

            if "ft_trade_id" in order_columns:
                result = connection.execute(
                    text(
                        "UPDATE orders SET position_side=(SELECT trades.position_side FROM trades "
                        "WHERE trades.id=orders.ft_trade_id) WHERE position_side IS NULL"
                    )
                )
                orders_updated += int(result.rowcount or 0)

            # Very old or intentionally minimal upstream schemas may not have
            # ft_trade_id.  Such orders are ordinary Freqtrade records and must
            # stay outside the side-aware Hedge identity until explicitly
            # upgraded by the application.
            result = connection.execute(
                text(
                    "UPDATE orders SET position_side='BOTH' "
                    "WHERE position_side IS NULL OR "
                    "position_side NOT IN ('LONG','SHORT','BOTH')"
                )
            )
            orders_updated += int(result.rowcount or 0)

            if "ft_is_open" in order_columns:
                submit_state_sql = (
                    "UPDATE orders SET submit_state=CASE "
                    "WHEN ft_is_open="
                    + true_literal
                    + " THEN 'ACKNOWLEDGED' ELSE 'TERMINAL' END "
                    "WHERE submit_state IS NULL"
                )
                order_backfill_mode = "ft_is_open"
            elif "status" in order_columns:
                # Preserve clearly active legacy exchange orders.  Unknown or
                # terminal statuses fail closed as TERMINAL, preventing a
                # restart from blindly resubmitting an ambiguous order.
                submit_state_sql = (
                    "UPDATE orders SET submit_state=CASE "
                    "WHEN LOWER(COALESCE(status, '')) IN "
                    "('open','new','partially_filled','partially filled') "
                    "THEN 'ACKNOWLEDGED' ELSE 'TERMINAL' END "
                    "WHERE submit_state IS NULL"
                )
                order_backfill_mode = "status"
            else:
                # Some upstream-isolation fixtures and ancient databases have
                # no order-lifecycle column at all.  Default to TERMINAL rather
                # than referencing a missing column or assuming an order is
                # safe to resubmit.
                submit_state_sql = (
                    "UPDATE orders SET submit_state='TERMINAL' "
                    "WHERE submit_state IS NULL"
                )
                order_backfill_mode = "fail-closed-terminal"

            result = connection.execute(text(submit_state_sql))
            orders_updated += int(result.rowcount or 0)
        trade_count = int(connection.execute(text("SELECT COUNT(*) FROM trades")).scalar_one())
    return {
        "trades": trade_count,
        "orders_updated": orders_updated,
        "order_backfill_mode": order_backfill_mode,
    }


def _collect_conflicts(engine: Engine) -> dict[str, Any]:
    conflicts: dict[str, Any] = {}
    if _table_exists(engine, "trades"):
        true_literal = "1" if engine.dialect.name == "sqlite" else "TRUE"
        id_aggregate = (
            "group_concat(id, ',')"
            if engine.dialect.name == "sqlite"
            else "string_agg(id::text, ',')"
        )
        with engine.connect() as connection:
            invalid_rows = connection.execute(
                text(
                    "SELECT id, account_id, pair, position_side FROM trades "
                    f"WHERE is_open={true_literal} AND "
                    "(open_slot_key IS NOT NULL OR hedge_version >= 2) AND ("
                    "account_id IS NULL OR account_id='' OR pair IS NULL OR pair='' OR "
                    "position_side IS NULL OR position_side NOT IN ('LONG','SHORT'))"
                )
            ).mappings().all()
            if invalid_rows:
                conflicts["invalid_active_trade_identity"] = [
                    dict(row) for row in invalid_rows
                ]
            rows = connection.execute(
                text(
                    "SELECT account_id, pair, position_side, COUNT(*) AS row_count, "
                    f"{id_aggregate} AS ids FROM trades WHERE is_open={true_literal} "
                    "AND open_slot_key IS NOT NULL "
                    "GROUP BY account_id, pair, position_side HAVING COUNT(*) > 1"
                )
            ).mappings().all()
            if rows:
                conflicts["active_trade_slots"] = [dict(row) for row in rows]
            key_rows = connection.execute(
                text(
                    "SELECT open_slot_key, COUNT(*) AS row_count, "
                    f"{id_aggregate} AS ids FROM trades WHERE open_slot_key IS NOT NULL "
                    "GROUP BY open_slot_key HAVING COUNT(*) > 1"
                )
            ).mappings().all()
            if key_rows:
                conflicts["open_slot_keys"] = [dict(row) for row in key_rows]
    if _table_exists(engine, "orders"):
        with engine.connect() as connection:
            duplicate_keys = connection.execute(
                text(
                    "SELECT idempotency_key, COUNT(*) AS row_count FROM orders "
                    "WHERE idempotency_key IS NOT NULL AND idempotency_key <> '' "
                    "GROUP BY idempotency_key HAVING COUNT(*) > 1"
                )
            ).mappings().all()
            if duplicate_keys:
                conflicts["order_idempotency_keys"] = [dict(row) for row in duplicate_keys]
    return conflicts


def _step_validate_legacy_conflicts(engine: Engine) -> dict[str, Any]:
    conflicts = _collect_conflicts(engine)
    if conflicts:
        raise HedgeMigrationConflict(
            "H3 migration found conflicting active hedge slots or idempotency keys; "
            "the database was left unmodified beyond reversible column/backfill steps.",
            conflicts,
        )
    return {"conflicts": 0}


def _step_create_ledger_tables(engine: Engine) -> dict[str, Any]:
    from freqtrade.persistence.hedge_models import create_hedge_tables

    create_hedge_tables(engine)
    return {"tables": sorted(REQUIRED_LEDGER_TABLES)}


def _create_index(engine: Engine, table: str, name: str, expression: str) -> bool:
    if name in _index_names(engine, table):
        return False
    with engine.begin() as connection:
        connection.execute(text(f"CREATE {expression}"))
    return True


def _step_create_core_indexes(engine: Engine) -> dict[str, Any]:
    created: list[str] = []
    if _table_exists(engine, "trades"):
        true_literal = "1" if engine.dialect.name == "sqlite" else "TRUE"
        if engine.dialect.name == "postgresql":
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE trades ALTER COLUMN account_id SET NOT NULL"))
                connection.execute(
                    text("ALTER TABLE trades ALTER COLUMN position_side SET NOT NULL")
                )
                connection.execute(
                    text("ALTER TABLE trades ALTER COLUMN hedge_version SET NOT NULL")
                )
        else:
            with engine.begin() as connection:
                connection.execute(text("DROP TRIGGER IF EXISTS trg_trades_h3_identity_insert"))
                connection.execute(text("DROP TRIGGER IF EXISTS trg_trades_h3_identity_update"))
                connection.execute(
                    text(
                        "CREATE TRIGGER trg_trades_h3_identity_insert "
                        "BEFORE INSERT ON trades WHEN NEW.open_slot_key IS NOT NULL AND "
                        "(NEW.is_open<>1 OR NEW.account_id IS NULL OR NEW.account_id='' "
                        "OR NEW.pair IS NULL OR NEW.pair='' OR NEW.position_side IS NULL "
                        "OR NEW.position_side NOT IN ('LONG','SHORT') OR NEW.hedge_version < 2) "
                        "BEGIN SELECT RAISE(ABORT, 'managed hedge trade identity is required'); END"
                    )
                )
                connection.execute(
                    text(
                        "CREATE TRIGGER trg_trades_h3_identity_update "
                        "BEFORE UPDATE OF is_open, account_id, pair, position_side, "
                        "open_slot_key, hedge_version ON trades "
                        "WHEN NEW.open_slot_key IS NOT NULL AND "
                        "(NEW.is_open<>1 OR NEW.account_id IS NULL OR NEW.account_id='' "
                        "OR NEW.pair IS NULL OR NEW.pair='' OR NEW.position_side IS NULL "
                        "OR NEW.position_side NOT IN ('LONG','SHORT') OR NEW.hedge_version < 2) "
                        "BEGIN SELECT RAISE(ABORT, 'managed hedge trade identity is required'); END"
                    )
                )
            created.extend(["trg_trades_h3_identity_insert", "trg_trades_h3_identity_update"])
        with engine.begin() as connection:
            connection.execute(text("DROP INDEX IF EXISTS uq_trades_active_account_symbol_side"))
        if _create_index(
            engine,
            "trades",
            "uq_trades_open_slot_key",
            "UNIQUE INDEX uq_trades_open_slot_key ON trades(open_slot_key) "
            "WHERE open_slot_key IS NOT NULL",
        ):
            created.append("uq_trades_open_slot_key")
    if _table_exists(engine, "orders"):
        if _create_index(
            engine,
            "orders",
            "uq_orders_idempotency_key",
            "UNIQUE INDEX uq_orders_idempotency_key ON orders(idempotency_key) "
            "WHERE idempotency_key IS NOT NULL AND idempotency_key <> ''",
        ):
            created.append("uq_orders_idempotency_key")
        if _create_index(
            engine,
            "orders",
            "ix_orders_account_side_recovery",
            "INDEX ix_orders_account_side_recovery "
            "ON orders(position_side, submit_state, client_order_id)",
        ):
            created.append("ix_orders_account_side_recovery")
    return {"created_indexes": created}


def _step_fix_fill_identity_scope(engine: Engine) -> dict[str, Any]:
    table = FILL_TABLE
    desired = ("exchange", "account_id", "symbol", "exchange_trade_id")
    if not _table_exists(engine, table):
        from freqtrade.persistence.hedge_models import FillEvent

        FillEvent.__table__.create(engine, checkfirst=True)
        return {"created": table, "rebuilt": False}
    if _has_unique_columns(engine, table, desired):
        return {"created": None, "rebuilt": False}

    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            for constraint in inspect(connection).get_unique_constraints(table):
                columns = tuple(constraint.get("column_names") or ())
                name = constraint.get("name")
                if name and columns == ("exchange", "account_id", "exchange_trade_id"):
                    connection.execute(
                        text(
                            f"ALTER TABLE {_quote(engine, table)} "
                            f"DROP CONSTRAINT {_quote(engine, name)}"
                        )
                    )
            connection.execute(
                text(
                    f"ALTER TABLE {_quote(engine, table)} ADD CONSTRAINT "
                    f"{_quote(engine, 'uq_hedge_fill_exchange_account_symbol_trade')} "
                    "UNIQUE (exchange, account_id, symbol, exchange_trade_id)"
                )
            )
        return {"created": None, "rebuilt": True}

    from freqtrade.persistence.hedge_models import FillEvent

    old_table = f"{table}_h3_v1_old"
    with engine.begin() as connection:
        tables = set(inspect(connection).get_table_names())
        if old_table not in tables:
            connection.execute(
                text(
                    f"ALTER TABLE {_quote(engine, table)} "
                    f"RENAME TO {_quote(engine, old_table)}"
                )
            )
        elif table in tables:
            connection.execute(text(f"DROP TABLE {_quote(engine, table)}"))

        index_rows = connection.exec_driver_sql(
            f"PRAGMA index_list({_quote(engine, old_table)})"
        ).mappings().all()
        for row in index_rows:
            name = str(row["name"])
            if not name.startswith("sqlite_autoindex_"):
                connection.execute(text(f"DROP INDEX IF EXISTS {_quote(engine, name)}"))

        FillEvent.__table__.create(connection, checkfirst=False)
        columns = [column.name for column in FillEvent.__table__.columns]
        quoted_columns = ", ".join(_quote(engine, column) for column in columns)
        connection.execute(
            text(
                f"INSERT INTO {_quote(engine, table)} ({quoted_columns}) "
                f"SELECT {quoted_columns} FROM {_quote(engine, old_table)}"
            )
        )
        connection.execute(text(f"DROP TABLE {_quote(engine, old_table)}"))
    return {"created": None, "rebuilt": True}



def _step_verify_v1(engine: Engine) -> dict[str, Any]:
    """Preserve the original H3-006 contract for interrupted v1 migrations.

    H3-006 predates the symbol-scoped Fill identity introduced by H3-007.
    Keeping this verifier backward-compatible allows a v1 database interrupted
    at H3-006 to advance to H3-007 instead of failing before the repair step.
    """

    missing_tables = sorted(LEGACY_REQUIRED_LEDGER_TABLES - set(inspect(engine).get_table_names()))
    missing_columns: list[str] = []
    for table, columns in CORE_COLUMNS.items():
        if not _table_exists(engine, table):
            continue
        available = _column_names(engine, table)
        missing_columns.extend(
            f"{table}.{column}" for column, _ in columns if column not in available
        )
    conflicts = _collect_conflicts(engine)
    if missing_tables or missing_columns or conflicts:
        raise HedgeMigrationError(
            _canonical_json(
                {
                    "missing_tables": missing_tables,
                    "missing_columns": missing_columns,
                    "conflicts": conflicts,
                }
            )
        )
    return {
        "schema_version": H3_SCHEMA_VERSION,
        "tables": len(LEGACY_REQUIRED_LEDGER_TABLES),
        "missing_columns": 0,
        "conflicts": 0,
    }

def _step_verify_v1_1(engine: Engine) -> dict[str, Any]:
    missing_tables = sorted(LEGACY_REQUIRED_LEDGER_TABLES - set(inspect(engine).get_table_names()))
    missing_columns: list[str] = []
    for table, columns in CORE_COLUMNS.items():
        if not _table_exists(engine, table):
            continue
        available = _column_names(engine, table)
        missing_columns.extend(
            f"{table}.{column}" for column, _ in columns if column not in available
        )
    missing_indexes: list[str] = []
    expected_indexes = {
        "trades": {
            "uq_trades_open_slot_key",
        },
        "orders": {
            "uq_orders_idempotency_key",
            "ix_orders_account_side_recovery",
        },
    }
    for table, expected in expected_indexes.items():
        if _table_exists(engine, table):
            missing_indexes.extend(
                f"{table}.{name}" for name in sorted(expected - _index_names(engine, table))
            )
    invalid_fill_identity = (
        _table_exists(engine, FILL_TABLE)
        and not _has_unique_columns(
            engine,
            FILL_TABLE,
            ("exchange", "account_id", "symbol", "exchange_trade_id"),
        )
    )
    conflicts = _collect_conflicts(engine)
    if missing_tables or missing_columns or missing_indexes or invalid_fill_identity or conflicts:
        raise HedgeMigrationError(
            _canonical_json(
                {
                    "missing_tables": missing_tables,
                    "missing_columns": missing_columns,
                    "missing_indexes": missing_indexes,
                    "invalid_fill_identity": invalid_fill_identity,
                    "conflicts": conflicts,
                }
            )
        )
    return {
        "schema_version": H3_SCHEMA_VERSION,
        "tables": len(LEGACY_REQUIRED_LEDGER_TABLES),
        "missing_columns": 0,
        "missing_indexes": 0,
        "invalid_fill_identity": False,
        "conflicts": 0,
    }


def _step_add_fill_sequence(engine: Engine) -> dict[str, Any]:
    added = _add_column_if_missing(engine, FILL_TABLE, "sequence_number", "INTEGER")
    created_index = False
    if _table_exists(engine, FILL_TABLE):
        created_index = _create_index(
            engine,
            FILL_TABLE,
            "ix_hedge_fill_sequence",
            "INDEX ix_hedge_fill_sequence ON hedge_fill_events "
            "(exchange, account_id, symbol, position_side, event_time, sequence_number)",
        )
    return {"added_column": added, "created_index": created_index}


def _step_enforce_single_current_position(engine: Engine) -> dict[str, Any]:
    table = "hedge_position_snapshots"
    if not _table_exists(engine, table):
        return {"repaired_rows": 0, "created_index": False}

    with engine.begin() as connection:
        rows = connection.execute(
            text(
                "SELECT id, account_id, symbol, position_side, source_event_time "
                "FROM hedge_position_snapshots WHERE is_current="
                + ("1" if engine.dialect.name == "sqlite" else "TRUE")
                + " ORDER BY account_id, symbol, position_side, "
                "source_event_time DESC, id DESC"
            )
        ).mappings().all()
        seen: set[tuple[str, str, str]] = set()
        stale_ids: list[int] = []
        for row in rows:
            key = (row["account_id"], row["symbol"], row["position_side"])
            if key in seen:
                stale_ids.append(int(row["id"]))
            else:
                seen.add(key)
        if stale_ids:
            connection.execute(
                text(
                    "UPDATE hedge_position_snapshots SET is_current="
                    + ("0" if engine.dialect.name == "sqlite" else "FALSE")
                    + " WHERE id IN :stale_ids"
                ).bindparams(bindparam("stale_ids", expanding=True)),
                {"stale_ids": stale_ids},
            )

    created = _create_index(
        engine,
        table,
        "uq_hedge_position_current",
        "UNIQUE INDEX uq_hedge_position_current ON hedge_position_snapshots "
        "(account_id, symbol, position_side) WHERE is_current="
        + ("1" if engine.dialect.name == "sqlite" else "TRUE"),
    )
    return {"repaired_rows": len(stale_ids), "created_index": created}


def _step_verify_v1_3(engine: Engine) -> dict[str, Any]:
    _step_verify_v1_1(engine)
    missing: dict[str, Any] = {}
    fill_columns = _column_names(engine, FILL_TABLE)
    if "sequence_number" not in fill_columns:
        missing["fill_columns"] = ["sequence_number"]
    fill_indexes = _index_names(engine, FILL_TABLE)
    if "ix_hedge_fill_sequence" not in fill_indexes:
        missing["fill_indexes"] = ["ix_hedge_fill_sequence"]
    position_indexes = _index_names(engine, "hedge_position_snapshots")
    if "uq_hedge_position_current" not in position_indexes:
        missing["position_indexes"] = ["uq_hedge_position_current"]
    if missing:
        raise HedgeMigrationError(_canonical_json(missing))
    return {"schema_version": H3_SCHEMA_VERSION, "release": H3_RELEASE}


def _step_harden_core_identity(engine: Engine) -> dict[str, Any]:
    """Harden active-trade identity and normalize optional order keys."""

    normalized_order_keys = 0
    if _table_exists(engine, "orders"):
        with engine.begin() as connection:
            result = connection.execute(
                text("UPDATE orders SET idempotency_key=NULL WHERE idempotency_key=''")
            )
            normalized_order_keys = int(result.rowcount or 0)

    conflicts = _collect_conflicts(engine)
    if conflicts:
        raise HedgeMigrationConflict(
            "H3 identity hardening found invalid or duplicate active trade identities.",
            conflicts,
        )

    triggers_recreated = False
    pair_not_null = False
    if _table_exists(engine, "trades"):
        if engine.dialect.name == "postgresql":
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE trades ALTER COLUMN pair SET NOT NULL"))
            pair_not_null = True
        else:
            with engine.begin() as connection:
                connection.execute(text("DROP TRIGGER IF EXISTS trg_trades_h3_identity_insert"))
                connection.execute(text("DROP TRIGGER IF EXISTS trg_trades_h3_identity_update"))
                connection.execute(
                    text(
                        "CREATE TRIGGER trg_trades_h3_identity_insert "
                        "BEFORE INSERT ON trades WHEN NEW.open_slot_key IS NOT NULL AND "
                        "(NEW.is_open<>1 OR NEW.account_id IS NULL OR NEW.account_id='' "
                        "OR NEW.pair IS NULL OR NEW.pair='' OR NEW.position_side IS NULL "
                        "OR NEW.position_side NOT IN ('LONG','SHORT') OR NEW.hedge_version < 2) "
                        "BEGIN SELECT RAISE(ABORT, "
                        "'managed hedge trade identity is required'); END"
                    )
                )
                connection.execute(
                    text(
                        "CREATE TRIGGER trg_trades_h3_identity_update "
                        "BEFORE UPDATE OF is_open, account_id, pair, position_side, "
                        "open_slot_key, hedge_version ON trades "
                        "WHEN NEW.open_slot_key IS NOT NULL AND "
                        "(NEW.is_open<>1 OR NEW.account_id IS NULL OR NEW.account_id='' "
                        "OR NEW.pair IS NULL OR NEW.pair='' OR NEW.position_side IS NULL "
                        "OR NEW.position_side NOT IN ('LONG','SHORT') OR NEW.hedge_version < 2) "
                        "BEGIN SELECT RAISE(ABORT, "
                        "'managed hedge trade identity is required'); END"
                    )
                )
            triggers_recreated = True
    return {
        "normalized_empty_idempotency_keys": normalized_order_keys,
        "sqlite_triggers_recreated": triggers_recreated,
        "postgresql_pair_not_null": pair_not_null,
    }


def _step_verify_hardened_identity(engine: Engine) -> dict[str, Any]:
    drift: dict[str, Any] = {}
    conflicts = _collect_conflicts(engine)
    if conflicts:
        drift["conflicts"] = conflicts

    if _table_exists(engine, "orders") and "idempotency_key" in _column_names(
        engine, "orders"
    ):
        with engine.connect() as connection:
            empty_keys = int(
                connection.execute(
                    text("SELECT COUNT(*) FROM orders WHERE idempotency_key=''")
                ).scalar_one()
            )
        if empty_keys:
            drift["empty_order_idempotency_keys"] = empty_keys

    if _table_exists(engine, "trades"):
        if engine.dialect.name == "sqlite":
            with engine.connect() as connection:
                trigger_rows = connection.execute(
                    text(
                        "SELECT name, sql FROM sqlite_master WHERE type='trigger' "
                        "AND name IN ('trg_trades_h3_identity_insert', "
                        "'trg_trades_h3_identity_update')"
                    )
                ).mappings().all()
            trigger_sql = {row["name"]: row["sql"] or "" for row in trigger_rows}
            required = {
                "trg_trades_h3_identity_insert",
                "trg_trades_h3_identity_update",
            }
            if set(trigger_sql) != required:
                drift["missing_trade_identity_triggers"] = sorted(
                    required - set(trigger_sql)
                )
            weak = sorted(name for name, sql in trigger_sql.items() if "NEW.pair" not in sql)
            if weak:
                drift["weak_trade_identity_triggers"] = weak
        else:
            pair_column = next(
                (
                    column
                    for column in inspect(engine).get_columns("trades")
                    if column["name"] == "pair"
                ),
                None,
            )
            if pair_column is None or pair_column.get("nullable", True):
                drift["postgresql_pair_nullable"] = True

    if drift:
        raise HedgeMigrationError(_canonical_json(drift))
    return {"schema_version": H3_SCHEMA_VERSION, "release": H3_RELEASE}


def _drop_index_if_exists(engine: Engine, name: str) -> bool:
    with engine.begin() as connection:
        connection.execute(text(f"DROP INDEX IF EXISTS {_quote(engine, name)}"))
    return True


def _step_add_full_function_columns(engine: Engine) -> dict[str, Any]:
    added: list[str] = []
    additions = (
        ("hedge_order_intents", "revision", "INTEGER NOT NULL DEFAULT 0"),
        ("hedge_order_snapshots", "action", "VARCHAR(32)"),
        ("hedge_fill_events", "action", "VARCHAR(32)"),
        ("hedge_position_snapshots", "sequence_number", "INTEGER"),
    )
    for table, column, sql_type in additions:
        if _add_column_if_missing(engine, table, column, sql_type):
            added.append(f"{table}.{column}")

    table = "hedge_position_snapshots"
    recreated: list[str] = []
    if _table_exists(engine, table):
        for name in ("uq_hedge_position_current_active", "uq_hedge_position_current"):
            _drop_index_if_exists(engine, name)
        true_literal = "1" if engine.dialect.name == "sqlite" else "TRUE"
        if _create_index(
            engine,
            table,
            "uq_hedge_position_current_active",
            "UNIQUE INDEX uq_hedge_position_current_active "
            "ON hedge_position_snapshots "
            "(exchange, account_id, symbol, position_side) "
            f"WHERE is_current={true_literal} AND is_active={true_literal}",
        ):
            recreated.append("uq_hedge_position_current_active")
        if _create_index(
            engine,
            table,
            "uq_hedge_position_current",
            "UNIQUE INDEX uq_hedge_position_current "
            "ON hedge_position_snapshots "
            "(exchange, account_id, symbol, position_side) "
            f"WHERE is_current={true_literal}",
        ):
            recreated.append("uq_hedge_position_current")
    return {"added_columns": added, "recreated_indexes": recreated}


def _step_verify_full_function(engine: Engine) -> dict[str, Any]:
    expected_columns = {
        "hedge_order_intents": {"revision"},
        "hedge_order_snapshots": {"action"},
        "hedge_fill_events": {"action"},
        "hedge_position_snapshots": {"sequence_number"},
    }
    missing: list[str] = []
    for table, columns in expected_columns.items():
        available = _column_names(engine, table)
        missing.extend(f"{table}.{column}" for column in sorted(columns - available))
    invalid_indexes: list[str] = []
    if _table_exists(engine, "hedge_position_snapshots"):
        indexes = {
            item.get("name"): tuple(item.get("column_names") or ())
            for item in inspect(engine).get_indexes("hedge_position_snapshots")
        }
        desired = ("exchange", "account_id", "symbol", "position_side")
        for name in ("uq_hedge_position_current", "uq_hedge_position_current_active"):
            if indexes.get(name) != desired:
                invalid_indexes.append(name)
    if missing or invalid_indexes:
        raise HedgeMigrationError(
            _canonical_json(
                {"missing_columns": missing, "invalid_indexes": invalid_indexes}
            )
        )
    return {"schema_version": H3_SCHEMA_VERSION, "release": H3_RELEASE}


def _step_add_v2_columns(engine: Engine) -> dict[str, Any]:
    """Add event-contract, approval, projection, risk, and resolution fields."""

    boolean = "BOOLEAN" if engine.dialect.name == "postgresql" else "INTEGER"
    additions: dict[str, tuple[tuple[str, str], ...]] = {
        "hedge_order_intents": (
            ("correlation_id", "VARCHAR(128)"),
            ("target_snapshot_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("approved_quantity", "VARCHAR(80)"),
            ("risk_snapshot_id", "VARCHAR(255)"),
            ("reason_codes_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("rules_version", "VARCHAR(64)"),
            ("expires_at", "TIMESTAMP"),
        ),
        "hedge_order_snapshots": (
            ("fact_key", "VARCHAR(64)"),
            ("correlation_id", "VARCHAR(128)"),
            ("remaining_quantity", "VARCHAR(80) NOT NULL DEFAULT '0'"),
            ("source_version", "VARCHAR(128)"),
            ("payload_version", "INTEGER NOT NULL DEFAULT 1"),
        ),
        "hedge_fill_events": (
            ("correlation_id", "VARCHAR(128)"),
            ("projection_status", "VARCHAR(16) NOT NULL DEFAULT 'APPLIED'"),
            ("projection_error", "TEXT"),
        ),
        "hedge_position_snapshots": (
            ("fact_key", "VARCHAR(64)"),
            ("venue_symbol", "VARCHAR(128)"),
            ("source_version", "VARCHAR(128)"),
            ("risk_source_snapshot_key", "VARCHAR(255)"),
        ),
        "hedge_account_risk_snapshots": (
            ("fact_key", "VARCHAR(64)"),
            ("equity", "VARCHAR(80) NOT NULL DEFAULT '0'"),
            ("gross_long_notional", "VARCHAR(80) NOT NULL DEFAULT '0'"),
            ("gross_short_notional", "VARCHAR(80) NOT NULL DEFAULT '0'"),
            ("pending_risk", "VARCHAR(80) NOT NULL DEFAULT '0'"),
            ("risk_data_valid", f"{boolean} NOT NULL DEFAULT 0"),
            ("source_snapshot_id", "VARCHAR(255)"),
            ("source_version", "VARCHAR(128)"),
            ("rules_version", "VARCHAR(64)"),
            ("reason_codes_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("projected_risk_json", "TEXT NOT NULL DEFAULT '{}'"),
        ),
        "hedge_account_events": (("fact_key", "VARCHAR(64)"),),
        "hedge_reconciliation_diffs": (
            ("resolved_by", "VARCHAR(128)"),
            ("resolution_note", "TEXT"),
        ),
        "hedge_strategy_side_states": (("exchange", "VARCHAR(64)"),),
        "hedge_event_outbox": (
            ("aggregate_sequence", "INTEGER"),
            ("correlation_id", "VARCHAR(128)"),
            ("causation_id", "VARCHAR(128)"),
            ("payload_version", "INTEGER NOT NULL DEFAULT 1"),
            ("event_version", "INTEGER NOT NULL DEFAULT 1"),
            ("contracts_version", "VARCHAR(64) NOT NULL DEFAULT 'hedge-contracts-v1'"),
            ("schema_version", "VARCHAR(64) NOT NULL DEFAULT 'h3-ledger-v2'"),
            ("exchange_time", "TIMESTAMP"),
            ("observed_time", "TIMESTAMP"),
            ("resolution_status", "VARCHAR(32)"),
            ("resolved_by", "VARCHAR(128)"),
            ("resolution_note", "TEXT"),
        ),
    }
    added: list[str] = []
    for table, columns in additions.items():
        for column, sql_type in columns:
            if _add_column_if_missing(engine, table, column, sql_type):
                added.append(f"{table}.{column}")

    from freqtrade.persistence.hedge_models import create_hedge_tables

    create_hedge_tables(engine)
    return {"added_columns": added, "created_v2_tables": sorted(REQUIRED_LEDGER_TABLES)}


def _normalize_order_status_sql() -> str:
    return (
        "CASE status "
        "WHEN 'NEW' THEN 'ACKNOWLEDGED' "
        "WHEN 'SUBMITTED' THEN 'ACKNOWLEDGED' "
        "WHEN 'PARTIALLY_FILLED' THEN 'PARTIAL' "
        "WHEN 'CANCELLED' THEN 'CANCELED' "
        "WHEN 'FAILED' THEN 'UNKNOWN' "
        "ELSE status END"
    )


def _step_backfill_v2_contracts(engine: Engine) -> dict[str, Any]:
    from freqtrade.persistence.hedge_contracts import canonical_symbol, stable_fact_key

    changed: dict[str, int] = {}
    with engine.begin() as connection:
        if _table_exists(engine, "hedge_order_intents"):
            result = connection.execute(
                text(
                    "UPDATE hedge_order_intents SET "
                    "correlation_id=COALESCE(NULLIF(correlation_id,''), idempotency_key), "
                    "target_snapshot_json=COALESCE(target_snapshot_json,'{}'), "
                    "reason_codes_json=COALESCE(reason_codes_json,'[]'), "
                    "approved_quantity=CASE WHEN approved_quantity IS NULL AND "
                    "status IN ('APPROVED','PREPARED','SUBMITTING','SUBMITTED',"
                    "'ACKNOWLEDGED','PARTIALLY_FILLED','PARTIAL','FILLED') "
                    "THEN requested_quantity ELSE approved_quantity END, "
                    f"status={_normalize_order_status_sql()}"
                )
            )
            changed["order_intents"] = int(result.rowcount or 0)

        if _table_exists(engine, "hedge_order_snapshots"):
            rows = connection.execute(
                text(
                    "SELECT id, snapshot_key, exchange, account_id, symbol, position_side, "
                    "exchange_order_id, source, source_event_time, sequence_number, "
                    "original_quantity, executed_quantity, intent_id, correlation_id "
                    "FROM hedge_order_snapshots ORDER BY id"
                )
            ).mappings().all()
            count = 0
            for row in rows:
                symbol = canonical_symbol(str(row["symbol"]))
                fact_key = stable_fact_key(
                    "order",
                    row["exchange"],
                    row["account_id"],
                    symbol,
                    row["position_side"],
                    row["exchange_order_id"],
                    row["source"],
                    row["source_event_time"],
                    row["sequence_number"],
                )
                original = max(Decimal(str(row["original_quantity"] or "0")), Decimal("0"))
                executed = max(Decimal(str(row["executed_quantity"] or "0")), Decimal("0"))
                remaining = max(original - executed, Decimal("0"))
                connection.execute(
                    text(
                        "UPDATE hedge_order_snapshots SET symbol=:symbol, fact_key=:fact_key, "
                        "remaining_quantity=:remaining, correlation_id=:correlation_id, "
                        f"status={_normalize_order_status_sql()} WHERE id=:id"
                    ),
                    {
                        "symbol": symbol,
                        "fact_key": fact_key,
                        "remaining": format(remaining, "f"),
                        "correlation_id": row["correlation_id"]
                        or row["intent_id"]
                        or row["snapshot_key"],
                        "id": row["id"],
                    },
                )
                count += 1
            changed["order_snapshots"] = count

        if _table_exists(engine, "hedge_fill_events"):
            rows = connection.execute(
                text("SELECT id, symbol, intent_id, event_id FROM hedge_fill_events")
            ).mappings().all()
            for row in rows:
                connection.execute(
                    text(
                        "UPDATE hedge_fill_events SET symbol=:symbol, "
                        "correlation_id=COALESCE(NULLIF(correlation_id,''), :correlation_id), "
                        "projection_status=COALESCE(projection_status,'APPLIED') WHERE id=:id"
                    ),
                    {
                        "symbol": canonical_symbol(str(row["symbol"])),
                        "correlation_id": row["intent_id"] or row["event_id"],
                        "id": row["id"],
                    },
                )
            changed["fills"] = len(rows)

        if _table_exists(engine, "hedge_position_snapshots"):
            rows = connection.execute(
                text(
                    "SELECT id, snapshot_key, exchange, account_id, symbol, position_side, "
                    "source, source_event_time, sequence_number FROM hedge_position_snapshots"
                )
            ).mappings().all()
            for row in rows:
                symbol = canonical_symbol(str(row["symbol"]))
                fact_key = stable_fact_key(
                    "position",
                    row["exchange"],
                    row["account_id"],
                    symbol,
                    row["position_side"],
                    row["source"],
                    row["source_event_time"],
                    row["sequence_number"],
                )
                connection.execute(
                    text(
                        "UPDATE hedge_position_snapshots SET symbol=:symbol, "
                        "venue_symbol=COALESCE(venue_symbol, symbol), fact_key=:fact_key, "
                        "risk_source_snapshot_key=COALESCE(risk_source_snapshot_key, snapshot_key) "
                        "WHERE id=:id"
                    ),
                    {"symbol": symbol, "fact_key": fact_key, "id": row["id"]},
                )
            changed["position_snapshots"] = len(rows)

        if _table_exists(engine, "hedge_account_risk_snapshots"):
            rows = connection.execute(
                text(
                    "SELECT id, snapshot_key, exchange, account_id, source, "
                    "source_event_time, margin_balance, gross_exposure, net_exposure "
                    "FROM hedge_account_risk_snapshots"
                )
            ).mappings().all()
            for row in rows:
                gross = Decimal(str(row["gross_exposure"] or "0"))
                net = Decimal(str(row["net_exposure"] or "0"))
                gross_long = max((gross + net) / Decimal("2"), Decimal("0"))
                gross_short = max((gross - net) / Decimal("2"), Decimal("0"))
                fact_key = stable_fact_key(
                    "risk",
                    row["exchange"],
                    row["account_id"],
                    row["source"],
                    row["source_event_time"],
                )
                connection.execute(
                    text(
                        "UPDATE hedge_account_risk_snapshots SET fact_key=:fact_key, "
                        "equity=COALESCE(NULLIF(equity,'0'), margin_balance), "
                        "gross_long_notional=:gross_long, "
                        "gross_short_notional=:gross_short, "
                        "source_snapshot_id=COALESCE(source_snapshot_id, snapshot_key), "
                        "risk_data_valid=CASE WHEN risk_state='READY' THEN 1 ELSE 0 END, "
                        "reason_codes_json=COALESCE(reason_codes_json,'[]'), "
                        "projected_risk_json=COALESCE(projected_risk_json,'{}') WHERE id=:id"
                    ),
                    {
                        "fact_key": fact_key,
                        "gross_long": format(gross_long, "f"),
                        "gross_short": format(gross_short, "f"),
                        "id": row["id"],
                    },
                )
            changed["risk_snapshots"] = len(rows)

        if _table_exists(engine, "hedge_account_events"):
            rows = connection.execute(
                text(
                    "SELECT id, exchange, account_id, event_type, asset, amount, symbol, "
                    "position_side, related_order_id, related_trade_id, transfer_direction, "
                    "source, event_time FROM hedge_account_events"
                )
            ).mappings().all()
            for row in rows:
                symbol = canonical_symbol(str(row["symbol"])) if row["symbol"] else ""
                fact_key = stable_fact_key(
                    "account-event",
                    row["exchange"],
                    row["account_id"],
                    row["event_type"],
                    row["asset"],
                    row["amount"],
                    symbol,
                    row["position_side"] or "",
                    row["related_order_id"] or "",
                    row["related_trade_id"] or "",
                    row["transfer_direction"] or "",
                    row["source"],
                    row["event_time"],
                )
                connection.execute(
                    text(
                        "UPDATE hedge_account_events SET fact_key=:fact_key, "
                        "symbol=CASE WHEN symbol IS NULL THEN NULL ELSE :symbol END "
                        "WHERE id=:id"
                    ),
                    {"fact_key": fact_key, "symbol": symbol, "id": row["id"]},
                )
            changed["account_events"] = len(rows)

        if _table_exists(engine, "hedge_strategy_side_states"):
            result = connection.execute(
                text(
                    "UPDATE hedge_strategy_side_states SET exchange='binance' "
                    "WHERE exchange IS NULL OR exchange=''"
                )
            )
            changed["strategy_states"] = int(result.rowcount or 0)

        if _table_exists(engine, "hedge_event_outbox"):
            rows = connection.execute(
                text(
                    "SELECT id, event_id, aggregate_type, aggregate_id, occurred_at, "
                    "correlation_id, observed_time FROM hedge_event_outbox "
                    "ORDER BY aggregate_type, aggregate_id, id"
                )
            ).mappings().all()
            sequences: dict[tuple[str, str], int] = {}
            for row in rows:
                key = (str(row["aggregate_type"]), str(row["aggregate_id"]))
                sequences[key] = sequences.get(key, 0) + 1
                connection.execute(
                    text(
                        "UPDATE hedge_event_outbox SET aggregate_sequence=:sequence, "
                        "correlation_id=COALESCE(NULLIF(correlation_id,''), event_id), "
                        "observed_time=COALESCE(observed_time, occurred_at), "
                        "payload_version=COALESCE(payload_version,1), "
                        "event_version=COALESCE(event_version,1), "
                        "contracts_version=COALESCE(contracts_version,'hedge-contracts-v1'), "
                        "schema_version='h3-ledger-v2' WHERE id=:id"
                    ),
                    {"sequence": sequences[key], "id": row["id"]},
                )
            changed["outbox"] = len(rows)
    return changed


def _step_rebuild_strategy_identity(engine: Engine) -> dict[str, Any]:
    table = "hedge_strategy_side_states"
    if not _table_exists(engine, table):
        return {"rebuilt": False}
    desired = ("exchange", "account_id", "symbol", "position_side", "strategy_name")
    if _has_unique_columns(engine, table, desired):
        return {"rebuilt": False}

    if engine.dialect.name == "postgresql":
        with engine.begin() as connection:
            for constraint in inspect(connection).get_unique_constraints(table):
                name = constraint.get("name")
                columns = tuple(constraint.get("column_names") or ())
                if name and columns != desired:
                    connection.execute(
                        text(
                            f"ALTER TABLE {_quote(engine, table)} "
                            f"DROP CONSTRAINT {_quote(engine, name)}"
                        )
                    )
            connection.execute(
                text(
                    f"ALTER TABLE {_quote(engine, table)} ADD CONSTRAINT "
                    f"{_quote(engine, 'uq_hedge_strategy_side_state')} UNIQUE "
                    "(exchange, account_id, symbol, position_side, strategy_name)"
                )
            )
        return {"rebuilt": True}

    from freqtrade.persistence.hedge_models import StrategySideState

    old_table = f"{table}_h3_v1_old"
    with engine.begin() as connection:
        connection.execute(text(f"ALTER TABLE {_quote(engine, table)} RENAME TO {old_table}"))
        for row in connection.exec_driver_sql(f"PRAGMA index_list({old_table})").mappings():
            name = str(row["name"])
            if not name.startswith("sqlite_autoindex_"):
                connection.execute(text(f"DROP INDEX IF EXISTS {_quote(engine, name)}"))
        StrategySideState.__table__.create(connection, checkfirst=False)
        new_columns = [column.name for column in StrategySideState.__table__.columns]
        old_columns = set(_column_names(engine, old_table))
        copy_columns = [column for column in new_columns if column in old_columns]
        quoted = ", ".join(_quote(engine, column) for column in copy_columns)
        connection.execute(
            text(
                f"INSERT INTO {_quote(engine, table)} ({quoted}) "
                f"SELECT {quoted} FROM {_quote(engine, old_table)}"
            )
        )
        connection.execute(text(f"DROP TABLE {_quote(engine, old_table)}"))
    return {"rebuilt": True}


def _step_create_v2_indexes(engine: Engine) -> dict[str, Any]:
    created: list[str] = []
    specs = (
        (
            "hedge_order_snapshots",
            "uq_hedge_order_snapshots_fact_key",
            "UNIQUE INDEX uq_hedge_order_snapshots_fact_key "
            "ON hedge_order_snapshots(fact_key) WHERE fact_key IS NOT NULL",
        ),
        (
            "hedge_position_snapshots",
            "uq_hedge_position_snapshots_fact_key",
            "UNIQUE INDEX uq_hedge_position_snapshots_fact_key "
            "ON hedge_position_snapshots(fact_key) WHERE fact_key IS NOT NULL",
        ),
        (
            "hedge_account_risk_snapshots",
            "uq_hedge_account_risk_fact_key",
            "UNIQUE INDEX uq_hedge_account_risk_fact_key "
            "ON hedge_account_risk_snapshots(fact_key) WHERE fact_key IS NOT NULL",
        ),
        (
            "hedge_account_events",
            "uq_hedge_account_events_fact_key",
            "UNIQUE INDEX uq_hedge_account_events_fact_key "
            "ON hedge_account_events(fact_key) WHERE fact_key IS NOT NULL",
        ),
        (
            "hedge_event_outbox",
            "uq_hedge_outbox_aggregate_sequence",
            "UNIQUE INDEX uq_hedge_outbox_aggregate_sequence ON hedge_event_outbox "
            "(aggregate_type, aggregate_id, aggregate_sequence)",
        ),
    )
    for table, name, expression in specs:
        if _table_exists(engine, table) and _create_index(engine, table, name, expression):
            created.append(name)
    return {"created_indexes": created}


def _step_verify_v2(engine: Engine) -> dict[str, Any]:
    missing_tables = sorted(REQUIRED_LEDGER_TABLES - set(inspect(engine).get_table_names()))
    expected_columns: dict[str, set[str]] = {
        "hedge_order_intents": {
            "correlation_id",
            "target_snapshot_json",
            "approved_quantity",
            "risk_snapshot_id",
            "reason_codes_json",
            "rules_version",
            "expires_at",
        },
        "hedge_order_snapshots": {
            "fact_key",
            "remaining_quantity",
            "correlation_id",
            "source_version",
            "payload_version",
        },
        "hedge_fill_events": {"correlation_id", "projection_status", "projection_error"},
        "hedge_position_snapshots": {
            "fact_key",
            "venue_symbol",
            "source_version",
            "risk_source_snapshot_key",
        },
        "hedge_account_risk_snapshots": {
            "fact_key",
            "equity",
            "gross_long_notional",
            "gross_short_notional",
            "pending_risk",
            "risk_data_valid",
            "source_snapshot_id",
            "source_version",
            "rules_version",
            "reason_codes_json",
            "projected_risk_json",
        },
        "hedge_account_events": {"fact_key"},
        "hedge_strategy_side_states": {"exchange"},
        "hedge_event_outbox": {
            "aggregate_sequence",
            "correlation_id",
            "payload_version",
            "event_version",
            "contracts_version",
            "schema_version",
            "observed_time",
        },
    }
    missing_columns: list[str] = []
    for table, columns in expected_columns.items():
        available = _column_names(engine, table)
        missing_columns.extend(f"{table}.{column}" for column in sorted(columns - available))
    desired_strategy_key = (
        "exchange",
        "account_id",
        "symbol",
        "position_side",
        "strategy_name",
    )
    invalid_strategy_identity = not _has_unique_columns(
        engine,
        "hedge_strategy_side_states",
        desired_strategy_key,
    )
    null_contract_rows: dict[str, int] = {}
    with engine.connect() as connection:
        for table, predicate in (
            ("hedge_order_intents", "correlation_id IS NULL OR correlation_id=''"),
            ("hedge_event_outbox", "aggregate_sequence IS NULL OR correlation_id IS NULL"),
        ):
            if _table_exists(engine, table):
                count = int(
                    connection.execute(
                        text(f"SELECT COUNT(*) FROM {_quote(engine, table)} WHERE {predicate}")
                    ).scalar_one()
                )
                if count:
                    null_contract_rows[table] = count
    if missing_tables or missing_columns or invalid_strategy_identity or null_contract_rows:
        raise HedgeMigrationError(
            _canonical_json(
                {
                    "missing_tables": missing_tables,
                    "missing_columns": missing_columns,
                    "invalid_strategy_identity": invalid_strategy_identity,
                    "null_contract_rows": null_contract_rows,
                }
            )
        )
    return {
        "schema_version": H3_SCHEMA_VERSION,
        "release": H3_RELEASE,
        "tables": len(REQUIRED_LEDGER_TABLES),
    }

def _step_scope_managed_trade_slots(engine: Engine) -> dict[str, Any]:
    """Remove the legacy global active-trade constraint and scope slots to Hedge rows.

    Releases through v1.2.9 marked every upstream Trade as LONG/SHORT and created
    a global unique index.  That broke normal Spot and one-way Futures behavior.
    Version 2 is the first explicit managed-trade marker.  Older ambiguous slot
    keys are cleared rather than allowed to block ordinary Freqtrade trades.
    """

    if not _table_exists(engine, "trades"):
        return {"repaired_legacy_rows": 0, "created_index": False}

    with engine.begin() as connection:
        connection.execute(text("DROP INDEX IF EXISTS uq_trades_active_account_symbol_side"))
        result = connection.execute(
            text(
                "UPDATE trades SET open_slot_key=NULL, position_side='BOTH', hedge_version=0 "
                "WHERE hedge_version IS NULL OR hedge_version < 2"
            )
        )
        repaired = int(result.rowcount or 0)
        connection.execute(
            text("UPDATE trades SET open_slot_key=NULL WHERE is_open="
                 + ("0" if engine.dialect.name == "sqlite" else "FALSE"))
        )

    created_index = _create_index(
        engine,
        "trades",
        "uq_trades_open_slot_key",
        "UNIQUE INDEX uq_trades_open_slot_key ON trades(open_slot_key) "
        "WHERE open_slot_key IS NOT NULL",
    )
    _step_harden_core_identity(engine)
    return {"repaired_legacy_rows": repaired, "created_index": created_index}


def _step_verify_scoped_trade_slots(engine: Engine) -> dict[str, Any]:
    if not _table_exists(engine, "trades"):
        return {"scoped": True}
    indexes = _index_names(engine, "trades")
    if "uq_trades_active_account_symbol_side" in indexes:
        raise HedgeMigrationError("Legacy global active-trade index is still present.")
    if "uq_trades_open_slot_key" not in indexes:
        raise HedgeMigrationError("Managed Hedge open-slot index is missing.")
    with engine.connect() as connection:
        invalid = int(
            connection.execute(
                text(
                    "SELECT COUNT(*) FROM trades WHERE open_slot_key IS NOT NULL AND "
                    "(position_side NOT IN ('LONG','SHORT') OR hedge_version < 2)"
                )
            ).scalar_one()
        )
    if invalid:
        raise HedgeMigrationError(f"Invalid managed Hedge slot rows remain: {invalid}")
    return {"scoped": True, "invalid_rows": 0}


def _step_create_stage_a_durable_paper_tables(engine: Engine) -> dict[str, Any]:
    from freqtrade.persistence.hedge_models import ActionGroupRow, PaperRuntimeCheckpointRow

    PaperRuntimeCheckpointRow.__table__.create(engine, checkfirst=True)
    ActionGroupRow.__table__.create(engine, checkfirst=True)
    return {
        "tables": (
            PaperRuntimeCheckpointRow.__tablename__,
            ActionGroupRow.__tablename__,
        )
    }


def _step_verify_stage_a_durable_paper_tables(engine: Engine) -> dict[str, Any]:
    required_by_table = {
        "hedge_paper_runtime_checkpoints": {
            "exchange",
            "account_id",
            "symbol",
            "source",
            "schema_version",
            "revision",
            "state_json",
        },
        "hedge_action_groups": {
            "action_group_id",
            "action_type",
            "account_id",
            "symbol",
            "members_json",
            "created_at",
            "updated_at",
        },
    }
    for table, required in required_by_table.items():
        if not _table_exists(engine, table):
            raise HedgeMigrationError(f"Stage A durable table is missing: {table}")
        missing = sorted(required - _column_names(engine, table))
        if missing:
            raise HedgeMigrationError(
                f"Stage A durable table {table} is missing columns: {missing}"
            )
    checkpoint_uniques = {
        tuple(sorted(item.get("column_names") or ()))
        for item in inspect(engine).get_unique_constraints(
            "hedge_paper_runtime_checkpoints"
        )
    }
    expected_checkpoint = tuple(sorted(("exchange", "account_id", "symbol", "source")))
    if expected_checkpoint not in checkpoint_uniques:
        raise HedgeMigrationError("Paper checkpoint source identity uniqueness is missing.")
    action_group_uniques = {
        tuple(sorted(item.get("column_names") or ()))
        for item in inspect(engine).get_unique_constraints("hedge_action_groups")
    }
    if ("action_group_id",) not in action_group_uniques:
        raise HedgeMigrationError("Action group identity uniqueness is missing.")
    return {"tables": tuple(required_by_table), "verified": True}


def _step_create_risk_approval_commits(engine: Engine) -> dict[str, Any]:
    from freqtrade.persistence.hedge_models import RiskApprovalCommitRow

    RiskApprovalCommitRow.__table__.create(engine, checkfirst=True)
    return {"table": RiskApprovalCommitRow.__tablename__}


def _step_verify_risk_approval_commits(engine: Engine) -> dict[str, Any]:
    table = "hedge_risk_approval_commits"
    if not _table_exists(engine, table):
        raise HedgeMigrationError("Durable risk approval commit table is missing.")
    columns = _column_names(engine, table)
    required = {
        "decision_id",
        "intent_id",
        "idempotency_key",
        "correlation_id",
        "risk_snapshot_id",
        "request_json",
        "risk_snapshot_json",
        "fencing_token",
        "approved_quantity",
        "approved_notional",
        "durable_reference",
    }
    missing = sorted(required - columns)
    if missing:
        raise HedgeMigrationError(
            f"Durable risk approval commit columns are missing: {missing}"
        )
    unique_sets = {
        tuple(sorted(item.get("column_names") or ()))
        for item in inspect(engine).get_unique_constraints(table)
    }
    for expected in (("decision_id",), ("idempotency_key",)):
        if expected not in unique_sets:
            raise HedgeMigrationError(
                f"Durable risk approval uniqueness is missing for {expected[0]}."
            )
    return {"table": table, "verified": True}



def _step_create_control_operations(engine: Engine) -> dict[str, Any]:
    from freqtrade.persistence.hedge_models import ControlOperationRow

    ControlOperationRow.__table__.create(engine, checkfirst=True)
    return {"table": ControlOperationRow.__tablename__}


def _step_verify_control_operations(engine: Engine) -> dict[str, Any]:
    table = "hedge_control_operations"
    if not _table_exists(engine, table):
        raise HedgeMigrationError("Durable control operation table is missing.")
    required = {
        "operation_id",
        "account_id",
        "idempotency_key",
        "action",
        "actor",
        "actor_role",
        "request_hash",
        "request_json",
        "state",
        "result_json",
        "created_at",
        "completed_at",
        "lease_owner",
        "lease_expires_at",
    }
    missing = sorted(required - _column_names(engine, table))
    if missing:
        raise HedgeMigrationError(
            f"Durable control operation columns are missing: {missing}"
        )
    unique_sets = {
        tuple(sorted(item.get("column_names") or ()))
        for item in inspect(engine).get_unique_constraints(table)
    }
    if tuple(sorted(("account_id", "idempotency_key"))) not in unique_sets:
        raise HedgeMigrationError("Control operation idempotency uniqueness is missing.")
    if ("operation_id",) not in unique_sets:
        raise HedgeMigrationError("Control operation identity uniqueness is missing.")
    return {"table": table, "verified": True}



def _step_add_control_operation_leases(engine: Engine) -> dict[str, Any]:
    table = "hedge_control_operations"
    if not _table_exists(engine, table):
        _step_create_control_operations(engine)
    added = {
        "lease_owner": _add_column_if_missing(
            engine, table, "lease_owner", "VARCHAR(128)"
        ),
        "lease_expires_at": _add_column_if_missing(
            engine, table, "lease_expires_at", "DATETIME"
        ),
    }
    return {"table": table, "added": added}


def _step_verify_control_operation_leases(engine: Engine) -> dict[str, Any]:
    table = "hedge_control_operations"
    required = {"lease_owner", "lease_expires_at", "revision", "state"}
    missing = sorted(required - _column_names(engine, table))
    if missing:
        raise HedgeMigrationError(
            f"Durable control operation lease columns are missing: {missing}"
        )
    return {"table": table, "verified": True}

def _step_create_r2_execution_authority(engine: Engine) -> dict[str, Any]:
    from freqtrade.persistence.hedge_models import (
        ExecutionIdempotencyRow,
        ExecutionOrderStateRow,
    )

    ExecutionOrderStateRow.__table__.create(engine, checkfirst=True)
    ExecutionIdempotencyRow.__table__.create(engine, checkfirst=True)
    return {
        "tables": (
            ExecutionOrderStateRow.__tablename__,
            ExecutionIdempotencyRow.__tablename__,
        )
    }


def _step_verify_r2_execution_authority(engine: Engine) -> dict[str, Any]:
    required_by_table = {
        "hedge_execution_order_states": {
            "client_order_id",
            "intent_id",
            "account_id",
            "symbol",
            "position_side",
            "idempotency_key",
            "approved_quantity",
            "lifecycle_status",
            "lifecycle_filled_quantity",
            "lifecycle_version",
        },
        "hedge_execution_idempotency": {
            "idempotency_key",
            "state",
            "client_order_id",
            "lease_owner",
            "lease_expires_at",
        },
    }
    for table, required in required_by_table.items():
        if not _table_exists(engine, table):
            raise HedgeMigrationError(f"R2 execution authority table is missing: {table}")
        missing = sorted(required - _column_names(engine, table))
        if missing:
            raise HedgeMigrationError(
                f"R2 execution authority table {table} is missing columns: {missing}"
            )
    order_uniques = {
        tuple(sorted(item.get("column_names") or ()))
        for item in inspect(engine).get_unique_constraints(
            "hedge_execution_order_states"
        )
    }
    for expected in (("client_order_id",), ("intent_id",), ("idempotency_key",)):
        if expected not in order_uniques:
            raise HedgeMigrationError(
                f"R2 execution authority uniqueness is missing for {expected[0]}."
            )
    idempotency_foreign_keys = inspect(engine).get_foreign_keys(
        "hedge_execution_idempotency"
    )
    if not any(
        tuple(item.get("constrained_columns") or ()) == ("client_order_id",)
        and item.get("referred_table") == "hedge_execution_order_states"
        for item in idempotency_foreign_keys
    ):
        raise HedgeMigrationError(
            "R2 idempotency completion pointer is missing its execution-order foreign key."
        )
    return {"tables": tuple(required_by_table), "verified": True}



def _step_create_execution_authority_tables(engine: Engine) -> dict[str, Any]:
    from freqtrade.persistence.hedge_models import (
        ExecutionBudgetReservationRow,
        ExecutionDailyBudgetRow,
        ExecutionIncomeEventRow,
    )

    tables = (ExecutionDailyBudgetRow, ExecutionBudgetReservationRow, ExecutionIncomeEventRow)
    for model in tables:
        model.__table__.create(engine, checkfirst=True)
    return {"tables": tuple(model.__tablename__ for model in tables)}


def _step_verify_execution_authority_tables(engine: Engine) -> dict[str, Any]:
    required_by_table = {
        "hedge_r5_daily_budgets": {
            "account_id", "utc_date", "orders", "turnover", "realized_loss",
            "gross_peak", "net_peak", "open_orders_peak", "revision",
        },
        "hedge_r5_budget_reservations": {
            "reservation_id", "account_id", "utc_date", "orders",
            "turnover", "state", "metadata_json",
        },
        "hedge_r5_income_events": {
            "exchange", "account_id", "external_event_id", "income_type",
            "amount", "occurred_at", "payload_json",
        },
    }
    for table, required in required_by_table.items():
        if not _table_exists(engine, table):
            raise HedgeMigrationError(f"execution authority table is missing: {table}")
        missing = sorted(required - _column_names(engine, table))
        if missing:
            raise HedgeMigrationError(
                f"execution authority table {table} is missing columns: {missing}"
            )
    budget_uniques = {
        tuple(sorted(item.get("column_names") or ()))
        for item in inspect(engine).get_unique_constraints("hedge_r5_daily_budgets")
    }
    if tuple(sorted(("account_id", "utc_date"))) not in budget_uniques:
        raise HedgeMigrationError("execution daily budget scope uniqueness is missing")
    income_uniques = {
        tuple(sorted(item.get("column_names") or ()))
        for item in inspect(engine).get_unique_constraints("hedge_r5_income_events")
    }
    if tuple(sorted(("exchange", "account_id", "external_event_id"))) not in income_uniques:
        raise HedgeMigrationError("execution income event identity uniqueness is missing")
    return {"tables": tuple(required_by_table), "verified": True}


def _verify_release(engine: Engine) -> None:
    _step_verify_v1_3(engine)
    _step_verify_hardened_identity(engine)
    _step_verify_full_function(engine)
    _step_verify_v2(engine)
    _step_verify_scoped_trade_slots(engine)
    _step_verify_risk_approval_commits(engine)
    _step_verify_stage_a_durable_paper_tables(engine)
    _step_verify_r2_execution_authority(engine)
    _step_verify_control_operations(engine)
    _step_verify_control_operation_leases(engine)
    _step_verify_execution_authority_tables(engine)


def _steps() -> tuple[MigrationStep, ...]:
    return (
        MigrationStep(
            "H3-001-core-columns",
            "Add nullable hedge fields to trades and orders",
            _step_add_core_columns,
        ),
        MigrationStep(
            "H3-002-core-backfill",
            "Backfill one-way rows and recovery state",
            _step_backfill_core,
        ),
        MigrationStep(
            "H3-003-conflict-check",
            "Fail closed on duplicate active slots",
            _step_validate_legacy_conflicts,
        ),
        MigrationStep(
            "H3-004-ledger-tables",
            "Create immutable ledger and outbox tables",
            _step_create_ledger_tables,
        ),
        MigrationStep(
            "H3-005-core-indexes",
            "Create active-slot and idempotency constraints",
            _step_create_core_indexes,
        ),
        MigrationStep("H3-006-verify", "Verify H3 schema and conflict-free state", _step_verify_v1),
        MigrationStep(
            "H3-007-fill-identity-scope",
            "Scope fill identity by exchange account and symbol",
            _step_fix_fill_identity_scope,
        ),
        MigrationStep(
            "H3-008-verify-v1.1",
            "Verify corrected fill identity and required indexes",
            _step_verify_v1_1,
        ),
        MigrationStep(
            "H3-009-fill-sequence",
            "Add deterministic fill sequence and replay index",
            _step_add_fill_sequence,
        ),
        MigrationStep(
            "H3-010-current-position",
            "Repair and enforce one current position projection per hedge side",
            _step_enforce_single_current_position,
        ),
        MigrationStep(
            "H3-011-verify-v1.3",
            "Verify hardened fill ordering and current projection constraints",
            _step_verify_v1_3,
        ),
        MigrationStep(
            "H3-012-core-identity-hardening",
            "Normalize optional order keys and enforce complete active trade identity",
            _step_harden_core_identity,
        ),
        MigrationStep(
            "H3-013-verify-core-identity",
            "Verify active trade identity guards and normalized order idempotency keys",
            _step_verify_hardened_identity,
        ),
        MigrationStep(
            "H3-014-full-function-columns",
            "Add intent revision, action facts, position sequence, and exchange-scoped projections",
            _step_add_full_function_columns,
        ),
        MigrationStep(
            "H3-015-verify-full-function",
            "Verify full-function ledger columns and projection identity",
            _step_verify_full_function,
        ),
        MigrationStep(
            "H3-016-v2-contract-columns",
            "Add event contracts, approval evidence, current orders, and ledger v2 tables",
            _step_add_v2_columns,
        ),
        MigrationStep(
            "H3-017-v2-contract-backfill",
            "Backfill canonical identities, approval evidence, risk facts, and event metadata",
            _step_backfill_v2_contracts,
        ),
        MigrationStep(
            "H3-018-strategy-exchange-identity",
            "Scope strategy side state by exchange",
            _step_rebuild_strategy_identity,
        ),
        MigrationStep(
            "H3-019-v2-natural-indexes",
            "Create natural fact and aggregate sequence uniqueness indexes",
            _step_create_v2_indexes,
        ),
        MigrationStep(
            "H3-020-verify-v2",
            "Verify complete H3 v2 ledger, event contract, and recovery schema",
            _step_verify_v2,
        ),
        MigrationStep(
            "H3-021-scope-managed-trade-slots",
            "Scope active Trade uniqueness to explicit Hedge-managed slot keys",
            _step_scope_managed_trade_slots,
        ),
        MigrationStep(
            "H3-022-verify-managed-trade-slots",
            "Verify ordinary Freqtrade trades are outside Hedge slot uniqueness",
            _step_verify_scoped_trade_slots,
        ),
        MigrationStep(
            "H3-023-durable-risk-approval-commits",
            "Create durable transaction handoff for approved risk reservations",
            _step_create_risk_approval_commits,
        ),
        MigrationStep(
            "H3-024-verify-durable-risk-approval-commits",
            "Verify durable risk approval ownership and idempotency constraints",
            _step_verify_risk_approval_commits,
        ),
        MigrationStep(
            "H3-025-stage-a-durable-paper-tables",
            "Create source-scoped Paper checkpoints and recoverable action groups",
            _step_create_stage_a_durable_paper_tables,
        ),
        MigrationStep(
            "H3-026-verify-stage-a-durable-paper-tables",
            "Verify source-scoped Paper checkpoint and action-group constraints",
            _step_verify_stage_a_durable_paper_tables,
        ),
        MigrationStep(
            "H3-027-r2-execution-authority",
            "Create SQL-authoritative execution-order and idempotency projections",
            _step_create_r2_execution_authority,
        ),
        MigrationStep(
            "H3-028-verify-r2-execution-authority",
            "Verify durable execution-order lifecycle and idempotency constraints",
            _step_verify_r2_execution_authority,
        ),
        MigrationStep(
            "H3-029-durable-control-operations",
            "Create durable idempotency and result records for control operations",
            _step_create_control_operations,
        ),
        MigrationStep(
            "H3-030-verify-durable-control-operations",
            "Verify control operation identity and idempotency constraints",
            _step_verify_control_operations,
        ),
        MigrationStep(
            "H3-031-control-operation-leases",
            "Add recoverable leases to dangerous control operations",
            _step_add_control_operation_leases,
        ),
        MigrationStep(
            "H3-032-verify-control-operation-leases",
            "Verify stale-operation recovery lease columns",
            _step_verify_control_operation_leases,
        ),
        MigrationStep(
            "H3-033-r5-authority-ledger",
            "Create R5 daily budget, reservation and immutable income authority",
            _step_create_execution_authority_tables,
        ),
        MigrationStep(
            "H3-034-verify-r5-authority-ledger",
            "Verify R5 budget lifecycle and income idempotency constraints",
            _step_verify_execution_authority_tables,
        ),
    )


def _write_conflict_report(
    backup_directory: str | Path | None,
    migration_id: str,
    conflicts: dict[str, Any],
) -> str | None:
    if backup_directory is None:
        return None
    directory = Path(backup_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{migration_id}-conflicts-{uuid4().hex[:8]}.json"
    payload = json.dumps(conflicts, ensure_ascii=False, indent=2, default=str)
    path.write_text(payload, encoding="utf-8")
    return str(path)


def _interrupted_backup_reference(
    engine: Engine,
    steps: tuple[MigrationStep, ...],
) -> str | None:
    """Reuse only the backup for an interrupted current migration batch."""

    if not _table_exists(engine, MIGRATION_TABLE):
        return None
    for step in steps:
        record = _load_record(engine, step.migration_id)
        if not record or record.get("state") == "APPLIED":
            continue
        reference = record.get("backup_reference")
        if reference:
            return str(reference)
    return None


def _partition_migration_steps(
    engine: Engine,
    steps: tuple[MigrationStep, ...],
) -> tuple[list[MigrationStep], list[str]]:
    pending: list[MigrationStep] = []
    skipped: list[str] = []
    for step in steps:
        record = _load_record(engine, step.migration_id)
        if record and record["state"] == "APPLIED":
            if record["checksum"] != step.checksum:
                raise HedgeMigrationError(
                    f"Checksum mismatch for applied migration {step.migration_id}."
                )
            skipped.append(step.migration_id)
        else:
            pending.append(step)
    return pending, skipped


def _apply_migration_step(
    engine: Engine,
    step: MigrationStep,
    *,
    backup_reference: str,
    backup_directory: str | Path | None,
    runner_id: str,
    fail_after_step: str | None,
) -> None:
    _upsert_running(
        engine,
        step,
        backup_reference=backup_reference,
        runner_id=runner_id,
    )
    try:
        details = step.apply(engine)
        if fail_after_step == step.migration_id:
            raise RuntimeError(f"Injected interruption after {step.migration_id}")
    except HedgeMigrationConflict as exc:
        conflicts = exc.conflicts
        report_path = _write_conflict_report(
            backup_directory,
            step.migration_id,
            conflicts,
        )
        if report_path:
            conflicts = {**conflicts, "report_path": report_path}
        _mark_failed(engine, step, exc, conflicts=conflicts)
        raise
    except Exception as exc:
        _mark_failed(engine, step, exc)
        raise HedgeMigrationError(
            f"H3 migration {step.migration_id} failed and can be retried: {exc}"
        ) from exc
    _mark_applied(engine, step, details)
    logger.info("Applied hedge database migration %s", step.migration_id)



def migration_plan_ids() -> tuple[str, ...]:
    """Return the ordered, immutable H3 migration identifiers.

    Tests and installers use this public view so adding a migration cannot
    leave hard-coded step counts or obsolete interruption cases behind.
    """

    return tuple(step.migration_id for step in _steps())


def hedge_migrations_pending(engine: Engine) -> bool:
    """Return whether any H3 migration is missing, failed, or checksum-drifted."""

    if engine.dialect.name not in {"sqlite", "postgresql"}:
        raise HedgeMigrationError(
            f"H3 supports SQLite and PostgreSQL only, got {engine.dialect.name!r}."
        )
    if not _table_exists(engine, MIGRATION_TABLE):
        return True
    for step in _steps():
        record = _load_record(engine, step.migration_id)
        if record is None or record["state"] != "APPLIED":
            return True
        if record["checksum"] != step.checksum:
            raise HedgeMigrationError(
                f"Checksum mismatch for applied migration {step.migration_id}."
            )
    return False


def prepare_hedge_migration_backup(
    engine: Engine,
    backup_directory: str | Path | None = None,
) -> str | None:
    """Create/reuse the backup before Freqtrade's native migration mutates the DB."""

    if not hedge_migrations_pending(engine):
        return None
    steps = _steps()
    interrupted = _interrupted_backup_reference(engine, steps)
    return interrupted or create_pre_migration_backup(engine, backup_directory)


def _run_hedge_migrations_unlocked(
    engine: Engine,
    decl_base: Any | None = None,
    previous_tables: list[str] | None = None,
    *,
    backup_directory: str | Path | None = None,
    fail_after_step: str | None = None,
    precreated_backup_reference: str | None = None,
) -> HedgeMigrationReport:
    """Run all H3 migrations and return an idempotence report.

    ``decl_base`` and ``previous_tables`` match Freqtrade's migration hook and
    are accepted for compatibility. Ledger tables use the shared ModelBase.
    ``fail_after_step`` exists solely for deterministic interruption tests.
    """

    del decl_base, previous_tables
    if engine.dialect.name not in {"sqlite", "postgresql"}:
        raise HedgeMigrationError(
            f"H3 supports SQLite and PostgreSQL only, got {engine.dialect.name!r}."
        )
    steps = _steps()
    migration_table_existed = _table_exists(engine, MIGRATION_TABLE)
    pristine_backup = precreated_backup_reference
    if not migration_table_existed and pristine_backup is None:
        pristine_backup = create_pre_migration_backup(engine, backup_directory)
    _ensure_migration_table(engine)
    pending, skipped = _partition_migration_steps(engine, steps)
    if not pending:
        _verify_release(engine)
        return HedgeMigrationReport(applied=(), skipped=tuple(skipped), backup_reference=None)

    backup_reference = (
        pristine_backup
        or _interrupted_backup_reference(engine, steps)
        or create_pre_migration_backup(engine, backup_directory)
    )
    runner_id = f"{socket.gethostname()}:{os.getpid()}:{uuid4().hex[:8]}"
    applied: list[str] = []
    for step in pending:
        _apply_migration_step(
            engine,
            step,
            backup_reference=backup_reference,
            backup_directory=backup_directory,
            runner_id=runner_id,
            fail_after_step=fail_after_step,
        )
        applied.append(step.migration_id)

    _verify_release(engine)
    return HedgeMigrationReport(
        applied=tuple(applied),
        skipped=tuple(skipped),
        backup_reference=backup_reference,
    )


def run_hedge_migrations(
    engine: Engine,
    decl_base: Any | None = None,
    previous_tables: list[str] | None = None,
    *,
    backup_directory: str | Path | None = None,
    fail_after_step: str | None = None,
    precreated_backup_reference: str | None = None,
    lock_timeout_seconds: float = 60.0,
) -> HedgeMigrationReport:
    """Run all H3 migrations under a cross-runner migration lock."""

    with _migration_lock(engine, timeout_seconds=lock_timeout_seconds):
        return _run_hedge_migrations_unlocked(
            engine,
            decl_base,
            previous_tables,
            backup_directory=backup_directory,
            fail_after_step=fail_after_step,
            precreated_backup_reference=precreated_backup_reference,
        )


def migration_status(engine: Engine) -> list[dict[str, Any]]:
    """Return migration records without mutating an unmigrated database."""

    if not _table_exists(engine, MIGRATION_TABLE):
        return []
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT migration_id, checksum, state, attempt_count, started_at, applied_at, "
                "failed_at, backup_reference, conflict_report_json, error_message "
                "FROM hedge_schema_migrations ORDER BY migration_id"
            )
        ).mappings().all()
        return [dict(row) for row in rows]
