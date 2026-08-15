"""Database-backed lease primitives for the single-writer invariant."""

from __future__ import annotations

import re
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Protocol


class LeaseUnavailable(RuntimeError):
    pass


class LeaseLost(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class LeaseRecord:
    lease_name: str
    owner_id: str
    fencing_token: int
    acquired_at_ms: int
    renewed_at_ms: int
    expires_at_ms: int

    def __post_init__(self) -> None:
        lease_name = _normalize_lease_name(self.lease_name)
        if not isinstance(self.owner_id, str) or not self.owner_id.strip():
            raise ValueError("owner_id must not be empty.")
        if len(self.owner_id.strip()) > 255:
            raise ValueError("owner_id must not exceed 255 characters.")
        if (
            isinstance(self.fencing_token, bool)
            or not isinstance(self.fencing_token, int)
            or self.fencing_token <= 0
        ):
            raise ValueError("fencing_token must be a positive integer.")
        for field_name in ("acquired_at_ms", "renewed_at_ms", "expires_at_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer.")
        if self.renewed_at_ms < self.acquired_at_ms:
            raise ValueError("renewed_at_ms must not precede acquired_at_ms.")
        if self.expires_at_ms != 0 and self.expires_at_ms <= self.renewed_at_ms:
            raise ValueError("A live lease expiry must be later than renewed_at_ms.")
        object.__setattr__(self, "lease_name", lease_name)
        object.__setattr__(self, "owner_id", self.owner_id.strip())

    def is_valid(self, *, now_ms: int) -> bool:
        if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
            raise ValueError("now_ms must be a nonnegative integer.")
        return now_ms < self.expires_at_ms


def _normalize_lease_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("lease_name must not be empty.")
    normalized = value.strip()
    if len(normalized) > 255:
        raise ValueError("lease_name must not exceed 255 characters.")
    return normalized


def _validate_lease_arguments(
    *,
    lease_name: str,
    owner_id: str,
    now_ms: int,
    ttl_ms: int,
) -> tuple[str, str, int, int]:
    lease_name = _normalize_lease_name(lease_name)
    if not isinstance(owner_id, str) or not owner_id.strip():
        raise ValueError("owner_id must not be empty.")
    if len(owner_id.strip()) > 255:
        raise ValueError("owner_id must not exceed 255 characters.")
    if isinstance(now_ms, bool) or not isinstance(now_ms, int) or now_ms < 0:
        raise ValueError("now_ms must be a nonnegative integer.")
    if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0:
        raise ValueError("ttl_ms must be a positive integer.")
    return lease_name, owner_id.strip(), now_ms, ttl_ms


class DatabaseLeaseStore(Protocol):
    def acquire(
        self,
        *,
        lease_name: str,
        owner_id: str,
        now_ms: int,
        ttl_ms: int,
    ) -> LeaseRecord | None: ...

    def renew(
        self,
        *,
        lease: LeaseRecord,
        now_ms: int,
        ttl_ms: int,
    ) -> LeaseRecord | None: ...

    def release(self, *, lease: LeaseRecord) -> bool: ...

    def read(self, *, lease_name: str) -> LeaseRecord | None: ...


class InMemoryDatabaseLeaseStore:
    """Deterministic test store with the same fencing semantics as the DB store."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[str, LeaseRecord] = {}

    def acquire(
        self,
        *,
        lease_name: str,
        owner_id: str,
        now_ms: int,
        ttl_ms: int,
    ) -> LeaseRecord | None:
        lease_name, owner_id, now_ms, ttl_ms = _validate_lease_arguments(
            lease_name=lease_name, owner_id=owner_id, now_ms=now_ms, ttl_ms=ttl_ms
        )
        with self._lock:
            current = self._records.get(lease_name)
            held_by_other = (
                current is not None
                and current.owner_id != owner_id
                and current.is_valid(now_ms=now_ms)
            )
            if held_by_other:
                return None
            if current is None:
                token = 1
                acquired_at = now_ms
            elif current.owner_id == owner_id and current.is_valid(now_ms=now_ms):
                token = current.fencing_token
                acquired_at = current.acquired_at_ms
            else:
                token = current.fencing_token + 1
                acquired_at = now_ms
            record = LeaseRecord(
                lease_name,
                owner_id,
                token,
                acquired_at,
                now_ms,
                now_ms + ttl_ms,
            )
            self._records[lease_name] = record
            return record

    def renew(
        self,
        *,
        lease: LeaseRecord,
        now_ms: int,
        ttl_ms: int,
    ) -> LeaseRecord | None:
        _validate_lease_arguments(
            lease_name=lease.lease_name,
            owner_id=lease.owner_id,
            now_ms=now_ms,
            ttl_ms=ttl_ms,
        )
        with self._lock:
            current = self._records.get(lease.lease_name)
            if (
                current is None
                or current.owner_id != lease.owner_id
                or current.fencing_token != lease.fencing_token
                or not current.is_valid(now_ms=now_ms)
            ):
                return None
            renewed = LeaseRecord(
                current.lease_name,
                current.owner_id,
                current.fencing_token,
                current.acquired_at_ms,
                now_ms,
                now_ms + ttl_ms,
            )
            self._records[current.lease_name] = renewed
            return renewed

    def release(self, *, lease: LeaseRecord) -> bool:
        with self._lock:
            current = self._records.get(lease.lease_name)
            if (
                current is None
                or current.owner_id != lease.owner_id
                or current.fencing_token != lease.fencing_token
            ):
                return False
            self._records[lease.lease_name] = LeaseRecord(
                current.lease_name,
                current.owner_id,
                current.fencing_token,
                current.acquired_at_ms,
                current.renewed_at_ms,
                0,
            )
            return True

    def read(self, *, lease_name: str) -> LeaseRecord | None:
        normalized_name = _normalize_lease_name(lease_name)
        with self._lock:
            return self._records.get(normalized_name)


class SqliteDatabaseLeaseStore:
    """Cross-process lease store using ``BEGIN IMMEDIATE`` and fencing tokens.

    The small lease table is intentionally isolated from Freqtrade migrations.
    Production deployments may pre-create it and set ``initialize_schema=False``.
    Direction three does not modify ``freqtrade/persistence/migrations.py``.
    """

    _SAFE_TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        database_path: str | Path,
        *,
        table_name: str = "hedge_single_writer_lease",
        initialize_schema: bool = True,
        busy_timeout_ms: int = 5000,
    ) -> None:
        if not self._SAFE_TABLE.fullmatch(table_name):
            raise ValueError("Unsafe lease table name.")
        self._database_path = str(database_path).strip()
        lowered_path = self._database_path.lower()
        if (
            not self._database_path
            or lowered_path == ":memory:"
            or lowered_path.startswith("file::memory:")
            or "mode=memory" in lowered_path
        ):
            raise ValueError(
                "SqliteDatabaseLeaseStore requires a file-backed database; "
                "use InMemoryDatabaseLeaseStore for in-memory tests."
            )
        if isinstance(busy_timeout_ms, bool) or not isinstance(busy_timeout_ms, int):
            raise ValueError("busy_timeout_ms must be an integer.")
        if busy_timeout_ms < 0:
            raise ValueError("busy_timeout_ms must be nonnegative.")
        self._table = table_name
        self._busy_timeout_ms = busy_timeout_ms
        if initialize_schema:
            self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database_path,
            timeout=max(self._busy_timeout_ms / 1000, 0.001),
            isolation_level=None,
        )
        connection.execute(f"PRAGMA busy_timeout={int(self._busy_timeout_ms)}")
        return connection

    def _initialize_schema(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(  # noqa: S608 - table name is regex validated.
                f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    lease_name TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    fencing_token INTEGER NOT NULL,
                    acquired_at_ms INTEGER NOT NULL,
                    renewed_at_ms INTEGER NOT NULL,
                    expires_at_ms INTEGER NOT NULL
                )
                """
            )

    @staticmethod
    def _row_to_record(row: tuple[object, ...] | None) -> LeaseRecord | None:
        if row is None:
            return None
        return LeaseRecord(
            lease_name=str(row[0]),
            owner_id=str(row[1]),
            fencing_token=int(row[2]),
            acquired_at_ms=int(row[3]),
            renewed_at_ms=int(row[4]),
            expires_at_ms=int(row[5]),
        )

    def acquire(
        self,
        *,
        lease_name: str,
        owner_id: str,
        now_ms: int,
        ttl_ms: int,
    ) -> LeaseRecord | None:
        lease_name, owner_id, now_ms, ttl_ms = _validate_lease_arguments(
            lease_name=lease_name, owner_id=owner_id, now_ms=now_ms, ttl_ms=ttl_ms
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            select_sql = (
                "SELECT lease_name, owner_id, fencing_token, acquired_at_ms, "
                "renewed_at_ms, expires_at_ms "
                f"FROM {self._table} WHERE lease_name = ?"
            )
            row = connection.execute(  # noqa: S608 - table name is regex validated.
                select_sql,
                (lease_name,),
            ).fetchone()
            current = self._row_to_record(row)
            held_by_other = (
                current is not None
                and current.owner_id != owner_id
                and current.is_valid(now_ms=now_ms)
            )
            if held_by_other:
                connection.rollback()
                return None
            if current is None:
                token = 1
                acquired_at = now_ms
                insert_sql = (
                    f"INSERT INTO {self._table} "
                    "(lease_name, owner_id, fencing_token, acquired_at_ms, "
                    "renewed_at_ms, expires_at_ms) VALUES (?, ?, ?, ?, ?, ?)"
                )
                connection.execute(  # noqa: S608 - table name is regex validated.
                    insert_sql,
                    (
                        lease_name,
                        owner_id,
                        token,
                        acquired_at,
                        now_ms,
                        now_ms + ttl_ms,
                    ),
                )
            else:
                same_live_owner = (
                    current.owner_id == owner_id
                    and current.is_valid(now_ms=now_ms)
                )
                token = (
                    current.fencing_token
                    if same_live_owner
                    else current.fencing_token + 1
                )
                acquired_at = current.acquired_at_ms if same_live_owner else now_ms
                update_sql = (
                    f"UPDATE {self._table} SET owner_id = ?, fencing_token = ?, "
                    "acquired_at_ms = ?, renewed_at_ms = ?, expires_at_ms = ? "
                    "WHERE lease_name = ?"
                )
                connection.execute(  # noqa: S608 - table name is regex validated.
                    update_sql,
                    (
                        owner_id,
                        token,
                        acquired_at,
                        now_ms,
                        now_ms + ttl_ms,
                        lease_name,
                    ),
                )
            connection.commit()
            return LeaseRecord(
                lease_name,
                owner_id,
                token,
                acquired_at,
                now_ms,
                now_ms + ttl_ms,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def renew(
        self,
        *,
        lease: LeaseRecord,
        now_ms: int,
        ttl_ms: int,
    ) -> LeaseRecord | None:
        _validate_lease_arguments(
            lease_name=lease.lease_name,
            owner_id=lease.owner_id,
            now_ms=now_ms,
            ttl_ms=ttl_ms,
        )
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            update_sql = (
                f"UPDATE {self._table} SET renewed_at_ms = ?, expires_at_ms = ? "
                "WHERE lease_name = ? AND owner_id = ? AND fencing_token = ? "
                "AND expires_at_ms > ?"
            )
            cursor = connection.execute(  # noqa: S608 - table name is regex validated.
                update_sql,
                (
                    now_ms,
                    now_ms + ttl_ms,
                    lease.lease_name,
                    lease.owner_id,
                    lease.fencing_token,
                    now_ms,
                ),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
            return LeaseRecord(
                lease.lease_name,
                lease.owner_id,
                lease.fencing_token,
                lease.acquired_at_ms,
                now_ms,
                now_ms + ttl_ms,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def release(self, *, lease: LeaseRecord) -> bool:
        with closing(self._connect()) as connection:
            release_sql = (
                f"UPDATE {self._table} SET expires_at_ms = 0 "
                "WHERE lease_name = ? AND owner_id = ? AND fencing_token = ?"
            )
            cursor = connection.execute(  # noqa: S608 - table name is regex validated.
                release_sql,
                (lease.lease_name, lease.owner_id, lease.fencing_token),
            )
            return cursor.rowcount == 1

    def read(self, *, lease_name: str) -> LeaseRecord | None:
        lease_name = _normalize_lease_name(lease_name)
        with closing(self._connect()) as connection:
            select_sql = (
                "SELECT lease_name, owner_id, fencing_token, acquired_at_ms, "
                "renewed_at_ms, expires_at_ms "
                f"FROM {self._table} WHERE lease_name = ?"
            )
            row = connection.execute(  # noqa: S608 - table name is regex validated.
                select_sql,
                (lease_name,),
            ).fetchone()
        return self._row_to_record(row)


class SqlAlchemyDatabaseLeaseStore:
    """SQLAlchemy lease store for SQLite and PostgreSQL deployments.

    SQLite writes start with ``BEGIN IMMEDIATE``. PostgreSQL uses row locks.
    ``INSERT ... ON CONFLICT DO NOTHING`` closes the absent-row acquisition
    race before the row is selected and conditionally updated.
    """

    _SAFE_TABLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

    def __init__(
        self,
        engine_or_url,
        *,
        table_name: str = "hedge_single_writer_lease",
        initialize_schema: bool = True,
    ) -> None:
        from sqlalchemy import create_engine
        from sqlalchemy.pool import NullPool

        if not self._SAFE_TABLE.fullmatch(table_name):
            raise ValueError("Unsafe lease table name.")
        self._owns_engine = isinstance(engine_or_url, str)
        self._engine = (
            create_engine(engine_or_url, future=True, poolclass=NullPool)
            if self._owns_engine
            else engine_or_url
        )
        try:
            self._table = table_name
            self._dialect = self._engine.dialect.name
            if self._dialect not in {"sqlite", "postgresql"}:
                raise ValueError(
                    "SqlAlchemyDatabaseLeaseStore supports only SQLite and PostgreSQL."
                )
            if self._dialect == "sqlite":
                database = str(
                    getattr(self._engine.url, "database", "") or ""
                ).strip().lower()
                query = getattr(self._engine.url, "query", {})
                if (
                    not database
                    or database == ":memory:"
                    or str(query.get("mode", "")).lower() == "memory"
                ):
                    raise ValueError(
                        "SqlAlchemyDatabaseLeaseStore requires file-backed SQLite; "
                        "use InMemoryDatabaseLeaseStore for in-memory tests."
                    )
            if initialize_schema:
                self._initialize_schema()
        except Exception:
            if self._owns_engine:
                self._engine.dispose()
            raise

    def close(self) -> None:
        """Dispose an internally-created SQLAlchemy engine."""

        if self._owns_engine:
            self._engine.dispose()

    def _initialize_schema(self) -> None:
        from sqlalchemy import text

        ddl = text(
            f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                lease_name VARCHAR(255) PRIMARY KEY,
                owner_id VARCHAR(255) NOT NULL,
                fencing_token BIGINT NOT NULL,
                acquired_at_ms BIGINT NOT NULL,
                renewed_at_ms BIGINT NOT NULL,
                expires_at_ms BIGINT NOT NULL
            )
            """
        )
        with self._engine.begin() as connection:
            connection.execute(ddl)

    def _begin_write(self):
        connection = self._engine.connect()
        try:
            if self._dialect == "sqlite":
                connection.exec_driver_sql("BEGIN IMMEDIATE")
            else:
                connection.begin()
            return connection
        except Exception:
            connection.close()
            raise

    @staticmethod
    def _mapping_to_record(row) -> LeaseRecord | None:
        if row is None:
            return None
        values = row._mapping
        return LeaseRecord(
            lease_name=str(values["lease_name"]),
            owner_id=str(values["owner_id"]),
            fencing_token=int(values["fencing_token"]),
            acquired_at_ms=int(values["acquired_at_ms"]),
            renewed_at_ms=int(values["renewed_at_ms"]),
            expires_at_ms=int(values["expires_at_ms"]),
        )

    def acquire(
        self,
        *,
        lease_name: str,
        owner_id: str,
        now_ms: int,
        ttl_ms: int,
    ) -> LeaseRecord | None:
        from sqlalchemy import text

        lease_name, owner_id, now_ms, ttl_ms = _validate_lease_arguments(
            lease_name=lease_name, owner_id=owner_id, now_ms=now_ms, ttl_ms=ttl_ms
        )
        connection = self._begin_write()
        try:
            expires_at = now_ms + ttl_ms
            insert_sql = text(
                f"""
                INSERT INTO {self._table} (
                    lease_name, owner_id, fencing_token, acquired_at_ms,
                    renewed_at_ms, expires_at_ms
                ) VALUES (
                    :lease_name, :owner_id, 1, :now_ms, :now_ms, :expires_at
                ) ON CONFLICT (lease_name) DO NOTHING
                """
            )
            inserted = connection.execute(
                insert_sql,
                {
                    "lease_name": lease_name,
                    "owner_id": owner_id,
                    "now_ms": now_ms,
                    "expires_at": expires_at,
                },
            )
            if inserted.rowcount == 1:
                record = LeaseRecord(
                    lease_name,
                    owner_id,
                    1,
                    now_ms,
                    now_ms,
                    expires_at,
                )
                connection.commit()
                return record

            lock_clause = " FOR UPDATE" if self._dialect == "postgresql" else ""
            select_sql = text(
                "SELECT lease_name, owner_id, fencing_token, acquired_at_ms, "
                "renewed_at_ms, expires_at_ms "
                f"FROM {self._table} WHERE lease_name = :lease_name{lock_clause}"
            )
            current = self._mapping_to_record(
                connection.execute(select_sql, {"lease_name": lease_name}).first()
            )
            if current is None:
                connection.rollback()
                return None
            if current.owner_id != owner_id and current.is_valid(now_ms=now_ms):
                connection.rollback()
                return None

            same_live_owner = (
                current.owner_id == owner_id and current.is_valid(now_ms=now_ms)
            )
            token = current.fencing_token if same_live_owner else current.fencing_token + 1
            acquired_at = current.acquired_at_ms if same_live_owner else now_ms
            update_sql = text(
                f"""
                UPDATE {self._table}
                SET owner_id = :owner_id,
                    fencing_token = :token,
                    acquired_at_ms = :acquired_at,
                    renewed_at_ms = :now_ms,
                    expires_at_ms = :expires_at
                WHERE lease_name = :lease_name
                """
            )
            connection.execute(
                update_sql,
                {
                    "owner_id": owner_id,
                    "token": token,
                    "acquired_at": acquired_at,
                    "now_ms": now_ms,
                    "expires_at": expires_at,
                    "lease_name": lease_name,
                },
            )
            connection.commit()
            return LeaseRecord(
                lease_name,
                owner_id,
                token,
                acquired_at,
                now_ms,
                expires_at,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def renew(
        self,
        *,
        lease: LeaseRecord,
        now_ms: int,
        ttl_ms: int,
    ) -> LeaseRecord | None:
        from sqlalchemy import text

        _validate_lease_arguments(
            lease_name=lease.lease_name,
            owner_id=lease.owner_id,
            now_ms=now_ms,
            ttl_ms=ttl_ms,
        )
        connection = self._begin_write()
        try:
            expires_at = now_ms + ttl_ms
            update_sql = text(
                f"""
                UPDATE {self._table}
                SET renewed_at_ms = :now_ms, expires_at_ms = :expires_at
                WHERE lease_name = :lease_name
                  AND owner_id = :owner_id
                  AND fencing_token = :token
                  AND expires_at_ms > :now_ms
                """
            )
            cursor = connection.execute(
                update_sql,
                {
                    "now_ms": now_ms,
                    "expires_at": expires_at,
                    "lease_name": lease.lease_name,
                    "owner_id": lease.owner_id,
                    "token": lease.fencing_token,
                },
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            connection.commit()
            return LeaseRecord(
                lease.lease_name,
                lease.owner_id,
                lease.fencing_token,
                lease.acquired_at_ms,
                now_ms,
                expires_at,
            )
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def release(self, *, lease: LeaseRecord) -> bool:
        from sqlalchemy import text

        connection = self._begin_write()
        try:
            release_sql = text(
                f"""
                UPDATE {self._table}
                SET expires_at_ms = 0
                WHERE lease_name = :lease_name
                  AND owner_id = :owner_id
                  AND fencing_token = :token
                """
            )
            cursor = connection.execute(
                release_sql,
                {
                    "lease_name": lease.lease_name,
                    "owner_id": lease.owner_id,
                    "token": lease.fencing_token,
                },
            )
            connection.commit()
            return cursor.rowcount == 1
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def read(self, *, lease_name: str) -> LeaseRecord | None:
        from sqlalchemy import text

        lease_name = _normalize_lease_name(lease_name)
        select_sql = text(
            "SELECT lease_name, owner_id, fencing_token, acquired_at_ms, "
            "renewed_at_ms, expires_at_ms "
            f"FROM {self._table} WHERE lease_name = :lease_name"
        )
        with self._engine.connect() as connection:
            row = connection.execute(select_sql, {"lease_name": lease_name}).first()
        return self._mapping_to_record(row)
