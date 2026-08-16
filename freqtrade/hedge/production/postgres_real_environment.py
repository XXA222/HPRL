"""PostgreSQL real-environment acceptance for HPRL Runtime Closure R3.

This module intentionally separates three evidence classes:

* core database semantics: real execution ledger / journal / checkpoint / visibility;
* logical backup and isolated restore using PostgreSQL's own pg_dump/pg_restore tools;
* failover identity and writer fencing across a routed target and the *direct* old node.

Nothing here promotes a replica, drops a database, creates a restore database, or rewrites
production configuration.  Destructive orchestration remains an operator responsibility.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Mapping

from .database_runtime import PostgresProbeRunner
from .postgres_runtime_closure import (
    PostgresRuntimeClosureReport,
    PostgresRuntimeClosureRunner,
    PostgresRestoreSnapshot,
    capture_postgres_restore_snapshot,
    verify_postgres_restore,
)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _execute(connection: Any, sql: str, params: tuple[object, ...] = ()) -> None:
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params)
    finally:
        close = getattr(cursor, "close", None)
        if callable(close):
            close()


def _close(connection: object | None) -> None:
    if connection is None:
        return
    close = getattr(connection, "close", None)
    if callable(close):
        close()


@dataclass(frozen=True, slots=True)
class PostgresNodeIdentity:
    database_name: str
    server_addr: str
    server_port: int | None
    backend_pid: int
    in_recovery: bool
    transaction_read_only: bool
    server_version: str
    system_identifier: str
    wal_position: str
    observed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _aware(self.observed_at))

    @property
    def endpoint(self) -> str:
        return f"{self.server_addr}:{self.server_port or 0}"

    @property
    def writable_primary(self) -> bool:
        return not self.in_recovery and not self.transaction_read_only


def capture_postgres_node_identity(connection: Any, *, now: datetime) -> PostgresNodeIdentity:
    database = str(_scalar(connection, "SELECT current_database()") or "")
    addr = str(_scalar(connection, "SELECT COALESCE(inet_server_addr()::text,'')") or "")
    raw_port = _scalar(connection, "SELECT inet_server_port()")
    port = None if raw_port is None else int(raw_port)
    pid = int(_scalar(connection, "SELECT pg_backend_pid()"))
    in_recovery = bool(_scalar(connection, "SELECT pg_is_in_recovery()"))
    read_only = str(_scalar(connection, "SHOW transaction_read_only") or "").lower() in {
        "on", "true", "1",
    }
    version = str(_scalar(connection, "SHOW server_version") or "")
    system_identifier = ""
    try:
        system_identifier = str(
            _scalar(connection, "SELECT system_identifier::text FROM pg_control_system()") or ""
        )
    except Exception:
        rollback = getattr(connection, "rollback", None)
        if callable(rollback):
            rollback()
    wal_position = ""
    try:
        wal_sql = (
            "SELECT COALESCE(pg_last_wal_replay_lsn()::text,'')"
            if in_recovery
            else "SELECT COALESCE(pg_current_wal_lsn()::text,'')"
        )
        wal_position = str(_scalar(connection, wal_sql) or "")
    except Exception:
        rollback = getattr(connection, "rollback", None)
        if callable(rollback):
            rollback()
    return PostgresNodeIdentity(
        database_name=database,
        server_addr=addr,
        server_port=port,
        backend_pid=pid,
        in_recovery=in_recovery,
        transaction_read_only=read_only,
        server_version=version,
        system_identifier=system_identifier,
        wal_position=wal_position,
        observed_at=_aware(now),
    )


@dataclass(frozen=True, slots=True)
class PostgresR3CoreReport:
    runtime: PostgresRuntimeClosureReport
    node: PostgresNodeIdentity
    reconnect_node: PostgresNodeIdentity
    same_database: bool
    same_cluster: bool
    direct_primary_writable: bool
    evidence_sha256: str
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.runtime.passed and not self.reasons


def run_postgres_r3_core(
    *,
    connection_factory: Callable[[], Any],
    session_factory: object,
    engine: object,
    symbol: str,
    now: datetime,
) -> PostgresR3CoreReport:
    observed = _aware(now)
    runtime = PostgresRuntimeClosureRunner(
        connection_factory=connection_factory,
        session_factory=session_factory,
        engine=engine,
        symbol=symbol,
    ).run(now=observed)
    first = second = None
    try:
        first = connection_factory()
        node = capture_postgres_node_identity(first, now=observed)
    finally:
        _close(first)
    try:
        second = connection_factory()
        reconnect = capture_postgres_node_identity(second, now=observed)
    finally:
        _close(second)
    reasons: list[str] = []
    same_database = node.database_name == reconnect.database_name and bool(node.database_name)
    if not same_database:
        reasons.append("POSTGRES_RECONNECT_DATABASE_IDENTITY_CHANGED")
    same_cluster = True
    if node.system_identifier and reconnect.system_identifier:
        same_cluster = node.system_identifier == reconnect.system_identifier
    if not same_cluster:
        reasons.append("POSTGRES_RECONNECT_CLUSTER_IDENTITY_CHANGED")
    writable = node.writable_primary and reconnect.writable_primary
    if not writable:
        reasons.append("POSTGRES_CORE_TARGET_NOT_WRITABLE_PRIMARY")
    if not runtime.passed:
        reasons.append("POSTGRES_RUNTIME_CLOSURE_FAILED")
    payload = {
        "runtime_passed": runtime.passed,
        "node": asdict(node),
        "reconnect": asdict(reconnect),
        "same_database": same_database,
        "same_cluster": same_cluster,
        "writable": writable,
    }
    digest = sha256(json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()
    return PostgresR3CoreReport(
        runtime=runtime,
        node=node,
        reconnect_node=reconnect,
        same_database=same_database,
        same_cluster=same_cluster,
        direct_primary_writable=writable,
        evidence_sha256=digest,
        reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True, slots=True)
class PostgresCliCapability:
    pg_dump_path: str
    pg_restore_path: str
    pg_dump_version: str
    pg_restore_version: str

    @property
    def passed(self) -> bool:
        return bool(self.pg_dump_path and self.pg_restore_path)


def _version(path: str) -> str:
    if not path:
        return ""
    completed = subprocess.run(
        [path, "--version"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=15, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else ""


def probe_postgres_cli() -> PostgresCliCapability:
    dump = shutil.which("pg_dump") or ""
    restore = shutil.which("pg_restore") or ""
    return PostgresCliCapability(dump, restore, _version(dump), _version(restore))


def _conninfo_to_pg_env(dsn: str) -> dict[str, str]:
    try:
        from psycopg.conninfo import conninfo_to_dict  # type: ignore
    except ImportError as exc:  # pragma: no cover - runtime bootstrap owns this
        raise RuntimeError("psycopg is required to parse PostgreSQL DSN safely") from exc
    info = conninfo_to_dict(dsn)
    mapping = {
        "host": "PGHOST", "hostaddr": "PGHOSTADDR", "port": "PGPORT",
        "dbname": "PGDATABASE", "user": "PGUSER", "password": "PGPASSWORD",
        "sslmode": "PGSSLMODE", "sslrootcert": "PGSSLROOTCERT",
        "sslcert": "PGSSLCERT", "sslkey": "PGSSLKEY", "target_session_attrs": "PGTARGETSESSIONATTRS",
        "connect_timeout": "PGCONNECT_TIMEOUT", "application_name": "PGAPPNAME",
    }
    env: dict[str, str] = {}
    for key, env_name in mapping.items():
        value = info.get(key)
        if value not in (None, ""):
            env[env_name] = str(value)
    if not env.get("PGDATABASE"):
        raise ValueError("PostgreSQL DSN must identify a database")
    return env


def _run_pg(command: list[str], *, dsn: str, timeout_seconds: int) -> tuple[int, str]:
    env = dict(os.environ)
    env.update(_conninfo_to_pg_env(dsn))
    env["PGAPPNAME"] = "hprl-r3-real-environment-acceptance"
    completed = subprocess.run(
        command,
        cwd=tempfile.gettempdir(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    # Output can include database/server names but never echo DSN/password from argv.
    return completed.returncode, completed.stdout[-12000:]


@dataclass(frozen=True, slots=True)
class PostgresLogicalBackupReport:
    passed: bool
    archive_path: str
    archive_sha256: str
    archive_bytes: int
    source_snapshot: PostgresRestoreSnapshot
    cli: PostgresCliCapability
    pg_dump_output_tail: str
    reasons: tuple[str, ...]
    source_snapshot_after: PostgresRestoreSnapshot | None = None
    source_stable_during_backup: bool = False
    archive_list_verified: bool = False
    pg_restore_list_output_tail: str = ""


def create_postgres_logical_backup(
    *,
    source_dsn: str,
    source_connection_factory: Callable[[], Any],
    archive_path: str | Path,
    now: datetime,
    timeout_seconds: int = 1800,
) -> PostgresLogicalBackupReport:
    cli = probe_postgres_cli()
    output = Path(archive_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    source_snapshot = capture_postgres_restore_snapshot(source_connection_factory, now=_aware(now))
    source_snapshot_after: PostgresRestoreSnapshot | None = None
    tail = ""
    list_tail = ""
    list_verified = False
    if output.exists():
        # Never silently overwrite a prior acceptance artifact: its SHA may already be
        # referenced by an evidence registry.
        reasons.append("POSTGRES_BACKUP_ARCHIVE_PREEXISTS")
    if not cli.passed:
        reasons.append("POSTGRES_CLIENT_TOOLS_UNAVAILABLE")
    if not reasons:
        command = [
            cli.pg_dump_path,
            "--format=custom",
            "--no-owner",
            "--no-privileges",
            "--file",
            str(output),
        ]
        rc, tail = _run_pg(command, dsn=source_dsn, timeout_seconds=timeout_seconds)
        if rc != 0:
            reasons.append(f"PG_DUMP_EXIT_NONZERO:{rc}")
        if rc == 0 and output.is_file() and output.stat().st_size > 0:
            listed = subprocess.run(
                [cli.pg_restore_path, "--list", str(output)],
                cwd=tempfile.gettempdir(), text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=min(timeout_seconds, 120), check=False,
            )
            list_tail = listed.stdout[-12000:]
            list_verified = listed.returncode == 0 and bool(listed.stdout.strip())
            if not list_verified:
                reasons.append(f"PG_RESTORE_LIST_FAILED:{listed.returncode}")
    # A dump can be transactionally consistent while concurrent writes continue, but an
    # exact restore-vs-source acceptance needs a quiescent source snapshot.  R3 therefore
    # proves that the acceptance tables did not change across the dump window.
    try:
        source_snapshot_after = capture_postgres_restore_snapshot(
            source_connection_factory, now=datetime.now(UTC)
        )
    except Exception as exc:
        reasons.append(f"POSTGRES_POST_DUMP_SNAPSHOT_FAILED:{type(exc).__name__}")
    stable = bool(
        source_snapshot_after is not None
        and source_snapshot.snapshot_sha256 == source_snapshot_after.snapshot_sha256
        and source_snapshot.database_name == source_snapshot_after.database_name
    )
    if not stable:
        reasons.append("POSTGRES_SOURCE_CHANGED_DURING_BACKUP")
    archive_sha = _sha256_path(output) if output.is_file() and output.stat().st_size else ""
    size = output.stat().st_size if output.is_file() else 0
    if not archive_sha:
        reasons.append("POSTGRES_BACKUP_ARCHIVE_MISSING_OR_EMPTY")
    return PostgresLogicalBackupReport(
        passed=not reasons,
        archive_path=str(output),
        archive_sha256=archive_sha,
        archive_bytes=size,
        source_snapshot=source_snapshot,
        cli=cli,
        pg_dump_output_tail=tail,
        reasons=tuple(dict.fromkeys(reasons)),
        source_snapshot_after=source_snapshot_after,
        source_stable_during_backup=stable,
        archive_list_verified=list_verified,
        pg_restore_list_output_tail=list_tail,
    )


def _count_user_tables(connection: Any) -> int:
    value = _scalar(
        connection,
        "SELECT count(*) FROM information_schema.tables "
        "WHERE table_type='BASE TABLE' AND table_schema NOT IN ('pg_catalog','information_schema')",
    )
    return int(value or 0)


@dataclass(frozen=True, slots=True)
class PostgresLogicalRestoreReport:
    passed: bool
    source_database: str
    target_database: str
    isolated_target: bool
    target_empty_before_restore: bool
    archive_sha256: str
    archive_sha256_matches: bool
    restore_exit_code: int
    restored_snapshot: PostgresRestoreSnapshot | None
    verification_passed: bool
    pg_restore_output_tail: str
    reasons: tuple[str, ...]


def restore_postgres_logical_backup(
    *,
    backup: PostgresLogicalBackupReport,
    target_dsn: str,
    target_connection_factory: Callable[[], Any],
    now: datetime,
    timeout_seconds: int = 1800,
) -> PostgresLogicalRestoreReport:
    reasons: list[str] = []
    target = target_connection_factory()
    try:
        target_database = str(_scalar(target, "SELECT current_database()") or "")
        target_empty = _count_user_tables(target) == 0
    finally:
        _close(target)
    source_database = backup.source_snapshot.database_name
    isolated = bool(source_database and target_database and source_database != target_database)
    if not backup.passed:
        reasons.append("POSTGRES_SOURCE_BACKUP_NOT_PASSED")
    if not isolated:
        reasons.append("RESTORE_TARGET_NOT_ISOLATED")
    if not target_empty:
        reasons.append("RESTORE_TARGET_NOT_EMPTY")
    archive = Path(backup.archive_path)
    current_sha = _sha256_path(archive) if archive.is_file() else ""
    sha_match = bool(current_sha and current_sha == backup.archive_sha256)
    if not sha_match:
        reasons.append("POSTGRES_BACKUP_ARCHIVE_SHA_MISMATCH")
    restore_rc = -1
    tail = ""
    restored_snapshot: PostgresRestoreSnapshot | None = None
    verification_passed = False
    if not reasons:
        cli = probe_postgres_cli()
        if not cli.passed:
            reasons.append("POSTGRES_CLIENT_TOOLS_UNAVAILABLE")
        else:
            command = [
                cli.pg_restore_path,
                "--exit-on-error",
                "--no-owner",
                "--no-privileges",
                "--dbname",
                os.environ.get("PGDATABASE", ""),  # replaced below: pg_restore accepts env PGDATABASE when empty
                str(archive),
            ]
            # An empty dbname argument is rejected.  Do not place the DSN in argv; use
            # a harmless explicit database name from parsed conninfo instead.
            target_env = _conninfo_to_pg_env(target_dsn)
            command[-2] = target_env["PGDATABASE"]
            env = dict(os.environ)
            env.update(target_env)
            env["PGAPPNAME"] = "hprl-r3-restore-acceptance"
            completed = subprocess.run(
                command, cwd=tempfile.gettempdir(), env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                timeout=timeout_seconds, check=False,
            )
            restore_rc = completed.returncode
            tail = completed.stdout[-12000:]
            if restore_rc != 0:
                reasons.append(f"PG_RESTORE_EXIT_NONZERO:{restore_rc}")
    if restore_rc == 0 and not reasons:
        restored_snapshot = capture_postgres_restore_snapshot(target_connection_factory, now=_aware(now))
        verification = verify_postgres_restore(backup.source_snapshot, restored_snapshot)
        verification_passed = verification.passed
        reasons.extend(verification.reasons)
    return PostgresLogicalRestoreReport(
        passed=not reasons,
        source_database=source_database,
        target_database=target_database,
        isolated_target=isolated,
        target_empty_before_restore=target_empty,
        archive_sha256=current_sha,
        archive_sha256_matches=sha_match,
        restore_exit_code=restore_rc,
        restored_snapshot=restored_snapshot,
        verification_passed=verification_passed,
        pg_restore_output_tail=tail,
        reasons=tuple(dict.fromkeys(reasons)),
    )


@dataclass(frozen=True, slots=True)
class PostgresR3FailoverToken:
    probe_id: str
    payload_sha256: str
    schema: str
    table: str
    primary: PostgresNodeIdentity
    writer_lock_key: int
    prepared_at: datetime


def prepare_postgres_r3_failover_token(
    connection_factory: Callable[[], Any],
    *,
    now: datetime,
    schema: str = "freqtrade_hedge_probe",
    table: str = "hprl_r3_failover_sentinel",
) -> PostgresR3FailoverToken:
    from uuid import uuid4

    observed = _aware(now)
    probe_id = "hprl-r3-failover-" + uuid4().hex
    payload = sha256(f"{probe_id}|{observed.isoformat()}".encode()).hexdigest()
    connection = connection_factory()
    try:
        identity = capture_postgres_node_identity(connection, now=observed)
        if not identity.writable_primary:
            raise RuntimeError("failover preparation requires a writable PostgreSQL primary")
        _execute(connection, f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        _execute(
            connection,
            f'CREATE TABLE IF NOT EXISTS "{schema}"."{table}" ('
            "probe_id text PRIMARY KEY, payload_sha256 text NOT NULL, prepared_at timestamptz NOT NULL, verified_at timestamptz NULL)"
        )
        connection.commit()
        _execute(
            connection,
            f'INSERT INTO "{schema}"."{table}" (probe_id,payload_sha256,prepared_at) VALUES (%s,%s,%s) '
            "ON CONFLICT (probe_id) DO UPDATE SET payload_sha256=EXCLUDED.payload_sha256, prepared_at=EXCLUDED.prepared_at",
            (probe_id, payload, observed),
        )
        connection.commit()
        key = PostgresProbeRunner(connection).lock_key
        return PostgresR3FailoverToken(probe_id, payload, schema, table, identity, key, observed)
    finally:
        _close(connection)


@dataclass(frozen=True, slots=True)
class PostgresR3FailoverReport:
    passed: bool
    sentinel_visible: bool
    routed_identity: PostgresNodeIdentity | None
    routed_endpoint_changed: bool
    cluster_identity_preserved: bool
    routed_writable_primary: bool
    writer_fence_acquired: bool
    old_primary_identity: PostgresNodeIdentity | None
    old_primary_matches_pre_failover_node: bool
    old_primary_fenced: bool
    cleanup_ok: bool
    reasons: tuple[str, ...]


def verify_postgres_r3_failover_token(
    token: PostgresR3FailoverToken,
    routed_factory: Callable[[], Any],
    *,
    old_primary_factory: Callable[[], Any],
    now: datetime,
) -> PostgresR3FailoverReport:
    reasons: list[str] = []
    routed = old = None
    routed_identity = old_identity = None
    sentinel = changed = cluster_ok = writable = fence = old_matches = old_fenced = cleanup = False
    try:
        routed = routed_factory()
        routed_identity = capture_postgres_node_identity(routed, now=_aware(now))
        found = _scalar(
            routed,
            f'SELECT payload_sha256 FROM "{token.schema}"."{token.table}" WHERE probe_id=%s',
            (token.probe_id,),
        )
        sentinel = found == token.payload_sha256
        if token.primary.server_addr and routed_identity.server_addr:
            changed = routed_identity.endpoint != token.primary.endpoint
        else:
            reasons.append("FAILOVER_NODE_ENDPOINT_UNAVAILABLE")
        cluster_ok = True
        if token.primary.system_identifier and routed_identity.system_identifier:
            cluster_ok = token.primary.system_identifier == routed_identity.system_identifier
        writable = routed_identity.writable_primary
        # Keep the routed writer fence held while the direct old node is checked.  A
        # split-brain old primary lives in a different lock domain and can acquire the
        # same application key; a correctly fenced old node cannot write/acquire it.
        fence = bool(_scalar(routed, "SELECT pg_try_advisory_lock(%s)", (token.writer_lock_key,)))
        if not fence:
            reasons.append("FAILOVER_WRITER_FENCE_NOT_ACQUIRED")

        try:
            old = old_primary_factory()
            old_identity = capture_postgres_node_identity(old, now=_aware(now))
            if token.primary.server_addr and old_identity.server_addr:
                old_matches = old_identity.endpoint == token.primary.endpoint
            else:
                old_matches = True
            old_fenced = old_identity.in_recovery or old_identity.transaction_read_only
            if not old_fenced:
                old_lock = bool(_scalar(old, "SELECT pg_try_advisory_lock(%s)", (token.writer_lock_key,)))
                if old_lock:
                    _scalar(old, "SELECT pg_advisory_unlock(%s)", (token.writer_lock_key,))
                old_fenced = not old_lock
        except Exception:
            # An unreachable direct old-primary endpoint is an acceptable fencing state.
            old_fenced = True
            old_matches = True
        finally:
            _close(old)
            old = None

        _execute(
            routed,
            f'UPDATE "{token.schema}"."{token.table}" SET verified_at=%s WHERE probe_id=%s',
            (_aware(now), token.probe_id),
        )
        routed.commit()
        _execute(
            routed,
            f'DELETE FROM "{token.schema}"."{token.table}" WHERE probe_id=%s',
            (token.probe_id,),
        )
        routed.commit()
        cleanup = True
    except Exception as exc:
        reasons.append(f"FAILOVER_VERIFY:{type(exc).__name__}:{exc}")
        if routed is not None:
            try:
                routed.rollback()
            except Exception:
                pass
    finally:
        if routed is not None and fence:
            try:
                _scalar(routed, "SELECT pg_advisory_unlock(%s)", (token.writer_lock_key,))
            except Exception:
                pass
        _close(routed)

    checks = (
        (sentinel, "FAILOVER_SENTINEL_NOT_VISIBLE"),
        (changed, "FAILOVER_ROUTED_NODE_DID_NOT_CHANGE"),
        (cluster_ok, "FAILOVER_CLUSTER_IDENTITY_CHANGED"),
        (writable, "FAILOVER_ROUTED_TARGET_NOT_WRITABLE_PRIMARY"),
        (fence, "FAILOVER_WRITER_FENCE_NOT_ACQUIRED"),
        (old_matches, "OLD_PRIMARY_DSN_NOT_BOUND_TO_PRE_FAILOVER_NODE"),
        (old_fenced, "FAILOVER_OLD_PRIMARY_NOT_FENCED"),
        (cleanup, "FAILOVER_SENTINEL_CLEANUP_FAILED"),
    )
    for ok, reason in checks:
        if not ok and reason not in reasons:
            reasons.append(reason)
    return PostgresR3FailoverReport(
        passed=not reasons,
        sentinel_visible=sentinel,
        routed_identity=routed_identity,
        routed_endpoint_changed=changed,
        cluster_identity_preserved=cluster_ok,
        routed_writable_primary=writable,
        writer_fence_acquired=fence,
        old_primary_identity=old_identity,
        old_primary_matches_pre_failover_node=old_matches,
        old_primary_fenced=old_fenced,
        cleanup_ok=cleanup,
        reasons=tuple(dict.fromkeys(reasons)),
    )
