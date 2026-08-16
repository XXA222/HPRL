"""Real PostgreSQL runtime-closure probes for HPRL V3.

The probes are intentionally bounded and use dedicated acceptance namespaces.  They
exercise the *actual* SQL execution store, closed-loop journal and recovery checkpoint
surfaces.  Failover and backup/restore are split into prepare/verify phases so the code
never silently restarts, promotes, drops, or rewrites a production database.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from typing import Any, Callable, Iterable
from uuid import uuid4

from .database_runtime import (
    PostgresConcurrencyProbeReport,
    PostgresConcurrencyProbeRunner,
    PostgresProbeReport,
    PostgresProbeRunner,
)
from .postgres_acceptance import (
    PostgresDurabilityProbeReport,
    PostgresDurabilityProbeRunner,
)

ZERO_HASH = "0" * 64


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _sha_payload(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(raw).hexdigest()


def _close(connection: object | None) -> None:
    if connection is None:
        return
    close = getattr(connection, "close", None)
    if callable(close):
        close()


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


@dataclass(frozen=True, slots=True)
class PostgresTableSurfaceReport:
    required_tables: tuple[str, ...]
    present_tables: tuple[str, ...]
    missing_tables: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.missing_tables


@dataclass(frozen=True, slots=True)
class PostgresExecutionLedgerProbeReport:
    inserted_unknown: bool
    second_session_visible: bool
    unresolved_unknown_visible: bool
    lifecycle_update_visible: bool
    unresolved_unknown_cleared: bool
    cleanup_ok: bool
    client_order_id: str
    account_id: str
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all((
            self.inserted_unknown,
            self.second_session_visible,
            self.unresolved_unknown_visible,
            self.lifecycle_update_visible,
            self.unresolved_unknown_cleared,
            self.cleanup_ok,
        )) and not self.errors


@dataclass(frozen=True, slots=True)
class PostgresClosedLoopJournalProbeReport:
    first_append_visible: bool
    lost_response_retry_idempotent: bool
    second_append_visible: bool
    stale_tip_rejected: bool
    chain_valid: bool
    cleanup_ok: bool
    account_id: str
    first_record_sha256: str
    second_record_sha256: str
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all((
            self.first_append_visible,
            self.lost_response_retry_idempotent,
            self.second_append_visible,
            self.stale_tip_rejected,
            self.chain_valid,
            self.cleanup_ok,
        )) and not self.errors


@dataclass(frozen=True, slots=True)
class PostgresCheckpointProbeReport:
    generation_one_visible: bool
    retry_idempotent: bool
    generation_two_visible: bool
    stale_generation_rejected: bool
    cleanup_ok: bool
    account_id: str
    checkpoint_one_sha256: str
    checkpoint_two_sha256: str
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all((
            self.generation_one_visible,
            self.retry_idempotent,
            self.generation_two_visible,
            self.stale_generation_rejected,
            self.cleanup_ok,
        )) and not self.errors


@dataclass(frozen=True, slots=True)
class PostgresRuntimeClosureReport:
    basic: PostgresProbeReport
    concurrency: PostgresConcurrencyProbeReport
    durability: PostgresDurabilityProbeReport
    tables: PostgresTableSurfaceReport
    execution_ledger: PostgresExecutionLedgerProbeReport
    closed_loop_journal: PostgresClosedLoopJournalProbeReport
    recovery_checkpoint: PostgresCheckpointProbeReport
    observed_at: datetime

    @property
    def passed(self) -> bool:
        return all((
            self.basic.passed,
            self.concurrency.passed,
            self.durability.passed,
            self.tables.passed,
            self.execution_ledger.passed,
            self.closed_loop_journal.passed,
            self.recovery_checkpoint.passed,
        ))


class PostgresRuntimeClosureRunner:
    REQUIRED_TABLES = (
        "hedge_execution_order_states",
        "hedge_audit_events",
        "hedge_schema_migrations",
    )

    def __init__(
        self,
        *,
        connection_factory: Callable[[], Any],
        session_factory: object,
        engine: object,
        symbol: str = "BTCUSDT",
        source_release: str = "freqtrade-hedge-hprl-v3-runtime-closure-r2",
    ) -> None:
        if not callable(connection_factory) or not callable(session_factory):
            raise TypeError("PostgreSQL connection/session factories must be callable")
        self.connection_factory = connection_factory
        self.session_factory = session_factory
        self.engine = engine
        self.symbol = symbol
        self.source_release = source_release

    def _table_surface(self) -> PostgresTableSurfaceReport:
        from sqlalchemy import inspect

        inspector = inspect(self.engine)
        present = tuple(sorted(name for name in self.REQUIRED_TABLES if inspector.has_table(name)))
        missing = tuple(name for name in self.REQUIRED_TABLES if name not in present)
        return PostgresTableSurfaceReport(self.REQUIRED_TABLES, present, missing)

    def _cleanup_account(self, account_id: str) -> bool:
        try:
            from sqlalchemy import delete
            from freqtrade.persistence.hedge_models import AuditEvent, ExecutionOrderStateRow

            with self.session_factory.begin() as session:  # type: ignore[operator]
                session.execute(delete(AuditEvent).where(AuditEvent.account_id == account_id))
                session.execute(
                    delete(ExecutionOrderStateRow).where(ExecutionOrderStateRow.account_id == account_id)
                )
            return True
        except Exception:
            return False

    def _execution_ledger_probe(self, *, now: datetime) -> PostgresExecutionLedgerProbeReport:
        account = "hprl-acceptance-" + uuid4().hex[:20]
        client_id = "hprlacc-" + uuid4().hex[:20]
        errors: list[str] = []
        inserted = visible = unresolved = updated = cleared = False
        try:
            from freqtrade.hedge.execution.service import (
                ExecutionOrder,
                IntentAction,
                OrderIntent,
                OrderType,
                PositionSide,
            )
            from freqtrade.hedge.execution.state_machine import OrderLifecycle, OrderState
            from freqtrade.persistence.hedge_execution_adapters import SqlExecutionStore

            quantity = Decimal("0.001")
            intent = OrderIntent(
                account_id=account,
                symbol=self.symbol,
                position_side=PositionSide.LONG,
                action=IntentAction.OPEN,
                quantity=quantity,
                idempotency_key="hprlacc:" + uuid4().hex,
                order_type=OrderType.MARKET,
                metadata={"exchange": "binance", "acceptance_probe": True},
            )
            unknown_lifecycle = OrderLifecycle(
                status=OrderState.UNKNOWN,
                filled_quantity=Decimal("0"),
                version=1,
                updated_at=now,
                reason="RUNTIME_ACCEPTANCE_UNKNOWN",
            )
            order = ExecutionOrder(intent, client_id, quantity, unknown_lifecycle, now)
            first = SqlExecutionStore(self.session_factory)
            second = SqlExecutionStore(self.session_factory)
            first.put(order)
            inserted = True
            observed = second.get_by_client_order_id(client_id)
            visible = observed is not None and observed.lifecycle.status is OrderState.UNKNOWN
            unresolved = second.has_unresolved_unknown(order.leg_key)
            acknowledged = ExecutionOrder(
                intent,
                client_id,
                quantity,
                unknown_lifecycle.transition(
                    OrderState.ACKNOWLEDGED,
                    ordered_quantity=quantity,
                    occurred_at=now + timedelta(microseconds=1),
                    reason="RUNTIME_ACCEPTANCE_RESOLVED",
                ),
                now,
            )
            second.put(acknowledged)
            observed2 = first.get_by_client_order_id(client_id)
            updated = observed2 is not None and observed2.lifecycle.status is OrderState.ACKNOWLEDGED
            cleared = not first.has_unresolved_unknown(order.leg_key)
        except Exception as exc:
            errors.append(f"EXECUTION_LEDGER:{type(exc).__name__}:{exc}")
        cleanup = self._cleanup_account(account)
        if not cleanup:
            errors.append("EXECUTION_LEDGER_CLEANUP_FAILED")
        return PostgresExecutionLedgerProbeReport(
            inserted, visible, unresolved, updated, cleared, cleanup,
            client_id, account, tuple(errors),
        )

    @staticmethod
    def _record(*, sequence: int, previous: str, now: datetime, account: str):
        from .closed_loop import ClosedLoopCycleRecord, ClosedLoopCycleStatus

        digest = _sha_payload({"account": account, "sequence": sequence})
        return ClosedLoopCycleRecord(
            sequence=sequence,
            cycle_id=f"hprl-sql-probe-{account}-{sequence}",
            observed_at=now + timedelta(microseconds=sequence),
            source_release="freqtrade-hedge-hprl-v3-runtime-closure-r2",
            model_id="runtime-acceptance-probe",
            symbol="BTCUSDT",
            projection_sequence=sequence,
            projection_observed_at=now + timedelta(microseconds=sequence),
            projection_source_sha256=digest,
            projection_semantic_sha256=digest,
            long_margin_ratio=Decimal("0.05"),
            short_margin_ratio=Decimal("0.05"),
            long_notional_ratio=Decimal("0.05"),
            short_notional_ratio=Decimal("0.05"),
            confidence=Decimal("1"),
            projection_accepted=True,
            projection_reasons=(),
            projection_chain_sha256=digest,
            planner_profile_sha256=digest,
            input_state_sha256=digest,
            planning_sha256=digest,
            execution_sha256=digest,
            reconciliation_digest=digest,
            evidence_digest=digest,
            safety_allows_reduce=True,
            safety_allows_new_risk=False,
            status=ClosedLoopCycleStatus.COMMITTED,
            writes_attempted=0,
            unresolved_client_order_ids=(),
            previous_record_sha256=previous,
        )

    def _journal_probe(self, *, now: datetime) -> PostgresClosedLoopJournalProbeReport:
        from .closed_loop import ClosedLoopJournalConcurrencyError, ZERO_HASH
        from .closed_loop_sql import SqlClosedLoopCycleJournalStore

        account = "hprl-journal-" + uuid4().hex[:20]
        errors: list[str] = []
        first_visible = retry_ok = second_visible = stale_rejected = chain_valid = False
        r1 = r2 = None
        try:
            first = SqlClosedLoopCycleJournalStore(self.session_factory, account_id=account)
            second = SqlClosedLoopCycleJournalStore(self.session_factory, account_id=account)
            r1 = self._record(sequence=1, previous=ZERO_HASH, now=now, account=account)
            j1 = first.append_atomic(r1, expected_previous_sha256=ZERO_HASH)
            first_visible = second.load().tip_sha256 == r1.record_sha256
            retried = second.append_atomic(r1, expected_previous_sha256=ZERO_HASH)
            retry_ok = len(retried.records) == 1 and retried.tip_sha256 == r1.record_sha256
            r2 = self._record(sequence=2, previous=j1.tip_sha256, now=now, account=account)
            j2 = second.append_atomic(r2, expected_previous_sha256=j1.tip_sha256)
            loaded = first.load()
            second_visible = loaded.tip_sha256 == r2.record_sha256 and len(loaded.records) == 2
            chain_valid = loaded.verify()
            try:
                first.append_atomic(r2, expected_previous_sha256=ZERO_HASH)
            except ClosedLoopJournalConcurrencyError:
                stale_rejected = True
        except Exception as exc:
            errors.append(f"CLOSED_LOOP_JOURNAL:{type(exc).__name__}:{exc}")
        cleanup = self._cleanup_account(account)
        if not cleanup:
            errors.append("CLOSED_LOOP_JOURNAL_CLEANUP_FAILED")
        return PostgresClosedLoopJournalProbeReport(
            first_visible, retry_ok, second_visible, stale_rejected, chain_valid, cleanup,
            account,
            ZERO_HASH if r1 is None else r1.record_sha256,
            ZERO_HASH if r2 is None else r2.record_sha256,
            tuple(errors),
        )

    def _checkpoint_probe(self, *, now: datetime) -> PostgresCheckpointProbeReport:
        from .closed_loop_sql import SqlRecoveryCheckpointStore
        from .recovery_checkpoint import DurableRecoveryCheckpoint

        account = "hprl-checkpoint-" + uuid4().hex[:20]
        digest = _sha_payload({"account": account})
        errors: list[str] = []
        one_visible = retry_ok = two_visible = stale_rejected = False
        c1 = c2 = None
        try:
            first = SqlRecoveryCheckpointStore(self.session_factory, account_id=account)
            second = SqlRecoveryCheckpointStore(self.session_factory, account_id=account)
            c1 = DurableRecoveryCheckpoint(
                generation=1,
                created_at=now,
                source_release=self.source_release,
                model_id="runtime-acceptance-probe",
                evidence_digest=digest,
                reconciliation_digest=digest,
                projection_chain_sha256=digest,
                last_market_sequence=1,
                last_user_sequence=1,
                unresolved_client_order_ids=(),
                metadata=(("probe", "postgres-runtime-closure"),),
            )
            first.save_atomic(c1)
            one_visible = second.load() == c1
            retry_ok = second.save_atomic(c1) == c1.checkpoint_sha256
            c2 = DurableRecoveryCheckpoint(
                generation=2,
                created_at=now + timedelta(microseconds=1),
                source_release=self.source_release,
                model_id="runtime-acceptance-probe",
                evidence_digest=digest,
                reconciliation_digest=digest,
                projection_chain_sha256=digest,
                last_market_sequence=2,
                last_user_sequence=2,
                unresolved_client_order_ids=(),
                metadata=(("probe", "postgres-runtime-closure"),),
            )
            second.save_atomic(c2)
            two_visible = first.load() == c2
            stale = DurableRecoveryCheckpoint(
                generation=1,
                created_at=now + timedelta(microseconds=2),
                source_release=self.source_release,
                model_id="different-probe",
                evidence_digest=digest,
                reconciliation_digest=digest,
                projection_chain_sha256=digest,
                last_market_sequence=3,
                last_user_sequence=3,
                unresolved_client_order_ids=(),
            )
            try:
                first.save_atomic(stale)
            except ValueError:
                stale_rejected = True
        except Exception as exc:
            errors.append(f"RECOVERY_CHECKPOINT:{type(exc).__name__}:{exc}")
        cleanup = self._cleanup_account(account)
        if not cleanup:
            errors.append("RECOVERY_CHECKPOINT_CLEANUP_FAILED")
        return PostgresCheckpointProbeReport(
            one_visible, retry_ok, two_visible, stale_rejected, cleanup, account,
            ZERO_HASH if c1 is None else c1.checkpoint_sha256,
            ZERO_HASH if c2 is None else c2.checkpoint_sha256,
            tuple(errors),
        )

    def run(self, *, now: datetime) -> PostgresRuntimeClosureReport:
        observed = _aware(now)
        primary = self.connection_factory()
        try:
            basic = PostgresProbeRunner(primary).run(now=observed)
        finally:
            _close(primary)
        concurrency = PostgresConcurrencyProbeRunner(self.connection_factory).run(now=observed)
        durability = PostgresDurabilityProbeRunner(
            self.connection_factory, self.connection_factory
        ).run(now=observed)
        tables = self._table_surface()
        execution = self._execution_ledger_probe(now=observed)
        journal = self._journal_probe(now=observed)
        checkpoint = self._checkpoint_probe(now=observed)
        return PostgresRuntimeClosureReport(
            basic, concurrency, durability, tables, execution, journal, checkpoint, observed
        )


@dataclass(frozen=True, slots=True)
class PostgresSnapshotTable:
    table: str
    rows: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PostgresRestoreSnapshot:
    database_name: str
    server_version: str
    tables: tuple[PostgresSnapshotTable, ...]
    snapshot_sha256: str
    observed_at: datetime


_SNAPSHOT_TABLES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "hedge_execution_order_states",
        "id",
        ("id", "client_order_id", "intent_id", "account_id", "exchange", "symbol", "position_side", "action", "quantity", "lifecycle_status", "lifecycle_filled_quantity", "lifecycle_version"),
    ),
    (
        "hedge_audit_events",
        "id",
        ("id", "audit_id", "account_id", "exchange", "event_type", "entity_type", "entity_id", "correlation_id", "payload_json", "occurred_at"),
    ),
    (
        "hedge_schema_migrations",
        "migration_id",
        ("migration_id", "checksum", "state", "attempt_count", "applied_at", "record_version"),
    ),
)


def capture_postgres_restore_snapshot(connection_factory: Callable[[], Any], *, now: datetime) -> PostgresRestoreSnapshot:
    connection = connection_factory()
    try:
        database_name = str(_scalar(connection, "SELECT current_database()") or "")
        server_version = str(_scalar(connection, "SHOW server_version") or "")
        table_rows: list[PostgresSnapshotTable] = []
        for table, order_by, columns in _SNAPSHOT_TABLES:
            digest = sha256()
            count = 0
            cursor = connection.cursor()
            try:
                cursor.execute(
                    f'SELECT {",".join(columns)} FROM "{table}" ORDER BY "{order_by}"'
                )
                while True:
                    rows = cursor.fetchmany(1000)
                    if not rows:
                        break
                    for row in rows:
                        payload = json.dumps(row, default=str, separators=(",", ":"), ensure_ascii=True)
                        digest.update(payload.encode("utf-8") + b"\n")
                        count += 1
            finally:
                close = getattr(cursor, "close", None)
                if callable(close):
                    close()
            table_rows.append(PostgresSnapshotTable(table, count, digest.hexdigest()))
        snapshot_hash = _sha_payload([asdict(item) for item in table_rows])
        return PostgresRestoreSnapshot(
            database_name=database_name,
            server_version=server_version,
            tables=tuple(table_rows),
            snapshot_sha256=snapshot_hash,
            observed_at=_aware(now),
        )
    finally:
        _close(connection)


@dataclass(frozen=True, slots=True)
class PostgresRestoreVerificationReport:
    source: PostgresRestoreSnapshot
    restored: PostgresRestoreSnapshot
    isolated_database: bool
    table_counts_match: bool
    table_hashes_match: bool
    snapshot_match: bool
    passed: bool
    reasons: tuple[str, ...]


def verify_postgres_restore(
    source: PostgresRestoreSnapshot,
    restored: PostgresRestoreSnapshot,
) -> PostgresRestoreVerificationReport:
    reasons: list[str] = []
    isolated = bool(source.database_name and restored.database_name and source.database_name != restored.database_name)
    if not isolated:
        reasons.append("RESTORE_TARGET_NOT_ISOLATED")
    source_by = {item.table: item for item in source.tables}
    restored_by = {item.table: item for item in restored.tables}
    counts = source_by.keys() == restored_by.keys() and all(
        source_by[name].rows == restored_by[name].rows for name in source_by
    )
    hashes = source_by.keys() == restored_by.keys() and all(
        source_by[name].sha256 == restored_by[name].sha256 for name in source_by
    )
    snapshot = source.snapshot_sha256 == restored.snapshot_sha256
    if not counts:
        reasons.append("RESTORE_TABLE_COUNTS_MISMATCH")
    if not hashes:
        reasons.append("RESTORE_TABLE_HASHES_MISMATCH")
    if not snapshot:
        reasons.append("RESTORE_SNAPSHOT_DIGEST_MISMATCH")
    return PostgresRestoreVerificationReport(
        source, restored, isolated, counts, hashes, snapshot, not reasons, tuple(reasons)
    )


@dataclass(frozen=True, slots=True)
class PostgresFailoverToken:
    probe_id: str
    payload_sha256: str
    schema: str
    table: str
    database_name: str
    primary_backend_pid: int
    prepared_at: datetime


def prepare_postgres_failover_token(
    connection_factory: Callable[[], Any],
    *,
    now: datetime,
    schema: str = "freqtrade_hedge_probe",
    table: str = "hprl_failover_sentinel",
) -> PostgresFailoverToken:
    observed = _aware(now)
    probe_id = "hprl-failover-" + uuid4().hex
    payload = sha256(f"{probe_id}|{observed.isoformat()}".encode()).hexdigest()
    connection = connection_factory()
    try:
        pid = int(_scalar(connection, "SELECT pg_backend_pid()"))
        database = str(_scalar(connection, "SELECT current_database()") or "")
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
        return PostgresFailoverToken(probe_id, payload, schema, table, database, pid, observed)
    finally:
        _close(connection)


@dataclass(frozen=True, slots=True)
class PostgresFailoverVerificationReport:
    sentinel_visible: bool
    routed_backend_changed: bool
    route_writable: bool
    writer_fence_acquired: bool
    old_primary_fenced: bool
    cleanup_ok: bool
    routed_backend_pid: int | None
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.reasons


def verify_postgres_failover_token(
    token: PostgresFailoverToken,
    routed_factory: Callable[[], Any],
    *,
    now: datetime,
    old_primary_factory: Callable[[], Any] | None = None,
) -> PostgresFailoverVerificationReport:
    from .database_runtime import PostgresProbeRunner

    reasons: list[str] = []
    routed = None
    routed_pid = None
    sentinel = changed = writable = fence = old_fenced = cleanup = False
    try:
        routed = routed_factory()
        routed_pid = int(_scalar(routed, "SELECT pg_backend_pid()"))
        found = _scalar(
            routed,
            f'SELECT payload_sha256 FROM "{token.schema}"."{token.table}" WHERE probe_id=%s',
            (token.probe_id,),
        )
        sentinel = found == token.payload_sha256
        changed = routed_pid != token.primary_backend_pid
        read_only = str(_scalar(routed, "SHOW transaction_read_only") or "").lower()
        writable = read_only in {"off", "false", "0"}
        key = PostgresProbeRunner(routed).lock_key
        # One acquisition / one release keeps the failover evidence unambiguous.
        # PostgreSQL advisory locks are session-scoped and re-entrant; acquiring the
        # same key twice would otherwise make a single successful unlock insufficient.
        fence = bool(_scalar(routed, "SELECT pg_try_advisory_lock(%s)", (key,)))
        if fence:
            _scalar(routed, "SELECT pg_advisory_unlock(%s)", (key,))
        _execute(
            routed,
            f'UPDATE "{token.schema}"."{token.table}" SET verified_at=%s WHERE probe_id=%s',
            (_aware(now), token.probe_id),
        )
        routed.commit()
        if old_primary_factory is None:
            old_fenced = False
            reasons.append("OLD_PRIMARY_FENCE_NOT_VERIFIED")
        else:
            old = None
            try:
                old = old_primary_factory()
                old_read_only = str(_scalar(old, "SHOW transaction_read_only") or "").lower()
                old_fenced = old_read_only in {"on", "true", "1"}
                if not old_fenced:
                    old_key = PostgresProbeRunner(old).lock_key
                    old_acquired = bool(_scalar(old, "SELECT pg_try_advisory_lock(%s)", (old_key,)))
                    if old_acquired:
                        _scalar(old, "SELECT pg_advisory_unlock(%s)", (old_key,))
                    old_fenced = not old_acquired
            except Exception:
                old_fenced = True
            finally:
                _close(old)
        _execute(
            routed,
            f'DELETE FROM "{token.schema}"."{token.table}" WHERE probe_id=%s',
            (token.probe_id,),
        )
        routed.commit()
        cleanup = True
    except Exception as exc:
        reasons.append(f"FAILOVER_VERIFY:{type(exc).__name__}:{exc}")
        try:
            if routed is not None:
                routed.rollback()
        except Exception:
            pass
    finally:
        _close(routed)
    for ok, reason in (
        (sentinel, "FAILOVER_SENTINEL_NOT_VISIBLE"),
        (changed, "FAILOVER_BACKEND_DID_NOT_CHANGE"),
        (writable, "FAILOVER_ROUTED_TARGET_NOT_WRITABLE"),
        (fence, "FAILOVER_WRITER_FENCE_NOT_ACQUIRED"),
        (old_fenced, "FAILOVER_OLD_PRIMARY_NOT_FENCED"),
        (cleanup, "FAILOVER_SENTINEL_CLEANUP_FAILED"),
    ):
        if not ok and reason not in reasons:
            reasons.append(reason)
    return PostgresFailoverVerificationReport(
        sentinel, changed, writable, fence, old_fenced, cleanup, routed_pid, tuple(reasons)
    )
