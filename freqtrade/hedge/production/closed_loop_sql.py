"""SQL-authoritative closed-loop cycle journal using the existing Hedge audit ledger.

No parallel order database is introduced.  Cycle evidence is stored as immutable audit
facts in ``hedge_audit_events``; PostgreSQL row locking plus the existing SingleWriter
fence provide the production serialization boundary.
"""
from __future__ import annotations

import json
from datetime import UTC
from hashlib import sha256

from sqlalchemy import select, text

from .closed_loop import (
    ClosedLoopCycleJournal,
    ClosedLoopCycleRecord,
    ClosedLoopJournalConcurrencyError,
    ZERO_HASH,
    _sha,
)

_ENTITY_TYPE = "HPRL_CLOSED_LOOP_CYCLE"
_EVENT_TYPE = "HPRL_CYCLE_COMMITTED"


def _lock_key(account_id: str, exchange: str) -> int:
    material = f"hprl-closed-loop|{exchange}|{account_id}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "big", signed=True)


def _audit_event_model():
    # Keep source validation and HPRL-only environments dependency-light.
    from freqtrade.persistence.hedge_models import AuditEvent

    return AuditEvent


class SqlClosedLoopCycleJournalStore:
    """Persist the hash chain in the canonical SQL audit table.

    The store implements the same ``load``/``append_atomic`` surface as the file store,
    so dry-run can use fsync JSON while production PostgreSQL uses the authoritative DB.
    """

    def __init__(
        self,
        session_factory: object,
        *,
        account_id: str,
        exchange: str = "binance",
        actor: str = "hprl-production-closed-loop",
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory
        self._account_id = str(account_id).strip()
        self._exchange = str(exchange).strip().lower()
        self._actor = str(actor).strip()
        if not self._account_id or not self._exchange or not self._actor:
            raise ValueError("closed-loop SQL identity is incomplete")

    def _query(self, *, for_update: bool = False):
        AuditEvent = _audit_event_model()
        statement = (
            select(AuditEvent)
            .where(
                AuditEvent.account_id == self._account_id,
                AuditEvent.exchange == self._exchange,
                AuditEvent.entity_type == _ENTITY_TYPE,
                AuditEvent.event_type == _EVENT_TYPE,
            )
            .order_by(AuditEvent.id.asc())
        )
        if for_update:
            statement = statement.with_for_update()
        return statement

    @staticmethod
    def _journal_from_rows(rows: object) -> ClosedLoopCycleJournal:
        records: list[ClosedLoopCycleRecord] = []
        for row in rows:
            payload = json.loads(str(row.payload_json))
            if not isinstance(payload, dict):
                raise ValueError("closed-loop SQL audit payload must be an object")
            records.append(ClosedLoopCycleRecord.from_payload(payload))
        return ClosedLoopCycleJournal(tuple(records))

    def _lock_writer(self, session: object) -> None:
        bind = session.get_bind()  # type: ignore[attr-defined]
        if getattr(getattr(bind, "dialect", None), "name", "") == "postgresql":
            session.execute(  # type: ignore[attr-defined]
                text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _lock_key(self._account_id, self._exchange)},
            )

    def load(self) -> ClosedLoopCycleJournal:
        with self._session_factory() as session:  # type: ignore[operator]
            rows = tuple(session.scalars(self._query()).all())
            return self._journal_from_rows(rows)

    def append_atomic(
        self,
        record: ClosedLoopCycleRecord,
        *,
        expected_previous_sha256: str,
    ) -> ClosedLoopCycleJournal:
        if not isinstance(record, ClosedLoopCycleRecord):
            raise TypeError("record must be ClosedLoopCycleRecord")
        expected = _sha(expected_previous_sha256, field="expected_previous_sha256")
        with self._session_factory.begin() as session:  # type: ignore[operator]
            # Row locks cannot serialize the very first append because no row exists yet.
            # PostgreSQL therefore takes a transaction-scoped advisory fence before reading
            # the chain.  Other SQL dialects retain the existing transaction semantics.
            self._lock_writer(session)
            rows = tuple(session.scalars(self._query(for_update=True)).all())
            journal = self._journal_from_rows(rows)
            # A caller can lose the commit response and retry with the original expected
            # previous tip.  Recognize that exact committed record *before* comparing the
            # now-advanced journal tip.  This makes the documented retry truly idempotent.
            if (
                journal.last is not None
                and journal.last.record_sha256 == record.record_sha256
                and record.previous_record_sha256 == expected
            ):
                return journal
            if journal.tip_sha256 != expected:
                raise ClosedLoopJournalConcurrencyError(
                    "SQL closed-loop journal changed since cycle planning"
                )
            journal.append(record)
            AuditEvent = _audit_event_model()
            session.add(
                AuditEvent(
                    account_id=self._account_id,
                    exchange=self._exchange,
                    event_type=_EVENT_TYPE,
                    entity_type=_ENTITY_TYPE,
                    entity_id=record.record_sha256,
                    severity="INFO",
                    reason_code=record.status.value[:64],
                    correlation_id=record.cycle_id[:128],
                    actor=self._actor,
                    payload_json=json.dumps(
                        record.payload(),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                    occurred_at=record.observed_at.astimezone(UTC).replace(tzinfo=None),
                )
            )
            return journal

    @property
    def account_id(self) -> str:
        return self._account_id

    @property
    def empty_tip_sha256(self) -> str:
        return ZERO_HASH

_CHECKPOINT_ENTITY_TYPE = "HPRL_RECOVERY_CHECKPOINT"
_CHECKPOINT_EVENT_TYPE = "HPRL_RECOVERY_CHECKPOINT_COMMITTED"


def _checkpoint_lock_key(account_id: str, exchange: str) -> int:
    material = f"hprl-recovery-checkpoint|{exchange}|{account_id}".encode("utf-8")
    return int.from_bytes(sha256(material).digest()[:8], "big", signed=True)


class SqlRecoveryCheckpointStore:
    """PostgreSQL/SQL authoritative recovery checkpoint history.

    Checkpoints are immutable audit facts.  The latest generation is the current value;
    historical generations remain available for recovery forensics.  PostgreSQL uses a
    transaction-scoped advisory fence so the monotonic generation check is safe even
    when no prior checkpoint row exists.
    """

    def __init__(
        self,
        session_factory: object,
        *,
        account_id: str,
        exchange: str = "binance",
        actor: str = "hprl-production-recovery",
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory
        self._account_id = str(account_id).strip()
        self._exchange = str(exchange).strip().lower()
        self._actor = str(actor).strip()
        if not self._account_id or not self._exchange or not self._actor:
            raise ValueError("recovery checkpoint SQL identity is incomplete")

    def _query(self, *, for_update: bool = False):
        AuditEvent = _audit_event_model()
        statement = (
            select(AuditEvent)
            .where(
                AuditEvent.account_id == self._account_id,
                AuditEvent.exchange == self._exchange,
                AuditEvent.entity_type == _CHECKPOINT_ENTITY_TYPE,
                AuditEvent.event_type == _CHECKPOINT_EVENT_TYPE,
            )
            .order_by(AuditEvent.id.asc())
        )
        if for_update:
            statement = statement.with_for_update()
        return statement

    def _lock_writer(self, session: object) -> None:
        bind = session.get_bind()  # type: ignore[attr-defined]
        if getattr(getattr(bind, "dialect", None), "name", "") == "postgresql":
            session.execute(  # type: ignore[attr-defined]
                text("SELECT pg_advisory_xact_lock(:key)"),
                {"key": _checkpoint_lock_key(self._account_id, self._exchange)},
            )

    @staticmethod
    def _checkpoint_from_row(row: object):
        from .recovery_checkpoint import DurableRecoveryCheckpoint

        payload = json.loads(str(row.payload_json))
        if not isinstance(payload, dict):
            raise ValueError("recovery checkpoint SQL audit payload must be an object")
        return DurableRecoveryCheckpoint.from_payload(payload)

    def load(self):
        with self._session_factory() as session:  # type: ignore[operator]
            rows = tuple(session.scalars(self._query()).all())
            if not rows:
                return None
            return self._checkpoint_from_row(rows[-1])

    def save_atomic(self, checkpoint: object) -> str:
        from .recovery_checkpoint import DurableRecoveryCheckpoint

        if not isinstance(checkpoint, DurableRecoveryCheckpoint):
            raise TypeError("checkpoint must be DurableRecoveryCheckpoint")
        with self._session_factory.begin() as session:  # type: ignore[operator]
            self._lock_writer(session)
            rows = tuple(session.scalars(self._query(for_update=True)).all())
            current = None if not rows else self._checkpoint_from_row(rows[-1])
            if current is not None:
                if current.checkpoint_sha256 == checkpoint.checkpoint_sha256:
                    return checkpoint.checkpoint_sha256
                if checkpoint.generation <= current.generation:
                    raise ValueError("recovery checkpoint generation must advance monotonically")
            AuditEvent = _audit_event_model()
            session.add(
                AuditEvent(
                    account_id=self._account_id,
                    exchange=self._exchange,
                    event_type=_CHECKPOINT_EVENT_TYPE,
                    entity_type=_CHECKPOINT_ENTITY_TYPE,
                    entity_id=checkpoint.checkpoint_sha256,
                    severity="INFO",
                    reason_code="CHECKPOINT_COMMITTED",
                    correlation_id=f"checkpoint:{checkpoint.generation}",
                    actor=self._actor,
                    payload_json=json.dumps(
                        checkpoint.payload(),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                    occurred_at=checkpoint.created_at.astimezone(UTC).replace(tzinfo=None),
                )
            )
        return checkpoint.checkpoint_sha256

    @property
    def account_id(self) -> str:
        return self._account_id
