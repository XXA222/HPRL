"""Transactional outbox claiming and publication.

The application transaction must call :func:`enqueue_outbox` with the same
SQLAlchemy session used for the state change. Publication always occurs in a
new transaction, so rolled-back work can never be observed by a publisher.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Event
from time import monotonic
from typing import Any
from uuid import uuid4

from sqlalchemy import and_, func, or_, select, text, update
from sqlalchemy.orm import Session, sessionmaker

from freqtrade.persistence.hedge_contracts import EventMetadata
from freqtrade.persistence.hedge_models import EventOutbox, canonical_json, utcnow


@dataclass(frozen=True)
class OutboxEnvelope:
    row_id: int
    event_id: str
    aggregate_type: str
    aggregate_id: str
    event_type: str
    aggregate_sequence: int
    correlation_id: str
    causation_id: str | None
    payload_version: int
    event_version: int
    contracts_version: str
    schema_version: str
    exchange_time: datetime | None
    observed_time: datetime
    payload_json: str
    headers_json: str
    occurred_at: datetime
    attempts: int
    lock_token: str


@dataclass(frozen=True)
class PublishBatchResult:
    claimed: int
    published: int
    failed: int


def enqueue_outbox(
    session: Session,
    *,
    aggregate_type: str,
    aggregate_id: str,
    event_type: str,
    payload: dict[str, Any],
    headers: dict[str, Any] | None = None,
    event_id: str | None = None,
    metadata: EventMetadata | None = None,
) -> EventOutbox:
    """Add an ordered outbox row to the caller's current transaction."""

    metadata = metadata or EventMetadata(
        correlation_id=event_id or str(uuid4()),
        observed_time=utcnow(),
    )
    if session.get_bind().dialect.name == "postgresql":
        scope = f"outbox|{aggregate_type}|{aggregate_id}"
        session.execute(
            text("SELECT pg_advisory_xact_lock(hashtextextended(:scope, 0))"),
            {"scope": scope},
        )
    sequence = session.scalar(
        select(func.coalesce(func.max(EventOutbox.aggregate_sequence), 0)).where(
            EventOutbox.aggregate_type == aggregate_type,
            EventOutbox.aggregate_id == aggregate_id,
        )
    )
    merged_headers = metadata.headers()
    merged_headers.update(headers or {})
    row = EventOutbox(
        event_id=event_id or str(uuid4()),
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        event_type=event_type,
        aggregate_sequence=int(sequence or 0) + 1,
        correlation_id=metadata.correlation_id,
        causation_id=metadata.causation_id,
        payload_version=metadata.payload_version,
        event_version=metadata.event_version,
        contracts_version=metadata.contracts_version,
        schema_version=metadata.schema_version,
        exchange_time=metadata.exchange_time,
        observed_time=metadata.observed_time or utcnow(),
        payload_json=canonical_json(payload),
        headers_json=canonical_json(merged_headers),
    )
    session.add(row)
    return row



class TransactionalOutboxPublisher:
    """Database-backed at-least-once publisher with dialect-safe claiming."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        lease_seconds: int = 60,
        max_attempts: int = 25,
        retry_base_seconds: int = 1,
        retry_max_seconds: int = 300,
    ):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if retry_base_seconds <= 0:
            raise ValueError("retry_base_seconds must be positive")
        if retry_max_seconds < retry_base_seconds:
            raise ValueError("retry_max_seconds must be >= retry_base_seconds")
        self.session_factory = session_factory
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.retry_base_seconds = retry_base_seconds
        self.retry_max_seconds = retry_max_seconds

    def _retry_delay(self, attempts: int) -> timedelta:
        remaining_doublings = max(0, attempts - 1)
        seconds = self.retry_base_seconds
        while remaining_doublings and seconds < self.retry_max_seconds:
            seconds = min(self.retry_max_seconds, seconds * 2)
            remaining_doublings -= 1
        return timedelta(seconds=seconds)

    def claim_batch(self, *, limit: int = 100) -> tuple[OutboxEnvelope, ...]:
        if limit <= 0:
            return ()
        token = str(uuid4())
        now = utcnow()
        lease_expired = now - timedelta(seconds=self.lease_seconds)
        claimed_ids: list[int] = []
        with self.session_factory.begin() as session:
            dialect = session.get_bind().dialect.name
            exhausted = or_(
                and_(
                    EventOutbox.status.in_(("PENDING", "FAILED")),
                    EventOutbox.attempts >= self.max_attempts,
                ),
                and_(
                    EventOutbox.status == "PROCESSING",
                    EventOutbox.attempts >= self.max_attempts,
                    or_(
                        EventOutbox.lock_token.is_(None),
                        EventOutbox.locked_at.is_(None),
                        EventOutbox.locked_at < lease_expired,
                    ),
                ),
            )
            session.execute(
                update(EventOutbox)
                .where(exhausted)
                .values(
                    status="DEAD",
                    available_at=now,
                    lock_token=None,
                    locked_at=None,
                    last_error="outbox retry limit exhausted",
                    updated_at=now,
                )
            )
            eligible = (
                EventOutbox.status.in_(("PENDING", "FAILED", "PROCESSING")),
                EventOutbox.available_at <= now,
                EventOutbox.attempts < self.max_attempts,
                or_(
                    EventOutbox.lock_token.is_(None),
                    EventOutbox.locked_at.is_(None),
                    EventOutbox.locked_at < lease_expired,
                ),
            )
            stmt = select(EventOutbox.id).where(*eligible).order_by(EventOutbox.id).limit(limit)
            if dialect == "postgresql":
                stmt = stmt.with_for_update(skip_locked=True)
            candidate_ids = list(session.scalars(stmt))
            for row_id in candidate_ids:
                candidate = session.get(EventOutbox, row_id)
                if candidate is None:
                    continue
                blocker = session.scalar(
                    select(EventOutbox.id)
                    .where(
                        EventOutbox.aggregate_type == candidate.aggregate_type,
                        EventOutbox.aggregate_id == candidate.aggregate_id,
                        EventOutbox.aggregate_sequence < candidate.aggregate_sequence,
                        EventOutbox.status != "PUBLISHED",
                    )
                    .limit(1)
                )
                if blocker is not None:
                    continue
                result = session.execute(
                    update(EventOutbox)
                    .where(EventOutbox.id == row_id, *eligible)
                    .values(
                        status="PROCESSING",
                        lock_token=token,
                        locked_at=now,
                        attempts=EventOutbox.attempts + 1,
                        updated_at=now,
                    )
                )
                if result.rowcount == 1:
                    claimed_ids.append(row_id)

        if not claimed_ids:
            return ()
        with self.session_factory() as session:
            rows = session.scalars(
                select(EventOutbox)
                .where(EventOutbox.id.in_(claimed_ids), EventOutbox.lock_token == token)
                .order_by(EventOutbox.id)
            ).all()
            return tuple(
                OutboxEnvelope(
                    row_id=row.id,
                    event_id=row.event_id,
                    aggregate_type=row.aggregate_type,
                    aggregate_id=row.aggregate_id,
                    event_type=row.event_type,
                    aggregate_sequence=row.aggregate_sequence,
                    correlation_id=row.correlation_id,
                    causation_id=row.causation_id,
                    payload_version=row.payload_version,
                    event_version=row.event_version,
                    contracts_version=row.contracts_version,
                    schema_version=row.schema_version,
                    exchange_time=row.exchange_time,
                    observed_time=row.observed_time,
                    payload_json=row.payload_json,
                    headers_json=row.headers_json,
                    occurred_at=row.occurred_at,
                    attempts=row.attempts,
                    lock_token=token,
                )
                for row in rows
            )

    def mark_published(self, envelope: OutboxEnvelope) -> bool:
        now = utcnow()
        with self.session_factory.begin() as session:
            result = session.execute(
                update(EventOutbox)
                .where(
                    EventOutbox.id == envelope.row_id,
                    EventOutbox.lock_token == envelope.lock_token,
                    EventOutbox.status == "PROCESSING",
                )
                .values(
                    status="PUBLISHED",
                    published_at=now,
                    lock_token=None,
                    locked_at=None,
                    last_error=None,
                    updated_at=now,
                )
            )
            return result.rowcount == 1

    def mark_failed(self, envelope: OutboxEnvelope, error: Exception | str) -> bool:
        now = utcnow()
        message = str(error)
        exhausted = envelope.attempts >= self.max_attempts
        with self.session_factory.begin() as session:
            result = session.execute(
                update(EventOutbox)
                .where(
                    EventOutbox.id == envelope.row_id,
                    EventOutbox.lock_token == envelope.lock_token,
                    EventOutbox.status == "PROCESSING",
                )
                .values(
                    status="DEAD" if exhausted else "FAILED",
                    available_at=(
                        now if exhausted else now + self._retry_delay(envelope.attempts)
                    ),
                    lock_token=None,
                    locked_at=None,
                    last_error=message[:8000],
                    updated_at=now,
                )
            )
            return result.rowcount == 1

    def publish_batch(
        self,
        publish: Callable[[OutboxEnvelope], None],
        *,
        limit: int = 100,
    ) -> PublishBatchResult:
        envelopes = self.claim_batch(limit=limit)
        published = 0
        failed = 0
        for envelope in envelopes:
            try:
                publish(envelope)
            except Exception as exc:  # publisher errors must be persisted before continuing
                self.mark_failed(envelope, exc)
                failed += 1
            else:
                if self.mark_published(envelope):
                    published += 1
                else:
                    failed += 1
        return PublishBatchResult(claimed=len(envelopes), published=published, failed=failed)

    def requeue_dead(
        self,
        *,
        event_id: str,
        resolved_by: str,
        note: str,
    ) -> bool:
        now = utcnow()
        with self.session_factory.begin() as session:
            result = session.execute(
                update(EventOutbox)
                .where(
                    EventOutbox.event_id == event_id,
                    EventOutbox.status == "DEAD",
                )
                .values(
                    status="PENDING",
                    attempts=0,
                    available_at=now,
                    lock_token=None,
                    locked_at=None,
                    last_error=None,
                    resolution_status="REQUEUED",
                    resolved_by=resolved_by,
                    resolution_note=note,
                    updated_at=now,
                )
            )
            return result.rowcount == 1

    def release_stale_leases(self) -> int:
        now = utcnow()
        lease_expired = now - timedelta(seconds=self.lease_seconds)
        with self.session_factory.begin() as session:
            rows = session.scalars(
                select(EventOutbox).where(
                    EventOutbox.status == "PROCESSING",
                    EventOutbox.locked_at < lease_expired,
                )
            ).all()
            for row in rows:
                exhausted = row.attempts >= self.max_attempts
                row.status = "DEAD" if exhausted else "FAILED"
                row.available_at = (
                    now if exhausted else now + self._retry_delay(row.attempts)
                )
                row.lock_token = None
                row.locked_at = None
                row.last_error = "outbox lease expired before acknowledgement"
                row.updated_at = now
            return len(rows)



class OutboxWorker:
    """Cooperative background worker around :class:`TransactionalOutboxPublisher`."""

    def __init__(
        self,
        publisher: TransactionalOutboxPublisher,
        publish: Callable[[OutboxEnvelope], None],
        *,
        poll_interval_seconds: float = 1.0,
        batch_size: int = 100,
    ):
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.publisher = publisher
        self.publish = publish
        self.poll_interval_seconds = poll_interval_seconds
        self.batch_size = batch_size
        self._stop = Event()

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> PublishBatchResult:
        return self.publisher.publish_batch(self.publish, limit=self.batch_size)

    def run_forever(self, *, max_idle_seconds: float | None = None) -> None:
        idle_started = monotonic()
        while not self._stop.is_set():
            result = self.run_once()
            if result.claimed:
                idle_started = monotonic()
                continue
            if max_idle_seconds is not None and monotonic() - idle_started >= max_idle_seconds:
                return
            self._stop.wait(self.poll_interval_seconds)


def pending_outbox_rows(session: Session) -> Sequence[EventOutbox]:
    return session.scalars(
        select(EventOutbox)
        .where(EventOutbox.status.in_(("PENDING", "FAILED")))
        .order_by(EventOutbox.id)
    ).all()


def dead_letter_rows(session: Session) -> Sequence[EventOutbox]:
    return session.scalars(
        select(EventOutbox)
        .where(EventOutbox.status == "DEAD")
        .order_by(EventOutbox.id)
    ).all()
