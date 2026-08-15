from __future__ import annotations

from datetime import timedelta

from sqlalchemy import func, select

from freqtrade.persistence.hedge_models import EventOutbox
from freqtrade.persistence.hedge_outbox import (
    TransactionalOutboxPublisher,
    dead_letter_rows,
    enqueue_outbox,
)


def test_outbox_publishes_only_committed_events(session_factory):
    with session_factory.begin() as session:
        enqueue_outbox(
            session,
            aggregate_type="OrderIntent",
            aggregate_id="intent-1",
            event_type="hedge.intent.created",
            payload={"id": "intent-1"},
        )

    published: list[str] = []
    publisher = TransactionalOutboxPublisher(session_factory)
    result = publisher.publish_batch(lambda envelope: published.append(envelope.event_id))
    assert result.claimed == 1
    assert result.published == 1
    assert result.failed == 0
    assert len(published) == 1

    with session_factory() as session:
        row = session.scalar(select(EventOutbox))
        assert row is not None
        assert row.status == "PUBLISHED"
        assert row.published_at is not None


def test_outbox_callback_failure_is_retryable(session_factory):
    with session_factory.begin() as session:
        enqueue_outbox(
            session,
            aggregate_type="FillEvent",
            aggregate_id="fill-1",
            event_type="hedge.fill.accepted",
            payload={"id": "fill-1"},
        )

    publisher = TransactionalOutboxPublisher(session_factory)
    first = publisher.publish_batch(lambda _: (_ for _ in ()).throw(RuntimeError("broker down")))
    assert first.failed == 1
    with session_factory() as session:
        row = session.scalar(select(EventOutbox))
        assert row is not None
        assert row.status == "FAILED"
        assert "broker down" in (row.last_error or "")

    with session_factory.begin() as session:
        row = session.scalar(select(EventOutbox))
        assert row is not None
        row.available_at = row.available_at - timedelta(minutes=1)

    second = publisher.publish_batch(lambda _: None)
    assert second.published == 1


def test_rolled_back_outbox_is_never_claimed(session_factory):
    session = session_factory()
    try:
        enqueue_outbox(
            session,
            aggregate_type="Risk",
            aggregate_id="risk-1",
            event_type="hedge.risk.changed",
            payload={"state": "HALT"},
        )
        session.flush()
        session.rollback()
    finally:
        session.close()

    publisher = TransactionalOutboxPublisher(session_factory)
    result = publisher.publish_batch(lambda _: None)
    assert result.claimed == 0
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(EventOutbox)) == 0


def test_outbox_moves_to_dead_letter_after_retry_limit(session_factory):
    with session_factory.begin() as session:
        enqueue_outbox(
            session,
            aggregate_type="FillEvent",
            aggregate_id="fill-dead",
            event_type="hedge.fill.accepted",
            payload={"id": "fill-dead"},
        )

    publisher = TransactionalOutboxPublisher(
        session_factory,
        max_attempts=2,
        retry_base_seconds=1,
    )
    failure = lambda _: (_ for _ in ()).throw(RuntimeError("broker remains down"))
    first = publisher.publish_batch(failure)
    assert first.failed == 1

    with session_factory.begin() as session:
        row = session.scalar(select(EventOutbox))
        assert row is not None
        row.available_at = row.available_at - timedelta(minutes=1)

    second = publisher.publish_batch(failure)
    assert second.failed == 1
    with session_factory() as session:
        rows = dead_letter_rows(session)
        assert len(rows) == 1
        assert rows[0].status == "DEAD"
        assert rows[0].attempts == 2
        assert "broker remains down" in (rows[0].last_error or "")


def test_claim_quarantines_preexisting_exhausted_rows(session_factory):
    with session_factory.begin() as session:
        row = enqueue_outbox(
            session,
            aggregate_type="Risk",
            aggregate_id="risk-exhausted",
            event_type="hedge.risk.changed",
            payload={"state": "HALT"},
        )
        session.flush([row])
        row.status = "FAILED"
        row.attempts = 3

    publisher = TransactionalOutboxPublisher(session_factory, max_attempts=3)
    assert publisher.claim_batch() == ()
    with session_factory() as session:
        row = session.scalar(select(EventOutbox))
        assert row is not None
        assert row.status == "DEAD"
        assert row.lock_token is None
        assert row.last_error == "outbox retry limit exhausted"
