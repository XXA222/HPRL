"""Reliable dispatcher for transactionally recorded execution outbox events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

from freqtrade.hedge.contracts.events import OutboxEvent
from freqtrade.hedge.contracts.ports import ClockPort, EventPublisherPort, SystemClock


class OutboxStorePort(Protocol):
    def outbox(self, *, unpublished_only: bool = False) -> Sequence[OutboxEvent]: ...

    def mark_published(
        self,
        event_id: str,
        *,
        published_at: datetime | None = None,
    ) -> None: ...

    def mark_publish_attempt(self, event_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class OutboxDispatchReport:
    attempted: int
    published: int
    failed: tuple[str, ...] = ()


class OutboxDispatcher:
    def __init__(
        self,
        store: OutboxStorePort,
        publisher: EventPublisherPort,
        *,
        clock: ClockPort | None = None,
    ) -> None:
        self._store = store
        self._publisher = publisher
        self._clock = clock or SystemClock()

    def dispatch(self, *, limit: int = 100) -> OutboxDispatchReport:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        events = tuple(self._store.outbox(unpublished_only=True))[:limit]
        published = 0
        failures: list[str] = []
        for event in events:
            try:
                self._publisher.publish(event)
            except Exception as exc:
                self._store.mark_publish_attempt(str(event.event_id))
                failures.append(f"{event.event_id}:{type(exc).__name__}")
                continue
            self._store.mark_published(
                str(event.event_id),
                published_at=self._clock.now(),
            )
            published += 1
        return OutboxDispatchReport(len(events), published, tuple(failures))
