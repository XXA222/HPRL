"""Synchronous outbox publisher adapters for telemetry and tests."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable
from threading import RLock

from freqtrade.hedge.contracts.events import OutboxEvent
from freqtrade.hedge.telemetry.events import HedgeEventType, HedgeTelemetryEvent


class InMemoryEventPublisher:
    def __init__(
        self,
        callback: Callable[[OutboxEvent], None] | None = None,
        *,
        event_capacity: int = 5000,
    ) -> None:
        if (
            not isinstance(event_capacity, int)
            or isinstance(event_capacity, bool)
            or event_capacity < 1
        ):
            raise ValueError("event_capacity must be a positive integer")
        self.event_capacity = event_capacity
        self._events: deque[OutboxEvent] = deque(maxlen=event_capacity)
        self._callbacks: list[Callable[[OutboxEvent], None]] = []
        if callback is not None:
            self._callbacks.append(callback)
        self._lock = RLock()

    def publish(self, event: OutboxEvent) -> None:
        if not isinstance(event, OutboxEvent):
            raise TypeError("event must be an OutboxEvent")
        with self._lock:
            self._events.append(event)
        with self._lock:
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            callback(event)

    def events(self) -> tuple[OutboxEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def set_callback(self, callback: Callable[[OutboxEvent], None] | None) -> None:
        """Replace callbacks for backward compatibility."""
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable or None")
        with self._lock:
            self._callbacks = [] if callback is None else [callback]

    def add_callback(self, callback: Callable[[OutboxEvent], None]) -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[OutboxEvent], None]) -> None:
        with self._lock:
            self._callbacks = [item for item in self._callbacks if item != callback]

    @property
    def callback_count(self) -> int:
        with self._lock:
            return len(self._callbacks)


class HedgeEventHubPublisher:
    """Publish execution outbox events to an async HedgeEventHub from sync code."""

    def __init__(self, hub: object) -> None:
        publish = getattr(hub, "publish", None)
        if not callable(publish):
            raise TypeError("hub must expose async publish(event)")
        self._hub = hub
        self._background_tasks: set[asyncio.Task[None]] = set()

    def publish(self, event: OutboxEvent) -> None:
        category = event.event_type.split("_", 1)[0]
        event_type = HedgeEventType.ORDER
        if category == "INTENT":
            event_type = HedgeEventType.INTENT
        elif category == "FILL":
            event_type = HedgeEventType.FILL
        elif category == "HALT":
            event_type = HedgeEventType.HALT
        telemetry = HedgeTelemetryEvent(
            event_type=event_type,
            payload={"event_type": event.event_type, **dict(event.payload)},
            account_id=str(event.payload.get("account_id", "default")),
            symbol=event.payload.get("symbol"),
            correlation_id=event.correlation_id,
            occurred_at=event.occurred_at,
        )
        coroutine = self._hub.publish(telemetry)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coroutine)
        else:
            task = loop.create_task(coroutine)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
