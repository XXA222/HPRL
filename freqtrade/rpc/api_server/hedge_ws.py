"""Versioned Hedge WebSocket hub with fail-closed RBAC and account scoping."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from numbers import Real
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from freqtrade.hedge.control.auth import HedgeRole
from freqtrade.hedge.telemetry.events import HedgeTelemetryEvent


@dataclass(frozen=True, slots=True)
class HedgeWsPrincipal:
    """Least-privilege WebSocket principal returned by the token validator."""

    subject: str
    role: HedgeRole = HedgeRole.VIEWER
    account_ids: frozenset[str] | None = None
    allow_sensitive: bool = False

    def __post_init__(self) -> None:
        subject = str(self.subject).strip()
        if not subject or len(subject) > 128:
            raise ValueError("WebSocket principal subject is invalid")
        try:
            role = self.role if isinstance(self.role, HedgeRole) else HedgeRole(self.role)
        except (TypeError, ValueError) as exc:
            raise ValueError("WebSocket principal role is invalid") from exc
        if role < HedgeRole.VIEWER:
            raise PermissionError("WebSocket access requires at least VIEWER")
        accounts = self.account_ids
        if accounts is not None:
            normalized = frozenset(
                value
                for item in accounts
                if (value := str(item).strip())
            )
            if not normalized:
                raise ValueError("account_ids cannot be an empty scope")
            accounts = normalized
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "account_ids", accounts)


TokenResult = bool | HedgeWsPrincipal
TokenValidator = Callable[[str | None], TokenResult | Awaitable[TokenResult]]


@dataclass(frozen=True, slots=True)
class SequencedHedgeEvent:
    sequence: int
    aggregate_key: str
    event: HedgeTelemetryEvent

    def as_dict(self) -> dict[str, Any]:
        payload = self.event.as_dict()
        payload["sequence"] = self.sequence
        payload["aggregate_key"] = self.aggregate_key
        return payload


def _gap_payload(start: int, end: int, latest: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "payload_version": 1,
        "event_type": "GAP",
        "sequence": end,
        "aggregate_key": "hedge-control-plane",
        "gap": {
            "from_sequence": start,
            "to_sequence": end,
            "latest_sequence": latest,
            "action": "FETCH_REST_SNAPSHOT_AND_RESUBSCRIBE",
        },
    }


class HedgeEventHub:
    """Bounded sequence/replay hub with explicit gap notification."""

    def __init__(self, *, queue_size: int = 256, history_size: int = 2048) -> None:
        for name, value in (("queue_size", queue_size), ("history_size", history_size)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._queue_size = queue_size
        self._history_size = history_size
        self._subscribers: set[asyncio.Queue[SequencedHedgeEvent]] = set()
        self._subscriber_gaps: dict[asyncio.Queue[SequencedHedgeEvent], tuple[int, int]] = {}
        self._history: deque[SequencedHedgeEvent] = deque(maxlen=history_size)
        self._lock = asyncio.Lock()
        self._published = 0
        self._dropped = 0
        self._sequence = 0

    @staticmethod
    def _aggregate_key(event: HedgeTelemetryEvent) -> str:
        symbol = event.symbol or "*"
        return f"{event.account_id}:{symbol}:{event.event_type.value}"

    async def publish(self, event: HedgeTelemetryEvent) -> None:
        if not isinstance(event, HedgeTelemetryEvent):
            raise TypeError("event must be a HedgeTelemetryEvent")
        async with self._lock:
            self._sequence += 1
            envelope = SequencedHedgeEvent(
                sequence=self._sequence,
                aggregate_key=self._aggregate_key(event),
                event=event,
            )
            self._history.append(envelope)
            subscribers = tuple(self._subscribers)
            self._published += 1
            for queue in subscribers:
                if queue.full():
                    try:
                        removed = queue.get_nowait()
                        self._dropped += 1
                        current = self._subscriber_gaps.get(queue)
                        start = removed.sequence if current is None else min(current[0], removed.sequence)
                        end = removed.sequence if current is None else max(current[1], removed.sequence)
                        self._subscriber_gaps[queue] = (start, end)
                    except asyncio.QueueEmpty:
                        pass
                queue.put_nowait(envelope)

    async def subscribe(self) -> AsyncIterator[HedgeTelemetryEvent]:
        queue: asyncio.Queue[SequencedHedgeEvent] = asyncio.Queue(maxsize=self._queue_size)
        async with self._lock:
            self._subscribers.add(queue)
        try:
            while True:
                envelope = await queue.get()
                async with self._lock:
                    self._subscriber_gaps.pop(queue, None)
                yield envelope.event
        finally:
            async with self._lock:
                self._subscribers.discard(queue)
                self._subscriber_gaps.pop(queue, None)

    async def subscribe_envelopes(
        self,
        *,
        after_sequence: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        if after_sequence is not None and (
            not isinstance(after_sequence, int)
            or isinstance(after_sequence, bool)
            or after_sequence < 0
        ):
            raise ValueError("after_sequence must be a non-negative integer")
        queue: asyncio.Queue[SequencedHedgeEvent] = asyncio.Queue(maxsize=self._queue_size)
        initial_gap: tuple[int, int] | None = None
        async with self._lock:
            history = tuple(self._history)
            latest = self._sequence
            if after_sequence is None:
                replay: tuple[SequencedHedgeEvent, ...] = ()
            else:
                if history and after_sequence < history[0].sequence - 1:
                    initial_gap = (after_sequence + 1, history[0].sequence - 1)
                replay = tuple(item for item in history if item.sequence > after_sequence)
            self._subscribers.add(queue)
        try:
            if initial_gap is not None:
                yield _gap_payload(initial_gap[0], initial_gap[1], latest)
            for envelope in replay:
                yield envelope.as_dict()
            while True:
                envelope = await queue.get()
                async with self._lock:
                    gap = self._subscriber_gaps.pop(queue, None)
                    latest = self._sequence
                if gap is not None:
                    yield _gap_payload(gap[0], gap[1], latest)
                yield envelope.as_dict()
        finally:
            async with self._lock:
                self._subscribers.discard(queue)
                self._subscriber_gaps.pop(queue, None)

    async def stats(self) -> dict[str, int]:
        async with self._lock:
            return {
                "subscribers": len(self._subscribers),
                "published": self._published,
                "dropped": self._dropped,
                "queue_size": self._queue_size,
                "history_size": self._history_size,
                "oldest_sequence": self._history[0].sequence if self._history else self._sequence,
                "latest_sequence": self._sequence,
            }


def _normalize_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    token = value.strip()
    if not token or len(token) > 4_096 or any(ord(ch) < 32 or ord(ch) == 127 for ch in token):
        return None
    return token


def _extract_token(websocket: WebSocket, *, allow_query_token: bool) -> tuple[bool, str | None]:
    authorization = websocket.headers.get("authorization")
    if authorization is not None:
        scheme, separator, value = authorization.partition(" ")
        if not separator or scheme.lower() != "bearer":
            return False, None
        token = _normalize_token(value)
        return (token is not None), token
    if allow_query_token and "token" in websocket.query_params:
        token = _normalize_token(websocket.query_params.get("token"))
        return (token is not None), token
    return True, None


async def _authorize(
    validator: TokenValidator,
    token: str | None,
    timeout: float,
) -> HedgeWsPrincipal | None:
    async def invoke() -> HedgeWsPrincipal | None:
        if inspect.iscoroutinefunction(validator):
            result = await validator(token)
        else:
            result = await asyncio.to_thread(validator, token)
            if inspect.isawaitable(result):
                result = await result
        # Boolean support is retained only for third-party/tests using the old API.
        # It maps to a least-role, all-account compatibility principal.
        if result is True:
            return HedgeWsPrincipal(subject="legacy-validator", role=HedgeRole.VIEWER)
        if result is False or not isinstance(result, HedgeWsPrincipal):
            return None
        return result if result.role >= HedgeRole.VIEWER else None

    try:
        return await asyncio.wait_for(invoke(), timeout=timeout)
    except Exception:
        return None


_SENSITIVE_FRAGMENTS = (
    "api_key",
    "apikey",
    "secret",
    "signature",
    "listenkey",
    "listen_key",
    "credential",
    "authorization",
    "token",
)


def _redact_payload(value: object, *, allow_sensitive: bool) -> object:
    if allow_sensitive:
        return value
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            lowered = key.lower().replace("-", "_")
            if any(fragment in lowered for fragment in _SENSITIVE_FRAGMENTS):
                result[key] = "[REDACTED]"
            else:
                result[key] = _redact_payload(item, allow_sensitive=False)
        return result
    if isinstance(value, list):
        return [_redact_payload(item, allow_sensitive=False) for item in value]
    return value


def _filter_event_for_principal(
    event: dict[str, Any], principal: HedgeWsPrincipal
) -> dict[str, Any] | None:
    if event.get("event_type") == "GAP":
        return event
    account_id = str(event.get("account_id", "")).strip()
    if principal.account_ids is not None and account_id not in principal.account_ids:
        return None
    return _redact_payload(event, allow_sensitive=principal.allow_sensitive)  # type: ignore[return-value]


def create_hedge_ws_router(
    hub: HedgeEventHub,
    *,
    token_validator: TokenValidator,
    allow_query_token: bool = False,
    auth_timeout_seconds: Real = 5.0,
) -> APIRouter:
    if not isinstance(hub, HedgeEventHub):
        raise TypeError("hub must be a HedgeEventHub")
    if not callable(token_validator):
        raise TypeError("token_validator must be callable")
    if not isinstance(allow_query_token, bool):
        raise TypeError("allow_query_token must be a boolean")
    if (
        not isinstance(auth_timeout_seconds, Real)
        or isinstance(auth_timeout_seconds, bool)
        or not math.isfinite(float(auth_timeout_seconds))
        or float(auth_timeout_seconds) <= 0
    ):
        raise ValueError("auth_timeout_seconds must be positive and finite")
    auth_timeout = float(auth_timeout_seconds)
    router = APIRouter(tags=["hedge-ws"])

    @router.websocket("/hedge/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        credential_valid, token = _extract_token(websocket, allow_query_token=allow_query_token)
        if not credential_valid:
            await websocket.close(code=1008)
            return
        principal = await _authorize(token_validator, token, auth_timeout)
        if principal is None:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        raw_after_sequence = websocket.query_params.get("after_sequence")
        try:
            after_sequence = None if raw_after_sequence is None else int(raw_after_sequence)
            if after_sequence is not None and after_sequence < 0:
                raise ValueError
        except ValueError:
            await websocket.close(code=1008)
            return
        iterator = hub.subscribe_envelopes(after_sequence=after_sequence)
        event_task: asyncio.Task[dict[str, Any]] | None = None
        receive_task: asyncio.Task[dict] | None = None
        try:
            event_task = asyncio.create_task(anext(iterator))
            receive_task = asyncio.create_task(websocket.receive())
            while True:
                done, _ = await asyncio.wait(
                    {event_task, receive_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if receive_task in done:
                    message = receive_task.result()
                    if message.get("type") == "websocket.disconnect":
                        break
                    receive_task = asyncio.create_task(websocket.receive())
                if event_task in done:
                    event = event_task.result()
                    filtered = _filter_event_for_principal(event, principal)
                    if filtered is not None:
                        await websocket.send_json(filtered)
                    event_task = asyncio.create_task(anext(iterator))
        except (asyncio.CancelledError, WebSocketDisconnect, StopAsyncIteration):
            return
        finally:
            for task in (event_task, receive_task):
                if task is not None and not task.done():
                    task.cancel()
            pending = [task for task in (event_task, receive_task) if task is not None]
            if pending:
                with suppress(asyncio.CancelledError):
                    await asyncio.gather(*pending, return_exceptions=True)
            with suppress(asyncio.CancelledError, RuntimeError):
                await iterator.aclose()

    return router


__all__ = [
    "HedgeEventHub",
    "HedgeWsPrincipal",
    "SequencedHedgeEvent",
    "create_hedge_ws_router",
]
