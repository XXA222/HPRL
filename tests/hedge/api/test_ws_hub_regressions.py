import asyncio

import pytest

from freqtrade.hedge.telemetry.events import HedgeEventType, HedgeTelemetryEvent
from freqtrade.rpc.api_server.hedge_ws import HedgeEventHub, _extract_token


def test_hub_rejects_unbounded_or_negative_queue_size() -> None:
    with pytest.raises(ValueError, match="positive"):
        HedgeEventHub(queue_size=0)
    with pytest.raises(ValueError, match="positive"):
        HedgeEventHub(queue_size=-1)


def test_hub_drops_oldest_when_subscriber_queue_is_full() -> None:
    async def scenario() -> None:
        hub = HedgeEventHub(queue_size=1)
        iterator = hub.subscribe()

        first_pending = asyncio.create_task(anext(iterator))
        await asyncio.sleep(0)
        await hub.publish(HedgeTelemetryEvent(HedgeEventType.ORDER, {"n": 1}))
        first = await first_pending
        assert first.payload["n"] == 1

        # The generator remains subscribed while no consumer is awaiting the next item.
        await hub.publish(HedgeTelemetryEvent(HedgeEventType.ORDER, {"n": 2}))
        await hub.publish(HedgeTelemetryEvent(HedgeEventType.ORDER, {"n": 3}))
        newest = await anext(iterator)
        assert newest.payload["n"] == 3
        await iterator.aclose()

    asyncio.run(scenario())


class _WebSocketLike:
    def __init__(self, *, authorization: str | None = None, query_token: str | None = None):
        self.headers = {} if authorization is None else {"authorization": authorization}
        self.query_params = {} if query_token is None else {"token": query_token}


def test_websocket_token_prefers_authorization_and_query_is_opt_in() -> None:
    header = _WebSocketLike(authorization="Bearer header-token", query_token="query-token")
    assert _extract_token(header, allow_query_token=True) == (True, "header-token")

    query_only = _WebSocketLike(query_token="query-token")
    assert _extract_token(query_only, allow_query_token=False) == (True, None)
    assert _extract_token(query_only, allow_query_token=True) == (True, "query-token")


def test_websocket_authentication_fails_closed_when_validator_raises() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    from freqtrade.rpc.api_server.hedge_ws import create_hedge_ws_router

    def broken_validator(token):
        raise RuntimeError("auth backend unavailable")

    app = FastAPI()
    app.include_router(
        create_hedge_ws_router(HedgeEventHub(), token_validator=broken_validator)
    )
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with TestClient(app).websocket_connect(
            "/hedge/ws", headers={"authorization": "Bearer token"}
        ):
            pass
    assert exc_info.value.code == 1008


def test_websocket_envelope_replays_from_sequence() -> None:
    async def scenario() -> None:
        hub = HedgeEventHub(queue_size=4, history_size=8)
        for number in range(1, 4):
            await hub.publish(HedgeTelemetryEvent(HedgeEventType.ORDER, {"n": number}))
        iterator = hub.subscribe_envelopes(after_sequence=1)
        second = await anext(iterator)
        third = await anext(iterator)
        assert second["sequence"] == 2
        assert second["payload"]["n"] == 2
        assert third["sequence"] == 3
        assert third["payload"]["n"] == 3
        await iterator.aclose()

    asyncio.run(scenario())


def test_websocket_replay_emits_explicit_gap_when_history_was_truncated() -> None:
    async def scenario() -> None:
        hub = HedgeEventHub(queue_size=4, history_size=2)
        for number in range(1, 5):
            await hub.publish(HedgeTelemetryEvent(HedgeEventType.ORDER, {"n": number}))
        iterator = hub.subscribe_envelopes(after_sequence=0)
        gap = await anext(iterator)
        third = await anext(iterator)
        fourth = await anext(iterator)
        assert gap["event_type"] == "GAP"
        assert gap["gap"] == {
            "from_sequence": 1,
            "to_sequence": 2,
            "latest_sequence": 4,
            "action": "FETCH_REST_SNAPSHOT_AND_RESUBSCRIBE",
        }
        assert third["sequence"] == 3
        assert fourth["sequence"] == 4
        await iterator.aclose()

    asyncio.run(scenario())


def test_websocket_queue_overflow_emits_gap_before_newest_event() -> None:
    async def scenario() -> None:
        hub = HedgeEventHub(queue_size=1, history_size=8)
        iterator = hub.subscribe_envelopes()
        first_pending = asyncio.create_task(anext(iterator))
        await asyncio.sleep(0)
        await hub.publish(HedgeTelemetryEvent(HedgeEventType.ORDER, {"n": 1}))
        first = await first_pending
        assert first["sequence"] == 1

        await hub.publish(HedgeTelemetryEvent(HedgeEventType.ORDER, {"n": 2}))
        await hub.publish(HedgeTelemetryEvent(HedgeEventType.ORDER, {"n": 3}))
        gap = await anext(iterator)
        newest = await anext(iterator)
        assert gap["event_type"] == "GAP"
        assert gap["gap"]["from_sequence"] == 2
        assert gap["gap"]["to_sequence"] == 2
        assert newest["sequence"] == 3
        assert newest["payload"]["n"] == 3
        await iterator.aclose()

    asyncio.run(scenario())
