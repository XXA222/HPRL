import asyncio
from datetime import timedelta

import pytest

from freqtrade.hedge.exchange.base import BinanceHttpResponse
from freqtrade.hedge.exchange.listen_key import (
    BinanceListenKeyClient,
    ListenKeyManager,
)
from freqtrade.hedge.exchange.rate_limit import BinanceTransportError

from ._helpers import FakeClock


class Transport:
    def __init__(self):
        self.created = 0
        self.keepalive_error = False
        self.calls = []

    async def request(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        if method == "POST":
            self.created += 1
            return BinanceHttpResponse(
                {"listenKey": f"key-{self.created}"},
                200,
                {},
            )
        if method == "PUT" and self.keepalive_error:
            self.keepalive_error = False
            raise BinanceTransportError(
                "expired",
                status=400,
                code=-1125,
            )
        return BinanceHttpResponse({}, 200, {})


def listen_key_manager(transport, clock, *, on_rebuilt=None):
    return ListenKeyManager(
        client=BinanceListenKeyClient(transport),
        clock=clock,
        ttl=timedelta(minutes=60),
        renew_interval=timedelta(minutes=30),
        on_rebuilt=on_rebuilt,
    )


def test_fake_clock_renew_and_expired_rebuild():
    clock = FakeClock()
    transport = Transport()
    manager = listen_key_manager(transport, clock)

    first = asyncio.run(manager.ensure())
    assert first.listen_key == "key-1"

    clock.advance(1801)
    renewed = asyncio.run(manager.renew_if_due())
    assert renewed.listen_key == "key-1"

    clock.advance(1801)
    transport.keepalive_error = True
    rebuilt = asyncio.run(manager.renew_if_due())
    assert rebuilt.listen_key == "key-2"
    assert rebuilt.generation == 2


def test_rebuild_callback_is_invoked_with_new_generation():
    clock = FakeClock()
    transport = Transport()
    rebuilt = []

    async def callback(lease):
        rebuilt.append(lease.generation)

    manager = listen_key_manager(
        transport,
        clock,
        on_rebuilt=callback,
    )
    asyncio.run(manager.ensure())
    asyncio.run(manager.force_rebuild())
    assert rebuilt == [2]


def test_failed_rebuild_does_not_close_or_forget_previous_lease():
    clock = FakeClock()
    transport = Transport()
    manager = listen_key_manager(transport, clock)
    first = asyncio.run(manager.ensure())
    original_request = transport.request

    async def failing_request(method, path, **kwargs):
        if method == "POST":
            raise BinanceTransportError("create failed", retryable=False)
        return await original_request(method, path, **kwargs)

    transport.request = failing_request
    with pytest.raises(BinanceTransportError):
        asyncio.run(manager.force_rebuild())

    assert manager.lease == first
    assert not any(
        method == "DELETE"
        for method, _path, _kwargs in transport.calls
    )


def test_renewal_loop_uses_injected_fake_clock():
    clock = FakeClock()
    stop = asyncio.Event()

    class Client:
        def __init__(self):
            self.keepalives = 0

        async def create(self):
            return "key"

        async def keepalive(self, key=None):
            self.keepalives += 1
            stop.set()

        async def close(self, key=None):
            return None

    client = Client()
    manager = ListenKeyManager(
        client=client,
        clock=clock,
        ttl=timedelta(seconds=20),
        renew_interval=timedelta(seconds=10),
    )
    asyncio.run(manager.run_renewal_loop(stop))
    assert client.keepalives == 1
    assert clock.mono == 110


def test_rebuild_never_deletes_after_create_because_delete_targets_current_key():
    clock = FakeClock()
    transport = Transport()
    manager = listen_key_manager(transport, clock)
    asyncio.run(manager.ensure())
    asyncio.run(manager.force_rebuild())

    methods = [method for method, _path, _kwargs in transport.calls]
    assert methods.count("POST") == 2
    assert "DELETE" not in methods


def test_renewal_loop_retries_transient_transport_failure():
    clock = FakeClock()
    stop = asyncio.Event()

    class Client:
        def __init__(self):
            self.keepalives = 0

        async def create(self):
            return "key"

        async def keepalive(self, key=None):
            self.keepalives += 1
            if self.keepalives == 1:
                raise BinanceTransportError("temporary", retryable=True)
            stop.set()

        async def close(self, key=None):
            return None

    client = Client()
    manager = ListenKeyManager(
        client=client,
        clock=clock,
        ttl=timedelta(seconds=20),
        renew_interval=timedelta(seconds=10),
        error_retry_interval=timedelta(seconds=2),
    )

    asyncio.run(manager.run_renewal_loop(stop))

    assert client.keepalives == 2
    assert clock.mono == 112
