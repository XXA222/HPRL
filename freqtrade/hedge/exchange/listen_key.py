from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

from .base import BinanceRestTransport, Clock, SystemClock, maybe_await
from .rate_limit import BinanceDataError, BinanceTransportError


logger = logging.getLogger(__name__)
DEFAULT_LISTEN_KEY_TTL = timedelta(minutes=60)
DEFAULT_LISTEN_KEY_RENEW_INTERVAL = timedelta(minutes=30)
DEFAULT_LISTEN_KEY_ERROR_RETRY_INTERVAL = timedelta(seconds=30)


@dataclass(frozen=True, slots=True)
class ListenKeyLease:
    listen_key: str
    created_at: datetime
    renewed_at: datetime
    expires_at: datetime
    generation: int


class BinanceListenKeyClient:
    def __init__(self, transport: BinanceRestTransport) -> None:
        self._transport = transport

    async def create(self) -> str:
        response = await self._transport.request(
            "POST", "/fapi/v1/listenKey", signed=False, weight=1
        )
        if not isinstance(response.data, Mapping):
            raise BinanceDataError("listenKey create returned invalid data")
        listen_key = str(response.data.get("listenKey") or "").strip()
        if not listen_key:
            raise BinanceDataError("listenKey create returned an empty key")
        return listen_key

    async def keepalive(self, listen_key: str | None = None) -> None:
        await self._transport.request(
            "PUT",
            "/fapi/v1/listenKey",
            signed=False,
            weight=1,
        )

    async def close(self, listen_key: str | None = None) -> None:
        await self._transport.request(
            "DELETE",
            "/fapi/v1/listenKey",
            signed=False,
            weight=1,
        )


class ListenKeyManager:
    """Creates, renews, closes and rebuilds a Binance Futures listenKey."""

    def __init__(
        self,
        *,
        client: BinanceListenKeyClient,
        clock: Clock | None = None,
        ttl: timedelta = DEFAULT_LISTEN_KEY_TTL,
        renew_interval: timedelta = DEFAULT_LISTEN_KEY_RENEW_INTERVAL,
        error_retry_interval: timedelta = DEFAULT_LISTEN_KEY_ERROR_RETRY_INTERVAL,
        on_rebuilt: Callable[[ListenKeyLease], Awaitable[None] | None] | None = None,
    ) -> None:
        if renew_interval <= timedelta(0) or renew_interval >= ttl:
            raise ValueError("renew_interval must be positive and smaller than ttl")
        if error_retry_interval <= timedelta(0):
            raise ValueError("error_retry_interval must be positive")
        self._client = client
        self._clock = clock or SystemClock()
        self._ttl = ttl
        self._renew_interval = renew_interval
        self._error_retry_interval = error_retry_interval
        self._on_rebuilt = on_rebuilt
        self._lease: ListenKeyLease | None = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._next_renew_at: datetime | None = None

    def set_on_rebuilt(
        self, callback: Callable[[ListenKeyLease], Awaitable[None] | None] | None
    ) -> None:
        self._on_rebuilt = callback

    @property
    def lease(self) -> ListenKeyLease | None:
        return self._lease

    @property
    def listen_key(self) -> str | None:
        return self._lease.listen_key if self._lease else None

    async def ensure(self) -> ListenKeyLease:
        async with self._lock:
            now = self._clock.now()
            if self._closed:
                raise RuntimeError("ListenKeyManager is closed")
            if self._lease is None or now >= self._lease.expires_at:
                return await self._create_locked(rebuilt=self._lease is not None)
            return self._lease

    async def renew_if_due(self) -> ListenKeyLease:
        async with self._lock:
            if self._closed:
                raise RuntimeError("ListenKeyManager is closed")
            now = self._clock.now()
            if self._lease is None or now >= self._lease.expires_at:
                return await self._create_locked(rebuilt=self._lease is not None)
            if self._next_renew_at is not None and now < self._next_renew_at:
                return self._lease
            try:
                await self._client.keepalive(self._lease.listen_key)
            except BinanceTransportError as exc:
                if exc.code == -1125 or exc.status in {400, 404}:
                    return await self._create_locked(rebuilt=True)
                raise
            renewed = ListenKeyLease(
                listen_key=self._lease.listen_key,
                created_at=self._lease.created_at,
                renewed_at=now,
                expires_at=now + self._ttl,
                generation=self._lease.generation,
            )
            self._lease = renewed
            self._next_renew_at = now + self._renew_interval
            return renewed

    async def force_rebuild(self) -> ListenKeyLease:
        async with self._lock:
            if self._closed:
                raise RuntimeError("ListenKeyManager is closed")
            return await self._create_locked(rebuilt=True)

    async def _create_locked(self, *, rebuilt: bool) -> ListenKeyLease:
        old = self._lease
        # Create first. Closing the old key before a successful create can leave
        # the manager pointing at a key it has already invalidated.
        listen_key = await self._client.create()
        now = self._clock.now()
        generation = 1 if old is None else old.generation + 1
        lease = ListenKeyLease(
            listen_key=listen_key,
            created_at=now,
            renewed_at=now,
            expires_at=now + self._ttl,
            generation=generation,
        )
        self._lease = lease
        self._next_renew_at = now + self._renew_interval
        # USDⓈ-M DELETE /listenKey has no listenKey parameter and closes the
        # account's current active stream. Calling it after POST would therefore
        # close the newly-created lease rather than a stale predecessor. The old
        # socket is closed by the stream reconnect path; only shutdown deletes
        # the current active listenKey.
        if rebuilt and self._on_rebuilt is not None:
            await maybe_await(self._on_rebuilt(lease))
        return lease

    async def _wait_or_stop(
        self, seconds: float, stop_event: asyncio.Event
    ) -> bool:
        sleeper = asyncio.create_task(self._clock.sleep(max(0.0, seconds)))
        stopper = asyncio.create_task(stop_event.wait())
        done, pending = await asyncio.wait(
            {sleeper, stopper}, return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if stopper in done and stopper.result():
            return True
        sleeper.result()
        return False

    async def run_renewal_loop(self, stop_event: asyncio.Event) -> None:
        while not stop_event.is_set():
            try:
                await self.ensure()
                now = self._clock.now()
                next_renew = self._next_renew_at or (now + self._renew_interval)
                delay = max(0.0, (next_renew - now).total_seconds())
                if await self._wait_or_stop(delay, stop_event):
                    return
                await self.renew_if_due()
            except asyncio.CancelledError:
                raise
            except BinanceTransportError:
                logger.warning(
                    "Binance listenKey renewal failed; retrying",
                    exc_info=True,
                )
                retry_delay = self._error_retry_interval.total_seconds()
                if await self._wait_or_stop(retry_delay, stop_event):
                    return

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            lease = self._lease
            self._lease = None
            self._next_renew_at = None
            if lease is not None:
                try:
                    await self._client.close(lease.listen_key)
                except BinanceTransportError:
                    logger.exception("Failed to close Binance listenKey during shutdown")
