from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from freqtrade.hedge.exchange.base import (
    AccountConfigurationFact,
    AccountSnapshotFact,
    BalanceFact,
    PositionFact,
)
from freqtrade.hedge.exchange.binance_readonly import BinanceAccountBundle
from freqtrade.hedge.exchange.binance_user_stream import BinanceUserStream
from freqtrade.hedge.exchange.listen_key import ListenKeyManager
from freqtrade.hedge.readonly.calibration import ReadonlyCalibration
from freqtrade.hedge.readonly.freshness import FreshnessPolicy, UserStreamFreshness
from freqtrade.hedge.readonly.runtime import (
    BinanceReadonlyRuntime,
    BinanceReadonlyRuntimeConfig,
)
from freqtrade.hedge.readonly.scheduler import (
    CalibrationSchedule,
    ReconciliationScheduler,
)
from freqtrade.hedge.readonly.service import BinanceReadonlyService

from ._helpers import FakeRepository


class RealtimeClock:
    def now(self):
        return datetime.now(UTC)

    def monotonic(self):
        return time.monotonic()

    async def sleep(self, seconds):
        await asyncio.sleep(seconds)


class Repository(FakeRepository):
    async def append_position_snapshots(
        self,
        facts,
        *,
        reconciliation_run_id=None,
    ):
        await super().append_position_snapshots(
            facts,
            reconciliation_run_id=reconciliation_run_id,
        )
        by_key = {(item.symbol, item.position_side): item for item in self.local_positions}
        for item in facts:
            by_key[(item.symbol, item.position_side)] = item
        self.local_positions = list(by_key.values())


class Transport:
    def __init__(self):
        self.closed = 0

    async def close(self):
        self.closed += 1


class Client:
    def __init__(self, clock):
        self.account_id = "acct"
        self.clock = clock
        self.transport = Transport()
        self.positions = (self._position("0", 1000),)
        self.bundle_calls = 0
        self.sync_calls = 0
        self.preflight_calls = 0

    def _position(self, quantity, update_time_ms):
        return PositionFact(
            account_id="acct",
            symbol="BTCUSDT",
            position_side="LONG",
            quantity=Decimal(quantity),
            entry_price=Decimal("100"),
            mark_price=Decimal("101"),
            unrealized_pnl=Decimal("0"),
            liquidation_price=None,
            leverage=5,
            margin_mode="cross",
            update_time_ms=update_time_ms,
            observed_at=self.clock.now(),
            source="BINANCE_REST",
        )

    async def synchronize_clock(self):
        self.sync_calls += 1

    async def preflight_permissions(self, policy):
        self.preflight_calls += 1

    async def fetch_bundle(self, *, include_fills, fill_start_time_ms=None):
        self.bundle_calls += 1
        now = self.clock.now()
        return BinanceAccountBundle(
            account_snapshot=AccountSnapshotFact(
                account_id="acct",
                total_wallet_balance=Decimal("1000"),
                total_available_balance=Decimal("900"),
                total_margin_balance=Decimal("1000"),
                total_initial_margin=Decimal("0"),
                total_maintenance_margin=Decimal("0"),
                total_unrealized_pnl=Decimal("0"),
                observed_at=now,
                collection_started_at=now,
                collection_completed_at=now,
            ),
            balances=(
                BalanceFact(
                    account_id="acct",
                    asset="USDT",
                    wallet_balance=Decimal("1000"),
                    available_balance=Decimal("900"),
                    cross_wallet_balance=Decimal("1000"),
                    unrealized_pnl=Decimal("0"),
                    observed_at=now,
                    source="BINANCE_REST",
                ),
            ),
            positions=self.positions,
            open_orders=(),
            fills=(),
            configuration=AccountConfigurationFact(
                account_id="acct",
                hedge_mode=True,
                active_margin_modes=("cross",),
                leverage_by_symbol_side={
                    "BTCUSDT:LONG": 5,
                    "BTCUSDT:SHORT": 5,
                },
                observed_at=now,
            ),
            income_events=(),
            collection_started_at=now,
            collection_completed_at=now,
        )


class ListenKeyClient:
    def __init__(self):
        self.created = 0
        self.closed = 0

    async def create(self):
        self.created += 1
        return f"key-{self.created}"

    async def keepalive(self, key=None):
        return None

    async def close(self, key=None):
        self.closed += 1


class Connection:
    def __init__(self, initial_messages=()):
        self.queue = asyncio.Queue()
        self.closed = False
        for message in initial_messages:
            self.queue.put_nowait(message)

    def __aiter__(self):
        return self._messages()

    async def _messages(self):
        while True:
            message = await self.queue.get()
            if message is None:
                return
            yield message

    async def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put_nowait(None)


class Connector:
    def __init__(self, first_message):
        self.first_message = first_message
        self.connections = []

    async def connect(self, url):
        messages = (self.first_message,) if not self.connections else ()
        connection = Connection(messages)
        self.connections.append(connection)
        return connection


async def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.005)


def account_update_message():
    return {
        "e": "ACCOUNT_UPDATE",
        "E": 2000,
        "T": 2000,
        "a": {
            "m": "ORDER",
            "B": [
                {
                    "a": "USDT",
                    "wb": "1000",
                    "cw": "1000",
                    "bc": "0",
                }
            ],
            "P": [
                {
                    "s": "BTCUSDT",
                    "ps": "LONG",
                    "pa": "1",
                    "ep": "100",
                    "up": "1",
                    "mt": "cross",
                }
            ],
        },
    }


@pytest.mark.asyncio
async def test_real_components_start_stream_apply_event_and_recalibrate_after_reconnect():
    clock = RealtimeClock()
    repository = Repository()
    client = Client(clock)
    listen_keys = ListenKeyManager(
        client=ListenKeyClient(),
        clock=clock,
        ttl=timedelta(minutes=60),
        renew_interval=timedelta(minutes=30),
        error_retry_interval=timedelta(seconds=1),
    )
    connector = Connector(account_update_message())
    stream = BinanceUserStream(
        account_id="acct",
        managed_symbols=("BTCUSDT",),
        repository=repository,
        listen_keys=listen_keys,
        connector=connector,
        websocket_base_url="ws://localhost:18082/ws",
        clock=clock,
        min_reconnect_delay_seconds=0.01,
        max_reconnect_delay_seconds=0.02,
        reconnect_reset_after_seconds=0.1,
    )
    calibration = ReadonlyCalibration(
        client=client,
        repository=repository,
        managed_symbols=("BTCUSDT",),
        clock=clock,
    )
    scheduler = ReconciliationScheduler(
        calibration=calibration,
        schedule=CalibrationSchedule(
            fast_interval=timedelta(hours=1),
            full_interval=timedelta(hours=2),
            error_retry_interval=timedelta(seconds=1),
        ),
        clock=clock,
    )
    service = BinanceReadonlyService(
        client=client,
        calibration=calibration,
        stream=stream,
        scheduler=scheduler,
        freshness=UserStreamFreshness(
            FreshnessPolicy(
                event_stale_after=None,
                calibration_stale_after=timedelta(hours=3),
            )
        ),
        clock=clock,
    )
    config = BinanceReadonlyRuntimeConfig(
        account_id="acct",
        managed_symbols=("BTCUSDT",),
        api_key="key",
        api_secret="secret",
    )
    runtime = BinanceReadonlyRuntime(
        config=config,
        transport=client.transport,
        client=client,
        listen_keys=listen_keys,
        stream=stream,
        calibration=calibration,
        scheduler=scheduler,
        service=service,
        clock=clock,
    )

    await runtime.start()
    try:
        await runtime.wait_until_ready(timeout_seconds=1)
        await wait_for(
            lambda: any(item.quantity == 1 for item in repository.positions)
        )
        assert repository.balance_snapshots
        assert client.bundle_calls >= 2

        client.positions = (client._position("1", 2000),)
        previous_bundle_calls = client.bundle_calls
        await runtime.request_reconnect()
        await wait_for(lambda: client.bundle_calls > previous_bundle_calls)
        await runtime.wait_until_ready(timeout_seconds=1)

        assert len(connector.connections) >= 2
        assert service.status.state.value == "READY"
    finally:
        await runtime.stop()

    assert client.transport.closed == 1
