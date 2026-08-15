import asyncio
from dataclasses import replace

import pytest

from freqtrade.hedge.exchange.base import EventDisposition
from freqtrade.hedge.exchange.binance_user_stream import (
    BinanceUserStream,
    DefaultWebSocketConnector,
)
from freqtrade.hedge.exchange.rate_limit import BinanceDataError

from ._helpers import (
    FakeClock,
    FakeRepository,
    order,
    position,
)


class Keys:
    async def close(self):
        return None


def test_default_websocket_connector_applies_explicit_proxy(monkeypatch):
    calls = []

    class Connection:
        pass

    async def connect(url, **kwargs):
        calls.append((url, kwargs))
        return Connection()

    monkeypatch.setitem(
        __import__("sys").modules,
        "websockets",
        __import__("types").SimpleNamespace(connect=connect),
    )
    connector = DefaultWebSocketConnector(
        proxy_url="http://127.0.0.1:7897",
    )

    result = asyncio.run(connector.connect("wss://fstream.binance.com/ws/key"))

    assert isinstance(result, Connection)
    assert calls[0][1]["proxy"] == "http://127.0.0.1:7897"


def make_stream(repository, clock, faults):
    async def fault(reason, payload):
        faults.append(reason)

    return BinanceUserStream(
        account_id="acct",
        managed_symbols=["BTCUSDT"],
        repository=repository,
        listen_keys=Keys(),
        clock=clock,
        on_integrity_fault=fault,
    )


def order_event(
    *,
    event_time=1000,
    cumulative_fill="0",
    last_fill="0",
    trade_id="-1",
    execution="NEW",
    commission="0",
):
    return {
        "e": "ORDER_TRADE_UPDATE",
        "E": event_time,
        "T": event_time,
        "o": {
            "s": "BTCUSDT",
            "ps": "LONG",
            "i": 10,
            "c": "c10",
            "S": "BUY",
            "o": "LIMIT",
            "X": "NEW",
            "q": "1",
            "z": cumulative_fill,
            "ap": "0",
            "R": False,
            "x": execution,
            "t": trade_id,
            "l": last_fill,
            "L": "100",
            "N": "USDT",
            "n": commission,
            "rp": "0",
        },
    }


def account_event(symbol="BTCUSDT", quantity="1"):
    return {
        "e": "ACCOUNT_UPDATE",
        "E": 1000,
        "T": 1000,
        "a": {
            "m": "ORDER",
            "B": [],
            "P": [
                {
                    "s": symbol,
                    "pa": quantity,
                    "ep": "100",
                    "up": "0",
                    "mt": "cross",
                    "ps": "LONG",
                }
            ],
        },
    }


def test_duplicate_and_out_of_order_do_not_overwrite():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)

    assert asyncio.run(
        stream.process_message(order_event())
    ) is EventDisposition.APPLY
    assert asyncio.run(
        stream.process_message(order_event())
    ) is EventDisposition.DUPLICATE
    assert asyncio.run(
        stream.process_message(
            order_event(
                event_time=999,
                cumulative_fill="0.1",
                last_fill="0.1",
                trade_id="1",
                execution="TRADE",
            )
        )
    ) is EventDisposition.OUT_OF_ORDER

    assert len(repository.orders) == 1
    assert faults == ["ORDER_EVENT_SEQUENCE_REGRESSION"]


def test_cumulative_fill_gap_forces_reconciliation():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)
    stream.seed_from_rest(
        [],
        [order(clock, filled="0.1", update=1000)],
    )

    disposition = asyncio.run(
        stream.process_message(
            order_event(
                event_time=1001,
                cumulative_fill="0.5",
                last_fill="0.1",
                trade_id="2",
                execution="TRADE",
            )
        )
    )

    assert disposition is EventDisposition.GAP
    assert faults == ["ORDER_CUMULATIVE_FILL_GAP"]
    assert not repository.orders


def test_trade_event_is_idempotently_persisted():
    repository = FakeRepository()
    clock = FakeClock()
    stream = make_stream(repository, clock, [])
    payload = order_event(
        event_time=1001,
        cumulative_fill="0.1",
        last_fill="0.1",
        trade_id="2",
        execution="TRADE",
    )

    asyncio.run(stream.process_message(payload))
    asyncio.run(stream.process_message(payload))

    assert len(repository.fills) == 1
    assert len(repository.orders) == 1


def test_trade_commission_creates_fee_account_event():
    repository = FakeRepository()
    clock = FakeClock()
    stream = make_stream(repository, clock, [])

    asyncio.run(
        stream.process_message(
            order_event(
                event_time=1001,
                cumulative_fill="0.1",
                last_fill="0.1",
                trade_id="9",
                execution="TRADE",
                commission="0.02",
            )
        )
    )

    assert any(item.event_type == "FEE" for item in repository.events)


def test_unmanaged_active_order_forces_fault_before_persistence():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)
    payload = order_event()
    payload["o"]["s"] = "ETHUSDT"

    assert asyncio.run(
        stream.process_message(payload)
    ) is EventDisposition.GAP
    assert faults == ["UNMANAGED_ACTIVE_ORDER"]
    assert not repository.orders


def test_unmanaged_nonzero_position_forces_fault_before_persistence():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)

    assert asyncio.run(
        stream.process_message(account_event("ETHUSDT"))
    ) is EventDisposition.GAP
    assert faults == ["UNMANAGED_NONZERO_POSITION"]
    assert not repository.positions


def test_repository_failure_does_not_commit_dedupe_or_sequence_watermark():
    class FailingRepository(FakeRepository):
        def __init__(self):
            super().__init__()
            self.fail_once = True

        async def append_order_snapshots(
            self,
            facts,
            *,
            reconciliation_run_id=None,
        ):
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("temporary repository failure")
            await super().append_order_snapshots(
                facts,
                reconciliation_run_id=reconciliation_run_id,
            )

    repository = FailingRepository()
    clock = FakeClock()
    stream = make_stream(repository, clock, [])
    payload = order_event(
        event_time=1001,
        cumulative_fill="0.1",
        last_fill="0.1",
        trade_id="2",
        execution="TRADE",
    )

    with pytest.raises(RuntimeError, match="temporary"):
        asyncio.run(stream.process_message(payload))
    assert asyncio.run(
        stream.process_message(payload)
    ) is EventDisposition.APPLY
    assert len(repository.orders) == 1
    assert len(repository.fills) == 1


def test_invalid_utf8_fails_closed_and_is_audited():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)

    with pytest.raises(BinanceDataError, match="UTF-8"):
        asyncio.run(stream.process_message(b"\xff\xfe"))

    assert faults == ["INVALID_UTF8"]
    assert any(
        item.event_type == "STREAM_INTEGRITY_FAULT"
        for item in repository.events
    )


def test_account_update_hydrates_fields_omitted_by_websocket_from_rest_context():
    repository = FakeRepository()
    clock = FakeClock()
    stream = make_stream(repository, clock, [])
    context = position(clock, qty="1", update=900)
    stream.seed_from_rest([context], [])
    payload = account_event(quantity="2")
    payload["a"]["P"][0].update(ep="101", up="3")

    assert asyncio.run(
        stream.process_message(payload)
    ) is EventDisposition.APPLY

    persisted = repository.positions[-1]
    assert persisted.leverage == context.leverage
    assert persisted.mark_price == context.mark_price
    assert persisted.liquidation_price == context.liquidation_price


def test_account_update_without_rest_context_fails_closed():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)

    assert asyncio.run(
        stream.process_message(account_event())
    ) is EventDisposition.GAP
    assert faults == ["POSITION_CONTEXT_MISSING"]
    assert not repository.positions


def test_same_timestamp_monotonic_fill_update_is_applied():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)

    assert asyncio.run(
        stream.process_message(order_event())
    ) is EventDisposition.APPLY
    monotonic_fill = order_event(
        cumulative_fill="0.1",
        last_fill="0.1",
        trade_id="7",
        execution="TRADE",
    )
    assert asyncio.run(
        stream.process_message(monotonic_fill)
    ) is EventDisposition.APPLY
    assert faults == []
    assert len(repository.orders) == 2


def test_same_timestamp_non_monotonic_state_still_fails_closed():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)

    assert asyncio.run(
        stream.process_message(
            order_event(cumulative_fill="0.2", last_fill="0.2", trade_id="7", execution="TRADE")
        )
    ) is EventDisposition.APPLY
    regressive = order_event(
        cumulative_fill="0.1",
        last_fill="0",
        trade_id="8",
        execution="TRADE",
    )
    assert asyncio.run(
        stream.process_message(regressive)
    ) is EventDisposition.OUT_OF_ORDER
    assert faults == ["ORDER_CUMULATIVE_FILL_REGRESSION"]
    assert len(repository.orders) == 1


def test_rest_seed_rejects_unmanaged_positions_even_if_directly_called():
    repository = FakeRepository()
    clock = FakeClock()
    stream = make_stream(repository, clock, [])
    unmanaged = replace(
        position(clock, qty="1", update=1),
        symbol="ETHUSDT",
    )

    with pytest.raises(BinanceDataError, match="unmanaged position"):
        stream.seed_from_rest([unmanaged], [])


def test_rest_seed_ignores_unmanaged_flat_position_rows():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)
    flat_unmanaged = replace(
        position(clock, qty="0", update=1),
        symbol="ETHUSDT",
    )

    stream.seed_from_rest([flat_unmanaged], [])

    assert stream.health.gap_count == 0
    assert faults == []


def test_fee_event_is_retried_after_fill_was_already_persisted():
    class FailingEventRepository(FakeRepository):
        def __init__(self):
            super().__init__()
            self.fail_once = True

        async def append_account_events(self, facts):
            should_fail = self.fail_once and any(
                fact.event_type == "ORDER_TRADE_UPDATE"
                for fact in facts
            )
            if should_fail:
                self.fail_once = False
                raise RuntimeError("temporary account event failure")
            await super().append_account_events(facts)

    repository = FailingEventRepository()
    clock = FakeClock()
    stream = make_stream(repository, clock, [])
    payload = order_event(
        event_time=1001,
        cumulative_fill="0.1",
        last_fill="0.1",
        trade_id="12",
        execution="TRADE",
        commission="0.02",
    )

    with pytest.raises(RuntimeError, match="temporary account event failure"):
        asyncio.run(stream.process_message(payload))
    assert len(repository.fills) == 1

    assert asyncio.run(
        stream.process_message(payload)
    ) is EventDisposition.APPLY
    assert len(repository.fills) == 1
    assert any(item.event_type == "FEE" for item in repository.events)


def test_invalid_numeric_event_timestamp_is_rejected_without_name_error():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)
    payload = {"e": "UNKNOWN_EVENT", "E": "not-an-integer"}

    with pytest.raises(BinanceDataError, match="integer millisecond"):
        asyncio.run(stream.process_message(payload))
    assert repository.events == []
    assert faults == []


def test_unknown_event_without_timestamp_fails_closed():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)

    disposition = asyncio.run(stream.process_message({"e": "UNKNOWN_EVENT"}))

    assert disposition is EventDisposition.GAP
    assert faults == ["EVENT_TIMESTAMP_MISSING"]
    assert any(
        event.event_type == "STREAM_INTEGRITY_FAULT"
        for event in repository.events
    )


def test_user_stream_rejects_nonfinite_delays_and_unsafe_url():
    kwargs = {
        "account_id": "acct",
        "managed_symbols": ["BTCUSDT"],
        "repository": FakeRepository(),
        "listen_keys": Keys(),
        "clock": FakeClock(),
    }
    with pytest.raises(ValueError, match="finite and positive"):
        BinanceUserStream(
            **kwargs,
            min_reconnect_delay_seconds=float("nan"),
        )
    with pytest.raises(ValueError, match="WSS"):
        BinanceUserStream(
            **kwargs,
            websocket_base_url="ws://example.com/ws",
        )


def test_account_update_persists_balance_snapshot_from_user_stream():
    from decimal import Decimal

    from freqtrade.hedge.exchange.base import BalanceFact

    repository = FakeRepository()
    clock = FakeClock()
    stream = make_stream(repository, clock, [])
    previous = BalanceFact(
        account_id="acct",
        asset="USDT",
        wallet_balance=Decimal("100"),
        available_balance=Decimal("80"),
        cross_wallet_balance=Decimal("90"),
        unrealized_pnl=Decimal("2"),
        observed_at=clock.now(),
        source="BINANCE_REST",
        raw={},
    )
    stream.seed_from_rest([], [], [previous])
    payload = account_event()
    payload["a"]["P"] = []
    payload["a"]["B"] = [{"a": "USDT", "wb": "110", "cw": "95"}]

    disposition = asyncio.run(stream.process_message(payload))

    assert disposition is EventDisposition.APPLY
    assert len(repository.balance_snapshots) == 1
    observed = repository.balance_snapshots[0]
    assert observed.wallet_balance == Decimal("110")
    assert observed.cross_wallet_balance == Decimal("95")
    assert observed.available_balance == Decimal("80")
    assert observed.unrealized_pnl == Decimal("2")


def test_account_configuration_event_requests_rest_recalibration():
    repository = FakeRepository()
    clock = FakeClock()
    recalibrations = []

    async def recalibrate(reason, payload):
        recalibrations.append((reason, payload["e"]))

    stream = BinanceUserStream(
        account_id="acct",
        managed_symbols=["BTCUSDT"],
        repository=repository,
        listen_keys=Keys(),
        clock=clock,
        on_recalibration_required=recalibrate,
    )
    payload = {
        "e": "ACCOUNT_CONFIG_UPDATE",
        "E": 1000,
        "T": 1000,
        "ac": {"s": "BTCUSDT", "l": 10},
    }

    disposition = asyncio.run(stream.process_message(payload))

    assert disposition is EventDisposition.RECALIBRATE
    assert recalibrations == [("ACCOUNT_CONFIG_UPDATE", "ACCOUNT_CONFIG_UPDATE")]
    assert repository.events[-1].event_type == "ACCOUNT_CONFIG_UPDATE"


def test_reconnect_backoff_only_resets_after_stable_connection():
    class RecordingClock(FakeClock):
        def __init__(self):
            super().__init__()
            self.delays = []

        async def sleep(self, seconds):
            self.delays.append(seconds)
            self.advance(seconds)

    class Lease:
        listen_key = "key"
        generation = 1

    class RunningKeys(Keys):
        async def ensure(self):
            return Lease()

    class EmptyConnection:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self):
            return None

    class Connector:
        async def connect(self, url):
            return EmptyConnection()

    clock = RecordingClock()
    repository = FakeRepository()
    disconnected = 0
    holder = {}

    async def on_disconnected():
        nonlocal disconnected
        disconnected += 1
        if disconnected >= 3:
            holder["stream"]._stop_event.set()

    stream = BinanceUserStream(
        account_id="acct",
        managed_symbols=["BTCUSDT"],
        repository=repository,
        listen_keys=RunningKeys(),
        connector=Connector(),
        clock=clock,
        on_disconnected=on_disconnected,
        min_reconnect_delay_seconds=1,
        max_reconnect_delay_seconds=8,
        reconnect_reset_after_seconds=30,
    )
    holder["stream"] = stream

    asyncio.run(stream.run())

    assert clock.delays == [1, 2]


def test_queued_order_event_covered_by_rest_baseline_is_ignored():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)
    stream.seed_from_rest(
        [],
        [order(clock, filled="0.5", update=2000)],
    )

    disposition = asyncio.run(
        stream.process_message(
            order_event(
                event_time=1000,
                cumulative_fill="0.1",
                last_fill="0.1",
                trade_id="2",
                execution="TRADE",
            )
        )
    )

    assert disposition is EventDisposition.DUPLICATE
    assert faults == []
    assert repository.orders == []
    assert repository.fills == []


def test_queued_position_event_covered_by_rest_baseline_does_not_disconnect():
    repository = FakeRepository()
    clock = FakeClock()
    faults = []
    stream = make_stream(repository, clock, faults)
    stream.seed_from_rest([position(clock, qty="2", update=2000)], [])
    payload = account_event(quantity="1")

    disposition = asyncio.run(stream.process_message(payload))

    assert disposition is EventDisposition.APPLY
    assert faults == []
    assert repository.positions == []
    assert repository.events[-1].event_type == "BALANCE"


def test_account_update_exposes_single_balance_change_as_accounting_fact():
    from decimal import Decimal

    repository = FakeRepository()
    clock = FakeClock()
    stream = make_stream(repository, clock, [])
    payload = account_event()
    payload["a"]["m"] = "FUNDING_FEE"
    payload["a"]["P"] = []
    payload["a"]["B"] = [
        {"a": "USDT", "wb": "1000.25", "cw": "1000.25", "bc": "0.25"}
    ]

    assert asyncio.run(stream.process_message(payload)) is EventDisposition.APPLY

    event = repository.events[-1]
    assert event.event_type == "FUNDING"
    assert event.currency == "USDT"
    assert event.amount == Decimal("0.25")
    assert event.economic_event_id


def test_stream_exposes_deterministic_latest_account_state():
    from decimal import Decimal

    from freqtrade.hedge.exchange.base import BalanceFact

    repository = FakeRepository()
    clock = FakeClock()
    stream = make_stream(repository, clock, [])
    initial_position = position(clock, qty="1", update=1000)
    initial_order = order(clock, oid="10", update=1000)
    initial_balance = BalanceFact(
        account_id="acct",
        asset="USDT",
        wallet_balance=Decimal("100"),
        available_balance=Decimal("90"),
        cross_wallet_balance=Decimal("100"),
        unrealized_pnl=Decimal("0"),
        observed_at=clock.now(),
        source="BINANCE_REST",
        raw={},
    )

    stream.seed_from_rest([initial_position], [initial_order], [initial_balance])
    assert stream.state_revision == 1
    assert stream.current_positions == (initial_position,)
    assert stream.current_active_orders == (initial_order,)
    assert stream.current_balances == (initial_balance,)

    update = order_event(event_time=1001)
    update["o"]["X"] = "CANCELED"
    asyncio.run(stream.process_message(update))

    assert stream.state_revision == 2
    assert stream.current_active_orders == ()
    assert stream.state_observed_at == clock.now()
