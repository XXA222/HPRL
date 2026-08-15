import asyncio

import pytest

from freqtrade.hedge.exchange.base import CalibrationKind
from freqtrade.hedge.exchange.binance_readonly import BinanceAccountBundle
from freqtrade.hedge.exchange.rate_limit import BinanceDataError
from freqtrade.hedge.readonly.calibration import (
    ReadonlyCalibration,
    ReadonlySafetyHalt,
)

from ._helpers import (
    FakeClock,
    FakeRepository,
    fill,
    order,
    position,
    snapshot,
)


class Client:
    account_id = "acct"

    def __init__(self, account_bundle):
        self.bundle = account_bundle
        self.calls = []
        self.clock_syncs = 0

    async def synchronize_clock(self):
        self.clock_syncs += 1

    async def fetch_bundle(self, **kwargs):
        self.calls.append(kwargs)
        return self.bundle


def bundle(
    clock,
    positions=(),
    orders=(),
    fills=(),
    income=(),
):
    return BinanceAccountBundle(
        snapshot(clock),
        (),
        tuple(positions),
        tuple(orders),
        tuple(fills),
        None,
        tuple(income),
        clock.now(),
        clock.now(),
    )


def calibration(client, repository, clock):
    return ReadonlyCalibration(
        client=client,
        repository=repository,
        managed_symbols=["BTCUSDT"],
        clock=clock,
    )


def test_unmanaged_nonzero_position_halts_and_is_audited():
    clock = FakeClock()
    repository = FakeRepository()
    client = Client(
        bundle(
            clock,
            [position(clock, symbol="ETHUSDT")],
        )
    )
    service = calibration(client, repository, clock)

    with pytest.raises(ReadonlySafetyHalt) as caught:
        asyncio.run(service.run(CalibrationKind.FULL))

    assert caught.value.reason == "UNMANAGED_POSITION"
    assert repository.runs["run-1"]["status"] == "HALT"
    assert repository.diffs[0].reason_code == "UNMANAGED_POSITION"


def test_unmanaged_active_order_halts():
    clock = FakeClock()
    repository = FakeRepository()
    client = Client(
        bundle(
            clock,
            orders=[order(clock, symbol="ETHUSDT")],
        )
    )
    service = calibration(client, repository, clock)

    with pytest.raises(ReadonlySafetyHalt):
        asyncio.run(service.run(CalibrationKind.FAST))
    assert repository.runs["run-1"]["status"] == "HALT"


def test_full_calibration_persists_rest_facts_and_income():
    clock = FakeClock()
    repository = FakeRepository()
    position_fact = position(clock)
    order_fact = order(clock)
    fill_fact = fill(clock)
    repository.local_positions = [position_fact]
    repository.local_orders = [order_fact]
    income = (
        {
            "incomeType": "FUNDING_FEE",
            "income": "1",
            "time": 1000,
            "tranId": 1,
        },
    )
    client = Client(
        bundle(
            clock,
            [position_fact],
            [order_fact],
            [fill_fact],
            income,
        )
    )
    service = calibration(client, repository, clock)

    result = asyncio.run(service.run(CalibrationKind.FULL))

    assert result.consistent
    assert len(repository.fills) == 1
    assert any(
        event.event_type == "FUNDING_FEE"
        for event in repository.events
    )
    assert client.calls[0]["include_fills"] is True


def test_position_drift_is_not_silently_overwritten():
    clock = FakeClock()
    repository = FakeRepository()
    repository.local_positions = [position(clock, qty="1")]
    client = Client(bundle(clock, [position(clock, qty="2")]))
    service = calibration(client, repository, clock)

    result = asyncio.run(service.run(CalibrationKind.FAST))

    assert not result.consistent
    assert result.diff_count == 1
    assert repository.diffs[0].reason_code == "POSITION_QUANTITY_MISMATCH"


def test_duplicate_position_facts_fail_reconciliation_instead_of_overwriting():
    clock = FakeClock()
    repository = FakeRepository()
    duplicate_bundle = bundle(
        clock,
        [
            position(clock, qty="1"),
            position(clock, qty="2"),
        ],
    )
    service = calibration(Client(duplicate_bundle), repository, clock)

    with pytest.raises(BinanceDataError, match="Duplicate active position"):
        asyncio.run(service.run(CalibrationKind.FAST))
    assert repository.runs["run-1"]["status"] == "FAILED"


def test_full_and_reconnect_refresh_clock_but_fast_does_not():
    clock = FakeClock()
    repository = FakeRepository()
    client = Client(bundle(clock))
    service = calibration(client, repository, clock)

    asyncio.run(service.run(CalibrationKind.FAST))
    assert client.clock_syncs == 0

    asyncio.run(service.run(CalibrationKind.FULL))
    assert client.clock_syncs == 1

    asyncio.run(service.run(CalibrationKind.RECONNECT))
    assert client.clock_syncs == 2


class DoubleFailureRepository(FakeRepository):
    async def append_account_snapshot(
        self,
        fact,
        *,
        reconciliation_run_id=None,
    ) -> None:
        raise ValueError("primary persistence failure")

    async def complete_reconciliation(
        self,
        run_id,
        *,
        completed_at,
        status,
        reason,
    ) -> None:
        raise RuntimeError("failure audit unavailable")


def test_failure_audit_error_does_not_mask_primary_calibration_failure():
    clock = FakeClock()
    repository = DoubleFailureRepository()
    service = calibration(Client(bundle(clock)), repository, clock)

    with pytest.raises(ValueError, match="primary persistence failure"):
        asyncio.run(service.run(CalibrationKind.FAST))


def test_history_cursor_reduces_repeated_full_history_window():
    class CursorRepository(FakeRepository):
        def __init__(self):
            super().__init__()
            self.cursor = None

        async def load_history_cursor(self, account_id, cursor_name):
            return self.cursor

        async def save_history_cursor(self, account_id, cursor_name, cursor_ms):
            self.cursor = cursor_ms

    clock = FakeClock()
    repository = CursorRepository()
    client = Client(bundle(clock))
    service = calibration(client, repository, clock)

    asyncio.run(service.run(CalibrationKind.FULL))
    first_start = client.calls[-1]["fill_start_time_ms"]
    first_cursor = repository.cursor
    clock.advance(600)
    asyncio.run(service.run(CalibrationKind.FULL))
    second_start = client.calls[-1]["fill_start_time_ms"]

    assert first_cursor is not None
    assert second_start == first_cursor - 5 * 60 * 1000
    assert second_start > first_start


def test_old_persistent_cursor_is_used_instead_of_silent_72h_truncation():
    class CursorRepository(FakeRepository):
        async def load_history_cursor(self, account_id, cursor_name):
            return int((clock.now().timestamp() - 10 * 24 * 3600) * 1000)

        async def save_history_cursor(self, account_id, cursor_name, cursor_ms):
            return None

    clock = FakeClock()
    repository = CursorRepository()
    client = Client(bundle(clock))
    service = ReadonlyCalibration(
        client=client,
        repository=repository,
        managed_symbols=["BTC/USDT:USDT"],
        clock=clock,
        max_history_backfill=None,
    )

    asyncio.run(service.run(CalibrationKind.FULL))

    expected = int((clock.now().timestamp() - 10 * 24 * 3600) * 1000) - 5 * 60 * 1000
    assert client.calls[0]["fill_start_time_ms"] == expected


def test_excessive_history_gap_requires_explicit_backfill():
    from datetime import timedelta

    from freqtrade.hedge.readonly.calibration import HistoryBackfillRequired

    class CursorRepository(FakeRepository):
        async def load_history_cursor(self, account_id, cursor_name):
            return int((clock.now().timestamp() - 40 * 24 * 3600) * 1000)

    clock = FakeClock()
    service = ReadonlyCalibration(
        client=Client(bundle(clock)),
        repository=CursorRepository(),
        managed_symbols=["BTCUSDT"],
        clock=clock,
        max_history_backfill=timedelta(days=30),
    )

    with pytest.raises(HistoryBackfillRequired, match="HISTORY_GAP_REQUIRES_BACKFILL"):
        asyncio.run(service.run(CalibrationKind.FULL))


def test_atomic_batch_repository_is_used_for_rest_fact_ingestion():
    class AtomicRepository(FakeRepository):
        def __init__(self):
            super().__init__()
            self.batches = []

        async def append_exchange_fact_batch(self, batch):
            self.batches.append(batch)
            self.positions.extend(batch.positions)
            self.orders.extend(batch.orders)
            self.fills.extend(batch.fills)
            self.events.extend(batch.account_events)
            self.diffs.extend(batch.reconciliation_diffs)

        async def append_account_snapshot(self, *args, **kwargs):
            raise AssertionError("non-atomic account snapshot write used")

    clock = FakeClock()
    repository = AtomicRepository()
    position_fact = position(clock)
    order_fact = order(clock)
    repository.local_positions = [position_fact]
    repository.local_orders = [order_fact]
    service = calibration(
        Client(bundle(clock, [position_fact], [order_fact], [fill(clock)])),
        repository,
        clock,
    )

    result = asyncio.run(service.run(CalibrationKind.FULL))

    assert result.consistent
    assert len(repository.batches) == 1
    batch = repository.batches[0]
    assert batch.reconciliation_run_id == "run-1"
    assert len(batch.positions) == 1
    assert len(batch.orders) == 1
    assert len(batch.fills) == 1
