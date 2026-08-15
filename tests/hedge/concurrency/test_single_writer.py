import multiprocessing as mp
import pytest

from tests.hedge.concurrency._spawn_lease_worker import acquire_worker

from freqtrade.hedge.concurrency import (
    InMemoryDatabaseLeaseStore,
    LeaseUnavailable,
    SingleWriterGuard,
    SqlAlchemyDatabaseLeaseStore,
    SqliteDatabaseLeaseStore,
)


def test_two_processes_cannot_both_be_writer(tmp_path) -> None:
    database_path = str(tmp_path / "lease.sqlite")
    SqliteDatabaseLeaseStore(database_path)
    context = mp.get_context("spawn")
    start = context.Event()
    release = context.Event()
    output = context.Queue()
    processes = [
        context.Process(
            target=acquire_worker,
            args=(database_path, f"writer-{index}", start, release, output),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    start.set()
    results = [output.get(timeout=10) for _ in processes]
    release.set()
    for process in processes:
        process.join(10)
    assert sum(1 for acquired, _token in results if acquired) == 1
    assert all(process.exitcode == 0 for process in processes)


def test_lost_lease_immediately_disables_new_risk() -> None:
    now = [1000]
    clock = lambda: now[0]
    store = InMemoryDatabaseLeaseStore()
    first = SingleWriterGuard(store, owner_id="first", ttl_ms=100, clock_ms=clock)
    second = SingleWriterGuard(store, owner_id="second", ttl_ms=100, clock_ms=clock)
    first.acquire()
    assert first.can_increase_risk()
    now[0] = 1200
    second.acquire()
    assert not first.can_increase_risk()


def test_same_owner_label_still_creates_only_one_writer_instance() -> None:
    store = InMemoryDatabaseLeaseStore()
    first = SingleWriterGuard(store, owner_id="worker", clock_ms=lambda: 1000)
    second = SingleWriterGuard(store, owner_id="worker", clock_ms=lambda: 1000)
    first.acquire()
    with pytest.raises(LeaseUnavailable):
        second.acquire()


def test_sqlalchemy_store_uses_fencing_on_sqlite(tmp_path) -> None:
    store = SqlAlchemyDatabaseLeaseStore(
        f"sqlite:///{tmp_path / 'sqlalchemy-lease.sqlite'}"
    )
    first = store.acquire(
        lease_name="writer",
        owner_id="first",
        now_ms=1000,
        ttl_ms=100,
    )
    assert first is not None
    assert store.acquire(
        lease_name="writer",
        owner_id="second",
        now_ms=1050,
        ttl_ms=100,
    ) is None
    second = store.acquire(
        lease_name="writer",
        owner_id="second",
        now_ms=1200,
        ttl_ms=100,
    )
    assert second is not None
    assert second.fencing_token == first.fencing_token + 1


def test_release_does_not_reset_fencing_token_in_memory() -> None:
    store = InMemoryDatabaseLeaseStore()
    first = SingleWriterGuard(store, owner_id="first", clock_ms=lambda: 1000)
    first_lease = first.acquire()
    assert first.release()
    second = SingleWriterGuard(store, owner_id="second", clock_ms=lambda: 1000)
    second_lease = second.acquire()
    assert second_lease.fencing_token == first_lease.fencing_token + 1


def test_release_does_not_reset_fencing_token_sqlite(tmp_path) -> None:
    store = SqliteDatabaseLeaseStore(tmp_path / "release-fencing.sqlite")
    first = store.acquire(lease_name="writer", owner_id="first", now_ms=1000, ttl_ms=100)
    assert first is not None
    assert store.release(lease=first)
    second = store.acquire(lease_name="writer", owner_id="second", now_ms=1000, ttl_ms=100)
    assert second is not None
    assert second.fencing_token == first.fencing_token + 1


def test_release_does_not_reset_fencing_token_sqlalchemy(tmp_path) -> None:
    store = SqlAlchemyDatabaseLeaseStore(
        f"sqlite:///{tmp_path / 'release-fencing-sa.sqlite'}"
    )
    first = store.acquire(lease_name="writer", owner_id="first", now_ms=1000, ttl_ms=100)
    assert first is not None
    assert store.release(lease=first)
    second = store.acquire(lease_name="writer", owner_id="second", now_ms=1000, ttl_ms=100)
    assert second is not None
    assert second.fencing_token == first.fencing_token + 1


def test_store_read_failure_fails_closed() -> None:
    class FailingReadStore(InMemoryDatabaseLeaseStore):
        fail_read = False

        def read(self, *, lease_name: str):
            if self.fail_read:
                raise OSError("database unavailable")
            return super().read(lease_name=lease_name)

    store = FailingReadStore()
    guard = SingleWriterGuard(store, owner_id="writer", clock_ms=lambda: 1000)
    guard.acquire()
    store.fail_read = True
    status = guard.status()
    assert not status.valid
    assert status.reason_code == "SINGLE_WRITER_STORE_UNAVAILABLE"


def test_sqlite_memory_database_is_rejected() -> None:
    with pytest.raises(ValueError, match="file-backed"):
        SqliteDatabaseLeaseStore(":memory:")


def test_lease_identity_validation_is_consistent() -> None:
    from freqtrade.hedge.concurrency.database_lease import LeaseRecord

    with pytest.raises(ValueError, match="owner_id must not exceed"):
        LeaseRecord("writer", "x" * 256, 1, 0, 0, 1)
    with pytest.raises(ValueError, match="owner_id must not be empty"):
        SingleWriterGuard(InMemoryDatabaseLeaseStore(), owner_id=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lease_name must not be empty"):
        SingleWriterGuard(
            InMemoryDatabaseLeaseStore(),
            owner_id="writer",
            lease_name=123,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="lease_name must not exceed"):
        SingleWriterGuard(
            InMemoryDatabaseLeaseStore(),
            owner_id="writer",
            lease_name="x" * 256,
        )


def test_sqlalchemy_memory_database_is_rejected() -> None:
    with pytest.raises(ValueError, match="file-backed"):
        SqlAlchemyDatabaseLeaseStore("sqlite:///:memory:")


def test_invalid_clock_fails_closed_and_clears_lease() -> None:
    now = [1000]
    store = InMemoryDatabaseLeaseStore()
    guard = SingleWriterGuard(store, owner_id="writer", clock_ms=lambda: now[0])
    guard.acquire()
    now[0] = -1
    status = guard.status()
    assert not status.valid
    assert status.reason_code == "SINGLE_WRITER_CLOCK_INVALID"
    assert guard.lease is None
