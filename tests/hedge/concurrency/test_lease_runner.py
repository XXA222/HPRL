from freqtrade.hedge.concurrency import (
    InMemoryDatabaseLeaseStore,
    LeaseRunnerState,
    SingleWriterGuard,
    SingleWriterLeaseRunner,
)


def test_lease_runner_acquires_and_renews_deterministically() -> None:
    now = [1000]
    guard = SingleWriterGuard(
        InMemoryDatabaseLeaseStore(),
        owner_id="writer",
        ttl_ms=300,
        clock_ms=lambda: now[0],
    )
    runner = SingleWriterLeaseRunner(guard, interval_seconds=0.05)
    first = runner.run_once()
    assert first.active
    assert first.state is LeaseRunnerState.ACTIVE
    assert first.acquisition_count == 1
    now[0] = 1100
    second = runner.run_once()
    assert second.active
    assert second.renewal_count == 1
    stopped = runner.stop()
    assert stopped.state is LeaseRunnerState.STOPPED
    assert not stopped.writer.valid


def test_lease_runner_reports_loss_and_can_reacquire() -> None:
    now = [1000]
    store = InMemoryDatabaseLeaseStore()
    guard = SingleWriterGuard(store, owner_id="writer", ttl_ms=100, clock_ms=lambda: now[0])
    runner = SingleWriterLeaseRunner(guard, interval_seconds=0.02)
    assert runner.run_once().active
    now[0] = 1200
    takeover = SingleWriterGuard(store, owner_id="other", ttl_ms=100, clock_ms=lambda: now[0])
    takeover.acquire()
    lost = runner.run_once()
    assert not lost.active
    assert lost.state is LeaseRunnerState.LOST
    now[0] = 1400
    reacquired = runner.run_once()
    assert reacquired.active
    assert reacquired.acquisition_count == 2


def test_lost_callback_only_fires_on_state_transition() -> None:
    now = [1000]
    store = InMemoryDatabaseLeaseStore()
    blocker = SingleWriterGuard(store, owner_id="blocker", ttl_ms=100, clock_ms=lambda: now[0])
    blocker.acquire()
    guard = SingleWriterGuard(store, owner_id="candidate", ttl_ms=100, clock_ms=lambda: now[0])
    lost = []
    active = []
    runner = SingleWriterLeaseRunner(
        guard,
        interval_seconds=0.01,
        on_lost=lambda status: lost.append(status.state),
        on_active=lambda status: active.append(status.state),
    )
    assert not runner.run_once().active
    assert not runner.run_once().active
    assert len(lost) == 1
    now[0] = 1200
    assert runner.run_once().active
    assert len(active) == 1
