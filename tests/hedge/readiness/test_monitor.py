from decimal import Decimal

from freqtrade.hedge.concurrency import InMemoryDatabaseLeaseStore, SingleWriterGuard
from freqtrade.hedge.readiness import ReadinessGate, ReadinessInputs, ReadinessMonitor


def _inputs():
    return ReadinessInputs(
        database_migration_succeeded=True,
        single_writer_lease_valid=False,
        position_mode="hedge",
        margin_mode="cross",
        configured_leverage=Decimal("3"),
        observed_leverages=(Decimal("3"),),
        unmanaged_position_count=0,
        unmanaged_order_count=0,
        rest_snapshot_valid=True,
        user_stream_fresh=True,
        unknown_order_count=0,
        reconciliation_converged=True,
        risk_data_valid=True,
    )


def test_monitor_binds_writer_and_updates_halt_reasons() -> None:
    guard = SingleWriterGuard(
        InMemoryDatabaseLeaseStore(),
        owner_id="writer",
        clock_ms=lambda: 1000,
    )
    gate = ReadinessGate(clock_ms=lambda: 1000)
    monitor = ReadinessMonitor(gate=gate, inputs=_inputs(), writer=guard)
    assert not monitor.refresh().ready
    guard.acquire()
    assert monitor.refresh().ready
    assert monitor.set_halt_reason("MANUAL_HALT").state.value == "HALT"
    assert monitor.clear_halt_reason("MANUAL_HALT").ready
    snapshot = monitor.snapshot()
    assert snapshot["report"]["ready"] is True
    assert snapshot["inputs"]["single_writer_lease_valid"] is True


def test_snapshot_refreshes_writer_fact_and_report_together() -> None:
    now = [1000]
    store = InMemoryDatabaseLeaseStore()
    writer = SingleWriterGuard(store, owner_id="writer", ttl_ms=100, clock_ms=lambda: now[0])
    gate = ReadinessGate(clock_ms=lambda: now[0])
    monitor = ReadinessMonitor(gate=gate, inputs=_inputs(), writer=writer)
    writer.acquire()
    snapshot = monitor.snapshot()
    assert snapshot["inputs"]["single_writer_lease_valid"] is True
    assert snapshot["report"]["state"] == "READY"
