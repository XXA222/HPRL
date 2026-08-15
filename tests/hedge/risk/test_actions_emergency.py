from decimal import Decimal

from freqtrade.enums.hedge import PositionAction, PositionSide
from freqtrade.hedge.concurrency import (
    InMemoryDatabaseLeaseStore,
    PositionLockManager,
    SingleWriterGuard,
)
from freqtrade.hedge.risk import (
    EmergencyReduceOnlyController,
    RiskActionStateMachine,
    RiskEvent,
    RiskMode,
)


def test_risk_action_state_machine_requires_ack_after_halt() -> None:
    machine = RiskActionStateMachine()
    assert (
        machine.transition(
            RiskEvent.ENTER_HALT,
            reason_codes=("CRITICAL",),
        ).mode
        is RiskMode.HALT
    )
    assert machine.transition(RiskEvent.RECOVERED).mode is RiskMode.HALT
    assert machine.transition(RiskEvent.OPERATOR_ACK).mode is RiskMode.REDUCE_ONLY
    assert machine.transition(RiskEvent.RECOVERED).mode is RiskMode.NORMAL


def test_emergency_mode_forbids_increase_and_clips_reduce() -> None:
    store = InMemoryDatabaseLeaseStore()
    writer = SingleWriterGuard(store, owner_id="writer", clock_ms=lambda: 1000)
    writer.acquire()
    locks = PositionLockManager()
    machine = RiskActionStateMachine()
    machine.transition(RiskEvent.ENTER_REDUCE_ONLY, reason_codes=("MARGIN_HIGH",))
    controller = EmergencyReduceOnlyController(locks=locks, writer=writer, state_machine=machine)

    rejected = controller.approve(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("1"),
        confirmed_quantity=Decimal("10"),
    )
    assert not rejected.allowed

    approved = controller.approve(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=PositionAction.REDUCE,
        requested_quantity=Decimal("12"),
        confirmed_quantity=Decimal("10"),
    )
    assert approved.allowed
    assert approved.approved_quantity == Decimal("10")
    assert approved.approved_quantity <= Decimal("12")
    assert approved.reservation is not None
    approved.reservation.release()


def _ready_gate():
    from freqtrade.hedge.readiness import ReadinessGate, ReadinessInputs

    gate = ReadinessGate(clock_ms=lambda: 1000)
    gate.evaluate(
        ReadinessInputs(
            database_migration_succeeded=True,
            single_writer_lease_valid=True,
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
    )
    return gate


def _risk_account():
    from freqtrade.hedge.risk import AccountRiskSnapshot

    return AccountRiskSnapshot(
        account_id="main",
        equity=Decimal("10000"),
        wallet_balance=Decimal("10000"),
        available_balance=Decimal("8000"),
        initial_margin=Decimal("1000"),
        maintenance_margin=Decimal("100"),
        gross_long_notional=Decimal("3000"),
        gross_short_notional=Decimal("1000"),
        net_notional=Decimal("2000"),
        liquidation_buffer_ratio=Decimal("0.5"),
    )


def _approval_coordinator(*, clock):
    from freqtrade.hedge.risk import (
        HedgeRiskEngine,
        RiskApprovalCoordinator,
        RiskLimits,
    )

    store = InMemoryDatabaseLeaseStore()
    writer = SingleWriterGuard(store, owner_id="writer", ttl_ms=100, clock_ms=clock)
    writer.acquire()
    readiness = _ready_gate()
    coordinator = RiskApprovalCoordinator(
        engine=HedgeRiskEngine(
            RiskLimits(
                max_margin_utilization=Decimal("0.55"),
                min_liquidation_buffer_ratio=Decimal("0.2"),
                max_gross_notional=Decimal("5000"),
                max_gross_exposure_ratio=Decimal("0.8"),
                max_leg_notional=Decimal("5000"),
                max_symbol_gross_notional=Decimal("5000"),
            )
        ),
        locks=PositionLockManager(),
        writer=writer,
        readiness=readiness,
        state_machine=RiskActionStateMachine(),
    )
    return store, writer, coordinator, readiness


def test_unified_coordinator_rechecks_lease_before_new_risk() -> None:
    from freqtrade.hedge.risk import RiskRequest

    now = [1000]
    store, _writer, coordinator, _readiness = _approval_coordinator(clock=lambda: now[0])
    now[0] = 1200
    takeover = SingleWriterGuard(store, owner_id="takeover", ttl_ms=100, clock_ms=lambda: now[0])
    takeover.acquire()
    approval = coordinator.approve(
        request=RiskRequest(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            action=PositionAction.INCREASE,
            requested_quantity=Decimal("0.5"),
            reference_price=Decimal("1000"),
            current_leg_notional=Decimal("3000"),
            current_symbol_gross_notional=Decimal("4000"),
        ),
        account=_risk_account(),
    )
    assert not approval.allowed
    assert approval.reason_codes == ("SINGLE_WRITER_LEASE_INVALID",)


def test_unified_coordinator_reserves_local_increase_capacity() -> None:
    from freqtrade.hedge.risk import RiskRequest

    _store, _writer, coordinator, _readiness = _approval_coordinator(clock=lambda: 1000)
    request = RiskRequest(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("0.8"),
        reference_price=Decimal("1000"),
        current_leg_notional=Decimal("3000"),
        current_symbol_gross_notional=Decimal("4000"),
    )
    first = coordinator.approve(request=request, account=_risk_account())
    second = coordinator.approve(request=request, account=_risk_account())
    assert first.allowed and second.allowed
    assert first.approved_quantity == Decimal("0.8")
    assert second.approved_quantity == Decimal("0.2")
    assert coordinator.pending_increase_notional(account_id="main") == Decimal("1000.0")
    assert first.reservation is not None
    assert second.reservation is not None
    first.reservation.release()
    second.reservation.release()
    assert coordinator.pending_increase_notional(account_id="main") == 0


def test_unified_coordinator_allows_not_ready_controlled_reduce() -> None:
    from freqtrade.hedge.readiness import ReadinessInputs
    from freqtrade.hedge.risk import RiskRequest

    _store, writer, coordinator, readiness = _approval_coordinator(clock=lambda: 1000)
    readiness.evaluate(
        ReadinessInputs(
            database_migration_succeeded=True,
            single_writer_lease_valid=True,
            position_mode="hedge",
            margin_mode="cross",
            configured_leverage=Decimal("3"),
            observed_leverages=(Decimal("3"),),
            unmanaged_position_count=0,
            unmanaged_order_count=0,
            rest_snapshot_valid=True,
            user_stream_fresh=False,
            unknown_order_count=0,
            reconciliation_converged=True,
            risk_data_valid=True,
        )
    )
    approval = coordinator.approve(
        request=RiskRequest(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.SHORT,
            action=PositionAction.REDUCE,
            requested_quantity=Decimal("12"),
            reference_price=Decimal("1000"),
            confirmed_quantity=Decimal("10"),
        ),
        account=_risk_account(),
    )
    assert writer.can_increase_risk()
    assert approval.allowed
    assert approval.approved_quantity == Decimal("10")
    assert approval.approved_quantity <= Decimal("12")
    assert approval.reservation is not None
    approval.reservation.release()


def test_allowed_approvals_carry_fencing_token() -> None:
    from freqtrade.hedge.risk import RiskRequest

    _store, _writer, coordinator, _readiness = _approval_coordinator(clock=lambda: 1000)
    approval = coordinator.approve(
        request=RiskRequest(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            action=PositionAction.INCREASE,
            requested_quantity=Decimal("0.25"),
            reference_price=Decimal("1000"),
            current_leg_notional=Decimal("3000"),
            current_symbol_gross_notional=Decimal("4000"),
        ),
        account=_risk_account(),
    )
    assert approval.allowed
    assert approval.fencing_token == 1
    assert approval.lease_expires_at_ms == 1100
    assert approval.reservation is not None
    approval.reservation.release()


def test_increase_reservation_context_releases_on_normal_exit() -> None:
    from freqtrade.hedge.risk import RiskRequest

    _store, _writer, coordinator, _readiness = _approval_coordinator(clock=lambda: 1000)
    approval = coordinator.approve(
        request=RiskRequest(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            action=PositionAction.INCREASE,
            requested_quantity=Decimal("0.25"),
            reference_price=Decimal("1000"),
            current_leg_notional=Decimal("3000"),
            current_symbol_gross_notional=Decimal("4000"),
        ),
        account=_risk_account(),
    )
    assert approval.reservation is not None
    with approval.reservation:
        assert coordinator.pending_increase_notional(account_id="main") == Decimal("250.00")
    assert coordinator.pending_increase_notional(account_id="main") == 0


def test_unified_coordinator_rechecks_lease_after_engine_evaluation() -> None:
    from freqtrade.hedge.risk import RiskRequest

    now = [1000]
    _store, _writer, coordinator, _readiness = _approval_coordinator(clock=lambda: now[0])
    original_evaluate = coordinator._engine.evaluate_request

    def evaluate_and_expire(*, request, account):
        decision = original_evaluate(request=request, account=account)
        now[0] = 1200
        return decision

    coordinator._engine.evaluate_request = evaluate_and_expire  # type: ignore[method-assign]
    approval = coordinator.approve(
        request=RiskRequest(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            action=PositionAction.INCREASE,
            requested_quantity=Decimal("0.25"),
            reference_price=Decimal("1000"),
            current_leg_notional=Decimal("3000"),
            current_symbol_gross_notional=Decimal("4000"),
        ),
        account=_risk_account(),
    )
    assert not approval.allowed
    assert approval.reason_codes == ("SINGLE_WRITER_LEASE_INVALID",)
    assert coordinator.pending_increase_notional(account_id="main") == 0


def test_reduce_rechecks_lease_after_waiting_for_position_lock() -> None:
    import threading
    import time

    from freqtrade.hedge.risk import RiskRequest

    now = [1000]
    _store, _writer, coordinator, _readiness = _approval_coordinator(clock=lambda: now[0])
    started = threading.Event()
    finished = threading.Event()
    result = []

    def approve_reduce() -> None:
        started.set()
        result.append(
            coordinator.approve(
                request=RiskRequest(
                    account_id="main",
                    symbol="ETH/USDT:USDT",
                    position_side=PositionSide.LONG,
                    action=PositionAction.REDUCE,
                    requested_quantity=Decimal("1"),
                    reference_price=Decimal("1000"),
                    confirmed_quantity=Decimal("10"),
                ),
                account=_risk_account(),
                timeout_seconds=1,
            )
        )
        finished.set()

    with coordinator._locks.lock(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
    ):
        thread = threading.Thread(target=approve_reduce)
        thread.start()
        assert started.wait(1)
        time.sleep(0.05)
        now[0] = 1200
    assert finished.wait(1)
    thread.join(timeout=1)
    assert len(result) == 1
    assert not result[0].allowed
    assert result[0].reason_codes == ("SINGLE_WRITER_LEASE_INVALID",)


def test_emergency_approval_carries_fencing_token() -> None:
    store = InMemoryDatabaseLeaseStore()
    writer = SingleWriterGuard(store, owner_id="writer", clock_ms=lambda: 1000)
    writer.acquire()
    machine = RiskActionStateMachine()
    machine.transition(RiskEvent.ENTER_REDUCE_ONLY, reason_codes=("MARGIN_HIGH",))
    controller = EmergencyReduceOnlyController(
        locks=PositionLockManager(),
        writer=writer,
        state_machine=machine,
    )
    approval = controller.approve(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=PositionAction.REDUCE,
        requested_quantity=Decimal("1"),
        confirmed_quantity=Decimal("10"),
    )
    assert approval.allowed
    assert approval.fencing_token == 1
    assert approval.lease_expires_at_ms is not None
    assert approval.reservation is not None
    approval.reservation.release()


def test_increase_rechecks_readiness_after_engine_evaluation() -> None:
    from freqtrade.hedge.readiness import ReadinessInputs
    from freqtrade.hedge.risk import RiskRequest

    _store, _writer, coordinator, readiness = _approval_coordinator(clock=lambda: 1000)
    original_evaluate = coordinator._engine.evaluate_request

    def evaluate_and_make_stale(*, request, account):
        decision = original_evaluate(request=request, account=account)
        readiness.evaluate(
            ReadinessInputs(
                database_migration_succeeded=True,
                single_writer_lease_valid=True,
                position_mode="hedge",
                margin_mode="cross",
                configured_leverage=Decimal("3"),
                observed_leverages=(Decimal("3"),),
                unmanaged_position_count=0,
                unmanaged_order_count=0,
                rest_snapshot_valid=True,
                user_stream_fresh=False,
                unknown_order_count=0,
                reconciliation_converged=True,
                risk_data_valid=True,
            )
        )
        return decision

    coordinator._engine.evaluate_request = evaluate_and_make_stale  # type: ignore[method-assign]
    approval = coordinator.approve(
        request=RiskRequest(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            action=PositionAction.INCREASE,
            requested_quantity=Decimal("0.25"),
            reference_price=Decimal("1000"),
            current_leg_notional=Decimal("3000"),
            current_symbol_gross_notional=Decimal("4000"),
        ),
        account=_risk_account(),
    )
    assert not approval.allowed
    assert approval.reason_codes[:2] == ("READINESS_NOT_READY", "USER_STREAM_STALE")
    assert coordinator.pending_increase_notional(account_id="main") == 0


def test_coordinator_counts_recovered_pending_reduce_orders() -> None:
    from freqtrade.hedge.risk import RiskRequest

    _store, _writer, coordinator, _readiness = _approval_coordinator(clock=lambda: 1000)
    approval = coordinator.approve(
        request=RiskRequest(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            action=PositionAction.REDUCE,
            requested_quantity=Decimal("5"),
            reference_price=Decimal("1000"),
            confirmed_quantity=Decimal("10"),
            pending_reduce_quantity=Decimal("8"),
        ),
        account=_risk_account(),
    )
    assert approval.allowed
    assert approval.approved_quantity == Decimal("2")
    assert approval.reservation is not None
    approval.reservation.release()


def test_reduce_rejects_cross_account_or_invalid_risk_snapshot() -> None:
    from dataclasses import replace

    from freqtrade.hedge.risk import RiskRequest

    _store, _writer, coordinator, _readiness = _approval_coordinator(clock=lambda: 1000)
    request = RiskRequest(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=PositionAction.REDUCE,
        requested_quantity=Decimal("1"),
        reference_price=Decimal("1000"),
        confirmed_quantity=Decimal("10"),
    )
    mismatch = coordinator.approve(
        request=request,
        account=replace(_risk_account(), account_id="other"),
    )
    assert not mismatch.allowed
    assert mismatch.reason_codes == ("ACCOUNT_ID_MISMATCH",)

    invalid = coordinator.approve(
        request=request,
        account=replace(_risk_account(), risk_data_valid=False),
    )
    assert not invalid.allowed
    assert invalid.reason_codes == ("RISK_DATA_INVALID",)


def test_state_machine_rejects_string_reason_container() -> None:
    import pytest

    machine = RiskActionStateMachine()
    with pytest.raises(ValueError, match="reason_codes must be a tuple"):
        machine.transition(
            RiskEvent.ENTER_HALT,
            reason_codes="CRITICAL",  # type: ignore[arg-type]
        )


def test_coordinator_converts_position_lock_timeout_to_stable_denial() -> None:
    import threading

    from freqtrade.hedge.risk import RiskRequest

    _store, _writer, coordinator, _readiness = _approval_coordinator(clock=lambda: 1000)
    entered = threading.Event()
    release = threading.Event()

    def holder() -> None:
        with coordinator._locks.lock(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
        ):
            entered.set()
            release.wait(2)

    thread = threading.Thread(target=holder)
    thread.start()
    assert entered.wait(1)
    try:
        approval = coordinator.approve(
            request=RiskRequest(
                account_id="main",
                symbol="ETH/USDT:USDT",
                position_side=PositionSide.LONG,
                action=PositionAction.REDUCE,
                requested_quantity=Decimal("1"),
                reference_price=Decimal("1000"),
                confirmed_quantity=Decimal("10"),
            ),
            account=_risk_account(),
            timeout_seconds=0.01,
        )
    finally:
        release.set()
        thread.join(2)
    assert not approval.allowed
    assert approval.reason_codes == ("POSITION_LOCK_TIMEOUT",)


def test_restrictive_risk_transition_requires_audit_reason() -> None:
    import pytest

    machine = RiskActionStateMachine()
    with pytest.raises(ValueError, match="requires at least one reason code"):
        machine.transition(RiskEvent.ENTER_HALT)
    with pytest.raises(ValueError, match="requires at least one reason code"):
        machine.transition(RiskEvent.ENTER_REDUCE_ONLY)
