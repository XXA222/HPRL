from decimal import Decimal

from freqtrade.enums.hedge import PositionAction, PositionSide
from freqtrade.hedge.concurrency import (
    InMemoryDatabaseLeaseStore,
    PositionLockKey,
    PositionLockManager,
    SingleWriterGuard,
)
from freqtrade.hedge.readiness import ReadinessGate, ReadinessInputs, ReadinessMonitor
from freqtrade.hedge.risk import (
    AccountRiskFacts,
    HedgeRiskEngine,
    HedgeRiskRuntime,
    OrderRiskIntent,
    PendingOrderRisk,
    PositionRiskLeg,
    RiskActionStateMachine,
    RiskApprovalCoordinator,
    RiskLimits,
    RiskMode,
    build_risk_portfolio,
)


def _ready_inputs(**overrides):
    values = dict(
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
        halt_reasons=(),
    )
    values.update(overrides)
    return ReadinessInputs(**values)


def _portfolio():
    positions = (
        PositionRiskLeg(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            quantity=Decimal("2"),
            mark_price=Decimal("2000"),
            leverage=Decimal("4"),
            maintenance_margin=Decimal("40"),
            liquidation_price=Decimal("1500"),
        ),
        PositionRiskLeg(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.SHORT,
            quantity=Decimal("1"),
            mark_price=Decimal("1900"),
            leverage=Decimal("4"),
            maintenance_margin=Decimal("20"),
            liquidation_price=Decimal("2400"),
        ),
    )
    pending = (
        PendingOrderRisk(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            action=PositionAction.INCREASE,
            remaining_quantity=Decimal("0.5"),
            reference_price=Decimal("2000"),
            leverage=Decimal("4"),
        ),
        PendingOrderRisk(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.SHORT,
            action=PositionAction.REDUCE,
            remaining_quantity=Decimal("0.25"),
            reference_price=Decimal("1900"),
            leverage=Decimal("4"),
        ),
    )
    return build_risk_portfolio(
        account_id="main",
        equity=Decimal("10000"),
        wallet_balance=Decimal("10000"),
        available_balance=Decimal("7500"),
        positions=positions,
        pending_orders=pending,
        observed_at_ms=1000,
    )


def _coordinator(*, limits=None, reservation_clock=None, lock_clock=None):
    store = InMemoryDatabaseLeaseStore()
    writer = SingleWriterGuard(store, owner_id="writer", clock_ms=lambda: 1000)
    writer.acquire()
    gate = ReadinessGate(clock_ms=lambda: 1000)
    gate.evaluate(_ready_inputs())
    locks = PositionLockManager(
        reservation_ttl_seconds=1,
        monotonic_clock=lock_clock or (lambda: 0.0),
    )
    state = RiskActionStateMachine()
    coordinator = RiskApprovalCoordinator(
        engine=HedgeRiskEngine(
            limits
            or RiskLimits(
                max_margin_utilization=Decimal("0.8"),
                min_liquidation_buffer_ratio=Decimal("0.1"),
                max_gross_notional=Decimal("20000"),
                max_gross_exposure_ratio=Decimal("2"),
                max_leg_notional=Decimal("10000"),
                max_symbol_gross_notional=Decimal("15000"),
            )
        ),
        locks=locks,
        writer=writer,
        readiness=gate,
        state_machine=state,
        reservation_ttl_ms=100,
        clock_ms=reservation_clock or (lambda: 1000),
    )
    return writer, gate, locks, state, coordinator


def test_portfolio_builds_account_and_side_aware_request() -> None:
    portfolio = _portfolio()
    account = portfolio.account
    assert account.gross_long_notional == Decimal("4000")
    assert account.gross_short_notional == Decimal("1900")
    assert account.net_notional == Decimal("2100")
    assert account.pending_order_notional == Decimal("1000.0")
    assert account.pending_long_notional == Decimal("1000.0")
    assert account.pending_short_notional == 0

    reduce_request = portfolio.build_request(
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.SHORT,
        action=PositionAction.REDUCE,
        requested_quantity=Decimal("1"),
        reference_price=Decimal("1900"),
        leverage=Decimal("4"),
    )
    assert reduce_request.confirmed_quantity == Decimal("1")
    assert reduce_request.pending_reduce_quantity == Decimal("0.25")
    assert reduce_request.current_leg_notional == Decimal("1900")
    assert reduce_request.current_symbol_gross_notional == Decimal("5900")


def test_side_specific_and_free_margin_limits_clip_quantity() -> None:
    portfolio = _portfolio()
    limits = RiskLimits(
        max_margin_utilization=Decimal("0.9"),
        min_liquidation_buffer_ratio=Decimal("0.1"),
        max_gross_notional=Decimal("50000"),
        max_gross_exposure_ratio=Decimal("5"),
        max_long_notional=Decimal("6000"),
        min_available_balance=Decimal("7300"),
    )
    engine = HedgeRiskEngine(limits)
    decision = engine.evaluate_portfolio_order(
        portfolio=portfolio,
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("2"),
        reference_price=Decimal("2000"),
        leverage=Decimal("4"),
    )
    # 200 USDT free-margin headroom at 4x = 800 USDT = 0.4 ETH.
    assert decision.allowed
    assert decision.approved_quantity == Decimal("0.4")
    assert "AVAILABLE_MARGIN_CLIPPED" in decision.reason_codes


def test_atomic_batch_rolls_back_all_reservations_on_denial() -> None:
    portfolio = _portfolio()
    limits = RiskLimits(
        max_margin_utilization=Decimal("0.9"),
        min_liquidation_buffer_ratio=Decimal("0.1"),
        max_gross_notional=Decimal("7500"),
        max_gross_exposure_ratio=Decimal("2"),
        max_single_order_notional=Decimal("1000"),
    )
    _writer, _gate, _locks, _state, coordinator = _coordinator(limits=limits)
    first = portfolio.build_request(
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.SHORT,
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("0.3"),
        reference_price=Decimal("2000"),
        leverage=Decimal("4"),
    )
    second = portfolio.build_request(
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("1"),
        reference_price=Decimal("2000"),
        leverage=Decimal("4"),
    )
    batch = coordinator.approve_batch(((first, portfolio.account), (second, portfolio.account)))
    assert not batch.allowed
    assert all(not item.allowed for item in batch.approvals)
    assert coordinator.pending_increase_notional(account_id="main") == 0


def test_reservations_expire_and_capacity_is_recovered() -> None:
    reservation_now = [1000]
    lock_now = [0.0]
    portfolio = _portfolio()
    _writer, _gate, locks, _state, coordinator = _coordinator(
        reservation_clock=lambda: reservation_now[0],
        lock_clock=lambda: lock_now[0],
    )
    increase_request = portfolio.build_request(
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.SHORT,
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("0.1"),
        reference_price=Decimal("2000"),
        leverage=Decimal("4"),
    )
    approval = coordinator.approve(request=increase_request, account=portfolio.account)
    assert approval.allowed
    assert coordinator.pending_increase_notional(account_id="main") == Decimal("200.0")
    reservation_now[0] = 1200
    assert coordinator.pending_increase_notional(account_id="main") == 0
    assert approval.reservation is not None and approval.reservation.expired

    reduce = locks.reserve_reduce(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        requested_quantity=Decimal("1"),
        confirmed_quantity=Decimal("2"),
    )
    key = PositionLockKey("main", "ETH/USDT:USDT", PositionSide.LONG)
    assert locks.pending_reduce_quantity(key) == Decimal("1")
    lock_now[0] = 2.0
    assert locks.pending_reduce_quantity(key) == 0
    assert reduce.expired


def test_runtime_switches_to_reduce_only_and_keeps_controlled_reduce() -> None:
    portfolio = _portfolio()
    writer, gate, _locks, state, coordinator = _coordinator()
    monitor = ReadinessMonitor(gate=gate, inputs=_ready_inputs(), writer=writer)
    runtime = HedgeRiskRuntime(
        coordinator=coordinator,
        readiness=monitor,
        state_machine=state,
    )
    assert runtime.refresh().readiness.ready

    increase = runtime.approve_order(
        portfolio=portfolio,
        intent=OrderRiskIntent(
            "ETH/USDT:USDT",
            PositionSide.SHORT,
            PositionAction.INCREASE,
            Decimal("0.1"),
            Decimal("1900"),
            Decimal("4"),
        ),
    )
    assert increase.allowed
    assert increase.reservation is not None
    increase.reservation.release()

    monitor.update(user_stream_fresh=False)
    status = runtime.refresh()
    assert status.risk_state.mode is RiskMode.REDUCE_ONLY
    denied = runtime.approve_order(
        portfolio=portfolio,
        intent=OrderRiskIntent(
            "ETH/USDT:USDT",
            PositionSide.LONG,
            PositionAction.INCREASE,
            Decimal("0.1"),
            Decimal("2000"),
            Decimal("4"),
        ),
    )
    assert not denied.allowed
    reduced = runtime.approve_order(
        portfolio=portfolio,
        intent=OrderRiskIntent(
            "ETH/USDT:USDT",
            PositionSide.SHORT,
            PositionAction.REDUCE,
            Decimal("1"),
            Decimal("1900"),
            Decimal("4"),
        ),
    )
    assert reduced.allowed
    assert reduced.approved_quantity == Decimal("0.75")
    assert reduced.reservation is not None
    reduced.reservation.release()


def test_limits_from_mapping_and_runtime_factory() -> None:
    from freqtrade.hedge.risk import build_hedge_risk_runtime

    limits = RiskLimits.from_mapping(
        {
            "max_margin_utilization": "0.8",
            "min_liquidation_buffer_ratio": "0.1",
            "max_gross_exposure_ratio": "2",
        }
    )
    store = InMemoryDatabaseLeaseStore()
    writer = SingleWriterGuard(store, owner_id="factory", clock_ms=lambda: 1000)
    runtime = build_hedge_risk_runtime(
        limits=limits,
        writer=writer,
        readiness_inputs=_ready_inputs(single_writer_lease_valid=False),
        enable_lease_runner=False,
        readiness_clock_ms=lambda: 1000,
        reservation_clock_ms=lambda: 1000,
        lock_monotonic_clock=lambda: 0.0,
    )
    writer.acquire()
    assert runtime.refresh().readiness.ready
    approval = runtime.approve_order(
        portfolio=_portfolio(),
        intent=OrderRiskIntent(
            "ETH/USDT:USDT",
            PositionSide.SHORT,
            PositionAction.INCREASE,
            Decimal("0.1"),
            Decimal("1900"),
            Decimal("4"),
        ),
    )
    assert approval.allowed
    assert runtime.coordinator is not None
    assert runtime.readiness.report.ready
    approval.reservation.release()


def test_runtime_updates_reduce_only_reason_codes() -> None:
    writer, gate, _locks, state, coordinator = _coordinator()
    monitor = ReadinessMonitor(gate=gate, inputs=_ready_inputs(), writer=writer)
    runtime = HedgeRiskRuntime(
        coordinator=coordinator,
        readiness=monitor,
        state_machine=state,
    )
    runtime.refresh()
    monitor.update(user_stream_fresh=False)
    first = runtime.refresh().risk_state
    assert "USER_STREAM_STALE" in first.reason_codes
    monitor.update(user_stream_fresh=True, reconciliation_converged=False)
    second = runtime.refresh().risk_state
    assert "RECONCILIATION_NOT_CONVERGED" in second.reason_codes
    assert "USER_STREAM_STALE" not in second.reason_codes


def test_runtime_approves_directly_from_authoritative_account_facts() -> None:
    source = _portfolio()
    facts = AccountRiskFacts(
        exchange="binance",
        account_id="main",
        equity=source.account.equity,
        wallet_balance=source.account.wallet_balance,
        available_balance=source.account.available_balance,
        positions=source.positions,
        pending_orders=source.pending_orders,
        initial_margin=source.account.initial_margin,
        maintenance_margin=source.account.maintenance_margin,
        snapshot_id="direction-two-17",
        source_version=17,
        exchange_time_ms=900,
        observed_at_ms=1000,
        reconciliation_converged=True,
    )
    writer, gate, _locks, state, coordinator = _coordinator()
    monitor = ReadinessMonitor(gate=gate, inputs=_ready_inputs(), writer=writer)
    runtime = HedgeRiskRuntime(
        coordinator=coordinator,
        readiness=monitor,
        state_machine=state,
    )
    approval = runtime.approve_order_from_facts(
        facts=facts,
        intent=OrderRiskIntent(
            "ETH/USDT:USDT",
            PositionSide.SHORT,
            PositionAction.INCREASE,
            Decimal("0.1"),
            Decimal("1900"),
            Decimal("4"),
            intent_id="facts-intent",
            idempotency_key="facts-idem",
            correlation_id="facts-corr",
        ),
    )
    assert approval.allowed
    assert approval.risk_snapshot_id == "direction-two-17"
    assert approval.idempotency_key == "facts-idem"
    assert approval.evaluated_at_ms == 1000
    assert approval.reservation is not None
    approval.reservation.release()
