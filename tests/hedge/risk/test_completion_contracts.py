from decimal import Decimal

import pytest

from freqtrade.enums.hedge import PositionAction, PositionSide
from freqtrade.hedge.concurrency import (
    InMemoryDatabaseLeaseStore,
    PositionLockManager,
    SingleWriterGuard,
)
from freqtrade.hedge.readiness import ReadinessGate, ReadinessInputs
from freqtrade.hedge.risk import (
    AccountRiskFacts,
    AccountRiskSnapshot,
    HedgeRiskEngine,
    InMemoryRiskApprovalCommitStore,
    PendingOrderRisk,
    PositionRiskLeg,
    RiskActionStateMachine,
    RiskApprovalCoordinator,
    RiskLimits,
    RiskPositionKey,
    RiskRequest,
    UnknownOrderRisk,
    build_risk_portfolio,
)


def _inputs(**overrides) -> ReadinessInputs:
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
    )
    values.update(overrides)
    return ReadinessInputs(**values)


def _account(**overrides) -> AccountRiskSnapshot:
    values = dict(
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
        snapshot_id="risk-1",
    )
    values.update(overrides)
    return AccountRiskSnapshot(**values)


def _limits(**overrides) -> RiskLimits:
    values = dict(
        max_margin_utilization=Decimal("0.8"),
        min_liquidation_buffer_ratio=Decimal("0.2"),
        max_gross_notional=Decimal("50000"),
    )
    values.update(overrides)
    return RiskLimits(**values)


def _coordinator(*, now: list[int] | None = None):
    clock = now if now is not None else [1000]
    store = InMemoryDatabaseLeaseStore()
    writer = SingleWriterGuard(store, owner_id="writer", ttl_ms=10_000, clock_ms=lambda: clock[0])
    writer.acquire()
    gate = ReadinessGate(clock_ms=lambda: clock[0])
    gate.evaluate(_inputs())
    coordinator = RiskApprovalCoordinator(
        engine=HedgeRiskEngine(_limits()),
        locks=PositionLockManager(),
        writer=writer,
        readiness=gate,
        state_machine=RiskActionStateMachine(),
        clock_ms=lambda: clock[0],
    )
    return coordinator, gate, writer


def test_low_level_reduce_fails_closed_without_confirmed_quantity() -> None:
    decision = HedgeRiskEngine(_limits()).evaluate(
        action=PositionAction.REDUCE,
        requested_quantity=Decimal("10"),
        reference_price=Decimal("2000"),
        account=_account(),
        position_side=PositionSide.LONG,
    )
    assert not decision.allowed
    assert decision.reason_codes == ("CONFIRMED_POSITION_REQUIRED",)


def test_unknown_order_quarantines_only_its_position_key() -> None:
    long_key = RiskPositionKey("binance", "main", "ETH/USDT:USDT", PositionSide.LONG)
    short_key = RiskPositionKey("binance", "main", "ETH/USDT:USDT", PositionSide.SHORT)
    unknown = UnknownOrderRisk(long_key, "client-long", 900)
    gate = ReadinessGate(clock_ms=lambda: 1000)
    report = gate.evaluate(
        _inputs(unknown_order_count=1, unknown_orders=(unknown,))
    )
    assert report.state.value == "DEGRADED"
    assert gate.allows_new_risk(short_key)
    assert not gate.allows_new_risk(long_key)
    assert gate.allows_controlled_reduce(short_key)
    assert not gate.allows_controlled_reduce(long_key)


def test_missing_liquidation_or_maintenance_data_fails_closed() -> None:
    portfolio = build_risk_portfolio(
        account_id="main",
        equity=Decimal("10000"),
        wallet_balance=Decimal("10000"),
        available_balance=Decimal("9000"),
        positions=(
            PositionRiskLeg(
                account_id="main",
                symbol="ETH/USDT:USDT",
                position_side=PositionSide.LONG,
                quantity=Decimal("1"),
                mark_price=Decimal("2000"),
                maintenance_margin=Decimal("0"),
                liquidation_price=None,
            ),
        ),
    )
    assert not portfolio.account.effective_risk_data_valid
    assert set(portfolio.account.risk_data_errors) == {
        "LIQUIDATION_DATA_INCOMPLETE",
        "MAINTENANCE_MARGIN_INCOMPLETE",
    }
    decision = HedgeRiskEngine(_limits()).evaluate_portfolio_order(
        portfolio=portfolio,
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("0.1"),
        reference_price=Decimal("2000"),
    )
    assert not decision.allowed


def test_post_fill_maintenance_buffer_clips_new_order() -> None:
    account = _account(
        equity=Decimal("1000"),
        wallet_balance=Decimal("1000"),
        available_balance=Decimal("1000"),
        initial_margin=Decimal("0"),
        maintenance_margin=Decimal("700"),
        gross_long_notional=Decimal("0"),
        gross_short_notional=Decimal("0"),
        net_notional=Decimal("0"),
    )
    decision = HedgeRiskEngine(_limits()).evaluate(
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("2"),
        reference_price=Decimal("1000"),
        account=account,
        position_side=PositionSide.LONG,
        leverage=Decimal("10"),
        maintenance_margin_rate=Decimal("0.1"),
    )
    assert decision.allowed
    assert decision.approved_quantity == Decimal("1")
    assert "PROJECTED_LIQUIDATION_BUFFER_CLIPPED" in decision.reason_codes


def test_reservation_releases_only_after_verified_durable_commit() -> None:
    coordinator, _gate, _writer = _coordinator()
    request = RiskRequest(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("0.1"),
        reference_price=Decimal("2000"),
        intent_id="intent-1",
        idempotency_key="idem-1",
        correlation_id="corr-1",
    )
    approval = coordinator.approve(request=request, account=_account())
    assert approval.allowed and approval.reservation is not None
    assert coordinator.pending_increase_notional(account_id="main") > 0
    with pytest.raises(TypeError):
        approval.reservation.confirm()  # type: ignore[call-arg]
    commit_store = InMemoryRiskApprovalCommitStore()
    record = approval.reservation.confirm(
        commit_port=commit_store,
        durable_reference="order_intents:1",
    )
    assert record.intent_id == "intent-1"
    assert record.idempotency_key == "idem-1"
    assert record.correlation_id == "corr-1"
    assert record.risk_snapshot_id == "risk-1"
    assert record.as_dict()["request"]["position_side"] == "LONG"
    assert record.as_dict()["risk_snapshot"]["snapshot_id"] == "risk-1"
    assert approval.reservation.committed
    assert coordinator.pending_increase_notional(account_id="main") == 0


def test_durable_reference_requires_nonempty_string() -> None:
    coordinator, _gate, _writer = _coordinator()
    request = RiskRequest(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("0.1"),
        reference_price=Decimal("2000"),
    )
    approval = coordinator.approve(request=request, account=_account())
    assert approval.allowed and approval.reservation is not None
    with pytest.raises(ValueError, match="durable_reference"):
        approval.reservation.build_commit_record(durable_reference="   ")
    with pytest.raises(ValueError, match="durable_reference"):
        approval.reservation.build_commit_record(durable_reference=None)  # type: ignore[arg-type]
    approval.reservation.release()


def test_expired_intent_is_denied_before_reservation() -> None:
    now = [1000]
    coordinator, _gate, _writer = _coordinator(now=now)
    request = RiskRequest(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=PositionAction.INCREASE,
        requested_quantity=Decimal("0.1"),
        reference_price=Decimal("2000"),
        expires_at_ms=999,
    )
    approval = coordinator.approve(request=request, account=_account())
    assert not approval.allowed
    assert approval.reason_codes == ("RISK_INTENT_EXPIRED",)


def test_authoritative_facts_port_payload_preserves_versions() -> None:
    facts = AccountRiskFacts(
        exchange="binance",
        account_id="main",
        equity=Decimal("10000"),
        wallet_balance=Decimal("10000"),
        available_balance=Decimal("9000"),
        positions=(
            PositionRiskLeg(
                account_id="main",
                symbol="ETH/USDT:USDT",
                position_side=PositionSide.LONG,
                quantity=Decimal("1"),
                mark_price=Decimal("2000"),
                leverage=Decimal("3"),
                maintenance_margin=Decimal("20"),
                liquidation_price=Decimal("1200"),
            ),
        ),
        pending_orders=(
            PendingOrderRisk(
                account_id="main",
                symbol="ETH/USDT:USDT",
                position_side=PositionSide.LONG,
                action=PositionAction.INCREASE,
                remaining_quantity=Decimal("0.1"),
                reference_price=Decimal("1900"),
                leverage=Decimal("3"),
            ),
        ),
        initial_margin=Decimal("700"),
        maintenance_margin=Decimal("20"),
        snapshot_id="snapshot-42",
        source_version=42,
        exchange_time_ms=900,
        observed_at_ms=950,
        reconciliation_converged=True,
    )
    portfolio = facts.to_portfolio()
    assert portfolio.account.snapshot_id == "snapshot-42"
    assert portfolio.account.source_version == 42
    assert portfolio.account.exchange_time_ms == 900
    assert portfolio.account.effective_risk_data_valid


def test_unscoped_unknown_order_fails_closed_for_every_leg() -> None:
    long_key = RiskPositionKey("binance", "main", "ETH/USDT:USDT", PositionSide.LONG)
    short_key = RiskPositionKey("binance", "main", "ETH/USDT:USDT", PositionSide.SHORT)
    gate = ReadinessGate(clock_ms=lambda: 1000)
    report = gate.evaluate(_inputs(unknown_order_count=1, unknown_orders=()))
    assert report.state.value == "HALT"
    assert "UNKNOWN_ORDER_SCOPE_INCOMPLETE" in {
        code.value for code in report.reason_codes
    }
    assert not gate.allows_new_risk(long_key)
    assert not gate.allows_new_risk(short_key)
    assert not gate.allows_controlled_reduce(long_key)
    assert not gate.allows_controlled_reduce(short_key)


def test_coordinator_enforces_same_leg_unknown_quarantine() -> None:
    coordinator, gate, _writer = _coordinator()
    long_key = RiskPositionKey("binance", "main", "ETH/USDT:USDT", PositionSide.LONG)
    gate.evaluate(
        _inputs(
            unknown_order_count=1,
            unknown_orders=(UnknownOrderRisk(long_key, "unknown-long", 900),),
        )
    )
    long_increase = coordinator.approve(
        request=RiskRequest(
            exchange="binance",
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            action=PositionAction.INCREASE,
            requested_quantity=Decimal("0.1"),
            reference_price=Decimal("2000"),
        ),
        account=_account(),
    )
    assert not long_increase.allowed
    assert "READINESS_NOT_READY" in long_increase.reason_codes

    short_increase = coordinator.approve(
        request=RiskRequest(
            exchange="binance",
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.SHORT,
            action=PositionAction.INCREASE,
            requested_quantity=Decimal("0.1"),
            reference_price=Decimal("2000"),
        ),
        account=_account(),
    )
    assert short_increase.allowed
    assert short_increase.reservation is not None
    short_increase.reservation.release()

    long_reduce = coordinator.approve(
        request=RiskRequest(
            exchange="binance",
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            action=PositionAction.REDUCE,
            requested_quantity=Decimal("0.5"),
            reference_price=Decimal("2000"),
            confirmed_quantity=Decimal("1"),
        ),
        account=_account(),
    )
    assert not long_reduce.allowed
    assert "CONTROLLED_REDUCE_NOT_READY" in long_reduce.reason_codes


class _TransientReadFailureCommitStore(InMemoryRiskApprovalCommitStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_reads = True

    def read_commit(self, decision_id: str):
        if self.fail_reads:
            return None
        return super().read_commit(decision_id)


def test_batch_durable_handoff_retains_capacity_on_verification_failure() -> None:
    coordinator, _gate, _writer = _coordinator()
    items = tuple(
        (
            RiskRequest(
                account_id="main",
                symbol="ETH/USDT:USDT",
                position_side=side,
                action=PositionAction.INCREASE,
                requested_quantity=Decimal("0.1"),
                reference_price=Decimal("2000"),
                intent_id=f"intent-{side.value.lower()}",
                idempotency_key=f"idem-{side.value.lower()}",
                correlation_id="corr-batch",
            ),
            _account(),
        )
        for side in (PositionSide.LONG, PositionSide.SHORT)
    )
    batch = coordinator.approve_batch(items)
    assert batch.allowed
    assert coordinator.pending_increase_notional(account_id="main") == Decimal("400")

    commit_store = _TransientReadFailureCommitStore()
    with pytest.raises(RuntimeError, match="verification failed"):
        batch.confirm(
            commit_port=commit_store,
            durable_references=("order_intents:long", "order_intents:short"),
        )
    # The durable transaction may already exist.  Keeping local reservations is
    # conservative until the read path can prove the handoff.
    assert coordinator.pending_increase_notional(account_id="main") == Decimal("400")

    commit_store.fail_reads = False
    records = batch.confirm(
        commit_port=commit_store,
        durable_references=("order_intents:long", "order_intents:short"),
    )
    assert len(records) == 2
    assert coordinator.pending_increase_notional(account_id="main") == Decimal("0")


def test_reduce_commit_uses_actual_account_risk_snapshot() -> None:
    coordinator, _gate, _writer = _coordinator()
    request = RiskRequest(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=PositionAction.REDUCE,
        requested_quantity=Decimal("0.5"),
        reference_price=Decimal("2000"),
        confirmed_quantity=Decimal("1"),
        intent_id="reduce-intent",
        idempotency_key="reduce-idem",
        correlation_id="reduce-corr",
    )
    approval = coordinator.approve(request=request, account=_account(snapshot_id="risk-reduce"))
    assert approval.allowed and approval.reservation is not None
    record = approval.reservation.confirm(
        commit_port=InMemoryRiskApprovalCommitStore(),
        durable_reference="order_intents:reduce",
    )
    assert record.risk_snapshot_id == "risk-reduce"
    assert record.as_dict()["risk_snapshot"]["snapshot_id"] == "risk-reduce"
    assert record.as_dict()["request"]["action"] == "REDUCE"


def test_accepted_durable_handoff_can_be_verified_after_lease_expiry() -> None:
    now = [1000]
    coordinator, _gate, _writer = _coordinator(now=now)
    approval = coordinator.approve(
        request=RiskRequest(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            action=PositionAction.INCREASE,
            requested_quantity=Decimal("0.1"),
            reference_price=Decimal("2000"),
            intent_id="late-verify-intent",
            idempotency_key="late-verify-idem",
            correlation_id="late-verify-corr",
        ),
        account=_account(),
    )
    assert approval.allowed and approval.reservation is not None
    store = _TransientReadFailureCommitStore()
    with pytest.raises(RuntimeError, match="could not be verified"):
        approval.reservation.confirm(
            commit_port=store,
            durable_reference="order_intents:late-verify",
        )

    now[0] = 50_000  # writer lease and local reservation are now expired
    store.fail_reads = False
    record = approval.reservation.confirm(
        commit_port=store,
        durable_reference="order_intents:late-verify",
    )
    assert record.intent_id == "late-verify-intent"
    assert approval.reservation.committed


def test_sql_risk_approval_commit_store_is_idempotent_and_fail_closed(tmp_path) -> None:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from freqtrade.hedge.risk import ApprovalCommitRecord, SqlRiskApprovalCommitStore
    from freqtrade.persistence.hedge_models import RiskApprovalCommitRow

    engine = create_engine(f"sqlite:///{tmp_path / 'risk-approval.sqlite'}")
    RiskApprovalCommitRow.__table__.create(engine)
    store = SqlRiskApprovalCommitStore(sessionmaker(bind=engine, expire_on_commit=False))
    record = ApprovalCommitRecord(
        decision_id="decision-1",
        intent_id="intent-1",
        idempotency_key="idem-1",
        correlation_id="corr-1",
        risk_snapshot_id="risk-1",
        request_json='{"action":"INCREASE"}',
        risk_snapshot_json='{"snapshot_id":"risk-1"}',
        rules_version="v1",
        fencing_token=1,
        approved_quantity=Decimal("0.1"),
        approved_notional=Decimal("200"),
        evaluated_at_ms=1000,
        committed_at_ms=1001,
        durable_reference="order_intents:1",
    )

    assert store.commit_approval(record)
    assert store.commit_approval(record)
    assert store.read_commit("decision-1") == record

    conflicting = ApprovalCommitRecord(
        decision_id="decision-1",
        intent_id="intent-1",
        idempotency_key="idem-1",
        correlation_id="corr-1",
        risk_snapshot_id="risk-1",
        request_json='{"action":"INCREASE"}',
        risk_snapshot_json='{"snapshot_id":"risk-1"}',
        rules_version="v1",
        fencing_token=1,
        approved_quantity=Decimal("0.2"),
        approved_notional=Decimal("400"),
        evaluated_at_ms=1000,
        committed_at_ms=1001,
        durable_reference="order_intents:1",
    )
    assert not store.commit_approval(conflicting)
