from __future__ import annotations

from datetime import datetime, timedelta
import json

import pytest
from sqlalchemy import func, select

from freqtrade.persistence.hedge_models import (
    AccountEvent,
    AccountRiskSnapshot,
    EventOutbox,
    FillEvent,
    OrderIntent,
    PositionSnapshot,
    ReconciliationDiff,
    ReconciliationRun,
    StrategySideState,
)
from freqtrade.persistence.hedge_outbox import OutboxWorker, TransactionalOutboxPublisher
from freqtrade.persistence.hedge_reconciliation import (
    HedgeReconciler,
    PositionFact,
    ReconciliationPolicy,
)
from freqtrade.persistence.hedge_recovery import LedgerRecoveryCoordinator
from freqtrade.persistence.hedge_repositories import HedgeLedgerRepository
from freqtrade.persistence.hedge_service import HedgePersistenceService


NOW = datetime(2026, 7, 27, 9, 0, 0)


def test_order_intent_state_machine_is_atomic_and_audited(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        intent, _ = repo.create_order_intent(
            account_id="main",
            exchange="binance",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            action="OPEN",
            side="BUY",
            order_type="LIMIT",
            requested_quantity="1",
            requested_price="2000",
            idempotency_key="intent-transition",
        )
        intent_id = intent.intent_id

    with session_factory.begin() as session:
        updated = HedgeLedgerRepository(session).transition_order_intent(
            intent_id=intent_id,
            new_status="APPROVED",
            expected_revision=0,
            approved_by="risk-engine",
        )
        assert updated.status == "APPROVED"
        assert updated.revision == 1

    with pytest.raises(ValueError, match="revision mismatch"):
        with session_factory.begin() as session:
            HedgeLedgerRepository(session).transition_order_intent(
                intent_id=intent_id,
                new_status="PREPARED",
                expected_revision=0,
            )

    with session_factory() as session:
        row = session.scalar(select(OrderIntent).where(OrderIntent.intent_id == intent_id))
        assert row is not None
        assert row.status == "APPROVED"
        assert row.approved_by == "risk-engine"
        assert session.scalar(select(func.count()).select_from(EventOutbox)) == 2


def test_fill_action_fee_event_and_strict_overclose(session_factory):
    with session_factory.begin() as session:
        result = HedgeLedgerRepository(session).apply_fill(
            exchange="binance",
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="open-fill",
            exchange_order_id="open-order",
            side="BUY",
            action="OPEN",
            quantity="1",
            price="2000",
            fee_amount="0.5",
            fee_currency="USDT",
            source="WEBSOCKET",
            event_time=NOW,
        )
        assert result.action == "OPEN"

    with session_factory() as session:
        fill = session.scalar(select(FillEvent))
        assert fill is not None and fill.action == "OPEN"
        fee = session.scalar(select(AccountEvent).where(AccountEvent.event_type == "FEE"))
        assert fee is not None
        assert fee.amount == "-0.5"
        assert fee.related_trade_id == "open-fill"

    with session_factory.begin() as session:
        blocked = HedgeLedgerRepository(session).apply_fill(
            exchange="binance",
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="too-large-close",
            exchange_order_id="close-order",
            side="SELL",
            action="CLOSE",
            quantity="2",
            price="2100",
            source="REST",
            event_time=NOW + timedelta(seconds=1),
            allow_overclose=False,
        )
        assert blocked.projection_blocked is True
        assert blocked.reason_code == "OVER_CLOSE_FILL"

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(FillEvent)) == 2
        blocked_fill = session.scalar(
            select(FillEvent).where(FillEvent.exchange_trade_id == "too-large-close")
        )
        assert blocked_fill is not None
        assert blocked_fill.projection_status == "BLOCKED"
        current = session.scalar(
            select(PositionSnapshot).where(PositionSnapshot.is_current.is_(True))
        )
        assert current is not None and current.quantity == "1"


def test_position_fact_sequence_and_exchange_scoped_current_projection(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.append_position_snapshot(
            snapshot_key="binance-1",
            account_id="main",
            exchange="binance",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            quantity="1",
            entry_price="100",
            source="REST",
            source_event_time=NOW,
            sequence_number=1,
        )
        repo.append_position_snapshot(
            snapshot_key="binance-2",
            account_id="main",
            exchange="binance",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            quantity="2",
            entry_price="101",
            source="REST",
            source_event_time=NOW,
            sequence_number=2,
        )
        repo.append_position_snapshot(
            snapshot_key="okx-1",
            account_id="main",
            exchange="okx",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            quantity="3",
            entry_price="102",
            source="REST",
            source_event_time=NOW,
            sequence_number=1,
        )

    with session_factory() as session:
        rows = session.scalars(
            select(PositionSnapshot).where(PositionSnapshot.is_current.is_(True))
        ).all()
        assert {(row.exchange, row.quantity) for row in rows} == {
            ("binance", "2"),
            ("okx", "3"),
        }


def test_reconciliation_detects_and_repairs_position_drift(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.append_position_snapshot(
            snapshot_key="local-base",
            account_id="main",
            exchange="binance",
            symbol="ETH/USDT:USDT",
            position_side="SHORT",
            quantity="1",
            entry_price="2500",
            source="REST",
            source_event_time=NOW,
        )

    with session_factory.begin() as session:
        summary = HedgeReconciler(HedgeLedgerRepository(session)).reconcile_positions(
            account_id="main",
            exchange="binance",
            facts=[
                PositionFact(
                    symbol="ETH/USDT:USDT",
                    position_side="SHORT",
                    quantity="2",
                    entry_price="2450",
                    event_time=NOW + timedelta(seconds=1),
                    sequence_number=10,
                )
            ],
            policy=ReconciliationPolicy(auto_repair_positions=True),
        )
        assert summary.status == "REPAIRED"
        assert summary.diff_count == 2
        assert summary.repaired_count == 1

    with session_factory() as session:
        run = session.scalar(select(ReconciliationRun))
        assert run is not None and run.status == "REPAIRED"
        assert session.scalar(select(func.count()).select_from(ReconciliationDiff)) == 2
        current = session.scalar(
            select(PositionSnapshot).where(
                PositionSnapshot.exchange == "binance",
                PositionSnapshot.is_current.is_(True),
            )
        )
        assert current is not None
        assert current.quantity == "2"
        assert current.entry_price == "2450"


def test_service_transaction_rollback_and_recovery_all(session_factory):
    service = HedgePersistenceService(session_factory)

    def broken(repo: HedgeLedgerRepository):
        repo.create_order_intent(
            account_id="main",
            exchange="binance",
            symbol="SOL/USDT:USDT",
            position_side="LONG",
            action="OPEN",
            side="BUY",
            order_type="LIMIT",
            requested_quantity="1",
            requested_price="100",
            idempotency_key="rolled-back-intent",
        )
        raise RuntimeError("rollback service transaction")

    with pytest.raises(RuntimeError):
        service.transaction(broken)

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(OrderIntent)) == 0
        assert session.scalar(select(func.count()).select_from(EventOutbox)) == 0

    service.record_fill(
        exchange="binance",
        account_id="main",
        symbol="SOL/USDT:USDT",
        position_side="LONG",
        exchange_trade_id="recovery-fill",
        exchange_order_id="recovery-order",
        side="BUY",
        quantity="2",
        price="100",
        source="WEBSOCKET",
        event_time=NOW,
    )
    report = LedgerRecoveryCoordinator(session_factory).recover_all()
    assert report.position_count == 1
    assert report.accounts[0].account_id == "main"


def test_outbox_worker_run_once(session_factory):
    service = HedgePersistenceService(session_factory)
    service.create_order_intent(
        account_id="main",
        exchange="binance",
        symbol="XRP/USDT:USDT",
        position_side="LONG",
        action="OPEN",
        side="BUY",
        order_type="LIMIT",
        requested_quantity="1",
        requested_price="1",
        idempotency_key="worker-intent",
    )
    published: list[str] = []
    worker = OutboxWorker(
        TransactionalOutboxPublisher(session_factory),
        lambda envelope: published.append(envelope.event_type),
        batch_size=10,
    )
    result = worker.run_once()
    assert result.published == 1
    assert published == ["hedge.order_intent.created"]


def test_bootstrap_runs_migrations_and_recovers_existing_ledger(engine, session_factory):
    from freqtrade.persistence.hedge_bootstrap import bootstrap_hedge_persistence

    with session_factory.begin() as session:
        HedgeLedgerRepository(session).apply_fill(
            exchange="binance",
            account_id="bootstrap-account",
            symbol="ADA/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="bootstrap-fill",
            exchange_order_id="bootstrap-order",
            side="BUY",
            quantity="10",
            price="0.5",
            source="WEBSOCKET",
            event_time=NOW,
        )

    report = bootstrap_hedge_persistence(
        engine,
        session_factory=session_factory,
        recover=True,
    )
    from freqtrade.persistence.hedge_migrations import migration_plan_ids

    assert len(report.migration.applied) == len(migration_plan_ids())
    assert report.recovery is not None
    assert report.recovery.position_count == 1


def test_fill_after_equal_timestamp_position_high_water_is_replayed(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.append_position_snapshot(
            snapshot_key="high-water-base",
            account_id="main",
            exchange="binance",
            symbol="DOGE/USDT:USDT",
            position_side="LONG",
            quantity="10",
            entry_price="0.1",
            source="WEBSOCKET",
            source_event_time=NOW,
            sequence_number=10,
        )
        applied = repo.apply_fill(
            exchange="binance",
            account_id="main",
            symbol="DOGE/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="after-high-water",
            exchange_order_id="after-high-water-order",
            side="BUY",
            quantity="5",
            price="0.2",
            source="WEBSOCKET",
            event_time=NOW,
            sequence_number=11,
        )
        assert applied.position_quantity == "15"

    with session_factory() as session:
        recovery = HedgeLedgerRepository(session).recover_projection(account_id="main")
        assert recovery.positions[0].quantity == "15"
        assert recovery.positions[0].last_sequence_number == 11


def test_service_exposes_risk_state_and_projection_recovery(session_factory):
    service = HedgePersistenceService(session_factory)
    risk, created = service.record_account_risk_snapshot(
        snapshot_key="risk-main-1",
        account_id="main",
        exchange="binance",
        source="REST",
        source_event_time=NOW,
        wallet_balance="1000",
        available_balance="800",
        margin_balance="1050",
        gross_exposure="500",
        net_exposure="100",
        margin_utilization="0.2",
        risk_state="READY",
    )
    assert created is True
    assert risk.wallet_balance == "1000"

    state = service.update_strategy_side_state(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side="LONG",
        strategy_name="GridStrategy",
        state_name="ACTIVE",
        state={"layer": 2},
        expected_revision=0,
    )
    assert state.revision == 1

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AccountRiskSnapshot)) == 1
        stored = session.scalar(select(StrategySideState))
        assert stored is not None and stored.state_name == "ACTIVE"

    service.record_fill(
        exchange="binance",
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side="LONG",
        exchange_trade_id="service-recovery-fill",
        exchange_order_id="service-recovery-order",
        side="BUY",
        quantity="1",
        price="2000",
        source="WEBSOCKET",
        event_time=NOW + timedelta(seconds=5),
    )
    recovery = service.recover_account("main", symbol="ETH/USDT:USDT")
    assert len(recovery.positions) == 1
    assert recovery.positions[0].quantity == "1"


def test_intent_and_fill_reject_action_direction_mismatch(session_factory):
    service = HedgePersistenceService(session_factory)
    with pytest.raises(ValueError, match="OPEN action has reducing order direction"):
        service.create_order_intent(
            account_id="main",
            exchange="binance",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            action="OPEN",
            side="SELL",
            order_type="LIMIT",
            requested_quantity="1",
            requested_price="2000",
            idempotency_key="invalid-open-direction",
        )

    with pytest.raises(ValueError, match="REDUCE order intent must be reduce-only"):
        service.create_order_intent(
            account_id="main",
            exchange="binance",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            action="REDUCE",
            side="SELL",
            order_type="LIMIT",
            requested_quantity="1",
            requested_price="2100",
            idempotency_key="invalid-reduce-only",
        )

    with pytest.raises(ValueError, match="CLOSE action has increasing order direction"):
        service.record_fill(
            exchange="binance",
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="invalid-close-fill",
            exchange_order_id="invalid-close-order",
            side="BUY",
            action="CLOSE",
            quantity="1",
            price="2100",
            source="WEBSOCKET",
            event_time=NOW,
        )


def test_position_projection_outbox_aggregate_is_exchange_scoped(session_factory):
    service = HedgePersistenceService(session_factory)
    for exchange, trade_id in (("binance", "binance-fill"), ("okx", "okx-fill")):
        service.record_fill(
            exchange=exchange,
            account_id="main",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            exchange_trade_id=trade_id,
            exchange_order_id=f"{exchange}-order",
            side="BUY",
            action="OPEN",
            quantity="1",
            price="100",
            source="WEBSOCKET",
            event_time=NOW,
        )

    with session_factory() as session:
        rows = session.scalars(
            select(EventOutbox).where(
                EventOutbox.event_type == "hedge.position.projected"
            )
        ).all()
        assert len(rows) == 2
        assert len({row.aggregate_id for row in rows}) == 2
        assert {json.loads(row.payload_json)["exchange"] for row in rows} == {
            "binance",
            "okx",
        }



def test_reconciliation_repairs_missing_exchange_position_to_zero(session_factory):
    service = HedgePersistenceService(session_factory)
    service.record_fill(
        exchange="binance",
        account_id="main",
        symbol="LTC/USDT:USDT",
        position_side="LONG",
        exchange_trade_id="missing-exchange-open",
        exchange_order_id="missing-exchange-order",
        side="BUY",
        action="OPEN",
        quantity="2",
        price="100",
        source="WEBSOCKET",
        event_time=NOW,
    )

    repaired = service.reconcile_positions(
        account_id="main",
        exchange="binance",
        facts=[],
        observed_at=NOW + timedelta(seconds=1),
        policy=ReconciliationPolicy(auto_repair_positions=True),
        trigger="REST_FULL",
    )
    assert repaired.status == "REPAIRED"
    assert repaired.repaired_count == 1

    with session_factory() as session:
        current = session.scalar(
            select(PositionSnapshot).where(
                PositionSnapshot.exchange == "binance",
                PositionSnapshot.symbol == "LTC/USDT:USDT",
                PositionSnapshot.is_current.is_(True),
            )
        )
        assert current is not None and current.quantity == "0"
