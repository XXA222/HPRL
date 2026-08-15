from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.orm import sessionmaker

from freqtrade.persistence.hedge_models import (
    AccountEvent,
    AuditEvent,
    CorePositionState,
    CurrentOrderProjection,
    EventOutbox,
    OrderSnapshot,
    PositionSnapshot,
    TacticalLot,
    TargetPosition,
)
from freqtrade.persistence.hedge_reconciliation import (
    AccountModeFact,
    HedgeReconciler,
    OrderFact,
    PositionFact,
    ReconciliationPolicy,
)
from freqtrade.persistence.hedge_repositories import HedgeLedgerRepository
from freqtrade.persistence.hedge_service import HedgePersistenceService


NOW = datetime(2026, 7, 27, 12, 0, 0)


def test_canonical_symbol_prevents_duplicate_position_identity(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.append_position_snapshot(
            snapshot_key="compact-symbol",
            account_id="main",
            exchange="binance",
            symbol="ETHUSDT",
            position_side="LONG",
            quantity="1",
            entry_price="2000",
            source="REST",
            source_event_time=NOW,
        )
        repo.append_position_snapshot(
            snapshot_key="canonical-symbol",
            account_id="main",
            exchange="binance",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            quantity="2",
            entry_price="1900",
            source="REST",
            source_event_time=NOW + timedelta(seconds=1),
        )

    with session_factory() as session:
        current = session.scalars(
            select(PositionSnapshot).where(PositionSnapshot.is_current.is_(True))
        ).all()
        assert len(current) == 1
        assert current[0].symbol == "ETH/USDT:USDT"
        assert current[0].quantity == "2"


def test_fill_atomically_updates_order_and_current_projection(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.append_order_snapshot(
            snapshot_key="order-ack",
            account_id="main",
            exchange="binance",
            symbol="ETHUSDT",
            position_side="LONG",
            exchange_order_id="order-1",
            side="BUY",
            action="OPEN",
            status="ACKNOWLEDGED",
            original_quantity="1",
            executed_quantity="0",
            cumulative_quote="0",
            source="REST",
            source_event_time=NOW,
        )
        partial = repo.apply_fill(
            exchange="binance",
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="fill-1",
            exchange_order_id="order-1",
            side="BUY",
            action="OPEN",
            quantity="0.4",
            price="2000",
            source="WEBSOCKET",
            event_time=NOW + timedelta(seconds=1),
        )
        assert partial.order_status == "PARTIAL"

    with session_factory() as session:
        current = session.scalar(select(CurrentOrderProjection))
        assert current is not None
        assert current.status == "PARTIAL"
        assert current.executed_quantity == "0.4"
        assert current.remaining_quantity == "0.6"
        local = session.scalar(
            select(OrderSnapshot)
            .where(OrderSnapshot.source == "LOCAL")
            .order_by(OrderSnapshot.id.desc())
        )
        assert local is not None and local.status == "PARTIAL"

    with session_factory.begin() as session:
        filled = HedgeLedgerRepository(session).apply_fill(
            exchange="binance",
            account_id="main",
            symbol="ETHUSDT",
            position_side="LONG",
            exchange_trade_id="fill-2",
            exchange_order_id="order-1",
            side="BUY",
            action="OPEN",
            quantity="0.6",
            price="2100",
            source="WEBSOCKET",
            event_time=NOW + timedelta(seconds=2),
        )
        assert filled.order_status == "FILLED"

    with session_factory() as session:
        current = session.scalar(select(CurrentOrderProjection))
        assert current is not None
        assert current.status == "FILLED"
        assert current.executed_quantity == "1"
        assert current.remaining_quantity == "0"
        assert current.average_price == "2060"


def test_recovery_preserves_exchange_risk_fields_and_is_idempotent(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.append_position_snapshot(
            snapshot_key="position-risk-fact",
            account_id="main",
            exchange="binance",
            symbol="ETHUSDT",
            position_side="LONG",
            quantity="1",
            entry_price="2000",
            mark_price="2100",
            notional="2100",
            unrealized_pnl="100",
            liquidation_price="900",
            leverage="3",
            margin_mode="cross",
            source="REST",
            source_event_time=NOW,
            source_version="rest-1",
        )
        repo.append_order_snapshot(
            snapshot_key="open-order",
            account_id="main",
            exchange="binance",
            symbol="ETHUSDT",
            position_side="LONG",
            exchange_order_id="order-risk",
            side="BUY",
            status="PARTIAL",
            original_quantity="2",
            executed_quantity="1",
            cumulative_quote="2000",
            source="REST",
            source_event_time=NOW,
        )

    with session_factory.begin() as session:
        session.execute(delete(PositionSnapshot).where(PositionSnapshot.source == "LOCAL"))
        session.execute(delete(CurrentOrderProjection))

    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        recovery = repo.rebuild_current_projections(account_id="main")
        assert len(recovery.orders) == 1
        assert len(recovery.positions) == 1

    with session_factory() as session:
        current = session.scalar(
            select(PositionSnapshot).where(PositionSnapshot.is_current.is_(True))
        )
        order = session.scalar(select(CurrentOrderProjection))
        assert current is not None
        assert current.mark_price == "2100"
        assert current.liquidation_price == "900"
        assert current.leverage == "3"
        assert current.margin_mode == "CROSS"
        assert order is not None and order.status == "PARTIAL"
        snapshot_count = session.scalar(select(func.count()).select_from(PositionSnapshot))
        outbox_count = session.scalar(select(func.count()).select_from(EventOutbox))

    with session_factory.begin() as session:
        HedgeLedgerRepository(session).rebuild_current_projections(account_id="main")

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(PositionSnapshot)) == snapshot_count
        assert session.scalar(select(func.count()).select_from(EventOutbox)) == outbox_count


def test_target_core_tactical_and_audit_ledgers_are_persistent(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        target, created = repo.record_target_position(
            exchange="binance",
            account_id="main",
            symbol="ETHUSDT",
            position_side="LONG",
            target_quantity="20",
            reason="core build",
            strategy_id="grid-v1",
            cycle_id="cycle-1",
            correlation_id="decision-1",
        )
        assert created is True
        core = repo.upsert_core_position_state(
            exchange="binance",
            account_id="main",
            symbol="ETHUSDT",
            position_side="LONG",
            core_quantity="10",
            core_floor="8",
            effective_cost="1500",
        )
        lot, lot_created = repo.record_tactical_lot(
            exchange="binance",
            account_id="main",
            symbol="ETHUSDT",
            position_side="LONG",
            strategy_name="grid-v1",
            lot_type="GRID_ENTRY",
            quantity="1",
            entry_price="1900",
            opened_at=NOW,
            lot_id="lot-1",
        )
        audit = repo.record_audit_event(
            account_id="main",
            exchange="binance",
            event_type="TARGET_ACCEPTED",
            correlation_id="decision-1",
            entity_type="TargetPosition",
            entity_id=target.target_id,
            payload={"cycle_id": "cycle-1"},
        )
        assert core.revision == 1
        assert lot_created is True
        assert lot.lot_id == "lot-1"
        assert audit.correlation_id == "decision-1"

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(TargetPosition)) == 1
        assert session.scalar(select(func.count()).select_from(CorePositionState)) == 1
        assert session.scalar(select(func.count()).select_from(TacticalLot)) == 1
        assert session.scalar(select(func.count()).select_from(AuditEvent)) == 1
        events = session.scalars(select(EventOutbox).order_by(EventOutbox.id)).all()
        assert events
        assert all(event.correlation_id for event in events)
        assert all(event.payload_version == 1 for event in events)
        assert all(event.schema_version == "h3-ledger-v2" for event in events)



def test_account_event_natural_fact_key_deduplicates_different_caller_keys(session_factory):
    values = {
        "account_id": "main",
        "exchange": "binance",
        "event_type": "FUNDING",
        "asset": "USDT",
        "amount": "-1.25",
        "symbol": "ETHUSDT",
        "position_side": "LONG",
        "source": "REST",
        "event_time": NOW,
    }
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        first, first_created = repo.record_account_event(event_key="caller-a", **values)
        second, second_created = repo.record_account_event(event_key="caller-b", **values)
        assert first_created is True
        assert second_created is False
        assert first.id == second.id
        assert first.fact_key

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AccountEvent)) == 1


def test_full_account_reconciliation_records_order_mode_and_unmanaged_diffs(
    session_factory,
):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.append_position_snapshot(
            snapshot_key="local-position",
            account_id="main",
            exchange="binance",
            symbol="ETHUSDT",
            position_side="LONG",
            quantity="1",
            entry_price="2000",
            source="REST",
            source_event_time=NOW,
        )
        repo.append_order_snapshot(
            snapshot_key="local-order",
            account_id="main",
            exchange="binance",
            symbol="ETHUSDT",
            position_side="LONG",
            exchange_order_id="order-reconcile",
            side="BUY",
            status="ACKNOWLEDGED",
            original_quantity="2",
            executed_quantity="0",
            cumulative_quote="0",
            source="REST",
            source_event_time=NOW,
        )
        summary = HedgeReconciler(repo).reconcile_account_facts(
            account_id="main",
            exchange="binance",
            positions=[
                PositionFact(
                    symbol="ETH/USDT:USDT",
                    position_side="LONG",
                    quantity="1",
                    entry_price="2000",
                    event_time=NOW + timedelta(seconds=1),
                )
            ],
            orders=[
                OrderFact(
                    symbol="ETHUSDT",
                    position_side="LONG",
                    exchange_order_id="order-reconcile",
                    side="BUY",
                    status="PARTIAL",
                    original_quantity="2",
                    executed_quantity="1",
                    cumulative_quote="2000",
                    average_price="2000",
                    event_time=NOW + timedelta(seconds=1),
                )
            ],
            mode=AccountModeFact(
                position_mode="ONEWAY",
                margin_mode="ISOLATED",
                leverage="5",
            ),
            expected_leverage="3",
            unmanaged_positions=["BTC/USDT:USDT|LONG"],
            unmanaged_orders=["SOL/USDT:USDT|SHORT|external-order"],
            policy=ReconciliationPolicy(auto_repair_orders=True),
            observed_at=NOW + timedelta(seconds=1),
        )
        assert summary.status == "DRIFT"
        assert summary.compared_positions == 1
        assert summary.compared_orders == 1
        assert summary.repaired_count == 1
        assert summary.severe_diff_count >= 6

    with session_factory() as session:
        order = session.scalar(
            select(CurrentOrderProjection).where(
                CurrentOrderProjection.exchange_order_id == "order-reconcile"
            )
        )
        assert order is not None
        assert order.status == "PARTIAL"
        assert order.executed_quantity == "1"

def test_service_returns_usable_entities_with_default_session_expiration(engine):
    factory = sessionmaker(bind=engine)
    service = HedgePersistenceService(factory)
    intent, created = service.create_order_intent(
        account_id="main",
        exchange="binance",
        symbol="ETHUSDT",
        position_side="LONG",
        action="OPEN",
        side="BUY",
        order_type="LIMIT",
        requested_quantity="1",
        requested_price="2000",
        idempotency_key="service-intent",
    )
    assert created is True
    assert intent.symbol == "ETH/USDT:USDT"
    assert intent.status == "PLANNED"
