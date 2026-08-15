from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from freqtrade.persistence.hedge_models import (
    AccountEvent,
    AccountRiskSnapshot,
    EventOutbox,
    FillEvent,
    OrderIntent,
    OrderSnapshot,
    PositionSnapshot,
    StrategySideState,
)
from freqtrade.persistence.hedge_repositories import HedgeLedgerRepository


NOW = datetime(2026, 7, 26, 8, 0, 0)


def test_order_intent_idempotency_key_is_unique(session_factory):
    with session_factory.begin() as session:
        first, created_first = HedgeLedgerRepository(session).create_order_intent(
            account_id="hedge-main",
            exchange="binance",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            action="INCREASE",
            side="BUY",
            order_type="LIMIT",
            requested_quantity="0.20",
            requested_price="2400.00",
            idempotency_key="intent-001",
        )
        first_intent_id = first.intent_id
        assert created_first is True

    with session_factory.begin() as session:
        second, created_second = HedgeLedgerRepository(session).create_order_intent(
            account_id="hedge-main",
            exchange="binance",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            action="INCREASE",
            side="BUY",
            order_type="LIMIT",
            requested_quantity="0.20",
            requested_price="2400.00",
            idempotency_key="intent-001",
        )
        assert created_second is False
        assert second.intent_id == first_intent_id

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(OrderIntent)) == 1
        assert session.scalar(select(func.count()).select_from(EventOutbox)) == 1


def test_duplicate_fill_does_not_double_position_and_partial_fills_accumulate(session_factory):
    with session_factory.begin() as session:
        first = HedgeLedgerRepository(session).apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="fill-1",
            exchange_order_id="order-1",
            side="BUY",
            quantity="0.4",
            price="2000",
            source="WEBSOCKET",
            event_time=NOW,
        )
        assert first.accepted is True

    with session_factory.begin() as session:
        duplicate = HedgeLedgerRepository(session).apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="fill-1",
            exchange_order_id="order-1",
            side="BUY",
            quantity="0.4",
            price="2000",
            source="REST",
            event_time=NOW,
        )
        assert duplicate.duplicate is True

    with session_factory.begin() as session:
        second = HedgeLedgerRepository(session).apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="fill-2",
            exchange_order_id="order-1",
            side="BUY",
            quantity="0.6",
            price="2200",
            source="WEBSOCKET",
            event_time=NOW,
        )
        assert second.cumulative_order_quantity == "1"
        assert second.cumulative_order_quote == "2120"
        assert second.position_quantity == "1"
        assert second.position_entry_price == "2120"

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(FillEvent)) == 2
        assert session.scalar(select(func.count()).select_from(EventOutbox)) == 6
        current = session.scalar(
            select(PositionSnapshot).where(PositionSnapshot.is_current.is_(True))
        )
        assert current is not None
        assert current.quantity == "1"


def test_short_side_fill_sign_is_independent(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        opened = repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="SHORT",
            exchange_trade_id="short-fill-1",
            exchange_order_id="short-order-1",
            side="SELL",
            quantity="2",
            price="2500",
            source="WEBSOCKET",
            event_time=NOW,
        )
        reduced = repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="SHORT",
            exchange_trade_id="short-fill-2",
            exchange_order_id="short-order-2",
            side="BUY",
            quantity="0.75",
            price="2300",
            source="WEBSOCKET",
            event_time=NOW,
        )
        assert opened.position_quantity == "2"
        assert reduced.position_quantity == "1.25"
        assert reduced.position_entry_price == "2500"


def test_fill_projection_accumulates_realized_pnl(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="pnl-open",
            exchange_order_id="pnl-order-open",
            side="BUY",
            quantity="2",
            price="2000",
            source="WEBSOCKET",
            event_time=NOW,
        )
        repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="pnl-reduce-1",
            exchange_order_id="pnl-order-reduce-1",
            side="SELL",
            quantity="0.5",
            price="2100",
            realized_pnl="50",
            source="WEBSOCKET",
            event_time=NOW,
        )
        repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="pnl-reduce-2",
            exchange_order_id="pnl-order-reduce-2",
            side="SELL",
            quantity="0.5",
            price="2050",
            realized_pnl="25",
            source="REST",
            event_time=NOW,
        )

    with session_factory() as session:
        current = session.scalar(
            select(PositionSnapshot).where(PositionSnapshot.is_current.is_(True))
        )
        assert current is not None
        assert current.quantity == "1"
        assert current.realized_pnl == "75"


def test_active_position_constraint_is_side_aware(session_factory):
    with session_factory.begin() as session:
        session.add_all(
            [
                PositionSnapshot(
                    snapshot_key="long-current",
                    account_id="a",
                    exchange="binance",
                    symbol="ETH/USDT:USDT",
                    position_side="LONG",
                    quantity="1",
                    entry_price="2000",
                    source="REST",
                    source_event_time=NOW,
                    is_active=True,
                    is_current=True,
                ),
                PositionSnapshot(
                    snapshot_key="short-current",
                    account_id="a",
                    exchange="binance",
                    symbol="ETH/USDT:USDT",
                    position_side="SHORT",
                    quantity="1",
                    entry_price="2100",
                    source="REST",
                    source_event_time=NOW,
                    is_active=True,
                    is_current=True,
                ),
            ]
        )

    with pytest.raises(IntegrityError):
        with session_factory.begin() as session:
            session.add(
                PositionSnapshot(
                    snapshot_key="long-conflict",
                    account_id="a",
                    exchange="binance",
                    symbol="ETH/USDT:USDT",
                    position_side="LONG",
                    quantity="2",
                    entry_price="2050",
                    source="REST",
                    source_event_time=NOW,
                    is_active=True,
                    is_current=True,
                )
            )


def test_account_event_types_and_deduplication(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        for index, event_type in enumerate(("FUNDING", "FEE", "BALANCE", "TRANSFER")):
            row, created = repo.record_account_event(
                event_key=f"account-event-{index}",
                account_id="hedge-main",
                exchange="binance",
                event_type=event_type,
                asset="USDT",
                amount="-0.12" if event_type in {"FUNDING", "FEE"} else "10",
                source="REST",
                event_time=NOW,
            )
            assert created is True
            assert row.event_type == event_type
        _, created = repo.record_account_event(
            event_key="account-event-0",
            account_id="hedge-main",
            exchange="binance",
            event_type="FUNDING",
            asset="USDT",
            amount="-0.12",
            source="REST",
            event_time=NOW,
        )
        assert created is False

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AccountEvent)) == 4


def test_transaction_rollback_removes_fill_projection_and_outbox(session_factory):
    with pytest.raises(RuntimeError):
        with session_factory.begin() as session:
            repo = HedgeLedgerRepository(session)
            repo.apply_fill(
                exchange="binance",
                account_id="hedge-main",
                symbol="ETH/USDT:USDT",
                position_side="LONG",
                exchange_trade_id="rollback-fill",
                exchange_order_id="rollback-order",
                side="BUY",
                quantity="1",
                price="2000",
                source="WEBSOCKET",
                event_time=NOW,
            )
            raise RuntimeError("force rollback")

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(FillEvent)) == 0
        assert session.scalar(select(func.count()).select_from(PositionSnapshot)) == 0
        assert session.scalar(select(func.count()).select_from(EventOutbox)) == 0


def test_rest_websocket_facts_reconciliation_and_strategy_state(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        position, created = repo.append_position_snapshot(
            snapshot_key="rest-position-1",
            account_id="hedge-main",
            exchange="binance",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            quantity="1.25",
            entry_price="2020",
            mark_price="2050",
            source="REST",
            source_event_time=NOW,
        )
        assert created is True
        assert position.source == "REST"
        risk, created = repo.append_account_risk_snapshot(
            snapshot_key="ws-risk-1",
            account_id="hedge-main",
            exchange="binance",
            wallet_balance="1000",
            available_balance="700",
            margin_balance="1010",
            total_initial_margin="300",
            total_maintenance_margin="40",
            gross_exposure="500",
            net_exposure="100",
            margin_utilization="0.297",
            liquidation_buffer="0.35",
            risk_state="READY",
            source="WEBSOCKET",
            source_event_time=NOW,
        )
        assert created is True
        assert risk.source == "WEBSOCKET"

        run = repo.start_reconciliation(
            account_id="hedge-main",
            exchange="binance",
            trigger="STARTUP",
        )
        diff, created = repo.add_reconciliation_diff(
            run_id=run.run_id,
            diff_key="position-qty",
            entity_type="POSITION",
            entity_key="ETH/USDT:USDT|LONG",
            field_name="quantity",
            local_value="1.20",
            exchange_value="1.25",
            severity="ERROR",
        )
        assert created is True
        assert diff.severity == "ERROR"
        completed = repo.complete_reconciliation(
            run_id=run.run_id,
            status="DEGRADED",
            summary={"requires_manual_review": True},
        )
        assert completed.diff_count == 1
        assert completed.severe_diff_count == 1

        state = repo.upsert_strategy_side_state(
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            strategy_name="HedgeBaselineStrategy",
            state_name="HOLDING",
            state={"target_quantity": "2"},
            expected_revision=0,
        )
        assert state.revision == 1
        updated = repo.upsert_strategy_side_state(
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            strategy_name="HedgeBaselineStrategy",
            state_name="ADDING",
            state={"target_quantity": "2.5"},
            expected_revision=1,
        )
        assert updated.revision == 2


def test_fill_trade_id_is_scoped_by_symbol(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        first = repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="shared-trade-id",
            exchange_order_id="btc-order",
            side="BUY",
            quantity="1",
            price="100",
            source="REST",
            event_time=NOW,
        )
        second = repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="shared-trade-id",
            exchange_order_id="eth-order",
            side="BUY",
            quantity="2",
            price="50",
            source="REST",
            event_time=NOW,
        )
        assert first.accepted is True
        assert second.accepted is True
        assert second.duplicate is False

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(FillEvent)) == 2


def test_order_fill_totals_are_scoped_by_exchange_and_symbol(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="btc-fill",
            exchange_order_id="shared-order-id",
            side="BUY",
            quantity="1",
            price="100",
            source="REST",
            event_time=NOW,
        )
        result = repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="eth-fill",
            exchange_order_id="shared-order-id",
            side="BUY",
            quantity="2",
            price="50",
            source="REST",
            event_time=NOW,
        )
        assert result.cumulative_order_quantity == "2"
        assert result.cumulative_order_quote == "100"


def test_idempotency_key_reuse_with_changed_payload_is_rejected(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.create_order_intent(
            account_id="hedge-main",
            exchange="binance",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            action="INCREASE",
            side="BUY",
            order_type="LIMIT",
            requested_quantity="1",
            requested_price="2000",
            idempotency_key="intent-conflict",
            payload={"decision": "one"},
        )

    with pytest.raises(ValueError, match="idempotency conflict"):
        with session_factory.begin() as session:
            HedgeLedgerRepository(session).create_order_intent(
                account_id="hedge-main",
                exchange="binance",
                symbol="ETH/USDT:USDT",
                position_side="LONG",
                action="INCREASE",
                side="BUY",
                order_type="LIMIT",
                requested_quantity="2",
                requested_price="2000",
                idempotency_key="intent-conflict",
                payload={"decision": "two"},
            )


def test_position_snapshot_rejects_invalid_active_price_and_leverage(session_factory):
    with pytest.raises(ValueError, match="entry price"):
        with session_factory.begin() as session:
            HedgeLedgerRepository(session).append_position_snapshot(
                snapshot_key="bad-price",
                account_id="hedge-main",
                exchange="binance",
                symbol="ETH/USDT:USDT",
                position_side="LONG",
                quantity="1",
                entry_price="0",
                leverage="1",
                source="REST",
                source_event_time=NOW,
            )

    with pytest.raises(ValueError, match="leverage"):
        with session_factory.begin() as session:
            HedgeLedgerRepository(session).append_position_snapshot(
                snapshot_key="bad-leverage",
                account_id="hedge-main",
                exchange="binance",
                symbol="ETH/USDT:USDT",
                position_side="LONG",
                quantity="1",
                entry_price="2000",
                leverage="0",
                source="REST",
                source_event_time=NOW,
            )


def test_conflicting_duplicate_fill_fails_closed_without_reprojection(session_factory):
    with session_factory.begin() as session:
        HedgeLedgerRepository(session).apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="conflicting-fill",
            exchange_order_id="order-one",
            side="BUY",
            quantity="1",
            price="100",
            source="WEBSOCKET",
            event_time=NOW,
        )

    with pytest.raises(ValueError, match="Conflicting duplicate fill"):
        with session_factory.begin() as session:
            HedgeLedgerRepository(session).apply_fill(
                exchange="binance",
                account_id="hedge-main",
                symbol="BTC/USDT:USDT",
                position_side="LONG",
                exchange_trade_id="conflicting-fill",
                exchange_order_id="order-one",
                side="BUY",
                quantity="2",
                price="100",
                source="REST",
                event_time=NOW,
            )

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(FillEvent)) == 1
        current = session.scalar(
            select(PositionSnapshot).where(PositionSnapshot.is_current.is_(True))
        )
        assert current is not None
        assert current.quantity == "1"


def test_generated_projection_and_outbox_keys_fit_postgresql_limits(session_factory):
    account_id = "a" * 128
    symbol = "S" * 117 + "/USDT:USDT"
    strategy_name = "t" * 128
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.append_position_snapshot(
            snapshot_key="fact-at-column-limit",
            account_id=account_id,
            exchange="binance",
            symbol=symbol,
            position_side="LONG",
            quantity="1",
            entry_price="100",
            source="REST",
            source_event_time=NOW,
        )
        repo.upsert_strategy_side_state(
            account_id=account_id,
            symbol=symbol,
            position_side="LONG",
            strategy_name=strategy_name,
            state_name="READY",
            state={},
        )

    with session_factory() as session:
        current = session.scalar(
            select(PositionSnapshot).where(PositionSnapshot.is_current.is_(True))
        )
        assert current is not None
        assert len(current.snapshot_key) <= 255
        aggregate_ids = session.scalars(select(EventOutbox.aggregate_id)).all()
        assert aggregate_ids
        assert all(len(value) <= 255 for value in aggregate_ids)


def test_conflicting_order_snapshot_duplicate_fails_closed(session_factory):
    common = {
        "snapshot_key": "order-snapshot-conflict",
        "account_id": "hedge-main",
        "exchange": "binance",
        "symbol": "ETH/USDT:USDT",
        "position_side": "LONG",
        "exchange_order_id": "order-snapshot-1",
        "side": "BUY",
        "status": "PARTIALLY_FILLED",
        "original_quantity": "2",
        "executed_quantity": "1",
        "cumulative_quote": "2000",
        "source": "WEBSOCKET",
        "source_event_time": NOW,
    }
    with session_factory.begin() as session:
        row, created = HedgeLedgerRepository(session).append_order_snapshot(**common)
        assert created is True
        assert row.executed_quantity == "1"

    with pytest.raises(ValueError, match="Order snapshot conflict"):
        with session_factory.begin() as session:
            HedgeLedgerRepository(session).append_order_snapshot(
                **{**common, "executed_quantity": "1.5", "cumulative_quote": "3000"}
            )

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(OrderSnapshot)) == 1


def test_conflicting_account_event_duplicate_fails_closed(session_factory):
    common = {
        "event_key": "funding-conflict",
        "account_id": "hedge-main",
        "exchange": "binance",
        "event_type": "FUNDING",
        "asset": "USDT",
        "amount": "-1.25",
        "source": "REST",
        "event_time": NOW,
    }
    with session_factory.begin() as session:
        _, created = HedgeLedgerRepository(session).record_account_event(**common)
        assert created is True

    with pytest.raises(ValueError, match="Account event conflict"):
        with session_factory.begin() as session:
            HedgeLedgerRepository(session).record_account_event(
                **{**common, "amount": "-2.50"}
            )

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AccountEvent)) == 1


def test_conflicting_position_fact_duplicate_fails_closed(session_factory):
    common = {
        "snapshot_key": "position-fact-conflict",
        "account_id": "hedge-main",
        "exchange": "binance",
        "symbol": "ETH/USDT:USDT",
        "position_side": "LONG",
        "quantity": "1",
        "entry_price": "2000",
        "source": "REST",
        "source_event_time": NOW,
    }
    with session_factory.begin() as session:
        _, created = HedgeLedgerRepository(session).append_position_snapshot(**common)
        assert created is True

    with pytest.raises(ValueError, match="Position snapshot conflict"):
        with session_factory.begin() as session:
            HedgeLedgerRepository(session).append_position_snapshot(
                **{**common, "quantity": "2"}
            )

    with session_factory() as session:
        facts = session.scalar(
            select(func.count()).select_from(PositionSnapshot).where(
                PositionSnapshot.source == "REST"
            )
        )
        assert facts == 1


def test_conflicting_account_risk_snapshot_duplicate_fails_closed(session_factory):
    common = {
        "snapshot_key": "risk-fact-conflict",
        "account_id": "hedge-main",
        "exchange": "binance",
        "wallet_balance": "1000",
        "available_balance": "800",
        "risk_state": "READY",
        "source": "REST",
        "source_event_time": NOW,
    }
    with session_factory.begin() as session:
        _, created = HedgeLedgerRepository(session).append_account_risk_snapshot(**common)
        assert created is True

    with pytest.raises(ValueError, match="Account risk snapshot conflict"):
        with session_factory.begin() as session:
            HedgeLedgerRepository(session).append_account_risk_snapshot(
                **{**common, "available_balance": "700"}
            )

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(AccountRiskSnapshot)) == 1


def test_strategy_side_state_uses_atomic_revision_compare_and_swap(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        created = repo.upsert_strategy_side_state(
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            strategy_name="CASStrategy",
            state_name="READY",
            state={"target": "1"},
            expected_revision=0,
        )
        assert created.revision == 1

    with pytest.raises(ValueError, match="revision mismatch"):
        with session_factory.begin() as session:
            HedgeLedgerRepository(session).upsert_strategy_side_state(
                account_id="hedge-main",
                symbol="ETH/USDT:USDT",
                position_side="LONG",
                strategy_name="CASStrategy",
                state_name="ADDING",
                state={"target": "2"},
                expected_revision=0,
            )

    with session_factory() as session:
        row = session.scalar(select(StrategySideState))
        assert row is not None
        assert row.revision == 1
        assert row.state_name == "READY"


def test_fill_sequence_orders_equal_timestamp_events_deterministically(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="a-close-arrived-first",
            exchange_order_id="close-order",
            side="SELL",
            quantity="0.5",
            price="110",
            source="WEBSOCKET",
            event_time=NOW,
            sequence_number=2,
        )
        result = repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="z-open-arrived-second",
            exchange_order_id="open-order",
            side="BUY",
            quantity="1",
            price="100",
            source="WEBSOCKET",
            event_time=NOW,
            sequence_number=1,
        )
        assert result.position_quantity == "0.5"
        assert result.position_entry_price == "100"

    with session_factory.begin() as session:
        recovery = HedgeLedgerRepository(session).rebuild_current_position_snapshots(
            account_id="hedge-main",
            symbol="BTC/USDT:USDT",
        )
        assert len(recovery.positions) == 1
        assert recovery.positions[0].quantity == "0.5"
        assert recovery.positions[0].entry_price == "100"
