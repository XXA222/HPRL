from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy import delete, select

from freqtrade.persistence.hedge_models import PositionSnapshot
from freqtrade.persistence.hedge_repositories import HedgeLedgerRepository


NOW = datetime(2026, 7, 26, 8, 0, 0)


def test_restart_replays_orders_and_positions_from_ledger(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.append_order_snapshot(
            snapshot_key="ws-order-1-new",
            account_id="hedge-main",
            exchange="binance",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_order_id="order-1",
            side="BUY",
            status="PARTIALLY_FILLED",
            original_quantity="2",
            executed_quantity="1",
            cumulative_quote="2000",
            average_price="2000",
            source="WEBSOCKET",
            source_event_time=NOW + timedelta(seconds=1),
            sequence_number=2,
        )
        repo.append_order_snapshot(
            snapshot_key="rest-order-1-old",
            account_id="hedge-main",
            exchange="binance",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_order_id="order-1",
            side="BUY",
            status="NEW",
            original_quantity="2",
            executed_quantity="0",
            cumulative_quote="0",
            source="REST",
            source_event_time=NOW,
            sequence_number=1,
        )
        repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="fill-1",
            exchange_order_id="order-1",
            side="BUY",
            quantity="0.4",
            price="1900",
            source="WEBSOCKET",
            event_time=NOW,
        )
        repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="fill-2",
            exchange_order_id="order-1",
            side="BUY",
            quantity="0.6",
            price="2100",
            source="WEBSOCKET",
            event_time=NOW + timedelta(seconds=1),
        )

    with session_factory.begin() as session:
        session.execute(delete(PositionSnapshot))

    with session_factory.begin() as session:
        recovery = HedgeLedgerRepository(session).rebuild_current_position_snapshots(
            account_id="hedge-main"
        )
        assert len(recovery.orders) == 1
        assert recovery.orders[0].status == "PARTIAL"
        assert recovery.orders[0].executed_quantity == "1"
        assert len(recovery.positions) == 1
        assert recovery.positions[0].quantity == "1"
        assert recovery.positions[0].entry_price == "2020"

    with session_factory() as session:
        current = session.scalar(
            select(PositionSnapshot).where(PositionSnapshot.is_current.is_(True))
        )
        assert current is not None
        assert current.source == "RECOVERY"
        assert current.quantity == "1"


def test_out_of_order_fill_projection_matches_restart_replay(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="late-close",
            exchange_order_id="close-order",
            side="SELL",
            quantity="1",
            price="110",
            source="REST",
            event_time=NOW + timedelta(seconds=2),
        )
        result = repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="early-open",
            exchange_order_id="open-order",
            side="BUY",
            quantity="1",
            price="100",
            source="WEBSOCKET",
            event_time=NOW,
        )
        recovery = repo.recover_projection(account_id="hedge-main")
        assert result.position_quantity == "0"
        assert recovery.positions[0].quantity == "0"
        assert result.position_entry_price == recovery.positions[0].entry_price


def test_recovery_uses_latest_external_position_fact_as_replay_base(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.append_position_snapshot(
            snapshot_key="rest-position-base",
            account_id="hedge-main",
            exchange="binance",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            quantity="2",
            entry_price="100",
            realized_pnl="10",
            source="REST",
            source_event_time=NOW,
        )
        result = repo.apply_fill(
            exchange="binance",
            account_id="hedge-main",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            exchange_trade_id="reduce-after-rest",
            exchange_order_id="reduce-order",
            side="SELL",
            quantity="0.5",
            price="110",
            realized_pnl="5",
            source="WEBSOCKET",
            event_time=NOW + timedelta(seconds=1),
        )
        assert result.position_quantity == "1.5"
        recovery = repo.recover_projection(account_id="hedge-main")
        assert recovery.positions[0].quantity == "1.5"
        assert recovery.positions[0].entry_price == "100"
        assert recovery.positions[0].realized_pnl == "15"


def test_recovery_preserves_external_position_without_fill_history(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.append_position_snapshot(
            snapshot_key="rest-only-position",
            account_id="hedge-main",
            exchange="binance",
            symbol="ETH/USDT:USDT",
            position_side="SHORT",
            quantity="3",
            entry_price="2500",
            source="REST",
            source_event_time=NOW,
        )
        recovery = repo.recover_projection(account_id="hedge-main")
        assert len(recovery.positions) == 1
        assert recovery.positions[0].quantity == "3"
        assert recovery.positions[0].entry_price == "2500"


def test_late_older_position_fact_does_not_replace_newer_projection(session_factory):
    older = NOW - timedelta(seconds=1)
    newer = NOW + timedelta(seconds=1)
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        repo.append_position_snapshot(
            snapshot_key="newer-rest-position",
            account_id="hedge-main",
            exchange="binance",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            quantity="2",
            entry_price="100",
            source="REST",
            source_event_time=newer,
        )
        repo.append_position_snapshot(
            snapshot_key="late-older-websocket-position",
            account_id="hedge-main",
            exchange="binance",
            symbol="BTC/USDT:USDT",
            position_side="LONG",
            quantity="1",
            entry_price="90",
            source="WEBSOCKET",
            source_event_time=older,
        )

    with session_factory() as session:
        current = session.scalar(
            select(PositionSnapshot).where(PositionSnapshot.is_current.is_(True))
        )
        assert current is not None
        assert current.source == "LOCAL"
        assert current.quantity == "2"
        assert current.entry_price == "100"
        recovery = HedgeLedgerRepository(session).recover_projection(account_id="hedge-main")
        assert recovery.positions[0].quantity == current.quantity
        assert recovery.positions[0].entry_price == current.entry_price


def test_order_recovery_is_scoped_by_exchange(session_factory):
    with session_factory.begin() as session:
        repo = HedgeLedgerRepository(session)
        for exchange in ("binance", "okx"):
            repo.append_order_snapshot(
                snapshot_key=f"{exchange}-shared-order",
                account_id="hedge-main",
                exchange=exchange,
                symbol="ETH/USDT:USDT",
                position_side="LONG",
                exchange_order_id="shared-order-id",
                side="BUY",
                status="ACKNOWLEDGED",
                original_quantity="1",
                executed_quantity="0",
                cumulative_quote="0",
                source="REST",
                source_event_time=NOW,
            )

    with session_factory() as session:
        recovery = HedgeLedgerRepository(session).recover_projection(account_id="hedge-main")
        assert len(recovery.orders) == 2
        assert {row.exchange for row in recovery.orders} == {"binance", "okx"}
