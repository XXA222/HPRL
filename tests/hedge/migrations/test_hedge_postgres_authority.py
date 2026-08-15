from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, func, select, text
from sqlalchemy.orm import sessionmaker

from freqtrade.hedge.execution.client_order_id import build_client_order_id
from freqtrade.hedge.execution.idempotency import ReservationState
from freqtrade.hedge.execution.service import (
    ExecutionOrder,
    ExecutionResult,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
)
from freqtrade.hedge.execution.state_machine import OrderLifecycle, OrderState
from freqtrade.persistence.hedge_execution_adapters import (
    SqlExecutionIdempotencyStore,
    SqlExecutionStore,
)
from freqtrade.persistence.hedge_migrations import (
    create_pre_migration_backup,
    restore_postgresql_backup,
    run_hedge_migrations,
)
from freqtrade.persistence.hedge_models import AccountEvent, EventOutbox
from freqtrade.persistence.hedge_service import HedgePersistenceService

POSTGRES_URL = os.environ.get("HEDGE_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="Set HEDGE_TEST_POSTGRES_URL to a disposable PostgreSQL database.",
)


def _order(key: str) -> ExecutionOrder:
    intent = OrderIntent(
        account_id="paper-pg",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=IntentAction.OPEN,
        quantity=Decimal("0.25"),
        idempotency_key=key,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("2000"),
        metadata={"exchange": "paper", "planner_intent_id": "pg-planner"},
    )
    return ExecutionOrder(
        intent=intent,
        client_order_id=build_client_order_id(
            account_id=intent.account_id,
            symbol=intent.symbol,
            position_side=intent.position_side.value,
            idempotency_key=key,
        ),
        approved_quantity=intent.quantity,
        lifecycle=OrderLifecycle(
            status=OrderState.ACKNOWLEDGED,
            version=2,
            updated_at=datetime.now(UTC),
        ),
        created_at=datetime.now(UTC),
    )


def test_postgres_execution_authority_concurrency_outbox_and_restore() -> None:
    assert POSTGRES_URL is not None
    admin = create_engine(POSTGRES_URL)
    schema = f"hedge_r20_{uuid4().hex[:12]}"
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine(POSTGRES_URL, pool_pre_ping=True)

    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute(f'SET search_path TO "{schema}", public')
        cursor.close()

    backup_ref: str | None = None
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE trades (id INTEGER PRIMARY KEY, exchange VARCHAR(25), "
                    "pair VARCHAR(64), is_open BOOLEAN NOT NULL, is_short BOOLEAN NOT NULL, "
                    "record_version INTEGER)"
                )
            )
            connection.execute(
                text(
                    "CREATE TABLE orders (id INTEGER PRIMARY KEY, ft_trade_id INTEGER NOT NULL, "
                    "order_id VARCHAR(255), ft_is_open BOOLEAN NOT NULL)"
                )
            )
        report = run_hedge_migrations(engine)
        assert "H3-027-r2-execution-authority" in report.applied
        sessions = sessionmaker(bind=engine, expire_on_commit=False)
        store = SqlExecutionStore(sessions)
        key = "pg-concurrent-idempotency"

        def reserve(worker: int):
            owner_id = f"worker-{worker}"
            state = SqlExecutionIdempotencyStore(
                sessions,
                SqlExecutionStore(sessions),
                owner_id=owner_id,
            ).reserve(key).state
            return owner_id, state

        with ThreadPoolExecutor(max_workers=8) as pool:
            reservations = list(pool.map(reserve, range(8)))
        states = [state for _, state in reservations]
        assert states.count(ReservationState.NEW) == 1
        assert states.count(ReservationState.IN_FLIGHT) == 7
        winner = next(owner for owner, state in reservations if state is ReservationState.NEW)

        order = _order(key)
        idempotency = SqlExecutionIdempotencyStore(sessions, store, owner_id=winner)
        idempotency.complete(key, ExecutionResult(order=order))
        replay = SqlExecutionIdempotencyStore(
            sessions, SqlExecutionStore(sessions), owner_id="restart"
        ).reserve(key)
        assert replay.state is ReservationState.COMPLETED
        assert replay.value is not None and replay.value.order == order

        service = HedgePersistenceService(sessions)
        service.record_account_event(
            event_key="pg-funding-1",
            account_id="paper-pg",
            exchange="paper",
            event_type="FUNDING",
            asset="USDT",
            amount=Decimal("-0.25"),
            source="LOCAL",
            event_time=datetime.now(UTC),
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            raw_payload={"source": "r2-pg-test"},
        )
        with sessions() as session:
            assert session.scalar(select(func.count(AccountEvent.id))) == 1
            assert session.scalar(select(func.count(EventOutbox.id))) >= 1

        backup_ref = create_pre_migration_backup(engine)
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM hedge_execution_idempotency"))
            connection.execute(text("DELETE FROM hedge_execution_order_states"))
            connection.execute(text("DELETE FROM hedge_event_outbox"))
            connection.execute(text("DELETE FROM hedge_account_events"))
        assert SqlExecutionStore(sessions).list_orders() == ()
        restored = restore_postgresql_backup(engine, backup_ref)
        assert restored["hedge_execution_order_states"] == 1
        assert restored["hedge_execution_idempotency"] == 1
        assert restored["hedge_account_events"] == 1
        assert restored["hedge_event_outbox"] >= 1
        assert len(SqlExecutionStore(sessions).list_orders()) == 1
        replay_after_restore = SqlExecutionIdempotencyStore(
            sessions, SqlExecutionStore(sessions), owner_id="post-restore"
        ).reserve(key)
        assert replay_after_restore.state is ReservationState.COMPLETED
        assert replay_after_restore.value is not None
        assert replay_after_restore.value.order == order
        with sessions() as session:
            assert session.scalar(select(func.count(AccountEvent.id))) == 1
            assert session.scalar(select(func.count(EventOutbox.id))) >= 1
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            if backup_ref and backup_ref.startswith("postgresql-schema:"):
                backup_schema = backup_ref.partition(":")[2]
                connection.execute(text(f'DROP SCHEMA IF EXISTS "{backup_schema}" CASCADE'))
        admin.dispose()
