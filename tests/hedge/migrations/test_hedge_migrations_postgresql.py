from __future__ import annotations

import os
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.exc import IntegrityError

from freqtrade.persistence.hedge_migrations import (
    HedgeMigrationError,
    create_pre_migration_backup,
    migration_status,
    restore_postgresql_backup,
    run_hedge_migrations,
)


POSTGRES_URL = os.environ.get("HEDGE_TEST_POSTGRES_URL")
pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason="Set HEDGE_TEST_POSTGRES_URL to a disposable PostgreSQL database.",
)


def test_postgresql_migration_and_internal_backup_schema():
    assert POSTGRES_URL is not None
    admin_engine = create_engine(POSTGRES_URL)
    schema = f"h3_test_{uuid4().hex[:12]}"
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))

    engine = create_engine(POSTGRES_URL)
    backup_schema: str | None = None
    full_backup_schema: str | None = None

    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_connection, _):
        cursor = dbapi_connection.cursor()
        cursor.execute(f'SET search_path TO "{schema}", public')
        cursor.close()

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE trades (id INTEGER PRIMARY KEY, exchange VARCHAR(25), "
                    "pair VARCHAR(25), is_open BOOLEAN NOT NULL, is_short BOOLEAN NOT NULL, "
                    "record_version INTEGER)"
                )
            )
            connection.execute(
                text(
                    "CREATE TABLE orders (id INTEGER PRIMARY KEY, ft_trade_id INTEGER NOT NULL, "
                    "order_id VARCHAR(255), ft_is_open BOOLEAN NOT NULL)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO trades VALUES "
                    "(1, 'binance', 'ETH/USDT:USDT', TRUE, FALSE, 2)"
                )
            )
        with pytest.raises(HedgeMigrationError, match="can be retried"):
            run_hedge_migrations(engine, fail_after_step="H3-004-ledger-tables")
        failed_status = {
            row["migration_id"]: row for row in migration_status(engine)
        }
        original_backup = failed_status["H3-001-core-columns"]["backup_reference"]

        first = run_hedge_migrations(engine)
        assert "H3-004-ledger-tables" in first.applied
        assert first.backup_reference == original_backup
        assert first.backup_reference.startswith("postgresql-schema:")
        backup_schema = first.backup_reference.partition(":")[2]
        with admin_engine.connect() as connection:
            backup_inspector = inspect(connection)
            backup_tables = backup_inspector.get_table_names(schema=backup_schema)
            assert {"trades", "orders"}.issubset(backup_tables)
            backup_trade_columns = {
                column["name"]
                for column in backup_inspector.get_columns("trades", schema=backup_schema)
            }
            assert "account_id" not in backup_trade_columns
            backup_count = connection.execute(
                text(f'SELECT COUNT(*) FROM "{backup_schema}"."trades"')
            ).scalar_one()
            assert backup_count == 1
        second = run_hedge_migrations(engine)
        assert second.applied == ()
        assert "hedge_event_outbox" in inspect(engine).get_table_names()

        full_backup = create_pre_migration_backup(engine)
        assert full_backup.startswith("postgresql-schema:")
        full_backup_schema = full_backup.partition(":")[2]
        with engine.begin() as connection:
            connection.execute(
                text("UPDATE trades SET pair='BTC/USDT:USDT' WHERE id=1")
            )
        restored = restore_postgresql_backup(engine, full_backup)
        assert restored["trades"] == 1
        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT pair FROM trades WHERE id=1")
            ).scalar_one() == "ETH/USDT:USDT"
        with pytest.raises(IntegrityError):
            with engine.begin() as connection:
                connection.execute(
                    text(
                        "INSERT INTO trades(id, exchange, pair, is_open, is_short, "
                        "record_version, account_id, position_side, open_slot_key, "
                        "hedge_version) VALUES (2, 'binance', 'ETH/USDT:USDT', "
                        "TRUE, FALSE, 2, 'default', 'LONG', 'duplicate-slot', 1)"
                    )
                )
    finally:
        engine.dispose()
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
            for candidate in (backup_schema, full_backup_schema):
                if candidate:
                    connection.execute(
                        text(f'DROP SCHEMA IF EXISTS "{candidate}" CASCADE')
                    )
        admin_engine.dispose()
