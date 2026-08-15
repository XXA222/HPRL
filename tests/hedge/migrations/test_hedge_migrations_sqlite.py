from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import IntegrityError

from freqtrade.persistence.base import ModelBase
from freqtrade.persistence.hedge_migrations import (
    HedgeMigrationConflict,
    HedgeMigrationError,
    migration_plan_ids,
    migration_status,
    prepare_hedge_migration_backup,
    restore_sqlite_backup,
    run_hedge_migrations,
)


MIGRATION_IDS = migration_plan_ids()


def _create_legacy_schema(engine, *, duplicate_open_slot: bool = False) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE trades ("
                "id INTEGER PRIMARY KEY, exchange VARCHAR(25), pair VARCHAR(25), "
                "is_open BOOLEAN NOT NULL, is_short BOOLEAN NOT NULL, record_version INTEGER)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE orders ("
                "id INTEGER PRIMARY KEY, ft_trade_id INTEGER NOT NULL, "
                "order_id VARCHAR(255), ft_is_open BOOLEAN NOT NULL)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO trades(id, exchange, pair, is_open, is_short, record_version) "
                "VALUES (1, 'binance', 'ETH/USDT:USDT', 1, 0, 2), "
                "(2, 'binance', 'ETH/USDT:USDT', 0, 1, 2)"
            )
        )
        if duplicate_open_slot:
            connection.execute(
                text(
                    "INSERT INTO trades(id, exchange, pair, is_open, is_short, record_version) "
                    "VALUES (3, 'binance', 'ETH/USDT:USDT', 1, 0, 2)"
                )
            )
        connection.execute(
            text(
                "INSERT INTO orders(id, ft_trade_id, order_id, ft_is_open) "
                "VALUES (1, 1, 'legacy-order-1', 1), (2, 2, 'legacy-order-2', 0)"
            )
        )


def test_in_memory_sqlite_migration_is_lossless_and_idempotent(tmp_path: Path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_legacy_schema(engine)

    first = run_hedge_migrations(
        engine,
        ModelBase,
        ["trades", "orders"],
        backup_directory=tmp_path,
    )
    assert first.applied == MIGRATION_IDS
    assert first.backup_reference == "sqlite-memory:no-file-backup"

    second = run_hedge_migrations(
        engine,
        ModelBase,
        ["trades", "orders"],
        backup_directory=tmp_path,
    )
    assert second.applied == ()
    assert second.skipped == MIGRATION_IDS

    inspector = inspect(engine)
    assert "hedge_fill_events" in inspector.get_table_names()
    assert {
        "account_id",
        "position_side",
        "open_slot_key",
        "hedge_version",
    }.issubset({column["name"] for column in inspector.get_columns("trades")})

    with engine.connect() as connection:
        rows = (
            connection.execute(
                text(
                    "SELECT id, exchange, pair, is_open, is_short, record_version, "
                    "account_id, position_side, open_slot_key, hedge_version "
                    "FROM trades ORDER BY id"
                )
            )
            .mappings()
            .all()
        )
        assert len(rows) == 2
        assert rows[0]["exchange"] == "binance"
        assert rows[0]["pair"] == "ETH/USDT:USDT"
        assert rows[0]["is_open"] == 1
        assert rows[0]["is_short"] == 0
        assert rows[0]["record_version"] == 2
        assert rows[0]["account_id"] == "default"
        assert rows[0]["position_side"] == "BOTH"
        assert rows[0]["open_slot_key"] is None
        assert rows[1]["position_side"] == "BOTH"
        assert rows[1]["open_slot_key"] is None
        assert rows[0]["hedge_version"] == 0
        order_rows = (
            connection.execute(
                text("SELECT id, ft_trade_id, order_id, ft_is_open FROM orders ORDER BY id")
            )
            .mappings()
            .all()
        )
        assert [dict(row) for row in order_rows] == [
            {
                "id": 1,
                "ft_trade_id": 1,
                "order_id": "legacy-order-1",
                "ft_is_open": 1,
            },
            {
                "id": 2,
                "ft_trade_id": 2,
                "order_id": "legacy-order-2",
                "ft_is_open": 0,
            },
        ]


def test_file_sqlite_creates_verified_pre_migration_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Linux permits fsync() on a read-only descriptor while Windows does not.
    # Require every descriptor passed to fsync by this test to be writable so
    # the Windows restore contract is enforced on every CI platform.
    real_fsync = os.fsync

    def writable_fsync(file_descriptor: int) -> None:
        os.write(file_descriptor, b"")
        real_fsync(file_descriptor)

    monkeypatch.setattr(os, "fsync", writable_fsync)

    database = tmp_path / "legacy.sqlite"
    engine = create_engine(f"sqlite+pysqlite:///{database}")
    _create_legacy_schema(engine)
    report = run_hedge_migrations(engine, backup_directory=tmp_path / "backups")
    assert report.backup_reference is not None
    backup = Path(report.backup_reference)
    assert backup.is_file()
    assert backup.with_suffix(backup.suffix + ".sha256").is_file()

    backup_engine = create_engine(f"sqlite+pysqlite:///{backup}")
    try:
        with backup_engine.connect() as connection:
            assert connection.execute(text("SELECT COUNT(*) FROM trades")).scalar_one() == 2
            legacy_columns = {
                column["name"] for column in inspect(backup_engine).get_columns("trades")
            }
            assert "account_id" not in legacy_columns
            assert "hedge_schema_migrations" not in inspect(backup_engine).get_table_names()
    finally:
        # Windows retains SQLite file handles in the connection pool until the
        # engine is disposed.  Release them before restore/corruption checks.
        backup_engine.dispose()

    restored = tmp_path / "restored.sqlite"
    restore_sqlite_backup(backup, restored)
    restored_engine = create_engine(f"sqlite+pysqlite:///{restored}")
    try:
        assert set(inspect(restored_engine).get_table_names()) == {"orders", "trades"}
        with restored_engine.connect() as connection:
            assert connection.execute(text("SELECT COUNT(*) FROM trades")).scalar_one() == 2
    finally:
        restored_engine.dispose()

    with backup.open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(HedgeMigrationError, match="checksum mismatch"):
        restore_sqlite_backup(backup, tmp_path / "must-not-restore.sqlite")


@pytest.mark.parametrize(
    "failed_step",
    MIGRATION_IDS,
)
def test_interrupted_migration_recovers_by_reexecution(
    tmp_path: Path,
    failed_step: str,
):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_legacy_schema(engine)
    with pytest.raises(HedgeMigrationError, match="can be retried"):
        run_hedge_migrations(
            engine,
            backup_directory=tmp_path,
            fail_after_step=failed_step,
        )

    failed = {row["migration_id"]: row for row in migration_status(engine)}
    original_backup = failed["H3-001-core-columns"]["backup_reference"]
    assert failed[failed_step]["state"] == "FAILED"

    recovered = run_hedge_migrations(engine, backup_directory=tmp_path)
    assert failed_step in recovered.applied
    assert recovered.backup_reference == original_backup
    states = {row["migration_id"]: row["state"] for row in migration_status(engine)}
    assert set(states.values()) == {"APPLIED"}


def test_duplicate_ordinary_active_trades_are_outside_hedge_slots(tmp_path: Path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_legacy_schema(engine, duplicate_open_slot=True)
    report = run_hedge_migrations(engine, backup_directory=tmp_path)
    assert report.applied == MIGRATION_IDS
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT position_side, hedge_version, open_slot_key "
                "FROM trades WHERE is_open=1 ORDER BY id"
            )
        ).all()
    assert rows == [("BOTH", 0, None), ("BOTH", 0, None)]
    assert "uq_trades_active_account_symbol_side" not in {
        index["name"] for index in inspect(engine).get_indexes("trades")
    }


def test_conflicting_explicit_hedge_slots_fail_closed(tmp_path: Path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_legacy_schema(engine, duplicate_open_slot=True)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE trades ADD COLUMN account_id VARCHAR(64)"))
        connection.execute(text("ALTER TABLE trades ADD COLUMN position_side VARCHAR(10)"))
        connection.execute(text("ALTER TABLE trades ADD COLUMN open_slot_key VARCHAR(255)"))
        connection.execute(text("ALTER TABLE trades ADD COLUMN hedge_version INTEGER"))
        connection.execute(
            text(
                "UPDATE trades SET account_id='default', position_side='LONG', "
                "open_slot_key='default|ETH/USDT:USDT|LONG', hedge_version=2 "
                "WHERE is_open=1"
            )
        )
    with pytest.raises(HedgeMigrationConflict) as exc_info:
        run_hedge_migrations(engine, backup_directory=tmp_path)
    conflicts = exc_info.value.conflicts
    assert conflicts["active_trade_slots"][0]["row_count"] == 2
    assert conflicts["open_slot_keys"][0]["row_count"] == 2

def test_active_slot_unique_constraint_is_enforced_after_upgrade(tmp_path: Path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_legacy_schema(engine)
    run_hedge_migrations(engine, backup_directory=tmp_path)
    with engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE trades SET position_side='LONG', hedge_version=2, "
                "open_slot_key='default|ETH/USDT:USDT|LONG' WHERE id=1"
            )
        )
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO trades(id, exchange, pair, is_open, is_short, record_version, "
                    "account_id, position_side, open_slot_key, hedge_version) VALUES "
                    "(99, 'binance', 'ETH/USDT:USDT', 1, 0, 2, 'default', 'LONG', "
                    "'default|ETH/USDT:USDT|LONG', 2)"
                )
            )


def test_applied_migration_detects_required_index_drift(tmp_path: Path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_legacy_schema(engine)
    run_hedge_migrations(engine, backup_directory=tmp_path)
    with engine.begin() as connection:
        connection.execute(text("DROP INDEX ix_orders_account_side_recovery"))
    with pytest.raises(HedgeMigrationError, match="missing_indexes"):
        run_hedge_migrations(engine, backup_directory=tmp_path)


def test_fill_identity_constraint_includes_symbol_after_migration(tmp_path: Path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_legacy_schema(engine)
    run_hedge_migrations(engine, backup_directory=tmp_path)
    constraints = inspect(engine).get_unique_constraints("hedge_fill_events")
    assert any(
        tuple(item.get("column_names") or ())
        == ("exchange", "account_id", "symbol", "exchange_trade_id")
        for item in constraints
    )


def test_prepared_backup_is_taken_before_native_schema_mutation(tmp_path: Path):
    database = tmp_path / "legacy-pre-native.sqlite"
    engine = create_engine(f"sqlite+pysqlite:///{database}")
    _create_legacy_schema(engine)
    backup_reference = prepare_hedge_migration_backup(
        engine,
        tmp_path / "pre-native-backups",
    )
    assert backup_reference is not None
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE trades ADD COLUMN native_mutation INTEGER"))
    run_hedge_migrations(
        engine,
        backup_directory=tmp_path / "unused",
        precreated_backup_reference=backup_reference,
    )

    backup_engine = create_engine(f"sqlite+pysqlite:///{backup_reference}")
    try:
        backup_columns = {item["name"] for item in inspect(backup_engine).get_columns("trades")}
        assert "native_mutation" not in backup_columns
        assert "hedge_schema_migrations" not in inspect(backup_engine).get_table_names()
    finally:
        backup_engine.dispose()
        engine.dispose()


def test_v1_interruption_at_verify_can_advance_to_fill_identity_repair(tmp_path: Path):
    database = tmp_path / "interrupted-v1.sqlite"
    engine = create_engine(f"sqlite+pysqlite:///{database}")
    _create_legacy_schema(engine)
    run_hedge_migrations(engine, backup_directory=tmp_path)

    with engine.begin() as connection:
        create_sql = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE type='table' AND name='hedge_fill_events'")
        ).scalar_one()
        old_create_sql = create_sql.replace(
            "CREATE TABLE hedge_fill_events",
            "CREATE TABLE hedge_fill_events_v1",
            1,
        ).replace(
            "CONSTRAINT uq_hedge_fill_exchange_account_symbol_trade "
            "UNIQUE (exchange, account_id, symbol, exchange_trade_id)",
            "CONSTRAINT uq_hedge_fill_exchange_account_trade "
            "UNIQUE (exchange, account_id, exchange_trade_id)",
            1,
        )
        connection.execute(text(old_create_sql))
        connection.execute(text("DROP TABLE hedge_fill_events"))
        connection.execute(text("ALTER TABLE hedge_fill_events_v1 RENAME TO hedge_fill_events"))
        connection.execute(
            text(
                "DELETE FROM hedge_schema_migrations "
                "WHERE migration_id IN ('H3-007-fill-identity-scope', 'H3-008-verify-v1.1')"
            )
        )
        connection.execute(
            text(
                "UPDATE hedge_schema_migrations SET state='FAILED' "
                "WHERE migration_id='H3-006-verify'"
            )
        )

    report = run_hedge_migrations(engine, backup_directory=tmp_path)
    assert report.applied == (
        "H3-006-verify",
        "H3-007-fill-identity-scope",
        "H3-008-verify-v1.1",
    )
    fill_uniques = inspect(engine).get_unique_constraints("hedge_fill_events")
    assert any(
        tuple(item["column_names"]) == ("exchange", "account_id", "symbol", "exchange_trade_id")
        for item in fill_uniques
    )
    engine.dispose()


def test_file_migration_removes_same_host_stale_lock(tmp_path: Path):
    database = tmp_path / "stale-lock.sqlite"
    engine = create_engine(f"sqlite+pysqlite:///{database}")
    _create_legacy_schema(engine)
    lock_path = database.with_suffix(database.suffix + ".h3-migration.lock")
    lock_path.write_text(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "pid": 2_147_483_647,
                "created_at": "2026-07-26T08:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    report = run_hedge_migrations(
        engine,
        backup_directory=tmp_path / "backups",
        lock_timeout_seconds=0.1,
    )
    assert report.applied == MIGRATION_IDS
    assert not lock_path.exists()
    engine.dispose()


def test_file_migration_refuses_live_lock_owner(tmp_path: Path):
    database = tmp_path / "live-lock.sqlite"
    engine = create_engine(f"sqlite+pysqlite:///{database}")
    _create_legacy_schema(engine)
    lock_path = database.with_suffix(database.suffix + ".h3-migration.lock")
    lock_path.write_text(
        json.dumps(
            {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "created_at": "2026-07-26T08:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(HedgeMigrationError, match="Timed out waiting"):
        run_hedge_migrations(
            engine,
            backup_directory=tmp_path / "backups",
            lock_timeout_seconds=0.01,
        )
    assert lock_path.exists()
    lock_path.unlink()
    engine.dispose()


def test_current_position_constraint_also_covers_flat_projection(tmp_path: Path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_legacy_schema(engine)
    run_hedge_migrations(engine, backup_directory=tmp_path)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO hedge_position_snapshots("
                "snapshot_key, account_id, exchange, symbol, position_side, quantity, "
                "entry_price, realized_pnl, unrealized_pnl, leverage, source, "
                "source_event_time, observed_at, is_active, is_current, raw_payload_json, "
                "record_version) VALUES ("
                "'flat-current-1', 'a', 'binance', 'ETH/USDT:USDT', 'LONG', '0', '0', "
                "'0', '0', '1', 'LOCAL', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, 1, '{}', 1)"
            )
        )
    with pytest.raises(IntegrityError):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO hedge_position_snapshots("
                    "snapshot_key, account_id, exchange, symbol, position_side, quantity, "
                    "entry_price, realized_pnl, unrealized_pnl, leverage, source, "
                    "source_event_time, observed_at, is_active, is_current, raw_payload_json, "
                    "record_version) VALUES ("
                    "'flat-current-2', 'a', 'binance', 'ETH/USDT:USDT', 'LONG', '0', '0', "
                    "'0', '0', '1', 'LOCAL', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, 1, "
                    "'{}', 1)"
                )
            )
    engine.dispose()


def test_migration_status_is_read_only_for_unmigrated_database():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_legacy_schema(engine)
    before = inspect(engine).get_table_names()
    assert migration_status(engine) == []
    assert inspect(engine).get_table_names() == before
    engine.dispose()


def test_empty_order_idempotency_keys_are_normalized_before_unique_index(tmp_path: Path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_legacy_schema(engine)
    with engine.begin() as connection:
        connection.execute(text("ALTER TABLE orders ADD COLUMN idempotency_key VARCHAR(255)"))
        connection.execute(text("UPDATE orders SET idempotency_key=''"))

    run_hedge_migrations(engine, backup_directory=tmp_path)
    with engine.connect() as connection:
        values = (
            connection.execute(text("SELECT idempotency_key FROM orders ORDER BY id"))
            .scalars()
            .all()
        )
    assert values == [None, None]
    assert "uq_orders_idempotency_key" in {
        item["name"] for item in inspect(engine).get_indexes("orders")
    }
    engine.dispose()


def test_active_trade_identity_rejects_empty_symbol_after_upgrade(tmp_path: Path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_legacy_schema(engine)
    run_hedge_migrations(engine, backup_directory=tmp_path)
    with pytest.raises(IntegrityError, match="managed hedge trade identity"):
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO trades(id, exchange, pair, is_open, is_short, record_version, "
                    "account_id, position_side, open_slot_key, hedge_version) VALUES "
                    "(100, 'binance', '', 1, 0, 2, 'default', 'LONG', 'default||LONG', 2)"
                )
            )
    engine.dispose()


def test_active_legacy_both_trade_remains_compatible_after_upgrade(tmp_path: Path):
    engine = create_engine("sqlite+pysqlite:///:memory:")
    _create_legacy_schema(engine)
    run_hedge_migrations(engine, backup_directory=tmp_path)
    with engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO trades(id, exchange, pair, is_open, is_short, record_version, "
                "account_id, position_side, open_slot_key, hedge_version) VALUES "
                "(101, 'binance', 'ETH/USDT', 1, 0, 2, 'default', 'BOTH', NULL, 1)"
            )
        )
    engine.dispose()
