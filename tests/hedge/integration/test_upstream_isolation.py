from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from freqtrade.persistence.base import ModelBase
from freqtrade.persistence.hedge_migrations import run_hedge_migrations
from freqtrade.persistence.hedge_models import HedgeModelBase
from freqtrade.wallets import normalize_wallet_symbol, position_wallet_key


def _legacy_core_schema(engine) -> None:
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE trades ("
                "id INTEGER PRIMARY KEY, exchange VARCHAR(25), pair VARCHAR(25), "
                "is_open BOOLEAN NOT NULL, is_short BOOLEAN NOT NULL DEFAULT 0)"
            )
        )
        connection.execute(
            text(
                "CREATE TABLE orders (id INTEGER PRIMARY KEY, ft_trade_id INTEGER, "
                "order_id VARCHAR(255))"
            )
        )
        connection.execute(
            text(
                "INSERT INTO trades(id, exchange, pair, is_open, is_short) VALUES "
                "(1, 'binance', 'ETH/BTC', 1, 0), "
                "(2, 'binance', 'ETH/BTC', 1, 0)"
            )
        )
        connection.execute(
            text(
                "INSERT INTO orders(id, ft_trade_id, order_id) "
                "VALUES (1, 1, 'legacy-without-ft-is-open')"
            )
        )


def test_hedge_models_do_not_pollute_native_metadata() -> None:
    assert HedgeModelBase.metadata is not ModelBase.metadata
    assert "hedge_order_intents" in HedgeModelBase.metadata.tables
    assert "hedge_order_intents" not in ModelBase.metadata.tables


def test_plain_wallet_pairs_are_not_forced_through_usdtm_codec() -> None:
    assert normalize_wallet_symbol("ETH/BTC") == "ETH/BTC"
    assert normalize_wallet_symbol("neo_usdt") == "NEO/USDT"
    assert position_wallet_key("ETH/BTC", "long") == ("ETH/BTC", "LONG")
    assert normalize_wallet_symbol("ETH/USDT:USDT") == "ETH/USDT:USDT"


def test_ordinary_duplicate_open_trades_do_not_reserve_hedge_slots(tmp_path) -> None:
    engine = create_engine(f"sqlite+pysqlite:///{tmp_path / 'legacy.sqlite'}")
    _legacy_core_schema(engine)
    run_hedge_migrations(engine, backup_directory=tmp_path / "backups")

    indexes = {item["name"] for item in inspect(engine).get_indexes("trades")}
    assert "uq_trades_active_account_symbol_side" not in indexes
    assert "uq_trades_open_slot_key" in indexes
    with engine.connect() as connection:
        rows = connection.execute(
            text(
                "SELECT position_side, hedge_version, open_slot_key "
                "FROM trades ORDER BY id"
            )
        ).all()
    assert rows == [("BOTH", 0, None), ("BOTH", 0, None)]
    with engine.connect() as connection:
        order_row = connection.execute(
            text(
                "SELECT position_side, submit_state FROM orders WHERE id=1"
            )
        ).one()
    assert order_row == ("BOTH", "TERMINAL")
