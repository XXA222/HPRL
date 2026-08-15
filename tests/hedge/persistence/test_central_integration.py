from __future__ import annotations

from pathlib import Path

import pytest

from freqtrade.hedge.integration.central_source import (
    IntegrationSafetyError,
    _patch_native_copy_text,
    integrate,
)


ROOT = Path(__file__).resolve().parents[3]


def _target(tmp_path: Path) -> tuple[Path, Path]:
    persistence = tmp_path / "freqtrade" / "persistence"
    persistence.mkdir(parents=True)
    return persistence / "trade_model.py", persistence / "migrations.py"


def test_central_integration_fails_closed_on_unreleased_trade_model(tmp_path: Path) -> None:
    trade_model, migrations = _target(tmp_path)
    trade_model.write_text(
        "class Order:\n    pass\n\n"
        "class LocalTrade:\n    pass\n\n"
        "class Trade:\n    pass\n",
        encoding="utf-8",
    )
    migrations.write_text(
        "def migrate_trades_and_orders_table(cols, cols_order):\n    return None\n",
        encoding="utf-8",
    )
    before_trade = trade_model.read_bytes()
    before_migrations = migrations.read_bytes()

    with pytest.raises(IntegrationSafetyError, match="Clean Mainline"):
        integrate(tmp_path)

    assert trade_model.read_bytes() == before_trade
    assert migrations.read_bytes() == before_migrations


def test_central_integration_is_idempotent_on_released_tree(tmp_path: Path) -> None:
    trade_model, migrations = _target(tmp_path)
    trade_model.write_bytes((ROOT / "freqtrade/persistence/trade_model.py").read_bytes())
    migrations.write_bytes((ROOT / "freqtrade/persistence/migrations.py").read_bytes())

    first = integrate(tmp_path)
    first_trade = trade_model.read_bytes()
    first_migrations = migrations.read_bytes()
    second = integrate(tmp_path)

    assert first.trade_model_changed is False
    assert first.migrations_changed is False
    assert second.trade_model_changed is False
    assert second.migrations_changed is False
    assert trade_model.read_bytes() == first_trade
    assert migrations.read_bytes() == first_migrations


def test_native_rebuild_copy_preserves_existing_hedge_columns_without_promotion() -> None:
    source = '''def migrate_trades_and_orders_table(cols, cols_order):
    is_short = get_column_def(cols, "is_short", "0")
    record_version = get_column_def(cols, "record_version", "1")
    sql = f"""insert into trades
            max_stake_amount, record_version
            {record_version} record_version
    """

def migrate_orders_table(cols_order):
    ft_order_tag = get_column_def(cols_order, "ft_order_tag", "null")
    sql = f"""insert into orders
            ft_amount, ft_price, ft_cancel_reason, ft_order_tag
            {ft_order_tag} ft_order_tag
    """
'''
    patched = _patch_native_copy_text(source)

    assert "coalesce(account_id, 'default')" in patched
    assert "coalesce(position_side, 'BOTH')" in patched
    assert "coalesce(hedge_version, 0)" in patched
    assert "case when coalesce(hedge_version, 0) >= 2" in patched
    assert "open_slot_key, hedge_version" in patched
    assert "client_order_id, idempotency_key, submit_state" in patched
    assert "case when {is_short} then 'SHORT' else 'LONG' end" not in patched
    assert 'hedge_version = get_column_def(cols, "hedge_version", "1")' not in patched
    assert _patch_native_copy_text(patched) == patched
