from __future__ import annotations

import pytest

from freqtrade.persistence import hedge_migrations
from freqtrade.persistence.hedge_migrations import HedgeMigrationError


class _Inspector:
    def __init__(self, foreign_keys: dict[str, list[dict[str, object]]]) -> None:
        self._foreign_keys = foreign_keys

    def get_foreign_keys(self, table: str, *, schema: str):
        assert schema == "tenant"
        return self._foreign_keys.get(table, [])


def _fk(parent: str, *, schema: str | None = None) -> dict[str, object]:
    return {"referred_table": parent, "referred_schema": schema}


def test_postgresql_restore_order_places_parents_before_children(monkeypatch) -> None:
    inspector = _Inspector(
        {
            "orders": [_fk("trades")],
            "fills": [_fk("orders")],
            "idempotency": [_fk("orders")],
            "external_child": [_fk("unselected")],
        }
    )
    monkeypatch.setattr(hedge_migrations, "inspect", lambda _: inspector)

    ordered = hedge_migrations._postgresql_restore_table_order(
        object(),
        "tenant",
        ("fills", "orders", "trades", "idempotency", "external_child"),
    )

    assert ordered.index("trades") < ordered.index("orders")
    assert ordered.index("orders") < ordered.index("fills")
    assert ordered.index("orders") < ordered.index("idempotency")
    assert set(ordered) == {"fills", "orders", "trades", "idempotency", "external_child"}


def test_postgresql_restore_order_ignores_self_reference(monkeypatch) -> None:
    inspector = _Inspector({"tree": [_fk("tree")]})
    monkeypatch.setattr(hedge_migrations, "inspect", lambda _: inspector)

    assert hedge_migrations._postgresql_restore_table_order(
        object(), "tenant", ("tree",)
    ) == ["tree"]


def test_postgresql_restore_order_fails_closed_on_cycle(monkeypatch) -> None:
    inspector = _Inspector({"a": [_fk("b")], "b": [_fk("a")]})
    monkeypatch.setattr(hedge_migrations, "inspect", lambda _: inspector)

    with pytest.raises(HedgeMigrationError, match="dependency cycle"):
        hedge_migrations._postgresql_restore_table_order(
            object(), "tenant", ("a", "b")
        )
