"""Clean-mainline safety checks for central Freqtrade persistence integration.

This module contains the small amount of central-source integration logic that
remains relevant to the current mainline.  It intentionally fails closed on an
unknown or incomplete persistence schema and never promotes ordinary Freqtrade
rows into Hedge identities.
"""

from __future__ import annotations

import ast
import py_compile
from dataclasses import dataclass
from pathlib import Path


_REQUIRED_ORDER_FIELDS = frozenset(
    {
        "position_side",
        "action",
        "client_order_id",
        "idempotency_key",
        "submit_state",
    }
)
_REQUIRED_TRADE_FIELDS = frozenset(
    {"account_id", "position_side", "open_slot_key", "hedge_version"}
)
_REQUIRED_ISOLATION_MARKERS = (
    "is_explicit_hedge_order",
    "is_explicit_hedge_trade",
    "Ordinary Freqtrade trades remain",
    'side not in {"LONG", "SHORT", "BOTH"}',
)


class IntegrationSafetyError(RuntimeError):
    """Raised when the central persistence source is not the verified mainline."""


@dataclass(frozen=True)
class IntegrationReport:
    """Result of validating and normalizing the central persistence source."""

    trade_model_changed: bool
    migrations_changed: bool


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise IntegrationSafetyError(f"Required class not found: {name}")


def _field_names(node: ast.ClassDef) -> set[str]:
    names: set[str] = set()
    for item in node.body:
        target: ast.expr | None = None
        if isinstance(item, ast.AnnAssign):
            target = item.target
        elif isinstance(item, ast.Assign) and len(item.targets) == 1:
            target = item.targets[0]
        if isinstance(target, ast.Name):
            names.add(target.id)
    return names


def _method_names(node: ast.ClassDef) -> set[str]:
    return {item.name for item in node.body if isinstance(item, ast.FunctionDef)}


def verify_trade_model(path: Path) -> None:
    """Verify that the current side-isolated Trade/Order contract is present."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    order = _class(tree, "Order")
    local_trade = _class(tree, "LocalTrade")
    trade = _class(tree, "Trade")

    missing_order = _REQUIRED_ORDER_FIELDS - _field_names(order)
    missing_local = _REQUIRED_TRADE_FIELDS - _field_names(local_trade)
    missing_trade = _REQUIRED_TRADE_FIELDS - _field_names(trade)
    missing_methods = {"refresh_hedge_identity"} - _method_names(local_trade)
    missing_markers = [
        marker for marker in _REQUIRED_ISOLATION_MARKERS if marker not in source
    ]

    problems: list[str] = []
    if missing_order:
        problems.append(f"Order fields={sorted(missing_order)}")
    if missing_local:
        problems.append(f"LocalTrade fields={sorted(missing_local)}")
    if missing_trade:
        problems.append(f"Trade fields={sorted(missing_trade)}")
    if missing_methods:
        problems.append(f"LocalTrade methods={sorted(missing_methods)}")
    if missing_markers:
        problems.append(f"isolation markers={missing_markers}")
    if problems:
        raise IntegrationSafetyError(
            "Unsafe central-source mutation is disabled. Use the verified Clean "
            "Mainline persistence source instead. Missing: "
            + "; ".join(problems)
        )


def patch_trade_model(path: Path) -> bool:
    """Validate the Trade model; current mainline never mutates it in place."""

    verify_trade_model(path)
    return False


def _patch_native_copy_text(source: str) -> str:
    """Normalize native copy SQL while preserving explicit Hedge identities.

    Existing explicit Hedge values are copied. Rows from older native schemas
    become BOTH with hedge_version=0 and no active slot key. Repeated execution
    is idempotent.
    """

    legacy_trade_block = '''    account_id = get_column_def(cols, "account_id", "'default'")
    position_side = get_column_def(
        cols,
        "position_side",
        f"case when {is_short} then 'SHORT' else 'LONG' end",
    )
    open_slot_key = get_column_def(cols, "open_slot_key", "null")
    hedge_version = get_column_def(cols, "hedge_version", "1")
'''
    source = source.replace(legacy_trade_block, "")

    normalized_trade_block = '''    account_id = (
        "coalesce(account_id, 'default')"
        if has_column(cols, "account_id")
        else "'default'"
    )
    position_side = (
        "coalesce(position_side, 'BOTH')"
        if has_column(cols, "position_side")
        else "'BOTH'"
    )
    hedge_version = (
        "coalesce(hedge_version, 0)"
        if has_column(cols, "hedge_version")
        else "0"
    )
    open_slot_key = (
        "case when coalesce(hedge_version, 0) >= 2 "
        "then open_slot_key else null end"
        if has_column(cols, "open_slot_key") and has_column(cols, "hedge_version")
        else "null"
    )
'''
    if "coalesce(position_side, 'BOTH')" not in source:
        anchor = '    record_version = get_column_def(cols, "record_version", "1")\n'
        if anchor not in source:
            raise IntegrationSafetyError("Native trade migration anchor is missing")
        source = source.replace(anchor, anchor + normalized_trade_block, 1)

    if "open_slot_key, hedge_version" not in source:
        source = source.replace(
            "            max_stake_amount, record_version\n",
            "            max_stake_amount, record_version, account_id, position_side,\n"
            "            open_slot_key, hedge_version\n",
            1,
        )
        source = source.replace(
            "            {record_version} record_version\n",
            "            {record_version} record_version, {account_id} account_id,\n"
            "            {position_side} position_side, {open_slot_key} open_slot_key,\n"
            "            {hedge_version} hedge_version\n",
            1,
        )

    legacy_order_block = '''    position_side = get_column_def(cols_order, "position_side", "null")
    action = get_column_def(cols_order, "action", "null")
    client_order_id = get_column_def(cols_order, "client_order_id", "null")
    idempotency_key = get_column_def(cols_order, "idempotency_key", "null")
    submit_state = get_column_def(cols_order, "submit_state", "null")
'''
    source = source.replace(legacy_order_block, "")

    order_start = source.find("def migrate_orders_table")
    order_section = source[order_start:] if order_start >= 0 else ""
    if "coalesce(position_side, 'BOTH')" not in order_section:
        normalized_order_block = '''    position_side = (
        "coalesce(position_side, 'BOTH')"
        if has_column(cols_order, "position_side")
        else "'BOTH'"
    )
    action = get_column_def(cols_order, "action", "null")
    client_order_id = get_column_def(cols_order, "client_order_id", "null")
    idempotency_key = get_column_def(cols_order, "idempotency_key", "null")
    submit_state = get_column_def(cols_order, "submit_state", "null")
'''
        anchor = '    ft_order_tag = get_column_def(cols_order, "ft_order_tag", "null")\n'
        if anchor not in source:
            raise IntegrationSafetyError("Native order migration anchor is missing")
        source = source.replace(anchor, anchor + normalized_order_block, 1)

    if "client_order_id, idempotency_key, submit_state" not in source:
        source = source.replace(
            "            ft_amount, ft_price, ft_cancel_reason, ft_order_tag\n",
            "            ft_amount, ft_price, ft_cancel_reason, ft_order_tag, position_side,\n"
            "            action, client_order_id, idempotency_key, submit_state\n",
            1,
        )
        source = source.replace(
            "            {ft_order_tag} ft_order_tag\n",
            "            {ft_order_tag} ft_order_tag, {position_side} position_side,\n"
            "            {action} action, {client_order_id} client_order_id,\n"
            "            {idempotency_key} idempotency_key, {submit_state} submit_state\n",
            1,
        )
    return source


def patch_migrations(path: Path) -> bool:
    """Normalize native migration copy SQL when an older form is encountered."""

    original = path.read_text(encoding="utf-8")
    patched = _patch_native_copy_text(original)
    if patched == original:
        return False
    ast.parse(patched, filename=str(path))
    path.write_text(patched, encoding="utf-8", newline="\n")
    py_compile.compile(str(path), doraise=True)
    return True


def integrate(root: Path) -> IntegrationReport:
    """Validate a candidate root and normalize only safe native copy SQL."""

    trade_model = root / "freqtrade" / "persistence" / "trade_model.py"
    migrations = root / "freqtrade" / "persistence" / "migrations.py"
    if not trade_model.is_file() or not migrations.is_file():
        raise FileNotFoundError("Target does not contain Freqtrade persistence central files")
    trade_changed = patch_trade_model(trade_model)
    migrations_changed = patch_migrations(migrations)
    return IntegrationReport(trade_changed, migrations_changed)
