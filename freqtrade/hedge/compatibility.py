"""Compatibility helpers isolating Hedge extensions from upstream Freqtrade APIs.

Database migrations add nullable/default Hedge columns to native Trade and Order
rows.  Those defaults must not change native JSON/RPC schemas or legacy futures
wallet lookups unless an entity carries an explicit LONG/SHORT Hedge identity.
"""

from __future__ import annotations

from typing import Any


_HEDGE_POSITION_SIDES = frozenset({"LONG", "SHORT"})
_DEFAULT_ACCOUNT_IDS = frozenset({None, "", "default"})


def normalize_compat_position_side(value: object | None) -> str:
    """Normalize an enum/string side without accepting arbitrary aliases."""

    raw = getattr(value, "value", value)
    normalized = str(raw or "BOTH").strip().upper()
    return normalized if normalized in {"LONG", "SHORT", "BOTH"} else "BOTH"


def is_explicit_hedge_trade(trade: Any) -> bool:
    """Return whether a Trade is explicitly owned by the Hedge subsystem."""

    side = normalize_compat_position_side(getattr(trade, "position_side", None))
    account_id = getattr(trade, "account_id", None)
    hedge_version = getattr(trade, "hedge_version", None)
    open_slot_key = getattr(trade, "open_slot_key", None)
    return (
        side in _HEDGE_POSITION_SIDES
        or account_id not in _DEFAULT_ACCOUNT_IDS
        or bool(open_slot_key)
        or (isinstance(hedge_version, int) and hedge_version >= 2)
    )


def is_explicit_hedge_order(order: Any) -> bool:
    """Return whether an Order contains explicit Hedge execution identity.

    ``submit_state`` is deliberately excluded because H3 migrations may
    backfill native historical orders with ``ACKNOWLEDGED`` or ``TERMINAL``.
    """

    side = normalize_compat_position_side(getattr(order, "position_side", None))
    return side in _HEDGE_POSITION_SIDES or any(
        getattr(order, field, None) not in {None, ""}
        for field in (
            "position_action",
            "action_group_id",
            "action",
            "client_order_id",
            "idempotency_key",
        )
    )


def effective_trade_position_side(trade: Any) -> str:
    """Resolve LONG/SHORT for wallet operations while preserving native trades."""

    side = normalize_compat_position_side(getattr(trade, "position_side", None))
    if side in _HEDGE_POSITION_SIDES:
        return side

    direction = getattr(trade, "trade_direction", None)
    normalized_direction = str(getattr(direction, "value", direction) or "").strip().upper()
    if normalized_direction in _HEDGE_POSITION_SIDES:
        return normalized_direction
    if normalized_direction == "BUY":
        return "LONG"
    if normalized_direction == "SELL":
        return "SHORT"
    return "SHORT" if bool(getattr(trade, "is_short", False)) else "LONG"
