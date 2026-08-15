"""Canonical cross-direction adapters.

Public contracts own identity and enum semantics. Planning and execution retain private
DTOs, but every conversion crosses this module so field/version drift cannot spread into
composition code.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from typing import Any, Iterable, Mapping
from uuid import NAMESPACE_URL, UUID, uuid5

from freqtrade.hedge.contracts.types import (
    IntentAction as ContractIntentAction,
    PositionSide as ContractPositionSide,
    expected_order_side,
)

from freqtrade.hedge.execution.service import IntentAction, OrderIntent, OrderType, PositionSide


def _enum_value(value: object, *, field_name: str) -> str:
    raw = getattr(value, "value", value)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{field_name} is invalid")
    return raw.strip().upper()


def _attribute(value: object, name: str, *, required: bool = True) -> Any:
    if not hasattr(value, name):
        if required:
            raise TypeError(f"planner intent does not expose {name}")
        return None
    return getattr(value, name)


def _planner_id(value: object) -> str:
    planner_id = str(_attribute(value, "intent_id")).strip()
    if not planner_id:
        raise ValueError("planner intent_id is required")
    return planner_id


def _deterministic_key(*, account_id: str, planner_id: str, cycle_id: str | None) -> str:
    payload = f"{account_id}|{planner_id}|{cycle_id or ''}"
    digest = sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"planner:{digest}"


def adapt_planner_intent(
    planner_intent: object,
    *,
    account_id: str,
    exchange: str = "binance",
    strategy_id: str | None = None,
    cycle_id: str | None = None,
    expires_at: datetime | None = None,
    idempotency_key: str | None = None,
) -> OrderIntent:
    """Convert a planning OrderIntent to the execution model.

    Planner ``UNSTUCK`` is normalized to a reduce/close execution action and retained as
    metadata.  BUY/SELL is validated against position side and increase/reduce semantics.
    """

    planner_id = _planner_id(planner_intent)
    position_side = PositionSide(
        _enum_value(
            _attribute(planner_intent, "position_side"),
            field_name="position_side",
        )
    )
    raw_action = _enum_value(_attribute(planner_intent, "action"), field_name="action")
    reason = str(_attribute(planner_intent, "reason", required=False) or "").strip()
    if raw_action == "UNSTUCK":
        action = IntentAction.REDUCE
        reason = reason or "unstuck"
    else:
        action = IntentAction(raw_action)

    quantity = Decimal(_attribute(planner_intent, "quantity"))
    order_type = OrderType(
        _enum_value(
            _attribute(planner_intent, "order_type", required=False) or "LIMIT",
            field_name="order_type",
        )
    )
    raw_price = _attribute(planner_intent, "price", required=False)
    limit_price = None if order_type is OrderType.MARKET else Decimal(raw_price)
    reduce_only = bool(_attribute(planner_intent, "reduce_only", required=False))

    planner_order_side = _attribute(planner_intent, "order_side", required=False)
    if planner_order_side is not None:
        planner_side_value = _enum_value(planner_order_side, field_name="order_side")
        expected = expected_order_side(
            ContractPositionSide(position_side.value),
            ContractIntentAction(action.value),
        ).value
        if planner_side_value != expected:
            raise ValueError(
                "planner order_side "
                f"{planner_side_value} conflicts with "
                f"{position_side.value}/{action.value}"
            )

    metadata: dict[str, object] = {
        "planner_intent_id": planner_id,
        "exchange": exchange,
    }
    for name in ("bucket", "layer", "time_in_force"):
        value = _attribute(planner_intent, name, required=False)
        if value is not None:
            metadata[name] = getattr(value, "value", value)
    if reason:
        metadata["reason"] = reason
    if raw_action == "UNSTUCK":
        metadata["strategy_action"] = "UNSTUCK"
    if strategy_id:
        metadata["strategy_id"] = strategy_id
    if cycle_id:
        metadata["cycle_id"] = cycle_id
    if expires_at is not None:
        metadata["expires_at"] = expires_at

    target = _attribute(planner_intent, "target_snapshot", required=False)
    if isinstance(target, Mapping):
        metadata["target_snapshot"] = dict(target)

    return OrderIntent(
        account_id=account_id,
        symbol=str(_attribute(planner_intent, "symbol")),
        position_side=position_side,
        action=action,
        quantity=quantity,
        idempotency_key=(
            idempotency_key
            or _deterministic_key(
                account_id=account_id,
                planner_id=planner_id,
                cycle_id=cycle_id,
            )
        ),
        order_type=order_type,
        limit_price=limit_price,
        reduce_only=reduce_only,
        intent_id=uuid5(NAMESPACE_URL, f"freqtrade-hedge:{account_id}:{planner_id}"),
        metadata=metadata,
    )


def adapt_planner_intents(
    planner_intents: Iterable[object],
    **kwargs: object,
) -> tuple[OrderIntent, ...]:
    return tuple(adapt_planner_intent(item, **kwargs) for item in planner_intents)


def assert_internal_contract_compatibility() -> None:
    """Fail fast when private enum values diverge from frozen public contracts."""

    from freqtrade.hedge.planning.context import (
        IntentAction as PlanningIntentAction,
        OrderType as PlanningOrderType,
        PositionSide as PlanningPositionSide,
    )

    if {item.value for item in PlanningPositionSide} != {
        item.value for item in ContractPositionSide
    }:
        raise RuntimeError("planning/public PositionSide values diverged")
    if {item.value for item in PositionSide} != {item.value for item in ContractPositionSide}:
        raise RuntimeError("execution/public PositionSide values diverged")
    if not {item.value for item in ContractIntentAction}.issubset(
        {item.value for item in PlanningIntentAction}
    ):
        raise RuntimeError("planning/public IntentAction values diverged")
    if {item.value for item in IntentAction} != {item.value for item in ContractIntentAction}:
        raise RuntimeError("execution/public IntentAction values diverged")
    if not {item.value for item in OrderType}.issubset(
        {item.value for item in PlanningOrderType}
    ):
        raise RuntimeError("planning/execution OrderType values diverged")
