"""Bridge Binance ORDER_TRADE_UPDATE events into the execution state machine."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

from .orchestrator import HedgeExecutionEngine
from .service import ExecutionResult, ExternalOrderSnapshot
from .state_machine import OrderState
from .unknown_resolver import UserStreamOrderCacheSinkPort

_STATUS_MAP = {
    "NEW": OrderState.ACKNOWLEDGED,
    "PARTIALLY_FILLED": OrderState.PARTIAL,
    "FILLED": OrderState.FILLED,
    "CANCELED": OrderState.CANCELED,
    "REJECTED": OrderState.REJECTED,
    "EXPIRED": OrderState.CANCELED,
    "EXPIRED_IN_MATCH": OrderState.CANCELED,
}


def order_trade_update_snapshot(
    event: Mapping[str, Any],
    *,
    allowed_symbols: tuple[str, ...],
) -> ExternalOrderSnapshot | None:
    if event.get("e") != "ORDER_TRADE_UPDATE":
        return None
    order = event.get("o")
    if not isinstance(order, Mapping):
        raise ValueError("ORDER_TRADE_UPDATE lacks order payload")
    symbol = str(order.get("s", "")).strip().upper()
    if symbol not in allowed_symbols:
        return None
    client_id = str(order.get("c", "")).strip()
    if not client_id:
        raise ValueError("ORDER_TRADE_UPDATE lacks client order id")
    raw_status = str(order.get("X", "")).strip().upper()
    status = _STATUS_MAP.get(raw_status, OrderState.UNKNOWN)
    filled = _decimal(order.get("z", "0"), "cumulative fill")
    average = _decimal(order.get("ap", "0"), "average price")
    average_value = average if average > 0 and filled > 0 else None
    fee = _decimal(order.get("n", "0") or "0", "commission")
    trade_id = str(order.get("t", "")).strip()
    return ExternalOrderSnapshot(
        client_order_id=client_id,
        status=status,
        filled_quantity=filled,
        average_price=average_value,
        exchange_order_id=str(order.get("i", "")).strip() or None,
        exchange_trade_id=trade_id if trade_id not in {"", "0", "-1"} else None,
        last_fill_fee=abs(fee),
        fee_currency=str(order.get("N", "USDT") or "USDT"),
        reason=f"BINANCE_USER_STREAM:{raw_status or 'UNKNOWN'}",
        observed_at=_timestamp(event.get("E") or event.get("T")),
    )


class ExecutionUserStreamBridge:
    def __init__(
        self,
        *,
        engine: HedgeExecutionEngine,
        cache: UserStreamOrderCacheSinkPort,
        allowed_symbols: tuple[str, ...],
    ) -> None:
        if not isinstance(engine, HedgeExecutionEngine):
            raise TypeError("engine must be HedgeExecutionEngine")
        if not callable(getattr(cache, "put", None)):
            raise TypeError("cache must implement put(snapshot)")
        normalized = tuple(dict.fromkeys(str(item).strip().upper() for item in allowed_symbols))
        if not normalized:
            raise ValueError("allowed_symbols must not be empty")
        self._engine = engine
        self._cache = cache
        self._allowed_symbols = normalized

    def handle(self, event: Mapping[str, Any]) -> ExecutionResult | None:
        snapshot = order_trade_update_snapshot(
            event,
            allowed_symbols=self._allowed_symbols,
        )
        if snapshot is None:
            return None
        self._cache.put(snapshot)
        try:
            return self._engine.apply_exchange_event(snapshot)
        except KeyError:
            return None


def _decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError(f"{field_name} must be exact")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is invalid") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{field_name} is invalid")
    return result


def _timestamp(value: object) -> datetime:
    try:
        millis = int(value)
    except (TypeError, ValueError):
        return datetime.now(UTC)
    return datetime.fromtimestamp(millis / 1000, tz=UTC) if millis > 0 else datetime.now(UTC)
