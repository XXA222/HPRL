"""Pure, monotonic order lifecycle state machine for hedge execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum


class OrderState(StrEnum):
    PREPARED = "PREPARED"
    SUBMITTING = "SUBMITTING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    UNKNOWN = "UNKNOWN"


TERMINAL_STATES = frozenset({OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED})
_INITIAL_TIME = datetime.min.replace(tzinfo=UTC)
_ALLOWED_TRANSITIONS: dict[OrderState, frozenset[OrderState]] = {
    OrderState.PREPARED: frozenset(
        {
            OrderState.SUBMITTING,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.SUBMITTING: frozenset(
        {
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIAL,
            OrderState.FILLED,
            OrderState.REJECTED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.ACKNOWLEDGED: frozenset(
        {
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIAL,
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.PARTIAL: frozenset(
        {
            OrderState.PARTIAL,
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.UNKNOWN,
        }
    ),
    OrderState.UNKNOWN: frozenset(
        {
            OrderState.UNKNOWN,
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIAL,
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
        }
    ),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELED: frozenset(),
    OrderState.REJECTED: frozenset(),
}


class InvalidOrderTransition(ValueError):
    """Raised when a lifecycle transition violates an invariant."""


def _decimal(value: object, *, field_name: str) -> Decimal:
    if isinstance(value, (bool, float)):
        raise InvalidOrderTransition(f"{field_name} must use an exact decimal value")
    try:
        result = Decimal(value)  # type: ignore[arg-type]
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise InvalidOrderTransition(f"{field_name} must be a valid decimal") from exc
    if not result.is_finite():
        raise InvalidOrderTransition(f"{field_name} must be finite")
    return result


def _aware(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise InvalidOrderTransition(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise InvalidOrderTransition(f"{field_name} must be timezone-aware")
    return value


def _optional_text(value: str | None, *, field_name: str, limit: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidOrderTransition(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        return None
    if len(result) > limit or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise InvalidOrderTransition(f"{field_name} is invalid")
    return result


@dataclass(frozen=True, slots=True)
class OrderLifecycle:
    status: OrderState = OrderState.PREPARED
    filled_quantity: Decimal = Decimal("0")
    average_price: Decimal | None = None
    exchange_order_id: str | None = None
    version: int = 0
    updated_at: datetime = _INITIAL_TIME
    reason: str | None = None

    def __post_init__(self) -> None:
        try:
            status = self.status if isinstance(self.status, OrderState) else OrderState(self.status)
        except (TypeError, ValueError) as exc:
            raise InvalidOrderTransition("status is invalid") from exc
        filled = _decimal(self.filled_quantity, field_name="filled_quantity")
        if filled < 0:
            raise InvalidOrderTransition("filled_quantity must not be negative")
        average = None
        if self.average_price is not None:
            average = _decimal(self.average_price, field_name="average_price")
            if average <= 0 or filled <= 0:
                raise InvalidOrderTransition(
                    "average_price must be positive and requires a positive fill"
                )
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 0:
            raise InvalidOrderTransition("version must be a non-negative integer")
        updated_at = _aware(self.updated_at, field_name="updated_at")
        exchange_order_id = _optional_text(
            self.exchange_order_id, field_name="exchange_order_id", limit=256
        )
        reason = _optional_text(self.reason, field_name="reason", limit=1024)
        if status is OrderState.ACKNOWLEDGED and filled != 0:
            raise InvalidOrderTransition("ACKNOWLEDGED requires filled == 0")
        if status is OrderState.PARTIAL and filled <= 0:
            raise InvalidOrderTransition("PARTIAL requires positive filled_quantity")
        if status is OrderState.FILLED and filled <= 0:
            raise InvalidOrderTransition("FILLED requires positive filled_quantity")
        if status is OrderState.REJECTED and filled != 0:
            raise InvalidOrderTransition("REJECTED requires filled == 0")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "filled_quantity", filled)
        object.__setattr__(self, "average_price", average)
        object.__setattr__(self, "exchange_order_id", exchange_order_id)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "reason", reason)

    def transition(
        self,
        target: OrderState | str,
        *,
        ordered_quantity: Decimal,
        filled_quantity: Decimal | None = None,
        average_price: Decimal | None = None,
        exchange_order_id: str | None = None,
        reason: str | None = None,
        occurred_at: datetime | None = None,
    ) -> "OrderLifecycle":
        try:
            target_state = target if isinstance(target, OrderState) else OrderState(target)
        except (TypeError, ValueError) as exc:
            raise InvalidOrderTransition("target state is invalid") from exc
        if target_state not in _ALLOWED_TRANSITIONS[self.status]:
            raise InvalidOrderTransition(f"{self.status} -> {target_state} is not allowed")
        return self._build_fact(
            target_state,
            ordered_quantity=ordered_quantity,
            filled_quantity=filled_quantity,
            average_price=average_price,
            exchange_order_id=exchange_order_id,
            reason=reason,
            occurred_at=occurred_at,
        )

    def reconcile_terminal_fact(
        self,
        target: OrderState | str,
        *,
        ordered_quantity: Decimal,
        filled_quantity: Decimal,
        average_price: Decimal | None = None,
        exchange_order_id: str | None = None,
        reason: str | None = None,
        occurred_at: datetime | None = None,
    ) -> "OrderLifecycle":
        """Absorb a late fill after cancel without permitting arbitrary terminal rewrites."""
        try:
            target_state = target if isinstance(target, OrderState) else OrderState(target)
        except (TypeError, ValueError) as exc:
            raise InvalidOrderTransition("target state is invalid") from exc
        if self.status is not OrderState.CANCELED:
            raise InvalidOrderTransition("only CANCELED can absorb a late terminal fill")
        ordered = _decimal(ordered_quantity, field_name="ordered_quantity")
        next_filled = _decimal(filled_quantity, field_name="filled_quantity")
        if next_filled < self.filled_quantity:
            raise InvalidOrderTransition("filled quantity must be monotonic")
        if target_state is OrderState.CANCELED and next_filled < ordered:
            pass
        elif target_state is OrderState.FILLED and next_filled == ordered:
            pass
        else:
            raise InvalidOrderTransition("invalid late fact for CANCELED order")
        return self._build_fact(
            target_state,
            ordered_quantity=ordered,
            filled_quantity=next_filled,
            average_price=average_price,
            exchange_order_id=exchange_order_id,
            reason=reason,
            occurred_at=occurred_at,
            allow_terminal=True,
        )

    def _build_fact(
        self,
        target_state: OrderState,
        *,
        ordered_quantity: Decimal,
        filled_quantity: Decimal | None,
        average_price: Decimal | None,
        exchange_order_id: str | None,
        reason: str | None,
        occurred_at: datetime | None,
        allow_terminal: bool = False,
    ) -> "OrderLifecycle":
        if self.terminal and not allow_terminal:
            raise InvalidOrderTransition("terminal lifecycle cannot transition")
        ordered = _decimal(ordered_quantity, field_name="ordered_quantity")
        if ordered <= 0:
            raise InvalidOrderTransition("ordered_quantity must be positive")
        next_filled = (
            self.filled_quantity
            if filled_quantity is None
            else _decimal(filled_quantity, field_name="filled_quantity")
        )
        if next_filled < self.filled_quantity:
            raise InvalidOrderTransition("filled quantity must be monotonic")
        if next_filled < 0 or next_filled > ordered:
            raise InvalidOrderTransition("filled quantity is outside [0, ordered_quantity]")
        if target_state is OrderState.ACKNOWLEDGED and next_filled != 0:
            raise InvalidOrderTransition("ACKNOWLEDGED requires filled == 0")
        if target_state is OrderState.PARTIAL and not (Decimal("0") < next_filled < ordered):
            raise InvalidOrderTransition("PARTIAL requires 0 < filled < ordered")
        if target_state is OrderState.FILLED and next_filled != ordered:
            raise InvalidOrderTransition("FILLED requires filled == ordered")
        if target_state is OrderState.REJECTED and next_filled != 0:
            raise InvalidOrderTransition("REJECTED requires filled == 0")

        next_average = self.average_price
        if average_price is not None:
            next_average = _decimal(average_price, field_name="average_price")
            if next_average <= 0:
                raise InvalidOrderTransition("average_price must be positive")
        if next_average is not None and next_filled == 0:
            raise InvalidOrderTransition("average_price requires a positive filled quantity")

        timestamp = _aware(occurred_at or datetime.now(UTC), field_name="occurred_at")
        if self.updated_at != _INITIAL_TIME and timestamp < self.updated_at:
            raise InvalidOrderTransition("order updates must not move backwards in time")

        return replace(
            self,
            status=target_state,
            filled_quantity=next_filled,
            average_price=next_average,
            exchange_order_id=(
                _optional_text(exchange_order_id, field_name="exchange_order_id", limit=256)
                or self.exchange_order_id
            ),
            version=self.version + 1,
            updated_at=timestamp,
            reason=_optional_text(reason, field_name="reason", limit=1024),
        )

    @property
    def terminal(self) -> bool:
        return self.status in TERMINAL_STATES
