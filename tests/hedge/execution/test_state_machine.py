from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from freqtrade.hedge.execution.state_machine import (
    InvalidOrderTransition,
    OrderLifecycle,
    OrderState,
)


def test_partial_fill_is_monotonic_and_reaches_filled() -> None:
    now = datetime.now(UTC)
    lifecycle = OrderLifecycle(updated_at=now)
    lifecycle = lifecycle.transition(
        OrderState.SUBMITTING, ordered_quantity=Decimal("2"), occurred_at=now
    )
    lifecycle = lifecycle.transition(
        OrderState.PARTIAL,
        ordered_quantity=Decimal("2"),
        filled_quantity=Decimal("0.5"),
        average_price=Decimal("100"),
        occurred_at=now + timedelta(milliseconds=1),
    )
    lifecycle = lifecycle.transition(
        OrderState.PARTIAL,
        ordered_quantity=Decimal("2"),
        filled_quantity=Decimal("1.5"),
        average_price=Decimal("101"),
        occurred_at=now + timedelta(milliseconds=2),
    )
    lifecycle = lifecycle.transition(
        OrderState.FILLED,
        ordered_quantity=Decimal("2"),
        filled_quantity=Decimal("2"),
        average_price=Decimal("102"),
        occurred_at=now + timedelta(milliseconds=3),
    )
    assert lifecycle.status is OrderState.FILLED
    assert lifecycle.filled_quantity == Decimal("2")
    assert lifecycle.terminal


def test_terminal_order_cannot_reopen() -> None:
    now = datetime.now(UTC)
    lifecycle = OrderLifecycle(updated_at=now).transition(
        OrderState.REJECTED,
        ordered_quantity=Decimal("1"),
        occurred_at=now,
    )
    with pytest.raises(InvalidOrderTransition):
        lifecycle.transition(
            OrderState.ACKNOWLEDGED,
            ordered_quantity=Decimal("1"),
            occurred_at=now + timedelta(seconds=1),
        )


def test_state_machine_rejects_non_finite_and_inconsistent_fills() -> None:
    now = datetime.now(UTC)
    lifecycle = OrderLifecycle(updated_at=now).transition(
        OrderState.SUBMITTING,
        ordered_quantity=Decimal("1"),
        occurred_at=now,
    )
    with pytest.raises(InvalidOrderTransition, match="finite"):
        lifecycle.transition(
            OrderState.PARTIAL,
            ordered_quantity=Decimal("1"),
            filled_quantity=Decimal("NaN"),
        )
    with pytest.raises(InvalidOrderTransition, match="ACKNOWLEDGED"):
        lifecycle.transition(
            OrderState.ACKNOWLEDGED,
            ordered_quantity=Decimal("1"),
            filled_quantity=Decimal("0.1"),
        )
    with pytest.raises(InvalidOrderTransition, match="average_price requires"):
        lifecycle.transition(
            OrderState.ACKNOWLEDGED,
            ordered_quantity=Decimal("1"),
            average_price=Decimal("100"),
        )


def test_state_machine_rejects_naive_or_backwards_time() -> None:
    now = datetime.now(UTC)
    lifecycle = OrderLifecycle(updated_at=now)
    with pytest.raises(InvalidOrderTransition, match="timezone-aware"):
        lifecycle.transition(
            OrderState.SUBMITTING,
            ordered_quantity=Decimal("1"),
            occurred_at=datetime.now(),
        )
    with pytest.raises(InvalidOrderTransition, match="backwards"):
        lifecycle.transition(
            OrderState.SUBMITTING,
            ordered_quantity=Decimal("1"),
            occurred_at=now - timedelta(seconds=1),
        )
