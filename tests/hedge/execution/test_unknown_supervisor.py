from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from freqtrade.hedge.execution import (
    ExecutionOrder,
    ExecutionResult,
    IntentAction,
    OrderIntent,
    OrderLifecycle,
    OrderState,
    OrderType,
    PositionSide,
    UnknownOrderSupervisor,
    UnknownRecoveryState,
)


@dataclass
class Clock:
    value: datetime

    def now(self) -> datetime:
        return self.value

    def advance(self, seconds: int) -> None:
        self.value += timedelta(seconds=seconds)


def unknown_result(client_order_id: str) -> ExecutionResult:
    intent = OrderIntent(
        account_id="hedge-main",
        symbol="ETHUSDT",
        position_side=PositionSide.LONG,
        action=IntentAction.OPEN,
        quantity=Decimal("1"),
        idempotency_key="unknown-supervisor",
        order_type=OrderType.MARKET,
    )
    now = datetime(2026, 7, 27, tzinfo=UTC)
    lifecycle = OrderLifecycle(updated_at=now).transition(
        OrderState.SUBMITTING, ordered_quantity=Decimal("1"), occurred_at=now
    )
    lifecycle = lifecycle.transition(
        OrderState.UNKNOWN, ordered_quantity=Decimal("1"), occurred_at=now
    )
    return ExecutionResult(
        ExecutionOrder(intent, client_order_id, Decimal("1"), lifecycle, now),
        message="UNKNOWN remains unresolved",
    )


class Resolver:
    def __init__(self, client_order_id: str) -> None:
        self.result = unknown_result(client_order_id)
        self.calls = 0

    def resolve_unknown(self, client_order_id: str) -> ExecutionResult:
        assert client_order_id == self.result.order.client_order_id
        self.calls += 1
        return self.result


def test_supervisor_retries_with_backoff_and_never_submits():
    client_id = "hx-test-long-open-unknown"
    clock = Clock(datetime(2026, 7, 27, tzinfo=UTC))
    resolver = Resolver(client_id)
    supervisor = UnknownOrderSupervisor(
        resolver,
        clock=clock,
        initial_backoff=timedelta(seconds=2),
        maximum_backoff=timedelta(seconds=8),
        maximum_attempts=3,
    )
    supervisor.register(client_id)
    first = supervisor.attempt(client_id)
    assert first.state is UnknownRecoveryState.RETRY_WAIT
    assert first.next_retry_at == clock.now() + timedelta(seconds=2)
    assert resolver.calls == 1
    assert supervisor.attempt(client_id) == first
    assert resolver.calls == 1
    clock.advance(2)
    second = supervisor.attempt(client_id)
    assert second.next_retry_at == clock.now() + timedelta(seconds=4)
    clock.advance(4)
    halted = supervisor.attempt(client_id)
    assert halted.state is UnknownRecoveryState.HALTED
    assert resolver.calls == 3
