from datetime import UTC, datetime
from decimal import Decimal

from freqtrade.hedge.execution import (
    ExecutionOrder,
    ExternalOrderSnapshot,
    FakeExchangeExecutionPort,
    InMemoryUserStreamOrderCache,
    IntentAction,
    OrderIntent,
    OrderLifecycle,
    OrderState,
    OrderType,
    PositionSide,
    UnknownOrderResolver,
)


def test_user_stream_fact_resolves_when_rest_has_no_order():
    now = datetime(2026, 7, 27, tzinfo=UTC)
    intent = OrderIntent(
        account_id="hedge-main",
        symbol="ETHUSDT",
        position_side=PositionSide.LONG,
        action=IntentAction.OPEN,
        quantity=Decimal("1"),
        idempotency_key="stream-recovery",
        order_type=OrderType.MARKET,
    )
    lifecycle = OrderLifecycle(updated_at=now).transition(
        OrderState.SUBMITTING, ordered_quantity=Decimal("1"), occurred_at=now
    )
    lifecycle = lifecycle.transition(
        OrderState.UNKNOWN, ordered_quantity=Decimal("1"), occurred_at=now
    )
    order = ExecutionOrder(intent, "hx-test-long-open-stream", Decimal("1"), lifecycle, now)
    cache = InMemoryUserStreamOrderCache()
    cache.put(
        ExternalOrderSnapshot(
            client_order_id=order.client_order_id,
            status=OrderState.PARTIAL,
            filled_quantity=Decimal("0.4"),
            average_price=Decimal("2000"),
            exchange_order_id="venue-1",
            exchange_trade_id="trade-1",
            observed_at=now,
        )
    )
    resolver = UnknownOrderResolver(
        FakeExchangeExecutionPort(),
        user_stream_cache=cache,
    )
    snapshot = resolver.resolve(order)
    assert snapshot is not None
    assert snapshot.status is OrderState.PARTIAL
    assert snapshot.filled_quantity == Decimal("0.4")
