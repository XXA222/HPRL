from decimal import Decimal

from freqtrade.hedge.execution import (
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
    build_integrated_fake_runtime,
)


def test_integrated_runtime_caches_user_stream_order_fact():
    runtime = build_integrated_fake_runtime()
    submitted = runtime.engine.submit(
        OrderIntent(
            account_id="hedge-main",
            symbol="ETHUSDT",
            position_side=PositionSide.LONG,
            action=IntentAction.OPEN,
            quantity=Decimal("0.01"),
            idempotency_key="stream-cache-runtime",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("2000"),
        )
    )
    snapshot = runtime.exchange.fill_order(
        submitted.order.client_order_id,
        quantity=Decimal("0.004"),
        price=Decimal("1999"),
    )
    runtime.engine.apply_exchange_event(snapshot)
    cached = runtime.user_stream_cache.get(submitted.order.client_order_id)
    assert cached == snapshot
