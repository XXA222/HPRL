from __future__ import annotations

from decimal import Decimal

import pytest

from freqtrade.hedge.execution import (
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
    build_integrated_fake_runtime,
)


def intent(*, action: IntentAction, key: str, qty: str = "1") -> OrderIntent:
    return OrderIntent(
        account_id="acct",
        symbol="ETHUSDT",
        position_side=PositionSide.LONG,
        action=action,
        quantity=Decimal(qty),
        idempotency_key=key,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
    )


def test_fill_updates_transaction_ledger_and_fake_position() -> None:
    runtime = build_integrated_fake_runtime()
    submitted = runtime.engine.submit(intent(action=IntentAction.OPEN, key="open"))
    snapshot = runtime.exchange.fill_order(
        submitted.order.client_order_id,
        quantity="0.4",
        price="100",
        exchange_trade_id="trade-1",
    )
    applied = runtime.engine.apply_exchange_event(snapshot)
    assert applied.order.lifecycle.filled_quantity == Decimal("0.4")
    assert len(runtime.ledger.fills()) == 1
    projection = runtime.ledger.positions(account_id="acct", symbol="ETHUSDT")[0]
    assert projection.quantity == Decimal("0.4")
    assert runtime.account.leg(
        account_id="acct", symbol="ETHUSDT", position_side=PositionSide.LONG
    ).quantity == Decimal("0.4")
    assert runtime.publisher.events()[-1].event_type == "FILL_RECORDED"


def test_duplicate_exchange_event_does_not_duplicate_fill() -> None:
    runtime = build_integrated_fake_runtime()
    submitted = runtime.engine.submit(intent(action=IntentAction.OPEN, key="open"))
    snapshot = runtime.exchange.fill_order(
        submitted.order.client_order_id,
        quantity="0.4",
        price="100",
        exchange_trade_id="trade-1",
    )
    runtime.engine.apply_exchange_event(snapshot)
    runtime.engine.apply_exchange_event(snapshot)
    assert len(runtime.ledger.fills()) == 1
    projection = runtime.ledger.positions(account_id="acct", symbol="ETHUSDT")[0]
    assert projection.quantity == Decimal("0.4")


def test_fake_reduce_cannot_reverse_position() -> None:
    runtime = build_integrated_fake_runtime()
    runtime.account.seed(
        account_id="acct",
        symbol="ETHUSDT",
        position_side=PositionSide.LONG,
        quantity=Decimal("0.5"),
        average_price=Decimal("100"),
    )
    submitted = runtime.engine.submit(
        intent(action=IntentAction.REDUCE, key="reduce", qty="1")
    )
    with pytest.raises(ValueError, match="exceeds confirmed"):
        runtime.exchange.fill_order(
            submitted.order.client_order_id,
            quantity="0.6",
            price="101",
        )


def test_reduce_fill_realizes_pnl_without_crossing_zero() -> None:
    runtime = build_integrated_fake_runtime()
    runtime.account.seed(
        account_id="acct",
        symbol="ETHUSDT",
        position_side=PositionSide.LONG,
        quantity=Decimal("1"),
        average_price=Decimal("100"),
    )
    # Seed the transactional projection through an opening fill.
    opened = runtime.engine.submit(intent(action=IntentAction.OPEN, key="seed"))
    open_snapshot = runtime.exchange.fill_order(
        opened.order.client_order_id, quantity="1", price="100", exchange_trade_id="seed-fill"
    )
    runtime.engine.apply_exchange_event(open_snapshot)
    reduced = runtime.engine.submit(
        intent(action=IntentAction.REDUCE, key="reduce", qty="0.25")
    )
    reduce_snapshot = runtime.exchange.fill_order(
        reduced.order.client_order_id,
        quantity="0.25",
        price="110",
        exchange_trade_id="reduce-fill",
    )
    runtime.engine.apply_exchange_event(reduce_snapshot)
    projection = runtime.ledger.positions(account_id="acct", symbol="ETHUSDT")[0]
    assert projection.quantity == Decimal("0.75")
    assert projection.realized_pnl == Decimal("2.50")
