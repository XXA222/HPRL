from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal

from freqtrade.hedge.execution.fake_exchange import FakeExchangeExecutionPort
from freqtrade.hedge.execution.service import (
    ExecutionOrder,
    ExternalOrderSnapshot,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
)
from freqtrade.hedge.execution.state_machine import OrderLifecycle, OrderState
from freqtrade.hedge.execution.unknown_resolver import (
    ResolutionSource,
    UnknownOrderResolver,
)


def unknown_order(quantity: str = "1") -> ExecutionOrder:
    intent = OrderIntent(
        account_id="main",
        symbol="ETHUSDT",
        position_side=PositionSide.LONG,
        action=IntentAction.OPEN,
        quantity=Decimal(quantity),
        idempotency_key="resolver-order",
        order_type=OrderType.LIMIT,
        limit_price=Decimal("3000"),
    )
    now = datetime.now(UTC)
    lifecycle = (
        OrderLifecycle(updated_at=now)
        .transition(OrderState.SUBMITTING, ordered_quantity=Decimal(quantity))
        .transition(OrderState.UNKNOWN, ordered_quantity=Decimal(quantity))
    )
    return ExecutionOrder(
        intent=intent,
        client_order_id="FTH-ETHUSDT-L0-ABCDEFGHIJKLMNOP",
        approved_quantity=Decimal(quantity),
        lifecycle=lifecycle,
        created_at=now,
    )


class MultiSourceExchange:
    def __init__(self, *, direct=None, opened=(), fills=()):
        self.direct = direct
        self.opened = opened
        self.fills = fills

    def query_order(self, *, client_order_id):
        if isinstance(self.direct, Exception):
            raise self.direct
        return self.direct

    def list_open_orders(self, *, account_id, symbol):
        if isinstance(self.opened, Exception):
            raise self.opened
        return self.opened

    def list_recent_fills(self, *, account_id, symbol):
        if isinstance(self.fills, Exception):
            raise self.fills
        return self.fills

    def submit_order(self, approved):
        raise AssertionError("resolver must not submit")

    def cancel_order(self, *, client_order_id):
        raise AssertionError("resolver must not cancel")


def snapshot(order, state, filled="0", price=None, trade=None, reason=None):
    return ExternalOrderSnapshot(
        client_order_id=order.client_order_id,
        status=state,
        filled_quantity=Decimal(filled),
        average_price=None if price is None else Decimal(price),
        exchange_trade_id=trade,
        reason=reason,
    )


def test_direct_unknown_does_not_short_circuit_open_order_fact() -> None:
    order = unknown_order()
    exchange = MultiSourceExchange(
        direct=snapshot(order, OrderState.UNKNOWN),
        opened=(snapshot(order, OrderState.ACKNOWLEDGED),),
    )
    resolver = UnknownOrderResolver(exchange)
    resolved = resolver.resolve(order)
    assert resolved.status is OrderState.ACKNOWLEDGED
    assert resolver.last_result.source is ResolutionSource.OPEN_ORDERS


def test_all_sources_are_attempted_after_earlier_failures() -> None:
    order = unknown_order()
    fill = snapshot(order, OrderState.PARTIAL, "1", "3000", "trade-1")
    exchange = MultiSourceExchange(
        direct=ConnectionError("down"),
        opened=RuntimeError("down"),
        fills=(fill,),
    )
    resolver = UnknownOrderResolver(exchange)
    resolved = resolver.resolve(order)
    assert resolved.status is OrderState.FILLED
    assert resolver.last_result.errors == (
        "DIRECT_QUERY:ConnectionError",
        "OPEN_ORDERS:RuntimeError",
    )


def test_malformed_candidates_are_ignored_and_reported() -> None:
    order = unknown_order()
    wrong = ExternalOrderSnapshot("other", OrderState.ACKNOWLEDGED)
    exchange = MultiSourceExchange(direct="bad", opened=(wrong, 123), fills=())
    resolver = UnknownOrderResolver(exchange)
    assert resolver.resolve(order) is None
    assert "DIRECT_QUERY:INVALID_TYPE" in resolver.last_result.errors
    assert "OPEN_ORDERS:CLIENT_ORDER_ID_MISMATCH" in resolver.last_result.errors
    assert "OPEN_ORDERS:INVALID_TYPE" in resolver.last_result.errors


def test_recent_fills_are_incremental_and_weighted() -> None:
    order = unknown_order("1")
    fills = (
        snapshot(order, OrderState.PARTIAL, "0.25", "3000", "t1"),
        snapshot(order, OrderState.PARTIAL, "0.75", "3100", "t2"),
    )
    resolved = UnknownOrderResolver(MultiSourceExchange(fills=fills)).resolve(order)
    assert resolved.status is OrderState.FILLED
    assert resolved.filled_quantity == Decimal("1")
    assert resolved.average_price == Decimal("3075")


def test_duplicate_trade_ids_are_counted_once() -> None:
    order = unknown_order("1")
    fills = (
        snapshot(order, OrderState.PARTIAL, "0.4", "3000", "same"),
        snapshot(order, OrderState.PARTIAL, "0.4", "3000", "same"),
    )
    resolved = UnknownOrderResolver(MultiSourceExchange(fills=fills)).resolve(order)
    assert resolved.status is OrderState.PARTIAL
    assert resolved.filled_quantity == Decimal("0.4")


def test_unpriced_incremental_fill_suppresses_weighted_average() -> None:
    order = unknown_order("1")
    fills = (
        snapshot(order, OrderState.PARTIAL, "0.4", "3000", "t1"),
        snapshot(order, OrderState.PARTIAL, "0.2", None, "t2"),
    )
    resolved = UnknownOrderResolver(MultiSourceExchange(fills=fills)).resolve(order)
    assert resolved.filled_quantity == Decimal("0.6")
    assert resolved.average_price is None


def test_overfill_is_capped_and_reported() -> None:
    order = unknown_order("1")
    fills = (
        snapshot(order, OrderState.PARTIAL, "0.7", "3000", "t1"),
        snapshot(order, OrderState.PARTIAL, "0.7", "3100", "t2"),
    )
    resolved = UnknownOrderResolver(MultiSourceExchange(fills=fills)).resolve(order)
    assert resolved.filled_quantity == Decimal("1")
    assert resolved.status is OrderState.FILLED
    assert "capped_overfill" in resolved.reason


def test_canceled_fact_is_merged_with_later_partial_fills() -> None:
    order = unknown_order("1")
    canceled = snapshot(order, OrderState.CANCELED, "0.2", "3000")
    fills = (snapshot(order, OrderState.PARTIAL, "0.4", "3010", "t1"),)
    resolver = UnknownOrderResolver(
        MultiSourceExchange(direct=canceled, fills=fills)
    )
    resolved = resolver.resolve(order)
    assert resolved.status is OrderState.CANCELED
    assert resolved.filled_quantity == Decimal("0.4")
    assert resolver.last_result.source is ResolutionSource.MERGED


def test_rejected_fact_with_real_fill_is_promoted_to_partial() -> None:
    order = unknown_order("1")
    rejected = snapshot(order, OrderState.REJECTED)
    fills = (snapshot(order, OrderState.PARTIAL, "0.2", "3000", "t1"),)
    resolved = UnknownOrderResolver(
        MultiSourceExchange(direct=rejected, fills=fills)
    ).resolve(order)
    assert resolved.status is OrderState.PARTIAL
    assert resolved.filled_quantity == Decimal("0.2")


def test_stronger_open_fact_can_override_direct_ack() -> None:
    order = unknown_order("1")
    direct = snapshot(order, OrderState.ACKNOWLEDGED)
    opened = (snapshot(order, OrderState.FILLED, "1", "3000"),)
    resolver = UnknownOrderResolver(
        MultiSourceExchange(direct=direct, opened=opened)
    )
    assert resolver.resolve(order).status is OrderState.FILLED


def test_bad_iterable_from_source_is_contained() -> None:
    order = unknown_order()

    def broken():
        yield snapshot(order, OrderState.ACKNOWLEDGED)
        raise RuntimeError("iteration failed")

    resolver = UnknownOrderResolver(MultiSourceExchange(opened=broken()))
    assert resolver.resolve(order) is None
    assert resolver.last_result.errors == ("OPEN_ORDERS:RuntimeError",)


def test_last_result_access_is_safe_under_concurrency() -> None:
    order = unknown_order()
    exchange = MultiSourceExchange(
        direct=snapshot(order, OrderState.ACKNOWLEDGED)
    )
    resolver = UnknownOrderResolver(exchange)
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: resolver.resolve(order), range(100)))
    assert all(item.status is OrderState.ACKNOWLEDGED for item in results)
    assert resolver.last_result.snapshot.status is OrderState.ACKNOWLEDGED


def test_more_complete_base_fact_keeps_authoritative_average_price() -> None:
    order = unknown_order("1")
    base = snapshot(order, OrderState.CANCELED, "0.8", "3050")
    fills = (snapshot(order, OrderState.PARTIAL, "0.4", "3000", "t1"),)

    resolved = UnknownOrderResolver(
        MultiSourceExchange(direct=base, fills=fills)
    ).resolve(order)

    assert resolved.status is OrderState.CANCELED
    assert resolved.filled_quantity == Decimal("0.8")
    assert resolved.average_price == Decimal("3050")


def test_canceled_zero_does_not_erase_partial_fill_from_open_orders() -> None:
    order = unknown_order("1")
    direct = snapshot(order, OrderState.CANCELED)
    opened = (snapshot(order, OrderState.PARTIAL, "0.8", "3040"),)

    resolver = UnknownOrderResolver(
        MultiSourceExchange(direct=direct, opened=opened)
    )
    resolved = resolver.resolve(order)

    assert resolved.status is OrderState.CANCELED
    assert resolved.filled_quantity == Decimal("0.8")
    assert resolved.average_price == Decimal("3040")
    assert resolver.last_result.source is ResolutionSource.MERGED


def test_rejected_zero_does_not_erase_partial_fill_from_open_orders() -> None:
    order = unknown_order("1")
    direct = snapshot(order, OrderState.REJECTED)
    opened = (snapshot(order, OrderState.PARTIAL, "0.3", "3020"),)

    resolved = UnknownOrderResolver(
        MultiSourceExchange(direct=direct, opened=opened)
    ).resolve(order)

    assert resolved.status is OrderState.PARTIAL
    assert resolved.filled_quantity == Decimal("0.3")


def test_malformed_filled_candidate_below_approved_is_ignored() -> None:
    order = unknown_order("1")
    malformed = snapshot(order, OrderState.FILLED, "0.4", "3000")
    resolver = UnknownOrderResolver(MultiSourceExchange(direct=malformed))

    assert resolver.resolve(order) is None
    assert resolver.last_result.errors == ("DIRECT_QUERY:INVALID_FILLED_QUANTITY",)
