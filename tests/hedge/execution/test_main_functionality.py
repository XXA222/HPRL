from __future__ import annotations

from decimal import Decimal

from freqtrade.hedge.execution.action_group import ActionGroupExecutor
from freqtrade.hedge.execution.cancel_replace import CancelReplaceCoordinator
from freqtrade.hedge.execution.fake_exchange import FakeExchangeExecutionPort
from freqtrade.hedge.execution.idempotency import InMemoryIdempotencyStore
from freqtrade.hedge.execution.kill_switch import KillSwitch
from freqtrade.hedge.execution.service import (
    AllowAllRiskApproval,
    ExecutionService,
    InMemoryExecutionStore,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
)
from freqtrade.hedge.execution.state_machine import OrderState
from freqtrade.hedge.execution.unknown_resolver import UnknownOrderResolver
from freqtrade.hedge.telemetry.metrics import HedgeMetrics


def make_service(exchange: FakeExchangeExecutionPort):
    store = InMemoryExecutionStore()
    metrics = HedgeMetrics()
    service = ExecutionService(
        risk=AllowAllRiskApproval(),
        exchange=exchange,
        store=store,
        idempotency=InMemoryIdempotencyStore(),
        unknown_resolver=UnknownOrderResolver(exchange),
        kill_switch=KillSwitch(),
        metrics=metrics,
    )
    return service, store, metrics


def make_intent(
    key: str,
    *,
    side: PositionSide = PositionSide.LONG,
    action: IntentAction = IntentAction.OPEN,
    quantity: str = "1",
    group_id=None,
) -> OrderIntent:
    return OrderIntent(
        account_id="main",
        symbol="BTC/USDT:USDT",
        position_side=side,
        action=action,
        quantity=Decimal(quantity),
        idempotency_key=key,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("100"),
        action_group_id=group_id,
    )


def test_fake_exchange_incremental_fills_drive_local_lifecycle() -> None:
    exchange = FakeExchangeExecutionPort()
    service, _, metrics = make_service(exchange)
    submitted = service.submit(make_intent("fill-flow"))

    partial = exchange.fill_order(
        submitted.order.client_order_id,
        quantity="0.4",
        price="100",
        exchange_trade_id="t1",
    )
    partial_result = service.apply_exchange_event(partial)
    assert partial_result.order.lifecycle.status is OrderState.PARTIAL
    assert partial_result.order.lifecycle.filled_quantity == Decimal("0.4")

    filled = exchange.fill_order(
        submitted.order.client_order_id,
        quantity="0.6",
        price="110",
        exchange_trade_id="t2",
    )
    filled_result = service.apply_exchange_event(filled)
    assert filled_result.order.lifecycle.status is OrderState.FILLED
    assert filled_result.order.lifecycle.average_price == Decimal("106")
    assert metrics.snapshot()["fill_quantity_total"]["_"] == "1.0"


def test_refresh_order_reads_fake_exchange_without_resubmitting() -> None:
    exchange = FakeExchangeExecutionPort()
    service, _, _ = make_service(exchange)
    submitted = service.submit(make_intent("refresh-one"))
    exchange.fill_order(
        submitted.order.client_order_id,
        quantity="0.25",
        price="101",
    )

    refreshed = service.refresh_order(submitted.order.client_order_id)
    assert refreshed.order.lifecycle.status is OrderState.PARTIAL
    assert refreshed.order.lifecycle.filled_quantity == Decimal("0.25")
    assert len(exchange.submit_calls) == 1


def test_batch_refresh_and_filtered_listing() -> None:
    exchange = FakeExchangeExecutionPort()
    service, _, _ = make_service(exchange)
    first = service.submit(make_intent("batch-long"))
    second = service.submit(
        make_intent("batch-short", side=PositionSide.SHORT)
    )
    exchange.fill_order(first.order.client_order_id, quantity="1", price="100")
    exchange.fill_order(second.order.client_order_id, quantity="0.5", price="99")

    report = service.refresh_orders(account_id="main", symbol="BTCUSDT")
    assert report.attempted == 2
    assert report.succeeded == 2
    assert report.failed == 0
    assert report.complete
    assert {
        order.lifecycle.status
        for order in service.list_orders(account_id="main")
    } == {OrderState.FILLED, OrderState.PARTIAL}
    short_orders = service.list_orders(position_side="SHORT")
    assert len(short_orders) == 1
    assert short_orders[0].client_order_id == second.order.client_order_id


def test_resolve_unknowns_operates_as_a_batch() -> None:
    exchange = FakeExchangeExecutionPort()
    service, _, _ = make_service(exchange)
    exchange.queue_timeout()
    result = service.submit(make_intent("unknown-batch"))
    assert result.order.lifecycle.status is OrderState.UNKNOWN

    exchange.set_order(
        exchange.acknowledge_order(result.order.client_order_id)
    )
    report = service.resolve_unknowns(account_id="main")
    assert report.attempted == 1
    assert report.results[0].order.lifecycle.status is OrderState.ACKNOWLEDGED
    assert report.unresolved == 0


def test_cancel_orders_filters_by_position_side() -> None:
    exchange = FakeExchangeExecutionPort()
    service, _, _ = make_service(exchange)
    long_order = service.submit(make_intent("cancel-long"))
    short_order = service.submit(
        make_intent("cancel-short", side=PositionSide.SHORT)
    )

    report = service.cancel_orders(position_side=PositionSide.LONG)
    assert report.attempted == 1
    assert report.results[0].order.lifecycle.status is OrderState.CANCELED
    assert service.get_order(short_order.order.client_order_id).lifecycle.status is OrderState.ACKNOWLEDGED
    assert exchange.cancel_calls == [long_order.order.client_order_id]


def test_action_group_close_both_can_be_executed_and_refreshed() -> None:
    exchange = FakeExchangeExecutionPort()
    service, _, _ = make_service(exchange)
    executor = ActionGroupExecutor(service)

    report = executor.execute_close_both(
        account_id="main",
        symbol="BTCUSDT",
        long_quantity=Decimal("0.7"),
        short_quantity=Decimal("0.3"),
        idempotency_key="close-both-main",
    )
    assert report.attempted == 2
    assert report.outcome == "ACCEPTED"
    assert len(service.action_group_orders(report.action_group_id)) == 2

    for result in report.results:
        exchange.fill_order(
            result.order.client_order_id,
            quantity=result.order.approved_quantity,
            price="100",
        )
        service.refresh_order(result.order.client_order_id)

    refreshed = executor.refresh(report)
    assert refreshed.terminal
    assert refreshed.outcome == "COMPLETED"
    assert refreshed.filled_quantity == Decimal("1.0")


def test_cancel_replace_remaining_builds_exact_replacement() -> None:
    exchange = FakeExchangeExecutionPort()
    service, _, _ = make_service(exchange)
    original = service.submit(make_intent("replace-original"))
    partial = exchange.fill_order(
        original.order.client_order_id,
        quantity="0.35",
        price="100",
    )
    service.apply_exchange_event(partial)

    result = CancelReplaceCoordinator(service).execute_remaining(
        original_client_order_id=original.order.client_order_id,
        idempotency_key="replace-new",
        limit_price="99",
        metadata={"reason": "reprice"},
    )
    assert result.completed
    assert result.canceled.order.lifecycle.status is OrderState.CANCELED
    assert result.replacement is not None
    assert result.replacement.order.approved_quantity == Decimal("0.65")
    assert result.replacement.order.intent.limit_price == Decimal("99")
    assert result.replacement.order.intent.metadata["replaces_client_order_id"] == original.order.client_order_id


def test_fake_execution_harness_is_ready_to_run() -> None:
    from freqtrade.hedge.execution.fake_exchange import build_fake_execution_harness

    harness = build_fake_execution_harness()
    result = harness.service.submit(make_intent("harness-order"))
    assert result.order.lifecycle.status is OrderState.ACKNOWLEDGED
    assert len(harness.exchange.submit_calls) == 1
    assert harness.store.get_by_client_order_id(result.order.client_order_id) is not None
    assert harness.audit.records
