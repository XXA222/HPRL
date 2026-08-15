from decimal import Decimal

from freqtrade.hedge.execution.action_group import (
    ActionGroupExecutor,
    build_close_both_intents,
)
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


def make_service(exchange: FakeExchangeExecutionPort) -> ExecutionService:
    return ExecutionService(
        risk=AllowAllRiskApproval(),
        exchange=exchange,
        store=InMemoryExecutionStore(),
        idempotency=InMemoryIdempotencyStore(),
        unknown_resolver=UnknownOrderResolver(exchange),
        kill_switch=KillSwitch(),
    )


def test_close_both_is_two_intents_with_shared_group_and_partial_report() -> None:
    intents = build_close_both_intents(
        account_id="main",
        symbol="ETHUSDT",
        long_quantity=Decimal("0.2"),
        short_quantity=Decimal("0.3"),
        idempotency_key="close-both-1",
    )
    assert len(intents) == 2
    assert intents[0].intent_id != intents[1].intent_id
    assert intents[0].action_group_id == intents[1].action_group_id
    assert {item.position_side for item in intents} == {PositionSide.LONG, PositionSide.SHORT}
    assert all(item.reduce_only for item in intents)

    exchange = FakeExchangeExecutionPort()
    exchange.queue_snapshot(OrderState.FILLED, filled_quantity="0.2", average_price="3000")
    exchange.queue_snapshot(OrderState.REJECTED, reason="second leg rejected by fake")
    report = ActionGroupExecutor(make_service(exchange)).execute(intents)

    assert len(report.results) == 2
    assert len(report.errors) == 0
    assert report.partially_successful
    assert not report.fully_successful
    by_side = {result.order.intent.position_side: result for result in report.results}
    assert by_side[PositionSide.LONG].order.lifecycle.status is OrderState.FILLED
    assert by_side[PositionSide.SHORT].order.lifecycle.status is OrderState.REJECTED


def test_cancel_replace_waits_for_confirmed_cancel() -> None:
    exchange = FakeExchangeExecutionPort()
    exchange.queue_snapshot(OrderState.ACKNOWLEDGED)
    service = make_service(exchange)
    original = service.submit(
        OrderIntent(
            account_id="main",
            symbol="ETHUSDT",
            position_side=PositionSide.LONG,
            action=IntentAction.OPEN,
            quantity=Decimal("0.1"),
            idempotency_key="original",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("3000"),
        )
    )
    exchange.queue_snapshot(OrderState.ACKNOWLEDGED)
    replacement = OrderIntent(
        account_id="main",
        symbol="ETHUSDT",
        position_side=PositionSide.LONG,
        action=IntentAction.OPEN,
        quantity=Decimal("0.1"),
        idempotency_key="replacement",
        order_type=OrderType.LIMIT,
        limit_price=Decimal("2990"),
    )
    result = CancelReplaceCoordinator(service).execute(
        original_client_order_id=original.order.client_order_id,
        replacement_intent=replacement,
    )
    assert result.completed
    assert result.canceled.order.lifecycle.status is OrderState.CANCELED
    assert result.replacement is not None
    assert len(exchange.submit_calls) == 2
