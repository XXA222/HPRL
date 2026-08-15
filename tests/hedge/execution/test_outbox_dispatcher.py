from decimal import Decimal

from freqtrade.hedge.execution import (
    IntentAction,
    OrderIntent,
    OrderType,
    OutboxDispatcher,
    PositionSide,
    build_integrated_fake_runtime,
)


class FailingPublisher:
    def __init__(self) -> None:
        self.fail = True
        self.events = []

    def publish(self, event) -> None:
        if self.fail:
            raise RuntimeError("publisher unavailable")
        self.events.append(event)


def intent(key: str) -> OrderIntent:
    return OrderIntent(
        account_id="hedge-main",
        symbol="ETHUSDT",
        position_side=PositionSide.LONG,
        action=IntentAction.OPEN,
        quantity=Decimal("0.01"),
        idempotency_key=key,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("2000"),
    )


def test_publisher_failure_does_not_turn_submitted_order_into_exception():
    publisher = FailingPublisher()
    runtime = build_integrated_fake_runtime(publisher=publisher)
    result = runtime.engine.submit(intent("publisher-failure"))
    assert result.order.lifecycle.status.value == "ACKNOWLEDGED"
    pending = runtime.ledger.outbox(unpublished_only=True)
    assert len(pending) == 1
    assert pending[0].attempts == 1

    publisher.fail = False
    report = OutboxDispatcher(runtime.ledger, publisher).dispatch()
    assert report.attempted == 1
    assert report.published == 1
    assert report.failed == ()
    assert len(runtime.ledger.outbox(unpublished_only=True)) == 0
