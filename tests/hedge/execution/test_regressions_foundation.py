from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import Event, Thread

import pytest

from freqtrade.hedge.execution.cancel_replace import CancelReplaceCoordinator
from freqtrade.hedge.execution.fake_exchange import FakeExchangeExecutionPort
from freqtrade.hedge.execution.idempotency import InMemoryIdempotencyStore
from freqtrade.hedge.execution.kill_switch import KillSwitch
from freqtrade.hedge.execution.service import (
    AllowAllRiskApproval,
    ExecutionBlockedError,
    ExecutionOrder,
    ExecutionService,
    ExternalOrderSnapshot,
    InMemoryExecutionStore,
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


def make_intent(
    key: str,
    *,
    side: PositionSide = PositionSide.LONG,
    quantity: str = "1",
) -> OrderIntent:
    return OrderIntent(
        account_id=" main ",
        symbol="ETH/USDT:USDT",
        position_side=side,
        action=IntentAction.OPEN,
        quantity=Decimal(quantity),
        idempotency_key=key,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("3000"),
    )


def make_service(exchange, *, idempotency=None):
    store = InMemoryExecutionStore()
    service = ExecutionService(
        risk=AllowAllRiskApproval(),
        exchange=exchange,
        store=store,
        idempotency=idempotency or InMemoryIdempotencyStore(),
        unknown_resolver=UnknownOrderResolver(exchange),
        kill_switch=KillSwitch(),
    )
    return service, store


def test_whitespace_is_normalized_before_idempotency_and_client_id() -> None:
    exchange = FakeExchangeExecutionPort()
    service, _ = make_service(exchange)

    first = service.submit(make_intent(" normalized-key "))
    second = service.submit(make_intent("normalized-key"))

    assert first.order.intent.account_id == "main"
    assert first.order.intent.idempotency_key == "normalized-key"
    assert second.idempotent_replay
    assert len(exchange.submit_calls) == 1


def test_duplicate_ack_is_idempotent_and_stale_snapshot_cannot_cancel_partial() -> None:
    exchange = FakeExchangeExecutionPort()
    service, _ = make_service(exchange)
    acknowledged = service.submit(make_intent("snapshots"))
    version = acknowledged.order.lifecycle.version

    duplicate = service.apply_snapshot(
        acknowledged.order,
        ExternalOrderSnapshot(
            client_order_id=acknowledged.order.client_order_id,
            status=OrderState.ACKNOWLEDGED,
            exchange_order_id=acknowledged.order.lifecycle.exchange_order_id,
            observed_at=acknowledged.order.lifecycle.updated_at + timedelta(milliseconds=1),
        ),
    )
    assert duplicate.lifecycle.version == version

    partial = service.apply_snapshot(
        duplicate,
        ExternalOrderSnapshot(
            client_order_id=duplicate.client_order_id,
            status=OrderState.PARTIAL,
            filled_quantity=Decimal("0.4"),
            average_price=Decimal("3001"),
            observed_at=duplicate.lifecycle.updated_at + timedelta(seconds=2),
        ),
    )
    stale_cancel = service.apply_snapshot(
        partial,
        ExternalOrderSnapshot(
            client_order_id=partial.client_order_id,
            status=OrderState.CANCELED,
            observed_at=partial.lifecycle.updated_at - timedelta(seconds=1),
        ),
    )
    assert stale_cancel.lifecycle.status is OrderState.PARTIAL
    assert stale_cancel.lifecycle.filled_quantity == Decimal("0.4")


class DirectQueryFailureExchange(FakeExchangeExecutionPort):
    def query_order(self, *, client_order_id: str):
        self.query_calls.append(client_order_id)
        raise ConnectionError("direct query unavailable")


def unknown_order() -> ExecutionOrder:
    intent = make_intent("recover")
    now = datetime.now(UTC)
    lifecycle = OrderLifecycle(updated_at=now)
    lifecycle = lifecycle.transition(
        OrderState.SUBMITTING, ordered_quantity=Decimal("1"), occurred_at=now
    )
    lifecycle = lifecycle.transition(
        OrderState.UNKNOWN,
        ordered_quantity=Decimal("1"),
        occurred_at=now + timedelta(milliseconds=1),
    )
    return ExecutionOrder(intent, "FTH-ETHUSDT-L0-ABCDEFGHIJKLMNOP", Decimal("1"), lifecycle, now)


def test_unknown_resolver_continues_after_source_error_and_deduplicates_trade_ids() -> None:
    exchange = DirectQueryFailureExchange()
    order = unknown_order()
    first = ExternalOrderSnapshot(
        client_order_id=order.client_order_id,
        status=OrderState.PARTIAL,
        filled_quantity=Decimal("0.4"),
        average_price=Decimal("3000"),
        exchange_order_id="order-1",
        exchange_trade_id="trade-1",
    )
    duplicate = ExternalOrderSnapshot(
        client_order_id=order.client_order_id,
        status=OrderState.PARTIAL,
        filled_quantity=Decimal("0.4"),
        average_price=Decimal("3000"),
        exchange_order_id="order-1",
        exchange_trade_id="trade-1",
    )
    second = ExternalOrderSnapshot(
        client_order_id=order.client_order_id,
        status=OrderState.PARTIAL,
        filled_quantity=Decimal("0.6"),
        average_price=Decimal("3010"),
        exchange_order_id="order-1",
        exchange_trade_id="trade-2",
    )
    exchange.add_recent_fill(first)
    exchange.add_recent_fill(duplicate)
    exchange.add_recent_fill(second)

    resolver = UnknownOrderResolver(exchange)
    resolved = resolver.resolve(order)

    assert resolved is not None
    assert resolved.status is OrderState.FILLED
    assert resolved.filled_quantity == Decimal("1")
    assert resolved.average_price == Decimal("3006")
    assert resolver.last_result.source is ResolutionSource.RECENT_FILLS
    assert resolver.last_result.errors == ("DIRECT_QUERY:ConnectionError",)


def test_cancel_replace_rejects_other_leg_and_excess_remaining_quantity() -> None:
    exchange = FakeExchangeExecutionPort()
    service, _ = make_service(exchange)
    original = service.submit(make_intent("original", quantity="1"))
    coordinator = CancelReplaceCoordinator(service)

    with pytest.raises(ValueError, match="same account, symbol and side"):
        coordinator.execute(
            original_client_order_id=original.order.client_order_id,
            replacement_intent=make_intent("other-side", side=PositionSide.SHORT),
        )
    with pytest.raises(ValueError, match="exceeds original remaining"):
        coordinator.execute(
            original_client_order_id=original.order.client_order_id,
            replacement_intent=make_intent("too-large", quantity="2"),
        )
    assert exchange.cancel_calls == []


class BlockingTimeoutExchange(FakeExchangeExecutionPort):
    def __init__(self) -> None:
        super().__init__()
        self.entered = Event()
        self.release = Event()

    def submit_order(self, approved):
        self.submit_calls.append(approved)
        self.entered.set()
        self.release.wait(timeout=5)
        raise TimeoutError("ambiguous timeout")


def test_same_leg_admission_waits_and_blocks_after_unknown() -> None:
    exchange = BlockingTimeoutExchange()
    service, _ = make_service(exchange)
    outcomes: list[object] = []

    def first() -> None:
        outcomes.append(service.submit(make_intent("first")))

    def second() -> None:
        try:
            service.submit(make_intent("second"))
        except Exception as exc:  # asserted below
            outcomes.append(exc)

    thread_one = Thread(target=first)
    thread_two = Thread(target=second)
    thread_one.start()
    assert exchange.entered.wait(timeout=2)
    thread_two.start()
    exchange.release.set()
    thread_one.join(timeout=5)
    thread_two.join(timeout=5)

    assert len(exchange.submit_calls) == 1
    assert any(isinstance(item, ExecutionBlockedError) for item in outcomes)


class FailingCompleteIdempotency(InMemoryIdempotencyStore):
    def complete(self, key, value):
        raise OSError("persistence unavailable")


def test_side_effect_started_never_releases_idempotency_on_completion_failure() -> None:
    exchange = FakeExchangeExecutionPort()
    idempotency = FailingCompleteIdempotency()
    service, _ = make_service(exchange, idempotency=idempotency)

    with pytest.raises(OSError, match="persistence unavailable"):
        service.submit(make_intent("persist-failure"))
    with pytest.raises(ExecutionBlockedError, match="in flight"):
        service.submit(make_intent("persist-failure"))
    assert len(exchange.submit_calls) == 1


def test_same_idempotency_key_with_different_intent_is_conflict() -> None:
    from freqtrade.hedge.execution.service import IdempotencyConflictError

    exchange = FakeExchangeExecutionPort()
    service, _ = make_service(exchange)
    service.submit(make_intent("conflict", quantity="1"))
    with pytest.raises(IdempotencyConflictError, match="different intent"):
        service.submit(make_intent("conflict", quantity="0.5"))
    assert len(exchange.submit_calls) == 1


def test_idempotent_replay_reads_latest_order_state_from_store() -> None:
    exchange = FakeExchangeExecutionPort()
    service, _ = make_service(exchange)
    first = service.submit(make_intent("latest-replay"))
    partial = service.apply_snapshot(
        first.order,
        ExternalOrderSnapshot(
            client_order_id=first.order.client_order_id,
            status=OrderState.PARTIAL,
            filled_quantity=Decimal("0.25"),
            average_price=Decimal("3000"),
            observed_at=first.order.lifecycle.updated_at + timedelta(seconds=1),
        ),
    )
    replay = service.submit(make_intent("latest-replay"))
    assert replay.idempotent_replay
    assert replay.order.lifecycle.version == partial.lifecycle.version
    assert replay.order.lifecycle.status is OrderState.PARTIAL
