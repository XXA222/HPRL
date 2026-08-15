from datetime import UTC, datetime
from decimal import Decimal

import pytest

from freqtrade.hedge.execution.fake_exchange import FakeExchangeExecutionPort
from freqtrade.hedge.execution.idempotency import InMemoryIdempotencyStore
from freqtrade.hedge.execution.kill_switch import ExecutionHaltedError, KillSwitch
from freqtrade.hedge.execution.service import (
    AllowAllRiskApproval,
    ExecutionBlockedError,
    ExecutionService,
    ExternalOrderSnapshot,
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


def intent(key: str, *, action: IntentAction = IntentAction.OPEN) -> OrderIntent:
    return OrderIntent(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        action=action,
        quantity=Decimal("0.1"),
        idempotency_key=key,
        order_type=OrderType.LIMIT,
        limit_price=Decimal("3000"),
    )


def test_same_idempotency_key_submits_only_once() -> None:
    exchange = FakeExchangeExecutionPort()
    exchange.queue_snapshot(OrderState.ACKNOWLEDGED)
    service, _, _ = make_service(exchange)

    first = service.submit(intent("same-key"))
    second = service.submit(intent("same-key"))

    assert first.order.client_order_id == second.order.client_order_id
    assert second.idempotent_replay
    assert len(exchange.submit_calls) == 1


def test_timeout_queries_before_any_retry_and_recovers() -> None:
    exchange = FakeExchangeExecutionPort()
    recovery = ExternalOrderSnapshot(
        client_order_id="placeholder",
        status=OrderState.ACKNOWLEDGED,
        exchange_order_id="ex-1",
        observed_at=datetime.now(UTC),
    )
    exchange.queue_timeout(recover_as=recovery)
    service, _, _ = make_service(exchange)

    result = service.submit(intent("timeout-key"))

    assert result.order.lifecycle.status is OrderState.ACKNOWLEDGED
    assert len(exchange.submit_calls) == 1
    assert exchange.query_calls == [result.order.client_order_id]
    assert "queried before any retry" in (result.message or "")


def test_unresolved_unknown_blocks_new_risk_on_same_leg() -> None:
    exchange = FakeExchangeExecutionPort()
    exchange.queue_timeout()
    service, _, metrics = make_service(exchange)
    unknown = service.submit(intent("unknown-1"))
    assert unknown.order.lifecycle.status is OrderState.UNKNOWN

    with pytest.raises(ExecutionBlockedError, match="UNKNOWN"):
        service.submit(intent("unknown-2"))

    exchange.queue_snapshot(OrderState.ACKNOWLEDGED)
    reduce_result = service.submit(intent("reduce-1", action=IntentAction.REDUCE))
    assert reduce_result.order.lifecycle.status is OrderState.ACKNOWLEDGED
    lock_metrics = metrics.snapshot()["lock_state"]
    assert any(value == "1" for value in lock_metrics.values())


def test_kill_switch_blocks_increase_but_allows_reduce() -> None:
    exchange = FakeExchangeExecutionPort()
    service, _, _ = make_service(exchange)
    service._kill_switch.activate(reason="operator halt", actor="admin")

    with pytest.raises(ExecutionHaltedError):
        service.submit(intent("halt-open"))

    exchange.queue_snapshot(OrderState.ACKNOWLEDGED)
    result = service.submit(intent("halt-reduce", action=IntentAction.REDUCE))
    assert result.order.lifecycle.status is OrderState.ACKNOWLEDGED
