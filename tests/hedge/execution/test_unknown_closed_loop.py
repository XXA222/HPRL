from __future__ import annotations

from decimal import Decimal

from freqtrade.hedge.execution.fake_exchange import build_fake_execution_harness
from freqtrade.hedge.execution.orchestrator import HedgeExecutionEngine
from freqtrade.hedge.execution.service import (
    ExternalOrderSnapshot,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
)
from freqtrade.hedge.execution.state_machine import OrderState
from freqtrade.hedge.execution.unknown_supervisor import (
    UnknownOrderSupervisor,
    UnknownRecoveryState,
)


def _intent() -> OrderIntent:
    return OrderIntent(
        account_id="acct",
        symbol="BTCUSDT",
        position_side=PositionSide.LONG,
        action=IntentAction.OPEN,
        quantity=Decimal("0.1"),
        idempotency_key="closed-loop-unknown",
        order_type=OrderType.MARKET,
        metadata={"reference_price": "100"},
    )


def test_engine_registers_unknown_and_query_recovers_without_resubmit() -> None:
    harness = build_fake_execution_harness()
    engine = HedgeExecutionEngine(harness.service)
    supervisor = UnknownOrderSupervisor(engine)
    engine.bind_unknown_supervisor(supervisor)

    harness.exchange.queue_timeout()
    result = engine.submit(_intent())
    assert result.order.lifecycle.status is OrderState.UNKNOWN
    record = supervisor.get(result.order.client_order_id)
    assert record is not None
    assert record.state is UnknownRecoveryState.PENDING
    submit_count = len(harness.exchange.submit_calls)

    harness.exchange.set_order(
        ExternalOrderSnapshot(
            client_order_id=result.order.client_order_id,
            status=OrderState.ACKNOWLEDGED,
        )
    )
    recovered = engine.run_unknown_recovery()
    assert len(recovered) == 1
    assert recovered[0].state is UnknownRecoveryState.RESOLVED
    assert recovered[0].resolved_state is OrderState.ACKNOWLEDGED
    assert len(harness.exchange.submit_calls) == submit_count
    assert harness.exchange.query_calls[-1] == result.order.client_order_id


def test_unknown_supervisor_can_only_be_bound_once() -> None:
    harness = build_fake_execution_harness()
    engine = HedgeExecutionEngine(harness.service)
    first = UnknownOrderSupervisor(engine)
    second = UnknownOrderSupervisor(engine)
    engine.bind_unknown_supervisor(first)
    engine.bind_unknown_supervisor(first)
    assert engine.unknown_supervisor is first

    try:
        engine.bind_unknown_supervisor(second)
    except RuntimeError as exc:
        assert "already bound" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("second supervisor binding must fail closed")
