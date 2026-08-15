import inspect

import pytest

from freqtrade.hedge.execution.fake_exchange import (
    BinanceExecutionAdapterStub,
    WriteFeatureDisabledError,
)


def test_binance_write_adapter_is_a_hard_gated_stub() -> None:
    adapter = BinanceExecutionAdapterStub(live_trading_enabled=False)
    with pytest.raises(WriteFeatureDisabledError, match="disabled"):
        adapter.query_order(client_order_id="x")


def test_stub_contains_no_real_exchange_client_call() -> None:
    source = inspect.getsource(BinanceExecutionAdapterStub)
    forbidden = ("ccxt", "requests.", "httpx.", ".create_order(")
    assert not any(token in source for token in forbidden)


def test_gated_stub_is_definitive_rejection_not_unknown_lock() -> None:
    from decimal import Decimal

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

    adapter = BinanceExecutionAdapterStub(live_trading_enabled=False)
    store = InMemoryExecutionStore()
    service = ExecutionService(
        risk=AllowAllRiskApproval(),
        exchange=adapter,
        store=store,
        idempotency=InMemoryIdempotencyStore(),
        unknown_resolver=UnknownOrderResolver(adapter),
        kill_switch=KillSwitch(),
    )
    result = service.submit(
        OrderIntent(
            account_id="main",
            symbol="ETHUSDT",
            position_side=PositionSide.LONG,
            action=IntentAction.OPEN,
            quantity=Decimal("0.1"),
            idempotency_key="stub-disabled",
            order_type=OrderType.LIMIT,
            limit_price=Decimal("3000"),
        )
    )
    assert result.order.lifecycle.status is OrderState.REJECTED
    assert not store.has_unresolved_unknown(result.order.leg_key)
