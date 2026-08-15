from datetime import datetime, timezone
from decimal import Decimal

import pytest

from freqtrade.hedge.exchange.base import ExchangeFactBatch, PositionFact
from freqtrade.hedge.execution.service import PositionSide as ExecutionPositionSide
from freqtrade.hedge.integration import (
    InMemoryReadonlyRepository,
    IntegratedFakeHedgeApplication,
)
from freqtrade.hedge.planning.context import (
    LegPosition,
    MarketSnapshot,
    PlannerConfig,
    PlanningContext,
    PositionSide,
    StrategyLegState,
    WalletSnapshot,
)

NOW = datetime(2026, 7, 27, tzinfo=timezone.utc)


@pytest.mark.asyncio
async def test_integrated_repository_accepts_atomic_exchange_batch():
    repository = InMemoryReadonlyRepository()
    fact = PositionFact(
        account_id="hedge-main",
        symbol="ETHUSDT",
        position_side="LONG",
        quantity=Decimal("1.25"),
        entry_price=Decimal("100"),
        mark_price=Decimal("101"),
        unrealized_pnl=Decimal("1.25"),
        liquidation_price=Decimal("50"),
        leverage=3,
        margin_mode="CROSSED",
        update_time_ms=1_722_000_000_000,
        observed_at=NOW,
        source="BINANCE_REST",
    )
    await repository.append_exchange_fact_batch(
        ExchangeFactBatch(
            account_id="hedge-main",
            source="BINANCE_REST",
            observed_at=NOW,
            positions=(fact,),
        )
    )
    assert await repository.load_active_positions("hedge-main") == (fact,)


def test_planner_to_fake_execution_to_fills_runs_end_to_end():
    context = PlanningContext(
        market=MarketSnapshot(
            "ETH/USDT:USDT",
            NOW,
            Decimal("99.9"),
            Decimal("100.1"),
            Decimal("100"),
            Decimal("0.1"),
            Decimal("0.001"),
        ),
        wallet=WalletSnapshot(
            balance=Decimal("1000"),
            equity=Decimal("1000"),
            available_balance=Decimal("1000"),
            long=LegPosition(PositionSide.LONG),
            short=LegPosition(PositionSide.SHORT),
            leverage=Decimal("3"),
        ),
        config=PlannerConfig(cooldown_seconds=0),
        long_state=StrategyLegState(PositionSide.LONG),
        short_state=StrategyLegState(PositionSide.SHORT),
        long_signal=Decimal("1"),
        short_signal=Decimal("1"),
    )
    application = IntegratedFakeHedgeApplication()
    cycle = application.run_cycle(context)
    assert cycle.planning.submit_orders
    assert len(cycle.executions) == len(cycle.planning.submit_orders)
    filled = application.apply_full_fills(cycle.executions)
    assert filled
    assert all(item.order.lifecycle.status.value == "FILLED" for item in filled)
    long_leg = application.execution.account.leg(
        account_id="hedge-main",
        symbol="ETHUSDT",
        position_side=ExecutionPositionSide.LONG,
    )
    short_leg = application.execution.account.leg(
        account_id="hedge-main",
        symbol="ETHUSDT",
        position_side=ExecutionPositionSide.SHORT,
    )
    assert long_leg.quantity > 0
    assert short_leg.quantity > 0
