from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum

import pytest

from freqtrade.hedge.contracts import (
    MarketRules,
    PositionKey,
    ReadinessDecision,
    ReadinessState,
)
from freqtrade.hedge.execution import (
    ExecutionBlockedError,
    IntentAction,
    OrderIntent,
    OrderType,
    PositionSide,
    adapt_planner_intent,
    build_integrated_fake_runtime,
)


class E(StrEnum):
    LONG = "LONG"
    BUY = "BUY"
    SELL = "SELL"
    OPEN = "OPEN"
    UNSTUCK = "UNSTUCK"
    CORE = "CORE"
    LIMIT = "LIMIT"
    MARKET = "MARKET"
    GTC = "GTC"


@dataclass
class PlannerIntent:
    intent_id: str = "hp-abc"
    symbol: str = "ETH/USDT:USDT"
    position_side: E = E.LONG
    order_side: E = E.BUY
    action: E = E.OPEN
    bucket: E = E.CORE
    quantity: Decimal = Decimal("0.123")
    price: Decimal = Decimal("100")
    reduce_only: bool = False
    order_type: E = E.LIMIT
    time_in_force: E = E.GTC
    layer: int = 1
    reason: str = "grid"


def test_planner_adapter_produces_deterministic_execution_intent() -> None:
    one = adapt_planner_intent(
        PlannerIntent(), account_id="acct", strategy_id="s", cycle_id="c"
    )
    two = adapt_planner_intent(
        PlannerIntent(), account_id="acct", strategy_id="s", cycle_id="c"
    )
    assert one == two
    assert one.metadata["planner_intent_id"] == "hp-abc"
    assert one.metadata["bucket"] == "CORE"
    assert one.idempotency_key.startswith("planner:")


def test_planner_unstuck_is_normalized_to_reduce() -> None:
    planner = PlannerIntent(
        order_side=E.SELL,
        action=E.UNSTUCK,
        reduce_only=True,
        order_type=E.MARKET,
        price=Decimal("99"),
    )
    result = adapt_planner_intent(planner, account_id="acct")
    assert result.action is IntentAction.REDUCE
    assert result.order_type is OrderType.MARKET
    assert result.limit_price is None
    assert result.metadata["strategy_action"] == "UNSTUCK"


def test_planner_side_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        adapt_planner_intent(
            PlannerIntent(order_side=E.SELL),
            account_id="acct",
        )


class HaltGate:
    def evaluate(self, position_key: PositionKey) -> ReadinessDecision:
        return ReadinessDecision(ReadinessState.HALT, ("STALE_USER_STREAM",), True)


def market_intent(*, action: IntentAction = IntentAction.OPEN, key: str = "k") -> OrderIntent:
    return OrderIntent(
        account_id="acct",
        symbol="ETHUSDT",
        position_side=PositionSide.LONG,
        action=action,
        quantity=Decimal("0.1004"),
        idempotency_key=key,
        order_type=OrderType.MARKET,
        metadata={"reference_price": "100", "exchange": "binance"},
    )


def test_readiness_halt_blocks_new_risk_but_allows_reduce_to_reach_core() -> None:
    runtime = build_integrated_fake_runtime()
    runtime.engine._readiness = HaltGate()  # contract fixture injection
    with pytest.raises(ExecutionBlockedError, match="STALE_USER_STREAM"):
        runtime.engine.submit(market_intent())

    runtime.account.seed(
        account_id="acct",
        symbol="ETHUSDT",
        position_side=PositionSide.LONG,
        quantity=Decimal("1"),
        average_price=Decimal("100"),
    )
    reduced = runtime.engine.submit(
        market_intent(action=IntentAction.REDUCE, key="reduce")
    )
    assert reduced.order.lifecycle.status.value == "ACKNOWLEDGED"


def test_single_writer_loss_blocks_submission() -> None:
    runtime = build_integrated_fake_runtime()
    runtime.single_writer.set_leader(False)
    with pytest.raises(RuntimeError, match="SINGLE_WRITER_LOST"):
        runtime.engine.submit(market_intent())
    assert runtime.exchange.submit_calls == []


def test_market_rules_round_and_min_notional() -> None:
    runtime = build_integrated_fake_runtime()
    key = PositionKey("binance", "acct", "ETHUSDT", "LONG")
    runtime.market_rules.set_rules(
        key,
        MarketRules(
            quantity_step=Decimal("0.01"),
            price_tick=Decimal("0.1"),
            minimum_quantity=Decimal("0.01"),
            minimum_notional=Decimal("10"),
        ),
    )
    result = runtime.engine.submit(market_intent())
    assert result.order.intent.quantity == Decimal("0.10")
    with pytest.raises(ExecutionBlockedError, match="MIN_NOTIONAL"):
        runtime.engine.submit(
            OrderIntent(
                account_id="acct",
                symbol="ETHUSDT",
                position_side=PositionSide.LONG,
                action=IntentAction.OPEN,
                quantity=Decimal("0.01"),
                idempotency_key="small",
                order_type=OrderType.MARKET,
                metadata={"reference_price": "100"},
            )
        )


def test_expired_intent_is_never_submitted() -> None:
    runtime = build_integrated_fake_runtime()
    intent = OrderIntent(
        account_id="acct",
        symbol="ETHUSDT",
        position_side=PositionSide.LONG,
        action=IntentAction.OPEN,
        quantity=Decimal("0.1"),
        idempotency_key="expired",
        order_type=OrderType.MARKET,
        metadata={
            "reference_price": "100",
            "expires_at": datetime.now(UTC) - timedelta(seconds=1),
        },
    )
    with pytest.raises(ExecutionBlockedError, match="INTENT_EXPIRED"):
        runtime.engine.submit(intent)
    assert runtime.exchange.submit_calls == []
