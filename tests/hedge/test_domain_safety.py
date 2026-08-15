from decimal import Decimal

from freqtrade.enums.hedge import PositionAction, PositionSide
from freqtrade.hedge.domain import HedgeAction, HedgeActionPlan, PositionKey
from freqtrade.hedge.local_reduce_only import calculate_safe_reduce


def test_frozen_action_is_not_mutated_by_plan() -> None:
    original = HedgeAction(
        "ETH/USDT:USDT",
        PositionSide.LONG,
        PositionAction.REDUCE,
        "1",
    )
    original_hash = hash(original)
    plan = HedgeActionPlan([original])
    assert hash(original) == original_hash
    assert plan.actions[0].action_group_id == plan.action_group_id
    assert original.action_group_id != plan.action_group_id


def test_position_key_contains_account() -> None:
    left = PositionKey("binance", "ETH/USDT:USDT", PositionSide.LONG, "a")
    right = PositionKey("binance", "ETH/USDT:USDT", PositionSide.LONG, "b")
    assert left != right


def test_local_reduce_only_clips() -> None:
    result = calculate_safe_reduce(
        requested_quantity=Decimal("8"),
        confirmed_quantity=Decimal("10"),
        pending_reduce_quantity=Decimal("4"),
    )
    assert result.allowed_quantity == Decimal("6")
    assert result.clipped
