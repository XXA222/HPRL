
from decimal import Decimal
from unittest import TestCase
from uuid import uuid4

from freqtrade.enums.hedge import PositionAction, PositionMode, PositionSide
from freqtrade.hedge.domain import HedgeAction, HedgeActionPlan, PositionKey


class TestHedgeDomain(TestCase):
    def test_enum_values_are_stable(self) -> None:
        self.assertEqual(PositionMode.ONEWAY.value, "oneway")
        self.assertEqual(PositionMode.HEDGE.value, "hedge")
        self.assertEqual(PositionSide.LONG.value, "LONG")
        self.assertEqual(PositionAction.REDUCE.value, "REDUCE")

    def test_position_key_keeps_side_in_identity(self) -> None:
        long_key = PositionKey("binance", "ETH/USDT:USDT", PositionSide.LONG)
        short_key = PositionKey("binance", "ETH/USDT:USDT", PositionSide.SHORT)

        self.assertNotEqual(long_key.identity, short_key.identity)

    def test_hedge_action_rejects_both_side(self) -> None:
        with self.assertRaisesRegex(ValueError, "never BOTH"):
            HedgeAction(
                symbol="ETH/USDT:USDT",
                position_side=PositionSide.BOTH,
                position_action=PositionAction.OPEN,
                quantity=Decimal("0.1"),
            )

    def test_plan_orders_reductions_before_increases(self) -> None:
        group_id = uuid4()
        increase = HedgeAction(
            "ETH/USDT:USDT",
            PositionSide.LONG,
            PositionAction.INCREASE,
            "0.10",
            group_id,
        )
        reduce = HedgeAction(
            "ETH/USDT:USDT",
            PositionSide.SHORT,
            PositionAction.REDUCE,
            "0.20",
            group_id,
        )

        plan = HedgeActionPlan([increase, reduce], group_id)

        self.assertEqual(plan.ordered_actions, (reduce, increase))
        self.assertTrue(plan.ordered_actions[0].reduces_risk)
        self.assertTrue(plan.ordered_actions[1].increases_risk)

    def test_quantity_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            HedgeAction(
                "ETH/USDT:USDT",
                PositionSide.LONG,
                PositionAction.OPEN,
                "0",
            )
