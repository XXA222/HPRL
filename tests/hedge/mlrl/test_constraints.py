import unittest

from freqtrade.freqai.hedge_rl.actions import HedgeActions
from freqtrade.freqai.hedge_rl.config import HedgeRLConfig
from freqtrade.freqai.hedge_rl.constraints import HedgeActionMasker
from freqtrade.freqai.hedge_rl.state import HedgeAccountState, HedgeLegSide, HedgeLegState


class TestRound08Constraints(unittest.TestCase):
    def setUp(self):
        self.masker = HedgeActionMasker(
            HedgeRLConfig(max_side_exposure=0.3, max_gross_exposure=0.5, max_net_exposure=0.3)
        )

    def test_flat_semantic_mask(self):
        account = HedgeAccountState.initial(1000)
        mask = self.masker.mask(account=account, mark=100)
        self.assertTrue(mask[HedgeActions.HOLD])
        self.assertTrue(mask[HedgeActions.LONG_OPEN_SMALL])
        self.assertFalse(mask[HedgeActions.LONG_ADD_SMALL])
        self.assertFalse(mask[HedgeActions.LONG_CLOSE])

    def test_projected_exposure_limit(self):
        account = HedgeAccountState(
            cash_balance=1000,
            equity=1000,
            peak_equity=1000,
            long=HedgeLegState(HedgeLegSide.LONG, 2.5, 100),
            short=HedgeLegState(HedgeLegSide.SHORT),
        )
        decision = self.masker.evaluate(HedgeActions.LONG_ADD_SMALL, account=account, mark=100)
        self.assertFalse(decision.allowed)
        self.assertIn("LONG_SIDE_EXPOSURE_LIMIT", decision.reasons)
        self.assertTrue(
            self.masker.evaluate(HedgeActions.LONG_REDUCE_SMALL, account=account, mark=100).allowed
        )

    def test_close_both_is_available_with_either_leg(self):
        flat = HedgeAccountState.initial(1000)
        flat_decision = self.masker.evaluate(HedgeActions.CLOSE_BOTH, account=flat, mark=100)
        self.assertFalse(flat_decision.allowed)
        self.assertIn("COMPOSITE_REDUCE_REQUIRES_ANY_POSITION", flat_decision.reasons)

        long_only = HedgeAccountState(
            cash_balance=1000,
            equity=1000,
            peak_equity=1000,
            long=HedgeLegState(HedgeLegSide.LONG, 1.0, 100),
            short=HedgeLegState(HedgeLegSide.SHORT),
        )
        self.assertTrue(
            self.masker.evaluate(HedgeActions.CLOSE_BOTH, account=long_only, mark=100).allowed
        )
        self.assertTrue(
            self.masker.evaluate(
                HedgeActions.EMERGENCY_REDUCE_BOTH, account=long_only, mark=100
            ).allowed
        )


if __name__ == "__main__":
    unittest.main()
