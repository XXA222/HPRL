import unittest

from freqtrade.freqai.hedge_rl.actions import (
    DEFAULT_ACTION_CATALOG,
    HedgeActions,
    LegCommand,
    Urgency,
)


class TestRound02Actions(unittest.TestCase):
    def test_catalog_is_dense_and_stable(self):
        self.assertEqual(len(DEFAULT_ACTION_CATALOG), 21)
        self.assertEqual(DEFAULT_ACTION_CATALOG.names()[0], "HOLD")
        self.assertEqual(DEFAULT_ACTION_CATALOG.names()[-1], "EMERGENCY_REDUCE_BOTH")
        self.assertEqual([int(a) for a in HedgeActions], list(range(21)))

    def test_dual_leg_and_emergency_semantics(self):
        both = DEFAULT_ACTION_CATALOG.decode(HedgeActions.BOTH_OPEN_SMALL)
        self.assertEqual(both.long_command, LegCommand.OPEN)
        self.assertEqual(both.short_command, LegCommand.OPEN)
        emergency = DEFAULT_ACTION_CATALOG.decode(20)
        self.assertEqual(emergency.urgency, Urgency.URGENT)
        self.assertEqual(emergency.long_fraction, 0.5)

    def test_unknown_action_rejected(self):
        with self.assertRaises(ValueError):
            DEFAULT_ACTION_CATALOG.decode(999)


if __name__ == "__main__":
    unittest.main()
