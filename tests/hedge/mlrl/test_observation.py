import unittest

import numpy as np

from freqtrade.freqai.hedge_rl.observation import HedgeObservationBuilder, ObservationSchema
from freqtrade.freqai.hedge_rl.state import HedgeAccountState, HedgeLegSide, HedgeLegState


class TestRound05Observation(unittest.TestCase):
    def test_causal_flat_observation(self):
        schema = ObservationSchema(("ret", "vol"), 3)
        builder = HedgeObservationBuilder(schema)
        features = np.arange(20, dtype=float).reshape(10, 2)
        account = HedgeAccountState(
            cash_balance=1000,
            equity=1010,
            peak_equity=1020,
            long=HedgeLegState(HedgeLegSide.LONG, 1, 100),
            short=HedgeLegState(HedgeLegSide.SHORT),
            step=5,
        )
        obs = builder.build(
            features,
            tick=4,
            account=account,
            mark=105,
            maintenance_rate=0.05,
            max_episode_steps=100,
        )
        self.assertEqual(obs.shape, (schema.flat_size,))
        self.assertTrue(np.array_equal(obs[:6], features[2:5].reshape(-1)))
        self.assertNotIn(10.0, obs[:6])

    def test_schema_signature_is_stable(self):
        a = ObservationSchema(("a", "b"), 4)
        b = ObservationSchema(("a", "b"), 4)
        self.assertEqual(a.signature, b.signature)
        self.assertNotEqual(a.signature, ObservationSchema(("b", "a"), 4).signature)

    def test_incomplete_window_rejected(self):
        with self.assertRaises(IndexError):
            HedgeObservationBuilder(ObservationSchema(("x",), 4)).build(
                np.ones((5, 1)),
                tick=2,
                account=HedgeAccountState.initial(1000),
                mark=100,
                maintenance_rate=0.05,
                max_episode_steps=10,
            )


if __name__ == "__main__":
    unittest.main()
