import unittest

from freqtrade.freqai.hedge_rl.config import HedgeRLConfig


class TestRound01Config(unittest.TestCase):
    def test_parse_and_roundtrip(self):
        cfg = HedgeRLConfig.from_config(
            {
                "freqai": {
                    "hedge_rl_config": {
                        "observation_window": 16,
                        "action_size_fractions": [0.1, 0.5],
                    }
                }
            }
        )
        self.assertEqual(cfg.observation_window, 16)
        self.assertEqual(cfg.action_size_fractions, (0.1, 0.5))
        self.assertEqual(cfg.to_dict()["action_size_fractions"], [0.1, 0.5])

    def test_rejects_invalid_exposure(self):
        with self.assertRaises(ValueError):
            HedgeRLConfig(max_side_exposure=1.0, max_gross_exposure=0.5)


if __name__ == "__main__":
    unittest.main()
