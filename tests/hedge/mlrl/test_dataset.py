import unittest

import numpy as np
import pandas as pd

from freqtrade.freqai.hedge_rl.dataset import (
    CausalWindowDataset,
    build_causal_market_features,
    validate_aligned_market_data,
)


class TestRound12Dataset(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2026-01-01", periods=30, freq="min", tz="UTC")
        close = np.linspace(100, 110, 30)
        self.prices = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.arange(30) + 1,
            },
            index=index,
        )

    def test_features_are_causal(self):
        before = build_causal_market_features(self.prices, volatility_window=5)
        changed = self.prices.copy()
        changed.iloc[-1, changed.columns.get_loc("close")] = 999
        after = build_causal_market_features(changed, volatility_window=5)
        pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])

    def test_window_returns_next_execution_tick(self):
        features = build_causal_market_features(self.prices).to_numpy()
        dataset = CausalWindowDataset(features, 4)
        window, next_tick = dataset[0]
        self.assertEqual(window.shape, (4, features.shape[1]))
        self.assertEqual(next_tick, 4)
        self.assertTrue(np.allclose(window, features[:4]))

    def test_alignment_and_suspicious_names(self):
        features = build_causal_market_features(self.prices)
        validate_aligned_market_data(features, self.prices)
        bad = features.rename(columns={features.columns[0]: "future_return"})
        with self.assertRaises(ValueError):
            validate_aligned_market_data(bad, self.prices)


if __name__ == "__main__":
    unittest.main()
