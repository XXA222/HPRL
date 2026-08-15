import unittest

import numpy as np

from freqtrade.freqai.hedge_rl.normalization import RunningNormalizer


class TestRound04Normalization(unittest.TestCase):
    def test_welford_and_roundtrip(self):
        norm = RunningNormalizer(2, clip=5)
        norm.update([[1, 10], [3, 14], [5, 18]])
        self.assertTrue(np.allclose(norm.mean, [3, 14]))
        restored = RunningNormalizer.from_state_dict(norm.state_dict())
        self.assertTrue(np.allclose(restored.normalize([3, 14]), [0, 0]))

    def test_finite_sanitization_and_clip(self):
        norm = RunningNormalizer(2, clip=2)
        norm.update([[0, 0], [1, 1]])
        out = norm.normalize([float("inf"), float("nan")])
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue((np.abs(out) <= 2).all())

    def test_shape_rejected(self):
        with self.assertRaises(ValueError):
            RunningNormalizer(3).normalize([1, 2])


if __name__ == "__main__":
    unittest.main()
