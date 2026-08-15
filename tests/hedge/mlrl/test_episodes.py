import unittest

from freqtrade.freqai.hedge_rl.episodes import (
    chronological_split,
    sample_episode_start,
    walk_forward_slices,
)


class TestRound11Episodes(unittest.TestCase):
    def test_chronological_split_has_embargo(self):
        train, validation, test = chronological_split(100, embargo=3)
        self.assertEqual(validation.start - train.stop, 3)
        self.assertEqual(test.start - validation.stop, 3)
        self.assertLess(train.stop, validation.start)
        self.assertLess(validation.stop, test.start)

    def test_walk_forward_never_looks_ahead(self):
        folds = walk_forward_slices(
            100, train_length=40, evaluation_length=10, step=10, embargo=2
        )
        self.assertGreater(len(folds), 1)
        for train, validation in folds:
            self.assertLess(train.stop, validation.start)
            self.assertEqual(validation.start - train.stop, 2)

    def test_episode_sampling_deterministic(self):
        train, _, _ = chronological_split(200)
        a = sample_episode_start(train, window_size=16, episode_steps=20, seed=42)
        b = sample_episode_start(train, window_size=16, episode_steps=20, seed=42)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
