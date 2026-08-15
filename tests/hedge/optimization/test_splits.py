import unittest
from datetime import UTC, datetime, timedelta

from freqtrade.hedge.optimization.splits import WalkForwardSpec, build_walk_forward_windows


class WalkForwardSplitTest(unittest.TestCase):
    def test_rolling_windows_have_expected_boundaries(self) -> None:
        windows = build_walk_forward_windows(
            130,
            WalkForwardSpec(train_size=60, validation_size=15, test_size=15, step_size=15),
        )
        self.assertEqual(len(windows), 3)
        self.assertEqual((windows[0].train.start, windows[0].train.stop), (0, 60))
        self.assertEqual((windows[1].train.start, windows[1].test.stop), (15, 105))

    def test_expanding_purge_and_embargo_do_not_overlap(self) -> None:
        windows = build_walk_forward_windows(
            100,
            WalkForwardSpec(
                train_size=50,
                validation_size=10,
                test_size=10,
                step_size=10,
                expanding=True,
                purge_size=5,
                embargo_size=2,
            ),
        )
        first = windows[0]
        self.assertEqual((first.train.start, first.train.stop), (0, 45))
        self.assertEqual((first.validation.start, first.validation.stop), (52, 62))
        self.assertEqual((first.test.start, first.test.stop), (64, 74))
        self.assertLess(first.train.stop, first.validation.start)

    def test_timestamps_must_be_strict_and_are_bound_to_windows(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        timestamps = [start + timedelta(minutes=i) for i in range(20)]
        window = build_walk_forward_windows(
            20,
            WalkForwardSpec(10, 5, 5),
            timestamps=timestamps,
        )[0]
        self.assertEqual(window.train_start_time, timestamps[0])
        self.assertEqual(window.test_end_time, timestamps[19])

    def test_insufficient_history_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least"):
            build_walk_forward_windows(10, WalkForwardSpec(10, 5, 5))


if __name__ == "__main__":
    unittest.main()
