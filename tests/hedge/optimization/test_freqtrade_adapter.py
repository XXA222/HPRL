import unittest
from datetime import UTC, datetime, timedelta

from freqtrade.hedge.optimization.engine import EvaluationContext
from freqtrade.hedge.optimization.freqtrade_adapter import window_timerange
from freqtrade.hedge.optimization.splits import IndexRange, WalkForwardWindow
from freqtrade.hedge.optimization.stress import BASELINE_SCENARIO


class FreqtradeAdapterTest(unittest.TestCase):
    def test_window_timerange_uses_exact_millisecond_boundaries(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        timestamps = tuple(start + timedelta(minutes=index) for index in range(10))
        context = EvaluationContext(
            trial_id=1,
            seed=42,
            stress_scenario=BASELINE_SCENARIO,
            window=WalkForwardWindow(
                0,
                train=IndexRange(0, 4),
                validation=IndexRange(4, 6),
                test=IndexRange(6, 9),
            ),
            evaluation_index=0,
        )
        timerange = window_timerange(timestamps, context)
        self.assertEqual(
            timerange,
            f"{int(timestamps[6].timestamp() * 1000)}-{int(timestamps[9].timestamp() * 1000)}",
        )

    def test_final_window_infers_one_candle_stop(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        timestamps = tuple(start + timedelta(minutes=index) for index in range(5))
        context = EvaluationContext(
            1,
            42,
            BASELINE_SCENARIO,
            WalkForwardWindow(0, IndexRange(0, 1), IndexRange(1, 2), IndexRange(2, 5)),
            0,
        )
        timerange = window_timerange(timestamps, context)
        expected_stop = timestamps[-1] + timedelta(minutes=1)
        self.assertTrue(timerange.endswith(str(int(expected_stop.timestamp() * 1000))))


if __name__ == "__main__":
    unittest.main()
