import unittest
from decimal import Decimal

from freqtrade.hedge.optimization.space import ParameterSpace
from freqtrade.hedge.optimization.types import ParameterKind, ParameterSpec


class ParameterSpaceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.space = ParameterSpace(
            (
                ParameterSpec(
                    "layers", "hedge.planner.max_grid_layers", ParameterKind.INTEGER,
                    low=1, high=3,
                ),
                ParameterSpec(
                    "spacing", "hedge.planner.grid_spacing", ParameterKind.DECIMAL,
                    low="0.01", high="0.02", step="0.01",
                ),
                ParameterSpec(
                    "limit_only", "hedge.planner.unstuck_limit_only", ParameterKind.BOOLEAN,
                ),
            )
        )

    def test_grid_is_complete_and_stable(self) -> None:
        first = tuple(self.space.iter_grid())
        second = tuple(self.space.iter_grid())
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertEqual(first[0]["spacing"], Decimal("0.01"))

    def test_grid_explosion_is_rejected_before_iteration(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeding"):
            tuple(self.space.iter_grid(max_candidates=3))

    def test_random_sampling_is_seeded_and_unique(self) -> None:
        one = self.space.sample_random(5, seed=42)
        two = self.space.sample_random(5, seed=42)
        self.assertEqual(one, two)
        self.assertEqual(len({tuple(item.items()) for item in one}), 5)

    def test_candidate_requires_exact_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly"):
            self.space.validate_candidate({"layers": 1})


if __name__ == "__main__":
    unittest.main()
