import unittest
from decimal import Decimal

from freqtrade.hedge.optimization.pruning import MedianPruningPolicy, should_prune
from freqtrade.hedge.optimization.types import ObjectiveDirection


class MedianPruningTest(unittest.TestCase):
    def test_does_not_prune_during_warmup_or_without_enough_peers(self) -> None:
        policy = MedianPruningPolicy(warmup_steps=2, minimum_completed_trials=2)
        histories = {1: (Decimal(1), Decimal(1), Decimal(1))}
        self.assertFalse(
            should_prune(
                step=1,
                current_value=Decimal(-10),
                completed_histories=histories,
                policy=policy,
            )
        )
        self.assertFalse(
            should_prune(
                step=2,
                current_value=Decimal(-10),
                completed_histories=histories,
                policy=policy,
            )
        )

    def test_maximize_trial_below_peer_median_is_pruned(self) -> None:
        policy = MedianPruningPolicy(warmup_steps=0, minimum_completed_trials=3)
        histories = {
            1: (Decimal(1),),
            2: (Decimal(2),),
            3: (Decimal(3),),
        }
        self.assertTrue(
            should_prune(
                step=0,
                current_value=Decimal("1.5"),
                completed_histories=histories,
                policy=policy,
            )
        )
        self.assertFalse(
            should_prune(
                step=0,
                current_value=Decimal("2.5"),
                completed_histories=histories,
                policy=policy,
            )
        )

    def test_minimize_direction_is_reversed(self) -> None:
        policy = MedianPruningPolicy(
            direction=ObjectiveDirection.MINIMIZE,
            warmup_steps=0,
            minimum_completed_trials=2,
        )
        histories = {1: (Decimal("0.1"),), 2: (Decimal("0.2"),)}
        self.assertTrue(
            should_prune(
                step=0,
                current_value=Decimal("0.3"),
                completed_histories=histories,
                policy=policy,
            )
        )


if __name__ == "__main__":
    unittest.main()
