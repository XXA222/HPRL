import unittest
from decimal import Decimal

from freqtrade.hedge.optimization.pareto import dominates, non_dominated_ranks, pareto_front
from freqtrade.hedge.optimization.types import (
    ObjectiveDirection,
    ObjectiveSpec,
    TrialRecord,
    TrialStatus,
)


OBJECTIVES = (
    ObjectiveSpec("return", ObjectiveDirection.MAXIMIZE),
    ObjectiveSpec("drawdown", ObjectiveDirection.MINIMIZE),
)


def trial(trial_id: int, values: tuple[str, str]) -> TrialRecord:
    return TrialRecord(
        trial_id=trial_id,
        parameter_hash=str(trial_id),
        parameters={},
        status=TrialStatus.COMPLETE,
        objective_values=tuple(Decimal(value) for value in values),
    )


class ParetoTest(unittest.TestCase):
    def test_direction_aware_dominance(self) -> None:
        self.assertTrue(
            dominates((Decimal(2), Decimal("0.1")), (Decimal(1), Decimal("0.2")), OBJECTIVES)
        )
        self.assertFalse(
            dominates((Decimal(2), Decimal("0.3")), (Decimal(1), Decimal("0.2")), OBJECTIVES)
        )

    def test_front_preserves_tradeoffs_and_excludes_dominated(self) -> None:
        trials = (
            trial(1, ("2", "0.3")),
            trial(2, ("1", "0.1")),
            trial(3, ("1", "0.2")),
        )
        self.assertEqual(tuple(item.trial_id for item in pareto_front(trials, OBJECTIVES)), (1, 2))
        self.assertEqual(non_dominated_ranks(trials, OBJECTIVES), {1: 0, 2: 0, 3: 1})

    def test_failed_trials_never_enter_front(self) -> None:
        failed = TrialRecord(
            4,
            "x",
            {},
            TrialStatus.FAILED,
            objective_values=(Decimal(99), Decimal(0)),
        )
        self.assertEqual(pareto_front((failed,), OBJECTIVES), ())


if __name__ == "__main__":
    unittest.main()
