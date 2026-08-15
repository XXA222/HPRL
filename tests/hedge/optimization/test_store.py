import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from freqtrade.hedge.optimization.store import StudyStore
from freqtrade.hedge.optimization.types import TrialRecord, TrialStatus


class StudyStoreTest(unittest.TestCase):
    def test_round_trip_and_resume_by_parameter_hash(self) -> None:
        with TemporaryDirectory() as directory:
            store = StudyStore(Path(directory) / "study.sqlite")
            store.initialize_study(
                study_name="demo",
                study_fingerprint="study-1",
                dataset_fingerprint="data-1",
                definition={"sampler": "grid"},
            )
            record = TrialRecord(
                trial_id=2,
                parameter_hash="param-2",
                parameters={"grid": "0.01"},
                status=TrialStatus.COMPLETE,
                metrics={"net_return": Decimal("0.12")},
                objective_values=(Decimal("0.12"),),
                scalar_score=Decimal("0.12"),
                duration_seconds=Decimal("1.5"),
                dataset_fingerprint="data-1",
                config_fingerprint="config-1",
                worker="thread-1",
            )
            store.save_trial("demo", record)
            loaded = store.load_trials("demo")
            self.assertEqual(loaded, (record,))
            self.assertIn("param-2", store.completed_by_parameter_hash("demo"))

    def test_study_name_cannot_be_reused_for_different_dataset(self) -> None:
        with TemporaryDirectory() as directory:
            store = StudyStore(Path(directory) / "study.sqlite")
            store.initialize_study(
                study_name="demo",
                study_fingerprint="one",
                dataset_fingerprint="data-one",
                definition={},
            )
            with self.assertRaisesRegex(ValueError, "different"):
                store.initialize_study(
                    study_name="demo",
                    study_fingerprint="two",
                    dataset_fingerprint="data-two",
                    definition={},
                )

    def test_upsert_does_not_duplicate_same_parameters(self) -> None:
        with TemporaryDirectory() as directory:
            store = StudyStore(Path(directory) / "study.sqlite")
            store.initialize_study(
                study_name="demo",
                study_fingerprint="one",
                dataset_fingerprint="data",
                definition={},
            )
            pending = TrialRecord(1, "same", {"x": 1}, TrialStatus.RUNNING)
            complete = TrialRecord(1, "same", {"x": 1}, TrialStatus.COMPLETE)
            store.save_trial("demo", pending)
            store.save_trial("demo", complete)
            self.assertEqual(store.load_trials("demo"), (complete,))


if __name__ == "__main__":
    unittest.main()
