import json
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from freqtrade.hedge.optimization.artifacts import export_optimization_result
from freqtrade.hedge.optimization.types import (
    ObjectiveDirection,
    ObjectiveSpec,
    OptimizationResult,
    TrialRecord,
    TrialStatus,
)


class OptimizationArtifactTest(unittest.TestCase):
    def test_exports_atomic_machine_readable_artifacts_without_base_config(self) -> None:
        trial = TrialRecord(
            trial_id=0,
            parameter_hash="hash",
            parameters={"grid": Decimal("0.01")},
            status=TrialStatus.COMPLETE,
            metrics={"net_return": Decimal("0.1")},
            objective_values=(Decimal("0.1"),),
            scalar_score=Decimal("0.1"),
            duration_seconds=Decimal("1.2"),
            dataset_fingerprint="data",
            config_fingerprint="config-secret-hash-only",
        )
        result = OptimizationResult(
            study_name="demo",
            trials=(trial,),
            pareto_trial_ids=(0,),
            best_trial_id=0,
            objective_specs=(ObjectiveSpec("net_return", ObjectiveDirection.MAXIMIZE),),
            dataset_fingerprint="data",
            study_fingerprint="study",
        )
        with TemporaryDirectory() as directory:
            artifacts = export_optimization_result(result, directory)
            self.assertTrue(artifacts.summary_json.exists())
            self.assertTrue(artifacts.trials_csv.exists())
            self.assertTrue(artifacts.best_parameters_json.exists())
            manifest = json.loads(artifacts.manifest_json.read_text(encoding="utf-8"))
            self.assertIn("optimization-summary.json", manifest["files"])
            summary_text = artifacts.summary_json.read_text(encoding="utf-8")
            self.assertNotIn("api_key", summary_text)
            self.assertNotIn(".tmp", " ".join(path.name for path in Path(directory).iterdir()))

    def test_no_best_file_for_all_failed_study(self) -> None:
        result = OptimizationResult(
            study_name="failed",
            trials=(TrialRecord(0, "x", {}, TrialStatus.FAILED),),
            pareto_trial_ids=(),
            best_trial_id=None,
            objective_specs=(ObjectiveSpec("return", ObjectiveDirection.MAXIMIZE),),
            dataset_fingerprint="data",
            study_fingerprint="study",
        )
        with TemporaryDirectory() as directory:
            artifacts = export_optimization_result(result, directory)
            self.assertIsNone(artifacts.best_parameters_json)


if __name__ == "__main__":
    unittest.main()
