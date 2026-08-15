import tempfile
import unittest
from pathlib import Path

import torch

from freqtrade.freqai.hedge_rl.networks import HedgeMultiTaskMLP
from freqtrade.freqai.hedge_rl.registry import HedgeModelRegistry, ModelManifest


class TestModelRegistry(unittest.TestCase):
    def test_atomic_save_load_and_schema_check(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = HedgeModelRegistry(directory)
            model = HedgeMultiTaskMLP(4, hidden_dim=8, n_layer=1)
            manifest = ModelManifest(
                model_version="registry-test",
                model_kind="multitask",
                observation_schema_signature="schema-1",
                source_version="clean-mainline",
            )
            artifact, manifest_path = registry.save("btc-model", model, manifest)
            self.assertTrue(artifact.is_file())
            self.assertTrue(manifest_path.is_file())
            payload = registry.load_state(
                "btc-model", expected_observation_schema="schema-1"
            )
            self.assertIn("model_state_dict", payload)
            with self.assertRaises(ValueError):
                registry.load_state("btc-model", expected_observation_schema="schema-2")

    def test_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = HedgeModelRegistry(directory)
            model = torch.nn.Linear(2, 1)
            registry.save(
                "model",
                model,
                ModelManifest("v1", "policy", "schema", "clean-mainline"),
            )
            artifact = Path(directory) / "model.pt"
            artifact.write_bytes(artifact.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "checksum"):
                registry.load_state("model", expected_observation_schema="schema")

    def test_unsafe_name_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                HedgeModelRegistry(directory).read_manifest("../escape")


if __name__ == "__main__":
    unittest.main()
