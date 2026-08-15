import unittest
from decimal import Decimal

from freqtrade.hedge.optimization.config_patch import apply_parameters
from freqtrade.hedge.optimization.types import ParameterKind, ParameterSpec


class SafeConfigPatchTest(unittest.TestCase):
    def test_patch_is_isolated_and_exact(self) -> None:
        original = {"hedge": {"planner": {"grid_spacing": "0.01"}}, "dry_run": True}
        spec = ParameterSpec(
            "grid", "hedge.planner.grid_spacing", ParameterKind.DECIMAL,
            low="0.001", high="0.02", step="0.001",
        )
        patched = apply_parameters(original, (spec,), {"grid": "0.007"})
        self.assertEqual(patched["hedge"]["planner"]["grid_spacing"], Decimal("0.007"))
        self.assertEqual(original["hedge"]["planner"]["grid_spacing"], "0.01")

    def test_live_gate_cannot_be_optimized(self) -> None:
        spec = ParameterSpec(
            "live", "hedge.live_trading_enabled", ParameterKind.BOOLEAN
        )
        with self.assertRaisesRegex(ValueError, "allowlisted"):
            apply_parameters({"hedge": {}}, (spec,), {"live": True})

    def test_missing_or_unknown_values_fail_closed(self) -> None:
        spec = ParameterSpec(
            "layers", "hedge.planner.max_grid_layers", ParameterKind.INTEGER,
            low=1, high=5,
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            apply_parameters({}, (spec,), {})
        with self.assertRaisesRegex(ValueError, "unknown"):
            apply_parameters({}, (spec,), {"layers": 2, "extra": 1})

    def test_step_alignment_is_enforced(self) -> None:
        spec = ParameterSpec(
            "layers", "hedge.planner.max_grid_layers", ParameterKind.INTEGER,
            low=1, high=7, step=2,
        )
        with self.assertRaisesRegex(ValueError, "step"):
            apply_parameters({}, (spec,), {"layers": 2})


if __name__ == "__main__":
    unittest.main()
