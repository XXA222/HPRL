import unittest
from decimal import Decimal

from freqtrade.hedge.optimization.fingerprint import (
    canonical_json,
    fingerprint,
    parameter_fingerprint,
)


class FingerprintTest(unittest.TestCase):
    def test_mapping_order_does_not_change_fingerprint(self) -> None:
        left = {"b": Decimal("1.0"), "a": [2, 3]}
        right = {"a": [2, 3], "b": Decimal("1.0")}
        self.assertEqual(fingerprint(left), fingerprint(right))

    def test_decimal_text_is_exact_and_stable(self) -> None:
        self.assertIn(b'"0.0100"', canonical_json({"x": Decimal("0.0100")}))
        self.assertEqual(
            parameter_fingerprint({"x": Decimal("0.1")}),
            parameter_fingerprint({"x": Decimal("0.1")}),
        )

    def test_nonfinite_float_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite"):
            fingerprint({"x": float("nan")})


class StudySemanticFingerprintTest(unittest.TestCase):
    def test_material_extra_definition_changes_study_fingerprint(self) -> None:
        from freqtrade.hedge.optimization.fingerprint import study_fingerprint

        common = {
            "parameter_specs": (),
            "objective_specs": (),
            "constraint_specs": (),
            "dataset_fingerprint": "data",
            "seed": 42,
            "sampler": "grid",
        }
        one = study_fingerprint(**common, extra_definition={"stress": "baseline"})
        two = study_fingerprint(**common, extra_definition={"stress": "fees_2x"})
        self.assertNotEqual(one, two)


if __name__ == "__main__":
    unittest.main()
