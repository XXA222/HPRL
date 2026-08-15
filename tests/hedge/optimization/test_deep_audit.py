from __future__ import annotations

import os
import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipInfo

from freqtrade.hedge.backtesting.advanced_metrics import (
    conditional_value_at_risk,
    exposure_ratio,
    omega_ratio,
    tail_ratio,
)
from freqtrade.hedge.backtesting.advanced_splits import (
    anchored_windows,
    leakage_audit,
    purged_kfold,
    regime_stratified_folds,
)
from freqtrade.hedge.backtesting.data_quality import (
    build_data_quality_report,
    canonical_event_fingerprint,
    detect_gaps,
    validate_ohlcv_bar,
    validate_strict_timestamps,
)
from freqtrade.hedge.backtesting.execution_realism import (
    fee_rate_by_volume,
    participation_cap,
    quantize_price,
    square_root_slippage_bps,
)
from freqtrade.hedge.deployment.release_validation import (
    atomic_write,
    overlay_plan,
    payload_manifest,
    safe_zip_member,
    validate_zip_members,
    validation_level_plan,
    verify_manifest,
)
from freqtrade.hedge.optimization.advanced_space import (
    candidate_hash,
    halton_samples,
    latin_hypercube,
    parameter_distance,
    validate_monotonic_constraints,
)
from freqtrade.hedge.optimization.orchestration import (
    deterministic_shards,
    failure_budget_exceeded,
    retry_policy,
    stable_merge_results,
)
from freqtrade.hedge.optimization.robust_selection import (
    epsilon_pareto,
    selection_entropy,
    spearman_rank_correlation,
    weighted_scenario_score,
)
from freqtrade.hedge.optimization.types import ParameterKind, ParameterSpec


T0 = datetime(2026, 1, 1, tzinfo=UTC)


class OptimizationDeepAuditTest(unittest.TestCase):
    def test_naive_timestamp_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_strict_timestamps((datetime(2026, 1, 1),))  # noqa: DTZ001

    def test_equal_timestamp_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_strict_timestamps((T0, T0))

    def test_ohlcv_outside_bounds_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_ohlcv_bar({"open": 10, "high": 9, "low": 8, "close": 10, "volume": 1})

    def test_irregular_gap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            detect_gaps((T0, T0 + timedelta(seconds=61)), timeframe_seconds=60)

    def test_quality_report_handles_empty_input(self) -> None:
        report = build_data_quality_report((), (), timeframe_seconds=60)
        self.assertEqual(report.score, Decimal(0))

    def test_fingerprint_rejects_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            canonical_event_fingerprint(({"value": float("inf")},))

    def test_round_up_price_quantization(self) -> None:
        self.assertEqual(quantize_price("1.001", "0.01", round_up=True), Decimal("1.01"))

    def test_fee_tier_requires_zero_origin(self) -> None:
        with self.assertRaises(ValueError):
            fee_rate_by_volume(10, ((1, "0.01"),))

    def test_participation_above_one_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            participation_cap(100, "1.1", 10)

    def test_square_root_slippage_rejects_zero_market(self) -> None:
        with self.assertRaises(ValueError):
            square_root_slippage_bps(1, 0, 1)

    def test_omega_no_losses_is_infinite(self) -> None:
        self.assertEqual(omega_ratio((1, 2, 3)), Decimal("Infinity"))

    def test_tail_ratio_requires_upper_quantile(self) -> None:
        with self.assertRaises(ValueError):
            tail_ratio((1, 2), quantile=Decimal("0.5"))

    def test_cvar_requires_valid_alpha(self) -> None:
        with self.assertRaises(ValueError):
            conditional_value_at_risk((1, 2), alpha=Decimal(1))

    def test_exposure_rejects_active_above_total(self) -> None:
        with self.assertRaises(ValueError):
            exposure_ratio(2, 1)

    def test_anchored_windows_require_history(self) -> None:
        with self.assertRaises(ValueError):
            anchored_windows(5, minimum_train=4, validation_size=1, test_size=1)

    def test_purged_folds_cover_all_test_indices(self) -> None:
        folds = purged_kfold(11, folds=3, purge=1, embargo=1)
        self.assertEqual(sorted(i for _, test in folds for i in test), list(range(11)))

    def test_leakage_audit_reports_all_overlap_types(self) -> None:
        issues = leakage_audit((0, 1), (1, 2), (0, 2))
        self.assertIn("train_validation_overlap", issues)
        self.assertIn("train_test_overlap", issues)
        self.assertIn("validation_test_overlap", issues)

    def test_regime_labels_cannot_be_empty(self) -> None:
        with self.assertRaises(ValueError):
            regime_stratified_folds(("bull", ""), folds=2)

    def test_lhs_rejects_integer_cardinality_overflow(self) -> None:
        specs = (ParameterSpec("x", "hedge.x", ParameterKind.INTEGER, low=1, high=2),)
        with self.assertRaises(ValueError):
            latin_hypercube(specs, count=3, seed=1)

    def test_halton_rejects_non_numeric_parameter(self) -> None:
        specs = (ParameterSpec("x", "hedge.x", ParameterKind.BOOLEAN),)
        with self.assertRaises(ValueError):
            halton_samples(specs, count=2)

    def test_candidate_hash_preserves_decimal_exactness(self) -> None:
        self.assertNotEqual(
            candidate_hash({"x": Decimal("1.0")}),
            candidate_hash({"x": Decimal("1.00")}),
        )

    def test_distance_requires_complete_candidates(self) -> None:
        specs = (ParameterSpec("x", "hedge.x", ParameterKind.INTEGER, low=1, high=2),)
        with self.assertRaises(ValueError):
            parameter_distance({}, {"x": 1}, specs)

    def test_monotonic_operator_is_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            validate_monotonic_constraints({"a": 1, "b": 2}, (("a", "!=", "b"),))

    def test_weighted_scenarios_require_matching_keys(self) -> None:
        with self.assertRaises(ValueError):
            weighted_scenario_score({"a": 1}, {"b": 1})

    def test_rank_correlation_rejects_ties(self) -> None:
        with self.assertRaises(ValueError):
            spearman_rank_correlation((1, 1, 2), (1, 2, 3))

    def test_entropy_zero_for_single_choice(self) -> None:
        self.assertEqual(selection_entropy({"only": 5}), Decimal("0.0"))

    def test_epsilon_pareto_dimension_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            epsilon_pareto({"a": (1, 2)}, epsilons=(0,), maximize=(True,))

    def test_retry_final_attempt_does_not_retry(self) -> None:
        decision = retry_policy(3, maximum_attempts=3, base_delay_seconds=1, retryable=True)
        self.assertFalse(decision.retry)

    def test_failure_budget_validates_counts(self) -> None:
        with self.assertRaises(ValueError):
            failure_budget_exceeded(2, 1, maximum_ratio="0.5")

    def test_more_workers_produce_empty_deterministic_shards(self) -> None:
        self.assertEqual(deterministic_shards((1, 2), workers=3), ((1,), (2,), ()))

    def test_result_merge_rejects_duplicate_trial_id(self) -> None:
        with self.assertRaises(ValueError):
            stable_merge_results(({"trial_id": 1}, {"trial_id": 1}))

    def test_zip_drive_and_backslash_traversal_are_rejected(self) -> None:
        self.assertFalse(safe_zip_member("C:/evil"))
        self.assertFalse(safe_zip_member("..\\evil"))

    def test_zip_duplicate_is_case_insensitive(self) -> None:
        with self.assertRaises(ValueError):
            validate_zip_members((ZipInfo("A.txt"), ZipInfo("a.txt")))

    def test_manifest_detects_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").write_text("one", encoding="utf-8")
            manifest = payload_manifest(root, ("a",))
            (root / "a").write_text("two", encoding="utf-8")
            self.assertEqual(verify_manifest(root, manifest), ("hash:a",))

    def test_atomic_write_leaves_no_temporary_file(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "a"
            atomic_write(target, b"x")
            self.assertEqual(tuple(root.glob("*.tmp")), ())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink unsupported")
    def test_manifest_blocks_symlink_escape(self) -> None:
        with TemporaryDirectory() as directory, TemporaryDirectory() as outside:
            root = Path(directory)
            external = Path(outside) / "secret"
            external.write_text("secret", encoding="utf-8")
            try:
                (root / "link").symlink_to(external)
            except OSError:
                self.skipTest("symlink creation unavailable")
            with self.assertRaises(ValueError):
                payload_manifest(root, ("link",))

    def test_overlay_requires_source_file(self) -> None:
        with TemporaryDirectory() as source, TemporaryDirectory() as target:
            with self.assertRaises(FileNotFoundError):
                overlay_plan(Path(source), Path(target), ("missing",))

    def test_full_validation_fails_without_dependencies(self) -> None:
        with self.assertRaises(ValueError):
            validation_level_plan("full", full_dependencies_available=False)


if __name__ == "__main__":
    unittest.main()
