from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from freqtrade.hedge.backtesting.advanced_splits import infer_timeframe_seconds
from freqtrade.hedge.backtesting.quality import validate_bar_spacing
from freqtrade.hedge.backtesting.cache import EvaluationCache
from freqtrade.hedge.backtesting.config import load_optimization_config
from freqtrade.hedge.backtesting.contracts import Candidate, ObjectiveConfig
from freqtrade.hedge.backtesting.data_quality import detect_gaps, validate_timeframe_alignment
from freqtrade.hedge.backtesting.dataset_io import load_dataset_json
from freqtrade.hedge.backtesting.decimal_utils import canonical_json
from freqtrade.hedge.backtesting.overfit import selection_entropy
from freqtrade.hedge.backtesting.runner import objective_score, planner_config_with_overrides
from freqtrade.hedge.backtesting.walkforward import run_walk_forward
from freqtrade.hedge.optimization.config import parse_optimization_config
from freqtrade.hedge.optimization.engine import OptimizationEngine
from freqtrade.hedge.optimization.freqtrade_adapter import run_freqtrade_hedge_optimization
from freqtrade.hedge.optimization.metrics import normalize_report
from freqtrade.hedge.optimization.splits import WalkForwardSpec
from freqtrade.hedge.optimization.types import ParameterKind, ParameterSpec, TrialStatus
from freqtrade.hedge.planning.context import PlannerConfig
from freqtrade.hedge.simulation.exchange import BarEvent


class OptimizationCodeQualityTest(unittest.TestCase):
    def _write_backtesting_config(self, directory: str, **overrides: object) -> Path:
        payload: dict[str, object] = {
            "schema_version": "hedge-optimization-config-v1",
            "method": "random",
            "space": {
                "x": {"type": "decimal", "low": "0.1", "high": "0.2", "log": False}
            },
        }
        payload.update(overrides)
        path = Path(directory) / "optimization.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_round_01_decimal_log_flag_is_strict_boolean(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_backtesting_config(
                directory,
                space={"x": {"type": "decimal", "low": "0.1", "high": "0.2", "log": "false"}},
            )
            with self.assertRaisesRegex(TypeError, "must be bool"):
                load_optimization_config(path)

    def test_round_02_integer_log_flag_is_strict_boolean(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_backtesting_config(
                directory,
                space={"x": {"type": "int", "low": 1, "high": 3, "log": "false"}},
            )
            with self.assertRaisesRegex(TypeError, "must be bool"):
                load_optimization_config(path)

    def test_round_03_integer_parameter_rejects_fractional_bound(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_backtesting_config(
                directory,
                space={"x": {"type": "int", "low": 1.5, "high": 3}},
            )
            with self.assertRaisesRegex(TypeError, "must be int"):
                load_optimization_config(path)

    def test_round_04_objective_boolean_does_not_use_truthiness(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_backtesting_config(
                directory,
                objective={"reject_liquidation": "false"},
            )
            with self.assertRaisesRegex(TypeError, "must be bool"):
                load_optimization_config(path)

    def test_round_05_top_level_worker_count_is_strict_integer(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_backtesting_config(directory, workers=1.5)
            with self.assertRaisesRegex(TypeError, "workers must be int"):
                load_optimization_config(path)

    def test_round_06_dataset_json_boolean_does_not_use_truthiness(self) -> None:
        payload = {
            "schema_version": "hedge-backtest-dataset-v1",
            "dataset_id": "strict-bool",
            "timeframe": "1m",
            "metadata": {},
            "events": [
                {
                    "kind": "signal",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "symbol": "BTC/USDT:USDT",
                    "long_signal": "1",
                    "short_signal": "0",
                    "allow_new_risk": "false",
                },
                {
                    "kind": "bar",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "symbol": "BTC/USDT:USDT",
                    "open": "100",
                    "high": "101",
                    "low": "99",
                    "close": "100",
                },
            ],
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(TypeError, "JSON boolean"):
                load_dataset_json(path)

    def test_round_07_set_canonicalization_is_sorted(self) -> None:
        self.assertEqual(
            canonical_json({"values": {"beta", "alpha"}}), b'{"values":["alpha","beta"]}'
        )

    def test_round_08_subsecond_timestamp_is_not_timeframe_aligned(self) -> None:
        timestamp = datetime(2026, 1, 1, 0, 1, 0, 500_000, tzinfo=UTC)
        self.assertFalse(validate_timeframe_alignment(timestamp, timeframe_seconds=60))

    def test_round_09_subsecond_gap_is_rejected(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        with self.assertRaisesRegex(ValueError, "not aligned"):
            detect_gaps(
                (start, start + timedelta(seconds=60, microseconds=1)), timeframe_seconds=60
            )

    def test_round_10_subsecond_bar_spacing_is_rejected(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        bars = (
            BarEvent(
                start,
                "BTC/USDT:USDT",
                Decimal(100),
                Decimal(101),
                Decimal(99),
                Decimal(100),
            ),
            BarEvent(
                start + timedelta(seconds=60, microseconds=1),
                "BTC/USDT:USDT",
                Decimal(100),
                Decimal(101),
                Decimal(99),
                Decimal(100),
            ),
        )
        with self.assertRaisesRegex(ValueError, "bar spacing"):
            validate_bar_spacing(bars, timeframe_seconds=60)

    def test_round_11_walk_forward_empty_fold_set_has_explicit_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one fold"):
            run_walk_forward(folds=(), candidates=(SimpleNamespace(candidate_id="candidate-x"),))

    def test_round_12_selection_entropy_is_order_independent(self) -> None:
        left = selection_entropy(("a", "b", "a", "c", "a", "b"))
        right = selection_entropy(("b", "a", "c", "b", "a", "a"))
        self.assertEqual(left, right)

    def _engine_config(self, directory: str) -> dict:
        return {
            "hedge": {
                "planner": {},
                "paper": {},
                "optimization": {
                    "study_name": "quality-engine",
                    "sampler": "grid",
                    "trials": 1,
                    "storage_path": str(Path(directory) / "study.sqlite"),
                    "output_directory": str(Path(directory) / "out"),
                    "parameters": [
                        {
                            "name": "layers",
                            "path": "hedge.planner.max_grid_layers",
                            "kind": "integer",
                            "low": 1,
                            "high": 1,
                        }
                    ],
                    "objectives": [{"metric": "net_return", "direction": "maximize"}],
                },
            }
        }

    def test_round_13_empty_evaluator_report_becomes_failed_trial(self) -> None:
        with TemporaryDirectory() as directory:
            raw = self._engine_config(directory)
            result = OptimizationEngine(
                base_config=raw,
                optimization_config=parse_optimization_config(raw),
                evaluator=lambda _candidate, _context: {},
                dataset_fingerprint="quality-dataset",
            ).run()
            self.assertIs(result.trials[0].status, TrialStatus.FAILED)
            self.assertIn("empty report", result.trials[0].error or "")

    def test_round_14_nonscalar_evaluator_metric_becomes_failed_trial(self) -> None:
        with TemporaryDirectory() as directory:
            raw = self._engine_config(directory)
            result = OptimizationEngine(
                base_config=raw,
                optimization_config=parse_optimization_config(raw),
                evaluator=lambda _candidate, _context: {"net_return": [1, 2]},
                dataset_fingerprint="quality-dataset",
            ).run()
            self.assertIs(result.trials[0].status, TrialStatus.FAILED)
            self.assertIn("supported scalar", result.trials[0].error or "")

    @staticmethod
    def _adapter_config(
        directory: str, *, walk_forward: bool = False, stress: bool = False
    ) -> dict:
        optimization: dict[str, object] = {
            "study_name": "adapter-quality",
            "sampler": "grid",
            "trials": 1,
            "storage_path": str(Path(directory) / "adapter.sqlite"),
            "output_directory": str(Path(directory) / "out"),
            "parameters": [
                {
                    "name": "layers",
                    "path": "hedge.planner.max_grid_layers",
                    "kind": "integer",
                    "low": 1,
                    "high": 1,
                }
            ],
            "objectives": [{"metric": "net_return", "direction": "maximize"}],
        }
        if walk_forward:
            optimization["walk_forward"] = {
                "train_size": 2,
                "validation_size": 1,
                "test_size": 2,
                "minimum_windows": 1,
            }
        if stress:
            optimization["stress_scenarios"] = [
                {"name": "funding_2x", "funding_rate_multiplier": "2"}
            ]
        return {
            "user_data_dir": directory,
            "hedge": {"planner": {}, "paper": {}, "optimization": optimization},
        }

    @staticmethod
    def _fake_run(*, fingerprint: str, bar_count: int, report: dict[str, object] | None = None):
        start = datetime(2026, 1, 1, tzinfo=UTC)
        events = tuple(
            BarEvent(
                start + timedelta(minutes=index),
                "BTC/USDT:USDT",
                Decimal(100), Decimal(101), Decimal(99), Decimal(100),
            )
            for index in range(bar_count)
        )
        dataset = SimpleNamespace(
            events=events,
            pair="BTC/USDT:USDT",
            timeframe="1m",
            start=events[0].timestamp,
            end=events[-1].timestamp,
            bar_count=bar_count,
            data_fingerprint=fingerprint,
        )
        result = SimpleNamespace(
            report=report or {
                "initial_balance": "1000",
                "final_equity": "1001",
                "net_return": "0.001",
            }
        )
        return SimpleNamespace(dataset=dataset, result=result)

    def test_round_15_full_baseline_fingerprint_drift_is_still_rejected(self) -> None:
        with TemporaryDirectory() as directory:
            calls = 0

            def runner(_config, *, export_path, export_events):
                nonlocal calls
                calls += 1
                return self._fake_run(
                    fingerprint="probe" if calls == 1 else "drift",
                    bar_count=6,
                )

            result = run_freqtrade_hedge_optimization(
                self._adapter_config(directory), backtest_runner=runner
            )
            self.assertIs(result.result.trials[0].status, TrialStatus.FAILED)
            self.assertIn("fingerprint drifted", result.result.trials[0].error or "")

    def test_round_16_walk_forward_slice_fingerprint_may_differ_from_full_probe(self) -> None:
        with TemporaryDirectory() as directory:
            calls = 0

            def runner(config, *, export_path, export_events):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return self._fake_run(fingerprint="probe", bar_count=6)
                self.assertIn("timerange", config)
                return self._fake_run(fingerprint="slice", bar_count=2)

            result = run_freqtrade_hedge_optimization(
                self._adapter_config(directory, walk_forward=True),
                backtest_runner=runner,
            )
            self.assertIs(result.result.trials[0].status, TrialStatus.COMPLETE)

    def test_round_17_explicit_stress_fingerprint_may_differ_from_probe(self) -> None:
        with TemporaryDirectory() as directory:
            calls = 0

            def runner(config, *, export_path, export_events):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return self._fake_run(fingerprint="probe", bar_count=6)
                self.assertEqual(
                    str(config["hedge_optimization_runtime"]["funding_rate_multiplier"]), "2"
                )
                return self._fake_run(fingerprint="stress", bar_count=6)

            result = run_freqtrade_hedge_optimization(
                self._adapter_config(directory, stress=True),
                backtest_runner=runner,
            )
            self.assertIs(result.result.trials[0].status, TrialStatus.COMPLETE)

    def test_round_18_probe_artifacts_are_cleaned_when_probe_raises(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory) / "out"

            def runner(_config, *, export_path, export_events):
                export_path.parent.mkdir(parents=True, exist_ok=True)
                export_path.write_text("partial", encoding="utf-8")
                export_path.with_suffix(export_path.suffix + ".sha256").write_text(
                    "partial", encoding="ascii"
                )
                raise RuntimeError("probe failed")

            with self.assertRaisesRegex(RuntimeError, "probe failed"):
                run_freqtrade_hedge_optimization(
                    self._adapter_config(directory), backtest_runner=runner
                )
            self.assertFalse((output / ".dataset-probe.json").exists())
            self.assertFalse((output / ".dataset-probe.json.sha256").exists())

    def test_round_19_trial_artifacts_are_cleaned_when_trial_raises(self) -> None:
        with TemporaryDirectory() as directory:
            calls = 0
            output = Path(directory) / "out"

            def runner(_config, *, export_path, export_events):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return self._fake_run(fingerprint="probe", bar_count=6)
                export_path.parent.mkdir(parents=True, exist_ok=True)
                export_path.write_text("partial", encoding="utf-8")
                export_path.with_suffix(export_path.suffix + ".sha256").write_text(
                    "partial", encoding="ascii"
                )
                raise RuntimeError("trial failed")

            result = run_freqtrade_hedge_optimization(
                self._adapter_config(directory), backtest_runner=runner
            )
            self.assertIs(result.result.trials[0].status, TrialStatus.FAILED)
            self.assertFalse((output / ".trial-artifacts").exists())

    def test_round_20_quality_fixes_keep_valid_backtesting_config_parseable(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_backtesting_config(
                directory,
                workers=2,
                random_count=4,
                seed=7,
                max_candidates=100,
                objective={"reject_liquidation": False},
            )
            parsed = load_optimization_config(path)
            self.assertEqual(parsed["workers"], 2)
            self.assertEqual(parsed["random_count"], 4)
            self.assertFalse(parsed["objective_config"].reject_liquidation)

    def test_regression_walk_forward_config_is_strict(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_backtesting_config(
                directory,
                walk_forward={"train_bars": 1000, "test_bars": 250, "anchored": "false"},
            )
            with self.assertRaisesRegex(TypeError, "must be bool"):
                load_optimization_config(path)
        with TemporaryDirectory() as directory:
            path = self._write_backtesting_config(
                directory,
                walk_forward={"train_bars": 1000.5, "test_bars": 250},
            )
            with self.assertRaisesRegex(TypeError, "must be int"):
                load_optimization_config(path)

    def test_regression_report_liquidated_is_strict_boolean(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a boolean"):
            normalize_report({"net_return": "0.1", "liquidated": "false"})

    def test_regression_timeframe_inference_rejects_subsecond_cadence(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=UTC)
        with self.assertRaisesRegex(ValueError, "whole number of seconds"):
            infer_timeframe_seconds((start, start + timedelta(seconds=60, microseconds=1)))

    def test_regression_objective_score_is_finite_when_infeasible(self) -> None:
        score, feasible, violations = objective_score(
            {"total_return_ratio": Decimal("0.1"), "max_drawdown_ratio": Decimal("0.8")},
            ObjectiveConfig(maximums={"max_drawdown_ratio": Decimal("0.35")}),
        )
        self.assertFalse(feasible)
        self.assertTrue(violations)
        self.assertTrue(score.is_finite())

    def test_regression_planner_integer_override_rejects_fractional_value(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be int"):
            planner_config_with_overrides(PlannerConfig(), {"max_grid_layers": 1.5})


    def test_regression_optimization_config_rejects_unknown_keys(self) -> None:
        raw = self._engine_config(".")
        raw["hedge"]["optimization"]["worker"] = 2
        with self.assertRaisesRegex(ValueError, "unsupported key"):
            parse_optimization_config(raw)

    def test_regression_parameter_spec_log_is_strict_boolean(self) -> None:
        with self.assertRaisesRegex(TypeError, "log flag must be boolean"):
            ParameterSpec(
                name="layers",
                path="hedge.planner.max_grid_layers",
                kind=ParameterKind.INTEGER,
                low=1,
                high=2,
                log="false",
            )

    def test_regression_walk_forward_contract_rejects_truthy_strings(self) -> None:
        with self.assertRaisesRegex(TypeError, "expanding must be a boolean"):
            WalkForwardSpec(2, 1, 1, expanding="false")

    def test_regression_cache_feasible_is_strict_boolean(self) -> None:
        with TemporaryDirectory() as directory:
            cache = EvaluationCache(Path(directory))
            candidate = Candidate("candidate", {}, 0)
            key = cache.key("a" * 64, candidate, "b" * 64)
            payload = {
                "schema_version": "hedge-bt-cache-v1",
                "candidate": {"candidate_id": "candidate", "parameters": {}, "ordinal": 0},
                "dataset_fingerprint": "a" * 64,
                "metrics": {},
                "objective_score": "0",
                "feasible": "false",
                "violations": [],
                "elapsed_seconds": "0",
                "evaluated_at": "2026-01-01T00:00:00+00:00",
            }
            cache.path(key).write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(TypeError, "feasible must be a boolean"):
                cache.get(key)


if __name__ == "__main__":
    unittest.main()
