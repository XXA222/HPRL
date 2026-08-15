"""Operational commands for Hedge native-feature convergence."""

from __future__ import annotations

import ast
from hashlib import sha256
from json import dumps, loads
from pathlib import Path
from typing import Any


def _print(payload: dict[str, Any], output: object = None) -> None:
    text = dumps(payload, indent=2, sort_keys=True, default=str)
    if output not in (None, ""):
        path = Path(str(output))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def start_hedge_native_audit(args: dict[str, Any]) -> int:
    from freqtrade.hedge.native.audit import AuditCheckResult, AuditStatus, NativeAuditRunner
    from freqtrade.hedge.native.exchange_capabilities import default_exchange_registry

    root = Path(str(args.get("project_root") or Path.cwd())).resolve()
    runner = NativeAuditRunner()
    required = (
        "freqtrade/hedge/native/state.py",
        "freqtrade/hedge/native/protections.py",
        "freqtrade/hedge/native/exits.py",
        "freqtrade/hedge/native/backtest.py",
        "freqtrade/hedge/native/hyperopt.py",
        "freqtrade/hedge/native/freqai.py",
        "freqtrade/hedge/native/rl.py",
        "freqtrade/hedge/native/rpc.py",
    )
    for index, relative in enumerate(required, start=1):
        runner.add(
            f"NATIVE-{index:02d}",
            f"Required module {relative}",
            lambda relative=relative: (root / relative).is_file(),
        )

    def parse_all() -> AuditCheckResult:
        failures: list[str] = []
        count = 0
        for path in sorted((root / "freqtrade").rglob("*.py")):
            count += 1
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except Exception as exc:
                failures.append(f"{path.relative_to(root)}:{type(exc).__name__}:{exc}")
        return AuditCheckResult(
            "NATIVE-AST",
            "Parse every Freqtrade Python source",
            AuditStatus.PASS if not failures else AuditStatus.FAIL,
            f"parsed={count}, failures={len(failures)}",
            {"failures": failures[:50]},
        )

    runner.add("NATIVE-AST", "Parse every Freqtrade Python source", parse_all)
    runner.add(
        "NATIVE-EXCHANGE",
        "Binance USD-M capability evidence remains explicit",
        lambda: default_exchange_registry().get("binance", "usdm").safe_for_live_write,
    )
    report = runner.run(name="Hedge native convergence source audit")
    payload = report.to_dict()
    _print(payload, args.get("hedge_native_output"))
    return 0 if report.passed else 1


def start_hedge_model_check(args: dict[str, Any]) -> int:
    from datetime import datetime, UTC
    from freqtrade.hedge.native.freqai import HedgeModelReadinessGate, manifest_from_mapping

    path = Path(str(args["hedge_model_manifest"]))
    payload = loads(path.read_text(encoding="utf-8"))
    manifest = manifest_from_mapping(payload)
    gate = HedgeModelReadinessGate(manifest, required=True)
    snapshot = gate.snapshot(at=datetime.now(UTC))
    result = {
        "schema": "hedge-model-check-v1",
        "ready": snapshot.ready,
        "model_version": snapshot.model_version,
        "feature_schema": snapshot.feature_schema,
        "trained_at": None if snapshot.trained_at is None else snapshot.trained_at.isoformat(),
        "expires_at": None if snapshot.expires_at is None else snapshot.expires_at.isoformat(),
        "reasons": list(snapshot.reasons),
        "manifest_sha256": sha256(path.read_bytes()).hexdigest(),
    }
    _print(result, args.get("hedge_native_output"))
    return 0 if snapshot.ready else 1


def start_hedge_contracts(args: dict[str, Any]) -> int:
    from dataclasses import fields
    from freqtrade.hedge.native.hyperopt import HedgeHyperoptSpace
    from freqtrade.hedge.native.rl import HedgeRLAction, HedgeRLState

    payload = {
        "schema": "hedge-native-contracts-v1",
        "hyperopt_parameters": [item.name for item in HedgeHyperoptSpace().spaces],
        "rl_actions": {item.name: int(item) for item in HedgeRLAction},
        "rl_state_fields": [item.name for item in fields(HedgeRLState)],
    }
    _print(payload, args.get("hedge_native_output"))
    return 0


def _load_json_file(path: object) -> dict[str, Any]:
    file_path = Path(str(path)).expanduser().resolve()
    payload = loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {file_path}")
    return payload


def _native_payload(payload: dict[str, Any]) -> dict[str, Any]:
    native = payload.get("hedge_native", payload)
    if not isinstance(native, dict):
        raise ValueError("result does not contain a Hedge native object")
    return native


def start_hedge_result_analysis(args: dict[str, Any]) -> int:
    """Rank one or more v4 Hedge result artifacts without ordinary Trade flattening."""
    from decimal import Decimal

    paths = tuple(args.get("hedge_result_files") or ())
    if not paths:
        raise ValueError("at least one --result is required")
    metric = str(args.get("hedge_result_metric") or "total_return")
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(str(raw)).expanduser().resolve()
        payload = _native_payload(_load_json_file(path))
        metrics = payload.get("metrics", {})
        if metric not in metrics:
            raise ValueError(f"metric {metric!r} not found in {path}")
        value = Decimal(str(metrics[metric]))
        rows.append({
            "file": str(path),
            "strategy": str(payload.get("strategy", "")),
            "pairs": list(payload.get("pairs", ())),
            "metric": metric,
            "value": str(value),
            "result_sha256": str(payload.get("result_sha256", "")),
            "metrics": metrics,
        })
    reverse = not bool(args.get("hedge_result_ascending", False))
    rows.sort(key=lambda item: Decimal(item["value"]), reverse=reverse)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    _print({"schema": "hedge-result-analysis-v1", "metric": metric, "results": rows}, args.get("hedge_native_output"))
    return 0


def start_hedge_lookahead_file_analysis(args: dict[str, Any]) -> int:
    from decimal import Decimal
    from freqtrade.hedge.native.analysis import HedgeLookaheadAnalyzer

    baseline = _native_payload(_load_json_file(args["hedge_baseline_result"]))
    candidates: dict[int, dict[str, Any]] = {}
    for item in args.get("hedge_candidate_results") or ():
        cutoff_text, separator, path_text = str(item).partition("=")
        if not separator:
            raise ValueError("candidate must use CUTOFF=PATH")
        cutoff = int(cutoff_text)
        candidates[cutoff] = _native_payload(_load_json_file(path_text))
    fields = tuple(args.get("hedge_analysis_fields") or ("events", "snapshots"))
    report = HedgeLookaheadAnalyzer(fields=fields).analyze(
        baseline,
        candidates,
        tolerance=Decimal(str(args.get("hedge_analysis_tolerance") or "0")),
    )
    _print(report.to_dict(), args.get("hedge_native_output"))
    return 0 if report.passed else 1


def start_hedge_recursive_file_analysis(args: dict[str, Any]) -> int:
    from decimal import Decimal
    from freqtrade.hedge.native.analysis import HedgeRecursiveAnalyzer

    outputs: dict[int, dict[str, Any]] = {}
    for item in args.get("hedge_recursive_results") or ():
        size_text, separator, path_text = str(item).partition("=")
        if not separator:
            raise ValueError("result must use STARTUP_CANDLES=PATH")
        outputs[int(size_text)] = _native_payload(_load_json_file(path_text))
    report = HedgeRecursiveAnalyzer().analyze(
        outputs,
        compare_tail=int(args.get("hedge_compare_tail") or 1),
        tolerance=Decimal(str(args.get("hedge_analysis_tolerance") or "0")),
    )
    _print(report.to_dict(), args.get("hedge_native_output"))
    return 0 if report.passed else 1


class _DeterministicTrial:
    """Small TrialLike implementation for reproducible native Hedge searches."""

    def __init__(self, seed: int) -> None:
        from random import Random
        self._random = Random(seed)

    def suggest_float(self, name: str, low: float, high: float, **kwargs: Any) -> float:
        from math import exp, log
        step = kwargs.get("step")
        if kwargs.get("log"):
            value = exp(self._random.uniform(log(low), log(high)))
        else:
            value = self._random.uniform(low, high)
        if step:
            value = low + round((value - low) / float(step)) * float(step)
        return min(max(value, low), high)

    def suggest_int(self, name: str, low: int, high: int, **kwargs: Any) -> int:
        step = int(kwargs.get("step", 1))
        count = (high - low) // step
        return low + self._random.randint(0, count) * step

    def suggest_categorical(self, name: str, choices: Any) -> Any:
        values = tuple(choices)
        return values[self._random.randrange(len(values))]


def start_hedge_native_hyperopt(args: dict[str, Any]) -> int:
    """Run reproducible Hedge-native search epochs through the shared backtester."""
    from freqtrade.commands.optimize_commands import setup_optimize_configuration
    from freqtrade.enums import RunMode
    from freqtrade.hedge.native.hyperopt import HedgeHyperoptRunner
    from freqtrade.optimize.hedge_backtesting import run_freqtrade_hedge_backtest

    epochs = int(args.get("hedge_epochs") or 10)
    if epochs < 1:
        raise ValueError("--hedge-epochs must be positive")
    seed = int(args.get("hedge_random_state") or 42)
    base_config = setup_optimize_configuration(args, RunMode.BACKTEST)
    raw_directory = args.get("hedge_hyperopt_directory")
    output_directory = Path(str(raw_directory or Path(base_config.get("user_data_dir", "user_data")) / "hyperopt_results" / "hedge-native")).expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    runner = HedgeHyperoptRunner()
    results = []
    for number in range(epochs):
        trial = _DeterministicTrial(seed + number)

        def backtest(trial_config: dict[str, Any], number: int = number):
            export = output_directory / f"epoch-{number:05d}.json"
            run = run_freqtrade_hedge_backtest(trial_config, export_path=export, export_events=False)
            if run.native_artifact is None:
                raise RuntimeError("Hedge backtest did not produce a native artifact")
            return run.native_artifact

        results.append(runner.evaluate(
            trial,
            number=number,
            base_config=base_config,
            backtest=backtest,
        ))
    manifest = runner.manifest(results)
    manifest["seed"] = seed
    manifest["epochs"] = epochs
    manifest["output_directory"] = str(output_directory)
    output = args.get("hedge_native_output") or output_directory / "hyperopt-manifest.json"
    _print(manifest, output)
    return 0
