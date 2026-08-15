#!/usr/bin/env python3
"""RTX 5070 V2.3 variance/orchestration/end-to-end pipeline hardware gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl.calibration import (  # noqa: E402
    balanced_interleaved_orders,
    paired_winner_confidence,
    robust_distribution_summary,
)

ALGORITHMS = ("fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2")
TARGET_ORCHESTRATION = ("fast_dsac", "simba_sac", "rebrac_v2")
V22_PROFILE = {
    "fast_td3": {"eager": 1, "compiled": 8, "alternative": 8},
    "fast_dsac": {"eager": 16, "compiled": 1, "alternative": 1},
    "simba_sac": {"eager": 4, "compiled": 32, "alternative": 32},
    "xqc": {"eager": 1, "compiled": 1, "alternative": 32},
    "rebrac_v2": {"eager": 16, "compiled": 16, "alternative": 1},
}


def _gpu_name() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        return completed.stdout.splitlines()[0].strip() if completed.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def _run_json(command: list[str], env: dict[str, str] | None = None) -> tuple[dict[str, Any] | None, str | None]:
    completed = subprocess.run(
        command, cwd=ROOT, env=env or os.environ.copy(), capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return None, detail[-16000:]
    try:
        return json.loads(completed.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}: {completed.stdout[-4000:]}"


def _precision_flags(algorithm: str) -> list[str]:
    if algorithm == "rebrac_v2":
        return ["--flow-likelihood-precision", "fp32"]
    return ["--mixed-precision"]


def _benchmark_command(
    args, algorithm: str, *, mode: str, scope: str, interop: int, condition_ms: int,
    iterations: int | None = None,
) -> list[str]:
    return [
        sys.executable, "-m", "freqtrade.hedge.hprl", "perf-benchmark",
        "--device", args.device,
        "--algorithm", algorithm,
        "--batch-size", str(args.batch_size),
        "--hidden-dim", str(args.hidden_dim),
        "--hidden-depth", str(args.hidden_depth),
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations if iterations is None else iterations),
        "--optimizer-backend", "auto",
        "--polyak-backend", "auto",
        "--grad-clip-backend", "auto",
        "--compile-mode", mode,
        "--compile-scope", scope,
        "--compile-cache-state", "warm",
        "--hardware-profile", "rtx5070_laptop",
        "--cpu-threads", str(args.cpu_threads),
        "--cpu-interop-threads", str(interop),
        "--obs-dim", "32", "--action-dim", "4",
        "--gpu-condition-ms", str(condition_ms),
        *_precision_flags(algorithm),
    ]


def _orchestration_command(args, algorithm: str, scope: str, interop: int) -> list[str]:
    return [
        sys.executable, "-m", "freqtrade.hedge.hprl", "perf-orchestration-profile",
        "--device", args.device, "--algorithm", algorithm,
        "--batch-size", str(args.batch_size), "--hidden-dim", str(args.hidden_dim),
        "--hidden-depth", str(args.hidden_depth), "--compile-mode", "reduce-overhead",
        "--compile-scope", scope, "--compile-cache-state", "warm",
        "--hardware-profile", "rtx5070_laptop", "--cpu-threads", str(args.cpu_threads),
        "--cpu-interop-threads", str(interop), "--active", str(args.profile_active),
        "--gpu-condition-ms", str(args.condition_ms), *_precision_flags(algorithm),
    ]


def _pipeline_command(args, algorithm: str, scope: str, interop: int) -> list[str]:
    return [
        sys.executable, "-m", "freqtrade.hedge.hprl", "perf-pipeline-benchmark",
        "--device", args.device, "--algorithm", algorithm,
        "--batch-size", str(args.batch_size), "--hidden-dim", str(args.hidden_dim),
        "--hidden-depth", str(args.hidden_depth), "--warmup", str(args.pipeline_warmup),
        "--iterations", str(args.pipeline_iterations), "--replay-capacity", str(args.replay_capacity),
        "--replay-device", "cpu", "--prefetch-slots", str(args.prefetch_slots),
        "--metrics-interval", str(args.metrics_interval),
        "--checkpoint-interval", str(args.checkpoint_interval),
        "--diagnostic-iterations", str(args.diagnostic_iterations),
        "--compile-mode", "reduce-overhead", "--compile-scope", scope,
        "--compile-cache-state", "warm", "--hardware-profile", "rtx5070_laptop",
        "--cpu-threads", str(args.cpu_threads), "--cpu-interop-threads", str(interop),
        "--gpu-condition-ms", str(args.condition_ms), *_precision_flags(algorithm),
    ]


def _aggregate_runs(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rates = [float(row["iterations_per_second"]) for row in rows if row.get("status", "PASS") == "PASS"]
    summary = robust_distribution_summary(rates)
    return {"summary": summary, "runs": rows}


def _variance_gate(args, algorithms: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    for algorithm in algorithms:
        candidates = sorted({V22_PROFILE[algorithm]["eager"], V22_PROFILE[algorithm]["alternative"]})
        if len(candidates) == 1:
            candidates.append(32 if candidates[0] != 32 else 1)
        conditions: dict[str, Any] = {}
        for condition_ms in (0, args.condition_ms):
            rows_by_candidate: dict[int, list[dict[str, Any]]] = {value: [] for value in candidates}
            paired_rounds: list[dict[str, Any]] = []
            orders = balanced_interleaved_orders(
                candidates, args.variance_repeats,
                seed=args.seed + sum(ord(ch) for ch in algorithm) + condition_ms,
            )
            for round_index, order in enumerate(orders):
                round_values: dict[int, float] = {}
                for interop in order:
                    payload, error = _run_json(_benchmark_command(
                        args, algorithm, mode="off", scope="module", interop=interop,
                        condition_ms=condition_ms,
                    ))
                    if payload is None or error:
                        errors.append({"algorithm": algorithm, "phase": "variance", "error": error or "unknown"})
                        continue
                    payload["round"] = round_index
                    rows_by_candidate[interop].append(payload)
                    round_values[interop] = float(payload["iterations_per_second"])
                paired_rounds.append({"round": round_index, "order": list(order), "rates": round_values})
            points = {
                str(interop): _aggregate_runs(rows)
                for interop, rows in rows_by_candidate.items()
            }
            ranked = sorted(
                candidates,
                key=lambda value: float(points[str(value)]["summary"].get("robust_median", 0.0)),
                reverse=True,
            )
            best, runner = ranked[0], ranked[1]
            best_pairs, runner_pairs = [], []
            for row in paired_rounds:
                rates = row["rates"]
                if best in rates and runner in rates:
                    best_pairs.append(rates[best]); runner_pairs.append(rates[runner])
            paired = paired_winner_confidence(
                best_pairs, runner_pairs, bootstrap_samples=args.bootstrap_samples,
                seed=args.seed + condition_ms + len(algorithm),
            ) if best_pairs else {"label": "none"}
            conditions[str(condition_ms)] = {
                "gpu_condition_ms": condition_ms,
                "balanced_interleaved": True,
                "points": points,
                "candidate_winner_threads": best,
                "paired_confidence": paired,
                "rounds": paired_rounds,
            }
        baseline_key, conditioned_key = "0", str(args.condition_ms)
        stable_thread = V22_PROFILE[algorithm]["eager"]
        raw_cv = float(conditions[baseline_key]["points"][str(stable_thread)]["summary"].get("robust_cv", 0.0))
        cond_cv = float(conditions[conditioned_key]["points"][str(stable_thread)]["summary"].get("robust_cv", 0.0))
        reduction = 100.0 * (1.0 - cond_cv / raw_cv) if raw_cv > 0 else 0.0
        result[algorithm] = {
            "stable_thread": stable_thread,
            "unconditioned_robust_cv": raw_cv,
            "conditioned_robust_cv": cond_cv,
            "conditioned_cv_reduction_pct": reduction,
            "recommended_gpu_conditioning": args.condition_ms if cond_cv <= raw_cv else 0,
            "conditions": conditions,
        }
    return {"schema": "hprl-v23-variance-gate-v1", "results": result, "errors": errors}


def _orchestration_gate(args) -> dict[str, Any]:
    results: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    for algorithm in TARGET_ORCHESTRATION:
        interop = int(V22_PROFILE[algorithm]["compiled"])
        scopes: dict[str, Any] = {}
        for scope in ("module", "loss"):
            micro = []
            for repeat in range(args.orchestration_repeats):
                payload, error = _run_json(_benchmark_command(
                    args, algorithm, mode="reduce-overhead", scope=scope, interop=interop,
                    condition_ms=args.condition_ms,
                ))
                if payload is None or error:
                    errors.append({"algorithm": algorithm, "phase": f"orchestration-{scope}", "error": error or "unknown"})
                    continue
                payload["repeat"] = repeat
                micro.append(payload)
            profile, profile_error = _run_json(_orchestration_command(args, algorithm, scope, interop))
            if profile_error:
                errors.append({"algorithm": algorithm, "phase": f"profile-{scope}", "error": profile_error})
            scopes[scope] = {
                "micro": _aggregate_runs(micro),
                "profile": profile,
            }
        stability, stability_error = _run_json(_benchmark_command(
            args, algorithm, mode="reduce-overhead", scope="loss", interop=interop,
            condition_ms=args.condition_ms, iterations=args.stability_updates,
        ))
        if stability_error:
            errors.append({"algorithm": algorithm, "phase": "loss-stability", "error": stability_error})
        module_rate = float(scopes["module"]["micro"]["summary"].get("robust_median", 0.0))
        loss_rate = float(scopes["loss"]["micro"]["summary"].get("robust_median", 0.0))
        module_launch = int(((scopes["module"].get("profile") or {}).get("categories") or {}).get("cuda_kernel_launches", 0))
        loss_launch = int(((scopes["loss"].get("profile") or {}).get("categories") or {}).get("cuda_kernel_launches", 0))
        throughput_speedup = loss_rate / module_rate if module_rate > 0 else 0.0
        launch_reduction = 100.0 * (1.0 - loss_launch / module_launch) if module_launch > 0 else 0.0
        stability_finite = bool(stability is not None and stability.get("parameters_finite", False))
        recommend_loss = bool(
            stability_finite
            and throughput_speedup >= 1.02
            and (module_launch == 0 or loss_launch <= module_launch)
        )
        results[algorithm] = {
            "interop": interop,
            "scopes": scopes,
            "loss_scope_stability": stability,
            "loss_scope_parameters_finite": stability_finite,
            "loss_scope_stability_updates": args.stability_updates,
            "loss_vs_module_speedup": throughput_speedup,
            "cuda_kernel_launch_reduction_pct": launch_reduction,
            "recommended_compile_scope": "loss" if recommend_loss else "module",
        }
    return {"schema": "hprl-v23-orchestration-gate-v1", "results": results, "errors": errors}


def _pipeline_gate(args, orchestration: dict[str, Any]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    errors: list[dict[str, str]] = []
    for algorithm in ALGORITHMS:
        if algorithm in TARGET_ORCHESTRATION:
            scope = orchestration["results"][algorithm]["recommended_compile_scope"]
        else:
            scope = "module"
        interop = int(V22_PROFILE[algorithm]["compiled"])
        pipeline, error = _run_json(_pipeline_command(args, algorithm, scope, interop))
        if pipeline is None or error:
            errors.append({"algorithm": algorithm, "phase": "pipeline", "error": error or "unknown"})
            results[algorithm] = {"status": "FAIL", "error": error}
            continue
        micro, micro_error = _run_json(_benchmark_command(
            args, algorithm, mode="reduce-overhead", scope=scope, interop=interop,
            condition_ms=args.condition_ms,
        ))
        if micro_error:
            errors.append({"algorithm": algorithm, "phase": "pipeline-micro-reference", "error": micro_error})
        micro_samples = float((micro or {}).get("samples_per_second", 0.0))
        pipe_samples = float(pipeline.get("samples_per_second", 0.0))
        results[algorithm] = {
            "status": "PASS",
            "compile_scope": scope,
            "interop": interop,
            "pipeline": pipeline,
            "micro_reference": micro,
            "pipeline_efficiency_vs_micro": pipe_samples / micro_samples if micro_samples > 0 else 0.0,
        }
    return {"schema": "hprl-v23-pipeline-gate-v1", "results": results, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--variance-repeats", type=int, default=7)
    parser.add_argument("--orchestration-repeats", type=int, default=3)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--condition-ms", type=int, default=350)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--profile-active", type=int, default=5)
    parser.add_argument("--stability-updates", type=int, default=2000)
    parser.add_argument("--pipeline-warmup", type=int, default=30)
    parser.add_argument("--pipeline-iterations", type=int, default=200)
    parser.add_argument("--replay-capacity", type=int, default=16384)
    parser.add_argument("--prefetch-slots", type=int, default=2)
    parser.add_argument("--metrics-interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--diagnostic-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=230814)
    parser.add_argument("--skip-variance", action="store_true")
    parser.add_argument("--skip-orchestration", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--allow-other-gpu", action="store_true")
    args = parser.parse_args()
    if args.variance_repeats < 5 or args.orchestration_repeats < 3:
        parser.error("V2.3 requires variance_repeats>=5 and orchestration_repeats>=3")
    gpu = _gpu_name()
    if not args.allow_other_gpu and "RTX 5070" not in gpu:
        parser.error(f"expected RTX 5070 hardware, got: {gpu}")

    variance = {"schema": "hprl-v23-variance-gate-v1", "status": "SKIP"}
    if not args.skip_variance:
        variance = _variance_gate(args, list(ALGORITHMS))
    orchestration = {"schema": "hprl-v23-orchestration-gate-v1", "status": "SKIP", "results": {}}
    if not args.skip_orchestration:
        orchestration = _orchestration_gate(args)
    pipeline = {"schema": "hprl-v23-pipeline-gate-v1", "status": "SKIP"}
    if not args.skip_pipeline:
        if not orchestration.get("results"):
            # Pipeline can still run using conservative module scope.
            orchestration = {"results": {a: {"recommended_compile_scope": "module"} for a in TARGET_ORCHESTRATION}, "errors": []}
        pipeline = _pipeline_gate(args, orchestration)

    errors = list(variance.get("errors", [])) + list(orchestration.get("errors", [])) + list(pipeline.get("errors", []))
    report = {
        "schema": "hprl-performance-v2.3-rtx5070-gate",
        "status": "PASS" if not errors else "FAIL",
        "gpu_name": gpu,
        "method": {
            "gpu_clock_conditioning": f"bounded GEMM preconditioning {args.condition_ms} ms outside timed interval",
            "robust_estimator": "MAD inliers + median/trimmed/winsorized locations",
            "paired_interleaving": "balanced per-round candidate order + paired bootstrap",
            "orchestration": "module-vs-loss compile scope + profiler launch counts + stability",
            "pipeline": "CPU replay -> pinned H2D prefetch -> update/target -> metrics log -> checkpoint",
        },
        "variance": variance,
        "orchestration": orchestration,
        "pipeline": pipeline,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
