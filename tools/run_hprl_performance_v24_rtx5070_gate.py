#!/usr/bin/env python3
"""RTX 5070 V2.4 production-scope, execution-region, XQC and sustained pipeline gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALGORITHMS = ("fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2")
LOSS_ALGORITHMS = ("fast_dsac", "simba_sac", "rebrac_v2")
INTEROP = {"fast_td3": 8, "fast_dsac": 1, "simba_sac": 32, "xqc": 1, "rebrac_v2": 16}
EXPECTED_AUTO_SCOPE = {
    "fast_td3": "module", "fast_dsac": "loss", "simba_sac": "loss",
    "xqc": "module", "rebrac_v2": "loss",
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


def _run_json(command: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    completed = subprocess.run(
        command, cwd=ROOT, env=os.environ.copy(), capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return None, detail[-20000:]
    try:
        return json.loads(completed.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}: {completed.stdout[-5000:]}"


def _precision_flags(algorithm: str) -> list[str]:
    if algorithm == "rebrac_v2":
        return ["--flow-likelihood-precision", "fp32"]
    return ["--mixed-precision"]


def _benchmark_command(args, algorithm: str, *, scope: str, iterations: int | None = None,
                       compile_mode: str = "reduce-overhead") -> list[str]:
    count = args.iterations if iterations is None else int(iterations)
    return [
        sys.executable, "-m", "freqtrade.hedge.hprl", "perf-benchmark",
        "--device", args.device, "--algorithm", algorithm,
        "--batch-size", str(args.batch_size), "--hidden-dim", str(args.hidden_dim),
        "--hidden-depth", str(args.hidden_depth), "--warmup", str(args.warmup),
        "--iterations", str(count), "--optimizer-backend", "auto",
        "--polyak-backend", "auto", "--grad-clip-backend", "auto",
        "--compile-mode", compile_mode, "--compile-scope", scope,
        "--expected-updates", str(max(count, 10000)), "--compile-cache-state", "warm",
        "--hardware-profile", "rtx5070_laptop", "--cpu-threads", str(args.cpu_threads),
        "--cpu-interop-threads", str(INTEROP[algorithm]), "--obs-dim", "32",
        "--action-dim", "4", "--gpu-condition-ms", "0", *_precision_flags(algorithm),
    ]


def _profile_command(args, algorithm: str, scope: str) -> list[str]:
    return [
        sys.executable, "-m", "freqtrade.hedge.hprl", "perf-orchestration-profile",
        "--device", args.device, "--algorithm", algorithm,
        "--batch-size", str(args.batch_size), "--hidden-dim", str(args.hidden_dim),
        "--hidden-depth", str(args.hidden_depth), "--compile-mode", "reduce-overhead",
        "--compile-scope", scope, "--compile-cache-state", "warm",
        "--hardware-profile", "rtx5070_laptop", "--cpu-threads", str(args.cpu_threads),
        "--cpu-interop-threads", str(INTEROP[algorithm]), "--active", str(args.profile_active),
        "--gpu-condition-ms", "0", *_precision_flags(algorithm),
    ]


def _pipeline_command(args, algorithm: str, *, sync: bool = False,
                      skip_reference: bool = False) -> list[str]:
    cmd = [
        sys.executable, "-m", "freqtrade.hedge.hprl", "perf-pipeline-benchmark",
        "--device", args.device, "--algorithm", algorithm,
        "--batch-size", str(args.batch_size), "--hidden-dim", str(args.hidden_dim),
        "--hidden-depth", str(args.hidden_depth), "--warmup", str(args.pipeline_warmup),
        "--iterations", str(args.pipeline_iterations), "--replay-capacity", str(args.replay_capacity),
        "--replay-device", "cpu", "--prefetch-slots", str(args.prefetch_slots),
        "--metrics-interval", str(args.metrics_interval), "--checkpoint-interval", str(args.checkpoint_interval),
        "--diagnostic-iterations", str(args.diagnostic_iterations), "--artifact-queue-size", str(args.artifact_queue_size),
        "--compile-mode", "reduce-overhead", "--compile-scope", "auto",
        "--compile-cache-state", "warm", "--hardware-profile", "rtx5070_laptop",
        "--cpu-threads", str(args.cpu_threads), "--cpu-interop-threads", str(INTEROP[algorithm]),
        "--gpu-condition-ms", "0", *_precision_flags(algorithm),
    ]
    if sync:
        cmd.append("--sync-artifacts")
    if skip_reference:
        cmd.append("--skip-micro-reference")
    return cmd


def _sustained_command(args, algorithm: str, updates: int) -> list[str]:
    return [
        sys.executable, "-m", "freqtrade.hedge.hprl", "perf-sustained-pipeline",
        "--device", args.device, "--algorithm", algorithm, "--batch-size", str(args.batch_size),
        "--hidden-dim", str(args.hidden_dim), "--hidden-depth", str(args.hidden_depth),
        "--warmup", str(args.sustained_warmup), "--iterations", str(updates),
        "--window-size", str(args.sustained_window), "--replay-capacity", str(args.replay_capacity),
        "--replay-device", "cpu", "--prefetch-slots", str(args.prefetch_slots),
        "--metrics-interval", str(args.metrics_interval), "--checkpoint-interval", str(args.sustained_checkpoint_interval),
        "--checkpoint-keep-last", str(args.checkpoint_keep_last), "--artifact-queue-size", str(args.artifact_queue_size),
        "--compile-mode", "auto", "--compile-scope", "auto", "--compile-cache-state", "warm",
        "--hardware-profile", "rtx5070_laptop", "--optimizer-backend", "auto",
        "--polyak-backend", "auto", "--grad-clip-backend", "auto",
        "--cpu-threads", str(args.cpu_threads), "--cpu-interop-threads", str(INTEROP[algorithm]),
        "--obs-dim", "32", "--action-dim", "4", *_precision_flags(algorithm),
    ]


def _median_rate(rows: list[dict[str, Any]]) -> float:
    values = [float(row.get("iterations_per_second", 0.0)) for row in rows]
    return statistics.median(values) if values else 0.0


def _production_scope_gate(args) -> dict[str, Any]:
    results, errors = {}, []
    for algorithm in ALGORITHMS:
        payload, error = _run_json(_benchmark_command(
            args, algorithm, scope="auto", iterations=max(args.iterations, 200), compile_mode="auto"
        ))
        if payload is None or error:
            errors.append({"algorithm": algorithm, "error": error or "unknown"})
            continue
        expected = EXPECTED_AUTO_SCOPE[algorithm]
        observed = str(payload.get("compile_scope"))
        results[algorithm] = {
            "expected_scope": expected, "observed_scope": observed,
            "compile_mode": payload.get("compile_mode"),
            "compiled_hotpaths": payload.get("compiled_hotpaths", []),
            "parameters_finite": payload.get("parameters_finite", False),
            "pass": observed == expected and bool(payload.get("parameters_finite", False)),
        }
        if not results[algorithm]["pass"]:
            errors.append({"algorithm": algorithm, "error": f"scope/finite mismatch: {results[algorithm]}"})
    return {"schema": "hprl-v24-production-scope-gate-v1", "results": results, "errors": errors}


def _execution_region_gate(args) -> dict[str, Any]:
    results, errors = {}, []
    for algorithm in LOSS_ALGORITHMS:
        scopes: dict[str, Any] = {}
        for scope in ("loss", "loss_post"):
            rows = []
            local_errors = []
            for _ in range(args.execution_repeats):
                payload, error = _run_json(_benchmark_command(args, algorithm, scope=scope))
                if payload is None or error:
                    local_errors.append(error or "unknown")
                else:
                    rows.append(payload)
            profile, profile_error = _run_json(_profile_command(args, algorithm, scope))
            if profile_error:
                local_errors.append(profile_error)
            scopes[scope] = {"runs": rows, "median_updates_per_second": _median_rate(rows),
                             "profile": profile, "errors": local_errors}
        stability, stability_error = _run_json(_benchmark_command(
            args, algorithm, scope="loss_post", iterations=args.post_stability_updates
        ))
        loss_rate = float(scopes["loss"]["median_updates_per_second"])
        post_rate = float(scopes["loss_post"]["median_updates_per_second"])
        loss_launch = int((((scopes["loss"].get("profile") or {}).get("categories") or {}).get("cuda_kernel_launches", 0)))
        post_launch = int((((scopes["loss_post"].get("profile") or {}).get("categories") or {}).get("cuda_kernel_launches", 0)))
        post_hotpaths = list((stability or {}).get("compiled_hotpaths", []))
        post_compiled = "update.post_update_surface" in post_hotpaths
        finite = bool(stability is not None and stability.get("parameters_finite", False))
        speedup = post_rate / loss_rate if loss_rate > 0 else 0.0
        loss_profile_ok = scopes["loss"].get("profile") is not None and not scopes["loss"].get("errors")
        post_profile_ok = scopes["loss_post"].get("profile") is not None and not scopes["loss_post"].get("errors")
        launch_profile_valid = bool(loss_profile_ok and post_profile_ok and loss_launch > 0 and post_launch > 0)
        launch_reduction = (
            100.0 * (1.0 - post_launch / loss_launch) if launch_profile_valid else None
        )
        recommend_post = bool(
            post_compiled and finite and speedup >= 1.01 and launch_profile_valid
            and post_launch <= loss_launch
        )
        results[algorithm] = {
            "scopes": scopes, "loss_post_stability": stability,
            "loss_post_stability_error": stability_error,
            "post_surface_compiled": post_compiled, "parameters_finite": finite,
            "loss_post_vs_loss_speedup": speedup,
            "launch_profile_valid": launch_profile_valid,
            "cuda_kernel_launch_reduction_pct": launch_reduction,
            "recommended_scope": "loss_post" if recommend_post else "loss",
        }
        # Experimental loss_post failure does not invalidate the already accepted production loss path.
    return {"schema": "hprl-v24-execution-region-gate-v1", "results": results, "errors": errors}


def _xqc_decomposition_gate(args) -> dict[str, Any]:
    command = [
        sys.executable, "-m", "freqtrade.hedge.hprl", "perf-xqc-decomposition",
        "--device", args.device, "--batch-size", str(args.batch_size),
        "--hidden-dim", str(args.hidden_dim), "--hidden-depth", str(args.hidden_depth),
        "--iterations", str(args.xqc_diagnostic_iterations), "--compile-mode", "reduce-overhead",
        "--compile-scope", "module", "--compile-cache-state", "warm",
        "--hardware-profile", "rtx5070_laptop", "--cpu-threads", str(args.cpu_threads),
        "--cpu-interop-threads", str(INTEROP["xqc"]), "--mixed-precision",
    ]
    payload, error = _run_json(command)
    required = {
        "replay.sample", "h2d.transfer", "forward_backward.critic_forward",
        "forward_backward.critic_backward_clip", "optimizer.critic", "target.polyak",
        "metrics.materialize", "logging.json_write", "checkpoint.snapshot", "checkpoint.write",
    }
    stages = set((((payload or {}).get("stage_attribution") or {}).get("stages") or {}))
    ok = payload is not None and bool(payload.get("parameters_finite", False)) and required.issubset(stages)
    return {"schema": "hprl-v24-xqc-decomposition-gate-v1", "status": "PASS" if ok else "FAIL",
            "required_stages": sorted(required), "observed_stages": sorted(stages),
            "result": payload, "error": error}


def _async_pipeline_gate(args) -> dict[str, Any]:
    results, errors = {}, []
    for algorithm in ALGORITHMS:
        async_payload, async_error = _run_json(_pipeline_command(args, algorithm, sync=False))
        sync_payload, sync_error = _run_json(_pipeline_command(args, algorithm, sync=True, skip_reference=True))
        if async_payload is None or async_error or sync_payload is None or sync_error:
            errors.append({"algorithm": algorithm, "async_error": async_error, "sync_error": sync_error})
            continue
        async_rate = float(async_payload.get("samples_per_second", 0.0))
        sync_rate = float(sync_payload.get("samples_per_second", 0.0))
        results[algorithm] = {
            "async": async_payload, "sync": sync_payload,
            "async_vs_sync_speedup": async_rate / sync_rate if sync_rate > 0 else 0.0,
            "pipeline_efficiency_same_scope": async_payload.get("pipeline_efficiency_same_scope"),
            "end_to_end_speedup_vs_module_baseline": async_payload.get("end_to_end_speedup_vs_module_baseline"),
            "denominator_semantics_pass": (
                async_payload.get("pipeline_efficiency_same_scope") is not None
                and async_payload.get("end_to_end_speedup_vs_module_baseline") is not None
            ),
            "parameters_finite": async_payload.get("parameters_finite", False),
        }
        if not results[algorithm]["denominator_semantics_pass"] or not results[algorithm]["parameters_finite"]:
            errors.append({"algorithm": algorithm, "error": "pipeline denominator/finite gate failed"})
    return {"schema": "hprl-v24-async-pipeline-gate-v1", "results": results, "errors": errors}


def _sustained_checks(payload: dict[str, Any], checkpoint_keep_last: int) -> dict[str, bool]:
    return {
        "parameters_finite": bool(payload.get("parameters_finite", False)),
        "replay_staging_stable": bool(payload.get("replay_staging_stable", False)),
        "memory_plateau": bool(payload.get("memory_plateau", False)),
        "throughput_stable": bool(payload.get("throughput_stable", False)),
        "logger_backpressure_ok": bool(payload.get("logger_backpressure_ok", False)),
        "checkpoint_retention_bounded": int(payload.get("checkpoints_retained", 0)) <= checkpoint_keep_last,
    }


def _sustained_gate(args, updates: int, *, retry_unstable: bool = False) -> dict[str, Any]:
    results, errors = {}, []
    for algorithm in ALGORITHMS:
        attempts: list[dict[str, Any]] = []
        payload, error = _run_json(_sustained_command(args, algorithm, updates))
        if payload is None or error:
            errors.append({"algorithm": algorithm, "error": error or "unknown"})
            continue
        checks = _sustained_checks(payload, args.checkpoint_keep_last)
        attempts.append({"attempt": 1, "checks": checks, "result": payload, "pass": all(checks.values())})
        safety_ok = all(value for key, value in checks.items() if key != "throughput_stable")
        if retry_unstable and safety_ok and not checks["throughput_stable"]:
            time.sleep(max(0.0, float(args.sustained_retry_cooldown_seconds)))
            retry_payload, retry_error = _run_json(_sustained_command(args, algorithm, updates))
            if retry_payload is not None and not retry_error:
                retry_checks = _sustained_checks(retry_payload, args.checkpoint_keep_last)
                attempts.append({
                    "attempt": 2, "checks": retry_checks, "result": retry_payload,
                    "pass": all(retry_checks.values()),
                })
                payload, checks = retry_payload, retry_checks
            else:
                attempts.append({"attempt": 2, "checks": {}, "result": None, "pass": False,
                                 "error": retry_error or "unknown retry failure"})
        passed = all(checks.values())
        results[algorithm] = {
            "checks": checks, "result": payload, "attempts": attempts,
            "retry_used": len(attempts) > 1, "pass": passed,
        }
        if not passed:
            errors.append({"algorithm": algorithm, "error": f"sustained checks failed after {len(attempts)} attempt(s): {checks}"})
    return {"schema": "hprl-v24-sustained-gate-v2", "updates": updates, "results": results, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--execution-repeats", type=int, default=3)
    parser.add_argument("--profile-active", type=int, default=5)
    parser.add_argument("--post-stability-updates", type=int, default=2000)
    parser.add_argument("--pipeline-warmup", type=int, default=30)
    parser.add_argument("--pipeline-iterations", type=int, default=200)
    parser.add_argument("--replay-capacity", type=int, default=16384)
    parser.add_argument("--prefetch-slots", type=int, default=2)
    parser.add_argument("--metrics-interval", type=int, default=50)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--diagnostic-iterations", type=int, default=3)
    parser.add_argument("--xqc-diagnostic-iterations", type=int, default=9)
    parser.add_argument("--artifact-queue-size", type=int, default=8)
    parser.add_argument("--sustained-updates", type=int, default=5000)
    parser.add_argument("--sustained-window", type=int, default=500)
    parser.add_argument("--sustained-warmup", type=int, default=50)
    parser.add_argument("--sustained-checkpoint-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-keep-last", type=int, default=2)
    parser.add_argument("--deep-sustained", action="store_true")
    parser.add_argument("--deep-sustained-updates", type=int, default=10000)
    parser.add_argument("--sustained-retry-cooldown-seconds", type=float, default=3.0)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--skip-sustained", action="store_true")
    args = parser.parse_args()
    if args.execution_repeats < 3 or args.sustained_updates < 1000:
        parser.error("V2.4 requires execution_repeats>=3 and sustained_updates>=1000")
    gpu = _gpu_name()
    if not args.allow_other_gpu and "RTX 5070" not in gpu:
        parser.error(f"expected RTX 5070 hardware, got: {gpu}")

    production = _production_scope_gate(args)
    execution = _execution_region_gate(args)
    xqc = _xqc_decomposition_gate(args)
    pipeline = _async_pipeline_gate(args)
    sustained = {"schema": "hprl-v24-sustained-gate-v1", "status": "SKIP", "errors": []}
    deep = {"schema": "hprl-v24-sustained-gate-v1", "status": "SKIP", "errors": []}
    if not args.skip_sustained:
        # 5k is a qualification/diagnostic pass when the authoritative 10k gate is enabled.
        sustained = _sustained_gate(args, args.sustained_updates, retry_unstable=not args.deep_sustained)
        if args.deep_sustained:
            deep = _sustained_gate(args, args.deep_sustained_updates, retry_unstable=True)
    errors = list(production.get("errors", [])) + list(pipeline.get("errors", []))
    if xqc.get("status") != "PASS":
        errors.append({"phase": "xqc_decomposition", "error": xqc.get("error") or "stage gate failed"})
    # With deep sustained enabled, 5k remains diagnostic evidence; 10k is authoritative.
    if not args.deep_sustained:
        errors += list(sustained.get("errors", []))
    errors += list(deep.get("errors", []))
    report = {
        "schema": "hprl-performance-v2.4.1-rtx5070-gate",
        "status": "PASS" if not errors else "FAIL",
        "gpu_name": gpu,
        "production_scope": production,
        "execution_region": execution,
        "xqc_decomposition": xqc,
        "async_pipeline": pipeline,
        "sustained_5k": sustained,
        "sustained_5k_role": "qualification" if args.deep_sustained else "authoritative",
        "sustained_10k": deep,
        "sustained_10k_role": "authoritative" if args.deep_sustained else "skipped",
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
