#!/usr/bin/env python3
"""RTX 5070 V2.2 cold/warm compiler-cache and host-dispatch calibration.

Deep hardware gate:
- real cold compile runs with TORCHINDUCTOR_FORCE_DISABLE_CACHES=1;
- warm-cache runs use a dedicated per-profile TorchInductor/Triton cache and fresh processes;
- eager/compiled host inter-op candidates are randomized and repeated;
- break-even uses a corrected startup estimator that removes warmup update time;
- winner confidence combines median margin, CV and bootstrap superiority.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl.calibration import (  # noqa: E402
    choose_threads_with_confidence,
    compile_cache_environment,
    distribution_summary,
)
from freqtrade.hedge.hprl.performance import (  # noqa: E402
    estimate_compile_break_even_updates,
    estimate_compile_startup_seconds,
)

ALGORITHMS = ("fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2")
V21_PROFILE = {
    "fast_td3": {"off": 1, "reduce-overhead": 8},
    "fast_dsac": {"off": 16, "reduce-overhead": 1},
    "simba_sac": {"off": 4, "reduce-overhead": 32},
    "xqc": {"off": 1, "reduce-overhead": 1},
    "rebrac_v2": {"off": 16, "reduce-overhead": 16},
}
V20_BASELINE = {
    "fast_td3": {"off": 8, "reduce-overhead": 1},
    "fast_dsac": {"off": 16, "reduce-overhead": 1},
    "simba_sac": {"off": 4, "reduce-overhead": 32},
    "xqc": {"off": 32, "reduce-overhead": 1},
    "rebrac_v2": {"off": 16, "reduce-overhead": 1},
}


@dataclass(frozen=True, slots=True)
class Job:
    algorithm: str
    mode: str
    interop: int
    cache_state: str
    repeat: int


def _run_json(command: list[str], env: dict[str, str]) -> tuple[dict[str, Any] | None, str | None]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return None, detail[-16000:]
    try:
        return json.loads(completed.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}: {completed.stdout[-4000:]}"


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return out.splitlines()[0].strip() if out.strip() else "unknown"
    except Exception:
        return "unknown"


def _telemetry() -> dict[str, Any] | None:
    query = (
        "temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,"
        "memory.used,memory.total"
    )
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception:
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    keys = (
        "temperature_c",
        "power_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "utilization_pct",
        "memory_used_mib",
        "memory_total_mib",
    )
    values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
    out: dict[str, Any] = {}
    for key, value in zip(keys, values, strict=False):
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def _candidate_threads(algorithm: str) -> list[int]:
    values = {
        1,
        V20_BASELINE[algorithm]["off"],
        V20_BASELINE[algorithm]["reduce-overhead"],
        V21_PROFILE[algorithm]["off"],
        V21_PROFILE[algorithm]["reduce-overhead"],
    }
    return sorted(int(value) for value in values)


def _benchmark_command(args, algorithm: str, mode: str, interop: int, cache_state: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "freqtrade.hedge.hprl",
        "perf-benchmark",
        "--device",
        args.device,
        "--algorithm",
        algorithm,
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--hidden-depth",
        str(args.hidden_depth),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--optimizer-backend",
        "auto",
        "--polyak-backend",
        "auto",
        "--grad-clip-backend",
        "auto",
        "--compile-mode",
        mode,
        "--compile-cache-state",
        cache_state,
        "--cpu-threads",
        str(args.cpu_threads),
        "--cpu-interop-threads",
        str(interop),
        "--hardware-profile",
        "rtx5070_laptop",
        "--obs-dim",
        "32",
        "--action-dim",
        "4",
    ]
    if algorithm == "rebrac_v2":
        command += ["--flow-likelihood-precision", "fp32"]
    else:
        command.append("--mixed-precision")
    return command


def _cache_dir(cache_root: Path, algorithm: str, interop: int, state: str, repeat: int | None = None) -> Path:
    stem = f"{algorithm}-i{interop}-{state}"
    if repeat is not None and state == "cold":
        stem += f"-r{repeat}"
    return cache_root / stem


def _seed_warm_cache(args, algorithm: str, interop: int, cache_root: Path) -> dict[str, Any]:
    cache_dir = _cache_dir(cache_root, algorithm, interop, "warm")
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = compile_cache_environment(os.environ, cache_state="warm", cache_dir=cache_dir)
    command = _benchmark_command(args, algorithm, "reduce-overhead", interop, "warm")
    payload, error = _run_json(command, env)
    return {
        "algorithm": algorithm,
        "interop": interop,
        "cache_dir": str(cache_dir),
        "status": "PASS" if payload is not None and not error else "FAIL",
        "warmup_seconds": float(payload.get("warmup_seconds", 0.0)) if payload else 0.0,
        "compiled_hotpaths": payload.get("compiled_hotpaths", []) if payload else [],
        "error": error,
    }


def _run_job(args, job: Job, cache_root: Path) -> dict[str, Any]:
    before = _telemetry()
    if job.mode == "off":
        # Eager has no compiler cache. Use a neutral warm label so policy output remains explicit.
        env = os.environ.copy()
        cache_dir = None
    else:
        cache_dir = _cache_dir(cache_root, job.algorithm, job.interop, job.cache_state, job.repeat)
        if job.cache_state == "cold":
            shutil.rmtree(cache_dir, ignore_errors=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        env = compile_cache_environment(os.environ, cache_state=job.cache_state, cache_dir=cache_dir)
    payload, error = _run_json(
        _benchmark_command(args, job.algorithm, job.mode, job.interop, job.cache_state), env
    )
    after = _telemetry()
    result: dict[str, Any] = {
        "algorithm": job.algorithm,
        "compile_mode": job.mode,
        "cpu_interop_threads": job.interop,
        "cache_state": "not_applicable" if job.mode == "off" else job.cache_state,
        "repeat": job.repeat,
        "status": "PASS" if payload is not None and not error else "FAIL",
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "telemetry_before": before,
        "telemetry_after": after,
        "error": error,
    }
    if payload is not None:
        result.update(
            {
                "updates_per_second": float(payload["iterations_per_second"]),
                "warmup_iterations": int(payload["warmup_iterations"]),
                "warmup_seconds": float(payload["warmup_seconds"]),
                "seconds": float(payload["seconds"]),
                "compiled_hotpaths": payload.get("compiled_hotpaths", []),
            }
        )
    return result


def _aggregate(jobs: list[dict[str, Any]], algorithm: str, mode: str, interop: int, state: str) -> dict[str, Any]:
    selected = [
        row
        for row in jobs
        if row["algorithm"] == algorithm
        and row["compile_mode"] == mode
        and int(row["cpu_interop_threads"]) == int(interop)
        and (mode == "off" or row["cache_state"] == state)
    ]
    good = [row for row in selected if row["status"] == "PASS"]
    rates = [float(row["updates_per_second"]) for row in good]
    warmups = [float(row["warmup_seconds"]) for row in good]
    summary = distribution_summary(rates)
    warm_summary = distribution_summary(warmups)
    return {
        "algorithm": algorithm,
        "compile_mode": mode,
        "cpu_interop_threads": interop,
        "cache_state": "not_applicable" if mode == "off" else state,
        "status": "PASS" if len(good) == len(selected) and bool(selected) else "FAIL",
        "median_updates_per_second": float(summary["median"]),
        "median_warmup_seconds": float(warm_summary["median"]),
        "rate_distribution": summary,
        "warmup_distribution": warm_summary,
        "runs": selected,
    }


def _mode_profile(args, points: list[dict[str, Any]], algorithm: str, mode: str, state: str) -> dict[str, Any]:
    selected = [
        point
        for point in points
        if point["algorithm"] == algorithm
        and point["compile_mode"] == mode
        and (mode == "off" or point["cache_state"] == state)
    ]
    previous = V21_PROFILE[algorithm][mode]
    choice = choose_threads_with_confidence(
        selected,
        previous_threads=previous,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed + sum(ord(ch) for ch in algorithm + mode + state),
    )
    recommended = choice.get("recommended_threads")
    point = next(
        (p for p in selected if int(p["cpu_interop_threads"]) == int(recommended)),
        None,
    ) if recommended is not None else None
    return {"choice": choice, "recommended_point": point, "all_points": selected}


def _break_even(args, eager: dict[str, Any] | None, compiled: dict[str, Any] | None, state: str) -> dict[str, Any]:
    if eager is None or compiled is None:
        return {"status": "UNAVAILABLE", "threshold": None}
    compiled_rate = float(compiled["median_updates_per_second"])
    eager_rate = float(eager["median_updates_per_second"])
    warmup_iterations = int(compiled["runs"][0].get("warmup_iterations", args.warmup))
    startup = estimate_compile_startup_seconds(
        compiled_warmup_seconds=float(compiled["median_warmup_seconds"]),
        compiled_updates_per_second=compiled_rate,
        warmup_iterations=warmup_iterations,
    )
    quantum = args.cold_quantum if state == "cold" else args.warm_quantum
    threshold = estimate_compile_break_even_updates(
        eager_updates_per_second=eager_rate,
        compiled_updates_per_second=compiled_rate,
        eager_warmup_seconds=float(eager["median_warmup_seconds"]),
        compiled_warmup_seconds=float(compiled["median_warmup_seconds"]),
        warmup_iterations=warmup_iterations,
        safety_margin=args.break_even_margin,
        quantum=quantum,
    )
    gain = compiled_rate / eager_rate if eager_rate > 0.0 else 0.0
    return {
        "status": "PASS" if threshold is not None else "NO_COMPILE_ADVANTAGE",
        "threshold": threshold,
        "cache_state": state,
        "quantum": quantum,
        "safety_margin": args.break_even_margin,
        "estimated_compile_startup_seconds": startup,
        "steady_state_speedup": gain,
        "warmup_iterations": warmup_iterations,
        "eager_updates_per_second": eager_rate,
        "compiled_updates_per_second": compiled_rate,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--break-even-margin", type=float, default=1.25)
    parser.add_argument("--cold-quantum", type=int, default=500)
    parser.add_argument("--warm-quantum", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=220814)
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    parser.add_argument("--cache-root", default="/tmp/hprl-v22-compile-cache")
    parser.add_argument("--keep-cache", action="store_true")
    parser.add_argument("--allow-other-gpu", action="store_true")
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("--repeats must be >= 5 for V2.2 confidence calibration")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    requested = [value.strip() for value in args.algorithms.split(",") if value.strip()]
    unknown = [value for value in requested if value not in ALGORITHMS]
    if unknown:
        parser.error(f"unknown algorithms: {unknown}")
    gpu = _gpu_name()
    if not args.allow_other_gpu and "RTX 5070" not in gpu:
        parser.error(f"expected RTX 5070 hardware, got: {gpu}")

    cache_root = Path(args.cache_root)
    if cache_root.exists() and not args.keep_cache:
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    seeds: list[dict[str, Any]] = []
    for algorithm in requested:
        for interop in _candidate_threads(algorithm):
            seeds.append(_seed_warm_cache(args, algorithm, interop, cache_root))
    seed_failures = [row for row in seeds if row["status"] != "PASS"]
    if seed_failures:
        print(json.dumps({"schema": "hprl-performance-v2.2-rtx5070-calibration", "status": "FAIL", "warm_cache_seed": seeds}, indent=2))
        return 1

    jobs: list[Job] = []
    for algorithm in requested:
        for interop in _candidate_threads(algorithm):
            for repeat in range(args.repeats):
                jobs.append(Job(algorithm, "off", interop, "warm", repeat))
                jobs.append(Job(algorithm, "reduce-overhead", interop, "cold", repeat))
                jobs.append(Job(algorithm, "reduce-overhead", interop, "warm", repeat))
    random.Random(args.seed).shuffle(jobs)
    runs = [_run_job(args, job, cache_root) for job in jobs]

    points: list[dict[str, Any]] = []
    for algorithm in requested:
        for interop in _candidate_threads(algorithm):
            points.append(_aggregate(runs, algorithm, "off", interop, "warm"))
            points.append(_aggregate(runs, algorithm, "reduce-overhead", interop, "cold"))
            points.append(_aggregate(runs, algorithm, "reduce-overhead", interop, "warm"))

    results: dict[str, Any] = {}
    for algorithm in requested:
        eager_profile = _mode_profile(args, points, algorithm, "off", "warm")
        state_payload: dict[str, Any] = {}
        for state in ("cold", "warm"):
            compiled_profile = _mode_profile(args, points, algorithm, "reduce-overhead", state)
            compiled_point = compiled_profile["recommended_point"]
            eager_same = None
            if compiled_point is not None:
                interop = int(compiled_point["cpu_interop_threads"])
                eager_same = next(
                    (
                        point
                        for point in points
                        if point["algorithm"] == algorithm
                        and point["compile_mode"] == "off"
                        and int(point["cpu_interop_threads"]) == interop
                    ),
                    None,
                )
            state_payload[state] = {
                "compiled_profile": compiled_profile,
                "eager_at_compiled_interop": eager_same,
                "break_even": _break_even(args, eager_same, compiled_point, state),
            }
        results[algorithm] = {
            "schema": "hprl-rtx5070-algorithm-calibration-v22",
            "status": "PASS",
            "candidate_interop_threads": _candidate_threads(algorithm),
            "eager_profile": eager_profile,
            "compile_cache_states": state_payload,
        }

    failed_runs = [run for run in runs if run["status"] != "PASS"]
    passed = not failed_runs and all(seed["status"] == "PASS" for seed in seeds)
    recommended_profile = {}
    for algorithm, payload in results.items():
        eager_choice = payload["eager_profile"]["choice"]
        cold = payload["compile_cache_states"]["cold"]
        warm = payload["compile_cache_states"]["warm"]
        cold_threshold = cold["break_even"]["threshold"]
        warm_threshold = warm["break_even"]["threshold"]
        recommended_profile[algorithm] = {
            "eager_interop_threads": eager_choice["recommended_threads"],
            "eager_confidence": eager_choice["confidence"],
            "compiled_cold_interop_threads": cold["compiled_profile"]["choice"]["recommended_threads"],
            "compiled_cold_confidence": cold["compiled_profile"]["choice"]["confidence"],
            "compiled_warm_interop_threads": warm["compiled_profile"]["choice"]["recommended_threads"],
            "compiled_warm_confidence": warm["compiled_profile"]["choice"]["confidence"],
            "cold_compile_break_even_updates": cold_threshold,
            "warm_compile_break_even_updates": warm_threshold,
            "cold_compile_mode_auto": "reduce-overhead" if cold_threshold is not None else "off",
            "warm_compile_mode_auto": "reduce-overhead" if warm_threshold is not None else "off",
            "compile_mode_auto": "cache-state-dependent",
            "rebrac_precision": "fp32" if algorithm == "rebrac_v2" else "not_applicable",
        }

    report = {
        "schema": "hprl-performance-v2.2-rtx5070-calibration",
        "status": "PASS" if passed else "FAIL",
        "device_request": args.device,
        "gpu_name": gpu,
        "method": {
            "cold_cache": "TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 + unique cache dir per repeat",
            "warm_cache": "dedicated seeded TORCHINDUCTOR_CACHE_DIR reused across fresh processes",
            "remote_compiler_caches_disabled": True,
            "fresh_process_per_measurement": True,
            "randomized_measurement_order": True,
            "repeats": args.repeats,
            "aggregation": "median+p10+p90+MAD+CV",
            "winner_confidence": "median margin + CV + bootstrap superiority",
            "bootstrap_samples": args.bootstrap_samples,
            "corrected_startup_estimator": True,
            "break_even_margin": args.break_even_margin,
            "cold_quantum": args.cold_quantum,
            "warm_quantum": args.warm_quantum,
        },
        "warm_cache_seed": seeds,
        "results": results,
        "recommended_profile": recommended_profile,
        "failed_runs": failed_runs,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not args.keep_cache:
        shutil.rmtree(cache_root, ignore_errors=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
