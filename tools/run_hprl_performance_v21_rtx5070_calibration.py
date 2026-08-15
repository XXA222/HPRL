#!/usr/bin/env python3
"""RTX 5070 V2.1 compile/host calibration after the V2.0 CUDAGraph fix.

The V2.0 acceptance proved correctness and supplied eager host winners, but host dispatch
is compile-mode specific. This calibration re-tests a reduced per-algorithm candidate set
in fresh processes, repeats each point, uses medians, and calculates compile break-even
from same-inter-op eager/compiled measurements.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl.performance import estimate_compile_break_even_updates  # noqa: E402

ALGORITHMS = ("fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2")
V20_EAGER_WINNERS = {
    "fast_td3": 8,
    "fast_dsac": 16,
    "simba_sac": 4,
    "xqc": 32,
    "rebrac_v2": 16,
}
V20_COMPILED_STABLE = {
    "fast_td3": 1,
    "fast_dsac": 1,
    "simba_sac": 32,
    "xqc": 1,
    "rebrac_v2": 1,
}
V20_REFERENCE_UPS = {
    "fast_dsac_10000_compiled": 107.18894455202923,
    "simba_sac_10000_compiled": 61.08480731164412,
    "rebrac_fp32_compiled_separate": 112.60429035753413,
}


def _run_json(command: list[str]) -> tuple[dict[str, Any] | None, str | None]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        return None, detail[-12000:]
    try:
        return json.loads(completed.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}: {completed.stdout[-3000:]}"


def _telemetry() -> dict[str, Any] | None:
    query = (
        "temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu," 
        "memory.used,memory.total"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None
    values = [value.strip() for value in completed.stdout.splitlines()[0].split(",")]
    keys = [
        "temperature_c",
        "power_w",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "utilization_pct",
        "memory_used_mib",
        "memory_total_mib",
    ]
    out: dict[str, Any] = {}
    for key, value in zip(keys, values, strict=False):
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def _benchmark_command(args, algorithm: str, mode: str, interop: int) -> list[str]:
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
        # V2.0 RTX 5070 evidence says FP32 is faster; do not enable AMP merely for flow.
    else:
        command.append("--mixed-precision")
    return command


def _point(args, algorithm: str, mode: str, interop: int) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    errors: list[str] = []
    for repeat in range(args.repeats):
        before = _telemetry()
        payload, error = _run_json(_benchmark_command(args, algorithm, mode, interop))
        after = _telemetry()
        if payload is not None:
            runs.append(
                {
                    "repeat": repeat,
                    "updates_per_second": float(payload["iterations_per_second"]),
                    "warmup_seconds": float(payload["warmup_seconds"]),
                    "seconds": float(payload["seconds"]),
                    "compiled_hotpaths": payload.get("compiled_hotpaths", []),
                    "telemetry_before": before,
                    "telemetry_after": after,
                }
            )
        if error:
            errors.append(error)
    rates = [row["updates_per_second"] for row in runs]
    warmups = [row["warmup_seconds"] for row in runs]
    return {
        "algorithm": algorithm,
        "compile_mode": mode,
        "cpu_interop_threads": interop,
        "status": "PASS" if len(runs) == args.repeats and not errors else "FAIL",
        "median_updates_per_second": statistics.median(rates) if rates else 0.0,
        "median_warmup_seconds": statistics.median(warmups) if warmups else 0.0,
        "runs": runs,
        "errors": errors,
    }


def _candidate_threads(algorithm: str) -> list[int]:
    values = {1, V20_EAGER_WINNERS[algorithm], V20_COMPILED_STABLE[algorithm]}
    return sorted(values)


def _algorithm_calibration(args, algorithm: str) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for interop in _candidate_threads(algorithm):
        for mode in ("off", "reduce-overhead"):
            points.append(_point(args, algorithm, mode, interop))
    good = [point for point in points if point["status"] == "PASS"]
    by_mode: dict[str, dict[str, Any] | None] = {}
    for mode in ("off", "reduce-overhead"):
        mode_points = [point for point in good if point["compile_mode"] == mode]
        by_mode[mode] = max(
            mode_points,
            key=lambda point: point["median_updates_per_second"],
            default=None,
        )
    compiled_best = by_mode["reduce-overhead"]
    eager_same: dict[str, Any] | None = None
    threshold: int | None = None
    if compiled_best is not None:
        interop = int(compiled_best["cpu_interop_threads"])
        eager_same = next(
            (
                point
                for point in good
                if point["compile_mode"] == "off"
                and int(point["cpu_interop_threads"]) == interop
            ),
            None,
        )
        if eager_same is not None:
            threshold = estimate_compile_break_even_updates(
                eager_updates_per_second=eager_same["median_updates_per_second"],
                compiled_updates_per_second=compiled_best["median_updates_per_second"],
                eager_warmup_seconds=eager_same["median_warmup_seconds"],
                compiled_warmup_seconds=compiled_best["median_warmup_seconds"],
                safety_margin=args.break_even_margin,
                quantum=args.break_even_quantum,
            )
    recommendation = {
        "eager_interop_threads": (
            int(by_mode["off"]["cpu_interop_threads"]) if by_mode["off"] else None
        ),
        "compiled_interop_threads": (
            int(compiled_best["cpu_interop_threads"]) if compiled_best else None
        ),
        "compile_mode_auto": "reduce-overhead" if threshold is not None else "off",
        "compile_break_even_updates": threshold,
    }
    passed = bool(by_mode["off"]) and bool(by_mode["reduce-overhead"]) and not [
        point for point in points if point["status"] != "PASS"
    ]
    return {
        "schema": "hprl-rtx5070-algorithm-calibration-v21",
        "status": "PASS" if passed else "FAIL",
        "algorithm": algorithm,
        "candidate_interop_threads": _candidate_threads(algorithm),
        "recommendation": recommendation,
        "best_eager": by_mode["off"],
        "best_compiled": compiled_best,
        "eager_at_compiled_interop": eager_same,
        "points": points,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--break-even-margin", type=float, default=1.25)
    parser.add_argument("--break-even-quantum", type=int, default=500)
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    requested = [value.strip() for value in args.algorithms.split(",") if value.strip()]
    unknown = [value for value in requested if value not in ALGORITHMS]
    if unknown:
        parser.error(f"unknown algorithms: {unknown}")
    results = {algorithm: _algorithm_calibration(args, algorithm) for algorithm in requested}
    passed = len(results) == len(requested) and all(
        payload.get("status") == "PASS" for payload in results.values()
    )
    report = {
        "schema": "hprl-performance-v2.1-rtx5070-calibration",
        "status": "PASS" if passed else "FAIL",
        "method": {
            "fresh_process_per_point": True,
            "repeats": args.repeats,
            "aggregation": "median",
            "same_interop_break_even": True,
            "break_even_margin": args.break_even_margin,
            "break_even_quantum": args.break_even_quantum,
            "telemetry": "nvidia-smi before/after when available",
        },
        "v20_evidence": {
            "eager_interop_winners": V20_EAGER_WINNERS,
            "compiled_stable_interop": V20_COMPILED_STABLE,
            "reference_updates_per_second": V20_REFERENCE_UPS,
            "rebrac_auto_precision": "fp32",
        },
        "results": results,
        "recommended_profile": {
            algorithm: payload["recommendation"] for algorithm, payload in results.items()
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
