#!/usr/bin/env python3
"""RTX 5070 HPRL V2.0 hardware acceptance: CUDAGraph, host dispatch and ReBRAC 2x2x2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ALGORITHMS = ("fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2")
BASELINE_REBRAC_MIXED_COMPILED_SEPARATE = 101.20051622108993


def _run_json(command: list[str]) -> tuple[dict | None, str | None]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None, (completed.stderr or completed.stdout).strip()[-12000:]
    try:
        return json.loads(completed.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}: {completed.stdout[-3000:]}"


def _base_benchmark_cmd(args, algorithm: str) -> list[str]:
    return [
        sys.executable, "-m", "freqtrade.hedge.hprl", "perf-benchmark",
        "--device", args.device,
        "--algorithm", algorithm,
        "--batch-size", str(args.batch_size),
        "--hidden-dim", str(args.hidden_dim),
        "--hidden-depth", str(args.hidden_depth),
        "--warmup", str(args.warmup),
        "--iterations", str(args.iterations),
        "--optimizer-backend", "auto",
        "--polyak-backend", "auto",
        "--grad-clip-backend", "auto",
        "--cpu-threads", str(args.cpu_threads),
        "--cpu-interop-threads", str(args.default_interop),
        "--obs-dim", "32", "--action-dim", "4",
    ]


def _stability_child(args) -> dict[str, object]:
    from freqtrade.hedge.hprl.action_space import configure_agent_action_levels
    from freqtrade.hedge.hprl.config import HPRLActionConfig, HPRLTrainingConfig
    from freqtrade.hedge.hprl.device import require_torch, resolve_device
    from freqtrade.hedge.hprl.performance import prepare_steady_state_agent, synchronize
    from freqtrade.hedge.hprl.registry import create_agent
    from freqtrade.hedge.hprl.replay import ReplayBatch

    if args.algorithm not in {"fast_dsac", "simba_sac"}:
        raise ValueError("CUDAGraph stability child supports FastDSAC/Simba only")
    if not 1000 <= args.stability_updates <= 10000:
        raise ValueError("stability-updates must be within [1000, 10000]")
    torch = require_torch()
    info = resolve_device(args.device)
    if info.resolved == "cpu":
        raise RuntimeError("CUDAGraph stability gate requires CUDA")
    interop = 32 if args.algorithm == "simba_sac" else args.default_interop
    cfg = HPRLTrainingConfig(
        algorithm=args.algorithm,
        device=info.resolved,
        replay_device="same",
        batch_size=args.batch_size,
        replay_capacity=max(2 * args.batch_size, 4096),
        warmup_steps=0,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        mixed_precision=True,
        optimizer_backend="auto",
        polyak_backend="auto",
        grad_clip_backend="auto",
        compile_mode="reduce-overhead",
        expected_updates=args.stability_updates,
        hardware_profile="rtx5070_laptop",
        cpu_threads=args.cpu_threads,
        cpu_interop_threads=interop,
        metrics_interval=args.stability_updates + 100,
    )
    agent = create_agent(args.algorithm, 32, 4, cfg, device=info.resolved)
    configure_agent_action_levels(agent, HPRLActionConfig().level_count)
    boundaries = agent._tier_buffers.gaussian_boundaries
    boundary_ptr = int(boundaries.data_ptr())
    boundary_copy = boundaries.detach().clone()
    batch = ReplayBatch(
        obs=torch.randn(args.batch_size, 32, device=info.resolved),
        action=torch.rand(args.batch_size, 4, device=info.resolved),
        reward=torch.randn(args.batch_size, 1, device=info.resolved) * 0.01,
        next_obs=torch.randn(args.batch_size, 32, device=info.resolved),
        done=torch.zeros(args.batch_size, 1, device=info.resolved),
    )
    prepare_steady_state_agent(agent)
    torch.cuda.reset_peak_memory_stats(info.resolved)
    synchronize(info.resolved)
    started = time.perf_counter()
    for _ in range(args.stability_updates):
        agent.update(batch, collect_metrics=False)
    synchronize(info.resolved)
    elapsed = time.perf_counter() - started
    modules = [agent.actor, agent.critic, agent.critic_target]
    finite = all(
        bool(torch.isfinite(parameter).all())
        for module in modules
        for parameter in module.parameters()
    )
    stable_storage = boundary_ptr == int(agent._tier_buffers.gaussian_boundaries.data_ptr())
    stable_values = bool(torch.equal(boundary_copy, agent._tier_buffers.gaussian_boundaries))
    compiled = set(getattr(agent, "compiled_hotpaths", ()))
    tier_compiled = {"tier_entropy", "selected_tier_log_prob"}.issubset(compiled)
    memory = {
        "allocated_bytes": int(torch.cuda.memory_allocated(info.resolved)),
        "reserved_bytes": int(torch.cuda.memory_reserved(info.resolved)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(info.resolved)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(info.resolved)),
    }
    passed = finite and stable_storage and stable_values and tier_compiled
    return {
        "schema": "hprl-cudagraph-stability-v2",
        "status": "PASS" if passed else "FAIL",
        "algorithm": args.algorithm,
        "device": info.resolved,
        "device_name": info.device_name,
        "updates": args.stability_updates,
        "seconds": elapsed,
        "updates_per_second": args.stability_updates / max(elapsed, 1e-12),
        "compile_mode": agent.performance_info.compile_mode,
        "compiled_hotpaths": sorted(compiled),
        "tier_buffer_registered": "gaussian_boundaries" in dict(agent._tier_buffers.named_buffers()),
        "tier_buffer_storage_stable": stable_storage,
        "tier_buffer_values_stable": stable_values,
        "parameters_finite": finite,
        "cpu_interop_threads": int(torch.get_num_interop_threads()),
        "cuda_memory": memory,
    }


def _stability_parent(args) -> dict[str, object]:
    results = {}
    errors = {}
    for algorithm in ("fast_dsac", "simba_sac"):
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--child-stability", "--algorithm", algorithm,
            "--device", args.device,
            "--stability-updates", str(args.stability_updates),
            "--batch-size", str(args.batch_size),
            "--hidden-dim", str(args.hidden_dim),
            "--hidden-depth", str(args.hidden_depth),
            "--cpu-threads", str(args.cpu_threads),
            "--default-interop", str(args.default_interop),
        ]
        result, error = _run_json(command)
        if result is not None:
            results[algorithm] = result
        if error:
            errors[algorithm] = error
    passed = len(results) == 2 and not errors and all(v.get("status") == "PASS" for v in results.values())
    return {
        "schema": "hprl-cudagraph-stability-matrix-v2",
        "status": "PASS" if passed else "FAIL",
        "updates_per_algorithm": args.stability_updates,
        "results": results,
        "errors": errors,
    }


def _host_matrix(args) -> dict[str, object]:
    results, errors = {}, {}
    for algorithm in ALGORITHMS:
        command = [
            sys.executable, "-m", "freqtrade.hedge.hprl", "perf-host-sweep",
            "--device", args.device, "--algorithm", algorithm,
            "--batch-size", str(args.batch_size),
            "--hidden-dim", str(args.hidden_dim), "--hidden-depth", str(args.hidden_depth),
            "--warmup", str(max(10, args.warmup // 2)),
            "--iterations", str(max(50, args.iterations // 2)),
            "--optimizer-backend", "auto", "--polyak-backend", "auto",
            "--grad-clip-backend", "auto", "--compile-mode", "off",
            "--cpu-threads", str(args.cpu_threads),
            "--interop-candidates", args.interop_candidates,
            "--mixed-precision",
        ]
        result, error = _run_json(command)
        if result is not None:
            results[algorithm] = result
        if error:
            errors[algorithm] = error
    recommendation = {
        algorithm: (payload.get("best") or {}).get("cpu_interop_threads")
        for algorithm, payload in results.items()
    }
    passed = len(results) == len(ALGORITHMS) and not errors
    return {
        "schema": "hprl-rtx5070-host-dispatch-matrix-v2",
        "status": "PASS" if passed else "FAIL",
        "candidates": args.interop_candidates,
        "recommended_interop_threads": recommendation,
        "simba_v19_expected_default": 32,
        "results": results,
        "errors": errors,
    }


def _profile_candidate(args, *, mixed: bool, compile_mode: str) -> dict | None:
    command = [
        sys.executable, "-m", "freqtrade.hedge.hprl", "perf-profile",
        "--device", args.device, "--algorithm", "rebrac_v2",
        "--batch-size", str(args.batch_size), "--hidden-dim", str(args.hidden_dim),
        "--hidden-depth", str(args.hidden_depth), "--optimizer-backend", "auto",
        "--polyak-backend", "auto", "--grad-clip-backend", "auto",
        "--compile-mode", compile_mode,
        "--flow-likelihood-precision", "mixed" if mixed else "fp32",
        "--cpu-threads", str(args.cpu_threads), "--cpu-interop-threads", str(args.default_interop),
        "--active", "3", "--row-limit", "120",
    ]
    if mixed:
        command.append("--mixed-precision")
    result, _ = _run_json(command)
    if result is None:
        return None
    interesting = {}
    for row in result.get("top_ops", []):
        name = str(row.get("name", ""))
        if any(token in name for token in ("_to_copy", "copy_", "cudaLaunchKernel", "to")):
            interesting[name] = {
                "count": int(row.get("count", 0)),
                "self_cpu_time_us": float(row.get("self_cpu_time_us", 0.0)),
                "self_cuda_time_us": float(row.get("self_cuda_time_us", 0.0)),
            }
    return {"interesting_ops": interesting, "raw": result}


def _rebrac_cross(args) -> dict[str, object]:
    results, errors = [], []
    for precision in ("fp32", "mixed"):
        for compile_mode in ("off", "reduce-overhead"):
            for paired in (False, True):
                command = _base_benchmark_cmd(args, "rebrac_v2")
                command += [
                    "--compile-mode", compile_mode,
                    "--flow-likelihood-precision", precision,
                ]
                if precision == "mixed":
                    command.append("--mixed-precision")
                if paired:
                    command.append("--flow-obs-projection-reuse")
                result, error = _run_json(command)
                candidate = {
                    "precision": precision,
                    "compile_mode": compile_mode,
                    "paired": paired,
                }
                if result is not None:
                    results.append({**candidate, "result": result})
                if error:
                    errors.append({**candidate, "error": error})
    ordered = sorted(
        results,
        key=lambda item: float(item["result"]["iterations_per_second"]),
        reverse=True,
    )
    def find(precision: str, compile_mode: str, paired: bool):
        return next((x for x in ordered if x["precision"] == precision and x["compile_mode"] == compile_mode and x["paired"] == paired), None)
    fp32_compiled_separate = find("fp32", "reduce-overhead", False)
    mixed_compiled_separate = find("mixed", "reduce-overhead", False)
    fp32_rate = float(fp32_compiled_separate["result"]["iterations_per_second"]) if fp32_compiled_separate else 0.0
    mixed_rate = float(mixed_compiled_separate["result"]["iterations_per_second"]) if mixed_compiled_separate else 0.0
    best = ordered[0] if ordered else None
    recommendation = "retain_mixed_until_complete"
    if fp32_rate > 0 and mixed_rate > 0:
        if fp32_rate >= mixed_rate * 1.02:
            recommendation = "prefer_fp32_and_remove_mixed_default_candidate"
        elif mixed_rate >= fp32_rate * 1.02:
            recommendation = "prefer_mixed_and_keep_coarse_cast_boundary"
        else:
            recommendation = "near_tie_prefer_fp32_for_lower_cast_overhead"
    profiles = {
        "fp32_reduce_overhead_separate": _profile_candidate(args, mixed=False, compile_mode="reduce-overhead"),
        "mixed_reduce_overhead_separate": _profile_candidate(args, mixed=True, compile_mode="reduce-overhead"),
    }
    return {
        "schema": "hprl-rebrac-cross-sweep-v2",
        "status": "PASS" if len(ordered) == 8 and not errors else "FAIL",
        "baseline_mixed_reduce_overhead_separate_updates_per_second": BASELINE_REBRAC_MIXED_COMPILED_SEPARATE,
        "fp32_reduce_overhead_separate_updates_per_second": fp32_rate,
        "fp32_breaks_101_2": fp32_rate > BASELINE_REBRAC_MIXED_COMPILED_SEPARATE,
        "mixed_reduce_overhead_separate_updates_per_second": mixed_rate,
        "recommendation": recommendation,
        "best": best,
        "results": ordered,
        "profiles": profiles,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--algorithm", default="fast_dsac")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--default-interop", type=int, default=1)
    parser.add_argument("--interop-candidates", default="1,2,4,8,16,32")
    parser.add_argument("--stability-updates", type=int, default=1000)
    parser.add_argument("--child-stability", action="store_true")
    parser.add_argument("--skip-host", action="store_true")
    parser.add_argument("--skip-rebrac", action="store_true")
    parser.add_argument("--skip-stability", action="store_true")
    args = parser.parse_args()

    if args.child_stability:
        report = _stability_child(args)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "PASS" else 1
    if not 1000 <= args.stability_updates <= 10000:
        parser.error("--stability-updates must be within [1000, 10000]")

    report: dict[str, object] = {
        "schema": "hprl-performance-v2.0-rtx5070-acceptance",
        "device_request": args.device,
        "stability_updates": args.stability_updates,
    }
    if not args.skip_stability:
        report["cudagraph_stability"] = _stability_parent(args)
    if not args.skip_host:
        report["host_dispatch"] = _host_matrix(args)
    if not args.skip_rebrac:
        report["rebrac_cross"] = _rebrac_cross(args)
    statuses = [
        value.get("status")
        for value in report.values()
        if isinstance(value, dict) and "status" in value
    ]
    report["status"] = "PASS" if statuses and all(s == "PASS" for s in statuses) else "FAIL"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
