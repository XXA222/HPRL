#!/usr/bin/env python3
"""Benchmark/autotune real HPRL gradient updates, replay sampling, and host dispatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl.action_space import configure_agent_action_levels  # noqa: E402
from freqtrade.hedge.hprl.config import HPRLActionConfig, HPRLTrainingConfig  # noqa: E402
from freqtrade.hedge.hprl.device import require_torch, resolve_device  # noqa: E402
from freqtrade.hedge.hprl.performance import (  # noqa: E402
    prepare_steady_state_agent,
    suggested_cpu_threads,
    timed_iterations_detailed,
)
from freqtrade.hedge.hprl.registry import create_agent  # noqa: E402
from freqtrade.hedge.hprl.replay import ReplayBatch, TensorReplayBuffer  # noqa: E402


torch = require_torch()
ALGORITHMS = ("fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2")


def csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def csv_strings(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def bench_agent(
    algorithm: str,
    *,
    device: str,
    batch_size: int,
    hidden_dim: int,
    hidden_depth: int,
    backend: str,
    polyak_backend: str,
    grad_clip_backend: str,
    compile_mode: str,
    cpu_threads: int,
    cpu_interop_threads: int,
    warmup: int,
    iterations: int,
    mixed_precision: bool,
    obs_dim: int,
    action_dim: int,
) -> dict[str, object]:
    config = HPRLTrainingConfig(
        algorithm=algorithm,
        device=device,
        replay_device="same",
        batch_size=batch_size,
        replay_capacity=max(batch_size * 2, 256),
        warmup_steps=0,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        mixed_precision=mixed_precision,
        optimizer_backend=backend,
        polyak_backend=polyak_backend,
        grad_clip_backend=grad_clip_backend,
        compile_mode=compile_mode,
        cpu_threads=cpu_threads,
        cpu_interop_threads=cpu_interop_threads,
        metrics_interval=10_000,
    )
    info = resolve_device(device)
    agent = create_agent(algorithm, obs_dim, action_dim, config, device=info.resolved)
    configure_agent_action_levels(agent, HPRLActionConfig().level_count)
    batch = ReplayBatch(
        obs=torch.randn(batch_size, obs_dim, device=info.resolved),
        action=torch.rand(batch_size, action_dim, device=info.resolved),
        reward=torch.randn(batch_size, 1, device=info.resolved) * 0.01,
        next_obs=torch.randn(batch_size, obs_dim, device=info.resolved),
        done=torch.zeros(batch_size, 1, device=info.resolved),
    )
    staged_warmup_updates = prepare_steady_state_agent(agent)
    timed = timed_iterations_detailed(
        lambda: agent.update(batch, collect_metrics=False),
        warmup=warmup,
        iterations=iterations,
        device=info.resolved,
        samples_per_iteration=batch_size,
    )
    return {
        "algorithm": algorithm,
        "device": info.resolved,
        "optimizer_backend": getattr(agent.actor_opt, "_hprl_backend", "unknown"),
        "polyak_backend": agent.performance_info.polyak_backend,
        "grad_clip_backend": agent.performance_info.grad_clip_backend,
        "compile_mode": compile_mode,
        "compiled_hotpaths": list(getattr(agent, "compiled_hotpaths", ())),
        "cpu_threads": int(torch.get_num_threads()),
        "cpu_interop_threads": int(torch.get_num_interop_threads()),
        "host_dispatch_tuned": agent.performance_info.host_dispatch_tuned,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "hidden_depth": hidden_depth,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "mixed_precision": mixed_precision,
        "staged_warmup_updates": staged_warmup_updates,
        "benchmark_stage": "steady_state",
        **timed,
    }


def bench_replay(
    *, capacity: int, batch_size: int, obs_dim: int, action_dim: int, iterations: int
) -> dict[str, object]:
    rb = TensorReplayBuffer(capacity, obs_dim, action_dim, device="cpu", pin_memory=False)
    chunk = min(capacity, 8192)
    for start in range(0, capacity, chunk):
        n = min(chunk, capacity - start)
        obs = torch.randn(n, obs_dim)
        rb.add(obs, torch.rand(n, action_dim), torch.randn(n), obs, torch.zeros(n))
    independent = timed_iterations_detailed(
        lambda: rb.sample(batch_size),
        warmup=10,
        iterations=iterations,
        device="cpu",
        samples_per_iteration=batch_size,
    )
    reusable = timed_iterations_detailed(
        lambda: rb.sample_reusable(batch_size),
        warmup=10,
        iterations=iterations,
        device="cpu",
        samples_per_iteration=batch_size,
    )
    return {
        "capacity": capacity,
        "batch_size": batch_size,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "persistent_bytes": rb.persistent_bytes,
        "sample_stage_bytes": rb.sample_stage_bytes,
        "independent": independent,
        "reusable": reusable,
        "reusable_speedup": reusable["iterations_per_second"]
        / max(independent["iterations_per_second"], 1e-12),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--algorithms", default=",".join(ALGORITHMS))
    p.add_argument("--batch-sizes", default="256,512,1024")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--hidden-depth", type=int, default=2)
    p.add_argument("--obs-dim", type=int, default=32)
    p.add_argument("--action-dim", type=int, default=4)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--cpu-threads", default="auto")
    p.add_argument("--cpu-interop-threads", default="1")
    p.add_argument("--optimizer-backends", default="auto")
    p.add_argument("--polyak-backends", default="auto")
    p.add_argument("--grad-clip-backends", default="auto")
    p.add_argument("--compile-modes", default="off")
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--replay-capacity", type=int, default=200_000)
    p.add_argument("--replay-batch", type=int, default=1024)
    p.add_argument("--replay-iterations", type=int, default=300)
    args = p.parse_args()

    info = resolve_device(args.device)
    algorithms = csv_strings(args.algorithms)
    batch_sizes = csv_ints(args.batch_sizes)
    backends = csv_strings(args.optimizer_backends)
    polyak_backends = csv_strings(args.polyak_backends)
    grad_clip_backends = csv_strings(args.grad_clip_backends)
    compile_modes = csv_strings(args.compile_modes)
    interop_threads = csv_ints(args.cpu_interop_threads)
    if args.cpu_threads == "auto":
        threads = suggested_cpu_threads() if info.resolved == "cpu" else (0,)
    else:
        threads = csv_ints(args.cpu_threads)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for algorithm in algorithms:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"unsupported algorithm {algorithm}")
        for batch_size in batch_sizes:
            for backend in backends:
                for polyak_backend in polyak_backends:
                    for grad_clip_backend in grad_clip_backends:
                        for compile_mode in compile_modes:
                            for cpu_threads in threads:
                                for cpu_interop_threads in interop_threads:
                                    try:
                                        rows.append(
                                            bench_agent(
                                                algorithm,
                                                device=info.resolved,
                                                batch_size=batch_size,
                                                hidden_dim=args.hidden_dim,
                                                hidden_depth=args.hidden_depth,
                                                backend=backend,
                                                polyak_backend=polyak_backend,
                                                grad_clip_backend=grad_clip_backend,
                                                compile_mode=compile_mode,
                                                cpu_threads=cpu_threads,
                                                cpu_interop_threads=cpu_interop_threads,
                                                warmup=args.warmup,
                                                iterations=args.iterations,
                                                mixed_precision=bool(args.mixed_precision),
                                                obs_dim=args.obs_dim,
                                                action_dim=args.action_dim,
                                            )
                                        )
                                    except Exception as exc:
                                        skipped.append(
                                            {
                                                "algorithm": algorithm,
                                                "batch_size": str(batch_size),
                                                "optimizer_backend": backend,
                                                "polyak_backend": polyak_backend,
                                                "grad_clip_backend": grad_clip_backend,
                                                "compile_mode": compile_mode,
                                                "cpu_threads": str(cpu_threads),
                                                "cpu_interop_threads": str(cpu_interop_threads),
                                                "error": f"{type(exc).__name__}: {exc}",
                                            }
                                        )
    best_updates: dict[str, dict[str, object]] = {}
    best_samples: dict[str, dict[str, object]] = {}
    for row in rows:
        algorithm = str(row["algorithm"])
        if algorithm not in best_updates or float(row["iterations_per_second"]) > float(
            best_updates[algorithm]["iterations_per_second"]
        ):
            best_updates[algorithm] = row
        if algorithm not in best_samples or float(row["samples_per_second"]) > float(
            best_samples[algorithm]["samples_per_second"]
        ):
            best_samples[algorithm] = row
    result = {
        "schema": "hprl-performance-autotune-v2",
        "device": info.resolved,
        "device_name": info.device_name,
        "torch": torch.__version__,
        "rows": rows,
        "best_updates_by_algorithm": best_updates,
        "best_samples_by_algorithm": best_samples,
        "replay_sampling": bench_replay(
            capacity=args.replay_capacity,
            batch_size=args.replay_batch,
            obs_dim=64,
            action_dim=8,
            iterations=args.replay_iterations,
        ),
        "skipped": skipped,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
