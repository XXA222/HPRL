"""Micro-benchmark helpers for HPRL vector environments and accelerator inference."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .device import cuda_memory_stats, require_torch, synchronize_device


torch = require_torch()


@dataclass(frozen=True, slots=True)
class ThroughputResult:
    steps: int
    environments: int
    transitions: int
    seconds: float
    transitions_per_second: float
    device: str = "cpu"
    max_allocated_bytes: int = 0
    max_reserved_bytes: int = 0


def benchmark_environment(env, *, steps: int = 100) -> ThroughputResult:
    if steps < 1:
        raise ValueError("benchmark steps must be positive")
    obs, _ = env.reset()
    action = torch.zeros((env.envs, env.action_dim), device=env.device)
    if torch.device(env.device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(env.device)
    synchronize_device(env.device)
    started = time.perf_counter()
    executed = 0
    for _ in range(steps):
        result = env.step(action)
        obs = result.observation
        executed += 1
        if bool(result.info.get("time_done", False)):
            obs, _ = env.reset()
        _ = obs
    synchronize_device(env.device)
    elapsed = max(time.perf_counter() - started, 1e-9)
    transitions = executed * env.envs
    memory = cuda_memory_stats(env.device)
    return ThroughputResult(
        steps=executed,
        environments=env.envs,
        transitions=transitions,
        seconds=elapsed,
        transitions_per_second=transitions / elapsed,
        device=str(env.device),
        max_allocated_bytes=memory["max_allocated_bytes"],
        max_reserved_bytes=memory["max_reserved_bytes"],
    )
