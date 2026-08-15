"""Sustained end-to-end HPRL training benchmark with drift/memory/backpressure gates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import statistics
import subprocess
import tempfile
import time
from typing import Any

from .async_io import AsyncArtifactWriter, SynchronousArtifactWriter
from .device import cuda_memory_stats, require_torch, synchronize_device, torch_device
from .artifact_policy import ArtifactWorkload, estimate_checkpoint_bytes, resolve_artifact_io_mode
from .performance import agent_finite_state
from .pipeline_benchmark import _make_replay, _next_batch


torch = require_torch()


@dataclass(frozen=True, slots=True)
class SustainedPipelineResult:
    schema: str
    iterations: int
    batch_size: int
    samples: int
    training_loop_seconds: float
    end_to_end_seconds: float
    training_loop_samples_per_second: float
    end_to_end_samples_per_second: float
    window_size: int
    windows: tuple[dict[str, Any], ...]
    throughput_drift_ratio: float
    throughput_reference_samples_per_second: float
    throughput_robust_cv: float
    throughput_collapse_fraction: float
    throughput_longest_collapse_run: int
    throughput_edge_ratio: float
    throughput_tail_health_ratio: float
    throughput_recovery_observed: bool
    throughput_stability_reasons: tuple[str, ...]
    throughput_stable: bool
    memory_reserved_growth_bytes: int
    memory_allocated_growth_bytes: int
    memory_plateau: bool
    replay_staging_stable: bool
    parameters_finite: bool
    metrics_events: int
    checkpoints_submitted: int
    checkpoints_retained: int
    checkpoint_bytes_written: int
    checkpoint_deletions: int
    artifact_queue_high_water: int
    artifact_producer_block_seconds: float
    artifact_worker_seconds: float
    artifact_backpressure_ratio: float
    logger_backpressure_ok: bool
    artifact_io_requested: str
    artifact_io_resolved: str
    artifact_io_policy: dict[str, Any]
    cuda_memory: dict[str, int]


def _median_edge_ratio(values: list[float]) -> float:
    if not values:
        return 0.0
    count = max(1, len(values) // 5)
    first = statistics.median(values[:count])
    last = statistics.median(values[-count:])
    return float(last / first) if first > 0.0 else 0.0


def _longest_true_run(values: list[bool]) -> int:
    longest = current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _throughput_health(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "reference": 0.0, "robust_cv": float("inf"), "collapse_fraction": 1.0,
            "longest_collapse_run": 0, "tail_health_ratio": 0.0,
            "recovery_observed": False, "stable": False, "reasons": ("no_windows",),
        }
    ordered = sorted(float(v) for v in values if float(v) > 0.0)
    if not ordered:
        return {
            "reference": 0.0, "robust_cv": float("inf"), "collapse_fraction": 1.0,
            "longest_collapse_run": len(values), "tail_health_ratio": 0.0,
            "recovery_observed": False, "stable": False, "reasons": ("nonpositive_windows",),
        }
    upper_count = max(1, (len(ordered) + 1) // 2)
    reference = float(statistics.median(ordered[-upper_count:]))
    median = float(statistics.median(ordered))
    mad = float(statistics.median(abs(v - median) for v in ordered))
    robust_cv = (1.4826 * mad / median) if median > 0.0 else float("inf")
    collapse_threshold = reference * 0.50
    collapsed = [float(v) < collapse_threshold for v in values]
    collapse_fraction = sum(collapsed) / len(collapsed)
    longest_collapse = _longest_true_run(collapsed)
    edge_count = max(1, len(values) // 5)
    head_rate = float(statistics.median(float(v) for v in values[:edge_count]))
    tail_rate = float(statistics.median(float(v) for v in values[-edge_count:]))
    edge_ratio = tail_rate / head_rate if head_rate > 0.0 else 0.0
    tail_health = tail_rate / reference if reference > 0.0 else 0.0
    recovery = any(collapsed[:-edge_count] if len(collapsed) > edge_count else []) and tail_health >= 0.80
    reasons: list[str] = []
    if collapse_fraction > 0.10:
        reasons.append("collapse_fraction")
    if longest_collapse > 1:
        reasons.append("consecutive_collapse")
    if edge_ratio < 0.70:
        reasons.append("edge_degradation")
    if tail_health < 0.80:
        reasons.append("terminal_degradation")
    if robust_cv > 0.25:
        reasons.append("high_robust_cv")
    return {
        "reference": reference, "robust_cv": robust_cv,
        "collapse_fraction": collapse_fraction, "longest_collapse_run": longest_collapse,
        "edge_ratio": edge_ratio, "tail_health_ratio": tail_health, "recovery_observed": recovery,
        "stable": not reasons, "reasons": tuple(reasons),
    }


def _gpu_window_telemetry(device) -> dict[str, float] | None:
    if getattr(device, "type", "") != "cuda":
        return None
    query = "temperature.gpu,power.draw,clocks.sm,utilization.gpu,memory.used,memory.total"
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False, timeout=3,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return None
        row = completed.stdout.strip().splitlines()[0].split(",")
        values = [float(part.strip()) for part in row[:6]]
        return dict(zip(
            ("temperature_c", "power_w", "sm_clock_mhz", "utilization_pct",
             "memory_used_mib", "memory_total_mib"), values, strict=True,
        ))
    except Exception:
        return None


def benchmark_sustained_training(
    agent,
    *,
    obs_dim: int,
    action_dim: int,
    batch_size: int,
    iterations: int,
    warmup: int = 50,
    window_size: int = 500,
    replay_capacity: int = 16_384,
    replay_device: str = "auto",
    pin_memory: bool = True,
    prefetch_slots: int = 2,
    metrics_interval: int = 100,
    checkpoint_interval: int = 1000,
    checkpoint_keep_last: int = 2,
    artifact_queue_size: int = 8,
    artifact_io_mode: str = "auto",
    estimated_logger_bytes_per_event: int = 1024,
    prior_queue_block_ratio: float = 0.0,
    work_dir: str | Path | None = None,
) -> SustainedPipelineResult:
    if iterations < 1 or batch_size < 1 or warmup < 0 or window_size < 1:
        raise ValueError("invalid sustained benchmark dimensions")
    if replay_capacity < batch_size:
        raise ValueError("replay_capacity must be >= batch_size")
    if checkpoint_interval < 0 or checkpoint_keep_last < 0:
        raise ValueError("checkpoint controls must be non-negative")
    device = torch_device(agent.device)
    buffer, prefetcher, _ = _make_replay(
        batch_size=batch_size,
        obs_dim=obs_dim,
        action_dim=action_dim,
        capacity=replay_capacity,
        replay_device=replay_device,
        training_device=device,
        pin_memory=pin_memory,
        prefetch_slots=prefetch_slots,
    )
    owns_tmp = work_dir is None
    temp = tempfile.TemporaryDirectory(prefix="hprl-sustained-") if owns_tmp else None
    root = Path(temp.name if temp is not None else work_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "sustained.jsonl"
    checkpoints: list[Path] = []
    metrics_events = 0
    writer = None
    io_decision = None
    try:
        for _ in range(int(warmup)):
            batch = _next_batch(buffer, prefetcher, batch_size, device)
            agent.update(batch, collect_metrics=False)
        synchronize_device(device)
        workload = ArtifactWorkload(
            checkpoint_bytes=estimate_checkpoint_bytes(agent),
            checkpoint_interval=int(checkpoint_interval),
            logger_bytes_per_event=max(0, int(estimated_logger_bytes_per_event)),
            metrics_interval=max(1, int(metrics_interval)),
            expected_updates=int(iterations),
            queue_block_ratio=max(0.0, float(prior_queue_block_ratio)),
        )
        io_decision = resolve_artifact_io_mode(artifact_io_mode, workload)
        writer = (AsyncArtifactWriter(queue_size=artifact_queue_size)
                  if io_decision.resolved == "async" else SynchronousArtifactWriter())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        stage_before = (
            prefetcher.staging_identity() if prefetcher is not None else buffer.staging_identity()
        )
        windows: list[dict[str, Any]] = []
        training_started = time.perf_counter()
        window_started = training_started
        window_begin = 0
        for index in range(int(iterations)):
            batch = _next_batch(buffer, prefetcher, batch_size, device)
            collect = (index + 1) % int(metrics_interval) == 0
            metrics = agent.update(batch, collect_metrics=collect)
            if metrics.values:
                writer.submit_metrics(log_path, metrics.values)
                metrics_events += 1
            if checkpoint_interval and (index + 1) % int(checkpoint_interval) == 0:
                path = root / f"checkpoint-{index + 1}.pt"
                writer.submit_checkpoint(
                    path, agent, {"sustained_iteration": index + 1}, cpu_snapshot=True
                )
                checkpoints.append(path)
                if checkpoint_keep_last and len(checkpoints) > checkpoint_keep_last:
                    writer.submit_delete(checkpoints[-checkpoint_keep_last - 1])
            window_done = ((index + 1) % int(window_size) == 0) or index + 1 == iterations
            if window_done:
                synchronize_device(device)
                now = time.perf_counter()
                updates = (index + 1) - window_begin
                elapsed = max(now - window_started, 1e-12)
                memory = cuda_memory_stats(device)
                windows.append({
                    "start_update": window_begin + 1,
                    "end_update": index + 1,
                    "updates": updates,
                    "seconds": elapsed,
                    "updates_per_second": float(updates) / elapsed,
                    "samples_per_second": float(updates * batch_size) / elapsed,
                    "allocated_bytes": memory["allocated_bytes"],
                    "reserved_bytes": memory["reserved_bytes"],
                    "queue_depth": int(writer.stats().pending),
                    "gpu_telemetry": _gpu_window_telemetry(device),
                })
                window_started = time.perf_counter()
                window_begin = index + 1
        synchronize_device(device)
        training_loop_seconds = max(time.perf_counter() - training_started, 1e-12)
        writer.close()
        end_to_end_seconds = max(time.perf_counter() - training_started, 1e-12)
        stats = writer.stats()
        stage_after = (
            prefetcher.staging_identity() if prefetcher is not None else buffer.staging_identity()
        )
        window_rates = [float(row["samples_per_second"]) for row in windows]
        drift = _median_edge_ratio(window_rates)
        health = _throughput_health(window_rates)
        first_memory = windows[0] if windows else {"reserved_bytes": 0, "allocated_bytes": 0}
        last_memory = windows[-1] if windows else first_memory
        reserved_growth = int(last_memory["reserved_bytes"]) - int(first_memory["reserved_bytes"])
        allocated_growth = int(last_memory["allocated_bytes"]) - int(first_memory["allocated_bytes"])
        plateau_limit = max(64 * 1024 * 1024, int(first_memory["reserved_bytes"] * 0.10))
        memory_plateau = reserved_growth <= plateau_limit and allocated_growth <= plateau_limit
        finite = agent_finite_state(agent)
        retained = sum(1 for path in checkpoints if path.exists())
        backpressure_ratio = stats.producer_block_seconds / end_to_end_seconds
        return SustainedPipelineResult(
            schema="hprl-sustained-training-benchmark-v2",
            iterations=int(iterations),
            batch_size=int(batch_size),
            samples=int(iterations) * int(batch_size),
            training_loop_seconds=training_loop_seconds,
            end_to_end_seconds=end_to_end_seconds,
            training_loop_samples_per_second=float(iterations * batch_size) / training_loop_seconds,
            end_to_end_samples_per_second=float(iterations * batch_size) / end_to_end_seconds,
            window_size=int(window_size),
            windows=tuple(windows),
            throughput_drift_ratio=drift,
            throughput_reference_samples_per_second=float(health["reference"]),
            throughput_robust_cv=float(health["robust_cv"]),
            throughput_collapse_fraction=float(health["collapse_fraction"]),
            throughput_longest_collapse_run=int(health["longest_collapse_run"]),
            throughput_edge_ratio=float(health["edge_ratio"]),
            throughput_tail_health_ratio=float(health["tail_health_ratio"]),
            throughput_recovery_observed=bool(health["recovery_observed"]),
            throughput_stability_reasons=tuple(health["reasons"]),
            throughput_stable=bool(health["stable"]),
            memory_reserved_growth_bytes=reserved_growth,
            memory_allocated_growth_bytes=allocated_growth,
            memory_plateau=bool(memory_plateau),
            replay_staging_stable=bool(stage_before == stage_after),
            parameters_finite=bool(finite["parameters_finite"]),
            metrics_events=metrics_events,
            checkpoints_submitted=len(checkpoints),
            checkpoints_retained=retained,
            checkpoint_bytes_written=stats.checkpoint_bytes,
            checkpoint_deletions=stats.deletions,
            artifact_queue_high_water=stats.queue_high_water,
            artifact_producer_block_seconds=stats.producer_block_seconds,
            artifact_worker_seconds=stats.worker_seconds,
            artifact_backpressure_ratio=backpressure_ratio,
            logger_backpressure_ok=bool(backpressure_ratio <= 0.10),
            artifact_io_requested=io_decision.requested,
            artifact_io_resolved=io_decision.resolved,
            artifact_io_policy={
                "score": io_decision.score, "reason": io_decision.reason,
                "workload": dict(io_decision.workload),
            },
            cuda_memory=cuda_memory_stats(device),
        )
    finally:
        if writer is not None and not getattr(writer, "_closed", False):
            try:
                writer.close()
            except Exception:
                pass
        if prefetcher is not None:
            prefetcher.release()
        buffer.release(aggressive=False)
        if temp is not None:
            temp.cleanup()
