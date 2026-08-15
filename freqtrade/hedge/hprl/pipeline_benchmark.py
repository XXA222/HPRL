"""End-to-end HPRL training-pipeline benchmark utilities.

The steady-state measurement preserves replay/H2D overlap and avoids per-stage synchronization.
A separate serialized diagnostic pass attributes time to replay, transfer/update, logging and
checkpoint surfaces without contaminating the production-throughput number.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from .async_io import AsyncArtifactWriter, SynchronousArtifactWriter
from .checkpoint import capture_checkpoint_payload, save_checkpoint, write_checkpoint_payload
from .artifact_policy import ArtifactWorkload, estimate_checkpoint_bytes, resolve_artifact_io_mode
from .device import cuda_memory_stats, require_torch, synchronize_device, torch_device
from .replay import CudaReplayPrefetcher, TensorReplayBuffer
from .stage_profiling import StageRecorder


torch = require_torch()


@dataclass(frozen=True, slots=True)
class PipelineBenchmarkResult:
    schema: str
    iterations: int
    batch_size: int
    samples: int
    seconds: float
    updates_per_second: float
    samples_per_second: float
    replay_device: str
    training_device: str
    prefetch_enabled: bool
    prefetch_slots: int
    metrics_events: int
    log_bytes: int
    checkpoints: int
    checkpoint_bytes: int
    stage_diagnostics: dict[str, Any]
    cuda_memory: dict[str, int]
    async_artifacts: bool
    artifact_queue_size: int
    artifact_queue_high_water: int
    artifact_producer_block_seconds: float
    artifact_worker_seconds: float
    artifact_io_requested: str
    artifact_io_resolved: str
    artifact_io_policy: dict[str, Any]


def _fill_replay(buffer: TensorReplayBuffer, rows: int, *, chunk_rows: int = 8192) -> None:
    if rows < buffer.capacity:
        rows = buffer.capacity
    device = buffer.device
    remaining = min(int(rows), buffer.capacity)
    while remaining > 0:
        count = min(int(chunk_rows), remaining)
        obs = torch.randn((count, buffer.obs_dim), device=device)
        action = torch.rand((count, buffer.action_dim), device=device)
        reward = torch.randn((count, 1), device=device) * 0.01
        next_obs = torch.randn((count, buffer.obs_dim), device=device)
        done = torch.zeros((count, 1), device=device)
        buffer.add(obs, action, reward, next_obs, done)
        remaining -= count


def _make_replay(
    *,
    batch_size: int,
    obs_dim: int,
    action_dim: int,
    capacity: int,
    replay_device: str,
    training_device,
    pin_memory: bool,
    prefetch_slots: int,
):
    target = torch_device(training_device)
    requested = str(replay_device).strip().lower()
    if requested == "same":
        storage = target
    elif requested == "auto":
        storage = torch.device("cpu") if target.type == "cuda" else target
    else:
        storage = torch_device(requested)
    buffer = TensorReplayBuffer(
        capacity,
        obs_dim,
        action_dim,
        device=str(storage),
        pin_memory=bool(pin_memory),
        validate_inputs=False,
    )
    _fill_replay(buffer, capacity)
    prefetcher = None
    if storage.type == "cpu" and target.type == "cuda" and buffer.pin_memory:
        prefetcher = CudaReplayPrefetcher(
            buffer, batch_size, target, slots=int(prefetch_slots)
        )
    return buffer, prefetcher, storage


def _next_batch(buffer, prefetcher, batch_size: int, training_device):
    if prefetcher is not None:
        return prefetcher.next()
    batch = buffer.sample_reusable(batch_size)
    if torch.device(batch.obs.device) != torch_device(training_device):
        batch = batch.to(training_device, non_blocking=False)
    return batch


def _serialized_stage_diagnostics(
    agent,
    buffer,
    prefetcher,
    *,
    batch_size: int,
    iterations: int,
    metrics_interval: int,
    directory: Path,
) -> dict[str, Any]:
    count = max(1, int(iterations))
    device = torch_device(agent.device)
    totals = {"replay_h2d_seconds": 0.0, "update_target_seconds": 0.0,
              "logging_seconds": 0.0, "checkpoint_seconds": 0.0}
    log_path = directory / "diagnostic.jsonl"
    checkpoint_path = directory / "diagnostic.pt"
    metrics_events = 0
    with log_path.open("w", encoding="utf-8") as handle:
        for index in range(count):
            synchronize_device(device)
            started = time.perf_counter()
            batch = _next_batch(buffer, prefetcher, batch_size, device)
            synchronize_device(device)
            totals["replay_h2d_seconds"] += time.perf_counter() - started

            collect = index == 0 or (index + 1) % max(1, metrics_interval) == 0
            started = time.perf_counter()
            metrics = agent.update(batch, collect_metrics=collect)
            synchronize_device(device)
            totals["update_target_seconds"] += time.perf_counter() - started

            if metrics.values:
                started = time.perf_counter()
                handle.write(json.dumps(dict(metrics.values), sort_keys=True) + "\n")
                handle.flush()
                totals["logging_seconds"] += time.perf_counter() - started
                metrics_events += 1

        started = time.perf_counter()
        save_checkpoint(checkpoint_path, agent, {"benchmark": "pipeline-stage-diagnostic"})
        totals["checkpoint_seconds"] += time.perf_counter() - started
    checkpoint_bytes = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
    return {
        "iterations": count,
        **totals,
        "per_iteration_replay_h2d_ms": 1000.0 * totals["replay_h2d_seconds"] / count,
        "per_iteration_update_target_ms": 1000.0 * totals["update_target_seconds"] / count,
        "metrics_events": metrics_events,
        "checkpoint_bytes": checkpoint_bytes,
    }




def profile_xqc_pipeline_decomposition(
    agent,
    *,
    obs_dim: int,
    action_dim: int,
    batch_size: int,
    iterations: int = 3,
    replay_capacity: int = 16_384,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Serialize XQC stages so Replay/H2D/update/I/O can be attributed independently.

    This diagnostic deliberately disables overlap. Production throughput must use
    ``benchmark_training_pipeline`` instead.
    """
    if type(agent).__name__ != "XQCAgent" and not hasattr(agent, "profile_update_stages"):
        raise TypeError("XQC pipeline decomposition requires an XQC agent")
    if iterations < 1 or replay_capacity < batch_size:
        raise ValueError("invalid XQC decomposition dimensions")
    device = torch_device(agent.device)
    storage_device = "cpu" if device.type == "cuda" else str(device)
    buffer = TensorReplayBuffer(
        replay_capacity, obs_dim, action_dim, device=storage_device,
        pin_memory=device.type == "cuda", validate_inputs=False,
    )
    _fill_replay(buffer, replay_capacity)
    owns_tmp = work_dir is None
    temp = tempfile.TemporaryDirectory(prefix="hprl-xqc-decompose-") if owns_tmp else None
    root = Path(temp.name if temp is not None else work_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "xqc-diagnostic.jsonl"
    checkpoint_path = root / "xqc-diagnostic.pt"
    recorder = StageRecorder(device)
    try:
        with log_path.open("w", encoding="utf-8") as handle:
            for index in range(int(iterations)):
                with recorder.record("replay.sample", domain="host"):
                    host_batch = buffer.sample_reusable(batch_size)
                if device.type == "cuda":
                    with recorder.record("h2d.transfer", domain="cuda"):
                        batch = host_batch.to(device, non_blocking=bool(buffer.pin_memory))
                else:
                    batch = host_batch
                metrics = agent.profile_update_stages(batch, recorder, collect_metrics=True)
                with recorder.record("logging.json_write", domain="host"):
                    handle.write(json.dumps(dict(metrics.values), sort_keys=True) + "\n")
                    handle.flush()
                if index + 1 == iterations:
                    with recorder.record("checkpoint.snapshot", domain="host"):
                        state, metadata_text = capture_checkpoint_payload(
                            agent, {"xqc_decomposition_iteration": index + 1}, cpu_snapshot=True
                        )
                    with recorder.record("checkpoint.write", domain="host"):
                        write_checkpoint_payload(checkpoint_path, state, metadata_text)
        return {
            "schema": "hprl-xqc-pipeline-decomposition-v2",
            "iterations": int(iterations),
            "batch_size": int(batch_size),
            "device": str(device),
            "replay_device": str(buffer.device),
            "pin_memory": bool(buffer.pin_memory),
            "stage_attribution": recorder.summary(),
            "checkpoint_bytes": checkpoint_path.stat().st_size if checkpoint_path.exists() else 0,
            "log_bytes": log_path.stat().st_size if log_path.exists() else 0,
            "staging_identity": buffer.staging_identity(),
        }
    finally:
        buffer.release(aggressive=False)
        if temp is not None:
            temp.cleanup()


def benchmark_training_pipeline(
    agent,
    *,
    obs_dim: int,
    action_dim: int,
    batch_size: int,
    iterations: int,
    warmup: int = 20,
    replay_capacity: int = 16_384,
    replay_device: str = "auto",
    pin_memory: bool = True,
    prefetch_slots: int = 2,
    metrics_interval: int = 100,
    checkpoint_interval: int = 0,
    diagnostic_iterations: int = 3,
    async_artifacts: bool = True,
    artifact_io_mode: str | None = None,
    artifact_queue_size: int = 8,
    checkpoint_cpu_snapshot: bool = True,
    estimated_logger_bytes_per_event: int = 1024,
    prior_queue_block_ratio: float = 0.0,
    work_dir: str | Path | None = None,
) -> PipelineBenchmarkResult:
    if batch_size < 1 or iterations < 1 or warmup < 0:
        raise ValueError("pipeline benchmark dimensions are invalid")
    if replay_capacity < batch_size:
        raise ValueError("pipeline replay_capacity must be >= batch_size")
    if metrics_interval < 1 or checkpoint_interval < 0:
        raise ValueError("metrics/checkpoint intervals are invalid")
    if artifact_queue_size < 1:
        raise ValueError("artifact_queue_size must be positive")
    device = torch_device(agent.device)
    buffer, prefetcher, storage = _make_replay(
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
    temp = tempfile.TemporaryDirectory(prefix="hprl-pipeline-") if owns_tmp else None
    root = Path(temp.name if temp is not None else work_dir)
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "train.jsonl"
    checkpoints = 0
    checkpoint_bytes = 0
    metrics_events = 0
    requested_io_mode = str(artifact_io_mode or ("async" if async_artifacts else "sync")).strip().lower()
    writer = None
    io_decision = None
    artifact_queue_high_water = 0
    artifact_producer_block_seconds = 0.0
    artifact_worker_seconds = 0.0
    try:
        log_path.unlink(missing_ok=True)
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
        io_decision = resolve_artifact_io_mode(requested_io_mode, workload)
        writer = (AsyncArtifactWriter(queue_size=artifact_queue_size)
                  if io_decision.resolved == "async" else SynchronousArtifactWriter())
        started = time.perf_counter()
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
                    path, agent, {"pipeline_iteration": index + 1},
                    cpu_snapshot=checkpoint_cpu_snapshot,
                )
                checkpoints += 1
        synchronize_device(device)
        writer.close()
        artifact_stats = writer.stats()
        checkpoint_bytes = artifact_stats.checkpoint_bytes
        artifact_queue_high_water = artifact_stats.queue_high_water
        artifact_producer_block_seconds = artifact_stats.producer_block_seconds
        artifact_worker_seconds = artifact_stats.worker_seconds
        elapsed = max(time.perf_counter() - started, 1e-12)
        stage = _serialized_stage_diagnostics(
            agent,
            buffer,
            prefetcher,
            batch_size=batch_size,
            iterations=diagnostic_iterations,
            metrics_interval=metrics_interval,
            directory=root,
        )
        log_bytes = log_path.stat().st_size if log_path.exists() else 0
        samples = int(iterations) * int(batch_size)
        return PipelineBenchmarkResult(
            schema="hprl-training-pipeline-benchmark-v1",
            iterations=int(iterations),
            batch_size=int(batch_size),
            samples=samples,
            seconds=elapsed,
            updates_per_second=float(iterations) / elapsed,
            samples_per_second=float(samples) / elapsed,
            replay_device=str(storage),
            training_device=str(device),
            prefetch_enabled=prefetcher is not None,
            prefetch_slots=int(prefetch_slots if prefetcher is not None else 0),
            metrics_events=metrics_events,
            log_bytes=log_bytes,
            checkpoints=checkpoints,
            checkpoint_bytes=checkpoint_bytes,
            stage_diagnostics=stage,
            cuda_memory=cuda_memory_stats(device),
            async_artifacts=io_decision.resolved == "async",
            artifact_queue_size=int(artifact_queue_size if io_decision.resolved == "async" else 0),
            artifact_queue_high_water=int(artifact_queue_high_water),
            artifact_producer_block_seconds=float(artifact_producer_block_seconds),
            artifact_worker_seconds=float(artifact_worker_seconds),
            artifact_io_requested=io_decision.requested,
            artifact_io_resolved=io_decision.resolved,
            artifact_io_policy={
                "score": io_decision.score, "reason": io_decision.reason,
                "workload": dict(io_decision.workload),
            },
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
