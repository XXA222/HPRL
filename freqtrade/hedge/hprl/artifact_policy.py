"""Workload-aware artifact I/O policy for HPRL logging and checkpoints."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

from .device import require_torch

torch = require_torch()


@dataclass(frozen=True, slots=True)
class ArtifactWorkload:
    checkpoint_bytes: int
    checkpoint_interval: int
    logger_bytes_per_event: int
    metrics_interval: int
    expected_updates: int
    queue_block_ratio: float = 0.0

    @property
    def checkpoint_bytes_per_update(self) -> float:
        if self.checkpoint_interval <= 0:
            return 0.0
        return float(self.checkpoint_bytes) / float(self.checkpoint_interval)

    @property
    def logger_bytes_per_update(self) -> float:
        return float(self.logger_bytes_per_event) / float(max(1, self.metrics_interval))

    @property
    def io_bytes_per_update(self) -> float:
        return self.checkpoint_bytes_per_update + self.logger_bytes_per_update

    @property
    def projected_io_bytes(self) -> float:
        return self.io_bytes_per_update * float(max(0, self.expected_updates))


@dataclass(frozen=True, slots=True)
class ArtifactIODecision:
    requested: str
    resolved: str
    score: float
    reason: str
    workload: Mapping[str, object]


def _tree_tensor_bytes(value) -> int:
    if torch.is_tensor(value):
        return int(value.numel()) * int(value.element_size())
    if isinstance(value, Mapping):
        return sum(_tree_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tree_tensor_bytes(item) for item in value)
    return 0


def estimate_checkpoint_bytes(agent) -> int:
    """Estimate serialized tensor payload without cloning live tensors or touching disk."""
    total = 0
    for name in ("actor", "critic", "actor_target", "critic_target"):
        module = getattr(agent, name, None)
        if module is not None and hasattr(module, "state_dict"):
            total += _tree_tensor_bytes(module.state_dict())
    for name in ("actor_opt", "critic_opt", "alpha_opt"):
        optimizer = getattr(agent, name, None)
        if optimizer is not None and hasattr(optimizer, "state_dict"):
            total += _tree_tensor_bytes(optimizer.state_dict())
    for name in ("log_alpha",):
        tensor = getattr(agent, name, None)
        if torch.is_tensor(tensor):
            total += int(tensor.numel()) * int(tensor.element_size())
    # torch.save metadata/storage headers are workload-dependent.  A small bounded allowance
    # keeps the policy conservative without serializing a live checkpoint just to choose a mode.
    return int(math.ceil(float(total) * 1.04 + 4096.0))


def resolve_artifact_io_mode(
    requested: str,
    workload: ArtifactWorkload,
    *,
    min_async_bytes_per_update: float = 16_384.0,
    min_projected_io_bytes: float = 64.0 * 1024.0 * 1024.0,
    max_queue_block_ratio: float = 0.0025,
) -> ArtifactIODecision:
    mode = str(requested or "auto").strip().lower()
    if mode not in {"auto", "sync", "async"}:
        raise ValueError("artifact I/O mode must be auto/sync/async")
    if mode != "auto":
        return ArtifactIODecision(
            requested=mode, resolved=mode, score=1.0 if mode == "async" else 0.0,
            reason="explicit_override", workload=asdict(workload),
        )
    block = max(0.0, float(workload.queue_block_ratio))
    pressure = max(0.0, float(workload.io_bytes_per_update))
    projected = max(0.0, float(workload.projected_io_bytes))
    score = min(1.0, pressure / max(1.0, float(min_async_bytes_per_update)))
    score *= min(1.0, projected / max(1.0, float(min_projected_io_bytes)))
    if block > float(max_queue_block_ratio):
        return ArtifactIODecision(
            requested="auto", resolved="sync", score=score,
            reason="queue_backpressure_exceeds_limit", workload=asdict(workload),
        )
    if workload.checkpoint_interval <= 0 and workload.logger_bytes_per_update < 4096.0:
        return ArtifactIODecision(
            requested="auto", resolved="sync", score=score,
            reason="light_logging_without_checkpoints", workload=asdict(workload),
        )
    if pressure >= float(min_async_bytes_per_update) and projected >= float(min_projected_io_bytes):
        return ArtifactIODecision(
            requested="auto", resolved="async", score=score,
            reason="sustained_io_pressure", workload=asdict(workload),
        )
    return ArtifactIODecision(
        requested="auto", resolved="sync", score=score,
        reason="io_pressure_below_async_threshold", workload=asdict(workload),
    )
