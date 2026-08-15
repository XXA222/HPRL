"""Bounded CPU/CUDA memory planning for HPRL and long historical datasets.

The module deliberately separates *storage* from *compute*.  Models and the vector
simulation may run on CUDA while large market datasets or replay buffers remain on
CPU when VRAM is constrained.  Sequential market data can be staged to the GPU in
bounded windows, avoiding a full multi-year tensor copy into VRAM.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .device import require_torch, torch_device
from .errors import HPRLConfigError


MIB = 1024**2
GIB = 1024**3


def tensor_nbytes(value: object | None) -> int:
    if value is None:
        return 0
    torch = require_torch()
    if not torch.is_tensor(value):
        return 0
    return int(value.numel() * value.element_size())


def dataset_nbytes(dataset: object) -> int:
    return sum(
        tensor_nbytes(getattr(dataset, name, None))
        for name in ("features", "forward_returns", "funding_rates", "available_notional")
    )


def replay_nbytes(capacity: int, obs_dim: int, action_dim: int) -> int:
    """Exact persistent storage size of :class:`TensorReplayBuffer` float32 tensors."""
    if min(capacity, obs_dim, action_dim) < 1:
        raise ValueError("replay dimensions must be positive")
    floats_per_transition = obs_dim * 2 + action_dim + 2
    return int(capacity * floats_per_transition * 4)


def process_rss_bytes() -> int | None:
    """Return Linux RSS without adding a psutil dependency."""
    status = Path("/proc/self/status")
    if not status.is_file():
        return None
    try:
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


@dataclass(frozen=True, slots=True)
class CudaMemoryState:
    device: str
    total_bytes: int
    free_bytes: int
    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int


def cuda_memory_state(device: str | object) -> CudaMemoryState | None:
    torch = require_torch()
    resolved = torch_device(device)
    if resolved.type != "cuda":
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(resolved)
    return CudaMemoryState(
        device=str(resolved),
        total_bytes=int(total_bytes),
        free_bytes=int(free_bytes),
        allocated_bytes=int(torch.cuda.memory_allocated(resolved)),
        reserved_bytes=int(torch.cuda.memory_reserved(resolved)),
        max_allocated_bytes=int(torch.cuda.max_memory_allocated(resolved)),
        max_reserved_bytes=int(torch.cuda.max_memory_reserved(resolved)),
    )


def _budget_bytes(state: CudaMemoryState, fraction: float, reserve_bytes: int) -> int:
    usable = max(0, state.free_bytes - reserve_bytes)
    return max(0, int(usable * float(fraction)))


def reserve_bytes_for(state: CudaMemoryState, memory_config) -> int:
    proportional = int(state.total_bytes * float(memory_config.cuda_reserve_fraction))
    return max(int(memory_config.min_cuda_reserve_bytes), proportional)


@dataclass(frozen=True, slots=True)
class DatasetMemoryPlan:
    requested_mode: str
    resolved_mode: str
    training_device: str
    dataset_bytes: int
    gpu_budget_bytes: int
    window_steps: int
    reason: str


@dataclass(frozen=True, slots=True)
class ReplayMemoryPlan:
    requested_device: str
    resolved_device: str
    replay_bytes: int
    gpu_budget_bytes: int
    reason: str


def plan_dataset(
    dataset: object,
    training_device: str | object,
    memory_config,
) -> DatasetMemoryPlan:
    device = torch_device(training_device)
    requested = str(memory_config.dataset_mode).strip().lower()
    size = dataset_nbytes(dataset)
    if device.type != "cuda":
        return DatasetMemoryPlan(
            requested, "resident", str(device), size, 0,
            int(memory_config.dataset_window_steps), "CPU training keeps market tensors resident",
        )
    state = cuda_memory_state(device)
    assert state is not None
    reserve = reserve_bytes_for(state, memory_config)
    budget = _budget_bytes(state, memory_config.dataset_gpu_fraction, reserve)
    if requested == "resident":
        resolved = "resident"
        reason = "explicit resident dataset mode"
    elif requested == "windowed":
        resolved = "windowed"
        reason = "explicit bounded window dataset mode"
    elif requested == "auto":
        resolved = "resident" if size <= budget else "windowed"
        reason = (
            "dataset fits configured CUDA dataset budget"
            if resolved == "resident"
            else "dataset exceeds configured CUDA dataset budget"
        )
    else:
        raise HPRLConfigError("dataset_mode must be auto/resident/windowed")
    if memory_config.strict_budget and resolved == "resident" and size > budget:
        raise HPRLConfigError(
            f"resident market dataset needs {size} bytes, above CUDA dataset budget {budget}; "
            "use dataset_mode='windowed'/'auto' or increase the memory budget"
        )
    return DatasetMemoryPlan(
        requested, resolved, str(device), size, budget,
        int(memory_config.dataset_window_steps), reason,
    )


def plan_replay(
    requested: str | object,
    training_device: str | object,
    *,
    capacity: int,
    obs_dim: int,
    action_dim: int,
    memory_config,
) -> ReplayMemoryPlan:
    training = torch_device(training_device)
    request = str(requested).strip().lower()
    if request == "gpu":
        request = "cuda"
    size = replay_nbytes(capacity, obs_dim, action_dim)
    if request == "same":
        resolved = training
        reason = "explicit same-as-training replay device"
    elif request == "auto":
        if training.type != "cuda":
            resolved = torch_device("cpu")
            reason = "CPU training uses CPU replay"
        else:
            state = cuda_memory_state(training)
            assert state is not None
            reserve = reserve_bytes_for(state, memory_config)
            budget = _budget_bytes(state, memory_config.replay_gpu_fraction, reserve)
            if size <= budget:
                resolved = training
                reason = "replay fits configured CUDA replay budget"
            else:
                resolved = torch_device("cpu")
                reason = "replay exceeds CUDA replay budget; using CPU storage"
    else:
        resolved = torch_device(request)
        reason = "explicit replay device"

    budget = 0
    if training.type == "cuda":
        state = cuda_memory_state(training)
        assert state is not None
        reserve = reserve_bytes_for(state, memory_config)
        budget = _budget_bytes(state, memory_config.replay_gpu_fraction, reserve)
        if (
            memory_config.strict_budget
            and resolved.type == "cuda"
            and size > budget
        ):
            raise HPRLConfigError(
                f"CUDA replay needs {size} bytes, above configured replay budget {budget}; "
                "use replay_device='auto'/'cpu' or reduce replay_capacity"
            )
    return ReplayMemoryPlan(request, str(resolved), size, budget, reason)


class MarketDatasetAccessor:
    """Resident or bounded-window view over sequential market tensors.

    In ``windowed`` CUDA mode the authoritative dataset stays on CPU and only a
    contiguous time window is staged to CUDA.  The old window is dropped before the
    next is materialized so CUDA allocator blocks can be reused instead of growing
    with historical length.
    """

    def __init__(self, dataset, training_device: str | object, memory_config) -> None:
        torch = require_torch()
        self.torch = torch
        self.device = torch_device(training_device)
        self.memory_config = memory_config
        self.plan = plan_dataset(dataset, self.device, memory_config)
        self.time_steps = int(dataset.features.shape[0])
        self.symbols = int(dataset.features.shape[1])
        self.features = int(dataset.features.shape[2])
        self._window_start = -1
        self._window_end = -1
        self._window: dict[str, object | None] | None = None
        self._gpu_window_buffers: dict[str, object] = {}
        self._pinned_window_buffers: dict[str, object] = {}

        if self.plan.resolved_mode == "resident":
            self.dataset = dataset.to(str(self.device))
            self.source = self.dataset
        else:
            # One authoritative CPU copy.  A GPU input is moved off-device here so the
            # sequential history does not consume VRAM for the whole training run.
            self.source = dataset.to("cpu")
            self.dataset = self.source

    def _stage_tensor(self, name: str, value, start: int, end: int):
        if value is None:
            return None
        piece = value[start:end]
        if self.device.type != "cuda":
            return piece

        # Reuse one fixed GPU window buffer per market tensor.  Allocating a new
        # CUDA tensor for every historical window can fragment the caching allocator
        # and temporarily hold both old/new blocks.  The last short window returns a
        # view over the same preallocated storage.
        length = int(end - start)
        capacity = int(self.memory_config.dataset_window_steps)
        tail_shape = tuple(int(dim) for dim in value.shape[1:])
        shape = (capacity, *tail_shape)
        gpu = self._gpu_window_buffers.get(name)
        if gpu is None or tuple(gpu.shape) != shape:
            gpu = self.torch.empty(shape, device=self.device, dtype=self.torch.float32)
            self._gpu_window_buffers[name] = gpu

        target = gpu[:length]
        if bool(self.memory_config.pin_staging_memory):
            host = self._pinned_window_buffers.get(name)
            if host is None or tuple(host.shape) != shape:
                try:
                    host = self.torch.empty(
                        shape,
                        device="cpu",
                        dtype=self.torch.float32,
                        pin_memory=True,
                    )
                except RuntimeError:
                    host = None
                if host is not None:
                    self._pinned_window_buffers[name] = host
            if host is not None:
                host_view = host[:length]
                host_view.copy_(piece, non_blocking=False)
                target.copy_(host_view, non_blocking=True)
                return target

        target.copy_(piece.to(dtype=self.torch.float32), non_blocking=False)
        return target

    def _ensure_window(self, index: int) -> None:
        if self.plan.resolved_mode == "resident":
            return
        if self._window_start <= index < self._window_end:
            return
        steps = int(self.memory_config.dataset_window_steps)
        start = (index // steps) * steps
        end = min(start + steps, self.time_steps)
        # Drop references first.  Do not call empty_cache here: allocator reuse is faster
        # and keeps steady-state window transitions cheap.
        self._window = None
        self._window_start = start
        self._window_end = end
        source = self.source
        self._window = {
            "features": self._stage_tensor("features", source.features, start, end),
            "forward_returns": self._stage_tensor(
                "forward_returns", source.forward_returns, start, end
            ),
            "funding_rates": self._stage_tensor(
                "funding_rates", source.funding_rates, start, end
            ),
            "available_notional": self._stage_tensor(
                "available_notional", source.available_notional, start, end
            ),
        }

    def _at(self, name: str, index: int):
        if not 0 <= index < self.time_steps:
            raise IndexError(index)
        if self.plan.resolved_mode == "resident":
            value = getattr(self.dataset, name)
            return None if value is None else value[index]
        self._ensure_window(index)
        assert self._window is not None
        value = self._window[name]
        if value is None:
            return None
        return value[index - self._window_start]

    def features_at(self, index: int):
        return self._at("features", index)

    def forward_returns_at(self, index: int):
        return self._at("forward_returns", index)

    def funding_rates_at(self, index: int):
        return self._at("funding_rates", index)

    def available_notional_at(self, index: int):
        return self._at("available_notional", index)

    def release(
        self,
        *,
        aggressive: bool = False,
        release_dataset: bool = False,
    ) -> None:
        self._window = None
        self._gpu_window_buffers.clear()
        self._pinned_window_buffers.clear()
        if release_dataset:
            self.dataset = None
            self.source = None
        if aggressive:
            gc.collect()
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()


def phase_boundary_cleanup(device: str | object, *, enabled: bool = True) -> None:
    """Explicit expensive cleanup for phase boundaries, never for each training step."""
    if not enabled:
        return
    from freqtrade.hedge.memory_lifecycle import (
        HedgeMemoryPolicy,
        phase_boundary_cleanup as process_phase_cleanup,
    )

    resolved = torch_device(device)
    process_phase_cleanup(
        policy=HedgeMemoryPolicy(),
        cuda_device=resolved,
        cuda_empty_cache=resolved.type == "cuda",
    )


def oom_diagnostics(device: str | object) -> dict[str, Any]:
    state = cuda_memory_state(device)
    return {
        "rss_bytes": process_rss_bytes(),
        "cuda": None if state is None else {
            "device": state.device,
            "total_bytes": state.total_bytes,
            "free_bytes": state.free_bytes,
            "allocated_bytes": state.allocated_bytes,
            "reserved_bytes": state.reserved_bytes,
            "max_allocated_bytes": state.max_allocated_bytes,
            "max_reserved_bytes": state.max_reserved_bytes,
        },
    }


def memory_budget_report(
    device: str | object,
    memory_config,
    *,
    dataset_bytes_estimate: int = 0,
    replay_capacity: int = 1_000_000,
    obs_dim: int = 64,
    action_dim: int = 8,
    replay_request: str = "auto",
) -> dict[str, Any]:
    resolved = torch_device(device)
    replay = plan_replay(
        replay_request,
        resolved,
        capacity=replay_capacity,
        obs_dim=obs_dim,
        action_dim=action_dim,
        memory_config=memory_config,
    )
    state = cuda_memory_state(resolved)
    if state is None:
        dataset_mode = "resident"
        dataset_budget = 0
        reserve = 0
    else:
        reserve = reserve_bytes_for(state, memory_config)
        dataset_budget = _budget_bytes(state, memory_config.dataset_gpu_fraction, reserve)
        if memory_config.dataset_mode == "auto":
            dataset_mode = "resident" if dataset_bytes_estimate <= dataset_budget else "windowed"
        else:
            dataset_mode = memory_config.dataset_mode
    return {
        "device": str(resolved),
        "rss_bytes": process_rss_bytes(),
        "cuda": None if state is None else {
            "total_bytes": state.total_bytes,
            "free_bytes": state.free_bytes,
            "allocated_bytes": state.allocated_bytes,
            "reserved_bytes": state.reserved_bytes,
        },
        "reserve_bytes": reserve,
        "dataset": {
            "requested_mode": memory_config.dataset_mode,
            "recommended_mode": dataset_mode,
            "estimated_bytes": int(dataset_bytes_estimate),
            "gpu_budget_bytes": dataset_budget,
            "window_steps": int(memory_config.dataset_window_steps),
        },
        "replay": {
            "requested_device": replay.requested_device,
            "recommended_device": replay.resolved_device,
            "estimated_bytes": replay.replay_bytes,
            "gpu_budget_bytes": replay.gpu_budget_bytes,
            "reason": replay.reason,
        },
    }
