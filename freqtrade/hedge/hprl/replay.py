"""Preallocated replay buffer with bounded pinned-memory staging for HPRL."""

from __future__ import annotations

from dataclasses import dataclass

from .device import require_torch, torch_device
from .memory import replay_nbytes


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    obs: object
    action: object
    reward: object
    next_obs: object
    done: object

    def to(self, device: str | object, *, non_blocking: bool = True) -> "ReplayBatch":
        target = torch_device(device)
        return ReplayBatch(
            obs=self.obs.to(target, non_blocking=non_blocking),
            action=self.action.to(target, non_blocking=non_blocking),
            reward=self.reward.to(target, non_blocking=non_blocking),
            next_obs=self.next_obs.to(target, non_blocking=non_blocking),
            done=self.done.to(target, non_blocking=non_blocking),
        )


class TensorReplayBuffer:
    """Fixed-capacity tensor replay.

    CPU replay storage is deliberately pageable.  Never page-lock the whole replay.  When CUDA
    is available and ``pin_memory`` is enabled, only a reusable sampled batch is page-locked.  This
    avoids locking an entire multi-gigabyte replay buffer while retaining fast
    non-blocking host-to-device batch transfers.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        *,
        device: str = "auto",
        pin_memory: bool = True,
        validate_inputs: bool = True,
    ) -> None:
        torch = require_torch()
        if capacity < 1 or obs_dim < 1 or action_dim < 1:
            raise ValueError("replay dimensions must be positive")
        self.torch = torch
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.device = torch_device(device)
        self.pin_memory = bool(
            pin_memory and self.device.type == "cpu" and torch.cuda.is_available()
        )
        self.validate_inputs = bool(validate_inputs)
        kwargs = {"dtype": torch.float32, "device": self.device}
        # One packed transition storage turns five replay gathers into one index_select.
        # Public field tensors are zero-copy views, preserving the existing ReplayBatch contract.
        self.transition_dim = self.obs_dim * 2 + self.action_dim + 2
        self._storage = torch.empty((capacity, self.transition_dim), **kwargs)
        self._bind_storage_views()
        self._cursor = 0
        self._size = 0
        self._stage_batch_sizes: dict[int, int] = {}
        self._pinned_stages: dict[int, ReplayBatch] = {}
        self._pinned_stage_storage: dict[int, object] = {}
        self._sample_stages: dict[int, ReplayBatch] = {}
        self._sample_stage_storage: dict[int, object] = {}
        self._sample_stage_batch_sizes: dict[int, int] = {}
        self._index_buffers: dict[int, object] = {}
        self._add_stage_storage = None
        self._add_stage_rows = 0
        self._released = False

    def _views_from_packed(self, packed) -> ReplayBatch:
        o1 = self.obs_dim
        a1 = o1 + self.action_dim
        r1 = a1 + 1
        n1 = r1 + self.obs_dim
        return ReplayBatch(
            packed[:, :o1],
            packed[:, o1:a1],
            packed[:, a1:r1],
            packed[:, r1:n1],
            packed[:, n1:n1 + 1],
        )

    def _bind_storage_views(self) -> None:
        views = self._views_from_packed(self._storage)
        self.obs = views.obs
        self.action = views.action
        self.reward = views.reward
        self.next_obs = views.next_obs
        self.done = views.done

    def _assert_open(self) -> None:
        if self._released:
            raise RuntimeError("replay buffer has been released")

    @property
    def persistent_bytes(self) -> int:
        return 0 if self._released else replay_nbytes(
            self.capacity, self.obs_dim, self.action_dim
        )

    @property
    def pinned_stage_bytes(self) -> int:
        return int(
            sum(
                value.numel() * value.element_size()
                for value in self._pinned_stage_storage.values()
            )
        )

    @property
    def sample_stage_bytes(self) -> int:
        return int(
            sum(
                value.numel() * value.element_size()
                for value in self._sample_stage_storage.values()
            )
        )

    @property
    def add_stage_bytes(self) -> int:
        if self._add_stage_storage is None:
            return 0
        return int(self._add_stage_storage.numel() * self._add_stage_storage.element_size())

    def __len__(self) -> int:
        return self._size

    def staging_identity(self) -> dict[str, object]:
        """Return stable storage identities for long-run allocator/reuse validation."""
        def ptr(value):
            return int(value.data_ptr()) if value is not None and value.numel() else 0
        return {
            "storage": ptr(self._storage),
            "pinned": {int(slot): ptr(value) for slot, value in self._pinned_stage_storage.items()},
            "sample": {int(slot): ptr(value) for slot, value in self._sample_stage_storage.items()},
            "indices": {int(size): ptr(value) for size, value in self._index_buffers.items()},
            "add": ptr(self._add_stage_storage),
        }


    def release(self, *, aggressive: bool = False) -> None:
        """Release replay storage and pinned staging at a training phase boundary."""
        if self._released:
            return
        torch = self.torch
        self._storage = torch.empty(
            (0, self.transition_dim), dtype=torch.float32, device=self.device
        )
        self._bind_storage_views()
        self._pinned_stages.clear()
        self._pinned_stage_storage.clear()
        self._stage_batch_sizes.clear()
        self._sample_stages.clear()
        self._sample_stage_storage.clear()
        self._sample_stage_batch_sizes.clear()
        self._index_buffers.clear()
        self._add_stage_storage = None
        self._add_stage_rows = 0
        self._cursor = 0
        self._size = 0
        self._released = True
        if aggressive:
            from .memory import phase_boundary_cleanup

            phase_boundary_cleanup(self.device, enabled=True)

    def _ensure_pinned_stage(self, batch_size: int, *, slot: int = 0) -> ReplayBatch:
        self._assert_open()
        if slot < 0:
            raise ValueError("pinned staging slot must be non-negative")
        existing = self._pinned_stages.get(slot)
        if existing is not None and self._stage_batch_sizes.get(slot) == batch_size:
            return existing
        torch = self.torch
        kwargs = {"dtype": torch.float32, "device": "cpu", "pin_memory": True}
        storage = torch.empty((batch_size, self.transition_dim), **kwargs)
        stage = self._views_from_packed(storage)
        self._pinned_stage_storage[slot] = storage
        self._pinned_stages[slot] = stage
        self._stage_batch_sizes[slot] = batch_size
        return stage

    def _ensure_sample_stage(self, batch_size: int, *, slot: int = 0) -> ReplayBatch:
        self._assert_open()
        existing = self._sample_stages.get(slot)
        if existing is not None and self._sample_stage_batch_sizes.get(slot) == batch_size:
            return existing
        storage = self.torch.empty(
            (batch_size, self.transition_dim),
            dtype=self.torch.float32,
            device=self.device,
        )
        stage = self._views_from_packed(storage)
        self._sample_stage_storage[slot] = storage
        self._sample_stages[slot] = stage
        self._sample_stage_batch_sizes[slot] = batch_size
        return stage

    def _draw_indices(self, batch_size: int):
        idx = self._index_buffers.get(batch_size)
        if idx is None:
            idx = self.torch.empty((batch_size,), dtype=self.torch.int64, device=self.device)
            self._index_buffers[batch_size] = idx
        idx.random_(0, self._size)
        return idx

    def _ensure_add_stage(self, rows: int):
        if self._add_stage_storage is None or self._add_stage_rows != rows:
            self._add_stage_storage = self.torch.empty(
                (rows, self.transition_dim),
                dtype=self.torch.float32,
                device=self.device,
            )
            self._add_stage_rows = int(rows)
        return self._add_stage_storage

    def add(self, obs, action, reward, next_obs, done) -> None:
        self._assert_open()
        torch = self.torch
        obs = obs.reshape(-1, self.obs_dim).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        action = action.reshape(-1, self.action_dim).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        reward = reward.reshape(-1, 1).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        next_obs = next_obs.reshape(-1, self.obs_dim).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        done = done.reshape(-1, 1).to(
            self.device, dtype=torch.float32, non_blocking=True
        )
        n = obs.shape[0]
        if n < 1:
            raise ValueError("replay add batch cannot be empty")
        if not (action.shape[0] == reward.shape[0] == next_obs.shape[0] == done.shape[0] == n):
            raise ValueError("replay batch dimensions are inconsistent")
        values = (obs, action, reward, next_obs, done)
        if self.validate_inputs:
            if any(not torch.isfinite(value).all() for value in values):
                raise ValueError("replay transitions must be finite")
            if ((done < 0) | (done > 1)).any():
                raise ValueError("replay done flags must be within [0, 1]")
        if n >= self.capacity:
            obs, action, reward, next_obs, done = (
                value[-self.capacity:] for value in values
            )
            n = self.capacity
        packed = self._ensure_add_stage(n)
        self.torch.cat((obs, action, reward, next_obs, done), dim=-1, out=packed)
        first = min(n, self.capacity - self._cursor)
        second = n - first
        start = self._cursor
        end = start + first
        self._storage[start:end].copy_(packed[:first], non_blocking=True)
        if second:
            self._storage[:second].copy_(packed[first:], non_blocking=True)
        self._cursor = (self._cursor + n) % self.capacity
        self._size = min(self.capacity, self._size + n)

    def _validate_sample_size(self, batch_size: int) -> None:
        self._assert_open()
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("replay batch_size must be a positive integer")
        if self._size < batch_size:
            raise ValueError("replay buffer does not contain enough samples")

    def sample(self, batch_size: int) -> ReplayBatch:
        """Return an independent sampled batch for external callers."""
        self._validate_sample_size(batch_size)
        idx = self._draw_indices(batch_size)
        packed = self.torch.index_select(self._storage, 0, idx)
        return self._views_from_packed(packed)

    def sample_reusable(self, batch_size: int, *, staging_slot: int = 0) -> ReplayBatch:
        """Sample into bounded reusable storage for the sequential training hot path.

        The returned batch is valid until the same staging slot is sampled again.  Trainers only
        reuse a slot after the prior update has consumed it, removing allocator churn without
        changing public ``sample()`` ownership semantics.
        """
        self._validate_sample_size(batch_size)
        idx = self._draw_indices(batch_size)
        if self.pin_memory:
            stage = self._ensure_pinned_stage(batch_size, slot=staging_slot)
            stage_storage = self._pinned_stage_storage[staging_slot]
        else:
            stage = self._ensure_sample_stage(batch_size, slot=staging_slot)
            stage_storage = self._sample_stage_storage[staging_slot]
        self.torch.index_select(self._storage, 0, idx, out=stage_storage)
        return stage


class CudaReplayPrefetcher:
    """Pipeline CPU replay batches into CUDA with bounded double/triple buffering.

    One batch is consumed on the default training stream while the copy stream transfers the next
    sampled batch.  Per-slot transfer and consume events prevent host or device staging from being
    reused before the prior asynchronous work has completed.
    """

    def __init__(
        self,
        buffer: TensorReplayBuffer,
        batch_size: int,
        device,
        *,
        slots: int = 2,
    ):
        torch = buffer.torch
        target = torch_device(device)
        if buffer.device.type != "cpu" or target.type != "cuda" or not buffer.pin_memory:
            raise ValueError("CUDA replay prefetch requires pinned CPU replay and CUDA target")
        if slots < 2 or slots > 4:
            raise ValueError("CUDA replay prefetch slots must be within [2, 4]")
        self.buffer = buffer
        self.batch_size = int(batch_size)
        self.device = target
        self.slots = int(slots)
        self.copy_stream = torch.cuda.Stream(device=target)
        self._transfer_events = [torch.cuda.Event(blocking=False) for _ in range(self.slots)]
        self._consume_events = [torch.cuda.Event(blocking=False) for _ in range(self.slots)]
        self._transfer_launched = [False] * self.slots
        self._consume_recorded = [False] * self.slots
        self._prefetched_slot: int | None = None
        self._last_returned_slot: int | None = None
        self._device_storage = [
            torch.empty(
                (self.batch_size, buffer.transition_dim),
                dtype=torch.float32,
                device=target,
            )
            for _ in range(self.slots)
        ]
        self._device_batches = [buffer._views_from_packed(value) for value in self._device_storage]

    def _launch(self, slot: int) -> None:
        torch = self.buffer.torch
        if self._transfer_launched[slot]:
            # Reusing the bounded pinned host slot is legal only after its previous H2D copy.
            self._transfer_events[slot].synchronize()
        self.buffer.sample_reusable(self.batch_size, staging_slot=slot)
        host_storage = self.buffer._pinned_stage_storage[slot]
        with torch.cuda.stream(self.copy_stream):
            if self._consume_recorded[slot]:
                self.copy_stream.wait_event(self._consume_events[slot])
            self._device_storage[slot].copy_(host_storage, non_blocking=True)
            self._transfer_events[slot].record(self.copy_stream)
        self._transfer_launched[slot] = True

    def next(self) -> ReplayBatch:
        torch = self.buffer.torch
        current_stream = torch.cuda.current_stream(self.device)
        if self._last_returned_slot is not None:
            consumed = self._last_returned_slot
            # This event is enqueued after every kernel from the previous agent.update() on the
            # default stream.  The copy stream waits before overwriting that device staging slot.
            self._consume_events[consumed].record(current_stream)
            self._consume_recorded[consumed] = True

        if self._prefetched_slot is None:
            ready = 0
            self._launch(ready)
        else:
            ready = self._prefetched_slot
        current_stream.wait_event(self._transfer_events[ready])

        next_slot = (ready + 1) % self.slots
        self._launch(next_slot)
        self._prefetched_slot = next_slot
        self._last_returned_slot = ready
        return self._device_batches[ready]

    def staging_identity(self) -> dict[str, object]:
        return {
            "device": tuple(int(value.data_ptr()) for value in self._device_storage),
            "pinned": self.buffer.staging_identity().get("pinned", {}),
            "slots": self.slots,
        }

    @property
    def device_stage_bytes(self) -> int:
        return int(sum(value.numel() * value.element_size() for value in self._device_storage))

    @property
    def prefetch_depth(self) -> int:
        return 1 if self._prefetched_slot is not None else 0

    def release(self) -> None:
        torch = self.buffer.torch
        # Phase-boundary cleanup may synchronize.  This guarantees no queued consumer or transfer
        # still references the staging storage when it is released.
        torch.cuda.synchronize(self.device)
        self._device_batches.clear()
        self._device_storage.clear()
        self._transfer_events.clear()
        self._consume_events.clear()
        self._transfer_launched.clear()
        self._consume_recorded.clear()
        self._prefetched_slot = None
        self._last_returned_slot = None
