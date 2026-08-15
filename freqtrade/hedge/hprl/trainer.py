"""Online and offline tensor-training orchestration for HPRL."""

from __future__ import annotations

from dataclasses import dataclass

from .action_space import canonicalize_offline_action_tensor
from .config import HPRLActionConfig, HPRLMemoryConfig, HPRLTrainingConfig
from .device import require_torch, seed_everything, torch_device
from .memory import (
    cuda_memory_state,
    oom_diagnostics,
    phase_boundary_cleanup,
    plan_replay,
    reserve_bytes_for,
)
from .replay import CudaReplayPrefetcher, ReplayBatch, TensorReplayBuffer


torch = require_torch()


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    environment_steps: int
    transitions: int
    updates: int
    final_equity_mean: float
    group_resets: int
    last_metrics: dict[str, float]


@dataclass(frozen=True, slots=True)
class OfflineTrainingSummary:
    updates: int
    samples_seen: int
    last_metrics: dict[str, float]


class DiscountedReturnNormalizer:
    """Normalize online rewards by running discounted-return standard deviation.

    Statistics remain on the selected device so CUDA training does not shuttle reward state back to
    the host.  Environment rewards remain raw; only replay rewards are normalized.
    """

    def __init__(
        self,
        num_envs: int,
        gamma: float,
        *,
        device: str | object = "auto",
        eps: float = 1e-8,
        validate_inputs: bool = True,
    ) -> None:
        if not isinstance(num_envs, int) or isinstance(num_envs, bool) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not 0.0 < float(gamma) <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if not 0.0 < float(eps) < 1.0:
            raise ValueError("eps must be in (0, 1)")
        self.num_envs = num_envs
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.device = torch_device(device)
        self.validate_inputs = bool(validate_inputs)
        self.discounted_return = torch.zeros(num_envs, device=self.device, dtype=torch.float32)
        self.count = 0
        self.mean = torch.zeros((), device=self.device, dtype=torch.float32)
        self.m2 = torch.zeros((), device=self.device, dtype=torch.float32)
        self.scale = torch.ones((), device=self.device, dtype=torch.float32)

    @torch.no_grad()
    def normalize(self, reward, done):
        original_shape = tuple(reward.shape)
        reward_flat = reward.to(self.device, dtype=torch.float32, non_blocking=True).reshape(-1)
        done_flat = done.to(self.device, dtype=torch.float32, non_blocking=True).reshape(-1)
        if reward_flat.numel() != self.num_envs or done_flat.numel() != self.num_envs:
            raise ValueError("reward/done must contain one value per vector environment")
        if self.validate_inputs:
            if not torch.isfinite(reward_flat).all() or not torch.isfinite(done_flat).all():
                raise ValueError("reward/done normalization inputs must be finite")
            if not bool(((done_flat == 0.0) | (done_flat == 1.0)).all()):
                raise ValueError("done normalization input must be boolean-like 0/1")

        self.discounted_return.mul_(self.gamma * (1.0 - done_flat)).add_(reward_flat)
        values = self.discounted_return.detach()
        batch_count = values.numel()
        batch_mean = values.mean()
        batch_m2 = (values - batch_mean).square().sum()
        previous_count = self.count
        total_count = previous_count + batch_count
        if previous_count == 0:
            self.mean.copy_(batch_mean)
            self.m2.copy_(batch_m2)
        else:
            delta = batch_mean - self.mean
            self.mean.add_(delta * (batch_count / total_count))
            correction = delta.square() * previous_count * batch_count / total_count
            self.m2.add_(batch_m2 + correction)
        self.count = total_count

        variance = self.m2 / max(self.count, 1)
        if self.count >= 2 * self.num_envs:
            self.scale.copy_(torch.sqrt(torch.clamp(variance, min=self.eps)))
        else:
            self.scale.fill_(1.0)
        return (reward_flat / self.scale).reshape(original_shape)

    def release(self) -> None:
        self.discounted_return = torch.empty(0, device=self.device)
        self.mean = torch.empty(0, device=self.device)
        self.m2 = torch.empty(0, device=self.device)
        self.scale = torch.empty(0, device=self.device)
        self.count = 0


class OnlineTrainer:
    """High-throughput trainer shared by online HPRL agents.

    Model, environment and optionally replay tensors can all remain resident on CUDA.  If replay is
    configured on CPU, sampled batches use non-blocking host-to-device transfer when pinned memory
    is available.
    """

    def __init__(
        self,
        env,
        agent,
        config: HPRLTrainingConfig,
        memory_config: HPRLMemoryConfig | None = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.config = config
        self.memory_config = memory_config or HPRLMemoryConfig()
        self.device = torch_device(env.device)
        if hasattr(agent, "device") and torch.device(agent.device) != self.device:
            raise ValueError("agent and environment must use the same device")

        self.replay_plan = plan_replay(
            config.replay_device,
            self.device,
            capacity=config.replay_capacity,
            obs_dim=env.observation_dim,
            action_dim=env.action_dim,
            memory_config=self.memory_config,
        )
        replay_device = torch_device(self.replay_plan.resolved_device)
        self.buffer = TensorReplayBuffer(
            config.replay_capacity,
            env.observation_dim,
            env.action_dim,
            device=str(replay_device),
            pin_memory=config.pin_memory,
            validate_inputs=config.runtime_checks,
        )
        self.replay_prefetcher = None
        if (
            config.replay_prefetch
            and self.device.type == "cuda"
            and self.buffer.device.type == "cpu"
            and self.buffer.pin_memory
        ):
            self.replay_prefetcher = CudaReplayPrefetcher(
                self.buffer,
                config.batch_size,
                self.device,
                slots=config.replay_prefetch_slots,
            )
        normalization = getattr(agent, "reward_normalization", None)
        if normalization not in (None, "return_std"):
            raise ValueError(f"unsupported HPRL reward normalization mode: {normalization!r}")
        self.reward_normalizer = (
            DiscountedReturnNormalizer(
                env.envs,
                config.gamma,
                device=self.device,
                validate_inputs=config.runtime_checks,
            )
            if normalization == "return_std"
            else None
        )

    def _warmup_action(self):
        # Environment owns the action contract. Tiered mode samples exact canonical level codes;
        # continuous mode preserves the prior exposure-domain warmup behavior.
        return self.env.sample_random_action()

    def _training_batch(self) -> ReplayBatch:
        if self.replay_prefetcher is not None:
            return self.replay_prefetcher.next()
        batch = (
            self.buffer.sample_reusable(self.config.batch_size)
            if self.config.replay_reuse_sample_buffers
            else self.buffer.sample(self.config.batch_size)
        )
        if torch.device(batch.obs.device) == torch.device(self.agent.device):
            return batch
        # Prefetch disabled: preserve correctness with the single sampled batch.
        # The high-throughput path above uses bounded multi-slot pinned/device staging.
        return batch.to(self.agent.device, non_blocking=not self.buffer.pin_memory)

    def run(self, environment_steps: int) -> TrainingSummary:
        if environment_steps < 1:
            raise ValueError("environment_steps must be positive")
        seed_everything(
            self.config.seed,
            deterministic=self.config.deterministic,
            device=self.device,
        )
        obs, _ = self.env.reset()
        updates = 0
        transitions = 0
        group_resets = 0
        last_metrics: dict[str, float] = {}
        last_equity = self.env.equity
        for _ in range(environment_steps):
            if transitions < self.config.warmup_steps:
                action = self._warmup_action()
            else:
                action = self.agent.act(obs, deterministic=False)
            step = self.env.step(action)
            done = torch.logical_or(step.terminated, step.truncated)
            # Per-environment hard terminals auto-reset on the accelerator. The only group reset is
            # the deterministic historical-data boundary, which is already a host-side bool.
            group_done = bool(step.info.get("time_done", False))
            replay_done = done
            executed_action = step.info.get("executed_action")
            if executed_action is None:
                raise RuntimeError("HPRL environment must report executed_action for replay")
            replay_reward = step.reward
            if self.reward_normalizer is not None:
                replay_reward = self.reward_normalizer.normalize(step.reward, replay_done)
            self.buffer.add(obs, executed_action, replay_reward, step.observation, replay_done)
            transitions += self.env.envs
            obs = step.observation
            last_equity = step.info["equity"]
            ready_to_update = (
                transitions >= self.config.warmup_steps
                and len(self.buffer) >= self.config.batch_size
            )
            if ready_to_update:
                for _ in range(self.config.gradient_steps):
                    collect_metrics = (
                        updates == 0
                        or (updates + 1) % self.config.metrics_interval == 0
                    )
                    try:
                        metrics = self.agent.update(
                            self._training_batch(), collect_metrics=collect_metrics
                        )
                    except torch.OutOfMemoryError as exc:
                        diagnostics = oom_diagnostics(self.device)
                        phase_boundary_cleanup(
                            self.device,
                            enabled=self.memory_config.phase_boundary_cleanup,
                        )
                        raise RuntimeError(
                            "HPRL CUDA out-of-memory during agent update. "
                            f"memory={diagnostics}; replay_plan={self.replay_plan}. "
                            "Use replay_device='auto'/'cpu', dataset_mode='windowed', "
                            "or reduce batch_size/replay_capacity/parallel_envs."
                        ) from exc
                    if metrics.values:
                        last_metrics = dict(metrics.values)
                    updates += 1
            if group_done:
                group_resets += 1
                obs, _ = self.env.reset()
        return TrainingSummary(
            environment_steps=environment_steps,
            transitions=transitions,
            updates=updates,
            final_equity_mean=float(last_equity.mean().item()),
            group_resets=group_resets,
            last_metrics=last_metrics,
        )


    def close(
        self,
        *,
        close_environment: bool = False,
        aggressive: bool = False,
    ) -> None:
        """Release temporary training storage without destroying the trained agent."""
        if self.replay_prefetcher is not None:
            self.replay_prefetcher.release()
            self.replay_prefetcher = None
        if getattr(self, "buffer", None) is not None:
            self.buffer.release(aggressive=False)
        if self.reward_normalizer is not None:
            self.reward_normalizer.release()
            self.reward_normalizer = None
        if close_environment:
            close = getattr(self.env, "close", None)
            if callable(close):
                close(aggressive=False)
        if aggressive:
            phase_boundary_cleanup(self.device, enabled=True)


class OfflineTrainer:
    """Device-resident sampling driver for ReBRAC-style offline HPRL agents."""

    def __init__(
        self,
        dataset,
        agent,
        config: HPRLTrainingConfig,
        *,
        device: str | None = None,
        memory_config: HPRLMemoryConfig | None = None,
        action_config: HPRLActionConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.agent = agent
        self.config = config
        self.memory_config = memory_config or HPRLMemoryConfig()
        self.action_config = action_config
        self.device = torch_device(config.device if device is None else device)
        if hasattr(agent, "device") and torch.device(agent.device) != self.device:
            raise ValueError("agent and offline trainer must use the same device")
        if dataset.observation_dim < 1 or dataset.action_dim < 1:
            raise ValueError("offline dataset dimensions must be positive")

    def run(self, updates: int) -> OfflineTrainingSummary:
        if updates < 1:
            raise ValueError("offline updates must be positive")
        seed_everything(
            self.config.seed,
            deterministic=self.config.deterministic,
            device=self.device,
        )
        storage_device = self.device
        tensor_device = self.device
        if self.device.type == "cuda" and self.memory_config.dataset_mode != "resident":
            # Decide storage before tensorization.  Materializing a full CPU dataset
            # merely to measure it and then copying another full dataset to CUDA
            # creates a large transient double-residency peak.  Offline transition
            # storage is exactly float32 and can be estimated from schema dimensions.
            total_bytes = int(
                len(self.dataset)
                * (
                    self.dataset.observation_dim * 2
                    + self.dataset.action_dim
                    + 2
                )
                * 4
            )
            state = cuda_memory_state(self.device)
            assert state is not None
            reserve = reserve_bytes_for(state, self.memory_config)
            budget = max(
                0,
                int(
                    max(0, state.free_bytes - reserve)
                    * float(self.memory_config.dataset_gpu_fraction)
                ),
            )
            use_gpu = (
                self.memory_config.dataset_mode == "auto" and total_bytes <= budget
            )
            tensor_device = self.device if use_gpu else torch.device("cpu")
            storage_device = tensor_device
        values = self.dataset.tensors(
            str(tensor_device),
            chunk_rows=self.memory_config.offline_tensorize_chunk_rows,
        )
        if self.action_config is not None:
            # Canonicalize in bounded chunks. Margin/notional conversion uses a small
            # [chunk, action_dim, levels] distance tensor and never duplicates the full dataset.
            rows_total = int(values["action"].shape[0])
            chunk_rows = self.memory_config.offline_tensorize_chunk_rows
            for start in range(0, rows_total, chunk_rows):
                end = min(start + chunk_rows, rows_total)
                action_slice = values["action"][start:end]
                canonical = canonicalize_offline_action_tensor(
                    action_slice,
                    self.action_config,
                    getattr(self.dataset, "action_unit", "policy_code"),
                )
                action_slice.copy_(canonical)
        if self.memory_config.release_offline_source_after_tensorize:
            release_source = getattr(self.dataset, "release_source", None)
            if callable(release_source):
                release_source()
        rows = int(values["obs"].shape[0])
        calibrate = getattr(self.agent, "calibrate_return_support", None)
        if callable(calibrate):
            calibrate(values["reward"], values["done"])
        last_metrics: dict[str, float] = {}
        for _ in range(updates):
            idx = torch.randint(
                0, rows, (self.config.batch_size,), device=storage_device
            )
            batch = ReplayBatch(
                obs=values["obs"][idx],
                action=values["action"][idx],
                reward=values["reward"][idx],
                next_obs=values["next_obs"][idx],
                done=values["done"][idx],
            )
            if torch.device(batch.obs.device) != self.device:
                batch = batch.to(self.device, non_blocking=False)
            collect_metrics = (
                _ == 0 or (_ + 1) % self.config.metrics_interval == 0 or _ + 1 == updates
            )
            metrics = self.agent.update(batch, collect_metrics=collect_metrics)
            if metrics.values:
                last_metrics = dict(metrics.values)
        return OfflineTrainingSummary(
            updates=updates,
            samples_seen=updates * self.config.batch_size,
            last_metrics=last_metrics,
        )


    def close(
        self,
        *,
        release_source: bool = False,
        aggressive: bool = False,
    ) -> None:
        """Release one-shot offline source rows after training/evaluation handoff."""
        if release_source:
            release = getattr(self.dataset, "release_source", None)
            if callable(release):
                release()
        if aggressive:
            phase_boundary_cleanup(self.device, enabled=True)
