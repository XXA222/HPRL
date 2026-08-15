"""Configuration schema for the isolated HPRL subsystem."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from .errors import HPRLConfigError


def _finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _strict_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class HPRLActionConfig:
    """Dual-leg action contract.

    ``tiered`` keeps a continuous policy latent in [0, 1] for HPRL algorithms while the executed
    LONG/SHORT state is quantized to configurable margin-budget tiers.  This preserves fast
    continuous-control learners without pretending the exchange position target is continuous.
    """

    mode: str = "tiered"
    position_levels: tuple[float, ...] = (0.0, 0.05, 0.12, 0.25, 0.40)
    leverage: float = 1.0
    max_leg_margin_ratio: float = 0.40
    max_gross_margin_ratio: float = 0.80
    max_abs_net_margin_ratio: float = 0.40
    max_increase_levels: int = 1
    max_decrease_levels: int = -1
    tier_hysteresis: float = 0.02
    # Continuous-mode compatibility envelope.
    max_leg_exposure: float = 1.0
    max_gross_exposure: float = 1.5
    max_abs_net_exposure: float = 1.0
    max_step_change: float = 0.20
    min_liquidation_buffer: float = 0.08

    def __post_init__(self) -> None:
        mode = self.mode.strip().lower() if isinstance(self.mode, str) else ""
        if mode not in {"tiered", "continuous"}:
            raise HPRLConfigError("action mode must be tiered/continuous")
        levels = tuple(float(value) for value in self.position_levels)
        if len(levels) < 2 or len(levels) > 16:
            raise HPRLConfigError("position_levels must contain 2..16 entries")
        if not _finite(*levels) or levels[0] != 0.0:
            raise HPRLConfigError("position_levels must be finite and start at 0")
        if any(value < 0 for value in levels) or any(b <= a for a, b in zip(levels, levels[1:])):
            raise HPRLConfigError("position_levels must be non-negative and strictly increasing")
        if levels[-1] > 1.0:
            raise HPRLConfigError("position_levels are margin-budget ratios and cannot exceed 1")
        margins = (
            self.leverage,
            self.max_leg_margin_ratio,
            self.max_gross_margin_ratio,
            self.max_abs_net_margin_ratio,
            self.tier_hysteresis,
        )
        continuous = (
            self.max_leg_exposure,
            self.max_gross_exposure,
            self.max_abs_net_exposure,
            self.max_step_change,
            self.min_liquidation_buffer,
        )
        if not _finite(*margins, *continuous):
            raise HPRLConfigError("action limits must be finite")
        if self.leverage <= 0:
            raise HPRLConfigError("leverage must be positive")
        if self.max_leg_margin_ratio <= 0 or self.max_gross_margin_ratio <= 0:
            raise HPRLConfigError("margin envelopes must be positive")
        if self.max_leg_margin_ratio > 1 or self.max_gross_margin_ratio > 1:
            raise HPRLConfigError("margin envelopes are equity ratios and cannot exceed 1")
        if self.max_abs_net_margin_ratio < 0 or self.max_abs_net_margin_ratio > 1:
            raise HPRLConfigError("absolute net margin ratio must be within [0, 1]")
        if levels[-1] > self.max_leg_margin_ratio + 1e-12:
            raise HPRLConfigError("largest position level exceeds max_leg_margin_ratio")
        if self.max_abs_net_margin_ratio > self.max_gross_margin_ratio:
            raise HPRLConfigError("absolute net margin cannot exceed gross margin")
        if not _strict_int(self.max_increase_levels) or not _strict_int(self.max_decrease_levels):
            raise HPRLConfigError("tier transition limits must be integers")
        max_jump = len(levels) - 1
        if not 0 <= self.max_increase_levels <= max_jump:
            raise HPRLConfigError("max_increase_levels is outside tier range")
        if self.max_decrease_levels != -1 and not 0 <= self.max_decrease_levels <= max_jump:
            raise HPRLConfigError("max_decrease_levels must be -1 (unlimited) or within tier range")
        half_policy_step = 0.5 / float(max_jump)
        if not 0 <= self.tier_hysteresis < half_policy_step:
            raise HPRLConfigError(
                "tier_hysteresis must be non-negative and smaller than half a policy tier step"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "position_levels", levels)
        positive = (self.max_leg_exposure, self.max_gross_exposure, self.max_step_change)
        if any(value <= 0 for value in positive) or self.max_abs_net_exposure < 0:
            raise HPRLConfigError("continuous exposure limits are invalid")
        if self.max_abs_net_exposure > self.max_gross_exposure:
            raise HPRLConfigError("absolute net exposure cannot exceed gross exposure")
        if not 0 <= self.min_liquidation_buffer < 1:
            raise HPRLConfigError("min_liquidation_buffer must be in [0, 1)")

    @property
    def level_count(self) -> int:
        return len(self.position_levels)

    @property
    def joint_states_per_symbol(self) -> int:
        return self.level_count * self.level_count

    @property
    def multi_discrete_nvec(self) -> tuple[int, int]:
        return (self.level_count, self.level_count)


@dataclass(frozen=True, slots=True)
class HPRLCostConfig:
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    base_slippage_bps: float = 0.5
    impact_coefficient_bps: float = 2.0
    max_participation: float = 0.05

    def __post_init__(self) -> None:
        costs = (
            self.maker_fee_bps,
            self.taker_fee_bps,
            self.base_slippage_bps,
            self.impact_coefficient_bps,
        )
        if not _finite(*costs, self.max_participation):
            raise HPRLConfigError("cost parameters must be finite")
        if any(value < 0 for value in costs):
            raise HPRLConfigError("cost parameters cannot be negative")
        if not 0 < self.max_participation <= 1:
            raise HPRLConfigError("max_participation must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class HPRLRewardConfig:
    """Scale-aware reward for one-step equity growth plus independent risk shaping."""

    # 100 means one percentage point of return is roughly one reward unit before shaping.
    return_scale: float = 100.0
    equity: float = 1.0
    drawdown: float = 0.35
    downside: float = 0.15
    cvar: float = 0.10
    turnover: float = 0.001
    # Net equity already contains fees/slippage/impact/funding. Extra shaping defaults to zero.
    fees: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    funding: float = 0.0
    quantization_alignment: float = 0.005
    risk_projection: float = 0.05
    gross_margin_risk: float = 0.05
    hedge_overlap: float = 0.0
    opportunity_cost: float = 0.0
    terminal_loss: float = 2.0
    gross_margin_soft_limit: float = 0.60
    reward_clip: float = 5.0

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in asdict(self).values())
        if not _finite(*values):
            raise HPRLConfigError("reward parameters must be finite")
        nonnegative = (
            self.return_scale,
            self.equity,
            self.drawdown,
            self.downside,
            self.cvar,
            self.turnover,
            self.fees,
            self.slippage,
            self.market_impact,
            self.funding,
            self.quantization_alignment,
            self.risk_projection,
            self.gross_margin_risk,
            self.hedge_overlap,
            self.opportunity_cost,
            self.terminal_loss,
            self.gross_margin_soft_limit,
            self.reward_clip,
        )
        if any(value < 0 for value in nonnegative):
            raise HPRLConfigError("reward weights/limits cannot be negative")
        if self.return_scale <= 0 or self.reward_clip <= 0:
            raise HPRLConfigError("return_scale and reward_clip must be positive")
        if not 0 <= self.gross_margin_soft_limit <= 1:
            raise HPRLConfigError("gross_margin_soft_limit must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class HPRLEnvironmentConfig:
    initial_equity: float = 1_000.0
    parallel_envs: int = 32
    annualization_periods: int = 525_600
    cvar_alpha: float = 0.05
    terminate_equity_ratio: float = 0.25
    # Disable per-step host-synchronizing tensor assertions in performance runs. Standalone
    # components remain strict by default when constructed directly.
    runtime_checks: bool = False
    # full preserves rich diagnostics; training emits only trainer-critical info.
    info_mode: str = "full"
    action: HPRLActionConfig = field(default_factory=HPRLActionConfig)
    costs: HPRLCostConfig = field(default_factory=HPRLCostConfig)
    reward: HPRLRewardConfig = field(default_factory=HPRLRewardConfig)

    def __post_init__(self) -> None:
        if not _finite(self.initial_equity, self.cvar_alpha, self.terminate_equity_ratio):
            raise HPRLConfigError("environment scalar parameters must be finite")
        if not _strict_int(self.parallel_envs) or not _strict_int(self.annualization_periods):
            raise HPRLConfigError("parallel_envs and annualization_periods must be integers")
        if not isinstance(self.runtime_checks, bool):
            raise HPRLConfigError("environment runtime_checks must be a boolean")
        info_mode = self.info_mode.strip().lower() if isinstance(self.info_mode, str) else ""
        if info_mode not in {"full", "training"}:
            raise HPRLConfigError("environment info_mode must be full/training")
        object.__setattr__(self, "info_mode", info_mode)
        if self.initial_equity <= 0 or self.parallel_envs < 1 or self.annualization_periods < 1:
            raise HPRLConfigError("environment dimensions must be positive")
        if not 0 < self.cvar_alpha <= 0.5:
            raise HPRLConfigError("cvar_alpha must be in (0, 0.5]")
        if not 0 < self.terminate_equity_ratio < 1:
            raise HPRLConfigError("terminate_equity_ratio must be in (0, 1)")


@dataclass(frozen=True, slots=True)
class HPRLTrainingConfig:
    algorithm: str = "xqc"
    seed: int = 42
    # auto prefers CUDA when a usable CUDA-enabled Torch runtime is present.
    device: str = "auto"
    # same keeps replay on the training device; cpu is useful when GPU memory is constrained.
    replay_device: str = "auto"
    batch_size: int = 1024
    replay_capacity: int = 1_000_000
    warmup_steps: int = 10_000
    gradient_steps: int = 1
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    hidden_dim: int = 512
    hidden_depth: int = 3
    gradient_clip_norm: float = 10.0
    mixed_precision: bool = False
    amp_dtype: str = "auto"
    allow_tf32: bool = True
    matmul_precision: str = "high"
    cudnn_benchmark: bool = False
    deterministic: bool = False
    pin_memory: bool = True
    # Expensive finite/range assertions can synchronize CUDA; enable when debugging data issues.
    runtime_checks: bool = False
    # Convert training metrics to host scalars only periodically to avoid per-update CUDA sync.
    metrics_interval: int = 100
    cuda_memory_fraction: float | None = None
    # Fraction of maximum categorical tier entropy targeted by SAC-family policies.
    tier_entropy_target_fraction: float = 0.65
    # Training-throughput controls. ``auto`` chooses fused Adam on CUDA and for-loop Adam on CPU.
    optimizer_backend: str = "auto"
    # ``auto`` uses device + expected update horizon + hardware profile.
    # Explicit modes always override the policy.
    compile_mode: str = "auto"
    # Expected optimizer updates for this training run. 0 means unknown/conservative.
    expected_updates: int = 0
    # auto detects known CUDA hardware; generic/cpu profiles remain conservative.
    hardware_profile: str = "auto"
    # compile_mode=auto uses cold thresholds by default. Set warm only when a compatible
    # TorchInductor/Triton disk cache is known to be pre-populated for this graph/device.
    compile_cache_state: str = "cold"
    compile_dynamic: bool = False
    compile_fullgraph: bool = False
    # RTX 5070 production auto uses validated loss-surface compilation for FastDSAC,
    # Simba-SAC and ReBRAC-v2. Generic hardware remains conservative at module scope.
    # ``loss_post`` is an experimental hardware-gated surface that also compiles target/
    # post-update tensor mutation and is never selected by auto until hardware acceptance.
    compile_scope: str = "auto"
    # 0 preserves the runtime/default intra-op thread count; positive values explicitly tune CPU.
    cpu_threads: int = 0
    # 0 = algorithm/hardware-aware host-dispatch policy; positive = explicit override.
    cpu_interop_threads: int = 0
    polyak_backend: str = "auto"
    grad_clip_backend: str = "auto"
    replay_prefetch: bool = True
    replay_prefetch_slots: int = 2
    replay_reuse_sample_buffers: bool = True
    return_scan_backend: str = "auto"
    # ReBRAC exact-likelihood policy. auto is hardware-aware: RTX 5070 Laptop uses FP32
    # after V2.0 hardware acceptance; generic CUDA may use mixed coupling MLPs.
    flow_likelihood_precision: str = "auto"
    # Experimental exact algebraic split of coupling first-layer obs/action projections.
    # Disabled until hardware profiling proves a benefit for the target batch/device.
    flow_obs_projection_reuse: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, str) or not self.algorithm.strip():
            raise HPRLConfigError("algorithm cannot be empty")
        if not isinstance(self.device, str) or not self.device.strip():
            raise HPRLConfigError("device cannot be empty")
        if not isinstance(self.replay_device, str) or not self.replay_device.strip():
            raise HPRLConfigError("replay_device cannot be empty")

        device = self.device.strip().lower()
        if device == "gpu":
            device = "cuda"
        valid_device = device in {"auto", "cpu", "cuda"} or (
            device.startswith("cuda:") and device[5:].isdigit()
        )
        if not valid_device:
            raise HPRLConfigError("device must be auto/cpu/cuda/gpu/cuda:<index>")
        replay = self.replay_device.strip().lower()
        if replay == "gpu":
            replay = "cuda"
        if not (
            replay in {"same", "auto", "cpu", "cuda"}
            or (replay.startswith("cuda:") and replay[5:].isdigit())
        ):
            raise HPRLConfigError("replay_device must be same/auto/cpu/cuda/gpu/cuda:<index>")

        integers = (
            self.seed,
            self.batch_size,
            self.replay_capacity,
            self.warmup_steps,
            self.gradient_steps,
            self.hidden_dim,
            self.hidden_depth,
            self.metrics_interval,
            self.cpu_threads,
            self.cpu_interop_threads,
            self.expected_updates,
            self.replay_prefetch_slots,
        )
        if any(not _strict_int(value) for value in integers):
            raise HPRLConfigError("training counters and dimensions must be integers")
        booleans = (
            self.mixed_precision,
            self.allow_tf32,
            self.cudnn_benchmark,
            self.deterministic,
            self.pin_memory,
            self.runtime_checks,
            self.compile_dynamic,
            self.compile_fullgraph,
            self.replay_prefetch,
            self.replay_reuse_sample_buffers,
            self.flow_obs_projection_reuse,
        )
        if any(not isinstance(value, bool) for value in booleans):
            raise HPRLConfigError("training acceleration flags must be booleans")
        amp_dtype = self.amp_dtype.strip().lower() if isinstance(self.amp_dtype, str) else ""
        if amp_dtype not in {"auto", "float16", "bfloat16"}:
            raise HPRLConfigError("amp_dtype must be auto/float16/bfloat16")
        if self.matmul_precision not in {"highest", "high", "medium"}:
            raise HPRLConfigError("matmul_precision must be highest/high/medium")
        optimizer_backend = (
            self.optimizer_backend.strip().lower()
            if isinstance(self.optimizer_backend, str)
            else ""
        )
        if optimizer_backend not in {"auto", "fused", "foreach", "for_loop"}:
            raise HPRLConfigError("optimizer_backend must be auto/fused/foreach/for_loop")
        polyak_backend = (
            self.polyak_backend.strip().lower()
            if isinstance(self.polyak_backend, str)
            else ""
        )
        if polyak_backend not in {"auto", "foreach", "for_loop"}:
            raise HPRLConfigError("polyak_backend must be auto/foreach/for_loop")
        grad_clip_backend = (
            self.grad_clip_backend.strip().lower()
            if isinstance(self.grad_clip_backend, str)
            else ""
        )
        if grad_clip_backend not in {"auto", "foreach", "for_loop"}:
            raise HPRLConfigError("grad_clip_backend must be auto/foreach/for_loop")
        return_scan_backend = (
            self.return_scan_backend.strip().lower()
            if isinstance(self.return_scan_backend, str)
            else ""
        )
        if return_scan_backend not in {"auto", "associative", "loop"}:
            raise HPRLConfigError("return_scan_backend must be auto/associative/loop")
        flow_likelihood_precision = (
            self.flow_likelihood_precision.strip().lower()
            if isinstance(self.flow_likelihood_precision, str)
            else ""
        )
        if flow_likelihood_precision not in {"auto", "fp32", "mixed"}:
            raise HPRLConfigError("flow_likelihood_precision must be auto/fp32/mixed")
        compile_mode = (
            self.compile_mode.strip().lower() if isinstance(self.compile_mode, str) else ""
        )
        compile_cache_state = (
            self.compile_cache_state.strip().lower()
            if isinstance(self.compile_cache_state, str)
            else ""
        )
        compile_scope = (
            self.compile_scope.strip().lower() if isinstance(self.compile_scope, str) else ""
        )
        hardware_profile = (
            self.hardware_profile.strip().lower() if isinstance(self.hardware_profile, str) else ""
        )
        if compile_cache_state not in {"auto", "cold", "warm"}:
            raise HPRLConfigError("compile_cache_state must be auto/cold/warm")
        if compile_scope not in {"auto", "module", "loss", "loss_post", "xqc_fused"}:
            raise HPRLConfigError("compile_scope must be auto/module/loss/loss_post/xqc_fused")
        if hardware_profile not in {"auto", "cpu", "generic_cuda", "rtx5070_laptop"}:
            raise HPRLConfigError(
                "hardware_profile must be auto/cpu/generic_cuda/rtx5070_laptop"
            )
        if compile_mode not in {
            "off", "auto", "default", "reduce-overhead", "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise HPRLConfigError(
                "compile_mode must be off/auto/default/reduce-overhead/max-autotune/"
                "max-autotune-no-cudagraphs"
            )
        object.__setattr__(self, "optimizer_backend", optimizer_backend)
        object.__setattr__(self, "polyak_backend", polyak_backend)
        object.__setattr__(self, "grad_clip_backend", grad_clip_backend)
        object.__setattr__(self, "return_scan_backend", return_scan_backend)
        object.__setattr__(self, "flow_likelihood_precision", flow_likelihood_precision)
        object.__setattr__(self, "compile_mode", compile_mode)
        object.__setattr__(self, "compile_cache_state", compile_cache_state)
        object.__setattr__(self, "compile_scope", compile_scope)
        object.__setattr__(self, "hardware_profile", hardware_profile)
        if self.cpu_threads < 0 or self.cpu_interop_threads < 0 or self.expected_updates < 0:
            raise HPRLConfigError(
                "cpu_threads/cpu_interop_threads/expected_updates must be >= 0"
            )
        if not 2 <= self.replay_prefetch_slots <= 4:
            raise HPRLConfigError("replay_prefetch_slots must be within [2, 4]")
        if self.seed < 0:
            raise HPRLConfigError("seed must be non-negative")
        if self.batch_size < 1 or self.replay_capacity < self.batch_size:
            raise HPRLConfigError("replay capacity must be at least batch_size")
        if self.warmup_steps < 0 or self.gradient_steps < 1 or self.metrics_interval < 1:
            raise HPRLConfigError("invalid training schedule")
        scalars = (
            self.gamma,
            self.tau,
            self.learning_rate,
            self.gradient_clip_norm,
            self.tier_entropy_target_fraction,
        )
        if not _finite(*scalars):
            raise HPRLConfigError("training scalar parameters must be finite")
        if not 0 < self.gamma <= 1 or not 0 < self.tau <= 1:
            raise HPRLConfigError("gamma/tau are outside their valid ranges")
        if self.learning_rate <= 0 or self.hidden_dim < 16 or self.hidden_depth < 1:
            raise HPRLConfigError("invalid optimizer/network dimensions")
        if self.gradient_clip_norm <= 0:
            raise HPRLConfigError("gradient_clip_norm must be positive")
        if not 0 <= self.tier_entropy_target_fraction <= 1:
            raise HPRLConfigError("tier_entropy_target_fraction must be in [0, 1]")
        if self.cuda_memory_fraction is not None:
            if not math.isfinite(float(self.cuda_memory_fraction)):
                raise HPRLConfigError("cuda_memory_fraction must be finite")
            if not 0 < float(self.cuda_memory_fraction) <= 1:
                raise HPRLConfigError("cuda_memory_fraction must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class HPRLMemoryConfig:
    """CPU/CUDA memory budget and long-history staging policy."""

    dataset_mode: str = "auto"
    dataset_window_steps: int = 16_384
    dataset_gpu_fraction: float = 0.20
    replay_gpu_fraction: float = 0.30
    cuda_reserve_fraction: float = 0.25
    min_cuda_reserve_bytes: int = 768 * 1024 * 1024
    pin_staging_memory: bool = True
    strict_budget: bool = False
    phase_boundary_cleanup: bool = True
    offline_tensorize_chunk_rows: int = 4096
    release_offline_source_after_tensorize: bool = False

    def __post_init__(self) -> None:
        mode = self.dataset_mode.strip().lower() if isinstance(self.dataset_mode, str) else ""
        if mode not in {"auto", "resident", "windowed"}:
            raise HPRLConfigError("dataset_mode must be auto/resident/windowed")
        if not _strict_int(self.dataset_window_steps) or self.dataset_window_steps < 2:
            raise HPRLConfigError("dataset_window_steps must be an integer >= 2")
        if not _strict_int(self.min_cuda_reserve_bytes) or self.min_cuda_reserve_bytes < 0:
            raise HPRLConfigError("min_cuda_reserve_bytes must be a non-negative integer")
        fractions = (
            self.dataset_gpu_fraction,
            self.replay_gpu_fraction,
            self.cuda_reserve_fraction,
        )
        if not _finite(*fractions):
            raise HPRLConfigError("memory fractions must be finite")
        if not 0 < self.dataset_gpu_fraction <= 1:
            raise HPRLConfigError("dataset_gpu_fraction must be in (0, 1]")
        if not 0 < self.replay_gpu_fraction <= 1:
            raise HPRLConfigError("replay_gpu_fraction must be in (0, 1]")
        if not 0 <= self.cuda_reserve_fraction < 1:
            raise HPRLConfigError("cuda_reserve_fraction must be in [0, 1)")
        if self.dataset_gpu_fraction + self.replay_gpu_fraction > 0.90:
            raise HPRLConfigError(
                "dataset_gpu_fraction + replay_gpu_fraction must be <= 0.90 to leave model headroom"
            )
        if (
            not _strict_int(self.offline_tensorize_chunk_rows)
            or self.offline_tensorize_chunk_rows < 1
        ):
            raise HPRLConfigError("offline_tensorize_chunk_rows must be a positive integer")
        flags = (
            self.pin_staging_memory,
            self.strict_budget,
            self.phase_boundary_cleanup,
            self.release_offline_source_after_tensorize,
        )
        if any(not isinstance(value, bool) for value in flags):
            raise HPRLConfigError("memory policy flags must be booleans")


@dataclass(frozen=True, slots=True)
class HPRLConfig:
    environment: HPRLEnvironmentConfig = field(default_factory=HPRLEnvironmentConfig)
    training: HPRLTrainingConfig = field(default_factory=HPRLTrainingConfig)
    memory: HPRLMemoryConfig = field(default_factory=HPRLMemoryConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "HPRLConfig":
        if not isinstance(values, Mapping):
            raise HPRLConfigError("HPRL configuration root must be a mapping")
        env_value = values.get("environment", {})
        training_value = values.get("training", {})
        memory_value = values.get("memory", {})
        if not isinstance(env_value, Mapping) or not isinstance(training_value, Mapping):
            raise HPRLConfigError("environment and training configuration must be mappings")
        if not isinstance(memory_value, Mapping):
            raise HPRLConfigError("memory configuration must be a mapping")
        env_raw = dict(env_value)
        action_value = env_raw.pop("action", {})
        costs_value = env_raw.pop("costs", {})
        reward_value = env_raw.pop("reward", {})
        nested_values = (action_value, costs_value, reward_value)
        if not all(isinstance(value, Mapping) for value in nested_values):
            raise HPRLConfigError("nested environment configuration must use mappings")
        try:
            env = HPRLEnvironmentConfig(
                **env_raw,
                action=HPRLActionConfig(**dict(action_value)),
                costs=HPRLCostConfig(**dict(costs_value)),
                reward=HPRLRewardConfig(**dict(reward_value)),
            )
            training = HPRLTrainingConfig(**dict(training_value))
            memory = HPRLMemoryConfig(**dict(memory_value))
        except TypeError as exc:
            raise HPRLConfigError(f"invalid HPRL configuration key/value: {exc}") from exc
        return cls(environment=env, training=training, memory=memory)
