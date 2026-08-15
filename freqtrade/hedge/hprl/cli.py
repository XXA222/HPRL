"""Standalone HPRL CLI; it does not modify Freqtrade's root CLI registry."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from . import HPRL_API_VERSION, HPRL_DEVICE_POLICY, HPRL_RELEASE
from .compatibility import inspect_clean_mainline
from .device import cuda_memory_stats, require_torch, resolve_device, synchronize_device
from .registry import available_algorithms
from .config import HPRLActionConfig, HPRLRewardConfig, HPRLTrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m freqtrade.hedge.hprl")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect", help="show HPRL capabilities without initializing a model")
    device = sub.add_parser("device", help="resolve cpu/cuda/auto execution device")
    device.add_argument("--device", default="auto")
    memory = sub.add_parser("memory", help="inspect CPU/CUDA memory budgets and placement")
    memory.add_argument("--device", default="auto")
    memory.add_argument("--dataset-bytes", type=int, default=0)
    memory.add_argument("--obs-dim", type=int, default=64)
    memory.add_argument("--action-dim", type=int, default=8)
    memory.add_argument("--replay-capacity", type=int, default=1_000_000)
    memory.add_argument("--replay-device", default="auto")
    compat = sub.add_parser("compat", help="check Clean Mainline integration contracts")
    compat.add_argument("--project-root", default=".")
    smoke = sub.add_parser("smoke", help="run dependency/device tensor smoke checks")
    smoke.add_argument("--device", default="auto")
    train_smoke = sub.add_parser(
        "train-smoke",
        help="run a real environment/replay/gradient-update smoke test on the selected device",
    )
    train_smoke.add_argument("--device", default="auto")
    train_smoke.add_argument("--algorithm", default="fast_td3", choices=available_algorithms())
    train_smoke.add_argument("--mixed-precision", action="store_true")
    train_smoke.add_argument("--optimizer-backend", default="auto")
    train_smoke.add_argument("--polyak-backend", default="auto")
    train_smoke.add_argument("--grad-clip-backend", default="auto")
    train_smoke.add_argument("--compile-mode", default="auto")
    train_smoke.add_argument("--compile-scope", default="auto", choices=("auto", "module", "loss", "loss_post", "xqc_fused"))
    train_smoke.add_argument("--expected-updates", type=int, default=0)
    train_smoke.add_argument("--hardware-profile", default="auto")
    train_smoke.add_argument("--compile-cache-state", default="cold", choices=("auto", "cold", "warm"))
    train_smoke.add_argument("--flow-likelihood-precision", default="auto")
    train_smoke.add_argument("--flow-obs-projection-reuse", action="store_true")
    train_smoke.add_argument("--cpu-threads", type=int, default=0)
    train_smoke.add_argument("--cpu-interop-threads", type=int, default=0)
    train_smoke.add_argument(
        "--replay-device",
        default="same",
        help="same/cpu/cuda/cuda:<index>; CPU replay is useful when VRAM is constrained",
    )
    perf = sub.add_parser("perf-benchmark", help="benchmark real HPRL gradient-update throughput")
    perf.add_argument("--device", default="auto")
    perf.add_argument("--algorithm", default="fast_td3", choices=available_algorithms())
    perf.add_argument("--batch-size", type=int, default=1024)
    perf.add_argument("--hidden-dim", type=int, default=512)
    perf.add_argument("--hidden-depth", type=int, default=3)
    perf.add_argument("--warmup", type=int, default=10)
    perf.add_argument("--iterations", type=int, default=50)
    perf.add_argument("--optimizer-backend", default="auto")
    perf.add_argument("--polyak-backend", default="auto")
    perf.add_argument("--grad-clip-backend", default="auto")
    perf.add_argument("--compile-mode", default="off")
    perf.add_argument("--compile-scope", default="auto", choices=("auto", "module", "loss", "loss_post", "xqc_fused"))
    perf.add_argument("--expected-updates", type=int, default=0)
    perf.add_argument("--hardware-profile", default="auto")
    perf.add_argument("--compile-cache-state", default="cold", choices=("auto", "cold", "warm"))
    perf.add_argument("--flow-likelihood-precision", default="auto")
    perf.add_argument("--flow-obs-projection-reuse", action="store_true")
    perf.add_argument("--mixed-precision", action="store_true")
    perf.add_argument("--cpu-threads", type=int, default=0)
    perf.add_argument("--cpu-interop-threads", type=int, default=0)
    perf.add_argument("--obs-dim", type=int, default=32)
    perf.add_argument("--action-dim", type=int, default=4)
    perf.add_argument("--gpu-condition-ms", type=int, default=0)
    perf.add_argument("--gpu-condition-matrix-size", type=int, default=512)

    pipeline = sub.add_parser(
        "perf-pipeline-benchmark",
        help="benchmark replay->H2D->update/target->logging/checkpoint training throughput",
    )
    pipeline.add_argument("--device", default="auto")
    pipeline.add_argument("--algorithm", default="fast_td3", choices=available_algorithms())
    pipeline.add_argument("--batch-size", type=int, default=1024)
    pipeline.add_argument("--hidden-dim", type=int, default=256)
    pipeline.add_argument("--hidden-depth", type=int, default=2)
    pipeline.add_argument("--warmup", type=int, default=20)
    pipeline.add_argument("--iterations", type=int, default=200)
    pipeline.add_argument("--replay-capacity", type=int, default=16384)
    pipeline.add_argument("--replay-device", default="auto")
    pipeline.add_argument("--prefetch-slots", type=int, default=2)
    pipeline.add_argument("--metrics-interval", type=int, default=100)
    pipeline.add_argument("--checkpoint-interval", type=int, default=0)
    pipeline.add_argument("--diagnostic-iterations", type=int, default=3)
    pipeline.add_argument("--optimizer-backend", default="auto")
    pipeline.add_argument("--polyak-backend", default="auto")
    pipeline.add_argument("--grad-clip-backend", default="auto")
    pipeline.add_argument("--compile-mode", default="auto")
    pipeline.add_argument("--compile-scope", default="auto", choices=("auto", "module", "loss", "loss_post", "xqc_fused"))
    pipeline.add_argument("--compile-cache-state", default="cold", choices=("auto", "cold", "warm"))
    pipeline.add_argument("--hardware-profile", default="auto")
    pipeline.add_argument("--flow-likelihood-precision", default="auto")
    pipeline.add_argument("--flow-obs-projection-reuse", action="store_true")
    pipeline.add_argument("--mixed-precision", action="store_true")
    pipeline.add_argument("--cpu-threads", type=int, default=0)
    pipeline.add_argument("--cpu-interop-threads", type=int, default=0)
    pipeline.add_argument("--obs-dim", type=int, default=32)
    pipeline.add_argument("--action-dim", type=int, default=4)
    pipeline.add_argument("--gpu-condition-ms", type=int, default=0)
    pipeline.add_argument("--artifact-io-mode", default="auto", choices=("auto", "sync", "async"))
    pipeline.add_argument("--sync-artifacts", action="store_true", help="compatibility override for --artifact-io-mode sync")
    pipeline.add_argument("--artifact-queue-size", type=int, default=8)
    pipeline.add_argument("--estimated-log-bytes-per-event", type=int, default=1024)
    pipeline.add_argument("--prior-artifact-block-ratio", type=float, default=0.0)
    pipeline.add_argument("--checkpoint-live-device-snapshot", action="store_true")
    pipeline.add_argument("--skip-micro-reference", action="store_true")
    xqc_decomp = sub.add_parser(
        "perf-xqc-decomposition",
        help="attribute XQC update stages with CUDA Events/NVTX without perturbing production",
    )
    xqc_decomp.add_argument("--device", default="auto")
    xqc_decomp.add_argument("--batch-size", type=int, default=1024)
    xqc_decomp.add_argument("--hidden-dim", type=int, default=256)
    xqc_decomp.add_argument("--hidden-depth", type=int, default=2)
    xqc_decomp.add_argument("--iterations", type=int, default=9)
    xqc_decomp.add_argument("--compile-mode", default="reduce-overhead")
    xqc_decomp.add_argument("--compile-scope", default="module", choices=("auto", "module", "loss", "loss_post", "xqc_fused"))
    xqc_decomp.add_argument("--compile-cache-state", default="warm", choices=("auto", "cold", "warm"))
    xqc_decomp.add_argument("--hardware-profile", default="rtx5070_laptop")
    xqc_decomp.add_argument("--cpu-threads", type=int, default=0)
    xqc_decomp.add_argument("--cpu-interop-threads", type=int, default=0)
    xqc_decomp.add_argument("--mixed-precision", action="store_true")
    xqc_decomp.add_argument("--obs-dim", type=int, default=32)
    xqc_decomp.add_argument("--action-dim", type=int, default=4)

    sustained = sub.add_parser(
        "perf-sustained-pipeline",
        help="run a long end-to-end HPRL training pipeline stability benchmark",
    )
    sustained.add_argument("--device", default="auto")
    sustained.add_argument("--algorithm", default="fast_td3", choices=available_algorithms())
    sustained.add_argument("--batch-size", type=int, default=1024)
    sustained.add_argument("--hidden-dim", type=int, default=256)
    sustained.add_argument("--hidden-depth", type=int, default=2)
    sustained.add_argument("--warmup", type=int, default=50)
    sustained.add_argument("--iterations", type=int, default=5000)
    sustained.add_argument("--window-size", type=int, default=500)
    sustained.add_argument("--replay-capacity", type=int, default=16384)
    sustained.add_argument("--replay-device", default="cpu")
    sustained.add_argument("--prefetch-slots", type=int, default=2)
    sustained.add_argument("--metrics-interval", type=int, default=100)
    sustained.add_argument("--checkpoint-interval", type=int, default=1000)
    sustained.add_argument("--checkpoint-keep-last", type=int, default=2)
    sustained.add_argument("--artifact-queue-size", type=int, default=8)
    sustained.add_argument("--artifact-io-mode", default="auto", choices=("auto", "sync", "async"))
    sustained.add_argument("--estimated-log-bytes-per-event", type=int, default=1024)
    sustained.add_argument("--prior-artifact-block-ratio", type=float, default=0.0)
    sustained.add_argument("--compile-mode", default="auto")
    sustained.add_argument("--compile-scope", default="auto", choices=("auto", "module", "loss", "loss_post", "xqc_fused"))
    sustained.add_argument("--compile-cache-state", default="warm", choices=("auto", "cold", "warm"))
    sustained.add_argument("--hardware-profile", default="auto")
    sustained.add_argument("--optimizer-backend", default="auto")
    sustained.add_argument("--polyak-backend", default="auto")
    sustained.add_argument("--grad-clip-backend", default="auto")
    sustained.add_argument("--flow-likelihood-precision", default="auto")
    sustained.add_argument("--mixed-precision", action="store_true")
    sustained.add_argument("--cpu-threads", type=int, default=0)
    sustained.add_argument("--cpu-interop-threads", type=int, default=0)
    sustained.add_argument("--obs-dim", type=int, default=32)
    sustained.add_argument("--action-dim", type=int, default=4)
    profile = sub.add_parser("perf-profile", help="profile real HPRL update hot operators")
    profile.add_argument("--device", default="auto")
    profile.add_argument("--algorithm", default="fast_td3", choices=available_algorithms())
    profile.add_argument("--batch-size", type=int, default=1024)
    profile.add_argument("--hidden-dim", type=int, default=256)
    profile.add_argument("--hidden-depth", type=int, default=2)
    profile.add_argument("--optimizer-backend", default="auto")
    profile.add_argument("--polyak-backend", default="auto")
    profile.add_argument("--grad-clip-backend", default="auto")
    profile.add_argument("--compile-mode", default="off")
    profile.add_argument("--compile-scope", default="auto", choices=("auto", "module", "loss", "loss_post", "xqc_fused"))
    profile.add_argument("--expected-updates", type=int, default=0)
    profile.add_argument("--hardware-profile", default="auto")
    profile.add_argument("--compile-cache-state", default="cold", choices=("auto", "cold", "warm"))
    profile.add_argument("--flow-likelihood-precision", default="auto")
    profile.add_argument("--flow-obs-projection-reuse", action="store_true")
    profile.add_argument("--mixed-precision", action="store_true")
    profile.add_argument("--cpu-threads", type=int, default=0)
    profile.add_argument("--cpu-interop-threads", type=int, default=0)
    profile.add_argument("--active", type=int, default=3)
    profile.add_argument("--row-limit", type=int, default=30)

    orchestration = sub.add_parser(
        "perf-orchestration-profile",
        help="profile kernel-launch/Python-orchestration surfaces for one HPRL update path",
    )
    orchestration.add_argument("--device", default="cuda")
    orchestration.add_argument("--algorithm", default="fast_dsac", choices=available_algorithms())
    orchestration.add_argument("--batch-size", type=int, default=1024)
    orchestration.add_argument("--hidden-dim", type=int, default=256)
    orchestration.add_argument("--hidden-depth", type=int, default=2)
    orchestration.add_argument("--optimizer-backend", default="auto")
    orchestration.add_argument("--polyak-backend", default="auto")
    orchestration.add_argument("--grad-clip-backend", default="auto")
    orchestration.add_argument("--compile-mode", default="reduce-overhead")
    orchestration.add_argument("--compile-scope", default="module", choices=("module", "loss", "loss_post", "xqc_fused"))
    orchestration.add_argument("--compile-cache-state", default="warm", choices=("cold", "warm"))
    orchestration.add_argument("--hardware-profile", default="rtx5070_laptop")
    orchestration.add_argument("--flow-likelihood-precision", default="auto")
    orchestration.add_argument("--mixed-precision", action="store_true")
    orchestration.add_argument("--cpu-threads", type=int, default=16)
    orchestration.add_argument("--cpu-interop-threads", type=int, default=1)
    orchestration.add_argument("--active", type=int, default=5)
    orchestration.add_argument("--gpu-condition-ms", type=int, default=350)

    compile_sweep = sub.add_parser(
        "perf-compile-sweep",
        help="compare eager vs reduce-overhead in isolated subprocesses",
    )
    compile_sweep.add_argument("--device", default="cuda")
    compile_sweep.add_argument("--algorithm", default="simba_sac", choices=available_algorithms())
    compile_sweep.add_argument("--batch-size", type=int, default=1024)
    compile_sweep.add_argument("--hidden-dim", type=int, default=256)
    compile_sweep.add_argument("--hidden-depth", type=int, default=2)
    compile_sweep.add_argument("--warmup", type=int, default=50)
    compile_sweep.add_argument("--iterations", type=int, default=200)
    compile_sweep.add_argument("--optimizer-backend", default="auto")
    compile_sweep.add_argument("--polyak-backend", default="auto")
    compile_sweep.add_argument("--grad-clip-backend", default="auto")
    compile_sweep.add_argument("--flow-likelihood-precision", default="auto")
    compile_sweep.add_argument("--flow-obs-projection-reuse", action="store_true")
    compile_sweep.add_argument("--mixed-precision", action="store_true")
    compile_sweep.add_argument("--cpu-threads", type=int, default=0)
    compile_sweep.add_argument("--cpu-interop-threads", type=int, default=0)
    compile_sweep.add_argument("--obs-dim", type=int, default=32)
    compile_sweep.add_argument("--action-dim", type=int, default=4)
    compile_sweep.add_argument("--min-speedup", type=float, default=1.05)

    backend_sweep = sub.add_parser(
        "perf-backend-sweep",
        help="benchmark Polyak/grad-clip backend combinations in isolated subprocesses",
    )
    backend_sweep.add_argument("--device", default="cuda")
    backend_sweep.add_argument("--algorithm", default="simba_sac", choices=available_algorithms())
    backend_sweep.add_argument("--batch-size", type=int, default=1024)
    backend_sweep.add_argument("--hidden-dim", type=int, default=256)
    backend_sweep.add_argument("--hidden-depth", type=int, default=2)
    backend_sweep.add_argument("--warmup", type=int, default=30)
    backend_sweep.add_argument("--iterations", type=int, default=150)
    backend_sweep.add_argument("--optimizer-backend", default="auto")
    backend_sweep.add_argument("--compile-mode", default="off")
    backend_sweep.add_argument("--flow-likelihood-precision", default="auto")
    backend_sweep.add_argument("--flow-obs-projection-reuse", action="store_true")
    backend_sweep.add_argument("--mixed-precision", action="store_true")
    backend_sweep.add_argument("--cpu-threads", type=int, default=0)
    backend_sweep.add_argument("--cpu-interop-threads", type=int, default=0)
    backend_sweep.add_argument("--obs-dim", type=int, default=32)
    backend_sweep.add_argument("--action-dim", type=int, default=4)

    host_sweep = sub.add_parser(
        "perf-host-sweep",
        help="benchmark GPU host-dispatch inter-op thread counts in isolated subprocesses",
    )
    host_sweep.add_argument("--device", default="cuda")
    host_sweep.add_argument("--algorithm", default="simba_sac", choices=available_algorithms())
    host_sweep.add_argument("--batch-size", type=int, default=1024)
    host_sweep.add_argument("--hidden-dim", type=int, default=256)
    host_sweep.add_argument("--hidden-depth", type=int, default=2)
    host_sweep.add_argument("--warmup", type=int, default=30)
    host_sweep.add_argument("--iterations", type=int, default=150)
    host_sweep.add_argument("--optimizer-backend", default="auto")
    host_sweep.add_argument("--polyak-backend", default="auto")
    host_sweep.add_argument("--grad-clip-backend", default="auto")
    host_sweep.add_argument("--compile-mode", default="off")
    host_sweep.add_argument("--flow-likelihood-precision", default="auto")
    host_sweep.add_argument("--flow-obs-projection-reuse", action="store_true")
    host_sweep.add_argument("--mixed-precision", action="store_true")
    host_sweep.add_argument("--cpu-threads", type=int, default=16)
    host_sweep.add_argument("--obs-dim", type=int, default=32)
    host_sweep.add_argument("--action-dim", type=int, default=4)
    host_sweep.add_argument(
        "--interop-candidates",
        default="1,2,4,8,16,32",
        help="comma-separated positive inter-op thread counts",
    )

    flow_profile = sub.add_parser(
        "perf-flow-profile",
        help="profile ReBRAC flow forward/inverse likelihood surfaces",
    )
    flow_profile.add_argument("--device", default="auto")
    flow_profile.add_argument("--batch-size", type=int, default=1024)
    flow_profile.add_argument("--obs-dim", type=int, default=32)
    flow_profile.add_argument("--action-dim", type=int, default=4)
    flow_profile.add_argument("--hidden-dim", type=int, default=256)
    flow_profile.add_argument("--flow-layers", type=int, default=4)
    flow_profile.add_argument("--warmup", type=int, default=20)
    flow_profile.add_argument("--iterations", type=int, default=100)
    flow_profile.add_argument("--mixed-precision", action="store_true")
    flow_profile.add_argument("--flow-likelihood-precision", default="auto")
    flow_profile.add_argument("--row-limit", type=int, default=20)

    flow_sweep = sub.add_parser(
        "perf-flow-sweep",
        help="compare ReBRAC flow likelihood precision/projection modes in isolated subprocesses",
    )
    flow_sweep.add_argument("--device", default="cuda")
    flow_sweep.add_argument("--batch-size", type=int, default=1024)
    flow_sweep.add_argument("--hidden-dim", type=int, default=256)
    flow_sweep.add_argument("--hidden-depth", type=int, default=2)
    flow_sweep.add_argument("--warmup", type=int, default=50)
    flow_sweep.add_argument("--iterations", type=int, default=200)
    flow_sweep.add_argument("--optimizer-backend", default="auto")
    flow_sweep.add_argument("--polyak-backend", default="auto")
    flow_sweep.add_argument("--grad-clip-backend", default="auto")
    flow_sweep.add_argument("--compile-mode", default="off")
    flow_sweep.add_argument("--cpu-threads", type=int, default=0)
    flow_sweep.add_argument("--cpu-interop-threads", type=int, default=0)
    flow_sweep.add_argument("--obs-dim", type=int, default=32)
    flow_sweep.add_argument("--action-dim", type=int, default=4)
    flow_sweep.add_argument("--min-speedup", type=float, default=1.02)
    return parser


def _train_smoke(args) -> dict[str, object]:
    from .config import HPRLConfig, HPRLEnvironmentConfig, HPRLTrainingConfig
    from .data import TensorMarketDataset
    from .runtime import build_online_runtime

    torch = require_torch()
    info = resolve_device(args.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)
    features = torch.randn((32, 2, 8), generator=generator, dtype=torch.float32) * 0.01
    returns = torch.randn((32, 2), generator=generator, dtype=torch.float32) * 0.001
    dataset = TensorMarketDataset(
        features=features,
        forward_returns=returns,
        symbols=("BTC/USDT:USDT", "ETH/USDT:USDT"),
    ).validate()
    training = HPRLTrainingConfig(
        algorithm=args.algorithm,
        device=args.device,
        replay_device=args.replay_device,
        batch_size=16,
        replay_capacity=256,
        warmup_steps=16,
        gradient_steps=1,
        hidden_dim=64,
        hidden_depth=2,
        mixed_precision=bool(args.mixed_precision),
        optimizer_backend=args.optimizer_backend,
        polyak_backend=args.polyak_backend,
        grad_clip_backend=args.grad_clip_backend,
        compile_mode=args.compile_mode,
        compile_scope=getattr(args, "compile_scope", "auto"),
        expected_updates=getattr(args, "expected_updates", 0),
        hardware_profile=getattr(args, "hardware_profile", "auto"),
        compile_cache_state=getattr(args, "compile_cache_state", "cold"),
        cpu_threads=args.cpu_threads,
        cpu_interop_threads=args.cpu_interop_threads,
        flow_likelihood_precision=args.flow_likelihood_precision,
        flow_obs_projection_reuse=bool(args.flow_obs_projection_reuse),
    )
    config = HPRLConfig(
        environment=HPRLEnvironmentConfig(
            parallel_envs=8,
            info_mode="training",
            action=HPRLActionConfig(),
        ),
        training=training,
    )
    runtime = build_online_runtime(dataset, config)
    summary = runtime.trainer.run(6)
    synchronize_device(info.resolved)
    env_device = str(runtime.env.device)
    agent_device = str(runtime.agent.device)
    replay_device = str(runtime.trainer.buffer.device)
    if env_device != info.resolved or agent_device != info.resolved:
        raise RuntimeError(
            "HPRL train-smoke device residency mismatch: "
            f"resolved={info.resolved}, env={env_device}, agent={agent_device}"
        )
    if args.replay_device == "same" and replay_device != info.resolved:
        raise RuntimeError(
            f"HPRL train-smoke replay residency mismatch: {replay_device} != {info.resolved}"
        )
    return {
        "status": "PASS",
        "algorithm": args.algorithm,
        "resolved_device": info.resolved,
        "device_name": info.device_name,
        "environment_device": env_device,
        "agent_device": agent_device,
        "replay_device": replay_device,
        "mixed_precision": training.mixed_precision,
        "amp_dtype": getattr(runtime.agent.precision, "dtype_name", "float32"),
        "transitions": summary.transitions,
        "updates": summary.updates,
        "dataset_memory_plan": asdict(runtime.env.market.plan),
        "replay_memory_plan": asdict(runtime.trainer.replay_plan),
        "replay_persistent_bytes": runtime.trainer.buffer.persistent_bytes,
        "replay_pinned_stage_bytes": runtime.trainer.buffer.pinned_stage_bytes,
        "replay_sample_stage_bytes": runtime.trainer.buffer.sample_stage_bytes,
        "action_mode": runtime.env.action_mode,
        "position_levels": list(runtime.env.config.action.position_levels),
        "joint_states_per_symbol": runtime.env.joint_action_states_per_symbol,
        "reward_return_scale": runtime.env.config.reward.return_scale,
        "optimizer_backend": getattr(runtime.agent.actor_opt, "_hprl_backend", "unknown"),
        "polyak_backend": runtime.agent.performance_info.polyak_backend,
        "grad_clip_backend": runtime.agent.performance_info.grad_clip_backend,
        "host_dispatch_tuned": runtime.agent.performance_info.host_dispatch_tuned,
        "compile_request": training.compile_mode,
        "compile_mode": runtime.agent.performance_info.compile_mode,
        "hardware_profile": runtime.agent.performance_info.hardware_profile,
        "compile_cache_state": runtime.agent.performance_info.compile_cache_state,
        "compile_cold_break_even_updates": runtime.agent.performance_info.compile_cold_break_even_updates,
        "compile_warm_break_even_updates": runtime.agent.performance_info.compile_warm_break_even_updates,
        "expected_updates": runtime.agent.performance_info.expected_updates,
        "compile_break_even_updates": runtime.agent.performance_info.compile_break_even_updates,
        "compiled_hotpaths": list(getattr(runtime.agent, "compiled_hotpaths", ())),
        "flow_likelihood_precision": getattr(
            runtime.agent, "flow_likelihood_precision", "not_applicable"
        ),
        "replay_prefetch": runtime.trainer.replay_prefetcher is not None,
        "replay_prefetch_device_stage_bytes": (
            0
            if runtime.trainer.replay_prefetcher is None
            else runtime.trainer.replay_prefetcher.device_stage_bytes
        ),
        "replay_prefetch_depth": (
            0
            if runtime.trainer.replay_prefetcher is None
            else runtime.trainer.replay_prefetcher.prefetch_depth
        ),
        "cuda_memory": cuda_memory_stats(info.resolved),
    }


def _perf_benchmark(args) -> dict[str, object]:
    from .action_space import configure_agent_action_levels
    from .performance import agent_finite_state, condition_cuda_device, prepare_steady_state_agent, timed_iterations_detailed
    from .registry import create_agent
    from .replay import ReplayBatch

    if args.batch_size < 1 or args.hidden_dim < 16 or args.hidden_depth < 1:
        raise ValueError("benchmark dimensions must be positive")
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("benchmark warmup/iterations are invalid")
    torch = require_torch()
    info = resolve_device(args.device)
    config = HPRLTrainingConfig(
        algorithm=args.algorithm,
        device=args.device,
        replay_device="same",
        batch_size=args.batch_size,
        replay_capacity=max(args.batch_size * 2, 256),
        warmup_steps=0,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        mixed_precision=bool(args.mixed_precision),
        optimizer_backend=args.optimizer_backend,
        polyak_backend=args.polyak_backend,
        grad_clip_backend=args.grad_clip_backend,
        compile_mode=args.compile_mode,
        compile_scope=getattr(args, "compile_scope", "auto"),
        expected_updates=getattr(args, "expected_updates", 0),
        hardware_profile=getattr(args, "hardware_profile", "auto"),
        compile_cache_state=getattr(args, "compile_cache_state", "cold"),
        cpu_threads=args.cpu_threads,
        cpu_interop_threads=args.cpu_interop_threads,
        flow_likelihood_precision=args.flow_likelihood_precision,
        flow_obs_projection_reuse=bool(args.flow_obs_projection_reuse),
        metrics_interval=max(args.iterations + args.warmup + 1, 1000),
    )
    action_dim = int(args.action_dim)
    obs_dim = int(args.obs_dim)
    agent = create_agent(args.algorithm, obs_dim, action_dim, config, device=info.resolved)
    configure_agent_action_levels(agent, HPRLActionConfig().level_count)
    batch = ReplayBatch(
        obs=torch.randn(args.batch_size, obs_dim, device=info.resolved),
        action=torch.rand(args.batch_size, action_dim, device=info.resolved),
        reward=torch.randn(args.batch_size, 1, device=info.resolved) * 0.01,
        next_obs=torch.randn(args.batch_size, obs_dim, device=info.resolved),
        done=torch.zeros(args.batch_size, 1, device=info.resolved),
    )
    staged_warmup_updates = prepare_steady_state_agent(agent)
    conditioning = condition_cuda_device(
        info.resolved,
        milliseconds=getattr(args, "gpu_condition_ms", 0),
        matrix_size=getattr(args, "gpu_condition_matrix_size", 512),
    )
    result = timed_iterations_detailed(
        lambda: agent.update(batch, collect_metrics=False),
        warmup=args.warmup,
        iterations=args.iterations,
        device=info.resolved,
        samples_per_iteration=args.batch_size,
    )
    finite_state = agent_finite_state(agent)
    result.update(
        {
            "schema": "hprl-performance-benchmark-v2",
            "staged_warmup_updates": staged_warmup_updates,
            "benchmark_stage": "steady_state",
            "algorithm": args.algorithm,
            "device": info.resolved,
            "device_name": info.device_name,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "hidden_depth": args.hidden_depth,
            "mixed_precision": config.mixed_precision,
            "amp_dtype": getattr(agent.precision, "dtype_name", "float32"),
            "optimizer_backend": getattr(agent.actor_opt, "_hprl_backend", "unknown"),
            "polyak_backend": agent.performance_info.polyak_backend,
            "grad_clip_backend": agent.performance_info.grad_clip_backend,
            "host_dispatch_tuned": agent.performance_info.host_dispatch_tuned,
            "compile_request": config.compile_mode,
            "compile_mode": agent.performance_info.compile_mode,
            "hardware_profile": agent.performance_info.hardware_profile,
            "compile_cache_state": agent.performance_info.compile_cache_state,
            "expected_updates": agent.performance_info.expected_updates,
            "compile_break_even_updates": agent.performance_info.compile_break_even_updates,
            "compile_cold_break_even_updates": agent.performance_info.compile_cold_break_even_updates,
            "compile_warm_break_even_updates": agent.performance_info.compile_warm_break_even_updates,
            "host_dispatch_confidence": agent.performance_info.host_dispatch_confidence,
            "host_dispatch_margin_pct": agent.performance_info.host_dispatch_margin_pct,
            "compiled_hotpaths": list(getattr(agent, "compiled_hotpaths", ())),
            "parameters_finite": bool(finite_state["parameters_finite"]),
            "finite_state": finite_state,
            "compile_scope": agent.performance_info.compile_scope,
            "compile_scope_request": getattr(config, "compile_scope", "auto"),
            "gpu_conditioning": conditioning,
            "flow_likelihood_precision": getattr(
                agent, "flow_likelihood_precision", "not_applicable"
            ),
            "cpu_threads": int(torch.get_num_threads()),
            "cpu_interop_threads": int(torch.get_num_interop_threads()),
        }
    )
    return result



def _pipeline_micro_reference(args, info, config, *, compile_scope: str) -> dict[str, object]:
    from .action_space import configure_agent_action_levels
    from .performance import agent_finite_state, prepare_steady_state_agent, timed_iterations_detailed
    from .registry import create_agent
    from .replay import ReplayBatch

    torch = require_torch()
    reference_config = replace(
        config,
        replay_device="same",
        compile_scope=compile_scope,
        metrics_interval=max(int(args.iterations) + int(args.warmup) + 1, 1000),
    )
    agent = create_agent(
        args.algorithm, args.obs_dim, args.action_dim, reference_config, device=info.resolved
    )
    configure_agent_action_levels(agent, HPRLActionConfig().level_count)
    batch = ReplayBatch(
        obs=torch.randn(args.batch_size, args.obs_dim, device=info.resolved),
        action=torch.rand(args.batch_size, args.action_dim, device=info.resolved),
        reward=torch.randn(args.batch_size, 1, device=info.resolved) * 0.01,
        next_obs=torch.randn(args.batch_size, args.obs_dim, device=info.resolved),
        done=torch.zeros(args.batch_size, 1, device=info.resolved),
    )
    prepare_steady_state_agent(agent)
    result = timed_iterations_detailed(
        lambda: agent.update(batch, collect_metrics=False),
        warmup=args.warmup,
        iterations=args.iterations,
        device=info.resolved,
        samples_per_iteration=args.batch_size,
    )
    finite = agent_finite_state(agent)
    result.update({
        "algorithm": args.algorithm,
        "compile_scope": agent.performance_info.compile_scope,
        "compile_mode": agent.performance_info.compile_mode,
        "compile_cache_state": agent.performance_info.compile_cache_state,
        "hardware_profile": agent.performance_info.hardware_profile,
        "flow_likelihood_precision": getattr(agent, "flow_likelihood_precision", "not_applicable"),
        "cpu_interop_threads": int(torch.get_num_interop_threads()),
        "mixed_precision": bool(reference_config.mixed_precision),
        "parameters_finite": bool(finite["parameters_finite"]),
    })
    return result


def _perf_pipeline_benchmark(args) -> dict[str, object]:
    from .action_space import configure_agent_action_levels
    from .performance import agent_finite_state, condition_cuda_device, prepare_steady_state_agent
    from .pipeline_benchmark import benchmark_training_pipeline
    from .registry import create_agent

    if args.batch_size < 1 or args.iterations < 1 or args.replay_capacity < args.batch_size:
        raise ValueError("pipeline benchmark dimensions are invalid")
    info = resolve_device(args.device)
    config = HPRLTrainingConfig(
        algorithm=args.algorithm,
        device=args.device,
        replay_device=args.replay_device,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        warmup_steps=0,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        mixed_precision=bool(args.mixed_precision),
        optimizer_backend=args.optimizer_backend,
        polyak_backend=args.polyak_backend,
        grad_clip_backend=args.grad_clip_backend,
        compile_mode=args.compile_mode,
        compile_scope=args.compile_scope,
        expected_updates=args.iterations,
        hardware_profile=args.hardware_profile,
        compile_cache_state=args.compile_cache_state,
        cpu_threads=args.cpu_threads,
        cpu_interop_threads=args.cpu_interop_threads,
        flow_likelihood_precision=args.flow_likelihood_precision,
        flow_obs_projection_reuse=bool(args.flow_obs_projection_reuse),
        replay_prefetch=True,
        replay_prefetch_slots=args.prefetch_slots,
        metrics_interval=args.metrics_interval,
    )
    agent = create_agent(
        args.algorithm, args.obs_dim, args.action_dim, config, device=info.resolved
    )
    configure_agent_action_levels(agent, HPRLActionConfig().level_count)
    staged_warmup_updates = prepare_steady_state_agent(agent)
    conditioning = condition_cuda_device(
        info.resolved, milliseconds=args.gpu_condition_ms, matrix_size=512
    )
    result = benchmark_training_pipeline(
        agent,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup=args.warmup,
        replay_capacity=args.replay_capacity,
        replay_device=args.replay_device,
        pin_memory=True,
        prefetch_slots=args.prefetch_slots,
        metrics_interval=args.metrics_interval,
        checkpoint_interval=args.checkpoint_interval,
        diagnostic_iterations=args.diagnostic_iterations,
        async_artifacts=not bool(args.sync_artifacts),
        artifact_io_mode=("sync" if bool(args.sync_artifacts) else args.artifact_io_mode),
        artifact_queue_size=args.artifact_queue_size,
        checkpoint_cpu_snapshot=not bool(args.checkpoint_live_device_snapshot),
        estimated_logger_bytes_per_event=args.estimated_log_bytes_per_event,
        prior_queue_block_ratio=args.prior_artifact_block_ratio,
    )
    finite_state = agent_finite_state(agent)
    payload = asdict(result)
    resolved_scope = agent.performance_info.compile_scope
    same_scope_micro = None
    module_micro = None
    if not bool(args.skip_micro_reference):
        same_scope_micro = _pipeline_micro_reference(
            args, info, config, compile_scope=resolved_scope
        )
        module_micro = (
            same_scope_micro
            if resolved_scope == "module"
            else _pipeline_micro_reference(args, info, config, compile_scope="module")
        )
    pipe_samples = float(payload.get("samples_per_second", 0.0))
    same_samples = float((same_scope_micro or {}).get("samples_per_second", 0.0))
    module_samples = float((module_micro or {}).get("samples_per_second", 0.0))
    payload.update({
        "schema": "hprl-training-pipeline-benchmark-v2",
        "algorithm": args.algorithm,
        "device_name": info.device_name,
        "compile_mode": agent.performance_info.compile_mode,
        "compile_scope": resolved_scope,
        "compile_scope_request": config.compile_scope,
        "compiled_hotpaths": list(getattr(agent, "compiled_hotpaths", ())),
        "parameters_finite": bool(finite_state["parameters_finite"]),
        "finite_state": finite_state,
        "flow_likelihood_precision": getattr(agent, "flow_likelihood_precision", "not_applicable"),
        "staged_warmup_updates": staged_warmup_updates,
        "gpu_conditioning": conditioning,
        "micro_reference_same_scope": same_scope_micro,
        "micro_reference_module_baseline": module_micro,
        "pipeline_efficiency_same_scope": pipe_samples / same_samples if same_samples > 0 else None,
        "end_to_end_speedup_vs_module_baseline": (
            pipe_samples / module_samples if module_samples > 0 else None
        ),
        "benchmark_identity": {
            "algorithm": args.algorithm,
            "compile_scope": resolved_scope,
            "precision": getattr(agent, "flow_likelihood_precision", (
                getattr(agent.precision, "dtype_name", "float32")
            )),
            "cpu_interop_threads": int(require_torch().get_num_interop_threads()),
            "cache_state": agent.performance_info.compile_cache_state,
        },
    })
    return payload

def _perf_xqc_decomposition(args) -> dict[str, object]:
    from .action_space import configure_agent_action_levels
    from .performance import agent_finite_state, prepare_steady_state_agent
    from .pipeline_benchmark import profile_xqc_pipeline_decomposition
    from .registry import create_agent

    info = resolve_device(args.device)
    config = HPRLTrainingConfig(
        algorithm="xqc", device=args.device, replay_device="cpu",
        batch_size=args.batch_size, replay_capacity=max(args.batch_size * 16, 16384),
        warmup_steps=0, hidden_dim=args.hidden_dim, hidden_depth=args.hidden_depth,
        mixed_precision=bool(args.mixed_precision), compile_mode=args.compile_mode,
        compile_scope=args.compile_scope, expected_updates=max(args.iterations, 10000),
        hardware_profile=args.hardware_profile, compile_cache_state=args.compile_cache_state,
        cpu_threads=args.cpu_threads, cpu_interop_threads=args.cpu_interop_threads,
        metrics_interval=max(args.iterations + 1, 1000),
    )
    agent = create_agent("xqc", args.obs_dim, args.action_dim, config, device=info.resolved)
    configure_agent_action_levels(agent, HPRLActionConfig().level_count)
    prepare_steady_state_agent(agent)
    result = profile_xqc_pipeline_decomposition(
        agent, obs_dim=args.obs_dim, action_dim=args.action_dim, batch_size=args.batch_size,
        iterations=args.iterations, replay_capacity=max(args.batch_size * 16, 16384),
    )
    finite = agent_finite_state(agent)
    result.update({
        "algorithm": "xqc",
        "device_name": info.device_name,
        "compile_mode": agent.performance_info.compile_mode,
        "compile_scope": agent.performance_info.compile_scope,
        "parameters_finite": bool(finite["parameters_finite"]),
        "finite_state": finite,
    })
    return result


def _perf_sustained_pipeline(args) -> dict[str, object]:
    from .action_space import configure_agent_action_levels
    from .performance import prepare_steady_state_agent
    from .registry import create_agent
    from .sustained_benchmark import benchmark_sustained_training

    info = resolve_device(args.device)
    config = HPRLTrainingConfig(
        algorithm=args.algorithm, device=args.device, replay_device=args.replay_device,
        batch_size=args.batch_size, replay_capacity=args.replay_capacity, warmup_steps=0,
        hidden_dim=args.hidden_dim, hidden_depth=args.hidden_depth,
        mixed_precision=bool(args.mixed_precision), optimizer_backend=args.optimizer_backend,
        polyak_backend=args.polyak_backend, grad_clip_backend=args.grad_clip_backend,
        compile_mode=args.compile_mode, compile_scope=args.compile_scope,
        expected_updates=args.iterations, hardware_profile=args.hardware_profile,
        compile_cache_state=args.compile_cache_state, cpu_threads=args.cpu_threads,
        cpu_interop_threads=args.cpu_interop_threads,
        flow_likelihood_precision=args.flow_likelihood_precision,
        replay_prefetch=True, replay_prefetch_slots=args.prefetch_slots,
        metrics_interval=args.metrics_interval,
    )
    agent = create_agent(args.algorithm, args.obs_dim, args.action_dim, config, device=info.resolved)
    configure_agent_action_levels(agent, HPRLActionConfig().level_count)
    staged = prepare_steady_state_agent(agent)
    result = benchmark_sustained_training(
        agent, obs_dim=args.obs_dim, action_dim=args.action_dim, batch_size=args.batch_size,
        iterations=args.iterations, warmup=args.warmup, window_size=args.window_size,
        replay_capacity=args.replay_capacity, replay_device=args.replay_device,
        pin_memory=True, prefetch_slots=args.prefetch_slots, metrics_interval=args.metrics_interval,
        checkpoint_interval=args.checkpoint_interval, checkpoint_keep_last=args.checkpoint_keep_last,
        artifact_queue_size=args.artifact_queue_size, artifact_io_mode=args.artifact_io_mode,
        estimated_logger_bytes_per_event=args.estimated_log_bytes_per_event,
        prior_queue_block_ratio=args.prior_artifact_block_ratio,
    )
    payload = asdict(result)
    payload.update({
        "algorithm": args.algorithm, "device": info.resolved, "device_name": info.device_name,
        "compile_mode": agent.performance_info.compile_mode,
        "compile_scope": agent.performance_info.compile_scope,
        "compile_cache_state": agent.performance_info.compile_cache_state,
        "compiled_hotpaths": list(getattr(agent, "compiled_hotpaths", ())),
        "staged_warmup_updates": staged,
        "flow_likelihood_precision": getattr(agent, "flow_likelihood_precision", "not_applicable"),
    })
    return payload


def _perf_profile(args) -> dict[str, object]:
    from .action_space import configure_agent_action_levels
    from .performance import prepare_steady_state_agent, profile_iterations
    from .registry import create_agent
    from .replay import ReplayBatch

    torch = require_torch()
    info = resolve_device(args.device)
    config = HPRLTrainingConfig(
        algorithm=args.algorithm,
        device=args.device,
        replay_device="same",
        batch_size=args.batch_size,
        replay_capacity=max(args.batch_size * 2, 256),
        warmup_steps=0,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        mixed_precision=bool(args.mixed_precision),
        optimizer_backend=args.optimizer_backend,
        polyak_backend=args.polyak_backend,
        grad_clip_backend=args.grad_clip_backend,
        compile_mode=args.compile_mode,
        compile_scope=getattr(args, "compile_scope", "auto"),
        expected_updates=getattr(args, "expected_updates", 0),
        hardware_profile=getattr(args, "hardware_profile", "auto"),
        compile_cache_state=getattr(args, "compile_cache_state", "cold"),
        cpu_threads=args.cpu_threads,
        cpu_interop_threads=args.cpu_interop_threads,
        flow_likelihood_precision=args.flow_likelihood_precision,
        flow_obs_projection_reuse=bool(args.flow_obs_projection_reuse),
        metrics_interval=10_000,
    )
    obs_dim, action_dim = 32, 4
    agent = create_agent(args.algorithm, obs_dim, action_dim, config, device=info.resolved)
    configure_agent_action_levels(agent, HPRLActionConfig().level_count)
    batch = ReplayBatch(
        obs=torch.randn(args.batch_size, obs_dim, device=info.resolved),
        action=torch.rand(args.batch_size, action_dim, device=info.resolved),
        reward=torch.randn(args.batch_size, 1, device=info.resolved) * 0.01,
        next_obs=torch.randn(args.batch_size, obs_dim, device=info.resolved),
        done=torch.zeros(args.batch_size, 1, device=info.resolved),
    )
    staged_warmup_updates = prepare_steady_state_agent(agent)
    # Keep profiler windows short and avoid shape/stack tracing so profiling itself does not
    # dominate the small-MLP workload.
    rows = profile_iterations(
        lambda: agent.update(batch, collect_metrics=False),
        device=info.resolved,
        wait=1,
        warmup=1,
        active=args.active,
        profile_memory=False,
        row_limit=args.row_limit,
    )
    return {
        "schema": "hprl-performance-profile-v1",
        "algorithm": args.algorithm,
        "device": info.resolved,
        "device_name": info.device_name,
        "batch_size": args.batch_size,
        "optimizer_backend": getattr(agent.actor_opt, "_hprl_backend", "unknown"),
        "polyak_backend": agent.performance_info.polyak_backend,
        "grad_clip_backend": agent.performance_info.grad_clip_backend,
        "compile_request": config.compile_mode,
        "compile_mode": agent.performance_info.compile_mode,
        "compiled_hotpaths": list(getattr(agent, "compiled_hotpaths", ())),
        "flow_likelihood_precision": getattr(
            agent, "flow_likelihood_precision", "not_applicable"
        ),
        "staged_warmup_updates": staged_warmup_updates,
        "benchmark_stage": "steady_state",
        "top_ops": rows,
    }



def _perf_orchestration_profile(args) -> dict[str, object]:
    from .action_space import configure_agent_action_levels
    from .performance import (
        condition_cuda_device, prepare_steady_state_agent, profile_iterations,
        summarize_profile_operations,
    )
    from .registry import create_agent
    from .replay import ReplayBatch

    torch = require_torch()
    info = resolve_device(args.device)
    config = HPRLTrainingConfig(
        algorithm=args.algorithm, device=args.device, replay_device="same",
        batch_size=args.batch_size, replay_capacity=max(256, args.batch_size * 2),
        warmup_steps=0, hidden_dim=args.hidden_dim, hidden_depth=args.hidden_depth,
        mixed_precision=bool(args.mixed_precision), optimizer_backend=args.optimizer_backend,
        polyak_backend=args.polyak_backend, grad_clip_backend=args.grad_clip_backend,
        compile_mode=args.compile_mode, compile_scope=args.compile_scope, expected_updates=100000,
        hardware_profile=args.hardware_profile, compile_cache_state=args.compile_cache_state,
        cpu_threads=args.cpu_threads, cpu_interop_threads=args.cpu_interop_threads,
        flow_likelihood_precision=args.flow_likelihood_precision, metrics_interval=100000,
    )
    obs_dim, action_dim = 32, 4
    agent = create_agent(args.algorithm, obs_dim, action_dim, config, device=info.resolved)
    configure_agent_action_levels(agent, HPRLActionConfig().level_count)
    prepare_steady_state_agent(agent)
    batch = ReplayBatch(
        obs=torch.randn(args.batch_size, obs_dim, device=info.resolved),
        action=torch.rand(args.batch_size, action_dim, device=info.resolved),
        reward=torch.randn(args.batch_size, 1, device=info.resolved) * 0.01,
        next_obs=torch.randn(args.batch_size, obs_dim, device=info.resolved),
        done=torch.zeros(args.batch_size, 1, device=info.resolved),
    )
    conditioning = condition_cuda_device(info.resolved, milliseconds=args.gpu_condition_ms)
    # Prime lazy torch.compile/CUDAGraph capture before profiler scheduling.
    for _ in range(5):
        agent.update(batch, collect_metrics=False)
    synchronize_device(info.resolved)
    rows = profile_iterations(
        lambda: agent.update(batch, collect_metrics=False),
        device=info.resolved, wait=1, warmup=1, active=args.active,
        profile_memory=False, row_limit=500,
    )
    summary = summarize_profile_operations(rows)
    return {
        "schema": "hprl-orchestration-profile-v1",
        "algorithm": args.algorithm, "device": info.resolved, "device_name": info.device_name,
        "compile_mode": agent.performance_info.compile_mode, "compile_scope": args.compile_scope,
        "compiled_hotpaths": list(getattr(agent, "compiled_hotpaths", ())),
        "flow_likelihood_precision": getattr(agent, "flow_likelihood_precision", "not_applicable"),
        "gpu_conditioning": conditioning, **summary, "top_ops": rows[:50],
    }

def _isolated_perf_benchmark(
    args,
    *,
    compile_mode: str,
    polyak_backend: str,
    grad_clip_backend: str,
    flow_likelihood_precision: str | None = None,
    flow_obs_projection_reuse: bool | None = None,
    cpu_interop_threads: int | None = None,
):
    command = [
        sys.executable,
        "-m",
        "freqtrade.hedge.hprl",
        "perf-benchmark",
        "--device",
        args.device,
        "--algorithm",
        args.algorithm,
        "--batch-size",
        str(args.batch_size),
        "--hidden-dim",
        str(args.hidden_dim),
        "--hidden-depth",
        str(args.hidden_depth),
        "--warmup",
        str(args.warmup),
        "--iterations",
        str(args.iterations),
        "--optimizer-backend",
        args.optimizer_backend,
        "--polyak-backend",
        polyak_backend,
        "--grad-clip-backend",
        grad_clip_backend,
        "--flow-likelihood-precision",
        (
            flow_likelihood_precision
            if flow_likelihood_precision is not None
            else getattr(args, "flow_likelihood_precision", "auto")
        ),
        "--cpu-threads",
        str(args.cpu_threads),
        "--cpu-interop-threads",
        str(
            cpu_interop_threads
            if cpu_interop_threads is not None
            else getattr(args, "cpu_interop_threads", 1)
        ),
        "--obs-dim",
        str(args.obs_dim),
        "--action-dim",
        str(args.action_dim),
        "--compile-mode",
        compile_mode,
    ]
    if args.mixed_precision:
        command.append("--mixed-precision")
    projection_reuse = (
        bool(flow_obs_projection_reuse)
        if flow_obs_projection_reuse is not None
        else bool(getattr(args, "flow_obs_projection_reuse", False))
    )
    if projection_reuse:
        command.append("--flow-obs-projection-reuse")
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    if completed.returncode != 0:
        return None, (completed.stderr or completed.stdout).strip()[-4000:]
    try:
        return json.loads(completed.stdout), None
    except json.JSONDecodeError as exc:
        return None, f"invalid benchmark JSON: {exc}: {completed.stdout[-1000:]}"


def _perf_compile_sweep(args) -> dict[str, object]:
    info = resolve_device(args.device)
    if info.resolved == "cpu":
        raise ValueError("perf-compile-sweep requires a CUDA device")
    if not (1.0 <= float(args.min_speedup) <= 5.0):
        raise ValueError("min-speedup must be within [1, 5]")

    results: dict[str, dict[str, object]] = {}
    errors: dict[str, str] = {}
    for mode in ("off", "reduce-overhead"):
        result, error = _isolated_perf_benchmark(
            args,
            compile_mode=mode,
            polyak_backend="auto",
            grad_clip_backend="auto",
        )
        if result is not None:
            results[mode] = result
        if error is not None:
            errors[mode] = error

    eager = results.get("off")
    compiled = results.get("reduce-overhead")
    speedup = 0.0
    recommended = "off"
    if eager is not None and compiled is not None:
        eager_rate = float(eager["iterations_per_second"])
        compiled_rate = float(compiled["iterations_per_second"])
        speedup = compiled_rate / max(eager_rate, 1e-12)
        if speedup >= float(args.min_speedup) and compiled.get("compiled_hotpaths"):
            recommended = "reduce-overhead"
    return {
        "schema": "hprl-compile-sweep-v1",
        "algorithm": args.algorithm,
        "device": info.resolved,
        "device_name": info.device_name,
        "min_speedup": float(args.min_speedup),
        "speedup": speedup,
        "recommended_compile_mode": recommended,
        "results": results,
        "errors": errors,
    }


def _perf_backend_sweep(args) -> dict[str, object]:
    info = resolve_device(args.device)
    results: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for polyak in ("for_loop", "foreach"):
        for grad_clip in ("for_loop", "foreach"):
            result, error = _isolated_perf_benchmark(
                args,
                compile_mode=args.compile_mode,
                polyak_backend=polyak,
                grad_clip_backend=grad_clip,
            )
            if result is not None:
                results.append(result)
            if error is not None:
                errors.append(
                    {
                        "polyak_backend": polyak,
                        "grad_clip_backend": grad_clip,
                        "error": error,
                    }
                )
    ordered = sorted(results, key=lambda item: float(item["iterations_per_second"]), reverse=True)
    best = ordered[0] if ordered else None
    return {
        "schema": "hprl-backend-sweep-v1",
        "algorithm": args.algorithm,
        "device": info.resolved,
        "device_name": info.device_name,
        "compile_mode": args.compile_mode,
        "best": best,
        "results": ordered,
        "errors": errors,
    }


def _perf_host_sweep(args) -> dict[str, object]:
    info = resolve_device(args.device)
    if info.resolved == "cpu":
        raise ValueError("perf-host-sweep requires a CUDA device")
    try:
        candidates = tuple(
            dict.fromkeys(
                int(value.strip())
                for value in str(args.interop_candidates).split(",")
                if value.strip()
            )
        )
    except ValueError as exc:
        raise ValueError("interop-candidates must be comma-separated integers") from exc
    if not candidates or any(value < 1 for value in candidates):
        raise ValueError("interop-candidates must contain positive integers")

    results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for interop_threads in candidates:
        result, error = _isolated_perf_benchmark(
            args,
            compile_mode=args.compile_mode,
            polyak_backend=args.polyak_backend,
            grad_clip_backend=args.grad_clip_backend,
            cpu_interop_threads=interop_threads,
        )
        if result is not None:
            results.append(result)
        if error is not None:
            errors.append({"cpu_interop_threads": interop_threads, "error": error})
    ordered = sorted(
        results,
        key=lambda item: float(item["iterations_per_second"]),
        reverse=True,
    )
    return {
        "schema": "hprl-host-dispatch-sweep-v1",
        "algorithm": args.algorithm,
        "device": info.resolved,
        "device_name": info.device_name,
        "compile_mode": args.compile_mode,
        "cpu_threads": args.cpu_threads,
        "candidates": list(candidates),
        "best": ordered[0] if ordered else None,
        "results": ordered,
        "errors": errors,
    }


def _perf_flow_profile(args) -> dict[str, object]:
    from contextlib import nullcontext

    from .algorithms.rebrac_v2 import ConditionalFlowActor
    from .device import PrecisionManager
    from .performance import profile_iterations, timed_iterations_detailed

    if min(args.batch_size, args.obs_dim, args.action_dim, args.hidden_dim, args.flow_layers) < 1:
        raise ValueError("flow profile dimensions must be positive")
    info = resolve_device(args.device)
    precision = PrecisionManager(
        info.resolved,
        enabled=bool(args.mixed_precision),
        dtype="auto",
    )
    mode = str(args.flow_likelihood_precision).strip().lower()
    if mode not in {"auto", "fp32", "mixed"}:
        raise ValueError("flow-likelihood-precision must be auto/fp32/mixed")
    use_mixed = mode == "mixed" or (mode == "auto" and precision.enabled)
    if use_mixed and not precision.enabled:
        raise ValueError("mixed flow likelihood requires --mixed-precision on CUDA")
    torch = require_torch()
    actor = ConditionalFlowActor(
        args.obs_dim,
        args.action_dim,
        args.hidden_dim,
        args.flow_layers,
    ).to(info.resolved)
    obs = torch.randn(args.batch_size, args.obs_dim, device=info.resolved)
    action = (
        torch.rand(args.batch_size, args.action_dim, device=info.resolved)
        .mul_(0.98)
        .add_(0.01)
    )

    def sample_fast():
        with precision.autocast():
            return actor.sample(obs, compute_log_prob=False)

    def sample_likelihood():
        with precision.autocast():
            return actor.sample(obs, compute_log_prob=True)

    def data_likelihood():
        context = precision.autocast() if use_mixed else nullcontext()
        with context:
            return actor.log_prob(obs.float(), action.float(), stable_fp32=use_mixed)

    def separate_actor_surfaces():
        with precision.autocast():
            sampled, log_prob, _ = actor.sample(obs, compute_log_prob=True)
        context = precision.autocast() if use_mixed else nullcontext()
        with context:
            data_log_prob = actor.log_prob(
                obs.float(), action.float(), stable_fp32=use_mixed
            )
        return sampled, log_prob, data_log_prob

    def paired_actor_surfaces():
        with precision.autocast():
            return actor.sample_and_data_log_prob(
                obs, action, stable_fp32=use_mixed
            )

    timing = {}
    for name, fn in (
        ("sample_no_logprob", sample_fast),
        ("sample_with_logprob", sample_likelihood),
        ("data_log_prob", data_likelihood),
        ("separate_actor_surfaces", separate_actor_surfaces),
        ("paired_actor_surfaces", paired_actor_surfaces),
    ):
        timing[name] = timed_iterations_detailed(
            fn,
            warmup=args.warmup,
            iterations=args.iterations,
            device=info.resolved,
            samples_per_iteration=args.batch_size,
        )
    top_ops = profile_iterations(
        data_likelihood,
        device=info.resolved,
        wait=1,
        warmup=1,
        active=3,
        profile_memory=False,
        row_limit=args.row_limit,
    )
    return {
        "schema": "hprl-flow-profile-v1",
        "device": info.resolved,
        "device_name": info.device_name,
        "batch_size": args.batch_size,
        "obs_dim": args.obs_dim,
        "action_dim": args.action_dim,
        "hidden_dim": args.hidden_dim,
        "flow_layers": args.flow_layers,
        "mixed_precision": bool(args.mixed_precision),
        "flow_likelihood_precision": "mixed" if use_mixed else "fp32",
        "timing": timing,
        "paired_projection_speedup": (
            float(timing["paired_actor_surfaces"]["iterations_per_second"])
            / max(float(timing["separate_actor_surfaces"]["iterations_per_second"]), 1e-12)
        ),
        "paired_projection_recommended": (
            float(timing["paired_actor_surfaces"]["iterations_per_second"])
            >= 1.03 * float(timing["separate_actor_surfaces"]["iterations_per_second"])
        ),
        "data_log_prob_top_ops": top_ops,
    }


def _perf_flow_sweep(args) -> dict[str, object]:
    info = resolve_device(args.device)
    if info.resolved == "cpu":
        raise ValueError("perf-flow-sweep requires a CUDA device")
    if not (1.0 <= float(args.min_speedup) <= 5.0):
        raise ValueError("min-speedup must be within [1, 5]")

    args.algorithm = "rebrac_v2"
    args.mixed_precision = True
    results: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for precision in ("fp32", "mixed"):
        for reuse in (False, True):
            result, error = _isolated_perf_benchmark(
                args,
                compile_mode=args.compile_mode,
                polyak_backend=args.polyak_backend,
                grad_clip_backend=args.grad_clip_backend,
                flow_likelihood_precision=precision,
                flow_obs_projection_reuse=reuse,
            )
            candidate: dict[str, object] = {
                "flow_likelihood_precision": precision,
                "flow_obs_projection_reuse": reuse,
            }
            if result is not None:
                candidate["result"] = result
                results.append(candidate)
            if error is not None:
                errors.append({**candidate, "error": error})

    ordered = sorted(
        results,
        key=lambda item: float(item["result"]["iterations_per_second"]),
        reverse=True,
    )
    baseline = next(
        (
            item
            for item in ordered
            if item["flow_likelihood_precision"] == "fp32"
            and not item["flow_obs_projection_reuse"]
        ),
        None,
    )
    best = ordered[0] if ordered else None
    speedup = 0.0
    recommended = {
        "flow_likelihood_precision": "fp32",
        "flow_obs_projection_reuse": False,
    }
    if best is not None and baseline is not None:
        baseline_rate = float(baseline["result"]["iterations_per_second"])
        best_rate = float(best["result"]["iterations_per_second"])
        speedup = best_rate / max(baseline_rate, 1e-12)
        if speedup >= float(args.min_speedup):
            recommended = {
                "flow_likelihood_precision": best["flow_likelihood_precision"],
                "flow_obs_projection_reuse": best["flow_obs_projection_reuse"],
            }
    return {
        "schema": "hprl-rebrac-flow-sweep-v1",
        "algorithm": "rebrac_v2",
        "device": info.resolved,
        "device_name": info.device_name,
        "compile_mode": args.compile_mode,
        "min_speedup": float(args.min_speedup),
        "speedup_vs_fp32_separate": speedup,
        "recommended": recommended,
        "best": best,
        "results": ordered,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect":
        print(
            json.dumps(
                {
                    "api_version": HPRL_API_VERSION,
                    "release": HPRL_RELEASE,
                    "algorithms": available_algorithms(),
                    "existing_rl_modified": False,
                    "live_order_write": False,
                    "device_policy": HPRL_DEVICE_POLICY,
                    "default_device": "auto",
                    "supported_devices": ["auto", "cpu", "cuda", "cuda:<index>"],
                    "action_space": {
                        "default_mode": HPRLActionConfig().mode,
                        "position_levels": list(HPRLActionConfig().position_levels),
                        "joint_states_per_symbol": HPRLActionConfig().joint_states_per_symbol,
                        "multi_discrete_nvec": list(HPRLActionConfig().multi_discrete_nvec),
                        "level_semantics": "margin_budget_ratio",
                        "policy_surface": "continuous_latent_with_tiered_execution",
                        "decrease_policy": "unlimited_by_default",
                        "stochastic_entropy": "executed_tier_distribution",
                        "tier_entropy_target_fraction": (
                            HPRLTrainingConfig().tier_entropy_target_fraction
                        ),
                    },
                    "reward": {
                        "return_scale": HPRLRewardConfig().return_scale,
                        "default_clip": HPRLRewardConfig().reward_clip,
                        "net_of_execution_costs": True,
                        "double_count_cost_shaping_by_default": False,
                    },
                    "performance": {
                        "optimizer_backend": HPRLTrainingConfig().optimizer_backend,
                        "compile_mode": HPRLTrainingConfig().compile_mode,
                        "auto_compile_policy": __import__(
                            "freqtrade.hedge.hprl.performance",
                            fromlist=["auto_compile_policy"],
                        ).auto_compile_policy(),
                        "cpu_threads": HPRLTrainingConfig().cpu_threads,
                        "cpu_interop_threads": HPRLTrainingConfig().cpu_interop_threads,
                        "polyak_backend": HPRLTrainingConfig().polyak_backend,
                        "grad_clip_backend": HPRLTrainingConfig().grad_clip_backend,
                        "return_scan_backend": HPRLTrainingConfig().return_scan_backend,
                        "flow_likelihood_precision": (
                            HPRLTrainingConfig().flow_likelihood_precision
                        ),
                        "flow_obs_projection_reuse": (
                            HPRLTrainingConfig().flow_obs_projection_reuse
                        ),
                        "replay_ring_write": "two-slice-copy",
                        "replay_sample_staging": HPRLTrainingConfig().replay_reuse_sample_buffers,
                        "gaussian_sample": "distribution-object-free",
                        "tier_tensor_cache": True,
                        "cpu_replay_cuda_prefetch": HPRLTrainingConfig().replay_prefetch,
                        "replay_prefetch_slots": HPRLTrainingConfig().replay_prefetch_slots,
                    },
                    "gpu_acceleration": {
                        "resident_environment": True,
                        "resident_replay": True,
                        "cpu_replay_non_blocking_transfer": True,
                        "mixed_precision": True,
                        "tf32": True,
                        "synchronized_benchmarking": True,
                        "bounded_dataset_windows": True,
                        "auto_replay_placement": True,
                        "bounded_pinned_replay_staging": True,
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    if args.command == "device":
        info = resolve_device(args.device)
        print(json.dumps(asdict(info), ensure_ascii=False, indent=2))
        return 0
    if args.command == "memory":
        from .config import HPRLMemoryConfig
        from .memory import memory_budget_report

        if args.dataset_bytes < 0:
            raise ValueError("dataset-bytes cannot be negative")
        report = memory_budget_report(
            args.device,
            HPRLMemoryConfig(),
            dataset_bytes_estimate=args.dataset_bytes,
            replay_capacity=args.replay_capacity,
            obs_dim=args.obs_dim,
            action_dim=args.action_dim,
            replay_request=args.replay_device,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "compat":
        report = inspect_clean_mainline(Path(args.project_root))
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
        return 0 if report.compatible else 1
    if args.command == "smoke":
        info = resolve_device(args.device)
        torch = require_torch()
        probe = torch.ones((256, 256), device=info.resolved)
        probe = probe @ torch.eye(256, device=info.resolved)
        synchronize_device(info.resolved)
        if not torch.isfinite(probe).all():
            raise RuntimeError("HPRL tensor smoke produced non-finite values")
        print(f"HPRL smoke PASS | device={info.resolved} | {info.device_name}")
        return 0
    if args.command == "train-smoke":
        print(json.dumps(_train_smoke(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "perf-benchmark":
        print(json.dumps(_perf_benchmark(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "perf-pipeline-benchmark":
        print(json.dumps(_perf_pipeline_benchmark(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "perf-xqc-decomposition":
        print(json.dumps(_perf_xqc_decomposition(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "perf-sustained-pipeline":
        print(json.dumps(_perf_sustained_pipeline(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "perf-orchestration-profile":
        print(json.dumps(_perf_orchestration_profile(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "perf-profile":
        print(json.dumps(_perf_profile(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "perf-compile-sweep":
        print(json.dumps(_perf_compile_sweep(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "perf-backend-sweep":
        print(json.dumps(_perf_backend_sweep(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "perf-host-sweep":
        print(json.dumps(_perf_host_sweep(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "perf-flow-profile":
        print(json.dumps(_perf_flow_profile(args), ensure_ascii=False, indent=2))
        return 0
    if args.command == "perf-flow-sweep":
        print(json.dumps(_perf_flow_sweep(args), ensure_ascii=False, indent=2))
        return 0
    raise AssertionError("unreachable")
