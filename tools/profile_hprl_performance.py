#!/usr/bin/env python3
"""Operator-level profiler for HPRL update hot paths."""

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
    profile_iterations,
)
from freqtrade.hedge.hprl.registry import create_agent  # noqa: E402
from freqtrade.hedge.hprl.replay import ReplayBatch  # noqa: E402


torch = require_torch()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--algorithm",
        default="rebrac_v2",
        choices=("fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2"),
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--optimizer-backend", default="auto")
    parser.add_argument("--polyak-backend", default="auto")
    parser.add_argument("--grad-clip-backend", default="auto")
    parser.add_argument("--compile-mode", default="off")
    parser.add_argument("--flow-likelihood-precision", default="auto")
    parser.add_argument("--flow-obs-projection-reuse", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--cpu-interop-threads", type=int, default=1)
    parser.add_argument("--active", type=int, default=5)
    parser.add_argument("--row-limit", type=int, default=40)
    args = parser.parse_args()

    info = resolve_device(args.device)
    config = HPRLTrainingConfig(
        algorithm=args.algorithm,
        device=info.resolved,
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
    rows = profile_iterations(
        lambda: agent.update(batch, collect_metrics=False),
        device=info.resolved,
        wait=1,
        warmup=1,
        active=args.active,
        row_limit=args.row_limit,
    )
    print(
        json.dumps(
            {
                "schema": "hprl-performance-profile-v1",
                "algorithm": args.algorithm,
                "device": info.resolved,
                "device_name": info.device_name,
                "torch": torch.__version__,
                "batch_size": args.batch_size,
                "hidden_dim": args.hidden_dim,
                "hidden_depth": args.hidden_depth,
                "optimizer_backend": getattr(agent.actor_opt, "_hprl_backend", "unknown"),
                "polyak_backend": agent.performance_info.polyak_backend,
                "grad_clip_backend": agent.performance_info.grad_clip_backend,
                "compile_request": config.compile_mode,
                "compile_mode": agent.performance_info.compile_mode,
                "flow_likelihood_precision": getattr(
                    agent, "flow_likelihood_precision", "not_applicable"
                ),
                "flow_obs_projection_reuse": config.flow_obs_projection_reuse,
                "compiled_hotpaths": list(getattr(agent, "compiled_hotpaths", ())),
                "staged_warmup_updates": staged_warmup_updates,
                "benchmark_stage": "steady_state",
                "top_ops": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
