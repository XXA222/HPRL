from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl.config import (
    HPRLActionConfig,
    HPRLCostConfig,
    HPRLEnvironmentConfig,
    HPRLMemoryConfig,
)
from freqtrade.hedge.hprl.data import TensorMarketDataset
from freqtrade.hedge.hprl.device import require_torch, synchronize_device
from freqtrade.hedge.hprl.env import VectorizedHedgeEnv
from freqtrade.hedge.hprl.memory import cuda_memory_state, memory_budget_report
from freqtrade.hedge.hprl.replay import TensorReplayBuffer


torch = require_torch()


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for HPRL memory CUDA smoke")
    device = "cuda:0"
    torch.manual_seed(20260813)
    steps = 96
    symbols = 2
    features = 8
    market_features = torch.randn((steps, symbols, features), dtype=torch.float32) * 0.01
    returns = torch.randn((steps, symbols), dtype=torch.float32) * 0.001
    funding = torch.randn((steps, symbols), dtype=torch.float32) * 1e-5
    available = torch.full((steps, symbols), 1_000_000.0, dtype=torch.float32)
    dataset = TensorMarketDataset(
        features=market_features,
        forward_returns=returns,
        funding_rates=funding,
        available_notional=available,
        symbols=("BTC/USDT:USDT", "ETH/USDT:USDT"),
    ).validate()
    env_cfg = HPRLEnvironmentConfig(
        parallel_envs=4,
        action=HPRLActionConfig(max_step_change=1.0),
        costs=HPRLCostConfig(
            maker_fee_bps=0.0,
            taker_fee_bps=0.0,
            base_slippage_bps=0.0,
            impact_coefficient_bps=0.0,
        ),
    )
    resident = VectorizedHedgeEnv(
        dataset,
        env_cfg,
        device=device,
        memory_config=HPRLMemoryConfig(dataset_mode="resident"),
    )
    windowed = VectorizedHedgeEnv(
        dataset,
        env_cfg,
        device=device,
        memory_config=HPRLMemoryConfig(
            dataset_mode="windowed",
            dataset_window_steps=8,
            pin_staging_memory=True,
        ),
    )
    obs_a, _ = resident.reset()
    obs_b, _ = windowed.reset()
    if not torch.allclose(obs_a, obs_b, atol=0, rtol=0):
        raise RuntimeError("resident/windowed reset observations differ")
    for index in range(40):
        action = torch.rand((env_cfg.parallel_envs, resident.action_dim), device=device) * 0.25
        one = resident.step(action)
        two = windowed.step(action)
        for left, right, label in (
            (one.observation, two.observation, "observation"),
            (one.reward, two.reward, "reward"),
            (one.info["equity"], two.info["equity"], "equity"),
            (one.info["position"], two.info["position"], "position"),
        ):
            if not torch.allclose(left, right, atol=1e-6, rtol=1e-6):
                raise RuntimeError(f"resident/windowed parity failure at {index}: {label}")
    if windowed.market.plan.resolved_mode != "windowed":
        raise RuntimeError("explicit windowed market mode did not remain windowed")
    if windowed.market._window_end - windowed.market._window_start > 8:
        raise RuntimeError("windowed market cache exceeded configured steps")

    # Exercise CPU replay + bounded pinned staging against the real CUDA runtime.
    replay = TensorReplayBuffer(
        1024,
        resident.observation_dim,
        resident.action_dim,
        device="cpu",
        pin_memory=True,
        validate_inputs=True,
    )
    obs = torch.randn((64, resident.observation_dim), dtype=torch.float32)
    act = torch.rand((64, resident.action_dim), dtype=torch.float32)
    rew = torch.randn((64, 1), dtype=torch.float32)
    nxt = torch.randn((64, resident.observation_dim), dtype=torch.float32)
    done = torch.zeros((64, 1), dtype=torch.float32)
    replay.add(obs, act, rew, nxt, done)
    batch = replay.sample(32)
    if not batch.obs.is_pinned():
        raise RuntimeError("CPU replay sample staging is not pinned")
    gpu_batch = batch.to(device, non_blocking=False)
    if gpu_batch.obs.device.type != "cuda" or not torch.isfinite(gpu_batch.obs).all():
        raise RuntimeError("pinned replay H2D staging failed")
    synchronize_device(device)

    report = {
        "schema": "hprl-memory-cuda-smoke-v1",
        "status": "PASS",
        "device": torch.cuda.get_device_name(0),
        "window_steps": 8,
        "parity_steps": 40,
        "dataset_window_mode": windowed.market.plan.resolved_mode,
        "cpu_replay_persistent_bytes": replay.persistent_bytes,
        "cpu_replay_pinned_stage_bytes": replay.pinned_stage_bytes,
        "budget_example": memory_budget_report(
            device,
            HPRLMemoryConfig(),
            dataset_bytes_estimate=2 * 1024**3,
            replay_capacity=1_000_000,
            obs_dim=64,
            action_dim=8,
            replay_request="auto",
        ),
        "cuda_memory": None,
    }
    state = cuda_memory_state(device)
    if state is not None:
        report["cuda_memory"] = {
            "total_bytes": state.total_bytes,
            "free_bytes": state.free_bytes,
            "allocated_bytes": state.allocated_bytes,
            "reserved_bytes": state.reserved_bytes,
            "max_allocated_bytes": state.max_allocated_bytes,
            "max_reserved_bytes": state.max_reserved_bytes,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
