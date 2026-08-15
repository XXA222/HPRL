"""Dependency-light runtime smoke for HPRL Action/Reward V1.6."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl.action_space import (
    TieredHedgeActionCodec,
    canonicalize_offline_action_tensor,
)
from freqtrade.hedge.hprl.config import HPRLActionConfig, HPRLRewardConfig
from freqtrade.hedge.hprl.device import require_torch
from freqtrade.hedge.hprl.reward import CompositeReward, RewardFactsTensor


def _scalar_reward(model: CompositeReward, value: float, **overrides: float) -> float:
    torch = require_torch()
    facts = {
        "equity_return": torch.tensor([value], dtype=torch.float32),
        "drawdown_increase": torch.tensor([0.0]),
        "downside_return": torch.tensor([value]),
        "cvar_loss": torch.tensor([0.0]),
        "turnover_ratio": torch.tensor([0.0]),
        "fee_ratio": torch.tensor([0.0]),
        "slippage_ratio": torch.tensor([0.0]),
        "impact_ratio": torch.tensor([0.0]),
        "funding_ratio": torch.tensor([0.0]),
    }
    for name, override in overrides.items():
        facts[name] = torch.tensor([override], dtype=torch.float32)
    total, _ = model.evaluate_tensor(RewardFactsTensor(**facts))
    return float(total[0])


def main() -> int:
    torch = require_torch()
    action_cfg = HPRLActionConfig()
    assert action_cfg.position_levels == (0.0, 0.05, 0.12, 0.25, 0.40)
    assert action_cfg.level_count == 5
    assert action_cfg.joint_states_per_symbol == 25

    codec = TieredHedgeActionCodec(action_cfg)
    raw = torch.tensor([[[0.02, 0.99], [0.63, 0.11]]], dtype=torch.float32)
    current = torch.zeros_like(raw)
    decoded = codec.decode(raw, current)
    levels = torch.tensor(action_cfg.position_levels)
    for value in decoded.target_margin.flatten():
        assert bool(torch.isclose(levels, value).any())
    gross = float(decoded.target_margin.sum())
    net = float((decoded.target_margin[..., 0] - decoded.target_margin[..., 1]).sum())
    assert gross <= action_cfg.max_gross_margin_ratio + 1e-6
    assert abs(net) <= action_cfg.max_abs_net_margin_ratio + 1e-6
    assert torch.allclose(decoded.target_notional, decoded.target_margin * action_cfg.leverage)

    margin = torch.tensor([[0.00, 0.05, 0.12, 0.25, 0.40]], dtype=torch.float32)
    canonical = canonicalize_offline_action_tensor(margin, action_cfg, "margin_budget")
    expected = torch.tensor([[0.0, 0.25, 0.5, 0.75, 1.0]], dtype=torch.float32)
    assert torch.allclose(canonical, expected)

    reward_cfg = HPRLRewardConfig()
    reward = CompositeReward(reward_cfg)
    negative = _scalar_reward(reward, -0.01)
    flat = _scalar_reward(reward, 0.0)
    positive = _scalar_reward(reward, 0.01)
    assert negative < flat < positive
    assert abs(positive - math.log1p(0.01) * reward_cfg.return_scale) < 1e-5
    projected = _scalar_reward(reward, 0.0, constraint_distance=0.5)
    assert projected < flat
    terminal = _scalar_reward(reward, 0.0, terminal=1.0)
    assert terminal < flat

    result = {
        "schema": "hprl-action-reward-v16-runtime-smoke",
        "status": "PASS",
        "action_mode": action_cfg.mode,
        "position_levels": list(action_cfg.position_levels),
        "joint_states_per_symbol": action_cfg.joint_states_per_symbol,
        "policy_codes": [0.0, 0.25, 0.5, 0.75, 1.0],
        "reward_return_scale": reward_cfg.return_scale,
        "reward_clip": reward_cfg.reward_clip,
        "gross_margin_ratio": gross,
        "net_margin_ratio": net,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
