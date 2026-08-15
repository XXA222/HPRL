"""200 deterministic runtime checks for HPRL Action/Reward V1.6 without pytest."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl.action_space import (  # noqa: E402
    TieredHedgeActionCodec,
    canonicalize_offline_action_tensor,
    hard_quantize_unit_action,
)
from freqtrade.hedge.hprl.config import (  # noqa: E402
    HPRLActionConfig,
    HPRLRewardConfig,
    HPRLTrainingConfig,
)
from freqtrade.hedge.hprl.device import require_torch  # noqa: E402
from freqtrade.hedge.hprl.reward import CompositeReward, RewardFactsTensor  # noqa: E402


EXPECTED = 200


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch = require_torch()
    checks: list[dict[str, object]] = []

    def check(group: str, case: str, condition: object, detail: str = "PASS") -> None:
        passed = bool(condition)
        checks.append(
            {
                "number": len(checks) + 1,
                "group": group,
                "case": case,
                "passed": passed,
                "detail": detail if passed else "FAIL",
            }
        )

    cfg = HPRLActionConfig(max_increase_levels=4)
    codec = TieredHedgeActionCodec(cfg)
    levels = torch.tensor(cfg.position_levels, dtype=torch.float32)
    codes = torch.linspace(0.0, 1.0, cfg.level_count)

    defaults = [
        cfg.mode == "tiered",
        cfg.position_levels == (0.0, 0.05, 0.12, 0.25, 0.40),
        cfg.level_count == 5,
        cfg.joint_states_per_symbol == 25,
        cfg.max_leg_margin_ratio == 0.40,
        cfg.max_gross_margin_ratio == 0.80,
        cfg.max_abs_net_margin_ratio == 0.40,
        cfg.max_decrease_levels == -1,
        cfg.tier_hysteresis == 0.02,
        cfg.leverage == 1.0,
    ]
    for i, value in enumerate(defaults, 1):
        check("G01_DEFAULT_CONTRACT", f"default={i}", value)

    raw_quant = torch.linspace(0.0, 1.0, 10).reshape(1, 10)
    hard = hard_quantize_unit_action(raw_quant, cfg.level_count)
    for i, value in enumerate(hard.flatten(), 1):
        check("G02_POLICY_QUANTIZATION", f"value={i}", torch.isclose(codes, value).any())

    leverages = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0)
    for i, leverage in enumerate(leverages, 1):
        lc = HPRLActionConfig(leverage=leverage, max_increase_levels=4)
        dec = TieredHedgeActionCodec(lc).decode(
            torch.tensor([[[1.0, 0.5]]]), torch.zeros(1, 1, 2)
        )
        check(
            "G03_MARGIN_NOTIONAL",
            f"leverage={leverage}",
            torch.allclose(dec.target_notional, dec.target_margin * leverage),
        )

    step_cfg = HPRLActionConfig(max_increase_levels=1, tier_hysteresis=0.0)
    step_codec = TieredHedgeActionCodec(step_cfg)
    for i in range(10):
        current_level = i % 4
        current = torch.full((1, 1, 2), step_cfg.position_levels[current_level])
        dec = step_codec.decode(torch.ones(1, 1, 2), current)
        maximum = min(current_level + 1, step_cfg.level_count - 1)
        check(
            "G04_STEP_UP_LIMIT",
            f"case={i + 1}",
            bool((dec.executed_level_index <= maximum).all()),
        )

    for i in range(10):
        current_level = 1 + (i % 4)
        current = torch.full((1, 1, 2), step_cfg.position_levels[current_level])
        dec = step_codec.decode(torch.zeros(1, 1, 2), current)
        check("G05_FAST_DERISK", f"case={i + 1}", bool((dec.target_margin == 0).all()))

    hcfg = HPRLActionConfig(max_increase_levels=4, tier_hysteresis=0.03)
    hcodec = TieredHedgeActionCodec(hcfg)
    for i in range(10):
        raw = 0.38 + 0.005 * i
        current = torch.full((1, 1, 2), 0.05)
        dec = hcodec.decode(torch.full((1, 1, 2), raw), current)
        expected_hold = raw < 0.405
        held = bool((dec.executed_level_index == 1).all())
        check("G06_HYSTERESIS", f"raw={raw:.3f}", held == expected_hold)

    gen = torch.Generator().manual_seed(1601)
    for i in range(10):
        action = torch.rand((16, 5, 2), generator=gen)
        current = torch.zeros_like(action)
        dec = codec.decode(action, current)
        gross = dec.target_margin.sum(dim=(-2, -1))
        check(
            "G07_GROSS_ENVELOPE",
            f"batch={i + 1}",
            bool((gross <= cfg.max_gross_margin_ratio + 1e-6).all()),
        )

    for i in range(10):
        action = torch.zeros(8, 4, 2)
        action[..., 0] = 1.0
        action[..., 1] = 0.1 * (i % 3)
        dec = codec.decode(action, torch.zeros_like(action))
        net = (dec.target_margin[..., 0] - dec.target_margin[..., 1]).sum(dim=-1)
        check(
            "G08_POSITIVE_NET_ENVELOPE",
            f"case={i + 1}",
            bool((net <= cfg.max_abs_net_margin_ratio + 1e-6).all()),
        )

    for i in range(10):
        action = torch.zeros(8, 4, 2)
        action[..., 1] = 1.0
        action[..., 0] = 0.1 * (i % 3)
        dec = codec.decode(action, torch.zeros_like(action))
        net = (dec.target_margin[..., 0] - dec.target_margin[..., 1]).sum(dim=-1)
        check(
            "G09_NEGATIVE_NET_ENVELOPE",
            f"case={i + 1}",
            bool((net >= -cfg.max_abs_net_margin_ratio - 1e-6).all()),
        )

    for i in range(10):
        values = codes.roll(i % cfg.level_count).reshape(1, -1)
        out = canonicalize_offline_action_tensor(values, cfg, "policy_code")
        check("G10_OFFLINE_POLICY", f"case={i + 1}", torch.allclose(out, values))

    for i in range(10):
        values = levels.roll(i % cfg.level_count).reshape(1, -1)
        out = canonicalize_offline_action_tensor(values, cfg, "margin_budget")
        expected = codes.roll(i % cfg.level_count).reshape(1, -1)
        check("G11_OFFLINE_MARGIN", f"case={i + 1}", torch.allclose(out, expected))

    for i, leverage in enumerate(leverages, 1):
        lc = HPRLActionConfig(leverage=leverage, max_increase_levels=4)
        values = levels.reshape(1, -1) * leverage
        out = canonicalize_offline_action_tensor(values, lc, "notional_exposure")
        check(
            "G12_OFFLINE_NOTIONAL",
            f"leverage={leverage}",
            torch.allclose(out, codes.reshape(1, -1)),
        )

    reward = CompositeReward(HPRLRewardConfig())

    def reward_value(
        equity_return: float,
        *,
        drawdown: float = 0.0,
        cvar: float = 0.0,
        constraint: float = 0.0,
        terminal: bool = False,
    ) -> float:
        t = lambda value: torch.tensor([value], dtype=torch.float32)
        total, _ = reward.evaluate_tensor(
            RewardFactsTensor(
                equity_return=t(equity_return),
                drawdown_increase=t(drawdown),
                downside_return=t(equity_return),
                cvar_loss=t(cvar),
                turnover_ratio=t(0.0),
                fee_ratio=t(0.0),
                slippage_ratio=t(0.0),
                impact_ratio=t(0.0),
                funding_ratio=t(0.0),
                constraint_distance=t(constraint),
                terminal=torch.tensor([terminal]),
            )
        )
        return float(total[0])

    returns = [-0.02 + i * 0.004 for i in range(11)]
    return_rewards = [reward_value(value) for value in returns]
    for i in range(10):
        check(
            "G13_RETURN_MONOTONIC",
            f"pair={i + 1}",
            return_rewards[i + 1] > return_rewards[i],
        )

    for i in range(10):
        low = reward_value(0.0, drawdown=i * 0.01)
        high = reward_value(0.0, drawdown=(i + 1) * 0.01)
        check("G14_DRAWDOWN_MONOTONIC", f"pair={i + 1}", high <= low)

    for i in range(10):
        low = reward_value(0.0, cvar=i * 0.01)
        high = reward_value(0.0, cvar=(i + 1) * 0.01)
        check("G15_CVAR_MONOTONIC", f"pair={i + 1}", high <= low)

    for i in range(10):
        low = reward_value(0.0, constraint=i * 0.05)
        high = reward_value(0.0, constraint=(i + 1) * 0.05)
        check("G16_CONSTRAINT_MONOTONIC", f"pair={i + 1}", high <= low)

    for i in range(10):
        base = reward_value(i * 0.0001)
        ended = reward_value(i * 0.0001, terminal=True)
        check("G17_TERMINAL_PENALTY", f"case={i + 1}", ended < base)

    gen = torch.Generator().manual_seed(1602)
    for i in range(10):
        value = float(torch.rand(1, generator=gen) * 0.08 - 0.04)
        draw = float(torch.rand(1, generator=gen) * 0.2)
        cvar = float(torch.rand(1, generator=gen) * 0.1)
        out = reward_value(value, drawdown=draw, cvar=cvar)
        check("G18_REWARD_FINITE_CLIPPED", f"case={i + 1}", math.isfinite(out) and abs(out) <= 5.0)

    projection_source = (ROOT / "freqtrade/hedge/hprl/action_space.py").read_text(encoding="utf-8")
    reward_source = (ROOT / "freqtrade/hedge/hprl/reward.py").read_text(encoding="utf-8")
    hotpath = [
        ".any().item()" not in projection_source,
        "straight_through_quantize_unit_action" in projection_source,
        "_floor_index" in projection_source,
        "gaussian_tier_probabilities" in projection_source,
        "gaussian_tier_entropy" in projection_source,
        "gaussian_selected_tier_log_prob" in projection_source,
        "float(\"-inf\")" not in projection_source,
        "torch.log1p" in reward_source,
        "reward_clip" in reward_source,
        "terminal_loss" in reward_source,
    ]
    for i, value in enumerate(hotpath, 1):
        check("G19_GPU_HOTPATH", f"source={i}", value)

    integration = [
        (ROOT / "freqtrade/hedge/hprl/action_space.py").is_file(),
        (ROOT / "freqtrade/hedge/hprl/reward.py").is_file(),
        (ROOT / "freqtrade/hedge/hprl/env.py").is_file(),
        (ROOT / "freqtrade/hedge/hprl/trainer.py").is_file(),
        (ROOT / "freqtrade/hedge/hprl/runtime.py").is_file(),
        cfg.joint_states_per_symbol == 25,
        HPRLRewardConfig().return_scale == 100.0,
        HPRLRewardConfig().fees == 0.0,
        HPRLRewardConfig().funding == 0.0,
        HPRLTrainingConfig().tier_entropy_target_fraction == 0.65,
    ]
    for i, value in enumerate(integration, 1):
        check("G20_INTEGRATION", f"case={i}", value)

    passed = sum(1 for row in checks if row["passed"])
    failed = len(checks) - passed
    result = {
        "schema": "hprl-action-reward-v16-runtime-200",
        "expected": EXPECTED,
        "executed": len(checks),
        "passed": passed,
        "failed": failed,
        "status": "PASS" if len(checks) == EXPECTED and failed == 0 else "FAIL",
        "checks": checks,
    }
    print(
        f"HPRL ACTION REWARD RUNTIME 200: {passed}/{EXPECTED} PASS; FAIL={failed}",
        flush=True,
    )
    visible = {key: value for key, value in result.items() if key != "checks"}
    print(json.dumps(visible if args.summary_only else result, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
