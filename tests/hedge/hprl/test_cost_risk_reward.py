from __future__ import annotations

import pytest
import torch

from freqtrade.hedge.hprl.config import HPRLActionConfig, HPRLCostConfig, HPRLRewardConfig
from freqtrade.hedge.hprl.costs import ExecutionCostModel
from freqtrade.hedge.hprl.reward import CompositeReward, RewardFactsTensor
from freqtrade.hedge.hprl.risk import HedgeActionProjector


def test_costs_increase_with_turnover() -> None:
    model = ExecutionCostModel(HPRLCostConfig())
    equity = torch.tensor([1000.0, 1000.0])
    result = model.evaluate(
        turnover_notional=torch.tensor([10.0, 100.0]),
        equity=equity,
        available_notional=torch.tensor([10000.0, 10000.0]),
    )
    assert result.total[1] > result.total[0]
    assert torch.all(result.fees >= 0)
    assert torch.all(result.market_impact >= 0)


def test_maker_fraction_reduces_fee() -> None:
    model = ExecutionCostModel(HPRLCostConfig(maker_fee_bps=1.0, taker_fee_bps=5.0))
    turnover = torch.tensor([100.0, 100.0])
    result = model.evaluate(
        turnover_notional=turnover,
        equity=torch.tensor([1000.0, 1000.0]),
        maker_fraction=torch.tensor([1.0, 0.0]),
    )
    assert result.fees[0] < result.fees[1]


def test_risk_projector_allows_independent_legs_and_caps_gross() -> None:
    cfg = HPRLActionConfig(
        max_leg_exposure=1.0,
        max_gross_exposure=1.0,
        max_abs_net_exposure=1.0,
        max_step_change=1.0,
    )
    projector = HedgeActionProjector(cfg)
    current = torch.zeros((2, 1, 2))
    raw = torch.tensor([[[0.7, 0.2]], [[1.0, 1.0]]])
    result = projector.project(raw, current)
    assert torch.all(result.target >= 0)
    assert result.target[0, 0, 0] == pytest.approx(0.7)
    assert result.target[0, 0, 1] == pytest.approx(0.2)
    assert result.target[1].sum() <= 1.0 + 1e-6


def test_risk_projector_caps_step_change() -> None:
    cfg = HPRLActionConfig(max_step_change=0.1)
    current = torch.zeros((1, 1, 2))
    raw = torch.ones((1, 1, 2))
    result = HedgeActionProjector(cfg).project(raw, current)
    assert torch.all(result.target <= 0.1 + 1e-6)
    assert bool(result.projected_mask[0])


def test_low_liquidation_buffer_is_reduce_only() -> None:
    cfg = HPRLActionConfig(max_step_change=1.0, min_liquidation_buffer=0.2)
    current = torch.tensor([[[0.4, 0.2]]])
    raw = torch.tensor([[[0.8, 0.4]]])
    result = HedgeActionProjector(cfg).project(
        raw,
        current,
        liquidation_buffer=torch.tensor([0.1]),
    )
    assert torch.all(result.target <= current)


def test_composite_reward_penalizes_costs_and_projection() -> None:
    model = CompositeReward(
        HPRLRewardConfig(fees=1.0, slippage=1.0, market_impact=1.0, funding=1.0)
    )
    facts = RewardFactsTensor(
        equity_return=torch.tensor([0.01]),
        drawdown_increase=torch.tensor([0.0]),
        downside_return=torch.tensor([0.01]),
        cvar_loss=torch.tensor([0.0]),
        turnover_ratio=torch.tensor([0.1]),
        fee_ratio=torch.tensor([0.001]),
        slippage_ratio=torch.tensor([0.001]),
        impact_ratio=torch.tensor([0.001]),
        funding_ratio=torch.tensor([0.0]),
        projected=torch.tensor([True]),
    )
    total, parts = model.evaluate_tensor(facts)
    assert parts["equity"].item() > 0
    assert parts["fees"].item() < 0
    assert parts["risk_projection"].item() < 0
    assert torch.isfinite(total).all()
