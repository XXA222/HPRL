"""Scale-aware, risk-sensitive reward decomposition for dual-leg HPRL."""

from __future__ import annotations

from dataclasses import dataclass

from .config import HPRLRewardConfig
from .contracts import RewardBreakdown
from .device import require_torch


@dataclass(frozen=True, slots=True)
class RewardFactsTensor:
    equity_return: object
    drawdown_increase: object
    downside_return: object
    cvar_loss: object
    turnover_ratio: object
    fee_ratio: object
    slippage_ratio: object
    impact_ratio: object
    funding_ratio: object
    quantization_distance: object | None = None
    constraint_distance: object | None = None
    gross_margin_ratio: object | None = None
    hedge_overlap_ratio: object | None = None
    opportunity_miss: object | None = None
    terminal: object | None = None
    # V1.5 compatibility: old callers supplied a boolean projection mask only.
    projected: object | None = None


class CompositeReward:
    """Reward with explicit separation between economic PnL and policy/risk shaping.

    Equity growth is already net of fees, slippage, impact and funding in the environment.  Those
    components therefore default to zero as extra shaping to avoid accidental double counting.
    """

    def __init__(
        self, config: HPRLRewardConfig | None = None, *, validate_inputs: bool = True
    ) -> None:
        self.config = config or HPRLRewardConfig()
        self.validate_inputs = bool(validate_inputs)

    def evaluate_tensor(self, facts: RewardFactsTensor, *, return_components: bool = True):
        torch = require_torch()
        cfg = self.config
        zero = torch.zeros_like(facts.equity_return)
        quantization_distance = (
            zero if facts.quantization_distance is None else facts.quantization_distance
        )
        if facts.constraint_distance is None:
            constraint_distance = (
                zero
                if facts.projected is None
                else facts.projected.to(dtype=facts.equity_return.dtype)
            )
        else:
            constraint_distance = facts.constraint_distance
        gross_margin_ratio = zero if facts.gross_margin_ratio is None else facts.gross_margin_ratio
        hedge_overlap_ratio = (
            zero if facts.hedge_overlap_ratio is None else facts.hedge_overlap_ratio
        )
        opportunity_miss = zero if facts.opportunity_miss is None else facts.opportunity_miss
        terminal = (
            torch.zeros_like(facts.equity_return, dtype=torch.bool)
            if facts.terminal is None
            else facts.terminal
        )
        numeric = (
            facts.equity_return,
            facts.drawdown_increase,
            facts.downside_return,
            facts.cvar_loss,
            facts.turnover_ratio,
            facts.fee_ratio,
            facts.slippage_ratio,
            facts.impact_ratio,
            facts.funding_ratio,
            quantization_distance,
            constraint_distance,
            gross_margin_ratio,
            hedge_overlap_ratio,
            opportunity_miss,
        )
        base_shape = tuple(facts.equity_return.shape)
        if any(tuple(value.shape) != base_shape for value in numeric):
            raise ValueError("reward fact tensors must share the same shape")
        if tuple(terminal.shape) != base_shape:
            raise ValueError("reward terminal mask must match reward fact shape")
        if self.validate_inputs and any(not torch.isfinite(value).all() for value in numeric):
            raise ValueError("reward facts must be finite")

        scale = float(cfg.return_scale)
        safe_return = torch.clamp(facts.equity_return, min=-0.999999)
        log_growth = torch.log1p(safe_return)
        gross_excess = torch.clamp(
            gross_margin_ratio - cfg.gross_margin_soft_limit,
            min=0.0,
        )
        components = {} if return_components else None
        active = []

        def add(name: str, value):
            active.append(value)
            if components is not None:
                components[name] = value

        def inactive(name: str):
            if components is not None:
                components[name] = zero

        add("equity", cfg.equity * scale * log_growth)
        add(
            "drawdown",
            -cfg.drawdown * scale * torch.clamp(facts.drawdown_increase, min=0.0),
        )
        add(
            "downside",
            -cfg.downside * scale * torch.clamp(-facts.downside_return, min=0.0),
        )
        add("cvar", -cfg.cvar * scale * torch.clamp(facts.cvar_loss, min=0.0))
        add("turnover", -cfg.turnover * scale * torch.abs(facts.turnover_ratio))
        for name, weight, value, sign in (
            ("fees", cfg.fees, facts.fee_ratio, -1.0),
            ("slippage", cfg.slippage, facts.slippage_ratio, -1.0),
            ("market_impact", cfg.market_impact, facts.impact_ratio, -1.0),
            ("funding", cfg.funding, facts.funding_ratio, 1.0),
        ):
            if weight == 0.0:
                inactive(name)
            else:
                add(name, sign * weight * scale * (value if sign > 0 else torch.abs(value)))
        add(
            "quantization_alignment",
            -cfg.quantization_alignment * torch.clamp(quantization_distance, min=0.0),
        )
        add(
            "risk_projection",
            -cfg.risk_projection * torch.clamp(constraint_distance, min=0.0),
        )
        add("gross_margin_risk", -cfg.gross_margin_risk * gross_excess.square())
        if cfg.hedge_overlap == 0.0:
            inactive("hedge_overlap")
        else:
            add(
                "hedge_overlap",
                -cfg.hedge_overlap * torch.clamp(hedge_overlap_ratio, min=0.0),
            )
        if cfg.opportunity_cost == 0.0:
            inactive("opportunity_cost")
        else:
            add(
                "opportunity_cost",
                -cfg.opportunity_cost * torch.clamp(opportunity_miss, min=0.0),
            )
        add("terminal_loss", -cfg.terminal_loss * terminal.to(facts.equity_return.dtype))
        total = active[0]
        for value in active[1:]:
            total = total + value
        total = torch.clamp(total, -cfg.reward_clip, cfg.reward_clip)
        return total, components

    def evaluate_scalar(self, **values: float) -> RewardBreakdown:
        import torch

        facts = RewardFactsTensor(
            **{
                name: torch.tensor(float(value), dtype=torch.float64)
                for name, value in values.items()
            }
        )
        total, components = self.evaluate_tensor(facts)
        return RewardBreakdown(
            total=float(total.item()),
            components={name: float(value.item()) for name, value in components.items()},
        )
