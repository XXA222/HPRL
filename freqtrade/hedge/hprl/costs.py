"""Execution-friction models used by HPRL simulation.

The square-root impact model is intentionally pluggable. It models impact as a function of
participation rather than treating slippage as a constant. Impact remains monotonic when the
configured participation threshold is exceeded, so oversized simulated trades are not rewarded
with an artificially capped impact cost.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import HPRLCostConfig
from .device import require_torch


@dataclass(frozen=True, slots=True)
class CostBreakdownTensor:
    fees: object
    slippage: object
    market_impact: object

    @property
    def total(self):
        return self.fees + self.slippage + self.market_impact


class ExecutionCostModel:
    """Vectorized proportional fee, slippage and square-root market impact model."""

    def __init__(
        self, config: HPRLCostConfig | None = None, *, validate_inputs: bool = True
    ) -> None:
        self.config = config or HPRLCostConfig()
        self.validate_inputs = bool(validate_inputs)

    def evaluate(
        self,
        *,
        turnover_notional,
        equity,
        maker_fraction=None,
        available_notional=None,
    ) -> CostBreakdownTensor:
        torch = require_torch()
        turnover = torch.as_tensor(turnover_notional)
        if turnover.dtype not in (torch.float32, torch.float64):
            turnover = turnover.to(dtype=torch.float32)
        equity_tensor = torch.as_tensor(equity, device=turnover.device, dtype=turnover.dtype)
        if self.validate_inputs:
            if not torch.isfinite(turnover).all() or not torch.isfinite(equity_tensor).all():
                raise ValueError("turnover and equity must be finite")
            if (turnover < 0).any() or (equity_tensor <= 0).any():
                raise ValueError("turnover must be non-negative and equity must be positive")
        equity_safe = torch.clamp(equity_tensor, min=torch.finfo(turnover.dtype).eps)

        if maker_fraction is None:
            maker = torch.zeros_like(turnover)
        else:
            maker = torch.as_tensor(
                maker_fraction,
                device=turnover.device,
                dtype=turnover.dtype,
            )
            if self.validate_inputs and (
                not torch.isfinite(maker).all() or ((maker < 0) | (maker > 1)).any()
            ):
                raise ValueError("maker_fraction must be finite and within [0, 1]")
        fee_bps = maker * self.config.maker_fee_bps + (1.0 - maker) * self.config.taker_fee_bps
        fees = turnover * fee_bps * 1e-4
        slippage = turnover * self.config.base_slippage_bps * 1e-4

        if available_notional is None:
            available = equity_safe / self.config.max_participation
        else:
            available = torch.as_tensor(
                available_notional,
                device=turnover.device,
                dtype=turnover.dtype,
            )
            if self.validate_inputs and (
                not torch.isfinite(available).all() or (available <= 0).any()
            ):
                raise ValueError("available_notional must be finite and positive")
        available = torch.clamp(available, min=equity_safe * 1e-6)
        if self.config.impact_coefficient_bps == 0.0:
            market_impact = torch.zeros_like(turnover)
        else:
            raw_participation = torch.clamp(turnover / available, min=0.0)
            normalized = raw_participation / self.config.max_participation
            impact_bps = self.config.impact_coefficient_bps * torch.sqrt(normalized)
            market_impact = turnover * impact_bps * 1e-4
        outputs = (fees, slippage, market_impact)
        if self.validate_inputs and any(not torch.isfinite(value).all() for value in outputs):
            raise OverflowError("execution-cost arithmetic produced a non-finite result")
        return CostBreakdownTensor(fees, slippage, market_impact)
