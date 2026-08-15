"""Hard continuous-action risk projection for dual-leg Hedge positions."""

from __future__ import annotations

from dataclasses import dataclass

from .config import HPRLActionConfig
from .device import require_torch


@dataclass(frozen=True, slots=True)
class TensorProjectionResult:
    target: object
    projected_mask: object
    projection_count: object


class HedgeActionProjector:
    """Project raw LONG/SHORT targets into a deterministic hard risk envelope.

    Expected action shape is ``[..., symbols, 2]`` where the final dimension is LONG, SHORT.
    The projection guarantees non-negative per-leg bounds, global gross exposure, and absolute net
    exposure. Risk-envelope reductions are allowed to override ``max_step_change`` because safety
    constraints take precedence over smooth policy movement.
    """

    def __init__(
        self, config: HPRLActionConfig | None = None, *, validate_inputs: bool = True
    ) -> None:
        self.config = config or HPRLActionConfig()
        self.validate_inputs = bool(validate_inputs)

    def project(self, raw_target, current, *, liquidation_buffer=None) -> TensorProjectionResult:
        torch = require_torch()
        if raw_target.shape != current.shape or raw_target.ndim < 2 or raw_target.shape[-1] != 2:
            raise ValueError(
                "HPRL actions and current positions must share [..., symbols, 2] shape"
            )
        if self.validate_inputs and (not torch.isfinite(current).all() or (current < 0).any()):
            raise ValueError("current positions must be finite and non-negative")
        original = raw_target
        target = torch.nan_to_num(
            original,
            nan=0.0,
            posinf=self.config.max_leg_exposure,
            neginf=0.0,
        )
        target = torch.clamp(target, 0.0, self.config.max_leg_exposure)
        delta = torch.clamp(
            target - current,
            -self.config.max_step_change,
            self.config.max_step_change,
        )
        target = torch.clamp(current + delta, 0.0, self.config.max_leg_exposure)

        gross = target.sum(dim=(-2, -1), keepdim=True)
        gross_scale = torch.clamp(
            self.config.max_gross_exposure / torch.clamp(gross, min=1e-12),
            max=1.0,
        )
        target = target * gross_scale

        # Enforce the net envelope exactly by reducing only the dominant side. For net > limit,
        # desired total long exposure is short_total + limit; the symmetric rule applies to shorts.
        long_total = target[..., 0].sum(dim=-1, keepdim=True)
        short_total = target[..., 1].sum(dim=-1, keepdim=True)
        net = long_total - short_total
        max_net = self.config.max_abs_net_exposure
        desired_long = torch.minimum(long_total, short_total + max_net)
        desired_short = torch.minimum(short_total, long_total + max_net)
        long_scale = desired_long / torch.clamp(long_total, min=1e-12)
        short_scale = desired_short / torch.clamp(short_total, min=1e-12)
        target = torch.stack(
            (target[..., 0] * long_scale, target[..., 1] * short_scale),
            dim=-1,
        )

        if liquidation_buffer is not None:
            buffer = torch.as_tensor(
                liquidation_buffer,
                device=target.device,
                dtype=target.dtype,
            )
            if self.validate_inputs and not torch.isfinite(buffer).all():
                raise ValueError("liquidation_buffer must be finite")
            defensive = buffer < self.config.min_liquidation_buffer
            while defensive.ndim < target.ndim:
                defensive = defensive.unsqueeze(-1)
            target = torch.where(defensive, torch.minimum(target, current), target)

        difference = (~torch.isclose(target, original, rtol=1e-5, atol=1e-7)).any(dim=(-2, -1))
        return TensorProjectionResult(
            target=target,
            projected_mask=difference,
            projection_count=difference.to(dtype=torch.int64),
        )
