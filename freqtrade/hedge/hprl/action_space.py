"""Tier-aware dual-leg action codec for HPRL continuous-control agents."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .config import HPRLActionConfig
from .device import require_torch


torch = require_torch()
nn = torch.nn


def gaussian_tier_boundaries(level_count: int, *, device=None):
    """Build immutable FP32 sigmoid-Gaussian tier boundaries.

    Boundary ownership intentionally lives in registered buffers at runtime.  This
    helper is allocation-only and contains no process-global Tensor cache, making the
    compiled probability kernels pure functions of their Tensor inputs.
    """
    if level_count < 2:
        raise ValueError("level_count must be >= 2")
    inner = (
        torch.arange(1, level_count, device=device, dtype=torch.float32) - 0.5
    ) / float(level_count - 1)
    return torch.log(inner) - torch.log1p(-inner)


class TierBoundaryBuffers(nn.Module):
    """Agent-owned static tier tensors safe across torch.compile/CUDAGraph replay."""

    def __init__(self, level_count: int) -> None:
        super().__init__()
        self.level_count = int(level_count)
        self.register_buffer(
            "gaussian_boundaries",
            gaussian_tier_boundaries(self.level_count),
            persistent=False,
        )


@dataclass(frozen=True, slots=True)
class TieredActionResult:
    requested_policy: object
    requested_level_index: object
    requested_margin: object
    executed_policy: object
    executed_level_index: object
    target_margin: object
    target_notional: object
    joint_action_index: object
    quantization_distance: object
    constraint_distance: object
    transition_limited: object
    risk_limited: object
    projected_mask: object


def hard_quantize_unit_action(action, level_count: int):
    """Map [0,1] continuous latents to canonical evenly spaced tier codes."""
    torch = require_torch()
    if level_count < 2:
        raise ValueError("level_count must be >= 2")
    clipped = torch.clamp(action, 0.0, 1.0)
    scale = float(level_count - 1)
    return torch.floor(clipped * scale + 0.5) / scale


def straight_through_quantize_unit_action(action, level_count: int):
    """Hard tier value in the forward pass with identity gradient for actor optimization."""
    hard = hard_quantize_unit_action(action, level_count)
    return action + (hard - action).detach()


def configure_agent_action_levels(agent, level_count: int) -> None:
    """Attach tier count plus agent-owned registered boundaries.

    The buffer module has stable storage for the lifetime of the agent and is passed
    explicitly into compiled tier kernels.  No compiled call reaches into a Python
    global Tensor cache.
    """
    if level_count < 2:
        raise ValueError("level_count must be >= 2")
    count = int(level_count)
    current = getattr(agent, "_tier_buffers", None)
    if not isinstance(current, TierBoundaryBuffers) or current.level_count != count:
        device = getattr(agent, "device", "cpu")
        current = TierBoundaryBuffers(count).to(device)
        setattr(agent, "_tier_buffers", current)
    setattr(agent, "action_level_count", count)


def agent_tier_boundaries(agent):
    buffers = getattr(agent, "_tier_buffers", None)
    if buffers is None:
        raise RuntimeError("agent tier buffers are not configured")
    return buffers.gaussian_boundaries


def action_for_critic(agent, action, *, straight_through: bool):
    level_count = int(getattr(agent, "action_level_count", 0) or 0)
    if level_count < 2:
        return action
    if straight_through:
        return straight_through_quantize_unit_action(action, level_count)
    return hard_quantize_unit_action(action, level_count)


def gaussian_tier_probabilities_from_boundaries(mean, log_std, boundaries):
    """Pure categorical tier probability kernel with explicit static boundaries."""
    mean32 = mean.float()
    log_std32 = log_std.float()
    std = log_std32.exp().clamp_min(1e-8)
    z = (boundaries - mean32.unsqueeze(-1)) / std.unsqueeze(-1)
    inner_cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    first = inner_cdf[..., :1]
    middle = inner_cdf[..., 1:] - inner_cdf[..., :-1]
    last = 1.0 - inner_cdf[..., -1:]
    probs = torch.cat((first, middle, last), dim=-1).clamp_min(1e-8)
    return probs / probs.sum(dim=-1, keepdim=True)


def gaussian_tier_entropy_from_boundaries(mean, log_std, boundaries):
    """Pure executed-tier entropy kernel for torch.compile/CUDAGraph hot paths."""
    probs = gaussian_tier_probabilities_from_boundaries(mean, log_std, boundaries)
    return -(probs * torch.log(probs)).sum(dim=-1)


def gaussian_selected_tier_log_prob_from_boundaries(mean, log_std, action, boundaries):
    """Pure selected-tier log-probability kernel with explicit boundaries."""
    level_count = int(boundaries.numel()) + 1
    mean32 = mean.float()
    log_std32 = log_std.float()
    std = log_std32.exp().clamp_min(1e-8)
    index = torch.round(action.float() * float(level_count - 1)).to(torch.int64)
    index = index.clamp(0, level_count - 1)
    lower_index = (index - 1).clamp(0, level_count - 2)
    upper_index = index.clamp(0, level_count - 2)
    lower_logit = boundaries[lower_index]
    upper_logit = boundaries[upper_index]
    inv_std = std.reciprocal()
    inv_sqrt_two = 1.0 / math.sqrt(2.0)
    lower_cdf = 0.5 * (1.0 + torch.erf((lower_logit - mean32) * inv_std * inv_sqrt_two))
    upper_cdf = 0.5 * (1.0 + torch.erf((upper_logit - mean32) * inv_std * inv_sqrt_two))
    selected = torch.where(
        index == 0,
        upper_cdf,
        torch.where(index == level_count - 1, 1.0 - lower_cdf, upper_cdf - lower_cdf),
    )
    return torch.log(selected.clamp_min(1e-8))


def gaussian_tier_probabilities(mean, log_std, level_count: int):
    """Compatibility wrapper; training hot paths use the explicit-boundary kernel."""
    boundaries = gaussian_tier_boundaries(level_count, device=mean.device)
    return gaussian_tier_probabilities_from_boundaries(mean, log_std, boundaries)


def gaussian_tier_entropy(mean, log_std, level_count: int):
    boundaries = gaussian_tier_boundaries(level_count, device=mean.device)
    return gaussian_tier_entropy_from_boundaries(mean, log_std, boundaries)


def gaussian_selected_tier_log_prob(mean, log_std, action, level_count: int):
    boundaries = gaussian_tier_boundaries(level_count, device=mean.device)
    return gaussian_selected_tier_log_prob_from_boundaries(mean, log_std, action, boundaries)


class TieredHedgeActionCodec(nn.Module):
    """Decode policy latents into exact margin-budget tiers and apply hard account envelopes.

    The policy surface remains [0, 1] per LONG/SHORT leg.  The executed surface is a finite grid,
    e.g. five levels produce 5 x 5 = 25 joint states per symbol.  Global gross/net constraints are
    enforced by monotonically de-risking tier indices, so the final target always remains on-grid.
    """

    def __init__(self, config: HPRLActionConfig, *, validate_inputs: bool = True, device=None) -> None:
        super().__init__()
        if config.mode.strip().lower() != "tiered":
            raise ValueError("TieredHedgeActionCodec requires action.mode='tiered'")
        self.config = config
        self.validate_inputs = bool(validate_inputs)
        self.register_buffer(
            "position_levels",
            torch.tensor(self.config.position_levels, dtype=torch.float32, device=device),
            persistent=False,
        )
        self.register_buffer(
            "level_indices",
            torch.arange(self.config.level_count, dtype=torch.int64, device=device),
            persistent=False,
        )

    def _levels(self, reference):
        levels = self.position_levels
        if levels.device != reference.device or levels.dtype != reference.dtype:
            return levels.to(device=reference.device, dtype=reference.dtype)
        return levels

    def _index_from_margin(self, margin):
        torch = require_torch()
        levels = self._levels(margin)
        distance = torch.abs(margin.unsqueeze(-1) - levels)
        return distance.argmin(dim=-1).to(torch.int64)

    def _margin_from_index(self, index, reference):
        levels = self._levels(reference)
        return levels[index]

    def _floor_index(self, desired_margin, reference):
        """Highest configured tier not exceeding desired margin, fully vectorized."""
        torch = require_torch()
        levels = self._levels(reference)
        eligible = levels.reshape(*([1] * desired_margin.ndim), -1) <= (
            desired_margin.unsqueeze(-1) + 1e-8
        )
        level_indices = self.level_indices
        if level_indices.device != levels.device:
            level_indices = level_indices.to(levels.device)
        candidates = torch.where(eligible, level_indices, torch.zeros_like(level_indices))
        return candidates.max(dim=-1).values

    def _enforce_account_envelope(self, indices, reference):
        """Project tier indices through gross/net account limits without iterative kernels."""
        torch = require_torch()
        cfg = self.config
        original = indices
        margin = self._margin_from_index(indices, reference)

        gross = margin.sum(dim=(-2, -1), keepdim=True)
        gross_scale = torch.clamp(
            cfg.max_gross_margin_ratio / torch.clamp(gross, min=1e-12),
            max=1.0,
        )
        gross_desired = margin * gross_scale
        indices = self._floor_index(gross_desired, reference)
        margin = self._margin_from_index(indices, reference)

        long_total = margin[..., 0].sum(dim=-1, keepdim=True)
        short_total = margin[..., 1].sum(dim=-1, keepdim=True)
        desired_long_total = torch.minimum(
            long_total,
            short_total + cfg.max_abs_net_margin_ratio,
        )
        long_scale = desired_long_total / torch.clamp(long_total, min=1e-12)
        long_desired = margin[..., 0] * long_scale
        long_index = self._floor_index(long_desired, reference)
        indices = torch.stack((long_index, indices[..., 1]), dim=-1)
        margin = self._margin_from_index(indices, reference)

        long_total = margin[..., 0].sum(dim=-1, keepdim=True)
        short_total = margin[..., 1].sum(dim=-1, keepdim=True)
        desired_short_total = torch.minimum(
            short_total,
            long_total + cfg.max_abs_net_margin_ratio,
        )
        short_scale = desired_short_total / torch.clamp(short_total, min=1e-12)
        short_desired = margin[..., 1] * short_scale
        short_index = self._floor_index(short_desired, reference)
        indices = torch.stack((indices[..., 0], short_index), dim=-1)

        risk_limited = (indices != original).any(dim=(-2, -1))
        return indices, risk_limited

    def decode(self, raw_action, current_margin, *, liquidation_buffer=None) -> TieredActionResult:
        torch = require_torch()
        if raw_action.shape != current_margin.shape or raw_action.ndim < 2:
            raise ValueError("raw_action/current_margin must share [..., symbols, 2] shape")
        if raw_action.shape[-1] != 2:
            raise ValueError("final action dimension must be LONG/SHORT")
        if self.validate_inputs:
            if not torch.isfinite(raw_action).all() or not torch.isfinite(current_margin).all():
                raise ValueError("tiered action inputs must be finite")
            if (current_margin < 0).any():
                raise ValueError("current_margin cannot be negative")

        cfg = self.config
        requested_policy = torch.clamp(raw_action, 0.0, 1.0)
        scale = float(cfg.level_count - 1)
        requested_index = torch.floor(requested_policy * scale + 0.5).to(torch.int64)
        current_index = self._index_from_margin(current_margin)
        if cfg.tier_hysteresis > 0.0:
            current_code = current_index.to(requested_policy.dtype) / scale
            half_step = 0.5 / scale
            up_threshold = current_code + half_step + float(cfg.tier_hysteresis)
            down_threshold = current_code - half_step - float(cfg.tier_hysteresis)
            hold_up = (requested_index > current_index) & (requested_policy < up_threshold)
            hold_down = (requested_index < current_index) & (requested_policy > down_threshold)
            requested_index = torch.where(hold_up | hold_down, current_index, requested_index)
        requested_margin = self._margin_from_index(requested_index, requested_policy)
        requested_canonical = requested_index.to(requested_policy.dtype) / scale
        quantization_distance = torch.abs(requested_policy - requested_canonical).mean(
            dim=(-2, -1)
        )

        lower = (
            torch.zeros_like(current_index)
            if cfg.max_decrease_levels == -1
            else current_index - cfg.max_decrease_levels
        )
        upper = current_index + cfg.max_increase_levels
        transitioned = torch.minimum(torch.maximum(requested_index, lower), upper)
        transitioned = torch.clamp(transitioned, 0, cfg.level_count - 1)
        transition_limited = (transitioned != requested_index).any(dim=(-2, -1))

        if liquidation_buffer is not None:
            buffer = torch.as_tensor(
                liquidation_buffer,
                device=transitioned.device,
                dtype=requested_policy.dtype,
            )
            defensive = buffer < cfg.min_liquidation_buffer
            while defensive.ndim < transitioned.ndim:
                defensive = defensive.unsqueeze(-1)
            transitioned = torch.where(
                defensive, torch.minimum(transitioned, current_index), transitioned
            )

        executed_index, risk_limited = self._enforce_account_envelope(
            transitioned, requested_policy
        )
        target_margin = self._margin_from_index(executed_index, requested_policy)
        target_notional = target_margin * float(cfg.leverage)
        joint_action_index = (
            executed_index[..., 0] * cfg.level_count + executed_index[..., 1]
        )
        executed_policy = executed_index.to(requested_policy.dtype) / scale
        constraint_distance = torch.abs(executed_policy - requested_canonical).mean(
            dim=(-2, -1)
        )
        projected_mask = transition_limited | risk_limited
        return TieredActionResult(
            requested_policy=requested_policy,
            requested_level_index=requested_index,
            requested_margin=requested_margin,
            executed_policy=executed_policy,
            executed_level_index=executed_index,
            target_margin=target_margin,
            target_notional=target_notional,
            joint_action_index=joint_action_index,
            quantization_distance=quantization_distance,
            constraint_distance=constraint_distance,
            transition_limited=transition_limited,
            risk_limited=risk_limited,
            projected_mask=projected_mask,
        )


def canonicalize_offline_action_tensor(action, config: HPRLActionConfig, action_unit: str):
    """Convert declared offline action units to the canonical [0,1] tier-code domain.

    Conversion is intentionally strict: offline datasets must declare whether stored values are
    policy codes, margin budgets, or notional exposures.  Values not lying on a configured tier are
    rejected instead of silently changing historical behavior data.
    """
    torch = require_torch()
    unit = action_unit.strip().lower() if isinstance(action_unit, str) else ""
    if unit not in {"policy_code", "margin_budget", "notional_exposure"}:
        raise ValueError(
            "offline action_unit must be policy_code/margin_budget/notional_exposure"
        )
    if config.mode != "tiered":
        if unit != "policy_code":
            raise ValueError("continuous action mode requires offline action_unit='policy_code'")
        return action
    if not torch.isfinite(action).all():
        raise ValueError("offline actions must be finite")

    scale = float(config.level_count - 1)
    if unit == "policy_code":
        if (action < -1e-6).any() or (action > 1.0 + 1e-6).any():
            raise ValueError("offline policy_code actions must be within [0, 1]")
        canonical = hard_quantize_unit_action(action, config.level_count)
        if not torch.allclose(action, canonical, rtol=0.0, atol=1e-5):
            raise ValueError("offline policy_code contains values outside the configured tier grid")
        return canonical

    margin = action if unit == "margin_budget" else action / float(config.leverage)
    if (margin < -1e-7).any():
        raise ValueError("offline margin/notional actions cannot be negative")
    levels = torch.tensor(config.position_levels, dtype=margin.dtype, device=margin.device)
    distances = torch.abs(margin.unsqueeze(-1) - levels)
    min_distance, index = distances.min(dim=-1)
    tolerance = max(1e-6, 1e-5 * float(max(config.position_levels[-1], 1.0)))
    if (min_distance > tolerance).any():
        raise ValueError("offline action is not aligned to a configured position tier")
    return index.to(margin.dtype) / scale
