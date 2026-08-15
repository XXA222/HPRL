"""Neural-network and training safety utilities (rounds 91-100)."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from .actions import DEFAULT_ACTION_CATALOG, HedgeActions
from .config import HedgeRLConfig
from .inference import HedgeInferenceGuard, InferenceDecision


_DEFAULT_ORTHOGONAL_GAIN = math.sqrt(2.0)


# Round 91 -------------------------------------------------------------------------------
def mask_action_logits(
    logits: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    invalid_value: float = -1e9,
) -> torch.Tensor:
    if logits.shape != action_mask.shape:
        raise ValueError("logits and action_mask must have identical shapes")
    if logits.shape[-1] != len(DEFAULT_ACTION_CATALOG):
        raise ValueError("last dimension must match the Hedge action catalogue")
    if action_mask.dtype is not torch.bool:
        action_mask = action_mask.to(dtype=torch.bool)
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite before masking")
    if not action_mask.any(dim=-1).all():
        raise ValueError("every batch row must permit at least one action")
    return torch.where(action_mask, logits, torch.full_like(logits, invalid_value))


# Round 92 -------------------------------------------------------------------------------
@dataclass(slots=True)
class RecurrentStateManager:
    layers: int
    batch_size: int
    hidden_size: int
    device: str = "cpu"
    _state: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if min(self.layers, self.batch_size, self.hidden_size) < 1:
            raise ValueError("recurrent state dimensions must be positive")
        self._state = torch.zeros(
            self.layers,
            self.batch_size,
            self.hidden_size,
            device=self.device,
        )

    @property
    def state(self) -> torch.Tensor:
        return self._state

    def update(self, state: torch.Tensor) -> None:
        if state.shape != self._state.shape or not torch.isfinite(state).all():
            raise ValueError("recurrent state has incompatible shape or non-finite values")
        self._state = state.detach()

    def reset(self, done: npt.ArrayLike | None = None) -> None:
        if done is None:
            self._state.zero_()
            return
        mask = np.asarray(done, dtype=np.bool_).reshape(-1)
        if mask.shape != (self.batch_size,):
            raise ValueError("done mask has incompatible batch size")
        self._state[:, torch.as_tensor(mask, device=self._state.device), :] = 0


# Round 93 -------------------------------------------------------------------------------
def orthogonal_initialize(
    module: nn.Module,
    *,
    gain: float = _DEFAULT_ORTHOGONAL_GAIN,
) -> nn.Module:
    if not math.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be finite and positive")
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=gain)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.GRU, nn.LSTM)):
            for name, parameter in layer.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(parameter)
                elif "weight_ih" in name:
                    nn.init.xavier_uniform_(parameter)
                elif "bias" in name:
                    nn.init.zeros_(parameter)
    return module


# Round 94 -------------------------------------------------------------------------------
class DistributionalValueHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        atoms: int = 51,
        minimum: float = -10.0,
        maximum: float = 10.0,
    ) -> None:
        super().__init__()
        if (
            min(input_dim, atoms) < 1
            or not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or maximum <= minimum
        ):
            raise ValueError("invalid distributional value dimensions or support")
        self.projection = nn.Linear(input_dim, atoms)
        self.register_buffer("support", torch.linspace(minimum, maximum, atoms))

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.projection(encoded)
        probabilities = torch.softmax(logits, dim=-1)
        expectation = torch.sum(probabilities * self.support, dim=-1)
        return logits, expectation


# Round 95 -------------------------------------------------------------------------------
class AuxiliaryRiskHead(nn.Module):
    OUTPUTS = ("drawdown_risk", "liquidation_risk", "volatility_risk")

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if min(input_dim, hidden_dim) < 1:
            raise ValueError("risk-head dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.OUTPUTS)),
        )

    def forward(self, encoded: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.network(encoded)
        probabilities = torch.sigmoid(raw)
        return {name: probabilities[..., index] for index, name in enumerate(self.OUTPUTS)}


# Round 96 -------------------------------------------------------------------------------
def finite_multitask_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim < 2:
        raise ValueError("prediction and target must have the same rank and shape")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("prediction and target must be finite")
    loss = torch.nn.functional.smooth_l1_loss(prediction, target, reduction="none")
    if weights is not None:
        if (
            weights.shape != (prediction.shape[-1],)
            or not torch.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("weights must be finite, non-negative, and match output width")
        loss = loss * weights
    result = loss.mean()
    if not torch.isfinite(result):
        raise FloatingPointError("multitask loss became non-finite")
    return result


# Round 97 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class GradientClipReport:
    total_norm_before: float
    maximum_norm: float
    clipped: bool


def clip_gradients(
    parameters: Iterable[nn.Parameter],
    *,
    maximum_norm: float,
) -> GradientClipReport:
    if not math.isfinite(maximum_norm) or maximum_norm <= 0:
        raise ValueError("maximum_norm must be finite and positive")
    trainable = [parameter for parameter in parameters if parameter.grad is not None]
    if not trainable:
        return GradientClipReport(0.0, maximum_norm, False)
    if any(not torch.isfinite(parameter.grad).all() for parameter in trainable):
        raise FloatingPointError("gradient contains non-finite values")
    norm = float(torch.nn.utils.clip_grad_norm_(trainable, maximum_norm))
    return GradientClipReport(norm, maximum_norm, norm > maximum_norm)


# Round 98 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    epsilon: float = 1e-8
    betas: tuple[float, float] = (0.9, 0.999)

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(float(item))
            for item in (self.learning_rate, self.weight_decay, self.epsilon, *self.betas)
        ):
            raise ValueError("optimizer values must be finite")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.epsilon <= 0:
            raise ValueError("invalid optimizer rates")
        if not 0 <= self.betas[0] < 1 or not 0 <= self.betas[1] < 1:
            raise ValueError("Adam betas must be within [0, 1)")

    def build(self, parameters: Iterable[nn.Parameter]) -> torch.optim.AdamW:
        return torch.optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=self.epsilon,
            betas=self.betas,
        )


# Round 99 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CheckpointCompatibility:
    source_version: str
    observation_signature: str
    action_signature: str
    architecture: str

    def __post_init__(self) -> None:
        for name in ("source_version", "observation_signature", "action_signature", "architecture"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")

    def validate(self, metadata: Mapping[str, Any]) -> tuple[bool, tuple[str, ...]]:
        expected = {
            "source_version": self.source_version,
            "observation_signature": self.observation_signature,
            "action_signature": self.action_signature,
            "architecture": self.architecture,
        }
        mismatches = tuple(
            name for name, value in expected.items() if str(metadata.get(name, "")) != value
        )
        return not mismatches, mismatches


# Round 100 ------------------------------------------------------------------------------
def fail_closed_policy_decision(
    logits: npt.ArrayLike,
    *,
    action_mask: npt.ArrayLike,
    feature_age_steps: int,
    config: HedgeRLConfig,
    model_compatible: bool,
    account_projection_fresh: bool,
) -> InferenceDecision:
    if not model_compatible or not account_projection_fresh:
        values = np.asarray(logits, dtype=np.float64).reshape(-1)
        requested = HedgeActions.HOLD
        if values.shape == (len(DEFAULT_ACTION_CATALOG),) and np.isfinite(values).any():
            requested = HedgeActions(int(np.nanargmax(values)))
        reasons = []
        if not model_compatible:
            reasons.append("MODEL_INCOMPATIBLE")
        if not account_projection_fresh:
            reasons.append("STALE_ACCOUNT_PROJECTION")
        return InferenceDecision(
            requested_action=requested,
            executed_action=HedgeActions.HOLD,
            confidence=0.0,
            normalized_entropy=1.0,
            shielded=True,
            reasons=tuple(reasons),
        )
    return HedgeInferenceGuard(config).decide(
        logits,
        action_mask=action_mask,
        feature_age_steps=feature_age_steps,
    )
