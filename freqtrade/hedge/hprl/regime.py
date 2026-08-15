"""Regime-adaptive continual policy layer inspired by ReCAP."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .device import require_torch


torch = require_torch()


@dataclass(frozen=True, slots=True)
class RegimeSignature:
    trend: float
    volatility: float
    downside: float

    def __post_init__(self) -> None:
        values = (self.trend, self.volatility, self.downside)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("regime signature values must be finite")
        if self.volatility < 0 or self.downside < 0:
            raise ValueError("regime volatility/downside values cannot be negative")

    def vector(self):
        return torch.tensor((self.trend, self.volatility, self.downside), dtype=torch.float32)


class AdaptiveRegimeDetector:
    """Deterministic rolling detector using trend, volatility, and downside intensity."""

    def __init__(self, *, trend_threshold: float = 0.5, high_vol_threshold: float = 0.01) -> None:
        if (
            not math.isfinite(trend_threshold)
            or not math.isfinite(high_vol_threshold)
            or trend_threshold <= 0
            or high_vol_threshold <= 0
        ):
            raise ValueError("regime thresholds must be positive")
        self.trend_threshold = trend_threshold
        self.high_vol_threshold = high_vol_threshold

    def signature(self, returns) -> RegimeSignature:
        values = torch.as_tensor(returns, dtype=torch.float32).flatten()
        if values.numel() < 4 or not torch.isfinite(values).all():
            raise ValueError("regime detection needs at least four finite returns")
        mean = values.mean()
        vol = values.std(unbiased=False).clamp_min(1e-8)
        downside = torch.clamp(-values, min=0.0).mean()
        return RegimeSignature(float(mean / vol), float(vol), float(downside / vol))

    def label(self, returns) -> str:
        sig = self.signature(returns)
        if sig.volatility >= self.high_vol_threshold:
            return "high_vol"
        if sig.trend >= self.trend_threshold:
            return "trend_up"
        if sig.trend <= -self.trend_threshold:
            return "trend_down"
        return "range"


class PolicyLibrary:
    """Named policy registry with regime signatures and soft composition weights."""

    def __init__(self) -> None:
        self._policies: dict[str, object] = {}
        self._signatures: dict[str, RegimeSignature] = {}

    def register(self, name: str, policy: object, signature: RegimeSignature) -> None:
        if not name.strip() or name in self._policies:
            raise ValueError("policy name must be unique and non-empty")
        self._policies[name] = policy
        self._signatures[name] = signature

    @property
    def policies(self) -> Mapping[str, object]:
        return dict(self._policies)

    def weights(self, current: RegimeSignature, *, temperature: float = 0.25) -> dict[str, float]:
        if not self._policies:
            raise ValueError("policy library is empty")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        target = current.vector()
        names = tuple(self._policies)
        distances = torch.stack(
            [torch.linalg.vector_norm(self._signatures[name].vector() - target) for name in names]
        )
        probs = torch.softmax(-distances / temperature, dim=0)
        return {name: float(prob) for name, prob in zip(names, probs, strict=True)}

    @torch.no_grad()
    def compose_action(self, obs, current: RegimeSignature, *, temperature: float = 0.25):
        weights = self.weights(current, temperature=temperature)
        result = None
        for name, weight in weights.items():
            policy = self._policies[name]
            action = policy.act(obs, deterministic=True)
            if action.shape != (obs.shape[0], action.shape[-1]) or not torch.isfinite(action).all():
                raise ValueError(f"policy {name!r} returned an invalid action batch")
            if result is not None and action.shape != result.shape:
                raise ValueError("regime policies must return the same action shape")
            result = action * weight if result is None else result + action * weight
        return result.clamp(0.0, 1.0)
