"""Risk-aware ensemble and OOD routing inspired by FineFT."""

from __future__ import annotations

from dataclasses import dataclass

from .device import require_torch


torch = require_torch()


@dataclass(frozen=True, slots=True)
class OODScore:
    score: object
    in_distribution: object


class GaussianStateBoundary:
    """Fast diagonal-density capability boundary for policy routing.

    FineFT uses VAE-based boundaries; HPRL keeps the boundary interface pluggable and ships this
    deterministic low-overhead baseline so routing can be tested without tying the module to one
    generative architecture.
    """

    def __init__(self, quantile: float = 0.995) -> None:
        if not 0.5 < quantile < 1.0:
            raise ValueError("OOD quantile must be in (0.5, 1)")
        self.quantile = quantile
        self.mean = None
        self.var = None
        self.threshold = None

    def fit(self, states) -> "GaussianStateBoundary":
        states = torch.as_tensor(states, dtype=torch.float32)
        if states.ndim != 2 or states.shape[0] < 8 or not torch.isfinite(states).all():
            raise ValueError("OOD fit requires a finite [samples, features] tensor with >=8 rows")
        self.mean = states.mean(dim=0)
        self.var = states.var(dim=0, unbiased=False).clamp_min(1e-6)
        scores = ((states - self.mean).square() / self.var).mean(dim=-1)
        self.threshold = torch.quantile(scores, self.quantile)
        return self

    def score(self, states) -> OODScore:
        if self.mean is None or self.var is None or self.threshold is None:
            raise RuntimeError("OOD boundary must be fitted before score()")
        states = torch.as_tensor(states, dtype=self.mean.dtype, device=self.mean.device)
        if (
            states.ndim != 2
            or states.shape[1] != self.mean.shape[0]
            or not torch.isfinite(states).all()
        ):
            raise ValueError("OOD score requires finite [samples, fitted_features] states")
        score = ((states - self.mean).square() / self.var).mean(dim=-1)
        return OODScore(score, score <= self.threshold)


class RiskAwareEnsembleRouter:
    """Route among profitable specialist policies and a conservative fallback."""

    def __init__(self, conservative_policy: object) -> None:
        self.conservative_policy = conservative_policy
        self.specialists: dict[str, tuple[object, GaussianStateBoundary, float]] = {}

    def register(
        self, name: str, policy: object, boundary: GaussianStateBoundary, profitability: float
    ) -> None:
        if not name.strip() or name in self.specialists:
            raise ValueError("specialist name must be unique and non-empty")
        if not torch.isfinite(torch.tensor(profitability)):
            raise ValueError("profitability must be finite")
        self.specialists[name] = (policy, boundary, float(profitability))

    @torch.no_grad()
    def act(self, obs):
        if obs.ndim != 2 or not torch.isfinite(obs).all():
            raise ValueError("ensemble observations must be a finite [batch, features] tensor")
        output = self.conservative_policy.act(obs, deterministic=True).clone()
        if (
            output.ndim != 2
            or output.shape[0] != obs.shape[0]
            or not torch.isfinite(output).all()
        ):
            raise ValueError("conservative policy returned an invalid action batch")
        if not self.specialists:
            return output.clamp(0.0, 1.0)

        selected: list[str | None] = [None] * obs.shape[0]
        selected_profit = [0.0] * obs.shape[0]
        for name, (_, boundary, profitability) in self.specialists.items():
            if profitability <= 0.0:
                continue
            in_distribution = boundary.score(obs).in_distribution.tolist()
            for row, eligible in enumerate(in_distribution):
                if eligible and profitability > selected_profit[row]:
                    selected[row] = name
                    selected_profit[row] = profitability

        for name in sorted({value for value in selected if value is not None}):
            rows = [index for index, value in enumerate(selected) if value == name]
            policy = self.specialists[name][0]
            index = torch.tensor(rows, device=obs.device, dtype=torch.long)
            action = policy.act(obs.index_select(0, index), deterministic=True)
            if (
                action.shape != (len(rows), output.shape[1])
                or not torch.isfinite(action).all()
            ):
                raise ValueError(f"specialist {name!r} returned an invalid action shape/value")
            output.index_copy_(0, index, action.to(device=output.device, dtype=output.dtype))
        return output.clamp(0.0, 1.0)
