"""Validated configuration for dual-leg Hedge reinforcement learning."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class RewardWeights:
    equity_return: float = 1.0
    realized_pnl: float = 0.25
    drawdown: float = 2.0
    turnover: float = 0.02
    gross_exposure: float = 0.05
    net_exposure: float = 0.05
    invalid_action: float = 1.0
    liquidation_risk: float = 4.0
    funding_cost: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not float(value) >= 0.0:
                raise ValueError(f"reward weight {name} must be a non-negative number")


@dataclass(frozen=True, slots=True)
class HedgeRLConfig:
    observation_window: int = 32
    max_episode_steps: int = 2048
    starting_balance: float = 10_000.0
    max_side_exposure: float = 0.80
    max_gross_exposure: float = 1.20
    max_net_exposure: float = 0.60
    maintenance_margin_ratio: float = 0.05
    drawdown_stop: float = 0.35
    fee_rate: float = 0.0004
    slippage_bps: float = 1.0
    funding_rate_per_step: float = 0.0
    reward_clip: float = 10.0
    action_size_fractions: tuple[float, ...] = (0.10, 0.25, 0.50)
    feature_clip: float = 10.0
    confidence_threshold: float = 0.55
    max_feature_age_steps: int = 1
    random_start: bool = True
    seed: int = 1
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    def __post_init__(self) -> None:
        self._validate_dimensions()
        self._validate_exposure_limits()
        self._validate_risk_and_costs()
        self._validate_action_fractions()

    def _validate_dimensions(self) -> None:
        if self.observation_window < 2:
            raise ValueError("observation_window must be at least 2")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be positive")
        if self.starting_balance <= 0:
            raise ValueError("starting_balance must be positive")

    def _validate_exposure_limits(self) -> None:
        for name in ("max_side_exposure", "max_gross_exposure", "max_net_exposure"):
            value = float(getattr(self, name))
            if not 0 < value <= 10:
                raise ValueError(f"{name} must be within (0, 10]")
        if self.max_gross_exposure < self.max_side_exposure:
            raise ValueError("max_gross_exposure cannot be below max_side_exposure")
        if self.max_net_exposure > self.max_gross_exposure:
            raise ValueError("max_net_exposure cannot exceed max_gross_exposure")

    def _validate_risk_and_costs(self) -> None:
        if not 0 < self.maintenance_margin_ratio < 1:
            raise ValueError("maintenance_margin_ratio must be within (0, 1)")
        if not 0 < self.drawdown_stop < 1:
            raise ValueError("drawdown_stop must be within (0, 1)")
        if not 0 <= self.fee_rate < 0.1:
            raise ValueError("fee_rate must be within [0, 0.1)")
        numeric_values = (
            self.slippage_bps,
            self.funding_rate_per_step,
            self.reward_clip,
            self.feature_clip,
        )
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("slippage, funding, and clip values must be finite")
        if self.slippage_bps < 0 or self.reward_clip <= 0 or self.feature_clip <= 0:
            raise ValueError("slippage_bps must be non-negative and clip values positive")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be within [0, 1]")
        if self.max_feature_age_steps < 0:
            raise ValueError("max_feature_age_steps cannot be negative")

    def _validate_action_fractions(self) -> None:
        if not self.action_size_fractions:
            raise ValueError("action_size_fractions cannot be empty")
        if tuple(sorted(set(self.action_size_fractions))) != self.action_size_fractions:
            raise ValueError("action_size_fractions must be unique and sorted")
        if any(not 0 < value <= 1 for value in self.action_size_fractions):
            raise ValueError("action_size_fractions must be within (0, 1]")

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> HedgeRLConfig:
        freqai = config.get("freqai", {})
        raw = freqai.get("hedge_rl_config", {})
        if not isinstance(raw, dict):
            raise TypeError("freqai.hedge_rl_config must be an object")
        values = dict(raw)
        reward_raw = values.pop("reward_weights", {})
        if not isinstance(reward_raw, dict):
            raise TypeError("reward_weights must be an object")
        if "action_size_fractions" in values:
            values["action_size_fractions"] = tuple(
                float(item) for item in values["action_size_fractions"]
            )
        values["reward_weights"] = RewardWeights(**reward_raw)
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["action_size_fractions"] = list(self.action_size_fractions)
        return result
