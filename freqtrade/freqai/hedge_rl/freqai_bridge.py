"""Shape-safe bridge between FreqAI rolling features and Hedge RL policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from .actions import DEFAULT_ACTION_CATALOG, HedgeActions
from .config import HedgeRLConfig
from .constraints import HedgeActionMasker
from .observation import HedgeObservationBuilder, ObservationSchema
from .state import HedgeAccountState


@dataclass(frozen=True, slots=True)
class HedgePolicyContext:
    """Account facts needed to reproduce the training observation at inference."""

    account: HedgeAccountState
    mark: float
    feature_age_steps: int = 0

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.mark)) or float(self.mark) <= 0:
            raise ValueError("policy context mark must be finite and positive")
        if self.feature_age_steps < 0:
            raise ValueError("feature_age_steps cannot be negative")

    @classmethod
    def neutral(cls, starting_balance: float, *, mark: float = 1.0) -> HedgePolicyContext:
        return cls(HedgeAccountState.initial(starting_balance), mark)


class HedgeFreqAIPolicyBridge:
    """Build exactly the flattened observation used by ``HedgeTradingEnv``.

    A live integration should supply a context provider backed by the Hedge account
    projection.  The neutral context is deliberately explicit and is suitable only
    for feature-only advisory/backtest inference.
    """

    def __init__(
        self,
        *,
        feature_names: tuple[str, ...],
        window_size: int,
        config: HedgeRLConfig,
    ) -> None:
        self.config = config
        self.schema = ObservationSchema(feature_names, window_size)
        self.builder = HedgeObservationBuilder(
            self.schema,
            feature_clip=config.feature_clip,
            normalize_market=False,
        )
        self.masker = HedgeActionMasker(config)

    def observation(
        self,
        features: npt.ArrayLike,
        *,
        tick: int,
        context: HedgePolicyContext,
    ) -> npt.NDArray[np.float32]:
        return self.builder.build(
            features,
            tick=tick,
            account=context.account,
            mark=context.mark,
            maintenance_rate=self.config.maintenance_margin_ratio,
            max_episode_steps=self.config.max_episode_steps,
        )

    def action_mask(self, context: HedgePolicyContext) -> npt.NDArray[np.bool_]:
        return self.masker.mask(account=context.account, mark=context.mark)

    def predict_action(
        self,
        model: Any,
        observation: npt.ArrayLike,
        *,
        context: HedgePolicyContext,
        use_masking: bool,
    ) -> HedgeActions:
        vector = np.asarray(observation, dtype=np.float32)
        if vector.shape != (self.schema.flat_size,) or not np.isfinite(vector).all():
            raise ValueError("policy observation shape or values are invalid")
        kwargs: dict[str, Any] = {"deterministic": True}
        if use_masking:
            kwargs["action_masks"] = self.action_mask(context)
        result, _ = model.predict(vector, **kwargs)
        action_values = np.asarray(result).reshape(-1)
        if action_values.size != 1:
            raise ValueError("Hedge policy must return exactly one discrete action")
        action = int(action_values[0])
        if not 0 <= action < len(DEFAULT_ACTION_CATALOG):
            raise ValueError(f"Hedge policy returned invalid action id {action}")
        return HedgeActions(action)
