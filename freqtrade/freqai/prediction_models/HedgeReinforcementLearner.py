"""Built-in FreqAI reinforcement learner using the dual-leg Hedge environment."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.hedge_rl.config import HedgeRLConfig
from freqtrade.freqai.hedge_rl.environment import HedgeTradingEnv
from freqtrade.freqai.hedge_rl.freqai_bridge import HedgeFreqAIPolicyBridge, HedgePolicyContext
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner


HedgeContextProvider = Callable[[str, int, object], HedgePolicyContext]


class HedgeReinforcementLearner(ReinforcementLearner):
    """Train PPO/MaskablePPO against a causal 21-action Hedge environment.

    Configure ``freqaimodel`` as ``HedgeReinforcementLearner`` and use one
    integer target column for the action id.  ``MaskablePPO`` is recommended.

    Live account-aware inference should register a context provider using
    :meth:`set_hedge_context_provider`.  Without one, inference intentionally
    uses a flat neutral account context; the resulting action remains advisory
    and must still pass the Hedge inference guard, planner, and risk engine.
    """

    MyRLEnv = HedgeTradingEnv

    def set_hedge_context_provider(self, provider: HedgeContextProvider | None) -> None:
        self._hedge_context_provider = provider

    def _policy_context(self, pair: str, tick: int, index_value: object) -> HedgePolicyContext:
        provider = None
        if hasattr(self, "_hedge_context_provider"):
            provider = self._hedge_context_provider
        if provider is None:
            config = HedgeRLConfig.from_config(self.config)
            return HedgePolicyContext.neutral(config.starting_balance)
        context = provider(pair, tick, index_value)
        if not isinstance(context, HedgePolicyContext):
            raise TypeError("Hedge context provider must return HedgePolicyContext")
        return context

    def rl_model_predict(
        self,
        dataframe: pd.DataFrame,
        dk: FreqaiDataKitchen,
        model,
    ) -> pd.DataFrame:
        """Predict one discrete action per complete causal feature window.

        The upstream implementation passes a two-dimensional rolling DataFrame
        directly to SB3.  Hedge training uses a flattened market window plus 12
        account features, so this override reconstructs that exact contract and
        prevents train/inference shape drift.
        """

        if len(dk.label_list) != 1:
            raise ValueError("HedgeReinforcementLearner requires exactly one action label")
        numeric = dataframe.apply(pd.to_numeric, errors="coerce")
        features = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(features).all():
            raise ValueError("Hedge RL prediction features must be finite")
        config = HedgeRLConfig.from_config(self.config)
        bridge = HedgeFreqAIPolicyBridge(
            feature_names=tuple(str(column) for column in dataframe.columns),
            window_size=self.CONV_WIDTH,
            config=config,
        )
        output = pd.DataFrame(0, index=dataframe.index, columns=dk.label_list, dtype=np.int64)
        use_masking = self.model_type == "MaskablePPO"
        for tick in range(self.CONV_WIDTH - 1, len(dataframe)):
            context = self._policy_context(dk.pair, tick, dataframe.index[tick])
            observation = bridge.observation(features, tick=tick, context=context)
            action = bridge.predict_action(
                model,
                observation,
                context=context,
                use_masking=use_masking,
            )
            output.iat[tick, 0] = int(action)
        return output
