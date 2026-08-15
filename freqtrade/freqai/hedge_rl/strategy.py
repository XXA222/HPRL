"""Strategy-side helpers for exposing RL actions as canonical Hedge signal columns."""

from __future__ import annotations

import pandas as pd

from .actions import DEFAULT_ACTION_CATALOG, LegCommand


def _score(command: LegCommand, fraction: float) -> float:
    return fraction if command in {LegCommand.OPEN, LegCommand.INCREASE} else 0.0


def apply_hedge_rl_action_columns(
    dataframe: pd.DataFrame,
    *,
    action_column: str = "&-hedge_action",
) -> pd.DataFrame:
    if action_column not in dataframe:
        raise ValueError(f"missing RL action column {action_column!r}")
    result = dataframe.copy()
    long_scores: list[float] = []
    short_scores: list[float] = []
    target_net: list[float] = []
    reasons: list[str] = []
    for value in result[action_column].fillna(0):
        try:
            spec = DEFAULT_ACTION_CATALOG.decode(int(value))
        except (TypeError, ValueError):
            spec = DEFAULT_ACTION_CATALOG.decode(0)
        long_score = _score(spec.long_command, spec.long_fraction)
        short_score = _score(spec.short_command, spec.short_fraction)
        long_scores.append(long_score)
        short_scores.append(short_score)
        target_net.append(max(-1.0, min(1.0, long_score - short_score)))
        reasons.append(f"RL:{spec.action.name}")
    result["hedge_long_score"] = long_scores
    result["hedge_short_score"] = short_scores
    result["hedge_target_net_ratio"] = target_net
    result["hedge_rl_reason"] = reasons
    return result
