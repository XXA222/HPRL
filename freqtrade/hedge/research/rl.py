"""RL experiment schedules, evaluation summaries, and promotion gates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean, pstdev


@dataclass(frozen=True, slots=True)
class RLExperimentConfig:
    algorithm: str = "MaskablePPO"
    total_timesteps: int = 100_000
    eval_interval: int = 10_000
    eval_episodes: int = 20
    seed: int = 1
    deterministic_eval: bool = True
    action_mask_required: bool = True

    def __post_init__(self) -> None:
        if not self.algorithm.strip():
            raise ValueError("RL algorithm cannot be empty")
        if self.total_timesteps < 1 or self.eval_interval < 1 or self.eval_episodes < 1:
            raise ValueError("RL training dimensions must be positive")
        if self.eval_interval > self.total_timesteps:
            raise ValueError("eval_interval cannot exceed total_timesteps")


def evaluation_schedule(config: RLExperimentConfig) -> tuple[int, ...]:
    values = list(range(config.eval_interval, config.total_timesteps + 1, config.eval_interval))
    if not values or values[-1] != config.total_timesteps:
        values.append(config.total_timesteps)
    return tuple(values)


def seed_schedule(seed: int, count: int) -> tuple[int, ...]:
    if count < 1:
        raise ValueError("seed count must be positive")
    return tuple(seed + index * 9973 for index in range(count))


def episode_summary(
    rewards: Sequence[float],
    drawdowns: Sequence[float] | None = None,
) -> dict[str, float]:
    values = tuple(float(item) for item in rewards)
    if not values or any(not math.isfinite(item) for item in values):
        raise ValueError("episode rewards must be finite")
    risk = tuple(float(item) for item in (drawdowns or (0.0,) * len(values)))
    if (
        len(risk) != len(values)
        or any(not 0 <= item <= 1 or not math.isfinite(item) for item in risk)
    ):
        raise ValueError("episode drawdowns are invalid")
    return {
        "mean_reward": fmean(values),
        "reward_std": pstdev(values) if len(values) > 1 else 0.0,
        "worst_reward": min(values),
        "best_reward": max(values),
        "mean_drawdown": fmean(risk),
        "max_drawdown": max(risk),
    }


def _action_mask_width(masks: Sequence[Sequence[bool]], hold_index: int) -> int:
    widths = {len(mask) for mask in masks}
    if len(widths) != 1:
        raise ValueError("action mask shapes are invalid")
    width = next(iter(widths))
    if width < 1 or not 0 <= hold_index < width:
        raise ValueError("action mask shapes are invalid")
    return width


def _validate_action_mask(mask: Sequence[bool], hold_index: int) -> None:
    if any(not isinstance(item, bool) for item in mask):
        raise ValueError("action mask shapes are invalid")
    if not any(mask) or not mask[hold_index]:
        raise ValueError("each action mask must retain HOLD and at least one valid action")


def action_mask_health(masks: Sequence[Sequence[bool]], *, hold_index: int = 0) -> dict[str, float]:
    if not masks:
        raise ValueError("action masks cannot be empty")
    width = _action_mask_width(masks, hold_index)
    for mask in masks:
        _validate_action_mask(mask, hold_index)
    valid_counts = [sum(mask) for mask in masks]
    mean_valid = fmean(valid_counts)
    return {
        "mean_valid_actions": mean_valid,
        "minimum_valid_actions": float(min(valid_counts)),
        "valid_action_ratio": mean_valid / width,
    }


def reward_component_balance(components: Sequence[dict[str, float]]) -> dict[str, float]:
    if not components:
        raise ValueError("reward components cannot be empty")
    names = sorted({name for row in components for name in row})
    result: dict[str, float] = {}
    for name in names:
        values = tuple(float(row.get(name, 0.0)) for row in components)
        if any(not math.isfinite(item) for item in values):
            raise ValueError("reward components must be finite")
        result[name] = fmean(values)
    return result


def promotion_decision(
    evaluation: dict[str, float],
    *,
    minimum_mean_reward: float,
    maximum_drawdown: float,
    maximum_reward_std: float | None = None,
) -> tuple[bool, tuple[str, ...]]:
    thresholds = (minimum_mean_reward, maximum_drawdown)
    if maximum_reward_std is not None:
        thresholds += (maximum_reward_std,)
    if any(not math.isfinite(float(item)) for item in thresholds):
        raise ValueError("RL promotion thresholds must be finite")
    for name in ("mean_reward", "max_drawdown", "reward_std"):
        if name in evaluation and not math.isfinite(float(evaluation[name])):
            raise ValueError("RL promotion metrics must be finite")
    violations: list[str] = []
    if evaluation.get("mean_reward", float("-inf")) < minimum_mean_reward:
        violations.append("mean_reward")
    if evaluation.get("max_drawdown", float("inf")) > maximum_drawdown:
        violations.append("max_drawdown")
    if (
        maximum_reward_std is not None
        and evaluation.get("reward_std", float("inf")) > maximum_reward_std
    ):
        violations.append("reward_std")
    return not violations, tuple(violations)


def compare_policies(rows: Sequence[dict[str, float]]) -> tuple[int, ...]:
    if not rows:
        raise ValueError("policy comparison requires rows")
    for row in rows:
        for name in ("mean_reward", "max_drawdown", "reward_std"):
            if name in row and not math.isfinite(float(row[name])):
                raise ValueError("policy comparison metrics must be finite")
    return tuple(
        sorted(
            range(len(rows)),
            key=lambda index: (
                float(rows[index].get("mean_reward", float("-inf"))),
                -float(rows[index].get("max_drawdown", float("inf"))),
                -float(rows[index].get("reward_std", float("inf"))),
            ),
            reverse=True,
        )
    )
