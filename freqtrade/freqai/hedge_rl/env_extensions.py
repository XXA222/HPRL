"""Environment reproducibility, snapshots, metrics, and scenarios (rounds 81-90)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import pandas as pd

from .actions import DEFAULT_ACTION_CATALOG, HedgeActions
from .curriculum import CurriculumScheduler
from .environment import HedgeTradingEnv
from .state import HedgeAccountState


# Round 81 -------------------------------------------------------------------------------
def verify_seed_determinism(
    env_factory,
    *,
    seed: int,
    actions: Sequence[int | HedgeActions],
) -> bool:
    env_a = env_factory()
    env_b = env_factory()
    obs_a, info_a = env_a.reset(seed=seed)
    obs_b, info_b = env_b.reset(seed=seed)
    if not np.array_equal(obs_a, obs_b) or info_a["tick"] != info_b["tick"]:
        return False
    for action in actions:
        transition_a = env_a.step(int(action))
        transition_b = env_b.step(int(action))
        if not np.array_equal(transition_a[0], transition_b[0]):
            return False
        if transition_a[1:4] != transition_b[1:4]:
            return False
        if transition_a[4]["executed_action"] != transition_b[4]["executed_action"]:
            return False
    return True


# Round 82 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HedgeEnvSnapshot:
    current_tick: int
    episode_steps: int
    terminated: bool
    truncated: bool
    account: HedgeAccountState
    rng_state: dict[str, object]

    @classmethod
    def capture(cls, env: HedgeTradingEnv) -> HedgeEnvSnapshot:
        return cls(
            current_tick=int(env._current_tick),
            episode_steps=int(env._episode_steps),
            terminated=bool(env._terminated),
            truncated=bool(env._truncated),
            account=env.simulator.state,
            rng_state=dict(env._rng.bit_generator.state),
        )

    def restore(self, env: HedgeTradingEnv) -> None:
        if not env._start_tick <= self.current_tick <= env._end_tick:
            raise ValueError("snapshot tick is outside the environment dataset")
        env._current_tick = self.current_tick
        env._episode_steps = self.episode_steps
        env._terminated = self.terminated
        env._truncated = self.truncated
        env.simulator.state = self.account
        env._rng.bit_generator.state = self.rng_state


# Round 83 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class NoLookaheadAudit:
    decision_tick: int
    execution_tick: int
    execution_open: float
    decision_close: float
    valid: bool


def audit_next_bar_execution(
    env: HedgeTradingEnv,
    *,
    action: int | HedgeActions,
) -> NoLookaheadAudit:
    env.reset(seed=env.seed_value)
    decision_tick = int(env._current_tick)
    decision_close = float(env.prices.iloc[decision_tick]["close"])
    execution_tick = decision_tick + 1
    execution_open = float(env.prices.iloc[execution_tick]["open"])
    _, _, _, _, info = env.step(int(action))
    valid = info["tick"] == execution_tick and env._current_tick == execution_tick
    # A distinct pair of prices makes accidental same-candle execution detectable.
    if execution_open != decision_close and info["tick"] != execution_tick:
        valid = False
    return NoLookaheadAudit(decision_tick, execution_tick, execution_open, decision_close, valid)


# Round 84 -------------------------------------------------------------------------------
def terminal_reason(
    account: HedgeAccountState,
    *,
    mark: float,
    maintenance_rate: float,
    drawdown_stop: float,
) -> str | None:
    if account.equity <= 0:
        return "NONPOSITIVE_EQUITY"
    if account.drawdown() >= drawdown_stop:
        return "DRAWDOWN_STOP"
    if account.maintenance_margin_ratio(mark, maintenance_rate) >= 1.0:
        return "LIQUIDATION_MARGIN"
    return None


# Round 85 -------------------------------------------------------------------------------
def truncation_reason(
    *,
    current_tick: int,
    end_tick: int,
    episode_steps: int,
    max_episode_steps: int,
) -> str | None:
    if min(current_tick, end_tick, episode_steps, max_episode_steps) < 0:
        raise ValueError("truncation counters cannot be negative")
    if current_tick >= end_tick:
        return "DATASET_END"
    if episode_steps >= max_episode_steps:
        return "MAX_EPISODE_STEPS"
    return None


# Round 86 -------------------------------------------------------------------------------
def invalid_action_fallback(
    requested: int | HedgeActions,
    action_mask: npt.ArrayLike,
) -> HedgeActions:
    mask = np.asarray(action_mask, dtype=np.bool_).reshape(-1)
    if mask.shape != (len(DEFAULT_ACTION_CATALOG),):
        raise ValueError("action mask has the wrong shape")
    action = HedgeActions(int(requested))
    if mask[int(action)]:
        return action
    if mask[int(HedgeActions.HOLD)]:
        return HedgeActions.HOLD
    valid = np.flatnonzero(mask)
    if not len(valid):
        raise RuntimeError("environment action mask contains no valid action")
    return HedgeActions(int(valid[0]))


# Round 87 -------------------------------------------------------------------------------
def assert_action_mask_invariants(mask: npt.ArrayLike) -> None:
    values = np.asarray(mask, dtype=np.bool_).reshape(-1)
    if values.shape != (len(DEFAULT_ACTION_CATALOG),):
        raise AssertionError("action mask size does not match the action catalogue")
    if not values.any():
        raise AssertionError("action mask must permit at least one action")
    if not values[int(HedgeActions.HOLD)]:
        raise AssertionError("HOLD must always remain available")


# Round 88 -------------------------------------------------------------------------------
@dataclass(slots=True)
class VectorObservationAdapter:
    observation_size: int

    def __post_init__(self) -> None:
        if self.observation_size < 1:
            raise ValueError("observation_size must be positive")

    def stack(self, observations: Iterable[npt.ArrayLike]) -> npt.NDArray[np.float32]:
        rows = [np.asarray(item, dtype=np.float32).reshape(-1) for item in observations]
        if not rows or any(row.shape != (self.observation_size,) for row in rows):
            raise ValueError("all observations must share the configured flat shape")
        result = np.stack(rows)
        if not np.isfinite(result).all():
            raise ValueError("vector observations must be finite")
        return result

    def unstack(self, batch: npt.ArrayLike) -> tuple[npt.NDArray[np.float32], ...]:
        array = np.asarray(batch, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != self.observation_size:
            raise ValueError("batch has an incompatible shape")
        return tuple(row.copy() for row in array)


# Round 89 -------------------------------------------------------------------------------
@dataclass(slots=True)
class EpisodeMetrics:
    starting_equity: float
    equity_curve: list[float]
    rewards: list[float]
    invalid_actions: int = 0
    turnover: float = 0.0

    @classmethod
    def start(cls, starting_equity: float) -> EpisodeMetrics:
        if not np.isfinite(starting_equity) or starting_equity <= 0:
            raise ValueError("starting_equity must be finite and positive")
        return cls(float(starting_equity), [float(starting_equity)], [])

    def update(
        self,
        *,
        equity: float,
        reward: float,
        invalid_action: bool,
        traded_notional: float,
    ) -> None:
        if (
            not all(np.isfinite(item) for item in (equity, reward, traded_notional))
            or traded_notional < 0
        ):
            raise ValueError("episode metrics inputs must be finite and turnover non-negative")
        self.equity_curve.append(float(equity))
        self.rewards.append(float(reward))
        self.invalid_actions += int(invalid_action)
        self.turnover += float(traded_notional)

    def summary(self) -> dict[str, float | int]:
        curve = np.asarray(self.equity_curve, dtype=float)
        running_peak = np.maximum.accumulate(curve)
        drawdowns = 1.0 - curve / running_peak
        returns = np.diff(np.log(np.maximum(curve, 1e-12)))
        return {
            "steps": len(self.rewards),
            "total_reward": float(sum(self.rewards)),
            "final_equity": float(curve[-1]),
            "equity_return": float(curve[-1] / self.starting_equity - 1.0),
            "max_drawdown": float(drawdowns.max(initial=0.0)),
            "return_volatility": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
            "invalid_actions": self.invalid_actions,
            "turnover": self.turnover,
        }


# Round 90 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ScenarioInjector:
    scheduler: CurriculumScheduler = field(default_factory=CurriculumScheduler)

    def inject(
        self,
        prices: pd.DataFrame,
        *,
        progress: float,
        seed: int,
        gap_at: int | None = None,
        price_shock_at: int | None = None,
        price_shock_fraction: float = 0.0,
    ) -> pd.DataFrame:
        result = self.scheduler.transform_prices(prices, progress=progress, seed=seed)
        if gap_at is not None:
            if not 0 <= gap_at < len(result):
                raise ValueError("gap_at is outside the dataset")
            result = result.drop(result.index[gap_at])
            result.attrs["injected_gap"] = gap_at
        if price_shock_at is not None:
            if not 0 <= price_shock_at < len(result) or not -0.95 < price_shock_fraction < 10:
                raise ValueError("invalid price shock parameters")
            multiplier = 1.0 + price_shock_fraction
            columns = [column for column in ("open", "high", "low", "close") if column in result]
            result.loc[result.index[price_shock_at] :, columns] *= multiplier
            result.attrs["injected_price_shock"] = price_shock_fraction
        return result
