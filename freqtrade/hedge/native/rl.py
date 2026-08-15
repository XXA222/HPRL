"""Dual-leg reinforcement-learning contracts for Hedge training.

The environment is vector-backend agnostic.  It exposes immutable numeric states,
explicit side/bucket actions and deterministic reward decomposition so callers can
place tensors and batched simulation on CPU or GPU without coupling live execution to
an RL library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import IntEnum
from typing import Any, Iterable, Mapping, Sequence

from .models import ONE, ZERO, finite_decimal


class HedgeRLAction(IntEnum):
    HOLD = 0
    INCREASE_LONG_CORE = 1
    INCREASE_LONG_TACTICAL = 2
    REDUCE_LONG = 3
    INCREASE_SHORT_CORE = 4
    INCREASE_SHORT_TACTICAL = 5
    REDUCE_SHORT = 6
    REDUCE_GROSS = 7
    CANCEL_ORDERS = 8
    EMERGENCY_REDUCE_ONLY = 9


@dataclass(frozen=True, slots=True)
class HedgeRLState:
    market_features: tuple[float, ...]
    long_quantity_ratio: float
    short_quantity_ratio: float
    long_profit_ratio: float
    short_profit_ratio: float
    core_long_ratio: float
    tactical_long_ratio: float
    core_short_ratio: float
    tactical_short_ratio: float
    gross_exposure_ratio: float
    net_exposure_ratio: float
    available_margin_ratio: float
    pending_margin_ratio: float
    funding_rate: float
    liquidation_buffer_ratio: float
    drawdown_ratio: float
    active_order_ratio: float
    regime_code: float = 0.0

    def vector(self) -> tuple[float, ...]:
        return self.market_features + (
            self.long_quantity_ratio,
            self.short_quantity_ratio,
            self.long_profit_ratio,
            self.short_profit_ratio,
            self.core_long_ratio,
            self.tactical_long_ratio,
            self.core_short_ratio,
            self.tactical_short_ratio,
            self.gross_exposure_ratio,
            self.net_exposure_ratio,
            self.available_margin_ratio,
            self.pending_margin_ratio,
            self.funding_rate,
            self.liquidation_buffer_ratio,
            self.drawdown_ratio,
            self.active_order_ratio,
            self.regime_code,
        )

    @property
    def dimension(self) -> int:
        return len(self.vector())

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
        *,
        market_features: Iterable[float] = (),
    ) -> "HedgeRLState":
        names = tuple(cls.__dataclass_fields__)[1:]
        kwargs = {name: float(values.get(name, 0.0)) for name in names}
        return cls(tuple(float(item) for item in market_features), **kwargs)


@dataclass(frozen=True, slots=True)
class HedgeRLRewardWeights:
    equity: Decimal = ONE
    risk_adjusted: Decimal = Decimal("0.25")
    funding: Decimal = Decimal("0.25")
    fees: Decimal = Decimal("0.50")
    slippage: Decimal = Decimal("0.50")
    drawdown: Decimal = Decimal("2")
    liquidation: Decimal = Decimal("5")
    turnover: Decimal = Decimal("0.10")
    core_damage: Decimal = Decimal("0.50")
    invalid_action: Decimal = ONE

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = finite_decimal(getattr(self, name), field_name=name)
            if value < ZERO:
                raise ValueError("reward weights cannot be negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class HedgeRLTransitionFacts:
    equity_change_ratio: Decimal = ZERO
    risk_adjusted_return: Decimal = ZERO
    funding_ratio: Decimal = ZERO
    fee_ratio: Decimal = ZERO
    slippage_ratio: Decimal = ZERO
    drawdown_increase: Decimal = ZERO
    liquidation_buffer_breach: Decimal = ZERO
    turnover_ratio: Decimal = ZERO
    core_damage_ratio: Decimal = ZERO
    invalid_action: bool = False

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name == "invalid_action":
                continue
            object.__setattr__(self, name, finite_decimal(getattr(self, name), field_name=name))


@dataclass(frozen=True, slots=True)
class HedgeRLReward:
    total: Decimal
    components: Mapping[str, Decimal]


class HedgeRLRewardFunction:
    def __init__(self, weights: HedgeRLRewardWeights | None = None) -> None:
        self.weights = weights or HedgeRLRewardWeights()

    def evaluate(self, facts: HedgeRLTransitionFacts) -> HedgeRLReward:
        components = {
            "equity": facts.equity_change_ratio * self.weights.equity,
            "risk_adjusted": facts.risk_adjusted_return * self.weights.risk_adjusted,
            "funding": facts.funding_ratio * self.weights.funding,
            "fees": -abs(facts.fee_ratio) * self.weights.fees,
            "slippage": -abs(facts.slippage_ratio) * self.weights.slippage,
            "drawdown": -max(facts.drawdown_increase, ZERO) * self.weights.drawdown,
            "liquidation": -max(facts.liquidation_buffer_breach, ZERO) * self.weights.liquidation,
            "turnover": -abs(facts.turnover_ratio) * self.weights.turnover,
            "core_damage": -max(facts.core_damage_ratio, ZERO) * self.weights.core_damage,
            "invalid_action": -self.weights.invalid_action if facts.invalid_action else ZERO,
        }
        return HedgeRLReward(sum(components.values(), ZERO), components)


@dataclass(frozen=True, slots=True)
class HedgeRLActionDecision:
    action: HedgeRLAction
    magnitude: Decimal
    allowed: bool
    reason: str
    reduce_only: bool


@dataclass(frozen=True, slots=True)
class HedgeRLActionLimits:
    max_step_ratio: Decimal = Decimal("0.05")
    min_liquidation_buffer: Decimal = Decimal("0.05")
    max_gross_exposure: Decimal = ONE
    require_confidence: Decimal = ZERO

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = finite_decimal(getattr(self, name), field_name=name)
            object.__setattr__(self, name, value)
        if not ZERO < self.max_step_ratio <= ONE:
            raise ValueError("max_step_ratio must be in (0, 1]")
        if self.min_liquidation_buffer < ZERO or self.max_gross_exposure <= ZERO:
            raise ValueError("invalid RL action risk limits")


class HedgeRLActionMask:
    """Return a conservative action mask from account risk state."""

    def __init__(self, limits: HedgeRLActionLimits | None = None) -> None:
        self.limits = limits or HedgeRLActionLimits()

    def mask(self, state: HedgeRLState, *, confidence: float = 1.0) -> tuple[bool, ...]:
        allowed = [True] * len(HedgeRLAction)
        new_risk = (
            state.gross_exposure_ratio < float(self.limits.max_gross_exposure)
            and state.liquidation_buffer_ratio >= float(self.limits.min_liquidation_buffer)
            and confidence >= float(self.limits.require_confidence)
        )
        for action in (
            HedgeRLAction.INCREASE_LONG_CORE,
            HedgeRLAction.INCREASE_LONG_TACTICAL,
            HedgeRLAction.INCREASE_SHORT_CORE,
            HedgeRLAction.INCREASE_SHORT_TACTICAL,
        ):
            allowed[int(action)] = new_risk
        allowed[int(HedgeRLAction.REDUCE_LONG)] = state.long_quantity_ratio > 0
        allowed[int(HedgeRLAction.REDUCE_SHORT)] = state.short_quantity_ratio > 0
        allowed[int(HedgeRLAction.REDUCE_GROSS)] = state.gross_exposure_ratio > 0
        allowed[int(HedgeRLAction.CANCEL_ORDERS)] = state.active_order_ratio > 0
        allowed[int(HedgeRLAction.EMERGENCY_REDUCE_ONLY)] = state.gross_exposure_ratio > 0
        return tuple(allowed)

    def decide(
        self,
        action: int | HedgeRLAction,
        magnitude: object,
        state: HedgeRLState,
        *,
        confidence: float = 1.0,
    ) -> HedgeRLActionDecision:
        selected = HedgeRLAction(int(action))
        size = min(max(finite_decimal(magnitude, field_name="magnitude"), ZERO), self.limits.max_step_ratio)
        allowed = self.mask(state, confidence=confidence)[int(selected)]
        reduce_only = selected in {
            HedgeRLAction.REDUCE_LONG,
            HedgeRLAction.REDUCE_SHORT,
            HedgeRLAction.REDUCE_GROSS,
            HedgeRLAction.EMERGENCY_REDUCE_ONLY,
        }
        return HedgeRLActionDecision(
            selected,
            size,
            allowed,
            "ACTION_ALLOWED" if allowed else "ACTION_MASKED_BY_RISK",
            reduce_only,
        )


@dataclass(slots=True)
class HedgeRLEpisodeLedger:
    """Deterministic episode accounting shared by single and vector environments."""

    initial_equity: Decimal
    peak_equity: Decimal = field(init=False)
    previous_equity: Decimal = field(init=False)
    cumulative_reward: Decimal = ZERO
    steps: int = 0

    def __post_init__(self) -> None:
        self.initial_equity = finite_decimal(self.initial_equity, field_name="initial_equity")
        if self.initial_equity <= ZERO:
            raise ValueError("initial_equity must be positive")
        self.peak_equity = self.initial_equity
        self.previous_equity = self.initial_equity

    def transition(
        self,
        *,
        equity: object,
        funding: object = ZERO,
        fees: object = ZERO,
        slippage: object = ZERO,
        turnover: object = ZERO,
        liquidation_buffer_breach: object = ZERO,
        core_damage: object = ZERO,
        invalid_action: bool = False,
        reward_function: HedgeRLRewardFunction | None = None,
    ) -> HedgeRLReward:
        current = finite_decimal(equity, field_name="equity")
        equity_change = (current - self.previous_equity) / self.initial_equity
        previous_drawdown = max(self.peak_equity - self.previous_equity, ZERO) / self.peak_equity
        self.peak_equity = max(self.peak_equity, current)
        current_drawdown = max(self.peak_equity - current, ZERO) / self.peak_equity
        facts = HedgeRLTransitionFacts(
            equity_change_ratio=equity_change,
            risk_adjusted_return=equity_change / max(Decimal("0.000001"), ONE + current_drawdown),
            funding_ratio=finite_decimal(funding, field_name="funding") / self.initial_equity,
            fee_ratio=finite_decimal(fees, field_name="fees") / self.initial_equity,
            slippage_ratio=finite_decimal(slippage, field_name="slippage") / self.initial_equity,
            drawdown_increase=max(current_drawdown - previous_drawdown, ZERO),
            liquidation_buffer_breach=finite_decimal(
                liquidation_buffer_breach, field_name="liquidation_buffer_breach"
            ),
            turnover_ratio=finite_decimal(turnover, field_name="turnover") / self.initial_equity,
            core_damage_ratio=finite_decimal(core_damage, field_name="core_damage") / self.initial_equity,
            invalid_action=invalid_action,
        )
        reward = (reward_function or HedgeRLRewardFunction()).evaluate(facts)
        self.previous_equity = current
        self.cumulative_reward += reward.total
        self.steps += 1
        return reward


class VectorHedgeRLEnvironment:
    """Small vector facade supporting hundreds of independent account ledgers."""

    def __init__(self, initial_equities: Sequence[object]) -> None:
        if not initial_equities:
            raise ValueError("at least one environment is required")
        self.ledgers = [HedgeRLEpisodeLedger(finite_decimal(item, field_name="initial_equity")) for item in initial_equities]

    @property
    def num_envs(self) -> int:
        return len(self.ledgers)

    def step_rewards(
        self,
        equities: Sequence[object],
        *,
        invalid_actions: Sequence[bool] | None = None,
    ) -> tuple[float, ...]:
        if len(equities) != self.num_envs:
            raise ValueError("equity batch length mismatch")
        invalid = tuple(invalid_actions or (False,) * self.num_envs)
        if len(invalid) != self.num_envs:
            raise ValueError("invalid-action batch length mismatch")
        return tuple(
            float(ledger.transition(equity=equity, invalid_action=bad).total)
            for ledger, equity, bad in zip(self.ledgers, equities, invalid, strict=True)
        )
