"""Decomposed account-level reward model for dual-leg Hedge reinforcement learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .config import HedgeRLConfig
from .portfolio import PortfolioTransition
from .state import HedgeAccountState


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    equity_return: float
    realized_pnl: float
    drawdown_penalty: float
    turnover_penalty: float
    gross_exposure_penalty: float
    net_exposure_penalty: float
    invalid_action_penalty: float
    liquidation_risk_penalty: float
    funding_cost_penalty: float
    unclipped_reward: float
    reward: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


class HedgeRewardModel:
    def __init__(self, config: HedgeRLConfig) -> None:
        self.config = config

    def calculate(
        self,
        *,
        transition: PortfolioTransition,
        account: HedgeAccountState,
        mark: float,
        invalid_action: bool = False,
    ) -> RewardBreakdown:
        base = max(abs(transition.previous_equity), 1e-12)
        weights = self.config.reward_weights
        equity_return = 100.0 * (transition.equity - transition.previous_equity) / base
        realized_return = 100.0 * transition.realized_pnl / base
        turnover_ratio = transition.traded_notional / base
        gross = account.gross_exposure(mark)
        net = abs(account.net_exposure(mark))
        margin_ratio = account.maintenance_margin_ratio(
            mark, self.config.maintenance_margin_ratio
        )
        drawdown_penalty = weights.drawdown * account.drawdown() ** 2
        turnover_penalty = weights.turnover * turnover_ratio
        gross_penalty = weights.gross_exposure * gross**2
        net_penalty = weights.net_exposure * net**2
        invalid_penalty = weights.invalid_action if invalid_action else 0.0
        liquidation_penalty = weights.liquidation_risk * max(0.0, margin_ratio - 0.5) ** 2
        funding_penalty = (
            weights.funding_cost * max(0.0, -transition.funding_cashflow) / base * 100.0
        )
        reward = (
            weights.equity_return * equity_return
            + weights.realized_pnl * realized_return
            - drawdown_penalty
            - turnover_penalty
            - gross_penalty
            - net_penalty
            - invalid_penalty
            - liquidation_penalty
            - funding_penalty
        )
        clipped = max(-self.config.reward_clip, min(self.config.reward_clip, reward))
        return RewardBreakdown(
            equity_return=equity_return,
            realized_pnl=realized_return,
            drawdown_penalty=drawdown_penalty,
            turnover_penalty=turnover_penalty,
            gross_exposure_penalty=gross_penalty,
            net_exposure_penalty=net_penalty,
            invalid_action_penalty=invalid_penalty,
            liquidation_risk_penalty=liquidation_penalty,
            funding_cost_penalty=funding_penalty,
            unclipped_reward=reward,
            reward=clipped,
        )
