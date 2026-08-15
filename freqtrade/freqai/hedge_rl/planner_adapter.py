"""Adapters from guarded RL actions to canonical Hedge planner signal semantics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from .actions import DEFAULT_ACTION_CATALOG, HedgeActions, LegCommand
from .config import HedgeRLConfig
from .inference import InferenceDecision
from .state import HedgeAccountState


@dataclass(frozen=True, slots=True)
class HedgeRLPlannerSignal:
    action: HedgeActions
    long_score: float
    short_score: float
    target_net_ratio: float
    risk_scale: float
    confidence: float
    normalized_entropy: float
    allow_new_risk: bool
    shielded: bool
    reason: str

    def __post_init__(self) -> None:
        if not 0 <= self.long_score <= 1 or not 0 <= self.short_score <= 1:
            raise ValueError("long_score and short_score must be within [0, 1]")
        if not -1 <= self.target_net_ratio <= 1:
            raise ValueError("target_net_ratio must be within [-1, 1]")
        if not 0 <= self.risk_scale <= 1:
            raise ValueError("risk_scale must be within [0, 1]")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be within [0, 1]")
        if not 0 <= self.normalized_entropy <= 1 + 1e-12:
            raise ValueError("normalized_entropy must be within [0, 1]")

    def strategy_columns(self) -> dict[str, float | int | bool | str]:
        return {
            "hedge_long_score": self.long_score,
            "hedge_short_score": self.short_score,
            "hedge_target_net_ratio": self.target_net_ratio,
            "hedge_risk_scale": self.risk_scale,
            "hedge_rl_confidence": self.confidence,
            "hedge_rl_entropy": self.normalized_entropy,
            "hedge_allow_new_risk": self.allow_new_risk,
            "hedge_rl_shielded": self.shielded,
            "hedge_rl_action": int(self.action),
            "hedge_rl_reason": self.reason,
        }


class HedgeRLPlannerAdapter:
    def __init__(self, config: HedgeRLConfig) -> None:
        self.config = config

    @staticmethod
    def _project(current: float, command: LegCommand, fraction: float) -> float:
        if command in {LegCommand.OPEN, LegCommand.INCREASE}:
            return current + fraction
        if command is LegCommand.REDUCE:
            return current * (1.0 - fraction)
        if command is LegCommand.CLOSE:
            return 0.0
        return current

    def from_decision(
        self,
        decision: InferenceDecision,
        *,
        account: HedgeAccountState,
        mark: float,
    ) -> HedgeRLPlannerSignal:
        spec = DEFAULT_ACTION_CATALOG.decode(decision.executed_action)
        equity = max(abs(account.equity), 1e-12)
        current_long = account.long.notional(mark) / equity
        current_short = account.short.notional(mark) / equity
        projected_long = max(
            0.0,
            self._project(current_long, spec.long_command, spec.long_fraction),
        )
        projected_short = max(
            0.0,
            self._project(current_short, spec.short_command, spec.short_fraction),
        )
        long_score = min(1.0, projected_long / self.config.max_side_exposure)
        short_score = min(1.0, projected_short / self.config.max_side_exposure)
        target_net = max(-1.0, min(1.0, projected_long - projected_short))
        risk_scale = max(
            0.0,
            min(1.0, 1.0 - account.drawdown() / self.config.drawdown_stop),
        )
        stale_or_uncertain = any(
            reason in {"STALE_FEATURES", "LOW_CONFIDENCE", "NONFINITE_LOGITS"}
            for reason in decision.reasons
        )
        allow_new_risk = not stale_or_uncertain and account.equity > 0 and risk_scale > 0
        reason = "RL:" + decision.executed_action.name
        if decision.reasons:
            reason += ":" + ",".join(decision.reasons)
        return HedgeRLPlannerSignal(
            action=decision.executed_action,
            long_score=long_score,
            short_score=short_score,
            target_net_ratio=target_net,
            risk_scale=risk_scale,
            confidence=decision.confidence,
            normalized_entropy=decision.normalized_entropy,
            allow_new_risk=allow_new_risk,
            shielded=decision.shielded,
            reason=reason,
        )

    def to_signal_snapshot(
        self,
        signal: HedgeRLPlannerSignal,
        *,
        pair: str,
        timeframe: str,
        candle_close_time: datetime,
        feature_timestamp: datetime,
        model_version: str,
    ):
        """Create the project's canonical SignalSnapshot without a hard import cycle."""

        from freqtrade.hedge.integration.signal_provider import SignalSnapshot

        return SignalSnapshot(
            symbol=pair,
            timeframe=timeframe,
            candle_close_time=candle_close_time,
            feature_timestamp=feature_timestamp,
            long_score=Decimal(str(signal.long_score)),
            short_score=Decimal(str(signal.short_score)),
            target_net=None,
            target_net_ratio=Decimal(str(signal.target_net_ratio)),
            model_version=model_version,
            reason=signal.reason,
            confidence=Decimal(str(signal.confidence)),
            risk_scale=Decimal(str(signal.risk_scale)),
            long_exposure_scale=Decimal(1),
            short_exposure_scale=Decimal(1),
            allow_new_risk=signal.allow_new_risk,
            regime="HEDGE_RL",
            strategy_reason=signal.reason,
        )
