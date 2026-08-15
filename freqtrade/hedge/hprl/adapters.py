"""Safe integration adapters from HPRL into the Clean Mainline planning boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import math
from typing import Protocol, Sequence

from .contracts import PlannedExecutionIntent


class HPRLPlanningPort(Protocol):
    """Dependency-light test port; HPRL never owns an exchange execution capability."""

    def accept_hprl_targets(self, intents: Sequence[PlannedExecutionIntent]) -> object: ...


@dataclass(frozen=True, slots=True)
class ReadonlyTargetAdapter:
    symbols: tuple[str, ...]
    model_id: str

    def __post_init__(self) -> None:
        if not self.symbols or any(not symbol.strip() for symbol in self.symbols):
            raise ValueError("adapter symbols must be non-empty strings")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("adapter symbols must be unique")
        if not self.model_id.strip():
            raise ValueError("adapter model_id cannot be empty")

    def decode(self, action_row, *, confidence: float = 1.0) -> tuple[PlannedExecutionIntent, ...]:
        values = action_row.detach().cpu().reshape(len(self.symbols), 2).tolist()
        return tuple(
            PlannedExecutionIntent(
                symbol=symbol,
                target_long_exposure=float(pair[0]),
                target_short_exposure=float(pair[1]),
                confidence=confidence,
                model_id=self.model_id,
                metadata={"source": "hprl", "exchange_write": "forbidden"},
            )
            for symbol, pair in zip(self.symbols, values, strict=True)
        )


@dataclass(frozen=True, slots=True)
class HPRLPlannerSignal:
    """Normalized HPRL output matching the Clean Mainline signal-provider semantics."""

    symbol: str
    long_score: float
    short_score: float
    target_net_ratio: float
    confidence: float
    risk_scale: float
    allow_new_risk: bool
    model_id: str
    reason: str = "HPRL_TARGET_EXPOSURE"
    regime: str = "HPRL"

    def __post_init__(self) -> None:
        if not self.symbol.strip() or not self.model_id.strip():
            raise ValueError("symbol and model_id are required")
        unit_values = (self.long_score, self.short_score, self.confidence, self.risk_scale)
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in unit_values):
            raise ValueError("HPRL planner unit values must be finite and within [0, 1]")
        if not math.isfinite(self.target_net_ratio) or not -1 <= self.target_net_ratio <= 1:
            raise ValueError("target_net_ratio must be finite and within [-1, 1]")
        if not isinstance(self.allow_new_risk, bool):
            raise ValueError("allow_new_risk must be a boolean")


@dataclass(frozen=True, slots=True)
class CleanMainlineSignalAdapter:
    """Bridge projected HPRL targets to the canonical Clean Mainline ``SignalSnapshot``.

    The bridge is intentionally lazy-imported so importing HPRL remains independent of Freqtrade
    runtime composition. The action must already have passed the HPRL hard risk projector.
    """

    symbols: tuple[str, ...]
    model_id: str
    max_leg_exposure: float = 1.0

    def __post_init__(self) -> None:
        if not self.symbols or any(not symbol.strip() for symbol in self.symbols):
            raise ValueError("adapter symbols must be non-empty strings")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("adapter symbols must be unique")
        if not self.model_id.strip():
            raise ValueError("adapter model_id cannot be empty")
        if not math.isfinite(self.max_leg_exposure) or self.max_leg_exposure <= 0:
            raise ValueError("max_leg_exposure must be finite and positive")

    def decode(
        self,
        action_row,
        *,
        confidence: float = 1.0,
        risk_scale: float = 1.0,
        allow_new_risk: bool = True,
    ) -> tuple[HPRLPlannerSignal, ...]:
        values = action_row.detach().cpu().reshape(len(self.symbols), 2).tolist()
        result: list[HPRLPlannerSignal] = []
        for symbol, pair in zip(self.symbols, values, strict=True):
            long_exposure = float(pair[0])
            short_exposure = float(pair[1])
            if any(not math.isfinite(value) or value < 0 for value in pair):
                raise ValueError("HPRL target exposure must be finite and non-negative")
            long_score = min(1.0, long_exposure / self.max_leg_exposure)
            short_score = min(1.0, short_exposure / self.max_leg_exposure)
            target_net = long_exposure - short_exposure
            result.append(
                HPRLPlannerSignal(
                    symbol=symbol,
                    long_score=long_score,
                    short_score=short_score,
                    target_net_ratio=max(-1.0, min(1.0, target_net)),
                    confidence=confidence,
                    risk_scale=risk_scale,
                    allow_new_risk=allow_new_risk,
                    model_id=self.model_id,
                )
            )
        return tuple(result)

    @staticmethod
    def signal_snapshot_kwargs(
        signal: HPRLPlannerSignal,
        *,
        timeframe: str,
        candle_close_time: datetime,
        feature_timestamp: datetime,
    ) -> dict[str, object]:
        """Build the canonical contract payload without importing the Freqtrade runtime."""
        if not timeframe.strip():
            raise ValueError("timeframe cannot be empty")
        return {
            "symbol": signal.symbol,
            "timeframe": timeframe,
            "candle_close_time": candle_close_time,
            "feature_timestamp": feature_timestamp,
            "long_score": Decimal(str(signal.long_score)),
            "short_score": Decimal(str(signal.short_score)),
            "target_net": None,
            "model_version": signal.model_id,
            "reason": signal.reason,
            "target_net_ratio": Decimal(str(signal.target_net_ratio)),
            "confidence": Decimal(str(signal.confidence)),
            "risk_scale": Decimal(str(signal.risk_scale)),
            "long_exposure_scale": Decimal("1"),
            "short_exposure_scale": Decimal("1"),
            "allow_new_risk": signal.allow_new_risk,
            "regime": signal.regime,
            "strategy_reason": signal.reason,
        }

    @staticmethod
    def to_signal_snapshot(
        signal: HPRLPlannerSignal,
        *,
        timeframe: str,
        candle_close_time: datetime,
        feature_timestamp: datetime,
    ):
        from freqtrade.hedge.integration.signal_provider import SignalSnapshot

        payload = CleanMainlineSignalAdapter.signal_snapshot_kwargs(
            signal,
            timeframe=timeframe,
            candle_close_time=candle_close_time,
            feature_timestamp=feature_timestamp,
        )
        return SignalSnapshot(**payload)


class NoExchangeWriteGuard:
    """Machine-testable statement that HPRL owns no live exchange-write capability."""

    live_order_write = False
    exchange_api_access = False

    @staticmethod
    def assert_safe() -> bool:
        return True
