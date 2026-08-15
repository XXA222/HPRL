"""drawdown circuit breaker with hysteresis and explicit reset."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from .common import ONE,ZERO
class BreakerState(StrEnum):NORMAL="NORMAL";WARNING="WARNING";PAUSED="PAUSED";KILLED="KILLED"
@dataclass(frozen=True,slots=True)
class BreakerDecision:state:BreakerState;drawdown:Decimal;new_risk_enabled:bool;reduce_only:bool;reason:str
class DrawdownCircuitBreaker:
    def __init__(self,*,warning:Decimal=Decimal("0.05"),pause:Decimal=Decimal("0.10"),kill:Decimal=Decimal("0.20"),recovery:Decimal=Decimal("0.03")):
        if not (ZERO<=recovery<warning<pause<kill<ONE):raise ValueError("invalid drawdown thresholds")
        self.warning=warning;self.pause=pause;self.kill=kill;self.recovery=recovery;self.peak=ZERO;self.state=BreakerState.NORMAL
    def evaluate(self,equity:Decimal)->BreakerDecision:
        if equity<ZERO:raise ValueError("equity must be nonnegative")
        self.peak=max(self.peak,equity);drawdown=ZERO if self.peak<=ZERO else (self.peak-equity)/self.peak
        if self.state is BreakerState.KILLED:return BreakerDecision(self.state,drawdown,False,True,"KILL_LATCHED")
        target=BreakerState.KILLED if drawdown>=self.kill else BreakerState.PAUSED if drawdown>=self.pause else BreakerState.WARNING if drawdown>=self.warning else BreakerState.NORMAL
        if target is not BreakerState.KILLED and self.state is BreakerState.PAUSED and drawdown>self.recovery:
            target=BreakerState.PAUSED
        self.state=target;return BreakerDecision(target,drawdown,target in {BreakerState.NORMAL,BreakerState.WARNING},target in {BreakerState.PAUSED,BreakerState.KILLED},f"DRAWDOWN_{target.value}")
    def manual_reset(self,*,equity:Decimal)->BreakerDecision:
        if equity<ZERO:raise ValueError("equity must be nonnegative")
        self.peak=equity;self.state=BreakerState.NORMAL;return self.evaluate(equity)
