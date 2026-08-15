"""strategy and informative-timeframe warmup gate."""
from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True,slots=True)
class WarmupRequirement:
    base_candles:int;informative_candles:tuple[tuple[str,int],...]=()
    def __post_init__(self):
        if self.base_candles<1 or any(n<1 for _,n in self.informative_candles):raise ValueError("warmup candles must be positive")
@dataclass(frozen=True,slots=True)
class WarmupDecision:ready:bool;progress:float;missing:tuple[str,...]
class StrategyWarmupGate:
    def __init__(self,requirement:WarmupRequirement):self.requirement=requirement
    def evaluate(self,*,base_available:int,informative_available:dict[str,int]|None=None)->WarmupDecision:
        info=informative_available or {};missing=[];ratios=[min(max(base_available,0)/self.requirement.base_candles,1.0)]
        if base_available<self.requirement.base_candles:missing.append(f"BASE:{self.requirement.base_candles-base_available}")
        for timeframe,required in self.requirement.informative_candles:
            available=max(int(info.get(timeframe,0)),0);ratios.append(min(available/required,1.0))
            if available<required:missing.append(f"{timeframe}:{required-available}")
        return WarmupDecision(not missing,min(ratios),tuple(missing))
