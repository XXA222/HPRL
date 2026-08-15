"""adaptive per-symbol cooldown after loss, rejection or volatility shock."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timedelta
from decimal import Decimal
from .common import ZERO,ensure_aware
@dataclass(frozen=True,slots=True)
class CooldownStatus:symbol:str;until:datetime|None;active:bool;reason:str;loss_streak:int
class AdaptiveCooldownManager:
    def __init__(self,*,base_seconds:int=60,max_seconds:int=3600,volatility_trigger:Decimal=Decimal("0.03")):
        if base_seconds<1 or max_seconds<base_seconds or volatility_trigger<=ZERO:raise ValueError("invalid cooldown settings")
        self.base_seconds=base_seconds;self.max_seconds=max_seconds;self.volatility_trigger=volatility_trigger;self._streak:dict[str,int]={};self._until:dict[str,datetime]={};self._reason:dict[str,str]={}
    def record_trade(self,symbol:str,*,pnl:Decimal,at:datetime)->CooldownStatus:
        ts=ensure_aware(at);streak=0 if pnl>=ZERO else self._streak.get(symbol,0)+1;self._streak[symbol]=streak
        if streak:
            seconds=min(self.base_seconds*(2**(streak-1)),self.max_seconds);self._until[symbol]=ts+timedelta(seconds=seconds);self._reason[symbol]="LOSS_STREAK"
        else:
            self._until.pop(symbol,None);self._reason.pop(symbol,None)
        return self.status(symbol,at=ts)
    def record_rejection(self,symbol:str,*,at:datetime)->CooldownStatus:
        ts=ensure_aware(at);self._until[symbol]=ts+timedelta(seconds=min(self.base_seconds*2,self.max_seconds));self._reason[symbol]="ORDER_REJECTION";return self.status(symbol,at=ts)
    def record_volatility(self,symbol:str,*,volatility:Decimal,at:datetime)->CooldownStatus:
        ts=ensure_aware(at)
        if volatility>=self.volatility_trigger:self._until[symbol]=ts+timedelta(seconds=min(self.base_seconds*4,self.max_seconds));self._reason[symbol]="VOLATILITY_SHOCK"
        return self.status(symbol,at=ts)
    def status(self,symbol:str,*,at:datetime)->CooldownStatus:
        ts=ensure_aware(at);until=self._until.get(symbol);active=until is not None and ts<until
        if not active and until is not None:self._until.pop(symbol,None);self._reason.pop(symbol,None)
        return CooldownStatus(symbol,until if active else None,active,self._reason.get(symbol,""),self._streak.get(symbol,0))
