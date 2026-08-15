"""pre-execution intent admission with reduce-only escape."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from .common import ZERO,ensure_aware
@dataclass(frozen=True,slots=True)
class AdmissionIntent: intent_id:str;symbol:str;side:str;quantity:Decimal;price:Decimal;reduce_only:bool=False
@dataclass(frozen=True,slots=True)
class AdmissionDecision:approved:bool;reasons:tuple[str,...];notional:Decimal
class IntentAdmissionGate:
    def __init__(self,*,min_notional:Decimal,max_notional:Decimal,max_open_orders:int=20):
        if min_notional<ZERO or max_notional<=ZERO or min_notional>max_notional or max_open_orders<1:raise ValueError("invalid admission limits")
        self.min_notional=min_notional;self.max_notional=max_notional;self.max_open_orders=max_open_orders;self._seen:set[str]=set()
    def evaluate(self,intent:AdmissionIntent,*,open_orders:int,new_risk_enabled:bool,cooldown_until:datetime|None,now:datetime)->AdmissionDecision:
        ts=ensure_aware(now);reasons=[];notional=intent.quantity*intent.price
        if not intent.intent_id or intent.quantity<=ZERO or intent.price<=ZERO:reasons.append("INTENT_INVALID")
        if intent.intent_id in self._seen:reasons.append("INTENT_DUPLICATE")
        if not intent.reduce_only:
            if not new_risk_enabled:reasons.append("NEW_RISK_PAUSED")
            if cooldown_until is not None and ts<ensure_aware(cooldown_until):reasons.append("COOLDOWN_ACTIVE")
            if open_orders>=self.max_open_orders:reasons.append("OPEN_ORDER_LIMIT")
            if notional<self.min_notional:reasons.append("MIN_NOTIONAL")
            if notional>self.max_notional:reasons.append("MAX_NOTIONAL")
        if not reasons:self._seen.add(intent.intent_id)
        return AdmissionDecision(not reasons,tuple(reasons),notional)
