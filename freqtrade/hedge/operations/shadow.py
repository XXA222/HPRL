"""active-versus-shadow strategy comparison without order submission."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from .common import ZERO
@dataclass(frozen=True,slots=True)
class ShadowObservation:cycle_id:str;active_target:Decimal;shadow_target:Decimal;active_risk:Decimal;shadow_risk:Decimal;active_pnl_delta:Decimal;shadow_pnl_delta:Decimal
@dataclass(frozen=True,slots=True)
class ShadowSummary:cycles:int;target_mae:Decimal;risk_excess_cycles:int;active_pnl:Decimal;shadow_pnl:Decimal;shadow_outperformed:bool
class StrategyShadowComparator:
    def __init__(self):self._items:dict[str,ShadowObservation]={}
    def record(self,item:ShadowObservation)->bool:
        existing=self._items.get(item.cycle_id)
        if existing is not None:
            if existing!=item:raise ValueError("shadow cycle collision")
            return False
        self._items[item.cycle_id]=item;return True
    def summary(self)->ShadowSummary:
        rows=tuple(self._items.values());count=len(rows)
        if not rows:return ShadowSummary(0,ZERO,0,ZERO,ZERO,False)
        mae=sum((abs(x.active_target-x.shadow_target) for x in rows),ZERO)/count;excess=sum(1 for x in rows if x.shadow_risk>x.active_risk);active=sum((x.active_pnl_delta for x in rows),ZERO);shadow=sum((x.shadow_pnl_delta for x in rows),ZERO)
        return ShadowSummary(count,mae,excess,active,shadow,shadow>active)
