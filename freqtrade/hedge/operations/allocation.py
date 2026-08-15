"""deterministic pair/side capital allocation with reserve."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from .common import ONE,ZERO
@dataclass(frozen=True,slots=True)
class PairAllocation: symbol:str;total_notional:Decimal;long_notional:Decimal;short_notional:Decimal
class DryRunCapitalAllocator:
    def __init__(self,*,reserve_ratio:Decimal=Decimal("0.10"),max_pair_ratio:Decimal=Decimal("0.50"),max_side_ratio:Decimal=Decimal("0.35")):
        for value in (reserve_ratio,max_pair_ratio,max_side_ratio):
            if value<ZERO or value>ONE:raise ValueError("allocation ratio must be in [0,1]")
        if reserve_ratio>=ONE:raise ValueError("reserve must leave deployable capital")
        self.reserve_ratio=reserve_ratio;self.max_pair_ratio=max_pair_ratio;self.max_side_ratio=max_side_ratio
    def allocate(self,*,equity:Decimal,weights:dict[str,Decimal],long_bias:dict[str,Decimal]|None=None)->tuple[PairAllocation,...]:
        if equity<ZERO:raise ValueError("equity must be nonnegative")
        clean={k:v for k,v in weights.items() if v>ZERO};total=sum(clean.values(),ZERO)
        if not clean or total<=ZERO:return ()
        deployable=equity*(ONE-self.reserve_ratio);bias=long_bias or {};rows=[]
        for symbol in sorted(clean):
            pair=min(deployable*clean[symbol]/total,equity*self.max_pair_ratio)
            lb=min(max(bias.get(symbol,Decimal("0.5")),ZERO),ONE)
            long=min(pair*lb,equity*self.max_side_ratio);short=min(pair*(ONE-lb),equity*self.max_side_ratio)
            rows.append(PairAllocation(symbol,long+short,long,short))
        return tuple(rows)
