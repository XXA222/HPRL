"""bounded deterministic spread/impact/volatility slippage model."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
from .common import ZERO
@dataclass(frozen=True,slots=True)
class SlippageQuote: reference_price:Decimal;execution_price:Decimal;slippage_bps:Decimal;components:dict[str,Decimal]
class DeterministicSlippageModel:
    def __init__(self,*,base_bps:Decimal=Decimal("1"),impact_bps:Decimal=Decimal("20"),volatility_bps:Decimal=Decimal("5"),cap_bps:Decimal=Decimal("100")):
        if min(base_bps,impact_bps,volatility_bps,cap_bps)<ZERO:raise ValueError("slippage parameters must be nonnegative")
        self.base_bps=base_bps;self.impact_bps=impact_bps;self.volatility_bps=volatility_bps;self.cap_bps=cap_bps
    def quote(self,*,side:str,reference_price:Decimal,order_notional:Decimal,available_notional:Decimal,volatility:Decimal)->SlippageQuote:
        if reference_price<=ZERO or order_notional<ZERO or available_notional<=ZERO:raise ValueError("invalid slippage input")
        participation=min(order_notional/available_notional,Decimal("1"));vol=max(volatility,ZERO)
        components={"base":self.base_bps,"impact":self.impact_bps*participation*participation,"volatility":self.volatility_bps*vol}
        bps=min(sum(components.values(),ZERO),self.cap_bps);direction=Decimal("1") if side.upper()=="BUY" else Decimal("-1") if side.upper()=="SELL" else None
        if direction is None:raise ValueError("side must be BUY or SELL")
        price=reference_price*(Decimal("1")+direction*bps/Decimal("10000"))
        return SlippageQuote(reference_price,price,bps,components)
