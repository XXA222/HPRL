"""deterministic participation-limited partial-fill engine."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal,ROUND_DOWN
from .common import ONE,ZERO
@dataclass(frozen=True,slots=True)
class FillOrder:order_id:str;remaining_quantity:Decimal;limit_price:Decimal|None;side:str
@dataclass(frozen=True,slots=True)
class FillDecision:order_id:str;filled_quantity:Decimal;remaining_quantity:Decimal;terminal:bool;reason:str
class PartialFillEngine:
    def __init__(self,*,participation_rate:Decimal=Decimal("0.10"),max_fill_ratio:Decimal=Decimal("0.50"),qty_step:Decimal=Decimal("0.0001")):
        if participation_rate<=ZERO or participation_rate>ONE or max_fill_ratio<=ZERO or max_fill_ratio>ONE or qty_step<=ZERO:raise ValueError("invalid fill model")
        self.participation_rate=participation_rate;self.max_fill_ratio=max_fill_ratio;self.qty_step=qty_step
    def fill(self,order:FillOrder,*,bar_volume:Decimal,market_price:Decimal)->FillDecision:
        if order.remaining_quantity<=ZERO:return FillDecision(order.order_id,ZERO,ZERO,True,"ALREADY_FILLED")
        if bar_volume<ZERO or market_price<=ZERO:raise ValueError("invalid bar input")
        marketable=order.limit_price is None or (order.side.upper()=="BUY" and order.limit_price>=market_price) or (order.side.upper()=="SELL" and order.limit_price<=market_price)
        if not marketable:return FillDecision(order.order_id,ZERO,order.remaining_quantity,False,"NOT_MARKETABLE")
        cap=min(order.remaining_quantity*self.max_fill_ratio,bar_volume*self.participation_rate,order.remaining_quantity)
        steps=(cap/self.qty_step).to_integral_value(rounding=ROUND_DOWN);filled=steps*self.qty_step
        if filled<=ZERO and order.remaining_quantity<=self.qty_step:
            filled=order.remaining_quantity
        remaining=max(order.remaining_quantity-filled,ZERO)
        reason="FILLED" if remaining==ZERO else "PARTIAL" if filled>ZERO else "VOLUME_BELOW_STEP"
        return FillDecision(order.order_id,filled,remaining,remaining==ZERO,reason)
