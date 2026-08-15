"""portfolio risk snapshots and limit evaluation."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from .common import ZERO,ensure_aware
@dataclass(frozen=True,slots=True)
class RiskLeg:symbol:str;long_notional:Decimal;short_notional:Decimal;margin_used:Decimal;unrealized_pnl:Decimal=ZERO
@dataclass(frozen=True,slots=True)
class PortfolioRiskSnapshot:
    timestamp:datetime;equity:Decimal;gross_notional:Decimal;net_notional:Decimal;margin_used:Decimal;margin_ratio:Decimal;gross_ratio:Decimal;unrealized_pnl:Decimal;reasons:tuple[str,...];ready:bool
class PortfolioRiskMonitor:
    def __init__(self,*,max_gross_ratio:Decimal=Decimal("0.80"),max_margin_ratio:Decimal=Decimal("0.55"),max_net_ratio:Decimal=Decimal("0.50")):
        if min(max_gross_ratio,max_margin_ratio,max_net_ratio)<=ZERO:raise ValueError("risk ratios must be positive")
        self.max_gross_ratio=max_gross_ratio;self.max_margin_ratio=max_margin_ratio;self.max_net_ratio=max_net_ratio
    def snapshot(self,*,timestamp:datetime,equity:Decimal,legs:tuple[RiskLeg,...])->PortfolioRiskSnapshot:
        ensure_aware(timestamp)
        if any(min(x.long_notional,x.short_notional,x.margin_used)<ZERO for x in legs):
            raise ValueError("risk notionals and margin must be nonnegative")
        gross=sum((x.long_notional+x.short_notional for x in legs),ZERO);net=sum((x.long_notional-x.short_notional for x in legs),ZERO);margin=sum((x.margin_used for x in legs),ZERO);upnl=sum((x.unrealized_pnl for x in legs),ZERO);reasons=[]
        if equity<=ZERO:
            reasons.append("EQUITY_NONPOSITIVE")
        denominator=max(equity,Decimal("0.00000001"))
        gross_ratio=gross/denominator;margin_ratio=margin/denominator;net_ratio=abs(net)/denominator
        if gross_ratio>self.max_gross_ratio:reasons.append("GROSS_LIMIT")
        if margin_ratio>self.max_margin_ratio:reasons.append("MARGIN_LIMIT")
        if net_ratio>self.max_net_ratio:reasons.append("NET_LIMIT")
        return PortfolioRiskSnapshot(timestamp,equity,gross,net,margin,margin_ratio,gross_ratio,upnl,tuple(reasons),not reasons)
