"""market-data freshness, sequence, gap and divergence gate."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from .common import ZERO,ensure_aware
@dataclass(frozen=True,slots=True)
class MarketHealthInput:
    symbol:str;candle_time:datetime;observed_at:datetime;timeframe_seconds:int;mark_price:Decimal;index_price:Decimal|None=None
@dataclass(frozen=True,slots=True)
class MarketHealthDecision:
    ready:bool;reasons:tuple[str,...];age_seconds:Decimal;gap_seconds:Decimal|None;divergence_bps:Decimal
class MarketDataHealthGate:
    def __init__(self,*,max_age_seconds:int=90,max_gap_candles:int=2,max_divergence_bps:Decimal=Decimal("50")):
        if max_age_seconds<=0 or max_gap_candles<0 or max_divergence_bps<ZERO:raise ValueError("invalid market gate limits")
        self.max_age_seconds=max_age_seconds;self.max_gap_candles=max_gap_candles;self.max_divergence_bps=max_divergence_bps;self._last:dict[str,datetime]={}
    def evaluate(self,item:MarketHealthInput)->MarketHealthDecision:
        candle=ensure_aware(item.candle_time);observed=ensure_aware(item.observed_at);reasons=[]
        if item.timeframe_seconds<=0 or item.mark_price<=ZERO:reasons.append("MARKET_INPUT_INVALID")
        age=Decimal(str((observed-candle).total_seconds()));
        if age<ZERO:reasons.append("MARKET_CANDLE_FROM_FUTURE")
        if age>self.max_age_seconds:reasons.append("MARKET_DATA_STALE")
        previous=self._last.get(item.symbol);gap=None
        if previous is not None:
            gap=Decimal(str((candle-previous).total_seconds()))
            if candle<=previous:reasons.append("MARKET_CANDLE_NON_MONOTONIC")
            elif gap>item.timeframe_seconds*(self.max_gap_candles+1):reasons.append("MARKET_CANDLE_GAP")
        divergence=ZERO
        if item.index_price is not None:
            if item.index_price<=ZERO:reasons.append("INDEX_PRICE_INVALID")
            else:
                divergence=abs(item.mark_price-item.index_price)/item.index_price*Decimal("10000")
                if divergence>self.max_divergence_bps:reasons.append("MARK_INDEX_DIVERGENCE")
        if not reasons or all(x not in {"MARKET_CANDLE_NON_MONOTONIC","MARKET_CANDLE_FROM_FUTURE","MARKET_INPUT_INVALID"} for x in reasons):
            if previous is None or candle>previous:self._last[item.symbol]=candle
        return MarketHealthDecision(not reasons,tuple(reasons),age,gap,divergence)
