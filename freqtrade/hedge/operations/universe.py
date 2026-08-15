"""managed-pair universe normalization, quarantine and health."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime,timedelta
from .common import ensure_aware

def normalize_symbol(value:str)->str:
    text=value.strip().upper().replace("/","").replace(":USDT","")
    if not text or not text.endswith("USDT"):raise ValueError(f"unsupported USD-M symbol: {value}")
    return text
@dataclass(frozen=True,slots=True)
class PairHealth: symbol:str;active:bool;quarantined_until:datetime|None;failures:int;reason:str=""
class PairUniversePolicy:
    def __init__(self,symbols:tuple[str,...],*,failure_threshold:int=3,quarantine_seconds:int=900):
        normalized=tuple(dict.fromkeys(normalize_symbol(x) for x in symbols))
        if not normalized or failure_threshold<1 or quarantine_seconds<1:raise ValueError("invalid universe policy")
        self.symbols=normalized;self.failure_threshold=failure_threshold;self.quarantine_seconds=quarantine_seconds;self._failures={x:0 for x in normalized};self._until:dict[str,datetime|None]={x:None for x in normalized};self._reason={x:"" for x in normalized}
    def record_success(self,symbol:str)->None:
        key=normalize_symbol(symbol);self._require(key);self._failures[key]=0;self._reason[key]=""
    def record_failure(self,symbol:str,*,at:datetime,reason:str)->None:
        key=normalize_symbol(symbol);self._require(key);ts=ensure_aware(at);self._failures[key]+=1;self._reason[key]=reason[:128]
        if self._failures[key]>=self.failure_threshold:self._until[key]=ts+timedelta(seconds=self.quarantine_seconds)
    def health(self,*,at:datetime)->tuple[PairHealth,...]:
        ts=ensure_aware(at);rows=[]
        for symbol in self.symbols:
            until=self._until[symbol]
            if until is not None and ts>=until:self._until[symbol]=None;self._failures[symbol]=0;self._reason[symbol]="";until=None
            rows.append(PairHealth(symbol,until is None,until,self._failures[symbol],self._reason[symbol]))
        return tuple(rows)
    def active_symbols(self,*,at:datetime)->tuple[str,...]:return tuple(x.symbol for x in self.health(at=at) if x.active)
    def _require(self,symbol:str)->None:
        if symbol not in self._failures:raise KeyError(symbol)
