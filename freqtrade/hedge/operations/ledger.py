"""idempotent exact-value fee/funding/realized-PnL simulation ledger."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from .common import ZERO,ensure_aware
class LedgerEventType(StrEnum):FEE="FEE";FUNDING="FUNDING";REALIZED_PNL="REALIZED_PNL";ADJUSTMENT="ADJUSTMENT"
@dataclass(frozen=True,slots=True)
class LedgerEvent:event_id:str;event_type:LedgerEventType;amount:Decimal;currency:str;timestamp:datetime;symbol:str|None=None;metadata:tuple[tuple[str,str],...]=()
@dataclass(frozen=True,slots=True)
class LedgerBalance:currency:str;fees:Decimal;funding:Decimal;realized_pnl:Decimal;adjustments:Decimal;net:Decimal
class FeeFundingLedger:
    def __init__(self):self._events:dict[str,LedgerEvent]={}
    def append(self,event:LedgerEvent)->bool:
        ensure_aware(event.timestamp)
        if not event.event_id or not event.currency:raise ValueError("event identity is required")
        existing=self._events.get(event.event_id)
        if existing is not None:
            if existing!=event:raise ValueError("ledger event id collision")
            return False
        self._events[event.event_id]=event;return True
    def balance(self,currency:str="USDT")->LedgerBalance:
        totals={x:ZERO for x in LedgerEventType}
        for event in self._events.values():
            if event.currency==currency:totals[event.event_type]+=event.amount
        net=totals[LedgerEventType.REALIZED_PNL]+totals[LedgerEventType.FUNDING]+totals[LedgerEventType.ADJUSTMENT]-totals[LedgerEventType.FEE]
        return LedgerBalance(currency,totals[LedgerEventType.FEE],totals[LedgerEventType.FUNDING],totals[LedgerEventType.REALIZED_PNL],totals[LedgerEventType.ADJUSTMENT],net)
    def events(self)->tuple[LedgerEvent,...]:return tuple(sorted(self._events.values(),key=lambda x:(x.timestamp,x.event_id)))
