"""alert deduplication, escalation, acknowledgement and expiry."""
from __future__ import annotations
from dataclasses import dataclass,replace
from datetime import datetime,timedelta
from enum import IntEnum
from .common import ensure_aware
class AlertSeverity(IntEnum):INFO=10;WARNING=20;ERROR=30;CRITICAL=40
@dataclass(frozen=True,slots=True)
class AlertRecord:key:str;severity:AlertSeverity;message:str;first_seen:datetime;last_seen:datetime;count:int=1;acknowledged:bool=False
class AlertManager:
    def __init__(self,*,dedup_seconds:int=300,escalate_after:int=3):
        if dedup_seconds<1 or escalate_after<2:raise ValueError("invalid alert settings")
        self.dedup_seconds=dedup_seconds;self.escalate_after=escalate_after;self._alerts:dict[str,AlertRecord]={}
    def emit(self,*,key:str,severity:AlertSeverity,message:str,at:datetime)->AlertRecord:
        ts=ensure_aware(at);current=self._alerts.get(key)
        if current is None or ts-current.last_seen>timedelta(seconds=self.dedup_seconds):record=AlertRecord(key,severity,message[:512],ts,ts)
        else:
            count=current.count+1;level=severity
            if count>=self.escalate_after:level=AlertSeverity(min(max(int(severity),int(current.severity))+10,int(AlertSeverity.CRITICAL)))
            record=replace(current,severity=level,message=message[:512],last_seen=ts,count=count,acknowledged=False)
        self._alerts[key]=record;return record
    def acknowledge(self,key:str)->AlertRecord:
        current=self._alerts[key];record=replace(current,acknowledged=True);self._alerts[key]=record;return record
    def active(self,*,minimum:AlertSeverity=AlertSeverity.INFO)->tuple[AlertRecord,...]:return tuple(sorted((x for x in self._alerts.values() if x.severity>=minimum and not x.acknowledged),key=lambda x:(-int(x.severity),x.first_seen)))
