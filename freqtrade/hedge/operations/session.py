"""deterministic Dry-run session and cycle identity."""
from __future__ import annotations
from dataclasses import dataclass,replace
from datetime import datetime
from enum import StrEnum
from .common import ensure_aware,sha256_value,utc_now

class SessionStatus(StrEnum):CREATED="CREATED";RUNNING="RUNNING";PAUSED="PAUSED";STOPPED="STOPPED";FAILED="FAILED"
@dataclass(frozen=True,slots=True)
class RunSession:
    session_id:str;account_id:str;symbols:tuple[str,...];config_sha256:str;started_at:datetime
    status:SessionStatus=SessionStatus.CREATED;cycle_sequence:int=0;last_cycle_at:datetime|None=None;reason:str=""
    def __post_init__(self):
        ensure_aware(self.started_at)
        if self.last_cycle_at is not None:ensure_aware(self.last_cycle_at)
        if not self.session_id or not self.account_id or not self.symbols:raise ValueError("session identity fields are required")
        if self.cycle_sequence<0:raise ValueError("cycle_sequence must be nonnegative")
    @classmethod
    def create(cls,*,account_id:str,symbols:tuple[str,...],config:object,started_at:datetime|None=None)->"RunSession":
        ts=ensure_aware(started_at or utc_now());normalized=tuple(dict.fromkeys(x.strip().upper() for x in symbols if x.strip()))
        if not normalized:raise ValueError("at least one symbol is required")
        config_hash=sha256_value(config);sid=sha256_value({"account":account_id,"symbols":normalized,"config":config_hash,"started":ts})[:24]
        return cls(sid,account_id,normalized,config_hash,ts)
    def transition(self,status:SessionStatus,*,reason:str="")->"RunSession":
        allowed={SessionStatus.CREATED:{SessionStatus.RUNNING,SessionStatus.STOPPED,SessionStatus.FAILED},SessionStatus.RUNNING:{SessionStatus.PAUSED,SessionStatus.STOPPED,SessionStatus.FAILED},SessionStatus.PAUSED:{SessionStatus.RUNNING,SessionStatus.STOPPED,SessionStatus.FAILED},SessionStatus.STOPPED:set(),SessionStatus.FAILED:set()}
        if status is self.status:return self
        if status not in allowed[self.status]:raise ValueError(f"invalid session transition {self.status}->{status}")
        return replace(self,status=status,reason=reason[:256])
    def next_cycle(self,at:datetime)->tuple["RunSession",str]:
        ts=ensure_aware(at)
        if self.status is not SessionStatus.RUNNING:raise RuntimeError("session must be RUNNING")
        if self.last_cycle_at is not None and ts<=self.last_cycle_at:raise ValueError("cycle time must be strictly monotonic")
        sequence=self.cycle_sequence+1;cycle_id=f"{self.session_id}:{sequence:012d}"
        return replace(self,cycle_sequence=sequence,last_cycle_at=ts),cycle_id
