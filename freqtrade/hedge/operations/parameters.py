"""versioned strategy parameters with bounds, activation and rollback."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from .common import ensure_aware,sha256_value
@dataclass(frozen=True,slots=True)
class ParameterBounds:minimum:Decimal;maximum:Decimal
@dataclass(frozen=True,slots=True)
class ParameterVersion:version_id:str;created_at:datetime;actor:str;values:tuple[tuple[str,Decimal],...];sha256:str
class StrategyParameterRegistry:
    def __init__(self,bounds:dict[str,ParameterBounds]):self.bounds=dict(bounds);self._versions:list[ParameterVersion]=[];self._active:int|None=None
    def create(self,values:dict[str,Decimal],*,actor:str,at:datetime)->ParameterVersion:
        ts=ensure_aware(at);unknown=set(values)-set(self.bounds)
        if unknown:raise ValueError("unknown parameter(s): "+",".join(sorted(unknown)))
        for name,bound in self.bounds.items():
            if name not in values:raise ValueError(f"missing parameter: {name}")
            if not bound.minimum<=values[name]<=bound.maximum:raise ValueError(f"parameter out of bounds: {name}")
        ordered=tuple(sorted(values.items()));digest=sha256_value({"at":ts,"actor":actor,"values":ordered});version=ParameterVersion(digest[:16],ts,actor[:128],ordered,digest);self._versions.append(version);return version
    def activate(self,version_id:str)->ParameterVersion:
        for index,version in enumerate(self._versions):
            if version.version_id==version_id:self._active=index;return version
        raise KeyError(version_id)
    def active(self)->ParameterVersion|None:return None if self._active is None else self._versions[self._active]
    def rollback(self)->ParameterVersion:
        if self._active is None or self._active<1:raise RuntimeError("no previous parameter version")
        self._active-=1;return self._versions[self._active]
    def versions(self)->tuple[ParameterVersion,...]:return tuple(self._versions)
