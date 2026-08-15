"""Shared exact-value and canonical serialization helpers for the operations runtime."""
from __future__ import annotations
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any

ZERO=Decimal("0");ONE=Decimal("1")

def decimal_value(value:object,default:Decimal=ZERO,*,minimum:Decimal|None=None,maximum:Decimal|None=None)->Decimal:
    try: result=value if isinstance(value,Decimal) else Decimal(str(value))
    except (InvalidOperation,TypeError,ValueError): result=default
    if not result.is_finite(): result=default
    if minimum is not None: result=max(minimum,result)
    if maximum is not None: result=min(maximum,result)
    return result

def utc_now()->datetime:return datetime.now(UTC)

def ensure_aware(value:datetime)->datetime:
    if value.tzinfo is None: raise ValueError("datetime must be timezone-aware")
    return value.astimezone(UTC)

def primitive(value:Any)->Any:
    if is_dataclass(value): return primitive(asdict(value))
    if isinstance(value,Decimal): return str(value)
    if isinstance(value,datetime): return ensure_aware(value).isoformat()
    if isinstance(value,Enum): return value.value
    if isinstance(value,Path): return str(value)
    if isinstance(value,dict): return {str(k):primitive(v) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
    if isinstance(value,(list,tuple,set,frozenset)): return [primitive(v) for v in value]
    if value is None or isinstance(value,(str,int,bool,float)): return value
    raise TypeError(f"unsupported canonical type: {type(value).__name__}")

def canonical_json(value:Any)->str:return json.dumps(primitive(value),sort_keys=True,separators=(",",":"),ensure_ascii=False)
def sha256_value(value:Any)->str:return hashlib.sha256(canonical_json(value).encode()).hexdigest()

def atomic_write_text(path:str|Path,text:str)->None:
    target=Path(path);target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix(target.suffix+".tmp")
    tmp.write_text(text,encoding="utf-8");tmp.replace(target)
