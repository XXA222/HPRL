"""Checksummed atomic operations runtime state persistence."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from threading import RLock
from .common import atomic_write_text,canonical_json,sha256_value
from .session import RunSession,SessionStatus

class StateCorruptionError(RuntimeError):pass
class AtomicRunStateStore:
    schema_version=1
    def __init__(self,path:str|Path):self.path=Path(path);self._lock=RLock()
    def save(self,session:RunSession,extras:dict[str,object]|None=None)->None:
        body={"schema_version":self.schema_version,"session":asdict(session),"extras":extras or {}}
        envelope={"body":body,"sha256":sha256_value(body)}
        with self._lock:atomic_write_text(self.path,canonical_json(envelope))
    def load(self)->tuple[RunSession,dict[str,object]]|None:
        with self._lock:
            if not self.path.exists():return None
            try: envelope=json.loads(self.path.read_text(encoding="utf-8"));body=envelope["body"]
            except Exception as exc:raise StateCorruptionError("runtime state is unreadable") from exc
            if envelope.get("sha256")!=sha256_value(body):raise StateCorruptionError("runtime state checksum mismatch")
            if int(body.get("schema_version",0))!=self.schema_version:raise StateCorruptionError("unsupported runtime state schema")
            raw=body["session"]
            try:
                session=RunSession(session_id=str(raw["session_id"]),account_id=str(raw["account_id"]),symbols=tuple(raw["symbols"]),config_sha256=str(raw["config_sha256"]),started_at=datetime.fromisoformat(raw["started_at"]),status=SessionStatus(raw["status"]),cycle_sequence=int(raw["cycle_sequence"]),last_cycle_at=None if raw.get("last_cycle_at") is None else datetime.fromisoformat(raw["last_cycle_at"]),reason=str(raw.get("reason","")))
            except Exception as exc:raise StateCorruptionError("runtime state session is invalid") from exc
            extras=body.get("extras",{});return session,extras if isinstance(extras,dict) else {}
    def load_fail_closed(self)->tuple[RunSession|None,dict[str,object],tuple[str,...]]:
        try:
            loaded=self.load();return (None,{},()) if loaded is None else (loaded[0],loaded[1],())
        except StateCorruptionError as exc:return None,{},(f"STATE_RECOVERY_FAILED:{exc}",)
