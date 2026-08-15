"""checksummed runtime checkpoints with retention and restore."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from .common import atomic_write_text,canonical_json,ensure_aware,sha256_value
@dataclass(frozen=True,slots=True)
class RuntimeCheckpoint:checkpoint_id:str;created_at:datetime;payload:dict[str,object];sha256:str
class RuntimeCheckpointManager:
    def __init__(self,directory:str|Path,*,retention:int=20):
        if retention<1:raise ValueError("retention must be positive")
        self.directory=Path(directory);self.retention=retention
    def create(self,payload:dict[str,object],*,at:datetime)->RuntimeCheckpoint:
        ts=ensure_aware(at);body={"created_at":ts,"payload":payload};digest=sha256_value(body);checkpoint=RuntimeCheckpoint(digest[:20],ts,payload,digest);self.directory.mkdir(parents=True,exist_ok=True);atomic_write_text(self.directory/f"{checkpoint.checkpoint_id}.json",canonical_json({"created_at":ts,"payload":payload,"sha256":digest}));self._prune();return checkpoint
    def restore(self,checkpoint_id:str)->RuntimeCheckpoint:
        path=self.directory/f"{checkpoint_id}.json";raw=json.loads(path.read_text(encoding="utf-8"));body={"created_at":datetime.fromisoformat(raw["created_at"]),"payload":raw["payload"]}
        if raw.get("sha256")!=sha256_value(body):raise ValueError("checkpoint checksum mismatch")
        return RuntimeCheckpoint(checkpoint_id,body["created_at"],body["payload"],raw["sha256"])
    def list_ids(self)->tuple[str,...]:return tuple(x.stem for x in sorted(self.directory.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True))
    def _prune(self)->None:
        for path in sorted(self.directory.glob("*.json"),key=lambda p:p.stat().st_mtime,reverse=True)[self.retention:]:path.unlink()
