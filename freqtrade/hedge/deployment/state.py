"""Atomic lifecycle state for the external Hedge supervisor."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping


class RuntimePhase(StrEnum):
    STOPPED = "STOPPED"
    PREFLIGHT = "PREFLIGHT"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    BACKOFF = "BACKOFF"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class RuntimeState:
    phase: RuntimePhase
    supervisor_pid: int
    instance_token: str
    child_pid: int | None
    restart_count: int
    started_at_utc: str
    updated_at_utc: str
    heartbeat_at_utc: str
    last_exit_code: int | None = None
    last_error: str | None = None
    stop_requested: bool = False

    def evolve(self, **changes: Any) -> "RuntimeState":
        now = datetime.now(UTC).isoformat()
        return replace(self, updated_at_utc=now, heartbeat_at_utc=now, **changes)


class RuntimeStateStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, state: RuntimeState) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(state)
        payload["phase"] = state.phase.value
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        fd, temporary = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )
        temp_path = Path(temporary)
        try:
            with os.fdopen(fd, "w", encoding="ascii", newline="\n") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.path)
        finally:
            temp_path.unlink(missing_ok=True)

    def read(self) -> RuntimeState | None:
        if not self.path.is_file():
            return None
        payload = json.loads(self.path.read_text(encoding="ascii"))
        if not isinstance(payload, Mapping):
            raise ValueError("runtime state file must contain a JSON object")
        return RuntimeState(
            phase=RuntimePhase(str(payload["phase"])),
            supervisor_pid=int(payload["supervisor_pid"]),
            instance_token=str(payload["instance_token"]),
            child_pid=None if payload.get("child_pid") is None else int(payload["child_pid"]),
            restart_count=int(payload["restart_count"]),
            started_at_utc=str(payload["started_at_utc"]),
            updated_at_utc=str(payload["updated_at_utc"]),
            heartbeat_at_utc=str(payload["heartbeat_at_utc"]),
            last_exit_code=(
                None if payload.get("last_exit_code") is None else int(payload["last_exit_code"])
            ),
            last_error=(None if payload.get("last_error") is None else str(payload["last_error"])),
            stop_requested=bool(payload.get("stop_requested", False)),
        )
