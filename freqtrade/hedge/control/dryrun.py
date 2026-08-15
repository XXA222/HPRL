"""Restart-durable local Dry-run control state."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
from pathlib import Path
from threading import RLock


class DryRunControlMode(StrEnum):
    RUNNING = "RUNNING"
    NEW_RISK_PAUSED = "NEW_RISK_PAUSED"


@dataclass(frozen=True, slots=True)
class DryRunControlSnapshot:
    mode: DryRunControlMode
    revision: int
    updated_at: datetime
    actor: str
    reason: str

    @property
    def new_risk_enabled(self) -> bool:
        return self.mode is DryRunControlMode.RUNNING


class DryRunControlState:
    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._lock = RLock()
        self._snapshot = DryRunControlSnapshot(
            DryRunControlMode.RUNNING,
            0,
            datetime.now(UTC),
            "system",
            "INITIAL",
        )
        self._load()

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._snapshot = DryRunControlSnapshot(
                DryRunControlMode(str(raw["mode"])),
                int(raw["revision"]),
                datetime.fromisoformat(str(raw["updated_at"])),
                str(raw.get("actor", "system")),
                str(raw.get("reason", "")),
            )
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            self._snapshot = DryRunControlSnapshot(
                DryRunControlMode.NEW_RISK_PAUSED,
                1,
                datetime.now(UTC),
                "system",
                "CONTROL_STATE_RECOVERY_FAILED",
            )

    def _persist(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        payload = asdict(self._snapshot)
        payload["mode"] = self._snapshot.mode.value
        payload["updated_at"] = self._snapshot.updated_at.isoformat()
        tmp.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        tmp.replace(self._path)

    def snapshot(self) -> DryRunControlSnapshot:
        with self._lock:
            return self._snapshot

    def _set(
        self,
        mode: DryRunControlMode,
        *,
        actor: str,
        reason: str,
    ) -> DryRunControlSnapshot:
        with self._lock:
            self._snapshot = DryRunControlSnapshot(
                mode,
                self._snapshot.revision + 1,
                datetime.now(UTC),
                actor[:128],
                reason[:256],
            )
            self._persist()
            return self._snapshot

    def pause_new_risk(
        self,
        *,
        actor: str = "operator",
        reason: str = "MANUAL_PAUSE",
    ) -> DryRunControlSnapshot:
        return self._set(
            DryRunControlMode.NEW_RISK_PAUSED,
            actor=actor,
            reason=reason,
        )

    def resume_new_risk(
        self,
        *,
        actor: str = "operator",
        reason: str = "MANUAL_RESUME",
    ) -> DryRunControlSnapshot:
        return self._set(
            DryRunControlMode.RUNNING,
            actor=actor,
            reason=reason,
        )

    def reset_fail_closed(
        self,
        *,
        actor: str = "operator",
    ) -> DryRunControlSnapshot:
        return self.pause_new_risk(
            actor=actor,
            reason="MANUAL_FAIL_CLOSED_RESET",
        )
