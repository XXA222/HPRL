"""Execution kill switch with risk-reducing exception, audit and HALT metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock
from typing import Any, Mapping, Protocol


class KillSwitchMode(StrEnum):
    RUNNING = "RUNNING"
    HALTED = "HALTED"


class ExecutionHaltedError(RuntimeError):
    pass


class _HaltMetricsPort(Protocol):
    def halt(self, active: bool, reason: str = "") -> None: ...


class _AuditPort(Protocol):
    def emit(self, event: str, payload: Mapping[str, Any]) -> None: ...


@dataclass(frozen=True, slots=True)
class KillSwitchSnapshot:
    mode: KillSwitchMode
    reason: str | None
    actor: str | None
    changed_at: datetime

    def __post_init__(self) -> None:
        mode = self.mode if isinstance(self.mode, KillSwitchMode) else KillSwitchMode(self.mode)
        if self.changed_at.tzinfo is None or self.changed_at.utcoffset() is None:
            raise ValueError("changed_at must be timezone-aware")
        object.__setattr__(self, "mode", mode)


def _required_text(value: object, *, field_name: str, limit: int) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{field_name} is required")
    if len(result) > limit or any(ord(ch) < 32 or ord(ch) == 127 for ch in result):
        raise ValueError(f"{field_name} is invalid")
    return result


class KillSwitch:
    def __init__(
        self,
        *,
        allow_risk_reduction_while_halted: bool = True,
        metrics: _HaltMetricsPort | None = None,
        audit: _AuditPort | None = None,
    ) -> None:
        if not isinstance(allow_risk_reduction_while_halted, bool):
            raise TypeError("allow_risk_reduction_while_halted must be a boolean")
        self._allow_reduce = allow_risk_reduction_while_halted
        self._metrics = metrics
        self._audit = audit
        self._snapshot = KillSwitchSnapshot(
            KillSwitchMode.RUNNING,
            None,
            None,
            datetime.now(UTC),
        )
        self._lock = RLock()

    def activate(self, *, reason: str, actor: str) -> KillSwitchSnapshot:
        reason_text = _required_text(reason, field_name="reason", limit=1024)
        actor_text = _required_text(actor, field_name="actor", limit=128)
        with self._lock:
            snapshot = KillSwitchSnapshot(
                KillSwitchMode.HALTED,
                reason_text,
                actor_text,
                datetime.now(UTC),
            )
            self._snapshot = snapshot
        self._emit(snapshot)
        return snapshot

    def deactivate(self, *, actor: str, confirmed: bool) -> KillSwitchSnapshot:
        if not isinstance(confirmed, bool):
            raise TypeError("confirmed must be a boolean")
        if not confirmed:
            raise PermissionError(
                "kill switch release requires secondary confirmation"
            )
        actor_text = _required_text(actor, field_name="actor", limit=128)
        with self._lock:
            snapshot = KillSwitchSnapshot(
                KillSwitchMode.RUNNING,
                None,
                actor_text,
                datetime.now(UTC),
            )
            self._snapshot = snapshot
        self._emit(snapshot)
        return snapshot

    def assert_allowed(self, *, reduces_risk: bool) -> None:
        if not isinstance(reduces_risk, bool):
            raise TypeError("reduces_risk must be a boolean")
        snapshot = self.snapshot()
        if snapshot.mode is KillSwitchMode.RUNNING:
            return
        if reduces_risk and self._allow_reduce:
            return
        raise ExecutionHaltedError(snapshot.reason or "execution is halted")

    def snapshot(self) -> KillSwitchSnapshot:
        with self._lock:
            return self._snapshot

    def _emit(self, snapshot: KillSwitchSnapshot) -> None:
        active = snapshot.mode is KillSwitchMode.HALTED
        if self._metrics is not None:
            try:
                self._metrics.halt(active, snapshot.reason or "")
            except Exception:
                pass
        if self._audit is not None:
            try:
                self._audit.emit(
                    "KILL_SWITCH_CHANGED",
                    {
                        "mode": snapshot.mode.value,
                        "reason": snapshot.reason,
                        "actor": snapshot.actor,
                        "changed_at": snapshot.changed_at.isoformat(),
                    },
                )
            except Exception:
                pass
