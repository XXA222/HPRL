from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from freqtrade.hedge.acceptance.models import RuntimeStage


@dataclass(frozen=True, slots=True)
class StreamAcceptanceState:
    stage: RuntimeStage
    connected: bool
    last_event_at: datetime | None
    reconnect_generation: int
    reconciliation_generation: int
    new_risk_enabled: bool
    reason: str


class StreamRecoveryGate:
    """Fail closed until every reconnect generation has a successful reconciliation."""

    def __init__(self, *, stale_after: timedelta | None) -> None:
        if stale_after is not None and stale_after <= timedelta(0):
            raise ValueError("stale_after must be positive or None")
        self.stale_after = stale_after
        self._connected = False
        self._last_event_at: datetime | None = None
        self._reconnect_generation = 0
        self._reconciliation_generation = 0
        self._stage = RuntimeStage.STREAM_STARTING
        self._reason = "NOT_CONNECTED"

    def connected(self, *, at: datetime | None = None) -> None:
        self._connected = True
        self._reconnect_generation += 1
        self._stage = RuntimeStage.RECOVERING
        self._reason = "RECONNECT_REQUIRES_RECONCILIATION"
        self._last_event_at = at or datetime.now(UTC)

    def disconnected(self) -> None:
        self._connected = False
        self._stage = RuntimeStage.STREAM_STALE
        self._reason = "DISCONNECTED"

    def event(self, *, at: datetime | None = None) -> None:
        self._last_event_at = at or datetime.now(UTC)
        if not self._connected:
            self._stage = RuntimeStage.STREAM_STALE
            self._reason = "EVENT_WHILE_DISCONNECTED"

    def reconciliation_passed(self) -> None:
        if not self._connected:
            raise RuntimeError("cannot mark reconciliation passed while disconnected")
        self._reconciliation_generation = self._reconnect_generation
        self._stage = RuntimeStage.READY
        self._reason = "READY"

    def reconciliation_failed(self, reason: str) -> None:
        self._stage = RuntimeStage.RECOVERING
        self._reason = f"RECONCILIATION_FAILED:{reason}"

    def assess(self, *, now: datetime | None = None) -> StreamAcceptanceState:
        current = now or datetime.now(UTC)
        stale = self.stale_after is not None and (
            self._last_event_at is None or current - self._last_event_at > self.stale_after
        )
        stage = self._stage
        reason = self._reason
        if self._connected and stale:
            stage = RuntimeStage.STREAM_STALE
            reason = "USER_STREAM_STALE"
        reconciled = self._reconciliation_generation == self._reconnect_generation
        new_risk_enabled = (
            self._connected and not stale and reconciled and stage is RuntimeStage.READY
        )
        return StreamAcceptanceState(
            stage=stage,
            connected=self._connected,
            last_event_at=self._last_event_at,
            reconnect_generation=self._reconnect_generation,
            reconciliation_generation=self._reconciliation_generation,
            new_risk_enabled=new_risk_enabled,
            reason=reason,
        )
