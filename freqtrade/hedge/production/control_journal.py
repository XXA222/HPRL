"""Hash-chained audit journal for operator control-plane commands."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
import json

from .control import ControlAction, ControlMode


@dataclass(frozen=True, slots=True)
class ControlJournalRecord:
    sequence: int
    action: ControlAction
    actor: str
    reason: str
    observed_at: datetime
    before: ControlMode
    after: ControlMode
    readiness_digest: str
    reconciliation_digest: str
    previous_sha256: str = "0" * 64
    record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.sequence <= 0:
            raise ValueError("control journal sequence must be positive")
        if not self.actor.strip() or not self.reason.strip():
            raise ValueError("control journal actor/reason are required")
        if self.observed_at.tzinfo is None:
            raise ValueError("control journal timestamp must be timezone-aware")
        for name in ("readiness_digest", "reconciliation_digest", "previous_sha256"):
            digest = str(getattr(self, name)).lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{name} must be sha256")
            object.__setattr__(self, name, digest)
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))
        payload = {
            "sequence": self.sequence,
            "action": self.action.value,
            "actor": self.actor,
            "reason": self.reason,
            "observed_at": self.observed_at.isoformat(),
            "before": self.before.value,
            "after": self.after.value,
            "readiness_digest": self.readiness_digest,
            "reconciliation_digest": self.reconciliation_digest,
            "previous_sha256": self.previous_sha256,
        }
        object.__setattr__(
            self,
            "record_sha256",
            sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        )


class ControlJournal:
    def __init__(self) -> None:
        self._records: list[ControlJournalRecord] = []

    def append(
        self,
        *,
        action: ControlAction,
        actor: str,
        reason: str,
        observed_at: datetime,
        before: ControlMode,
        after: ControlMode,
        readiness_digest: str,
        reconciliation_digest: str,
    ) -> ControlJournalRecord:
        previous = self._records[-1].record_sha256 if self._records else "0" * 64
        record = ControlJournalRecord(
            len(self._records) + 1,
            action,
            actor,
            reason,
            observed_at,
            before,
            after,
            readiness_digest,
            reconciliation_digest,
            previous,
        )
        self._records.append(record)
        return record

    def verify(self) -> bool:
        previous = "0" * 64
        for index, record in enumerate(self._records, start=1):
            if record.sequence != index or record.previous_sha256 != previous:
                return False
            rebuilt = ControlJournalRecord(
                record.sequence,
                record.action,
                record.actor,
                record.reason,
                record.observed_at,
                record.before,
                record.after,
                record.readiness_digest,
                record.reconciliation_digest,
                record.previous_sha256,
            )
            if rebuilt.record_sha256 != record.record_sha256:
                return False
            previous = record.record_sha256
        return True

    @property
    def records(self) -> tuple[ControlJournalRecord, ...]:
        return tuple(self._records)
