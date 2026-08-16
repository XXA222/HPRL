"""Production incident ledger used to block unsafe resume/promotion."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from .contracts import Severity


class IncidentStatus(StrEnum):
    OPEN = "OPEN"
    MITIGATED = "MITIGATED"
    CLOSED = "CLOSED"


@dataclass(frozen=True, slots=True)
class Incident:
    incident_id: str
    severity: Severity
    code: str
    opened_at: datetime
    status: IncidentStatus = IncidentStatus.OPEN
    closed_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.incident_id.strip() or not self.code.strip():
            raise ValueError("incident_id and code are required")
        if self.opened_at.tzinfo is None:
            raise ValueError("opened_at must be timezone-aware")
        if self.closed_at is not None and self.closed_at.tzinfo is None:
            raise ValueError("closed_at must be timezone-aware")
        if self.status is IncidentStatus.CLOSED and self.closed_at is None:
            raise ValueError("closed incident requires closed_at")


class IncidentLedger:
    def __init__(self) -> None:
        self._items: dict[str, Incident] = {}

    def open(self, incident: Incident) -> None:
        if incident.incident_id in self._items:
            raise ValueError("duplicate incident id")
        self._items[incident.incident_id] = incident

    def close(self, incident_id: str, *, closed_at: datetime) -> Incident:
        current = self._items[incident_id]
        if closed_at.tzinfo is None or closed_at < current.opened_at:
            raise ValueError("invalid closed_at")
        updated = Incident(current.incident_id, current.severity, current.code, current.opened_at, IncidentStatus.CLOSED, closed_at.astimezone(UTC))
        self._items[incident_id] = updated
        return updated

    def close_checked(
        self,
        incident_id: str,
        *,
        closed_at: datetime,
        reconciliation_converged: bool,
        readiness_passed: bool,
        operator_acknowledged: bool,
    ) -> Incident:
        current = self._items[incident_id]
        if current.severity in {Severity.HALT_NEW_RISK, Severity.HALT_ACCOUNT}:
            if not reconciliation_converged:
                raise PermissionError("incident close requires reconciliation convergence")
            if not readiness_passed:
                raise PermissionError("incident close requires current readiness")
            if not operator_acknowledged:
                raise PermissionError("incident close requires operator acknowledgement")
        return self.close(incident_id, closed_at=closed_at)

    @property
    def open_incidents(self) -> tuple[Incident, ...]:
        return tuple(sorted((x for x in self._items.values() if x.status is not IncidentStatus.CLOSED), key=lambda x: x.incident_id))

    @property
    def blocks_new_risk(self) -> bool:
        return any(x.severity in {Severity.HALT_NEW_RISK, Severity.HALT_ACCOUNT} for x in self.open_incidents)

    @property
    def blocks_account(self) -> bool:
        return any(x.severity is Severity.HALT_ACCOUNT for x in self.open_incidents)
