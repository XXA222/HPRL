"""Durable handoff contract for approved risk reservations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import json
from threading import RLock
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from freqtrade.hedge.errors import HedgeConfigurationError


def _nonempty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HedgeConfigurationError(f"{field_name} must not be empty.")
    return value.strip()


def _canonical_json(value: str, *, field_name: str) -> str:
    normalized = _nonempty(value, field_name=field_name)
    try:
        payload = json.loads(normalized)
    except (TypeError, ValueError) as exc:
        raise HedgeConfigurationError(f"{field_name} must contain valid JSON.") from exc
    if not isinstance(payload, dict):
        raise HedgeConfigurationError(f"{field_name} must contain a JSON object.")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class ApprovalCommitRecord:
    """Evidence that durable pending risk took ownership of a local reservation.

    The record contains immutable request and risk-snapshot JSON so the direction-one
    transaction can persist the exact approval evidence instead of only a foreign key.
    """

    decision_id: str
    intent_id: str
    idempotency_key: str
    correlation_id: str
    risk_snapshot_id: str
    request_json: str
    risk_snapshot_json: str
    rules_version: str
    fencing_token: int
    approved_quantity: Decimal
    approved_notional: Decimal
    evaluated_at_ms: int
    committed_at_ms: int
    durable_reference: str
    target_snapshot_version: int | None = None
    intent_expires_at_ms: int | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "decision_id",
            "intent_id",
            "idempotency_key",
            "correlation_id",
            "risk_snapshot_id",
            "rules_version",
            "durable_reference",
        ):
            object.__setattr__(
                self,
                field_name,
                _nonempty(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "request_json",
            _canonical_json(self.request_json, field_name="request_json"),
        )
        object.__setattr__(
            self,
            "risk_snapshot_json",
            _canonical_json(self.risk_snapshot_json, field_name="risk_snapshot_json"),
        )
        if (
            isinstance(self.fencing_token, bool)
            or not isinstance(self.fencing_token, int)
            or self.fencing_token <= 0
        ):
            raise HedgeConfigurationError("fencing_token must be a positive integer.")
        if self.approved_quantity <= 0 or self.approved_notional <= 0:
            raise HedgeConfigurationError("Committed approval values must be positive.")
        for field_name in ("evaluated_at_ms", "committed_at_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise HedgeConfigurationError(
                    f"{field_name} must be nonnegative integer milliseconds."
                )
        for field_name in ("target_snapshot_version", "intent_expires_at_ms"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise HedgeConfigurationError(
                    f"{field_name} must be a nonnegative integer or None."
                )

    def as_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "intent_id": self.intent_id,
            "idempotency_key": self.idempotency_key,
            "correlation_id": self.correlation_id,
            "risk_snapshot_id": self.risk_snapshot_id,
            "request": json.loads(self.request_json),
            "risk_snapshot": json.loads(self.risk_snapshot_json),
            "rules_version": self.rules_version,
            "fencing_token": self.fencing_token,
            "approved_quantity": str(self.approved_quantity),
            "approved_notional": str(self.approved_notional),
            "evaluated_at_ms": self.evaluated_at_ms,
            "committed_at_ms": self.committed_at_ms,
            "durable_reference": self.durable_reference,
            "target_snapshot_version": self.target_snapshot_version,
            "intent_expires_at_ms": self.intent_expires_at_ms,
        }


class RiskApprovalCommitPort(Protocol):
    """Implemented by the ledger/execution integration transaction.

    Implementations must persist the approval, exact risk snapshot, request evidence,
    fencing token and durable pending-risk ownership in one database transaction.
    """

    def commit_approval(self, record: ApprovalCommitRecord) -> bool: ...

    def commit_approval_batch(self, records: tuple[ApprovalCommitRecord, ...]) -> bool: ...

    def read_commit(self, decision_id: str) -> ApprovalCommitRecord | None: ...


class InMemoryRiskApprovalCommitStore:
    """Thread-safe fake implementing idempotent commit semantics."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._records: dict[str, ApprovalCommitRecord] = {}

    def commit_approval(self, record: ApprovalCommitRecord) -> bool:
        with self._lock:
            current = self._records.get(record.decision_id)
            if current is None:
                self._records[record.decision_id] = record
                return True
            return current == record

    def commit_approval_batch(self, records: tuple[ApprovalCommitRecord, ...]) -> bool:
        if not records:
            raise ValueError("records must not be empty.")
        if len({record.decision_id for record in records}) != len(records):
            raise ValueError("Batch decision_id values must be unique.")
        with self._lock:
            for record in records:
                current = self._records.get(record.decision_id)
                if current is not None and current != record:
                    return False
            for record in records:
                self._records.setdefault(record.decision_id, record)
            return True

    def read_commit(self, decision_id: str) -> ApprovalCommitRecord | None:
        normalized = _nonempty(decision_id, field_name="decision_id")
        with self._lock:
            return self._records.get(normalized)


class SqlRiskApprovalCommitStore:
    """SQL-backed, idempotent approval handoff store.

    The direction-three reservation is considered durable only after this row is
    committed. A decision or idempotency collision with different evidence fails
    closed instead of silently reusing an unrelated approval.
    """

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory

    @staticmethod
    def _row_values(record: ApprovalCommitRecord) -> dict[str, object]:
        return {
            "decision_id": record.decision_id,
            "intent_id": record.intent_id,
            "idempotency_key": record.idempotency_key,
            "correlation_id": record.correlation_id,
            "risk_snapshot_id": record.risk_snapshot_id,
            "request_json": record.request_json,
            "risk_snapshot_json": record.risk_snapshot_json,
            "rules_version": record.rules_version,
            "fencing_token": record.fencing_token,
            "approved_quantity": str(record.approved_quantity),
            "approved_notional": str(record.approved_notional),
            "evaluated_at_ms": record.evaluated_at_ms,
            "committed_at_ms": record.committed_at_ms,
            "durable_reference": record.durable_reference,
            "target_snapshot_version": record.target_snapshot_version,
            "intent_expires_at_ms": record.intent_expires_at_ms,
        }

    @staticmethod
    def _from_row(row: object) -> ApprovalCommitRecord:
        return ApprovalCommitRecord(
            decision_id=str(getattr(row, "decision_id")),
            intent_id=str(getattr(row, "intent_id")),
            idempotency_key=str(getattr(row, "idempotency_key")),
            correlation_id=str(getattr(row, "correlation_id")),
            risk_snapshot_id=str(getattr(row, "risk_snapshot_id")),
            request_json=str(getattr(row, "request_json")),
            risk_snapshot_json=str(getattr(row, "risk_snapshot_json")),
            rules_version=str(getattr(row, "rules_version")),
            fencing_token=int(getattr(row, "fencing_token")),
            approved_quantity=Decimal(str(getattr(row, "approved_quantity"))),
            approved_notional=Decimal(str(getattr(row, "approved_notional"))),
            evaluated_at_ms=int(getattr(row, "evaluated_at_ms")),
            committed_at_ms=int(getattr(row, "committed_at_ms")),
            durable_reference=str(getattr(row, "durable_reference")),
            target_snapshot_version=getattr(row, "target_snapshot_version"),
            intent_expires_at_ms=getattr(row, "intent_expires_at_ms"),
        )

    def commit_approval(self, record: ApprovalCommitRecord) -> bool:
        return self.commit_approval_batch((record,))

    def commit_approval_batch(self, records: tuple[ApprovalCommitRecord, ...]) -> bool:
        if not records:
            raise ValueError("records must not be empty.")
        if len({record.decision_id for record in records}) != len(records):
            raise ValueError("Batch decision_id values must be unique.")
        if len({record.idempotency_key for record in records}) != len(records):
            raise ValueError("Batch idempotency_key values must be unique.")

        from freqtrade.persistence.hedge_models import RiskApprovalCommitRow

        try:
            with self._session_factory.begin() as session:
                for record in records:
                    existing = session.scalar(
                        select(RiskApprovalCommitRow).where(
                            RiskApprovalCommitRow.decision_id == record.decision_id
                        )
                    )
                    if existing is not None:
                        if self._from_row(existing) != record:
                            return False
                        continue
                    conflict = session.scalar(
                        select(RiskApprovalCommitRow).where(
                            RiskApprovalCommitRow.idempotency_key
                            == record.idempotency_key
                        )
                    )
                    if conflict is not None:
                        return self._from_row(conflict) == record
                    session.add(RiskApprovalCommitRow(**self._row_values(record)))
                session.flush()
            return True
        except IntegrityError:
            # A concurrent writer may have won the unique-key race. Re-read and
            # accept only byte-equivalent approval evidence.
            return all(self.read_commit(item.decision_id) == item for item in records)

    def read_commit(self, decision_id: str) -> ApprovalCommitRecord | None:
        normalized = _nonempty(decision_id, field_name="decision_id")
        from freqtrade.persistence.hedge_models import RiskApprovalCommitRow

        with self._session_factory() as session:
            row = session.scalar(
                select(RiskApprovalCommitRow).where(
                    RiskApprovalCommitRow.decision_id == normalized
                )
            )
            return None if row is None else self._from_row(row)
