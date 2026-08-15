"""Durable idempotency store for control-plane operations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from threading import RLock
from typing import Protocol
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from freqtrade.hedge.control.models import (
    ControlOperationResult,
    ControlOperationState,
    ControlRequest,
)


class ControlOperationConflict(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ControlOperationClaim:
    operation_id: UUID
    created_at: datetime
    existing_result: ControlOperationResult | None = None
    in_progress: bool = False
    recovered_stale: bool = False


class ControlOperationStore(Protocol):
    def lookup(self, *, request: ControlRequest) -> ControlOperationClaim | None: ...

    def latest_results(
        self,
        *,
        account_id: str,
        actions: tuple[str, ...],
    ) -> tuple[ControlOperationResult, ...]: ...

    def claim(
        self,
        *,
        request: ControlRequest,
        actor: str,
        actor_role: str,
    ) -> ControlOperationClaim: ...

    def complete(self, result: ControlOperationResult) -> None: ...

    def fail(self, result: ControlOperationResult) -> None: ...


@dataclass(slots=True)
class _MemoryRecord:
    operation_id: UUID
    request_hash: str
    actor: str
    actor_role: str
    created_at: datetime
    state: ControlOperationState
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    result: ControlOperationResult | None = None


class InMemoryControlOperationStore:
    def __init__(self, *, lease_seconds: int = 60, owner: str | None = None) -> None:
        if not isinstance(lease_seconds, int) or not 5 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be in [5, 3600]")
        self._records: dict[tuple[str, str], _MemoryRecord] = {}
        self._lock = RLock()
        self._lease_seconds = lease_seconds
        self._owner = str(owner or f"control-{uuid4().hex}")

    def lookup(self, *, request: ControlRequest) -> ControlOperationClaim | None:
        key = (request.account_id, request.idempotency_key)
        with self._lock:
            existing = self._records.get(key)
            if existing is None:
                return None
            if existing.request_hash != request.request_hash:
                raise ControlOperationConflict("idempotency key payload mismatch")
            return ControlOperationClaim(
                existing.operation_id,
                existing.created_at,
                existing_result=existing.result,
                in_progress=(
                    existing.result is None
                    and existing.lease_expires_at is not None
                    and existing.lease_expires_at > datetime.now(UTC)
                ),
                recovered_stale=False,
            )

    def latest_results(
        self,
        *,
        account_id: str,
        actions: tuple[str, ...],
    ) -> tuple[ControlOperationResult, ...]:
        action_set = set(actions)
        with self._lock:
            rows = [
                record.result
                for (record_account, _), record in self._records.items()
                if record_account == account_id
                and record.result is not None
                and record.result.action.value in action_set
            ]
        return tuple(sorted(rows, key=lambda item: item.completed_at))

    def claim(
        self,
        *,
        request: ControlRequest,
        actor: str,
        actor_role: str,
    ) -> ControlOperationClaim:
        key = (request.account_id, request.idempotency_key)
        with self._lock:
            existing = self._records.get(key)
            if existing is not None:
                if existing.request_hash != request.request_hash:
                    raise ControlOperationConflict("idempotency key payload mismatch")
                if existing.result is not None:
                    return ControlOperationClaim(
                        existing.operation_id,
                        existing.created_at,
                        existing_result=existing.result,
                    )
                now = datetime.now(UTC)
                if existing.lease_expires_at is not None and existing.lease_expires_at > now:
                    return ControlOperationClaim(
                        existing.operation_id,
                        existing.created_at,
                        in_progress=True,
                    )
                if existing.actor != actor:
                    raise ControlOperationConflict("stale operation actor mismatch")
                existing.lease_owner = self._owner
                existing.lease_expires_at = now + timedelta(seconds=self._lease_seconds)
                return ControlOperationClaim(
                    existing.operation_id,
                    existing.created_at,
                    recovered_stale=True,
                )
            operation_id = uuid4()
            created_at = datetime.now(UTC)
            self._records[key] = _MemoryRecord(
                operation_id=operation_id,
                request_hash=request.request_hash,
                actor=actor,
                actor_role=actor_role,
                created_at=created_at,
                state=ControlOperationState.CLAIMED,
                lease_owner=self._owner,
                lease_expires_at=created_at + timedelta(seconds=self._lease_seconds),
            )
            return ControlOperationClaim(operation_id, created_at)

    def complete(self, result: ControlOperationResult) -> None:
        self._set_result(result, ControlOperationState.COMPLETED)

    def fail(self, result: ControlOperationResult) -> None:
        self._set_result(result, ControlOperationState.FAILED)

    def _set_result(
        self,
        result: ControlOperationResult,
        state: ControlOperationState,
    ) -> None:
        key = (result.account_id, result.idempotency_key)
        with self._lock:
            record = self._records.get(key)
            if record is None or record.operation_id != result.operation_id:
                raise KeyError("control operation claim not found")
            record.state = state
            record.result = result
            record.lease_owner = None
            record.lease_expires_at = None


class SqlControlOperationStore:
    def __init__(
        self,
        session_factory: object,
        *,
        lease_seconds: int = 60,
        owner: str | None = None,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        if not isinstance(lease_seconds, int) or not 5 <= lease_seconds <= 3600:
            raise ValueError("lease_seconds must be in [5, 3600]")
        self._session_factory = session_factory
        self._lock = RLock()
        self._lease_seconds = lease_seconds
        self._owner = str(owner or f"control-{uuid4().hex}")

    def lookup(self, *, request: ControlRequest) -> ControlOperationClaim | None:
        from freqtrade.persistence.hedge_models import ControlOperationRow

        with self._session_factory() as session:
            row = session.scalar(
                select(ControlOperationRow).where(
                    ControlOperationRow.account_id == request.account_id,
                    ControlOperationRow.idempotency_key == request.idempotency_key,
                )
            )
            return None if row is None else self._claim_from_row(row, request)

    def latest_results(
        self,
        *,
        account_id: str,
        actions: tuple[str, ...],
    ) -> tuple[ControlOperationResult, ...]:
        from freqtrade.persistence.hedge_models import ControlOperationRow

        if not actions:
            return ()
        with self._session_factory() as session:
            rows = tuple(
                session.scalars(
                    select(ControlOperationRow)
                    .where(
                        ControlOperationRow.account_id == account_id,
                        ControlOperationRow.action.in_(actions),
                        ControlOperationRow.result_json.is_not(None),
                    )
                    .order_by(
                        ControlOperationRow.completed_at.asc(),
                        ControlOperationRow.id.asc(),
                    )
                )
            )
        return tuple(
            ControlOperationResult.from_dict(json.loads(row.result_json))
            for row in rows
            if row.result_json
        )

    def claim(
        self,
        *,
        request: ControlRequest,
        actor: str,
        actor_role: str,
    ) -> ControlOperationClaim:
        from freqtrade.persistence.hedge_models import ControlOperationRow

        with self._lock:
            with self._session_factory.begin() as session:
                existing = session.scalar(
                    select(ControlOperationRow).where(
                        ControlOperationRow.account_id == request.account_id,
                        ControlOperationRow.idempotency_key == request.idempotency_key,
                    )
                )
                if existing is not None:
                    return self._claim_from_row(existing, request, acquire_stale=True, actor=actor)
                row = ControlOperationRow(
                    operation_id=str(uuid4()),
                    account_id=request.account_id,
                    idempotency_key=request.idempotency_key,
                    action=request.action.value,
                    symbol=request.symbol,
                    actor=actor,
                    actor_role=actor_role,
                    reason=request.reason,
                    request_hash=request.request_hash,
                    request_json=json.dumps(
                        request.canonical_payload(),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ),
                    state=ControlOperationState.CLAIMED.value,
                    created_at=datetime.now(UTC).replace(tzinfo=None),
                    lease_owner=self._owner,
                    lease_expires_at=(
                        datetime.now(UTC) + timedelta(seconds=self._lease_seconds)
                    ).replace(tzinfo=None),
                )
                session.add(row)
                try:
                    session.flush([row])
                except IntegrityError:
                    session.rollback()
                    with self._session_factory.begin() as retry_session:
                        existing = retry_session.scalar(
                            select(ControlOperationRow).where(
                                ControlOperationRow.account_id == request.account_id,
                                ControlOperationRow.idempotency_key == request.idempotency_key,
                            )
                        )
                        if existing is None:
                            raise
                        return self._claim_from_row(existing, request, acquire_stale=True, actor=actor)
                return ControlOperationClaim(
                    UUID(row.operation_id),
                    row.created_at.replace(tzinfo=UTC),
                )

    def _claim_from_row(
        self,
        row: object,
        request: ControlRequest,
        *,
        acquire_stale: bool = False,
        actor: str | None = None,
    ) -> ControlOperationClaim:
        request_hash = str(getattr(row, "request_hash"))
        if request_hash != request.request_hash:
            raise ControlOperationConflict("idempotency key payload mismatch")
        raw_result = getattr(row, "result_json")
        result = (
            None
            if not raw_result
            else ControlOperationResult.from_dict(json.loads(str(raw_result)))
        )
        operation_id = UUID(str(getattr(row, "operation_id")))
        created_at = getattr(row, "created_at").replace(tzinfo=UTC)
        if result is not None:
            return ControlOperationClaim(
                operation_id,
                created_at,
                existing_result=result,
            )
        expires = getattr(row, "lease_expires_at", None)
        expires_aware = None if expires is None else expires.replace(tzinfo=UTC)
        now = datetime.now(UTC)
        if expires_aware is not None and expires_aware > now:
            return ControlOperationClaim(operation_id, created_at, in_progress=True)
        if not acquire_stale:
            return ControlOperationClaim(operation_id, created_at, recovered_stale=True)
        if actor is None or str(getattr(row, "actor")) != actor:
            raise ControlOperationConflict("stale operation actor mismatch")
        setattr(row, "lease_owner", self._owner)
        setattr(
            row,
            "lease_expires_at",
            (now + timedelta(seconds=self._lease_seconds)).replace(tzinfo=None),
        )
        setattr(row, "revision", int(getattr(row, "revision", 0)) + 1)
        return ControlOperationClaim(operation_id, created_at, recovered_stale=True)

    def complete(self, result: ControlOperationResult) -> None:
        self._set_result(result, ControlOperationState.COMPLETED)

    def fail(self, result: ControlOperationResult) -> None:
        self._set_result(result, ControlOperationState.FAILED)

    def _set_result(
        self,
        result: ControlOperationResult,
        state: ControlOperationState,
    ) -> None:
        from freqtrade.persistence.hedge_models import ControlOperationRow

        with self._session_factory.begin() as session:
            row = session.scalar(
                select(ControlOperationRow).where(
                    ControlOperationRow.operation_id == str(result.operation_id)
                )
            )
            if row is None:
                raise KeyError("control operation claim not found")
            row.state = state.value
            row.outcome = result.outcome.value
            row.outcome_code = result.code
            row.result_json = json.dumps(
                result.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            row.completed_at = result.completed_at.astimezone(UTC).replace(tzinfo=None)
            row.lease_owner = None
            row.lease_expires_at = None
            row.revision += 1
            session.flush([row])
