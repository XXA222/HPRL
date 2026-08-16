"""Durable HPRL/Execution recovery checkpoint and fail-closed convergence barrier."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from freqtrade.hedge.execution.service import ExecutionOrder, OrderState

from .recovery import CrashPoint, RecoveryAction, RecoveryContext, RecoveryPlan, build_recovery_plan

ZERO_HASH = "0" * 64


def _hash(value: object, *, field_name: str, allow_zero: bool = True) -> str:
    result = str(value).lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ValueError(f"{field_name} must be SHA-256 hex")
    if not allow_zero and result == ZERO_HASH:
        raise ValueError(f"{field_name} cannot be zero hash")
    return result


def _aware(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class DurableRecoveryCheckpoint:
    generation: int
    created_at: datetime
    source_release: str
    model_id: str
    evidence_digest: str
    reconciliation_digest: str
    projection_chain_sha256: str
    last_market_sequence: int
    last_user_sequence: int
    unresolved_client_order_ids: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()
    checkpoint_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.generation <= 0:
            raise ValueError("checkpoint generation must be positive")
        object.__setattr__(self, "created_at", _aware(self.created_at, field_name="created_at"))
        if not self.source_release.strip() or not self.model_id.strip():
            raise ValueError("source_release/model_id are required")
        for name in ("evidence_digest", "reconciliation_digest", "projection_chain_sha256"):
            object.__setattr__(self, name, _hash(getattr(self, name), field_name=name))
        if self.last_market_sequence < 0 or self.last_user_sequence < 0:
            raise ValueError("checkpoint sequences must be nonnegative")
        ids = tuple(str(item).strip() for item in self.unresolved_client_order_ids)
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("unresolved client order ids must be unique non-empty strings")
        object.__setattr__(self, "unresolved_client_order_ids", tuple(sorted(ids)))
        metadata = tuple(sorted((str(k), str(v)) for k, v in self.metadata))
        if len(metadata) != len({key for key, _ in metadata}):
            raise ValueError("checkpoint metadata keys must be unique")
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "checkpoint_sha256", self._compute_hash())

    def _body(self) -> dict[str, object]:
        return {
            "generation": self.generation,
            "created_at": self.created_at.isoformat(),
            "source_release": self.source_release,
            "model_id": self.model_id,
            "evidence_digest": self.evidence_digest,
            "reconciliation_digest": self.reconciliation_digest,
            "projection_chain_sha256": self.projection_chain_sha256,
            "last_market_sequence": self.last_market_sequence,
            "last_user_sequence": self.last_user_sequence,
            "unresolved_client_order_ids": list(self.unresolved_client_order_ids),
            "metadata": [list(item) for item in self.metadata],
        }

    def _compute_hash(self) -> str:
        raw = json.dumps(self._body(), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        return sha256(raw).hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "schema": "freqtrade-hedge-hprl-v3-recovery-checkpoint-v1",
            **self._body(),
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "DurableRecoveryCheckpoint":
        if raw.get("schema") != "freqtrade-hedge-hprl-v3-recovery-checkpoint-v1":
            raise ValueError("unsupported recovery checkpoint schema")
        metadata_raw = raw.get("metadata", [])
        if not isinstance(metadata_raw, list):
            raise ValueError("checkpoint metadata must be a list")
        metadata: list[tuple[str, str]] = []
        for item in metadata_raw:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise ValueError("checkpoint metadata entry is invalid")
            metadata.append((str(item[0]), str(item[1])))
        unresolved_raw = raw.get("unresolved_client_order_ids", [])
        if not isinstance(unresolved_raw, list):
            raise ValueError("unresolved_client_order_ids must be a list")
        checkpoint = cls(
            generation=int(raw["generation"]),
            created_at=datetime.fromisoformat(str(raw["created_at"])),
            source_release=str(raw["source_release"]),
            model_id=str(raw["model_id"]),
            evidence_digest=str(raw["evidence_digest"]),
            reconciliation_digest=str(raw["reconciliation_digest"]),
            projection_chain_sha256=str(raw["projection_chain_sha256"]),
            last_market_sequence=int(raw["last_market_sequence"]),
            last_user_sequence=int(raw["last_user_sequence"]),
            unresolved_client_order_ids=tuple(str(item) for item in unresolved_raw),
            metadata=tuple(metadata),
        )
        expected = str(raw.get("checkpoint_sha256", ""))
        if expected != checkpoint.checkpoint_sha256:
            raise ValueError("recovery checkpoint hash mismatch")
        return checkpoint


class RecoveryCheckpointStore:
    """Atomic fsync-backed checkpoint store with monotonic generation enforcement."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> DurableRecoveryCheckpoint | None:
        if not self.path.exists():
            return None
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("recovery checkpoint root must be an object")
        return DurableRecoveryCheckpoint.from_payload(raw)

    def save_atomic(self, checkpoint: DurableRecoveryCheckpoint) -> Path:
        if not isinstance(checkpoint, DurableRecoveryCheckpoint):
            raise TypeError("checkpoint must be DurableRecoveryCheckpoint")
        current = self.load()
        if current is not None and checkpoint.generation <= current.generation:
            raise ValueError("recovery checkpoint generation must advance monotonically")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        raw = json.dumps(checkpoint.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        temporary.write_text(raw, encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)
        try:
            directory_fd = os.open(str(self.path.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return self.path


@dataclass(frozen=True, slots=True)
class RecoveryBarrierPolicy:
    max_checkpoint_age: timedelta = timedelta(minutes=2)
    require_reconciliation_digest_match: bool = True
    require_evidence_digest_match: bool = True

    def __post_init__(self) -> None:
        if self.max_checkpoint_age <= timedelta(0):
            raise ValueError("max_checkpoint_age must be positive")


@dataclass(frozen=True, slots=True)
class RecoveryBarrierReport:
    passed: bool
    allow_reduce: bool
    allow_new_risk: bool
    reasons: tuple[str, ...]
    unresolved_client_order_ids: tuple[str, ...]
    plan: RecoveryPlan
    checkpoint_sha256: str | None


class RecoveryConvergenceBarrier:
    """Cross-check checkpoint state against the authoritative execution store projection."""

    def __init__(self, policy: RecoveryBarrierPolicy | None = None) -> None:
        self.policy = policy or RecoveryBarrierPolicy()

    def evaluate(
        self,
        checkpoint: DurableRecoveryCheckpoint | None,
        *,
        orders: Iterable[ExecutionOrder],
        now: datetime,
        current_evidence_digest: str,
        current_reconciliation_digest: str,
    ) -> RecoveryBarrierReport:
        current = _aware(now, field_name="now")
        evidence = _hash(current_evidence_digest, field_name="current_evidence_digest")
        reconciliation = _hash(current_reconciliation_digest, field_name="current_reconciliation_digest")
        order_list = tuple(orders)
        if any(not isinstance(item, ExecutionOrder) for item in order_list):
            raise TypeError("orders must contain ExecutionOrder values")
        unknown = tuple(sorted(
            item.client_order_id
            for item in order_list
            if item.lifecycle.status is OrderState.UNKNOWN
        ))
        reasons: list[str] = []
        if checkpoint is None:
            reasons.append("RECOVERY_CHECKPOINT_MISSING")
        else:
            if checkpoint.created_at > current:
                reasons.append("RECOVERY_CHECKPOINT_FROM_FUTURE")
            elif current - checkpoint.created_at > self.policy.max_checkpoint_age:
                reasons.append("RECOVERY_CHECKPOINT_STALE")
            if self.policy.require_evidence_digest_match and checkpoint.evidence_digest != evidence:
                reasons.append("RECOVERY_EVIDENCE_DIGEST_CHANGED")
            if (
                self.policy.require_reconciliation_digest_match
                and checkpoint.reconciliation_digest != reconciliation
            ):
                reasons.append("RECOVERY_RECONCILIATION_DIGEST_CHANGED")
            if set(checkpoint.unresolved_client_order_ids) != set(unknown):
                reasons.append("RECOVERY_UNKNOWN_ORDER_SET_CHANGED")
        if unknown:
            reasons.append("RECOVERY_UNKNOWN_ORDERS_PRESENT")
        crash_point = (
            CrashPoint.STALE_OR_CORRUPT_CHECKPOINT if checkpoint is None or reasons
            else CrashPoint.AFTER_DB_COMMIT
        )
        plan = build_recovery_plan(
            RecoveryContext(
                crash_point=crash_point,
                intent_committed=True,
                exchange_submit_maybe_sent=bool(unknown),
                ack_persisted=not bool(unknown),
                unknown_order_present=bool(unknown),
                checkpoint_valid=checkpoint is not None and not reasons,
                durable_facts_available=True,
            )
        )
        allow_new = not reasons and not unknown and plan.new_risk_allowed_before_convergence
        # build_recovery_plan deliberately keeps new risk closed until convergence.  A clean
        # checkpoint plus no UNKNOWN orders represents convergence at this higher barrier.
        if not reasons and not unknown:
            allow_new = True
        allow_reduce = not any(
            reason in {"RECOVERY_CHECKPOINT_FROM_FUTURE"}
            for reason in reasons
        )
        return RecoveryBarrierReport(
            passed=not reasons,
            allow_reduce=allow_reduce,
            allow_new_risk=allow_new,
            reasons=tuple(dict.fromkeys(reasons)),
            unresolved_client_order_ids=unknown,
            plan=plan,
            checkpoint_sha256=None if checkpoint is None else checkpoint.checkpoint_sha256,
        )


def mandatory_recovery_actions(report: RecoveryBarrierReport) -> tuple[RecoveryAction, ...]:
    """Expose a stable execution-oriented action list for startup orchestration."""
    return report.plan.actions
