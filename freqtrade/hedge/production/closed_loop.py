"""End-to-end HPRL production-cycle contract and durable hash-chained journal.

This module closes the semantic gap between HPRL policy output, the canonical Hedge
planner/execution path, reconciliation evidence and crash recovery.  It deliberately
owns no exchange API capability: all writes remain inside ``HedgeExecutionEngine`` via
``ProductionEquivalentHedgeMainLoop``.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping, Protocol

from freqtrade.hedge.execution.state_machine import OrderState
from freqtrade.hedge.integration.production_main_loop import (
    HedgeMainLoopCycle,
    ProductionEquivalentHedgeMainLoop,
)
from freqtrade.hedge.planning.context import PlanningContext
from freqtrade.hedge.symbols import raw_symbol

from .hprl_hedge_adapter import HprlHedgeAdapter, HprlTargetProjection
from .recovery_checkpoint import DurableRecoveryCheckpoint, RecoveryCheckpointStore

ZERO_HASH = "0" * 64


def _sha(value: object, *, field: str, zero_ok: bool = True) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(ch not in "0123456789abcdef" for ch in result):
        raise ValueError(f"{field} must be SHA-256 hex")
    if not zero_ok and result == ZERO_HASH:
        raise ValueError(f"{field} cannot be zero hash")
    return result


def _aware(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _digest(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(raw.encode("utf-8")).hexdigest()


def _context_payload(context: PlanningContext) -> dict[str, object]:
    wallet = context.wallet
    return {
        "symbol": raw_symbol(context.market.symbol),
        "timestamp": context.market.timestamp.isoformat(),
        "bid": str(context.market.bid),
        "ask": str(context.market.ask),
        "mark": str(context.market.mark),
        "equity": str(wallet.equity),
        "balance": str(wallet.balance),
        "available_balance": str(wallet.available_balance),
        "leverage": str(wallet.leverage),
        "long_quantity": str(wallet.long.quantity),
        "long_average_price": str(wallet.long.average_price),
        "short_quantity": str(wallet.short.quantity),
        "short_average_price": str(wallet.short.average_price),
        "active_orders": [
            {
                "order_id": item.order_id,
                "client_order_id": item.client_order_id,
                "side": item.position_side.value,
                "quantity": str(item.quantity),
                "price": str(item.price),
                "reduce_only": item.reduce_only,
                "action": item.action.value,
            }
            for item in sorted(wallet.active_orders, key=lambda order: order.order_id)
        ],
        "long_state_sequence": context.long_state.sequence,
        "short_state_sequence": context.short_state.sequence,
        "long_signal": str(context.long_signal),
        "short_signal": str(context.short_signal),
        "planner_config": repr(context.config),
    }


def context_state_sha256(context: PlanningContext) -> str:
    return _digest(_context_payload(context))


def _planning_sha256(cycle: HedgeMainLoopCycle | None) -> str:
    if cycle is None or cycle.planning is None:
        return ZERO_HASH
    planning = cycle.planning
    payload = {
        "ideal": [item.intent_id for item in planning.ideal_orders],
        "submit": [item.intent_id for item in planning.submit_orders],
        "cancel": list(planning.cancel_order_ids),
        "modify": list(planning.modify_order_ids),
        "delete": list(planning.delete_order_ids),
        "risk_cancel": list(planning.risk_cancel_order_ids),
        "kept": list(planning.kept_order_ids),
        "target_net_quantity": str(planning.target_net_quantity),
        "net_gap_quantity": str(planning.net_gap_quantity),
        "long_target_quantity": str(planning.long_target_quantity),
        "short_target_quantity": str(planning.short_target_quantity),
        "long_state_sequence": planning.long_state.sequence,
        "short_state_sequence": planning.short_state.sequence,
        "diagnostics": list(planning.diagnostics),
    }
    return _digest(payload)


def _execution_result_payload(result: object) -> dict[str, object]:
    order = result.order
    lifecycle = order.lifecycle
    return {
        "client_order_id": order.client_order_id,
        "intent_id": str(order.intent.intent_id),
        "idempotency_key": order.intent.idempotency_key,
        "position_side": order.intent.position_side.value,
        "action": order.intent.action.value,
        "quantity": str(order.intent.quantity),
        "status": lifecycle.status.value,
        "filled_quantity": str(lifecycle.filled_quantity),
        "average_price": None if lifecycle.average_price is None else str(lifecycle.average_price),
        "version": lifecycle.version,
        "message": result.message,
    }


def _execution_sha256(cycle: HedgeMainLoopCycle | None) -> str:
    if cycle is None:
        return ZERO_HASH
    payload = {
        "cycle_id": cycle.cycle_id,
        "submissions": [_execution_result_payload(item) for item in cycle.submissions],
        "cancellations": [_execution_result_payload(item) for item in cycle.cancellations],
        "blocked_submit": list(cycle.blocked_submit_intent_ids),
        "blocked_cancel": list(cycle.blocked_cancel_order_ids),
        "deferred_submit": list(cycle.deferred_submit_intent_ids),
        "deferred_cancel": list(cycle.deferred_cancel_order_ids),
        "external": list(cycle.external_order_ids),
        "orphan": list(cycle.orphan_order_ids),
        "errors": [
            {
                "operation": item.operation,
                "reference": item.reference,
                "error_type": item.error_type,
                "message": item.message,
            }
            for item in cycle.errors
        ],
        "strategy_state_committed": cycle.strategy_state_committed,
        "writes_attempted": cycle.writes_attempted,
    }
    return _digest(payload)


class ClosedLoopCycleStatus(StrEnum):
    COMMITTED = "COMMITTED"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"
    HALTED = "HALTED"


@dataclass(frozen=True, slots=True)
class ClosedLoopCycleRecord:
    sequence: int
    cycle_id: str
    observed_at: datetime
    source_release: str
    model_id: str
    symbol: str
    projection_sequence: int
    projection_observed_at: datetime
    projection_source_sha256: str
    projection_semantic_sha256: str
    long_margin_ratio: Decimal
    short_margin_ratio: Decimal
    long_notional_ratio: Decimal
    short_notional_ratio: Decimal
    confidence: Decimal
    projection_accepted: bool
    projection_reasons: tuple[str, ...]
    projection_chain_sha256: str
    planner_profile_sha256: str
    input_state_sha256: str
    planning_sha256: str
    execution_sha256: str
    reconciliation_digest: str
    evidence_digest: str
    safety_allows_reduce: bool
    safety_allows_new_risk: bool
    status: ClosedLoopCycleStatus
    writes_attempted: int
    unresolved_client_order_ids: tuple[str, ...] = ()
    previous_record_sha256: str = ZERO_HASH
    record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if self.sequence <= 0:
            raise ValueError("closed-loop sequence must be positive")
        if not self.cycle_id.strip() or not self.source_release.strip() or not self.model_id.strip():
            raise ValueError("cycle_id/source_release/model_id are required")
        if not self.symbol.strip():
            raise ValueError("symbol is required")
        if self.projection_sequence < 0:
            raise ValueError("projection_sequence must be nonnegative")
        if self.writes_attempted < 0:
            raise ValueError("writes_attempted cannot be negative")
        object.__setattr__(self, "observed_at", _aware(self.observed_at, field="observed_at"))
        object.__setattr__(
            self,
            "projection_observed_at",
            _aware(self.projection_observed_at, field="projection_observed_at"),
        )
        for name in (
            "projection_source_sha256",
            "projection_semantic_sha256",
            "projection_chain_sha256",
            "planner_profile_sha256",
            "input_state_sha256",
            "planning_sha256",
            "execution_sha256",
            "reconciliation_digest",
            "evidence_digest",
            "previous_record_sha256",
        ):
            object.__setattr__(self, name, _sha(getattr(self, name), field=name))
        for name in (
            "long_margin_ratio",
            "short_margin_ratio",
            "long_notional_ratio",
            "short_notional_ratio",
            "confidence",
        ):
            value = Decimal(getattr(self, name))
            if not value.is_finite() or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        if self.confidence > 1:
            raise ValueError("confidence cannot exceed one")
        ids = tuple(sorted(str(item).strip() for item in self.unresolved_client_order_ids))
        if any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError("unresolved client order ids must be unique")
        object.__setattr__(self, "unresolved_client_order_ids", ids)
        object.__setattr__(self, "projection_reasons", tuple(str(x) for x in self.projection_reasons))
        object.__setattr__(self, "record_sha256", _digest(self._body()))

    def _body(self) -> dict[str, object]:
        return {
            "sequence": self.sequence,
            "cycle_id": self.cycle_id,
            "observed_at": self.observed_at.isoformat(),
            "source_release": self.source_release,
            "model_id": self.model_id,
            "symbol": self.symbol,
            "projection_sequence": self.projection_sequence,
            "projection_observed_at": self.projection_observed_at.isoformat(),
            "projection_source_sha256": self.projection_source_sha256,
            "projection_semantic_sha256": self.projection_semantic_sha256,
            "long_margin_ratio": str(self.long_margin_ratio),
            "short_margin_ratio": str(self.short_margin_ratio),
            "long_notional_ratio": str(self.long_notional_ratio),
            "short_notional_ratio": str(self.short_notional_ratio),
            "confidence": str(self.confidence),
            "projection_accepted": self.projection_accepted,
            "projection_reasons": list(self.projection_reasons),
            "projection_chain_sha256": self.projection_chain_sha256,
            "planner_profile_sha256": self.planner_profile_sha256,
            "input_state_sha256": self.input_state_sha256,
            "planning_sha256": self.planning_sha256,
            "execution_sha256": self.execution_sha256,
            "reconciliation_digest": self.reconciliation_digest,
            "evidence_digest": self.evidence_digest,
            "safety_allows_reduce": self.safety_allows_reduce,
            "safety_allows_new_risk": self.safety_allows_new_risk,
            "status": self.status.value,
            "writes_attempted": self.writes_attempted,
            "unresolved_client_order_ids": list(self.unresolved_client_order_ids),
            "previous_record_sha256": self.previous_record_sha256,
        }

    def payload(self) -> dict[str, object]:
        return {**self._body(), "record_sha256": self.record_sha256}

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "ClosedLoopCycleRecord":
        result = cls(
            sequence=int(raw["sequence"]),
            cycle_id=str(raw["cycle_id"]),
            observed_at=datetime.fromisoformat(str(raw["observed_at"])),
            source_release=str(raw["source_release"]),
            model_id=str(raw["model_id"]),
            symbol=str(raw["symbol"]),
            projection_sequence=int(raw["projection_sequence"]),
            projection_observed_at=datetime.fromisoformat(str(raw["projection_observed_at"])),
            projection_source_sha256=str(raw["projection_source_sha256"]),
            projection_semantic_sha256=str(raw["projection_semantic_sha256"]),
            long_margin_ratio=Decimal(str(raw["long_margin_ratio"])),
            short_margin_ratio=Decimal(str(raw["short_margin_ratio"])),
            long_notional_ratio=Decimal(str(raw["long_notional_ratio"])),
            short_notional_ratio=Decimal(str(raw["short_notional_ratio"])),
            confidence=Decimal(str(raw["confidence"])),
            projection_accepted=bool(raw["projection_accepted"]),
            projection_reasons=tuple(str(x) for x in raw.get("projection_reasons", [])),
            projection_chain_sha256=str(raw["projection_chain_sha256"]),
            planner_profile_sha256=str(raw["planner_profile_sha256"]),
            input_state_sha256=str(raw["input_state_sha256"]),
            planning_sha256=str(raw["planning_sha256"]),
            execution_sha256=str(raw["execution_sha256"]),
            reconciliation_digest=str(raw["reconciliation_digest"]),
            evidence_digest=str(raw["evidence_digest"]),
            safety_allows_reduce=bool(raw["safety_allows_reduce"]),
            safety_allows_new_risk=bool(raw["safety_allows_new_risk"]),
            status=ClosedLoopCycleStatus(str(raw["status"])),
            writes_attempted=int(raw["writes_attempted"]),
            unresolved_client_order_ids=tuple(str(x) for x in raw.get("unresolved_client_order_ids", [])),
            previous_record_sha256=str(raw.get("previous_record_sha256", ZERO_HASH)),
        )
        if str(raw.get("record_sha256", "")) != result.record_sha256:
            raise ValueError("closed-loop cycle record hash mismatch")
        return result

    def as_previous_projection(self) -> HprlTargetProjection:
        return HprlTargetProjection(
            sequence=self.projection_sequence,
            observed_at=self.projection_observed_at,
            symbol=self.symbol,
            model_id=self.model_id,
            long_margin_ratio=self.long_margin_ratio,
            short_margin_ratio=self.short_margin_ratio,
            long_notional_ratio=self.long_notional_ratio,
            short_notional_ratio=self.short_notional_ratio,
            confidence=self.confidence,
            accepted=self.projection_accepted,
            reasons=self.projection_reasons,
            source_sha256=self.projection_source_sha256,
        )


class ClosedLoopCycleJournal:
    def __init__(self, records: tuple[ClosedLoopCycleRecord, ...] = ()) -> None:
        self._records = list(records)
        if not self.verify():
            raise ValueError("closed-loop journal hash chain is invalid")

    @property
    def records(self) -> tuple[ClosedLoopCycleRecord, ...]:
        return tuple(self._records)

    @property
    def tip_sha256(self) -> str:
        return self._records[-1].record_sha256 if self._records else ZERO_HASH

    @property
    def projection_chain_sha256(self) -> str:
        return self._records[-1].projection_chain_sha256 if self._records else ZERO_HASH

    @property
    def last(self) -> ClosedLoopCycleRecord | None:
        return self._records[-1] if self._records else None

    def append(self, record: ClosedLoopCycleRecord) -> None:
        if record.sequence != len(self._records) + 1:
            raise ValueError("closed-loop journal sequence is not monotonic")
        if record.previous_record_sha256 != self.tip_sha256:
            raise ValueError("closed-loop journal previous hash mismatch")
        self._records.append(record)

    def verify(self) -> bool:
        previous = ZERO_HASH
        for index, record in enumerate(self._records, start=1):
            if record.sequence != index or record.previous_record_sha256 != previous:
                return False
            rebuilt = ClosedLoopCycleRecord.from_payload(record.payload())
            if rebuilt.record_sha256 != record.record_sha256:
                return False
            previous = record.record_sha256
        return True

    def payload(self) -> dict[str, object]:
        return {
            "schema": "freqtrade-hedge-hprl-v3-closed-loop-journal-v1",
            "records": [item.payload() for item in self._records],
            "tip_sha256": self.tip_sha256,
        }

    @classmethod
    def from_payload(cls, raw: Mapping[str, object]) -> "ClosedLoopCycleJournal":
        if raw.get("schema") != "freqtrade-hedge-hprl-v3-closed-loop-journal-v1":
            raise ValueError("unsupported closed-loop journal schema")
        values = raw.get("records")
        if not isinstance(values, list):
            raise ValueError("closed-loop journal records must be a list")
        records = tuple(ClosedLoopCycleRecord.from_payload(item) for item in values if isinstance(item, Mapping))
        if len(records) != len(values):
            raise ValueError("closed-loop journal contains invalid records")
        journal = cls(records)
        if str(raw.get("tip_sha256", journal.tip_sha256)) != journal.tip_sha256:
            raise ValueError("closed-loop journal tip mismatch")
        return journal


class ClosedLoopJournalConcurrencyError(RuntimeError):
    pass


class ClosedLoopJournalStorePort(Protocol):
    def load(self) -> ClosedLoopCycleJournal: ...

    def append_atomic(
        self,
        record: ClosedLoopCycleRecord,
        *,
        expected_previous_sha256: str,
    ) -> ClosedLoopCycleJournal: ...


class RecoveryCheckpointStorePort(Protocol):
    def load(self) -> DurableRecoveryCheckpoint | None: ...

    def save_atomic(self, checkpoint: DurableRecoveryCheckpoint) -> object: ...


class ClosedLoopCycleJournalStore:
    """Atomic append-only journal store with compare-and-swap tip enforcement."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> ClosedLoopCycleJournal:
        if not self.path.exists():
            return ClosedLoopCycleJournal()
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise ValueError("closed-loop journal root must be an object")
        return ClosedLoopCycleJournal.from_payload(raw)

    @contextmanager
    def _exclusive_lock(self):
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            try:
                import fcntl
            except ImportError:  # pragma: no cover - Windows-only local tooling
                yield
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def append_atomic(
        self,
        record: ClosedLoopCycleRecord,
        *,
        expected_previous_sha256: str,
    ) -> ClosedLoopCycleJournal:
        expected = _sha(expected_previous_sha256, field="expected_previous_sha256")
        with self._exclusive_lock():
            journal = self.load()
            if journal.tip_sha256 != expected:
                raise ClosedLoopJournalConcurrencyError(
                    "closed-loop journal changed since cycle planning"
                )
            journal.append(record)
            self._save_atomic(journal)
            return journal

    def _save_atomic(self, journal: ClosedLoopCycleJournal) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        raw = json.dumps(journal.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
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


@dataclass(frozen=True, slots=True)
class HprlClosedLoopOutcome:
    projection: HprlTargetProjection
    planning_context: PlanningContext
    main_loop_cycle: HedgeMainLoopCycle | None
    record: ClosedLoopCycleRecord
    checkpoint: DurableRecoveryCheckpoint | None


class HprlProductionClosedLoop:
    """Coordinates one model target through planner/execution/evidence/recovery semantics."""

    def __init__(
        self,
        *,
        adapter: HprlHedgeAdapter,
        main_loop: ProductionEquivalentHedgeMainLoop,
        source_release: str,
        journal_store: ClosedLoopJournalStorePort,
        checkpoint_store: RecoveryCheckpointStorePort | None = None,
    ) -> None:
        if not source_release.strip():
            raise ValueError("source_release is required")
        self.adapter = adapter
        self.main_loop = main_loop
        self.source_release = source_release.strip()
        self.journal_store = journal_store
        self.checkpoint_store = checkpoint_store

    def run(
        self,
        intent: object,
        *,
        projection_sequence: int,
        observed_at: datetime,
        now: datetime,
        context: PlanningContext,
        evidence_digest: str,
        reconciliation_digest: str,
        last_market_sequence: int,
        last_user_sequence: int,
        safety_allows_reduce: bool,
        safety_allows_new_risk: bool,
    ) -> HprlClosedLoopOutcome:
        evidence = _sha(evidence_digest, field="evidence_digest")
        reconciliation = _sha(reconciliation_digest, field="reconciliation_digest")
        current = _aware(now, field="now")
        journal = self.journal_store.load()
        previous_record = journal.last
        previous_projection = None if previous_record is None else previous_record.as_previous_projection()
        projection = self.adapter.adapt(
            intent,
            sequence=projection_sequence,
            observed_at=observed_at,
            now=current,
            previous=previous_projection,
        )
        planning_context, profile_sha = self.adapter.apply_to_context(
            context,
            projection,
            allow_new_risk=safety_allows_new_risk,
        )
        input_sha = context_state_sha256(planning_context)
        cycle_material = {
            "source_release": self.source_release,
            "projection": projection.semantic_sha256,
            "planner_profile": profile_sha,
            "input_state": input_sha,
            "previous_cycle": journal.tip_sha256,
        }
        cycle_id = "hprl-cycle-" + _digest(cycle_material)[:32]

        if safety_allows_reduce:
            main_cycle = self.main_loop.run_cycle(
                planning_context,
                strategy_allows_new_risk=(safety_allows_new_risk and projection.accepted),
                cycle_id_override=cycle_id,
            )
            if main_cycle.errors:
                status = ClosedLoopCycleStatus.ERROR
            elif not main_cycle.successful or not projection.accepted:
                status = ClosedLoopCycleStatus.BLOCKED
            else:
                status = ClosedLoopCycleStatus.COMMITTED
            writes_attempted = main_cycle.writes_attempted
        else:
            main_cycle = None
            status = ClosedLoopCycleStatus.HALTED
            writes_attempted = 0

        unknown = tuple(
            sorted(
                order.client_order_id
                for order in self.main_loop.engine.core.list_orders(
                    statuses=(OrderState.UNKNOWN,), include_terminal=False
                )
            )
        )
        projection_chain = _digest(
            {
                "previous": journal.projection_chain_sha256,
                "projection": projection.semantic_sha256,
            }
        )
        record = ClosedLoopCycleRecord(
            sequence=len(journal.records) + 1,
            cycle_id=cycle_id,
            observed_at=current,
            source_release=self.source_release,
            model_id=projection.model_id,
            symbol=raw_symbol(projection.symbol),
            projection_sequence=projection.sequence,
            projection_observed_at=projection.observed_at,
            projection_source_sha256=projection.source_sha256,
            projection_semantic_sha256=projection.semantic_sha256,
            long_margin_ratio=projection.long_margin_ratio,
            short_margin_ratio=projection.short_margin_ratio,
            long_notional_ratio=projection.long_notional_ratio,
            short_notional_ratio=projection.short_notional_ratio,
            confidence=projection.confidence,
            projection_accepted=projection.accepted,
            projection_reasons=projection.reasons,
            projection_chain_sha256=projection_chain,
            planner_profile_sha256=profile_sha,
            input_state_sha256=input_sha,
            planning_sha256=_planning_sha256(main_cycle),
            execution_sha256=_execution_sha256(main_cycle),
            reconciliation_digest=reconciliation,
            evidence_digest=evidence,
            safety_allows_reduce=bool(safety_allows_reduce),
            safety_allows_new_risk=bool(safety_allows_new_risk),
            status=status,
            writes_attempted=writes_attempted,
            unresolved_client_order_ids=unknown,
            previous_record_sha256=journal.tip_sha256,
        )
        persisted = self.journal_store.append_atomic(
            record,
            expected_previous_sha256=journal.tip_sha256,
        )

        checkpoint = None
        if self.checkpoint_store is not None:
            previous_checkpoint = self.checkpoint_store.load()
            generation = 1 if previous_checkpoint is None else previous_checkpoint.generation + 1
            checkpoint = DurableRecoveryCheckpoint(
                generation=generation,
                created_at=current,
                source_release=self.source_release,
                model_id=projection.model_id,
                evidence_digest=evidence,
                reconciliation_digest=reconciliation,
                projection_chain_sha256=projection_chain,
                last_market_sequence=last_market_sequence,
                last_user_sequence=last_user_sequence,
                unresolved_client_order_ids=unknown,
                metadata=(
                    ("closed_loop_cycle_id", cycle_id),
                    ("closed_loop_cycle_sha256", record.record_sha256),
                    ("closed_loop_journal_tip_sha256", persisted.tip_sha256),
                    ("closed_loop_status", status.value),
                ),
            )
            self.checkpoint_store.save_atomic(checkpoint)
        return HprlClosedLoopOutcome(
            projection=projection,
            planning_context=planning_context,
            main_loop_cycle=main_cycle,
            record=record,
            checkpoint=checkpoint,
        )
