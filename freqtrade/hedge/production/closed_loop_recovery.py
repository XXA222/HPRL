"""Restart convergence barrier binding recovery checkpoints to the closed-loop journal."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from freqtrade.hedge.execution.service import ExecutionOrder

from .closed_loop import ClosedLoopCycleJournal, ZERO_HASH
from .recovery_checkpoint import (
    DurableRecoveryCheckpoint,
    RecoveryBarrierReport,
    RecoveryConvergenceBarrier,
)


@dataclass(frozen=True, slots=True)
class ClosedLoopRecoveryReport:
    passed: bool
    allow_reduce: bool
    allow_new_risk: bool
    reasons: tuple[str, ...]
    base: RecoveryBarrierReport
    journal_tip_sha256: str
    checkpoint_cycle_sha256: str | None


class ClosedLoopRecoveryBarrier:
    """Fail closed unless execution, checkpoint and hash-chained cycle journal converge."""

    def __init__(self, base: RecoveryConvergenceBarrier | None = None) -> None:
        self.base = base or RecoveryConvergenceBarrier()

    def evaluate(
        self,
        checkpoint: DurableRecoveryCheckpoint | None,
        journal: ClosedLoopCycleJournal,
        *,
        orders: Iterable[ExecutionOrder],
        now: datetime,
        current_evidence_digest: str,
        current_reconciliation_digest: str,
    ) -> ClosedLoopRecoveryReport:
        if not isinstance(journal, ClosedLoopCycleJournal):
            raise TypeError("journal must be ClosedLoopCycleJournal")
        if not journal.verify():
            raise ValueError("closed-loop journal hash chain is invalid")
        order_values = tuple(orders)
        base = self.base.evaluate(
            checkpoint,
            orders=order_values,
            now=now,
            current_evidence_digest=current_evidence_digest,
            current_reconciliation_digest=current_reconciliation_digest,
        )
        reasons = list(base.reasons)
        checkpoint_cycle_sha = None
        if checkpoint is None:
            reasons.append("CLOSED_LOOP_CHECKPOINT_MISSING")
        else:
            metadata = dict(checkpoint.metadata)
            checkpoint_cycle_sha = metadata.get("closed_loop_cycle_sha256")
            checkpoint_tip = metadata.get("closed_loop_journal_tip_sha256")
            checkpoint_cycle_id = metadata.get("closed_loop_cycle_id")
            if journal.last is None:
                if checkpoint_tip not in {None, ZERO_HASH}:
                    reasons.append("CLOSED_LOOP_JOURNAL_MISSING")
            else:
                if checkpoint_tip != journal.tip_sha256:
                    reasons.append("CLOSED_LOOP_JOURNAL_TIP_MISMATCH")
                if checkpoint_cycle_sha != journal.last.record_sha256:
                    reasons.append("CLOSED_LOOP_CYCLE_HASH_MISMATCH")
                if checkpoint_cycle_id != journal.last.cycle_id:
                    reasons.append("CLOSED_LOOP_CYCLE_ID_MISMATCH")
                if checkpoint.projection_chain_sha256 != journal.projection_chain_sha256:
                    reasons.append("CLOSED_LOOP_PROJECTION_CHAIN_MISMATCH")
                current_unknown = tuple(sorted(
                    item.client_order_id
                    for item in order_values
                    if item.lifecycle.status.value == "UNKNOWN"
                ))
                if journal.last.unresolved_client_order_ids != current_unknown:
                    reasons.append("CLOSED_LOOP_JOURNAL_UNKNOWN_SET_MISMATCH")
        unique = tuple(dict.fromkeys(reasons))
        link_failure = any(reason.startswith("CLOSED_LOOP_") for reason in unique)
        return ClosedLoopRecoveryReport(
            passed=base.passed and not link_failure,
            allow_reduce=base.allow_reduce and "CLOSED_LOOP_JOURNAL_MISSING" not in unique,
            allow_new_risk=base.allow_new_risk and not link_failure,
            reasons=unique,
            base=base,
            journal_tip_sha256=journal.tip_sha256,
            checkpoint_cycle_sha256=checkpoint_cycle_sha,
        )
