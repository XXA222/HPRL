"""Bind Binance dry-run telemetry to the authoritative HPRL closed-loop cycle journal."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from freqtrade.hedge.telemetry.dryrun import DryRunCycleTelemetry

from .binance_dryrun import (
    BinanceDryRunAcceptanceReport,
    BinanceDryRunPolicy,
    BinanceDryRunSafetyContext,
    evaluate_binance_dryrun,
)
from .closed_loop import ClosedLoopCycleJournal


@dataclass(frozen=True, slots=True)
class ClosedLoopDryRunAcceptanceReport:
    passed: bool
    base: BinanceDryRunAcceptanceReport
    journal_cycle_count: int
    telemetry_cycle_count: int
    linked_cycle_count: int
    journal_tip_sha256: str
    reasons: tuple[str, ...]


def evaluate_closed_loop_binance_dryrun(
    telemetry: Iterable[DryRunCycleTelemetry],
    *,
    journal: ClosedLoopCycleJournal,
    safety: BinanceDryRunSafetyContext,
    policy: BinanceDryRunPolicy | None = None,
    require_exact_cycle_set: bool = True,
) -> ClosedLoopDryRunAcceptanceReport:
    if not isinstance(journal, ClosedLoopCycleJournal):
        raise TypeError("journal must be ClosedLoopCycleJournal")
    if not journal.verify():
        raise ValueError("closed-loop journal hash chain is invalid")
    items = tuple(telemetry)
    base = evaluate_binance_dryrun(items, safety=safety, policy=policy)
    telemetry_ids = tuple(item.cycle_id for item in items)
    journal_ids = tuple(item.cycle_id for item in journal.records)
    telemetry_set = set(telemetry_ids)
    journal_set = set(journal_ids)
    linked = telemetry_set & journal_set
    reasons = list(base.reasons)
    if not journal.records:
        reasons.append("BINANCE_DRYRUN_CLOSED_LOOP_JOURNAL_EMPTY")
    if any(cycle_id not in journal_set for cycle_id in telemetry_ids):
        reasons.append("BINANCE_DRYRUN_UNJOURNALED_CYCLE")
    if require_exact_cycle_set and telemetry_set != journal_set:
        reasons.append("BINANCE_DRYRUN_CYCLE_SET_MISMATCH")
    if journal.records and items:
        # Sequence/time order must preserve the journal order for the telemetry subset.
        journal_order = {cycle_id: index for index, cycle_id in enumerate(journal_ids)}
        observed_order = [journal_order.get(cycle_id, -1) for cycle_id in telemetry_ids]
        if observed_order != sorted(observed_order):
            reasons.append("BINANCE_DRYRUN_JOURNAL_ORDER_MISMATCH")
    return ClosedLoopDryRunAcceptanceReport(
        passed=not reasons,
        base=base,
        journal_cycle_count=len(journal.records),
        telemetry_cycle_count=len(items),
        linked_cycle_count=len(linked),
        journal_tip_sha256=journal.tip_sha256,
        reasons=tuple(dict.fromkeys(reasons)),
    )
