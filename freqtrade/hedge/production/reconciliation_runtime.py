"""Convergence supervisor and corrective-action planner for continuous reconciliation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from .contracts import Severity
from .reconciliation import DiffKind, ReconciliationResult


class ReconciliationAction(StrEnum):
    REFRESH_EXCHANGE_SNAPSHOT = "REFRESH_EXCHANGE_SNAPSHOT"
    QUERY_ORDER_BY_CLIENT_ID = "QUERY_ORDER_BY_CLIENT_ID"
    IMPORT_EXTERNAL_ORDER = "IMPORT_EXTERNAL_ORDER"
    REBUILD_POSITION_PROJECTION = "REBUILD_POSITION_PROJECTION"
    REBUILD_BALANCE_PROJECTION = "REBUILD_BALANCE_PROJECTION"
    REPLAY_MISSING_EVENTS = "REPLAY_MISSING_EVENTS"
    VERIFY_ACCOUNT_MODE = "VERIFY_ACCOUNT_MODE"
    VERIFY_MARGIN_MODE = "VERIFY_MARGIN_MODE"
    HALT_NEW_RISK = "HALT_NEW_RISK"
    HALT_ACCOUNT = "HALT_ACCOUNT"
    REQUIRE_MANUAL_REVIEW = "REQUIRE_MANUAL_REVIEW"


@dataclass(frozen=True, slots=True)
class ReconciliationPlan:
    actions: tuple[ReconciliationAction, ...]
    requires_manual_review: bool
    maximum_severity: Severity | None


@dataclass(frozen=True, slots=True)
class ReconciliationSupervisorPolicy:
    confirmations_for_new_risk: int = 3
    confirmations_for_reduce: int = 1
    max_snapshot_age: timedelta = timedelta(seconds=15)
    max_nonconverged_duration: timedelta = timedelta(minutes=2)

    def __post_init__(self) -> None:
        if self.confirmations_for_new_risk <= 0 or self.confirmations_for_reduce <= 0:
            raise ValueError("reconciliation confirmation counts must be positive")
        if self.max_snapshot_age <= timedelta(0) or self.max_nonconverged_duration <= timedelta(0):
            raise ValueError("reconciliation durations must be positive")


@dataclass(frozen=True, slots=True)
class ReconciliationSupervisorSnapshot:
    consecutive_converged: int
    allow_new_risk: bool
    allow_reduce: bool
    first_nonconverged_at: datetime | None
    last_observed_at: datetime | None
    reasons: tuple[str, ...]


def build_reconciliation_plan(result: ReconciliationResult) -> ReconciliationPlan:
    actions: list[ReconciliationAction] = []
    severities = [item.severity for item in result.diffs]
    for item in result.diffs:
        if item.kind is DiffKind.POSITION:
            actions.extend((ReconciliationAction.REFRESH_EXCHANGE_SNAPSHOT, ReconciliationAction.REBUILD_POSITION_PROJECTION))
        elif item.kind is DiffKind.OPEN_ORDER:
            actions.extend((ReconciliationAction.QUERY_ORDER_BY_CLIENT_ID, ReconciliationAction.REFRESH_EXCHANGE_SNAPSHOT))
        elif item.kind is DiffKind.UNKNOWN_ORDER:
            actions.extend((ReconciliationAction.IMPORT_EXTERNAL_ORDER, ReconciliationAction.REQUIRE_MANUAL_REVIEW))
        elif item.kind is DiffKind.BALANCE:
            actions.extend((ReconciliationAction.REFRESH_EXCHANGE_SNAPSHOT, ReconciliationAction.REBUILD_BALANCE_PROJECTION))
        elif item.kind is DiffKind.CURSOR:
            actions.extend((ReconciliationAction.REPLAY_MISSING_EVENTS, ReconciliationAction.REFRESH_EXCHANGE_SNAPSHOT))
        elif item.kind is DiffKind.MODE:
            actions.extend((ReconciliationAction.VERIFY_ACCOUNT_MODE, ReconciliationAction.VERIFY_MARGIN_MODE, ReconciliationAction.REQUIRE_MANUAL_REVIEW))
        elif item.kind is DiffKind.LEVERAGE:
            actions.extend((ReconciliationAction.REFRESH_EXCHANGE_SNAPSHOT, ReconciliationAction.REQUIRE_MANUAL_REVIEW))
    if any(level is Severity.HALT_ACCOUNT for level in severities):
        actions.insert(0, ReconciliationAction.HALT_ACCOUNT)
    elif any(level is Severity.HALT_NEW_RISK for level in severities):
        actions.insert(0, ReconciliationAction.HALT_NEW_RISK)
    ordered = tuple(dict.fromkeys(actions))
    maximum = None
    if severities:
        maximum = Severity.HALT_ACCOUNT if Severity.HALT_ACCOUNT in severities else Severity.HALT_NEW_RISK
    return ReconciliationPlan(
        ordered,
        ReconciliationAction.REQUIRE_MANUAL_REVIEW in ordered,
        maximum,
    )


class ReconciliationSupervisor:
    """Requires repeated fresh convergence before restoring write capability."""

    def __init__(self, policy: ReconciliationSupervisorPolicy | None = None) -> None:
        self.policy = policy or ReconciliationSupervisorPolicy()
        self._consecutive = 0
        self._first_nonconverged_at: datetime | None = None
        self._last_observed_at: datetime | None = None
        self._last_result: ReconciliationResult | None = None

    def observe(
        self,
        result: ReconciliationResult,
        *,
        observed_at: datetime,
        now: datetime,
    ) -> ReconciliationSupervisorSnapshot:
        if observed_at.tzinfo is None or now.tzinfo is None:
            raise ValueError("reconciliation timestamps must be timezone-aware")
        observed_at = observed_at.astimezone(UTC)
        now = now.astimezone(UTC)
        if self._last_observed_at is not None and observed_at < self._last_observed_at:
            self._consecutive = 0
            self._first_nonconverged_at = self._first_nonconverged_at or now
            return self._snapshot(("RECONCILIATION_TIMESTAMP_REGRESSION",))
        self._last_observed_at = observed_at
        self._last_result = result
        reasons: list[str] = []
        if now - observed_at > self.policy.max_snapshot_age:
            self._consecutive = 0
            self._first_nonconverged_at = self._first_nonconverged_at or now
            reasons.append("RECONCILIATION_SNAPSHOT_STALE")
        elif result.converged:
            self._consecutive += 1
            self._first_nonconverged_at = None
        else:
            self._consecutive = 0
            self._first_nonconverged_at = self._first_nonconverged_at or now
            reasons.extend(f"DIFF:{item.kind.value}:{item.key}" for item in result.diffs)
        if (
            self._first_nonconverged_at is not None
            and now - self._first_nonconverged_at > self.policy.max_nonconverged_duration
        ):
            reasons.append("RECONCILIATION_NONCONVERGED_SLA")
        return self._snapshot(tuple(reasons))

    def _snapshot(self, reasons: tuple[str, ...]) -> ReconciliationSupervisorSnapshot:
        result = self._last_result
        raw_reduce = bool(result and result.allow_reduce)
        stale_or_regressed = any(
            reason in {"RECONCILIATION_SNAPSHOT_STALE", "RECONCILIATION_TIMESTAMP_REGRESSION"}
            for reason in reasons
        )
        # A position/balance drift may legitimately block new risk while a controlled
        # reduction remains the safest action.  Do not accidentally require *full*
        # convergence before allowing such reduction; only hard-account diffs or stale
        # facts may block it.
        allow_reduce = raw_reduce and not stale_or_regressed
        allow_new = bool(result and result.allow_new_risk) and self._consecutive >= self.policy.confirmations_for_new_risk
        if self._first_nonconverged_at is not None:
            allow_new = False
        return ReconciliationSupervisorSnapshot(
            self._consecutive,
            allow_new,
            allow_reduce,
            self._first_nonconverged_at,
            self._last_observed_at,
            reasons,
        )
