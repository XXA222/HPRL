"""Deterministic startup/crash recovery matrix.

The invariant is simple: an ambiguous submit is queried/reconciled, never blindly resent.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class CrashPoint(StrEnum):
    BEFORE_INTENT_COMMIT = "BEFORE_INTENT_COMMIT"
    AFTER_INTENT_COMMIT = "AFTER_INTENT_COMMIT"
    BEFORE_SUBMIT = "BEFORE_SUBMIT"
    AFTER_SUBMIT_BEFORE_ACK = "AFTER_SUBMIT_BEFORE_ACK"
    AFTER_ACK_BEFORE_PERSIST = "AFTER_ACK_BEFORE_PERSIST"
    AFTER_PARTIAL_FILL = "AFTER_PARTIAL_FILL"
    DURING_CANCEL = "DURING_CANCEL"
    AFTER_CANCEL_BEFORE_PERSIST = "AFTER_CANCEL_BEFORE_PERSIST"
    DURING_RECONCILIATION = "DURING_RECONCILIATION"
    AFTER_DB_COMMIT = "AFTER_DB_COMMIT"
    AFTER_WS_EVENT_BEFORE_REST = "AFTER_WS_EVENT_BEFORE_REST"
    STALE_OR_CORRUPT_CHECKPOINT = "STALE_OR_CORRUPT_CHECKPOINT"


class RecoveryAction(StrEnum):
    DISCARD_UNCOMMITTED = "DISCARD_UNCOMMITTED"
    LOAD_DURABLE_FACTS = "LOAD_DURABLE_FACTS"
    QUERY_BY_CLIENT_ORDER_ID = "QUERY_BY_CLIENT_ORDER_ID"
    REFRESH_ORDER = "REFRESH_ORDER"
    RECONCILE_ACCOUNT = "RECONCILE_ACCOUNT"
    REPLAY_FILLS = "REPLAY_FILLS"
    RESUME_CANCEL = "RESUME_CANCEL"
    INVALIDATE_CHECKPOINT = "INVALIDATE_CHECKPOINT"
    HALT_NEW_RISK = "HALT_NEW_RISK"
    RELEASE_IF_CONVERGED = "RELEASE_IF_CONVERGED"


@dataclass(frozen=True, slots=True)
class RecoveryContext:
    crash_point: CrashPoint
    intent_committed: bool
    exchange_submit_maybe_sent: bool
    ack_persisted: bool
    unknown_order_present: bool
    checkpoint_valid: bool
    durable_facts_available: bool


@dataclass(frozen=True, slots=True)
class RecoveryPlan:
    actions: tuple[RecoveryAction, ...]
    blind_resubmit_allowed: bool
    new_risk_allowed_before_convergence: bool


def build_recovery_plan(ctx: RecoveryContext) -> RecoveryPlan:
    actions: list[RecoveryAction] = [RecoveryAction.HALT_NEW_RISK]
    if not ctx.intent_committed and not ctx.exchange_submit_maybe_sent:
        actions.append(RecoveryAction.DISCARD_UNCOMMITTED)
    else:
        if ctx.durable_facts_available:
            actions.append(RecoveryAction.LOAD_DURABLE_FACTS)
        if ctx.exchange_submit_maybe_sent and not ctx.ack_persisted:
            actions.append(RecoveryAction.QUERY_BY_CLIENT_ORDER_ID)
        if ctx.crash_point in {CrashPoint.AFTER_PARTIAL_FILL, CrashPoint.AFTER_DB_COMMIT}:
            actions.append(RecoveryAction.REPLAY_FILLS)
        if ctx.crash_point in {CrashPoint.DURING_CANCEL, CrashPoint.AFTER_CANCEL_BEFORE_PERSIST}:
            actions.append(RecoveryAction.RESUME_CANCEL)
        actions.append(RecoveryAction.RECONCILE_ACCOUNT)
    if not ctx.checkpoint_valid:
        actions.append(RecoveryAction.INVALIDATE_CHECKPOINT)
        if RecoveryAction.LOAD_DURABLE_FACTS not in actions and ctx.durable_facts_available:
            actions.append(RecoveryAction.LOAD_DURABLE_FACTS)
    if ctx.unknown_order_present and RecoveryAction.QUERY_BY_CLIENT_ORDER_ID not in actions:
        actions.append(RecoveryAction.QUERY_BY_CLIENT_ORDER_ID)
    if RecoveryAction.RECONCILE_ACCOUNT not in actions:
        actions.append(RecoveryAction.RECONCILE_ACCOUNT)
    actions.append(RecoveryAction.RELEASE_IF_CONVERGED)
    return RecoveryPlan(tuple(dict.fromkeys(actions)), False, False)


def recovery_matrix() -> tuple[RecoveryPlan, ...]:
    return tuple(
        build_recovery_plan(
            RecoveryContext(
                crash_point=point,
                intent_committed=point is not CrashPoint.BEFORE_INTENT_COMMIT,
                exchange_submit_maybe_sent=point not in {
                    CrashPoint.BEFORE_INTENT_COMMIT,
                    CrashPoint.AFTER_INTENT_COMMIT,
                    CrashPoint.BEFORE_SUBMIT,
                },
                ack_persisted=point not in {
                    CrashPoint.BEFORE_INTENT_COMMIT,
                    CrashPoint.AFTER_INTENT_COMMIT,
                    CrashPoint.BEFORE_SUBMIT,
                    CrashPoint.AFTER_SUBMIT_BEFORE_ACK,
                },
                unknown_order_present=point is CrashPoint.AFTER_SUBMIT_BEFORE_ACK,
                checkpoint_valid=point is not CrashPoint.STALE_OR_CORRUPT_CHECKPOINT,
                durable_facts_available=True,
            )
        )
        for point in CrashPoint
    )
