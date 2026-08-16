"""Auditable production control-plane state machine."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum


class ControlMode(StrEnum):
    RUN = "RUN"
    PAUSE_NEW_RISK = "PAUSE_NEW_RISK"
    CLOSE_ONLY = "CLOSE_ONLY"
    HALT = "HALT"


class ControlAction(StrEnum):
    START = "START"
    PAUSE_NEW_RISK = "PAUSE_NEW_RISK"
    CLOSE_ONLY = "CLOSE_ONLY"
    HALT = "HALT"
    RESUME = "RESUME"
    CANCEL_ALL = "CANCEL_ALL"
    REDUCE_LONG = "REDUCE_LONG"
    REDUCE_SHORT = "REDUCE_SHORT"
    FLATTEN = "FLATTEN"


@dataclass(frozen=True, slots=True)
class ControlEvent:
    sequence: int
    action: ControlAction
    actor: str
    reason: str
    observed_at: datetime
    before: ControlMode
    after: ControlMode


class ProductionControlPlane:
    def __init__(self) -> None:
        self._mode = ControlMode.HALT
        self._events: list[ControlEvent] = []

    @property
    def mode(self) -> ControlMode:
        return self._mode

    @property
    def events(self) -> tuple[ControlEvent, ...]:
        return tuple(self._events)

    @property
    def allows_new_risk(self) -> bool:
        return self._mode is ControlMode.RUN

    @property
    def allows_reduce(self) -> bool:
        return self._mode in {ControlMode.RUN, ControlMode.PAUSE_NEW_RISK, ControlMode.CLOSE_ONLY}

    def apply(
        self,
        action: ControlAction,
        *,
        actor: str,
        reason: str,
        readiness_passed: bool,
        reconciliation_converged: bool,
        observed_at: datetime,
    ) -> ControlEvent:
        if observed_at.tzinfo is None:
            raise ValueError("observed_at must be timezone-aware")
        actor = actor.strip()
        reason = reason.strip()
        if not actor or not reason:
            raise ValueError("actor and reason are required")
        before = self._mode
        after = before
        if action in {ControlAction.START, ControlAction.RESUME}:
            if not readiness_passed or not reconciliation_converged:
                raise PermissionError("resume requires readiness and reconciliation convergence")
            after = ControlMode.RUN
        elif action is ControlAction.PAUSE_NEW_RISK:
            after = ControlMode.PAUSE_NEW_RISK
        elif action is ControlAction.CLOSE_ONLY:
            after = ControlMode.CLOSE_ONLY
        elif action is ControlAction.HALT:
            after = ControlMode.HALT
        elif action in {ControlAction.CANCEL_ALL, ControlAction.REDUCE_LONG, ControlAction.REDUCE_SHORT, ControlAction.FLATTEN}:
            if not self.allows_reduce and action is not ControlAction.CANCEL_ALL:
                raise PermissionError("risk-reducing action requires controlled-reduce capability")
        event = ControlEvent(len(self._events) + 1, action, actor, reason, observed_at.astimezone(UTC), before, after)
        self._events.append(event)
        self._mode = after
        return event
