"""Policy-output bridge between discrete RL predictions and Hedge controls."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import DEFAULT_ACTION_CATALOG, HedgeActions, LegCommand, Urgency


@dataclass(frozen=True, slots=True)
class HedgePolicyOutput:
    action: HedgeActions
    long_delta: float
    short_delta: float
    close_long: bool
    close_short: bool
    urgency: Urgency


def _signed_delta(command: LegCommand, fraction: float) -> float:
    if command in {LegCommand.OPEN, LegCommand.INCREASE}:
        return fraction
    if command is LegCommand.REDUCE:
        return -fraction
    if command is LegCommand.CLOSE:
        return -1.0
    return 0.0


def decode_policy_action(action: int) -> HedgePolicyOutput:
    spec = DEFAULT_ACTION_CATALOG.decode(action)
    return HedgePolicyOutput(
        action=spec.action,
        long_delta=_signed_delta(spec.long_command, spec.long_fraction),
        short_delta=_signed_delta(spec.short_command, spec.short_fraction),
        close_long=spec.long_command is LegCommand.CLOSE,
        close_short=spec.short_command is LegCommand.CLOSE,
        urgency=spec.urgency,
    )
