"""Stable discrete action catalogue for dual-leg Hedge reinforcement learning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class LegCommand(StrEnum):
    HOLD = "HOLD"
    OPEN = "OPEN"
    INCREASE = "INCREASE"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"


class Urgency(StrEnum):
    PASSIVE = "PASSIVE"
    NORMAL = "NORMAL"
    URGENT = "URGENT"


class HedgeActions(IntEnum):
    HOLD = 0
    LONG_OPEN_SMALL = 1
    LONG_OPEN_MEDIUM = 2
    LONG_ADD_SMALL = 3
    LONG_ADD_MEDIUM = 4
    LONG_REDUCE_SMALL = 5
    LONG_REDUCE_MEDIUM = 6
    LONG_CLOSE = 7
    SHORT_OPEN_SMALL = 8
    SHORT_OPEN_MEDIUM = 9
    SHORT_ADD_SMALL = 10
    SHORT_ADD_MEDIUM = 11
    SHORT_REDUCE_SMALL = 12
    SHORT_REDUCE_MEDIUM = 13
    SHORT_CLOSE = 14
    BOTH_OPEN_SMALL = 15
    BOTH_REDUCE_SMALL = 16
    REBALANCE_TO_LONG = 17
    REBALANCE_TO_SHORT = 18
    CLOSE_BOTH = 19
    EMERGENCY_REDUCE_BOTH = 20


@dataclass(frozen=True, slots=True)
class HedgeActionSpec:
    action: HedgeActions
    long_command: LegCommand = LegCommand.HOLD
    short_command: LegCommand = LegCommand.HOLD
    long_fraction: float = 0.0
    short_fraction: float = 0.0
    urgency: Urgency = Urgency.NORMAL

    def __post_init__(self) -> None:
        for name in ("long_fraction", "short_fraction"):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.long_command is LegCommand.HOLD and self.long_fraction != 0:
            raise ValueError("HOLD long command requires zero fraction")
        if self.short_command is LegCommand.HOLD and self.short_fraction != 0:
            raise ValueError("HOLD short command requires zero fraction")
        if self.long_command is LegCommand.CLOSE and self.long_fraction != 1:
            raise ValueError("CLOSE long command requires fraction 1")
        if self.short_command is LegCommand.CLOSE and self.short_fraction != 1:
            raise ValueError("CLOSE short command requires fraction 1")


_DEFAULT_ACTIONS = (
    HedgeActionSpec(HedgeActions.HOLD),
    HedgeActionSpec(HedgeActions.LONG_OPEN_SMALL, LegCommand.OPEN, long_fraction=0.10),
    HedgeActionSpec(HedgeActions.LONG_OPEN_MEDIUM, LegCommand.OPEN, long_fraction=0.25),
    HedgeActionSpec(HedgeActions.LONG_ADD_SMALL, LegCommand.INCREASE, long_fraction=0.10),
    HedgeActionSpec(HedgeActions.LONG_ADD_MEDIUM, LegCommand.INCREASE, long_fraction=0.25),
    HedgeActionSpec(HedgeActions.LONG_REDUCE_SMALL, LegCommand.REDUCE, long_fraction=0.25),
    HedgeActionSpec(HedgeActions.LONG_REDUCE_MEDIUM, LegCommand.REDUCE, long_fraction=0.50),
    HedgeActionSpec(HedgeActions.LONG_CLOSE, LegCommand.CLOSE, long_fraction=1.0),
    HedgeActionSpec(
        HedgeActions.SHORT_OPEN_SMALL,
        short_command=LegCommand.OPEN,
        short_fraction=0.10,
    ),
    HedgeActionSpec(
        HedgeActions.SHORT_OPEN_MEDIUM,
        short_command=LegCommand.OPEN,
        short_fraction=0.25,
    ),
    HedgeActionSpec(
        HedgeActions.SHORT_ADD_SMALL,
        short_command=LegCommand.INCREASE,
        short_fraction=0.10,
    ),
    HedgeActionSpec(
        HedgeActions.SHORT_ADD_MEDIUM,
        short_command=LegCommand.INCREASE,
        short_fraction=0.25,
    ),
    HedgeActionSpec(
        HedgeActions.SHORT_REDUCE_SMALL,
        short_command=LegCommand.REDUCE,
        short_fraction=0.25,
    ),
    HedgeActionSpec(
        HedgeActions.SHORT_REDUCE_MEDIUM,
        short_command=LegCommand.REDUCE,
        short_fraction=0.50,
    ),
    HedgeActionSpec(HedgeActions.SHORT_CLOSE, short_command=LegCommand.CLOSE, short_fraction=1.0),
    HedgeActionSpec(
        HedgeActions.BOTH_OPEN_SMALL,
        LegCommand.OPEN,
        LegCommand.OPEN,
        0.10,
        0.10,
        Urgency.PASSIVE,
    ),
    HedgeActionSpec(
        HedgeActions.BOTH_REDUCE_SMALL,
        LegCommand.REDUCE,
        LegCommand.REDUCE,
        0.25,
        0.25,
    ),
    HedgeActionSpec(
        HedgeActions.REBALANCE_TO_LONG,
        LegCommand.INCREASE,
        LegCommand.REDUCE,
        0.10,
        0.25,
    ),
    HedgeActionSpec(
        HedgeActions.REBALANCE_TO_SHORT,
        LegCommand.REDUCE,
        LegCommand.INCREASE,
        0.25,
        0.10,
    ),
    HedgeActionSpec(
        HedgeActions.CLOSE_BOTH,
        LegCommand.CLOSE,
        LegCommand.CLOSE,
        1.0,
        1.0,
        Urgency.URGENT,
    ),
    HedgeActionSpec(
        HedgeActions.EMERGENCY_REDUCE_BOTH,
        LegCommand.REDUCE,
        LegCommand.REDUCE,
        0.50,
        0.50,
        Urgency.URGENT,
    ),
)


class HedgeActionCatalog:
    """Immutable action-id decoder with a stable serialization contract."""

    def __init__(self, actions: tuple[HedgeActionSpec, ...] = _DEFAULT_ACTIONS) -> None:
        self._actions = actions
        expected = list(range(len(HedgeActions)))
        actual = sorted(int(spec.action) for spec in actions)
        if actual != expected:
            raise ValueError("action catalogue must contain every HedgeActions id exactly once")
        self._by_id = {int(spec.action): spec for spec in actions}

    def __len__(self) -> int:
        return len(self._actions)

    def decode(self, action: int | HedgeActions) -> HedgeActionSpec:
        try:
            return self._by_id[int(action)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"unknown Hedge action id: {action!r}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(spec.action.name for spec in self._actions)

    def specs(self) -> tuple[HedgeActionSpec, ...]:
        return self._actions


DEFAULT_ACTION_CATALOG = HedgeActionCatalog()
