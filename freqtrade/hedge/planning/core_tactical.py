from __future__ import annotations

from dataclasses import replace
from datetime import timedelta

from .close_grid import build_close_grid
from .context import (
    OrderIntent,
    PlanningContext,
    PositionSide,
    StrategyLegState,
    TrailingPhase,
)
from .grid import build_entry_grid
from .target import TargetPosition, calculate_target
from .trailing import update_trailing_state
from .unstuck import build_unstuck_intent


def _with_state(
    context: PlanningContext,
    side: PositionSide,
    state: StrategyLegState,
) -> PlanningContext:
    return replace(
        context,
        long_state=state if side is PositionSide.LONG else context.long_state,
        short_state=state if side is PositionSide.SHORT else context.short_state,
    )


def _cooldown_elapsed(
    context: PlanningContext,
    side: PositionSide,
    state: StrategyLegState,
) -> bool:
    if state.trailing_phase is TrailingPhase.COOLDOWN:
        return False
    last = state.last_entry_at
    return last is None or context.market.timestamp - last >= timedelta(
        seconds=context.config.cooldown_seconds
    )


def plan_leg(
    context: PlanningContext,
    side: PositionSide,
) -> tuple[tuple[OrderIntent, ...], StrategyLegState, TargetPosition, tuple[str, ...]]:
    target = calculate_target(context, side)
    state = update_trailing_state(context, side)
    effective = _with_state(context, side, state)
    diagnostics: list[str] = [
        f"{side.value}:target_gap:{target.quantity_gap}",
        f"{side.value}:net_gap:{target.net_gap_quantity}",
    ]

    if not context.config.enabled(side):
        return (), state, target, tuple(diagnostics + [f"{side.value}:disabled"])

    unstuck = build_unstuck_intent(effective, side, target)
    if unstuck is not None:
        return (unstuck,), replace(state, sequence=state.sequence + 1), target, tuple(
            diagnostics + [f"{side.value}:unstuck"]
        )

    orders = list(build_close_grid(effective, side, target))
    has_active_entry = any(
        active.symbol == context.market.symbol
        and active.position_side is side
        and not active.reduce_only
        for active in context.wallet.active_orders
    )
    if not _cooldown_elapsed(context, side, state):
        diagnostics.append(f"{side.value}:entry_cooldown")
        if has_active_entry:
            orders.extend(
                build_entry_grid(
                    effective,
                    side,
                    target,
                    allow_new_entries=False,
                )
            )
    elif state.trailing_phase is TrailingPhase.CONFIRMED or context.wallet.leg(side).quantity == 0:
        orders.extend(build_entry_grid(effective, side, target))
    else:
        diagnostics.append(f"{side.value}:trailing_{state.trailing_phase.value.lower()}")
        if has_active_entry:
            orders.extend(
                build_entry_grid(
                    effective,
                    side,
                    target,
                    allow_new_entries=False,
                )
            )

    state = replace(state, sequence=state.sequence + 1)
    return tuple(orders), state, target, tuple(diagnostics)
