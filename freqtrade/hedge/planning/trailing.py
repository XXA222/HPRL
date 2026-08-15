from __future__ import annotations

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

from .context import (
    PlanningContext,
    PositionSide,
    StrategyLegState,
    TrailingPhase,
)


def _idle(state: StrategyLegState) -> StrategyLegState:
    return replace(
        state,
        trailing_phase=TrailingPhase.IDLE,
        trailing_trigger_price=None,
        trailing_extreme=None,
        trailing_started_at=None,
        trailing_confirmed_at=None,
        trailing_cooldown_until=None,
        trailing_armed=False,
    )


def enter_trailing_cooldown(
    state: StrategyLegState,
    *,
    timestamp,
    cooldown_seconds: int,
) -> StrategyLegState:
    return replace(
        state,
        trailing_phase=TrailingPhase.COOLDOWN,
        trailing_trigger_price=None,
        trailing_extreme=None,
        trailing_started_at=None,
        trailing_confirmed_at=None,
        trailing_cooldown_until=timestamp + timedelta(seconds=cooldown_seconds),
        trailing_armed=False,
    )


def update_trailing_state(context: PlanningContext, side: PositionSide) -> StrategyLegState:
    """Advance the persisted IDLE/ARMED/CONFIRMED/COOLDOWN state machine."""
    state = context.state(side)
    now = context.market.timestamp
    price = context.market.mark
    cfg = context.config
    leg = context.wallet.leg(side)

    if state.trailing_phase is TrailingPhase.COOLDOWN:
        if state.trailing_cooldown_until is not None and now < state.trailing_cooldown_until:
            return state
        state = _idle(state)

    if state.trailing_phase is TrailingPhase.CONFIRMED:
        return state

    if state.trailing_phase is TrailingPhase.ARMED:
        if (
            state.trailing_started_at is not None
            and cfg.trailing_timeout_seconds > 0
            and now - state.trailing_started_at
            >= timedelta(seconds=cfg.trailing_timeout_seconds)
        ):
            return _idle(state)
        extreme = state.trailing_extreme or price
        if side is PositionSide.LONG:
            extreme = min(extreme, price)
            confirmed = price >= extreme * (Decimal("1") + cfg.trailing_rebound)
        else:
            extreme = max(extreme, price)
            confirmed = price <= extreme * (Decimal("1") - cfg.trailing_rebound)
        if confirmed:
            return replace(
                state,
                trailing_phase=TrailingPhase.CONFIRMED,
                trailing_extreme=extreme,
                trailing_confirmed_at=now,
                trailing_armed=True,
            )
        return replace(state, trailing_extreme=extreme, trailing_armed=False)

    if leg.quantity <= 0:
        return _idle(state)

    anchor = leg.average_price
    trigger = anchor * (
        Decimal("1") - cfg.trailing_trigger_distance
        if side is PositionSide.LONG
        else Decimal("1") + cfg.trailing_trigger_distance
    )
    crossed = price <= trigger if side is PositionSide.LONG else price >= trigger
    if crossed:
        return replace(
            state,
            trailing_phase=TrailingPhase.ARMED,
            trailing_trigger_price=trigger,
            trailing_extreme=price,
            trailing_started_at=now,
            trailing_confirmed_at=None,
            trailing_cooldown_until=None,
            trailing_armed=False,
        )
    return replace(
        state,
        trailing_phase=TrailingPhase.IDLE,
        trailing_trigger_price=trigger,
        trailing_extreme=None,
        trailing_started_at=None,
        trailing_confirmed_at=None,
        trailing_cooldown_until=None,
        trailing_armed=False,
    )


def trailing_confirmed(context: PlanningContext, side: PositionSide) -> bool:
    return update_trailing_state(context, side).trailing_phase is TrailingPhase.CONFIRMED
