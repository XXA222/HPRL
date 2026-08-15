from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from decimal import Decimal
from uuid import UUID, uuid4

from freqtrade.enums.hedge import PositionAction, PositionSide
from freqtrade.hedge.errors import HedgeInvariantError
from freqtrade.hedge.numeric import require_positive
from freqtrade.hedge.symbols import canonicalize_symbol


@dataclass(frozen=True, slots=True, order=True)
class PositionKey:
    exchange: str
    symbol: str
    position_side: PositionSide
    account_id: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.exchange, str) or not self.exchange.strip():
            raise HedgeInvariantError("exchange must not be empty.")
        if not isinstance(self.account_id, str) or not self.account_id.strip():
            raise HedgeInvariantError("account_id must not be empty.")
        if not isinstance(self.position_side, PositionSide):
            object.__setattr__(self, "position_side", PositionSide(self.position_side))
        if self.position_side is PositionSide.BOTH:
            raise HedgeInvariantError("PositionKey must identify LONG or SHORT.")
        object.__setattr__(self, "exchange", self.exchange.strip().lower())
        object.__setattr__(self, "account_id", self.account_id.strip())
        object.__setattr__(self, "symbol", canonicalize_symbol(self.symbol))

    @property
    def identity(self) -> tuple[str, str, str, PositionSide]:
        return (
            self.exchange,
            self.account_id,
            self.symbol,
            self.position_side,
        )


@dataclass(frozen=True, slots=True)
class HedgeAction:
    symbol: str
    position_side: PositionSide
    position_action: PositionAction
    quantity: Decimal
    action_group_id: UUID
    account_id: str = "default"
    strategy_id: str = "default"

    def __init__(
        self,
        symbol: str,
        position_side: PositionSide | str,
        position_action: PositionAction | str,
        quantity: Decimal | str | float | int,
        action_group_id: UUID | None = None,
        account_id: str = "default",
        strategy_id: str = "default",
    ) -> None:
        side = (
            position_side
            if isinstance(position_side, PositionSide)
            else PositionSide(position_side)
        )
        action = (
            position_action
            if isinstance(position_action, PositionAction)
            else PositionAction(position_action)
        )
        if side is PositionSide.BOTH:
            raise HedgeInvariantError("Hedge actions must never BOTH; target LONG or SHORT.")
        if not account_id.strip():
            raise HedgeInvariantError("account_id must not be empty.")
        if not strategy_id.strip():
            raise HedgeInvariantError("strategy_id must not be empty.")
        object.__setattr__(self, "symbol", canonicalize_symbol(symbol))
        object.__setattr__(self, "position_side", side)
        object.__setattr__(self, "position_action", action)
        object.__setattr__(self, "quantity", require_positive(quantity, field="quantity"))
        object.__setattr__(self, "action_group_id", action_group_id or uuid4())
        object.__setattr__(self, "account_id", account_id.strip())
        object.__setattr__(self, "strategy_id", strategy_id.strip())

    @property
    def increases_risk(self) -> bool:
        return self.position_action.increases_risk

    @property
    def reduces_risk(self) -> bool:
        return self.position_action.reduces_risk


@dataclass(frozen=True, slots=True)
class HedgeActionPlan:
    actions: tuple[HedgeAction, ...]
    action_group_id: UUID

    def __init__(
        self,
        actions: Iterable[HedgeAction],
        action_group_id: UUID | None = None,
    ) -> None:
        source_actions = tuple(actions)
        if not source_actions:
            raise HedgeInvariantError("A hedge action plan must contain an action.")
        if len({action.symbol for action in source_actions}) != 1:
            raise HedgeInvariantError("MVP plans may manage only one symbol.")
        if len({action.account_id for action in source_actions}) != 1:
            raise HedgeInvariantError("A plan may manage only one account.")
        group_id = action_group_id or uuid4()
        normalized_actions = tuple(
            replace(action, action_group_id=group_id) for action in source_actions
        )
        object.__setattr__(self, "actions", normalized_actions)
        object.__setattr__(self, "action_group_id", group_id)

    @property
    def reduce_risk_actions(self) -> tuple[HedgeAction, ...]:
        return tuple(action for action in self.actions if action.reduces_risk)

    @property
    def increase_risk_actions(self) -> tuple[HedgeAction, ...]:
        return tuple(action for action in self.actions if action.increases_risk)

    @property
    def ordered_actions(self) -> tuple[HedgeAction, ...]:
        return self.reduce_risk_actions + self.increase_risk_actions
