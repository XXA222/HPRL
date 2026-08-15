"""Deterministic cross-wallet simulator with independent LONG and SHORT legs."""

from __future__ import annotations

from dataclasses import dataclass, replace

from .actions import HedgeActionSpec, LegCommand
from .costs import ExecutionCostModel
from .state import HedgeAccountState, HedgeLegSide, HedgeLegState


@dataclass(frozen=True, slots=True)
class PortfolioTransition:
    previous_equity: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees: float
    funding_cashflow: float
    traded_notional: float
    long_quantity_delta: float
    short_quantity_delta: float


class HedgePortfolioSimulator:
    def __init__(self, starting_balance: float, cost_model: ExecutionCostModel) -> None:
        self.cost_model = cost_model
        self.state = HedgeAccountState.initial(starting_balance)

    def reset(self, starting_balance: float | None = None) -> HedgeAccountState:
        balance = self.state.peak_equity if starting_balance is None else float(starting_balance)
        self.state = HedgeAccountState.initial(balance)
        return self.state

    @staticmethod
    def _is_buy(side: HedgeLegSide, command: LegCommand) -> bool:
        increasing = command in {LegCommand.OPEN, LegCommand.INCREASE}
        return increasing if side is HedgeLegSide.LONG else not increasing

    def _apply_leg(
        self,
        leg: HedgeLegState,
        *,
        command: LegCommand,
        fraction: float,
        reference_price: float,
        sizing_equity: float,
        urgency,
    ) -> tuple[HedgeLegState, float, float, float, float]:
        if command is LegCommand.HOLD:
            return leg, 0.0, 0.0, 0.0, 0.0
        increasing = command in {LegCommand.OPEN, LegCommand.INCREASE}
        if increasing:
            quantity = max(0.0, sizing_equity * fraction / reference_price)
        elif command is LegCommand.CLOSE:
            quantity = leg.quantity
        else:
            quantity = leg.quantity * fraction
        quantity = min(quantity, leg.quantity) if not increasing else quantity
        if quantity <= 1e-15:
            return leg, 0.0, 0.0, 0.0, 0.0

        estimate = self.cost_model.estimate(
            reference_price=reference_price,
            quantity=quantity,
            is_buy=self._is_buy(leg.side, command),
            urgency=urgency,
        )
        realized = 0.0
        if increasing:
            new_quantity = leg.quantity + quantity
            new_average = (
                leg.average_price * leg.quantity + estimate.fill_price * quantity
            ) / new_quantity
        else:
            realized = (
                (estimate.fill_price - leg.average_price)
                * quantity
                * leg.side.direction
            )
            new_quantity = max(0.0, leg.quantity - quantity)
            new_average = leg.average_price if new_quantity > 1e-15 else 0.0
        updated = replace(
            leg,
            quantity=new_quantity,
            average_price=new_average,
            realized_pnl=leg.realized_pnl + realized,
            fees_paid=leg.fees_paid + estimate.fee,
        )
        signed_delta = quantity if increasing else -quantity
        return updated, realized, estimate.fee, estimate.notional, signed_delta

    def apply_action(
        self,
        spec: HedgeActionSpec,
        *,
        reference_price: float,
        mark_price: float | None = None,
        funding_rate: float = 0.0,
    ) -> PortfolioTransition:
        previous = self.state
        sizing_equity = max(previous.equity, 0.0)
        long, long_realized, long_fee, long_turnover, long_delta = self._apply_leg(
            previous.long,
            command=spec.long_command,
            fraction=spec.long_fraction,
            reference_price=reference_price,
            sizing_equity=sizing_equity,
            urgency=spec.urgency,
        )
        short, short_realized, short_fee, short_turnover, short_delta = self._apply_leg(
            previous.short,
            command=spec.short_command,
            fraction=spec.short_fraction,
            reference_price=reference_price,
            sizing_equity=sizing_equity,
            urgency=spec.urgency,
        )
        fees = long_fee + short_fee
        realized = long_realized + short_realized
        cash = previous.cash_balance + realized - fees
        mark = float(reference_price if mark_price is None else mark_price)

        long_funding = self.cost_model.funding_cashflow(
            side=HedgeLegSide.LONG,
            notional=long.notional(mark),
            funding_rate=funding_rate,
        )
        short_funding = self.cost_model.funding_cashflow(
            side=HedgeLegSide.SHORT,
            notional=short.notional(mark),
            funding_rate=funding_rate,
        )
        funding_cashflow = long_funding + short_funding
        cash += funding_cashflow
        long = replace(long, funding_paid=long.funding_paid - long_funding)
        short = replace(short, funding_paid=short.funding_paid - short_funding)
        unrealized = long.unrealized_pnl(mark) + short.unrealized_pnl(mark)
        equity = cash + unrealized
        peak = max(previous.peak_equity, equity)
        turnover = long_turnover + short_turnover
        self.state = HedgeAccountState(
            cash_balance=cash,
            equity=equity,
            peak_equity=peak,
            long=long,
            short=short,
            step=previous.step + 1,
            turnover=previous.turnover + turnover,
        )
        return PortfolioTransition(
            previous_equity=previous.equity,
            equity=equity,
            realized_pnl=realized,
            unrealized_pnl=unrealized,
            fees=fees,
            funding_cashflow=funding_cashflow,
            traded_notional=turnover,
            long_quantity_delta=long_delta,
            short_quantity_delta=short_delta,
        )

    def mark_to_market(self, mark_price: float) -> HedgeAccountState:
        state = self.state
        unrealized = state.long.unrealized_pnl(mark_price) + state.short.unrealized_pnl(mark_price)
        equity = state.cash_balance + unrealized
        self.state = replace(state, equity=equity, peak_equity=max(state.peak_equity, equity))
        return self.state
