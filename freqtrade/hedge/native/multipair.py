"""Account-level multi-pair Paper orchestration.

Each pair retains an independent planner/ledger identity while this coordinator owns
universe assignment, aggregate equity/gross limits and deterministic cycle ordering.
It is suitable for portfolio backtests and Paper; real Binance writes remain behind the
existing production composition and promotion gates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from threading import RLock
from typing import Any, Callable, Iterable, Mapping

from .models import ZERO, finite_decimal
from .universe import HedgeUniverseManager, SymbolAssignmentState


@dataclass(frozen=True, slots=True)
class MultiPairAccountSnapshot:
    equity: Decimal
    available_balance: Decimal
    gross_notional: Decimal
    net_notional: Decimal
    pair_equity: Mapping[str, Decimal]
    pair_gross: Mapping[str, Decimal]
    active_pairs: tuple[str, ...]
    draining_pairs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MultiPairCycleResult:
    results: Mapping[str, Any]
    errors: Mapping[str, str]
    snapshot: MultiPairAccountSnapshot


class MultiPairPaperHedgeRuntime:
    """Coordinate per-pair applications under one aggregate capital envelope."""

    def __init__(
        self,
        *,
        universe: HedgeUniverseManager,
        application_factory: Callable[[str, Decimal], Any],
        initial_balance: object,
        max_gross_ratio: object = Decimal("0.80"),
    ) -> None:
        self.universe = universe
        self.application_factory = application_factory
        self.initial_balance = finite_decimal(initial_balance, field_name="initial_balance")
        self.max_gross_ratio = finite_decimal(max_gross_ratio, field_name="max_gross_ratio")
        if self.initial_balance <= ZERO or self.max_gross_ratio <= ZERO:
            raise ValueError("initial balance and max gross ratio must be positive")
        self._applications: dict[str, Any] = {}
        self._lock = RLock()

    @property
    def applications(self) -> Mapping[str, Any]:
        with self._lock:
            return dict(self._applications)

    def refresh_universe(
        self,
        pairs: Iterable[str],
        *,
        blacklist: Iterable[str] = (),
    ) -> None:
        open_pairs = []
        active_order_pairs = []
        for pair, app in self.applications.items():
            wallet = app.wallet()
            if wallet.long.quantity > ZERO or wallet.short.quantity > ZERO:
                open_pairs.append(pair)
            if wallet.active_orders:
                active_order_pairs.append(pair)
        snapshot = self.universe.refresh(
            pairs,
            blacklist=blacklist,
            account_capital=self.initial_balance,
            open_position_pairs=open_pairs,
            active_order_pairs=active_order_pairs,
        )
        with self._lock:
            for assignment in snapshot.assignments:
                if assignment.state is SymbolAssignmentState.ACTIVE and assignment.pair not in self._applications:
                    app = self.application_factory(assignment.pair, assignment.capital_limit)
                    app.add_new_risk_provider(
                        lambda pair=assignment.pair: self._pair_new_risk_allowed(pair)
                    )
                    app.bind_order_admission_provider(self.universe.admit)
                    self._applications[assignment.pair] = app
            removable = [
                pair
                for pair, app in self._applications.items()
                if snapshot.assignment(pair) is None
                and app.wallet().long.quantity == ZERO
                and app.wallet().short.quantity == ZERO
                and not app.wallet().active_orders
            ]
            for pair in removable:
                del self._applications[pair]

    def _pair_new_risk_allowed(self, pair: str) -> bool:
        assignment = self.universe.snapshot.assignment(pair)
        if assignment is None or assignment.state is not SymbolAssignmentState.ACTIVE:
            return False
        snapshot = self.snapshot()
        return snapshot.gross_notional < snapshot.equity * self.max_gross_ratio

    def snapshot(self) -> MultiPairAccountSnapshot:
        pair_equity: dict[str, Decimal] = {}
        pair_gross: dict[str, Decimal] = {}
        net = ZERO
        available = ZERO
        active: list[str] = []
        draining: list[str] = []
        with self._lock:
            rows = tuple(self._applications.items())
        for pair, app in rows:
            wallet = app.wallet()
            market = app.last_market
            mark = Decimal("1") if market is None else market.mark
            pair_equity[pair] = wallet.equity
            pair_gross[pair] = wallet.gross_notional(mark)
            net += (wallet.long.quantity - wallet.short.quantity) * mark
            available += wallet.available_balance
            assignment = self.universe.snapshot.assignment(pair)
            if assignment is not None and assignment.state is SymbolAssignmentState.ACTIVE:
                active.append(pair)
            else:
                draining.append(pair)
        # Allocated pair balances partition one logical account. Unallocated capital
        # remains available and is not duplicated.
        allocated_initial = sum(
            (
                item.capital_limit
                for item in self.universe.snapshot.assignments
                if item.state is SymbolAssignmentState.ACTIVE
            ),
            ZERO,
        )
        equity = sum(pair_equity.values(), ZERO) + max(self.initial_balance - allocated_initial, ZERO)
        gross = sum(pair_gross.values(), ZERO)
        return MultiPairAccountSnapshot(
            equity=equity,
            available_balance=min(available + max(self.initial_balance - allocated_initial, ZERO), equity),
            gross_notional=gross,
            net_notional=net,
            pair_equity=pair_equity,
            pair_gross=pair_gross,
            active_pairs=tuple(sorted(active)),
            draining_pairs=tuple(sorted(draining)),
        )

    def run_cycles(self, inputs: Mapping[str, Mapping[str, Any]]) -> MultiPairCycleResult:
        results: dict[str, Any] = {}
        errors: dict[str, str] = {}
        with self._lock:
            applications = dict(self._applications)
        # Stable pair order prevents strategy ordering from changing results.
        for pair in sorted(inputs):
            app = applications.get(pair)
            if app is None:
                errors[pair] = "PAIR_NOT_ASSIGNED"
                continue
            try:
                results[pair] = app.run_market_cycle(**dict(inputs[pair]))
            except Exception as exc:
                errors[pair] = f"{type(exc).__name__}:{exc}"
        return MultiPairCycleResult(results, errors, self.snapshot())
