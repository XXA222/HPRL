"""Idempotent dual-leg accounting and invariant audits (rounds 51-60)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from .state import HedgeAccountState, HedgeLegSide


def _positive_finite(name: str, value: float, *, allow_zero: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0 or (not allow_zero and result == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


# Round 51 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FillRecord:
    fill_id: str
    order_id: str
    side: HedgeLegSide
    increasing: bool
    quantity: float
    price: float
    fee: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.fill_id.strip() or not self.order_id.strip():
            raise ValueError("fill_id and order_id cannot be empty")
        _positive_finite("quantity", self.quantity)
        _positive_finite("price", self.price)
        _positive_finite("fee", self.fee, allow_zero=True)
        if self.timestamp.tzinfo is None:
            raise ValueError("fill timestamp must be timezone-aware")


class IdempotentFillLedger:
    def __init__(self) -> None:
        self._fills: dict[str, FillRecord] = {}

    def record(self, fill: FillRecord) -> bool:
        existing = self._fills.get(fill.fill_id)
        if existing is None:
            self._fills[fill.fill_id] = fill
            return True
        if existing != fill:
            raise ValueError("duplicate fill_id carries conflicting data")
        return False

    def fills(self) -> tuple[FillRecord, ...]:
        return tuple(sorted(self._fills.values(), key=lambda item: (item.timestamp, item.fill_id)))


# Round 52 -------------------------------------------------------------------------------
def validate_trade_price_quantity(
    *,
    price: float,
    quantity: float,
    min_price: float = 0.0,
    min_quantity: float = 0.0,
    tick_size: float | None = None,
    step_size: float | None = None,
    tolerance: float = 1e-9,
) -> None:
    price = _positive_finite("price", price)
    quantity = _positive_finite("quantity", quantity)
    if price + tolerance < min_price or quantity + tolerance < min_quantity:
        raise ValueError("trade is below exchange minimum")
    for label, value, increment in (
        ("price", price, tick_size),
        ("quantity", quantity, step_size),
    ):
        if increment is None:
            continue
        increment = _positive_finite(f"{label} increment", increment)
        units = value / increment
        if not math.isclose(units, round(units), rel_tol=0, abs_tol=tolerance):
            raise ValueError(f"{label} is not aligned to its exchange increment")


# Round 53 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PositionAccumulator:
    side: HedgeLegSide
    quantity: float = 0.0
    average_price: float = 0.0
    realized_pnl: float = 0.0

    def apply_fill(
        self,
        *,
        increasing: bool,
        quantity: float,
        price: float,
    ) -> PositionAccumulator:
        quantity = _positive_finite("quantity", quantity)
        price = _positive_finite("price", price)
        if increasing:
            new_quantity = self.quantity + quantity
            new_average = (
                self.average_price * self.quantity + price * quantity
            ) / new_quantity
            return replace(self, quantity=new_quantity, average_price=new_average)
        if quantity > self.quantity + 1e-12:
            raise ValueError("reducing fill exceeds the open position")
        realized = realized_pnl(
            side=self.side,
            entry_price=self.average_price,
            exit_price=price,
            quantity=quantity,
        )
        remaining = max(0.0, self.quantity - quantity)
        return replace(
            self,
            quantity=remaining,
            average_price=self.average_price if remaining > 1e-12 else 0.0,
            realized_pnl=self.realized_pnl + realized,
        )


# Rounds 54 and 55 ------------------------------------------------------------------------
def realized_pnl(
    *,
    side: HedgeLegSide,
    entry_price: float,
    exit_price: float,
    quantity: float,
) -> float:
    entry = _positive_finite("entry_price", entry_price)
    exit_value = _positive_finite("exit_price", exit_price)
    qty = _positive_finite("quantity", quantity, allow_zero=True)
    return (exit_value - entry) * qty * side.direction


# Round 56 -------------------------------------------------------------------------------
@dataclass(slots=True)
class FeeLedger:
    total: float = 0.0
    by_order: dict[str, float] = field(default_factory=dict)

    def post(self, order_id: str, fee: float) -> None:
        if not order_id.strip():
            raise ValueError("order_id cannot be empty")
        value = _positive_finite("fee", fee, allow_zero=True)
        self.by_order[order_id] = self.by_order.get(order_id, 0.0) + value
        self.total += value

    def reconcile(self) -> float:
        difference = self.total - sum(self.by_order.values())
        if abs(difference) > 1e-9:
            raise RuntimeError("fee ledger is internally inconsistent")
        return difference


# Round 57 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FundingPosting:
    event_id: str
    side: HedgeLegSide
    notional: float
    rate: float

    @property
    def cashflow(self) -> float:
        return -self.side.direction * self.notional * self.rate


class FundingLedger:
    def __init__(self) -> None:
        self._postings: dict[str, FundingPosting] = {}

    def post(self, posting: FundingPosting) -> bool:
        if not posting.event_id.strip():
            raise ValueError("funding event_id cannot be empty")
        _positive_finite("notional", posting.notional, allow_zero=True)
        if not math.isfinite(posting.rate):
            raise ValueError("funding rate must be finite")
        existing = self._postings.get(posting.event_id)
        if existing is None:
            self._postings[posting.event_id] = posting
            return True
        if existing != posting:
            raise ValueError("funding event id conflict")
        return False

    @property
    def cashflow(self) -> float:
        return sum(item.cashflow for item in self._postings.values())


# Round 58 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MarkToMarketResult:
    long_unrealized: float
    short_unrealized: float
    total_unrealized: float
    equity: float


def mark_to_market_account(account: HedgeAccountState, *, mark: float) -> MarkToMarketResult:
    mark = _positive_finite("mark", mark)
    long_unrealized = account.long.unrealized_pnl(mark)
    short_unrealized = account.short.unrealized_pnl(mark)
    total = long_unrealized + short_unrealized
    return MarkToMarketResult(
        long_unrealized,
        short_unrealized,
        total,
        account.cash_balance + total,
    )


# Round 59 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LiquidationBuffer:
    maintenance_margin: float
    equity: float
    buffer_amount: float
    buffer_ratio: float


def liquidation_buffer(
    account: HedgeAccountState,
    *,
    mark: float,
    maintenance_rate: float,
) -> LiquidationBuffer:
    mark = _positive_finite("mark", mark)
    rate = float(maintenance_rate)
    if not math.isfinite(rate) or not 0 <= rate < 1:
        raise ValueError("maintenance_rate must be finite and within [0, 1)")
    maintenance = account.gross_notional(mark) * rate
    buffer_amount = account.equity - maintenance
    ratio = buffer_amount / max(abs(account.equity), 1e-12)
    return LiquidationBuffer(maintenance, account.equity, buffer_amount, ratio)


# Round 60 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class AccountInvariantReport:
    valid: bool
    violations: tuple[str, ...]
    recomputed_equity: float
    equity_difference: float


def audit_account_invariants(
    account: HedgeAccountState,
    *,
    mark: float,
    tolerance: float = 1e-8,
) -> AccountInvariantReport:
    mtm = mark_to_market_account(account, mark=mark)
    violations: list[str] = []
    difference = account.equity - mtm.equity
    if abs(difference) > tolerance:
        violations.append("EQUITY_DOES_NOT_MATCH_CASH_PLUS_UNREALIZED")
    if account.peak_equity + tolerance < account.equity:
        violations.append("PEAK_EQUITY_BELOW_EQUITY")
    if account.long.side is not HedgeLegSide.LONG or account.short.side is not HedgeLegSide.SHORT:
        violations.append("LEG_SIDE_IDENTITY_INVALID")
    if account.turnover < -tolerance:
        violations.append("NEGATIVE_TURNOVER")
    for label, leg in (("LONG", account.long), ("SHORT", account.short)):
        if leg.quantity <= tolerance and abs(leg.average_price) > tolerance:
            violations.append(f"{label}_FLAT_WITH_NONZERO_AVERAGE")
        if leg.quantity > tolerance and leg.average_price <= 0:
            violations.append(f"{label}_OPEN_WITH_INVALID_AVERAGE")
    return AccountInvariantReport(not violations, tuple(violations), mtm.equity, difference)
