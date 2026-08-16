"""Continuous reconciliation policy between local projection and exchange truth."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Iterable

from .contracts import Severity


class DiffKind(StrEnum):
    POSITION = "POSITION"
    OPEN_ORDER = "OPEN_ORDER"
    BALANCE = "BALANCE"
    MODE = "MODE"
    LEVERAGE = "LEVERAGE"
    UNKNOWN_ORDER = "UNKNOWN_ORDER"
    CURSOR = "CURSOR"


@dataclass(frozen=True, slots=True)
class PositionTruth:
    symbol: str
    side: str
    quantity: Decimal

    def __post_init__(self) -> None:
        symbol = self.symbol.strip().upper()
        side = self.side.strip().upper()
        quantity = Decimal(self.quantity)
        if not symbol:
            raise ValueError("position symbol is required")
        if side not in {"LONG", "SHORT"}:
            raise ValueError("position side must be LONG or SHORT")
        if not quantity.is_finite() or quantity < 0:
            raise ValueError("position quantity must be finite and nonnegative")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "quantity", quantity)


@dataclass(frozen=True, slots=True)
class ReconciliationPlane:
    positions: tuple[PositionTruth, ...]
    open_order_ids: frozenset[str]
    wallet_balance: Decimal
    hedge_mode: bool
    cross_symbols: frozenset[str]
    cursor: int

    @classmethod
    def build(
        cls,
        *,
        positions: Iterable[PositionTruth],
        open_order_ids: Iterable[str],
        wallet_balance: Decimal,
        hedge_mode: bool,
        cross_symbols: Iterable[str],
        cursor: int,
    ) -> "ReconciliationPlane":
        if cursor < 0:
            raise ValueError("cursor must be nonnegative")
        normalized_positions = tuple(sorted(positions, key=lambda p: (p.symbol, p.side)))
        position_keys = [(p.symbol, p.side) for p in normalized_positions]
        if len(position_keys) != len(set(position_keys)):
            raise ValueError("duplicate position truth for symbol/side")
        normalized_orders = tuple(str(x).strip() for x in open_order_ids)
        if any(not x for x in normalized_orders):
            raise ValueError("open order ids must be non-empty")
        wallet = Decimal(wallet_balance)
        if not wallet.is_finite() or wallet < 0:
            raise ValueError("wallet_balance must be finite and nonnegative")
        normalized_cross = tuple(str(x).strip().upper() for x in cross_symbols)
        if any(not x for x in normalized_cross):
            raise ValueError("cross symbols must be non-empty")
        return cls(
            normalized_positions,
            frozenset(normalized_orders),
            wallet,
            bool(hedge_mode),
            frozenset(normalized_cross),
            int(cursor),
        )


@dataclass(frozen=True, slots=True)
class ReconciliationDiff:
    kind: DiffKind
    key: str
    local: str
    exchange: str
    severity: Severity


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    converged: bool
    allow_new_risk: bool
    allow_reduce: bool
    diffs: tuple[ReconciliationDiff, ...]


def reconcile(
    local: ReconciliationPlane,
    exchange: ReconciliationPlane,
    *,
    balance_tolerance: Decimal = Decimal("0.00000001"),
    quantity_tolerance: Decimal = Decimal("0.00000001"),
) -> ReconciliationResult:
    diffs: list[ReconciliationDiff] = []
    lp = {(p.symbol, p.side): p.quantity for p in local.positions}
    ep = {(p.symbol, p.side): p.quantity for p in exchange.positions}
    for key in sorted(set(lp) | set(ep)):
        lv, ev = lp.get(key, Decimal("0")), ep.get(key, Decimal("0"))
        if abs(lv - ev) > quantity_tolerance:
            diffs.append(ReconciliationDiff(DiffKind.POSITION, ":".join(key), str(lv), str(ev), Severity.HALT_NEW_RISK))
    local_only = local.open_order_ids - exchange.open_order_ids
    exchange_only = exchange.open_order_ids - local.open_order_ids
    for order_id in sorted(local_only):
        diffs.append(ReconciliationDiff(DiffKind.OPEN_ORDER, order_id, "OPEN", "MISSING", Severity.HALT_NEW_RISK))
    for order_id in sorted(exchange_only):
        diffs.append(ReconciliationDiff(DiffKind.UNKNOWN_ORDER, order_id, "MISSING", "OPEN", Severity.HALT_ACCOUNT))
    if abs(local.wallet_balance - exchange.wallet_balance) > balance_tolerance:
        diffs.append(ReconciliationDiff(DiffKind.BALANCE, "wallet", str(local.wallet_balance), str(exchange.wallet_balance), Severity.HALT_NEW_RISK))
    if local.hedge_mode != exchange.hedge_mode:
        diffs.append(ReconciliationDiff(DiffKind.MODE, "hedge_mode", str(local.hedge_mode), str(exchange.hedge_mode), Severity.HALT_ACCOUNT))
    if local.cross_symbols != exchange.cross_symbols:
        diffs.append(ReconciliationDiff(DiffKind.MODE, "cross_symbols", repr(sorted(local.cross_symbols)), repr(sorted(exchange.cross_symbols)), Severity.HALT_ACCOUNT))
    if local.cursor > exchange.cursor:
        diffs.append(ReconciliationDiff(DiffKind.CURSOR, "event_cursor", str(local.cursor), str(exchange.cursor), Severity.HALT_ACCOUNT))
    elif local.cursor < exchange.cursor:
        diffs.append(ReconciliationDiff(DiffKind.CURSOR, "event_cursor", str(local.cursor), str(exchange.cursor), Severity.HALT_NEW_RISK))
    halt_account = any(item.severity is Severity.HALT_ACCOUNT for item in diffs)
    halt_new = halt_account or any(item.severity is Severity.HALT_NEW_RISK for item in diffs)
    return ReconciliationResult(
        converged=not diffs,
        allow_new_risk=not halt_new,
        allow_reduce=not halt_account,
        diffs=tuple(diffs),
    )
