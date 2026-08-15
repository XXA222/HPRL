"""Read-only comparison of exchange, wallet and database position snapshots."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from freqtrade.enums import PositionSide
from freqtrade.hedge.domain import PositionKey
from freqtrade.hedge.numeric import require_nonnegative
from freqtrade.hedge.position_book import PositionRecord, SideAwarePositionBook
from freqtrade.hedge.symbols import canonicalize_symbol


@dataclass(frozen=True, slots=True)
class ReconciliationIssue:
    code: str
    key: PositionKey | tuple[str, str]
    local_amount: Decimal | None
    remote_amount: Decimal | None
    detail: str


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    consistent: bool
    issues: tuple[ReconciliationIssue, ...]
    local: SideAwarePositionBook
    remote: SideAwarePositionBook
    amount_tolerance: Decimal


def _value(source: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(source, Mapping) and name in source:
            return source[name]
        if hasattr(source, name):
            return getattr(source, name)
    return default


def records_from_trades(
    trades: Iterable[Any],
    *,
    exchange: str = "unknown",
    account_id: str = "default",
    managed_pair: str | None = None,
) -> tuple[PositionRecord, ...]:
    records: list[PositionRecord] = []
    for trade in trades:
        if not _value(trade, "is_open", default=True):
            continue
        side = _value(trade, "position_side")
        if side in (None, PositionSide.BOTH.value):
            side = (
                PositionSide.SHORT
                if _value(trade, "is_short", default=False)
                else PositionSide.LONG
            )
        records.append(
            PositionRecord(
                exchange=exchange,
                account_id=_value(trade, "account_id", default=account_id),
                symbol=canonicalize_symbol(
                    _value(trade, "pair", "symbol"),
                    managed_pair=managed_pair,
                ),
                position_side=side,
                amount=_value(trade, "amount", default=0),
                entry_price=_value(trade, "open_rate", "entry_price", default=0),
                leverage=_value(trade, "leverage", default=1),
                collateral=_value(trade, "stake_amount", "collateral", default=0),
                source="database",
                version=_value(trade, "hedge_version", "record_version", default=0),
            )
        )
    return tuple(records)


def records_from_wallet_positions(
    positions: Mapping[Any, Any] | Iterable[Any],
    *,
    exchange: str = "unknown",
    account_id: str = "default",
    managed_pair: str | None = None,
) -> tuple[PositionRecord, ...]:
    values = positions.values() if isinstance(positions, Mapping) else positions
    records: list[PositionRecord] = []
    for position in values:
        raw_amount = Decimal(str(_value(position, "position", "amount", default=0)))
        side = _value(position, "position_side", "side")
        records.append(
            PositionRecord(
                exchange=exchange,
                account_id=account_id,
                symbol=canonicalize_symbol(
                    _value(position, "symbol", "pair"),
                    managed_pair=managed_pair,
                ),
                position_side=str(getattr(side, "value", side)).upper(),
                amount=abs(raw_amount),
                entry_price=_value(position, "entry_price", "open_rate", default=0),
                leverage=_value(position, "leverage", default=1),
                collateral=_value(position, "collateral", "stake_amount", default=0),
                source="exchange",
            )
        )
    return tuple(records)


def reconcile_positions(
    local_records: Iterable[PositionRecord],
    remote_records: Iterable[PositionRecord],
    *,
    amount_tolerance: Decimal | str | float = Decimal("0.00000001"),
) -> ReconciliationResult:
    tolerance = require_nonnegative(amount_tolerance, field="amount_tolerance")
    local = SideAwarePositionBook(local_records)
    remote = SideAwarePositionBook(remote_records)
    local_map = local.as_dict()
    remote_map = remote.as_dict()
    issues: list[ReconciliationIssue] = []

    def issue_key(key: PositionKey) -> PositionKey | tuple[str, str]:
        """Preserve the original public key shape for unscoped records."""
        if key.exchange == "unknown" and key.account_id == "default":
            return (key.symbol, key.position_side.value)
        return key

    for key in sorted(set(local_map) | set(remote_map)):
        local_record = local_map.get(key)
        remote_record = remote_map.get(key)
        if local_record is None:
            issues.append(
                ReconciliationIssue(
                    "MISSING_LOCAL",
                    issue_key(key),
                    None,
                    remote_record.amount if remote_record else None,
                    "Exchange position has no matching local open trade.",
                )
            )
            continue
        if remote_record is None:
            issues.append(
                ReconciliationIssue(
                    "MISSING_REMOTE",
                    issue_key(key),
                    local_record.amount,
                    None,
                    "Local open trade has no matching exchange position.",
                )
            )
            continue
        if abs(local_record.amount - remote_record.amount) > tolerance:
            issues.append(
                ReconciliationIssue(
                    "AMOUNT_MISMATCH",
                    issue_key(key),
                    local_record.amount,
                    remote_record.amount,
                    "Position amount drift exceeds the configured tolerance.",
                )
            )
    return ReconciliationResult(
        consistent=not issues,
        issues=tuple(issues),
        local=local,
        remote=remote,
        amount_tolerance=tolerance,
    )
