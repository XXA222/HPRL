"""Account and side-aware position book."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from freqtrade.enums import PositionSide
from freqtrade.hedge.domain import PositionKey
from freqtrade.hedge.errors import HedgeDataError, HedgeInvariantError, HedgeSafetyError
from freqtrade.hedge.numeric import require_nonnegative, to_decimal
from freqtrade.hedge.symbols import canonicalize_symbol


@dataclass(frozen=True, slots=True)
class PositionRecord:
    symbol: str
    position_side: PositionSide
    amount: Decimal
    entry_price: Decimal = Decimal("0")
    mark_price: Decimal | None = None
    liquidation_price: Decimal | None = None
    leverage: Decimal = Decimal("1")
    collateral: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    margin_mode: str = "cross"
    source: str = "unknown"
    exchange: str = "unknown"
    account_id: str = "default"
    source_time_ms: int | None = None
    version: int = 0

    def __post_init__(self) -> None:
        side = (
            self.position_side
            if isinstance(self.position_side, PositionSide)
            else PositionSide(str(self.position_side).upper())
        )
        if side is PositionSide.BOTH:
            raise HedgeDataError("A nonzero position must be LONG or SHORT.")
        object.__setattr__(self, "position_side", side)
        object.__setattr__(self, "symbol", canonicalize_symbol(self.symbol))
        object.__setattr__(
            self,
            "amount",
            require_nonnegative(self.amount, field="amount"),
        )
        for name in ("entry_price", "leverage", "collateral"):
            object.__setattr__(
                self,
                name,
                require_nonnegative(getattr(self, name), field=name),
            )
        for name in ("mark_price", "liquidation_price"):
            value = to_decimal(getattr(self, name), field=name, allow_none=True)
            if value is not None and value < 0:
                raise HedgeDataError(f"{name} must be nonnegative.")
            object.__setattr__(self, name, value)
        pnl = to_decimal(self.unrealized_pnl, field="unrealized_pnl")
        if pnl is None:  # pragma: no cover - allow_none is deliberately false.
            raise HedgeDataError("unrealized_pnl is required.")
        object.__setattr__(self, "unrealized_pnl", pnl)
        if not self.exchange.strip():
            raise HedgeDataError("exchange must not be empty.")
        if not self.account_id.strip():
            raise HedgeDataError("account_id must not be empty.")
        if self.margin_mode.lower() not in {"cross", "isolated", "unknown"}:
            raise HedgeDataError("margin_mode must be cross, isolated or unknown.")
        if self.source_time_ms is not None and self.source_time_ms < 0:
            raise HedgeDataError("source_time_ms must not be negative.")
        if self.version < 0:
            raise HedgeDataError("version must not be negative.")
        object.__setattr__(self, "exchange", self.exchange.strip().lower())
        object.__setattr__(self, "account_id", self.account_id.strip())
        object.__setattr__(self, "margin_mode", self.margin_mode.lower())

    @property
    def key(self) -> PositionKey:
        return PositionKey(
            exchange=self.exchange,
            account_id=self.account_id,
            symbol=self.symbol,
            position_side=self.position_side,
        )

    @property
    def reference_price(self) -> Decimal:
        if self.mark_price is not None and self.mark_price > 0:
            return self.mark_price
        return self.entry_price

    @property
    def notional(self) -> Decimal:
        return self.amount * self.reference_price


class SideAwarePositionBook:
    def __init__(self, records: Iterable[PositionRecord] = ()) -> None:
        self._records: dict[PositionKey, PositionRecord] = {}
        for record in records:
            self.upsert(record, reject_duplicate=True)

    def upsert(
        self,
        record: PositionRecord,
        *,
        reject_duplicate: bool = False,
    ) -> None:
        current = self._records.get(record.key)
        if reject_duplicate and current is not None:
            raise HedgeInvariantError(f"Duplicate position record: {record.key}")
        if current is not None and record.version < current.version:
            raise HedgeSafetyError(
                f"Stale position version for {record.key}: {record.version} < {current.version}."
            )
        self._records[record.key] = record

    def get(
        self,
        symbol: str,
        position_side: PositionSide | str,
        *,
        account_id: str = "default",
        exchange: str = "unknown",
    ) -> PositionRecord | None:
        side = (
            position_side
            if isinstance(position_side, PositionSide)
            else PositionSide(str(position_side).upper())
        )
        return self._records.get(
            PositionKey(
                exchange=exchange,
                account_id=account_id,
                symbol=symbol,
                position_side=side,
            )
        )

    def remove(
        self,
        symbol: str,
        position_side: PositionSide | str,
        *,
        account_id: str = "default",
        exchange: str = "unknown",
    ) -> None:
        side = (
            position_side
            if isinstance(position_side, PositionSide)
            else PositionSide(str(position_side).upper())
        )
        self._records.pop(
            PositionKey(
                exchange=exchange,
                account_id=account_id,
                symbol=symbol,
                position_side=side,
            ),
            None,
        )

    def all(self) -> tuple[PositionRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def as_dict(self) -> dict[PositionKey, PositionRecord]:
        return dict(self._records)

    def _legs(
        self,
        symbol: str,
        *,
        account_id: str = "default",
        exchange: str = "unknown",
    ) -> tuple[PositionRecord | None, PositionRecord | None]:
        return (
            self.get(
                symbol,
                PositionSide.LONG,
                account_id=account_id,
                exchange=exchange,
            ),
            self.get(
                symbol,
                PositionSide.SHORT,
                account_id=account_id,
                exchange=exchange,
            ),
        )

    def net_amount(
        self,
        symbol: str,
        *,
        account_id: str = "default",
        exchange: str = "unknown",
    ) -> Decimal:
        long_record, short_record = self._legs(
            symbol,
            account_id=account_id,
            exchange=exchange,
        )
        return (long_record.amount if long_record else Decimal("0")) - (
            short_record.amount if short_record else Decimal("0")
        )

    def gross_amount(
        self,
        symbol: str,
        *,
        account_id: str = "default",
        exchange: str = "unknown",
    ) -> Decimal:
        long_record, short_record = self._legs(
            symbol,
            account_id=account_id,
            exchange=exchange,
        )
        return (long_record.amount if long_record else Decimal("0")) + (
            short_record.amount if short_record else Decimal("0")
        )

    def hedge_ratio(
        self,
        symbol: str,
        *,
        account_id: str = "default",
        exchange: str = "unknown",
    ) -> Decimal | None:
        long_record, short_record = self._legs(
            symbol,
            account_id=account_id,
            exchange=exchange,
        )
        long_amount = long_record.amount if long_record else Decimal("0")
        short_amount = short_record.amount if short_record else Decimal("0")
        if long_amount == 0 and short_amount == 0:
            return Decimal("0")
        if long_amount == 0:
            return None
        return short_amount / long_amount
