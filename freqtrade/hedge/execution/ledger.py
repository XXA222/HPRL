"""In-memory transactional execution ledger used by Fake/Dry-run and contract tests."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from threading import RLock
from types import MappingProxyType
from typing import Mapping, Sequence

from freqtrade.hedge.contracts.events import FillEvent, OutboxEvent
from freqtrade.hedge.contracts.types import IntentAction, PositionKey, PositionSide


@dataclass(frozen=True, slots=True)
class PositionProjection:
    position_key: PositionKey
    quantity: Decimal = Decimal("0")
    average_entry_price: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    funding: Decimal = Decimal("0")
    version: int = 0
    updated_at: datetime = datetime.min.replace(tzinfo=UTC)

    def apply_fill(self, fill: FillEvent) -> "PositionProjection":
        if fill.position_key != self.position_key:
            raise ValueError("fill position key does not match projection")
        qty = self.quantity
        avg = self.average_entry_price
        realized = self.realized_pnl
        if fill.action in {IntentAction.OPEN, IntentAction.INCREASE}:
            next_qty = qty + fill.quantity
            next_avg = ((qty * avg) + (fill.quantity * fill.price)) / next_qty
        else:
            if fill.quantity > qty:
                raise ValueError("reduce fill would cross the position through zero")
            next_qty = qty - fill.quantity
            if self.position_key.position_side is PositionSide.LONG:
                realized += (fill.price - avg) * fill.quantity
            else:
                realized += (avg - fill.price) * fill.quantity
            next_avg = Decimal("0") if next_qty == 0 else avg
        return replace(
            self,
            quantity=next_qty,
            average_entry_price=next_avg,
            realized_pnl=realized,
            fees=self.fees + fill.fee,
            version=self.version + 1,
            updated_at=fill.observed_time,
        )


@dataclass(frozen=True, slots=True)
class LedgerAuditRecord:
    event_type: str
    payload: Mapping[str, object]
    occurred_at: datetime


class InMemoryExecutionLedger:
    """Atomic order/fill/projection/outbox fake for direction-one integration."""

    def __init__(self) -> None:
        self._orders: dict[str, object] = {}
        self._fills: dict[tuple[str, str, str], FillEvent] = {}
        self._positions: dict[PositionKey, PositionProjection] = {}
        self._outbox: dict[str, OutboxEvent] = {}
        self._audit: list[LedgerAuditRecord] = []
        self._lock = RLock()

    def seed_position(
        self,
        *,
        position_key: PositionKey,
        quantity: Decimal,
        average_entry_price: Decimal,
        realized_pnl: Decimal = Decimal("0"),
        fees: Decimal = Decimal("0"),
        funding: Decimal = Decimal("0"),
        observed_at: datetime | None = None,
    ) -> PositionProjection:
        """Idempotently seed an adopted/live position before reduce-only fills.

        A nonzero local projection may only be reseeded with identical facts; callers
        must reconcile conflicting exchange evidence instead of overwriting fills.
        """
        values = (quantity, average_entry_price, realized_pnl, fees, funding)
        if any(not isinstance(item, Decimal) or not item.is_finite() for item in values):
            raise ValueError("position seed values must be finite Decimal values")
        if quantity < 0 or average_entry_price < 0:
            raise ValueError("position seed quantity/price cannot be negative")
        timestamp = observed_at or datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("position seed observed_at must be timezone-aware")
        candidate = PositionProjection(
            position_key=position_key,
            quantity=quantity,
            average_entry_price=(Decimal("0") if quantity == 0 else average_entry_price),
            realized_pnl=realized_pnl,
            fees=fees,
            funding=funding,
            version=0,
            updated_at=timestamp,
        )
        with self._lock:
            existing = self._positions.get(position_key)
            if existing is not None and existing.version > 0:
                if (existing.quantity, existing.average_entry_price) != (
                    candidate.quantity, candidate.average_entry_price
                ):
                    raise RuntimeError("live position conflicts with applied execution fills")
                return existing
            self._positions[position_key] = candidate
            return candidate

    def record(
        self,
        *,
        order: object,
        event_type: str,
        fill: FillEvent | None = None,
        outbox: OutboxEvent | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        client_order_id = str(getattr(order, "client_order_id", "")).strip()
        if not client_order_id:
            raise ValueError("order must expose client_order_id")
        event_name = str(event_type).strip().upper()
        if not event_name:
            raise ValueError("event_type is required")
        now = datetime.now(UTC)
        with self._lock:
            if fill is not None:
                fill_key = (
                    fill.position_key.exchange,
                    fill.position_key.account_id,
                    fill.trade_id,
                )
                existing = self._fills.get(fill_key)
                if existing is not None and existing != fill:
                    raise ValueError("duplicate trade_id contains conflicting fill data")
                if existing is None:
                    current = self._positions.get(
                        fill.position_key,
                        PositionProjection(fill.position_key),
                    )
                    projected = current.apply_fill(fill)
                    self._fills[fill_key] = fill
                    self._positions[fill.position_key] = projected
            self._orders[client_order_id] = order
            if outbox is not None:
                self._outbox[str(outbox.event_id)] = outbox
            self._audit.append(
                LedgerAuditRecord(
                    event_type=event_name,
                    payload=MappingProxyType(dict(payload or {})),
                    occurred_at=now,
                )
            )

    def order(self, client_order_id: str) -> object | None:
        with self._lock:
            return self._orders.get(client_order_id)

    def fills(self) -> tuple[FillEvent, ...]:
        with self._lock:
            return tuple(sorted(self._fills.values(), key=lambda item: (item.observed_time, item.trade_id)))

    def position(self, position_key: PositionKey) -> PositionProjection:
        with self._lock:
            return self._positions.get(position_key, PositionProjection(position_key))

    def positions(self, *, account_id: str | None = None, symbol: str | None = None) -> tuple[PositionProjection, ...]:
        with self._lock:
            values = tuple(self._positions.values())
        return tuple(
            item
            for item in sorted(values, key=lambda item: item.position_key.lock_name)
            if (account_id is None or item.position_key.account_id == account_id)
            and (symbol is None or item.position_key.symbol == symbol)
        )

    def audit(self, *, limit: int = 100) -> tuple[LedgerAuditRecord, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        with self._lock:
            return tuple(reversed(self._audit[-limit:]))

    def outbox(self, *, unpublished_only: bool = False) -> tuple[OutboxEvent, ...]:
        with self._lock:
            values = tuple(self._outbox.values())
        if unpublished_only:
            values = tuple(item for item in values if item.published_at is None)
        return tuple(sorted(values, key=lambda item: item.occurred_at))

    def mark_published(self, event_id: str, *, published_at: datetime | None = None) -> None:
        with self._lock:
            event = self._outbox[event_id]
            self._outbox[event_id] = replace(
                event,
                published_at=published_at or datetime.now(UTC),
                attempts=event.attempts + 1,
            )

    def mark_publish_attempt(self, event_id: str) -> None:
        with self._lock:
            event = self._outbox[event_id]
            self._outbox[event_id] = replace(
                event,
                attempts=event.attempts + 1,
            )
