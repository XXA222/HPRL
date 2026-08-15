"""Read-only RPC projections preserving dual-leg Hedge authority."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from decimal import Decimal
from threading import RLock
from typing import Any, Iterable, Mapping

from .models import HedgeBucket, HedgeEvent, HedgeSide, LegSnapshot, utc_datetime


def _serial(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return utc_datetime(value).isoformat()
    if isinstance(value, Mapping):
        return {str(key): _serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_serial(item) for item in value]
    if is_dataclass(value):
        return _serial(asdict(value))
    raw = getattr(value, "value", None)
    return raw if raw is not None else value


@dataclass(frozen=True, slots=True)
class HedgePositionProjection:
    pair: str
    side: HedgeSide
    quantity: Decimal
    average_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    core_quantity: Decimal
    tactical_quantity: Decimal
    funding: Decimal = Decimal("0")
    fees: Decimal = Decimal("0")
    liquidation_price: Decimal | None = None
    updated_at: datetime = field(default_factory=utc_datetime)

    @classmethod
    def from_legs(cls, legs: Iterable[LegSnapshot], **kwargs: Any) -> "HedgePositionProjection":
        rows = tuple(legs)
        if not rows:
            raise ValueError("at least one leg bucket is required")
        pair = rows[0].pair
        side = rows[0].side
        if any(item.pair != pair or item.side is not side for item in rows):
            raise ValueError("position projection legs must share pair and side")
        quantity = sum((item.quantity for item in rows), Decimal("0"))
        notional_cost = sum((item.quantity * item.average_price for item in rows), Decimal("0"))
        mark = rows[-1].mark_price
        return cls(
            pair=pair,
            side=side,
            quantity=quantity,
            average_price=Decimal("0") if quantity == 0 else notional_cost / quantity,
            mark_price=mark,
            unrealized_pnl=sum((item.unrealized_pnl for item in rows), Decimal("0")),
            realized_pnl=sum((item.realized_pnl for item in rows), Decimal("0")),
            core_quantity=sum((item.quantity for item in rows if item.bucket is HedgeBucket.CORE), Decimal("0")),
            tactical_quantity=sum((item.quantity for item in rows if item.bucket is HedgeBucket.TACTICAL), Decimal("0")),
            funding=sum((item.funding for item in rows), Decimal("0")),
            fees=sum((item.fees for item in rows), Decimal("0")),
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class HedgeAccountProjection:
    account_id: str
    equity: Decimal
    wallet_balance: Decimal
    available_balance: Decimal
    gross_notional: Decimal
    net_notional: Decimal
    margin_utilization: Decimal
    positions: tuple[HedgePositionProjection, ...]
    readiness: Mapping[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=utc_datetime)

    def to_dict(self) -> dict[str, Any]:
        return _serial(self)

    def standard_balance_projection(self) -> dict[str, Any]:
        """Subset compatible with ordinary balance consumers plus Hedge extension."""
        return {
            "currencies": [
                {
                    "currency": "USDT",
                    "free": float(self.available_balance),
                    "balance": float(self.wallet_balance),
                    "used": float(max(self.wallet_balance - self.available_balance, Decimal("0"))),
                    "est_stake": float(self.equity),
                    "stake": "USDT",
                }
            ],
            "total": float(self.equity),
            "symbol": "",
            "value": float(self.equity),
            "note": "Hedge account projection; dual-leg positions are in hedge_native",
            "hedge_native": self.to_dict(),
        }

    def standard_status_projection(self) -> list[dict[str, Any]]:
        """Return one read-only row per side, never collapsing LONG and SHORT."""
        rows: list[dict[str, Any]] = []
        for position in self.positions:
            if position.quantity <= 0:
                continue
            rows.append(
                {
                    "trade_id": f"hedge:{position.pair}:{position.side.value}",
                    "pair": position.pair,
                    "is_open": True,
                    "is_short": position.side is HedgeSide.SHORT,
                    "amount": float(position.quantity),
                    "open_rate": float(position.average_price),
                    "current_rate": float(position.mark_price),
                    "profit_abs": float(position.unrealized_pnl),
                    "realized_profit": float(position.realized_pnl),
                    "position_side": position.side.value,
                    "core_quantity": str(position.core_quantity),
                    "tactical_quantity": str(position.tactical_quantity),
                    "authority": "hedge-ledger-readonly-projection",
                    "force_exit_supported": False,
                }
            )
        return rows

    def profit_projection(self) -> dict[str, Any]:
        realized = sum((item.realized_pnl for item in self.positions), Decimal("0"))
        unrealized = sum((item.unrealized_pnl for item in self.positions), Decimal("0"))
        fees = sum((item.fees for item in self.positions), Decimal("0"))
        funding = sum((item.funding for item in self.positions), Decimal("0"))
        return {
            "profit_closed_coin": float(realized),
            "profit_all_coin": float(realized + unrealized),
            "profit_closed_percent_mean": 0.0,
            "profit_all_percent_mean": 0.0,
            "trade_count": len([item for item in self.positions if item.quantity > 0]),
            "closed_trade_count": 0,
            "first_trade_date": "",
            "latest_trade_date": "",
            "avg_duration": "",
            "best_pair": "",
            "winning_trades": 0,
            "losing_trades": 0,
            "hedge_native": {
                "realized_pnl": str(realized),
                "unrealized_pnl": str(unrealized),
                "fees": str(fees),
                "funding": str(funding),
                "gross_notional": str(self.gross_notional),
                "net_notional": str(self.net_notional),
            },
        }


class HedgeRpcProjectionService:
    """Atomic in-memory projection cache for REST, FreqUI and command clients."""

    def __init__(self, *, event_capacity: int = 5000) -> None:
        if event_capacity <= 0:
            raise ValueError("event_capacity must be positive")
        self.event_capacity = event_capacity
        self._account: HedgeAccountProjection | None = None
        self._events: deque[HedgeEvent] = deque(maxlen=event_capacity)
        self._lock = RLock()

    def update_account(self, account: HedgeAccountProjection) -> None:
        with self._lock:
            self._account = account

    def publish(self, event: HedgeEvent) -> None:
        with self._lock:
            self._events.append(event)

    def account(self) -> HedgeAccountProjection | None:
        with self._lock:
            return self._account

    def events(
        self,
        *,
        limit: int = 100,
        pair: str | None = None,
        side: HedgeSide | None = None,
        severity: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        if limit <= 0:
            return ()
        with self._lock:
            rows = tuple(self._events)
        selected: list[dict[str, Any]] = []
        for event in reversed(rows):
            if pair and event.pair != pair.upper():
                continue
            if side and event.side is not HedgeSide.parse(side):
                continue
            if severity and event.severity != severity.upper():
                continue
            selected.append(_serial(event))
            if len(selected) >= limit:
                break
        return tuple(selected)

    def api_payload(self) -> dict[str, Any]:
        account = self.account()
        return {
            "schema": "hedge-rpc-projection-v1",
            "account": None if account is None else account.to_dict(),
            "events": list(self.events(limit=100)),
        }
