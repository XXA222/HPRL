"""Canonical production state fingerprint used across replay/recovery/shadow evidence."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from typing import Iterable


@dataclass(frozen=True, slots=True)
class CanonicalLeg:
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    realized_pnl: Decimal


@dataclass(frozen=True, slots=True)
class CanonicalOrder:
    client_order_id: str
    exchange_order_id: str | None
    symbol: str
    side: str
    position_side: str
    status: str
    requested_quantity: Decimal
    filled_quantity: Decimal


@dataclass(frozen=True, slots=True)
class CanonicalProductionState:
    account_id: str
    wallet_balance: Decimal
    available_balance: Decimal
    legs: tuple[CanonicalLeg, ...]
    orders: tuple[CanonicalOrder, ...]
    funding_total: Decimal
    fee_total: Decimal
    event_cursor: int
    fencing_token: int

    @classmethod
    def build(
        cls,
        *,
        account_id: str,
        wallet_balance: Decimal,
        available_balance: Decimal,
        legs: Iterable[CanonicalLeg],
        orders: Iterable[CanonicalOrder],
        funding_total: Decimal,
        fee_total: Decimal,
        event_cursor: int,
        fencing_token: int,
    ) -> "CanonicalProductionState":
        if not account_id.strip():
            raise ValueError("account_id is required")
        if event_cursor < 0 or fencing_token <= 0:
            raise ValueError("cursor must be nonnegative and fencing token positive")
        return cls(
            account_id.strip(),
            Decimal(wallet_balance),
            Decimal(available_balance),
            tuple(sorted(legs, key=lambda x: (x.symbol, x.side))),
            tuple(sorted(orders, key=lambda x: x.client_order_id)),
            Decimal(funding_total),
            Decimal(fee_total),
            int(event_cursor),
            int(fencing_token),
        )

    @property
    def semantic_hash(self) -> str:
        body = {
            "account_id": self.account_id,
            "wallet_balance": str(self.wallet_balance),
            "available_balance": str(self.available_balance),
            "funding_total": str(self.funding_total),
            "fee_total": str(self.fee_total),
            "event_cursor": self.event_cursor,
            "fencing_token": self.fencing_token,
            "legs": [
                {
                    "symbol": x.symbol,
                    "side": x.side,
                    "quantity": str(x.quantity),
                    "entry_price": str(x.entry_price),
                    "realized_pnl": str(x.realized_pnl),
                }
                for x in self.legs
            ],
            "orders": [
                {
                    "client_order_id": x.client_order_id,
                    "exchange_order_id": x.exchange_order_id,
                    "symbol": x.symbol,
                    "side": x.side,
                    "position_side": x.position_side,
                    "status": x.status,
                    "requested_quantity": str(x.requested_quantity),
                    "filled_quantity": str(x.filled_quantity),
                }
                for x in self.orders
            ],
        }
        return sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()
