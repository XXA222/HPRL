"""Authoritative order ownership classification for hedge execution.

Client-order-id prefixes are only hints. Durable ExecutionStore membership is the
source of truth, preventing foreign orders with a similar prefix from being canceled.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Sequence

from .service import ExecutionOrder, ExecutionStorePort


class OrderOwnership(StrEnum):
    MANAGED = "MANAGED"
    EXTERNAL = "EXTERNAL"
    ORPHAN_PREFIX = "ORPHAN_PREFIX"


@dataclass(frozen=True, slots=True)
class OwnershipDecision:
    client_order_id: str
    ownership: OrderOwnership
    order: ExecutionOrder | None = None
    planner_intent_id: str | None = None


class ExecutionOrderOwnershipRegistry:
    """Classify and resolve orders using the durable execution store."""

    def __init__(
        self,
        store: ExecutionStorePort,
        *,
        managed_prefixes: Sequence[str] = ("FTH-",),
    ) -> None:
        if not callable(getattr(store, "get_by_client_order_id", None)):
            raise TypeError("store must implement get_by_client_order_id")
        if not callable(getattr(store, "list_orders", None)):
            raise TypeError("store must implement list_orders")
        prefixes = tuple(dict.fromkeys(str(item).strip().upper() for item in managed_prefixes))
        if not prefixes or any(not item for item in prefixes):
            raise ValueError("managed_prefixes must contain non-empty values")
        self._store = store
        self._prefixes = prefixes

    @property
    def store(self) -> ExecutionStorePort:
        return self._store

    def classify(self, client_order_id: str) -> OwnershipDecision:
        client_id = str(client_order_id).strip()
        if not client_id:
            raise ValueError("client_order_id is required")
        order = self._store.get_by_client_order_id(client_id)
        if order is not None:
            planner_id = str(order.intent.metadata.get("planner_intent_id", "")).strip() or None
            return OwnershipDecision(client_id, OrderOwnership.MANAGED, order, planner_id)
        upper = client_id.upper()
        if any(upper.startswith(prefix) for prefix in self._prefixes):
            return OwnershipDecision(client_id, OrderOwnership.ORPHAN_PREFIX)
        return OwnershipDecision(client_id, OrderOwnership.EXTERNAL)

    def resolve_managed_reference(self, reference: str) -> ExecutionOrder | None:
        """Resolve a client id or planner intent id to one authoritative order."""

        value = str(reference).strip()
        if not value:
            raise ValueError("order reference is required")
        direct = self._store.get_by_client_order_id(value)
        if direct is not None:
            return direct
        matches = [
            order
            for order in self._store.list_orders()
            if str(order.intent.metadata.get("planner_intent_id", "")).strip() == value
        ]
        if not matches:
            return None
        nonterminal = [order for order in matches if not order.lifecycle.terminal]
        candidates = nonterminal or matches
        candidates.sort(key=lambda item: (item.created_at, item.client_order_id), reverse=True)
        return candidates[0]

    def managed_open_orders(
        self,
        *,
        account_id: str | None = None,
        symbol: str | None = None,
    ) -> tuple[ExecutionOrder, ...]:
        selected: list[ExecutionOrder] = []
        for order in self._store.list_orders():
            if order.lifecycle.terminal:
                continue
            if account_id is not None and order.intent.account_id != account_id:
                continue
            if symbol is not None and order.intent.symbol != symbol:
                continue
            selected.append(order)
        return tuple(sorted(selected, key=lambda item: (item.created_at, item.client_order_id)))

    def classify_many(self, client_order_ids: Iterable[str]) -> tuple[OwnershipDecision, ...]:
        return tuple(self.classify(item) for item in client_order_ids)
