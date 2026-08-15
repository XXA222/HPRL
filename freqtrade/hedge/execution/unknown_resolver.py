"""Multi-source UNKNOWN order recovery without blind resubmission."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from threading import RLock
from typing import Protocol

from .service import ExecutionOrder, ExchangeExecutionPort, ExternalOrderSnapshot
from .state_machine import OrderState


class ResolutionSource(StrEnum):
    DIRECT_QUERY = "DIRECT_QUERY"
    OPEN_ORDERS = "OPEN_ORDERS"
    RECENT_FILLS = "RECENT_FILLS"
    USER_STREAM = "USER_STREAM"
    MERGED = "MERGED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    snapshot: ExternalOrderSnapshot | None
    source: ResolutionSource
    errors: tuple[str, ...] = ()


class UserStreamOrderCachePort(Protocol):
    def get(self, client_order_id: str) -> ExternalOrderSnapshot | None: ...


class UserStreamOrderCacheSinkPort(UserStreamOrderCachePort, Protocol):
    def put(self, snapshot: ExternalOrderSnapshot) -> None: ...


class InMemoryUserStreamOrderCache:
    """Last validated order fact observed on the account user stream."""

    def __init__(self) -> None:
        self._items: dict[str, ExternalOrderSnapshot] = {}
        self._lock = RLock()

    def put(self, snapshot: ExternalOrderSnapshot) -> None:
        if not isinstance(snapshot, ExternalOrderSnapshot):
            raise TypeError("snapshot must be an ExternalOrderSnapshot")
        with self._lock:
            current = self._items.get(snapshot.client_order_id)
            if current is None or snapshot.observed_at >= current.observed_at:
                self._items[snapshot.client_order_id] = snapshot

    def get(self, client_order_id: str) -> ExternalOrderSnapshot | None:
        with self._lock:
            return self._items.get(client_order_id)


class UnknownOrderResolver:
    def __init__(
        self,
        exchange: ExchangeExecutionPort,
        *,
        user_stream_cache: UserStreamOrderCachePort | None = None,
    ) -> None:
        self._exchange = exchange
        self._user_stream_cache = user_stream_cache
        self._last_result = ResolutionResult(None, ResolutionSource.UNRESOLVED)
        self._lock = RLock()

    @property
    def last_result(self) -> ResolutionResult:
        with self._lock:
            return self._last_result

    def resolve(self, order: ExecutionOrder) -> ExternalOrderSnapshot | None:
        if not isinstance(order, ExecutionOrder):
            raise TypeError("order must be an ExecutionOrder")
        errors: list[str] = []
        candidates: list[tuple[ResolutionSource, ExternalOrderSnapshot]] = []

        try:
            direct = self._exchange.query_order(client_order_id=order.client_order_id)
        except Exception as exc:
            errors.append(f"DIRECT_QUERY:{type(exc).__name__}")
        else:
            candidate = self._valid_candidate(order, direct, errors, "DIRECT_QUERY")
            if candidate is not None and candidate.status is not OrderState.UNKNOWN:
                candidates.append((ResolutionSource.DIRECT_QUERY, candidate))

        try:
            open_orders = tuple(
                self._exchange.list_open_orders(
                    account_id=order.intent.account_id,
                    symbol=order.intent.symbol,
                )
            )
        except Exception as exc:
            errors.append(f"OPEN_ORDERS:{type(exc).__name__}")
            open_orders = ()
        for item in open_orders:
            candidate = self._valid_candidate(order, item, errors, "OPEN_ORDERS")
            if candidate is not None and candidate.status is not OrderState.UNKNOWN:
                candidates.append((ResolutionSource.OPEN_ORDERS, candidate))

        try:
            recent_fills = tuple(
                self._exchange.list_recent_fills(
                    account_id=order.intent.account_id,
                    symbol=order.intent.symbol,
                )
            )
        except Exception as exc:
            errors.append(f"RECENT_FILLS:{type(exc).__name__}")
            recent_fills = ()

        if self._user_stream_cache is not None:
            try:
                stream_value = self._user_stream_cache.get(order.client_order_id)
            except Exception as exc:
                errors.append(f"USER_STREAM:{type(exc).__name__}")
            else:
                candidate = self._valid_candidate(
                    order, stream_value, errors, "USER_STREAM"
                )
                if candidate is not None and candidate.status is not OrderState.UNKNOWN:
                    candidates.append((ResolutionSource.USER_STREAM, candidate))

        fill_fact = self._aggregate_fills(order, recent_fills, errors)
        base, base_source = self._merge_order_candidates(order, candidates)
        resolved, source = self._merge(order, base, base_source, fill_fact)
        result = ResolutionResult(resolved, source, tuple(errors))
        with self._lock:
            self._last_result = result
        return resolved

    @staticmethod
    def _valid_candidate(
        order: ExecutionOrder,
        value: object,
        errors: list[str],
        source: str,
    ) -> ExternalOrderSnapshot | None:
        if value is None:
            return None
        if not isinstance(value, ExternalOrderSnapshot):
            errors.append(f"{source}:INVALID_TYPE")
            return None
        if value.client_order_id != order.client_order_id:
            errors.append(f"{source}:CLIENT_ORDER_ID_MISMATCH")
            return None
        if value.filled_quantity > order.approved_quantity:
            errors.append(f"{source}:FILL_EXCEEDS_APPROVED")
            return None
        if (
            value.status is OrderState.FILLED
            and value.filled_quantity != order.approved_quantity
        ):
            errors.append(f"{source}:INVALID_FILLED_QUANTITY")
            return None
        return value

    @staticmethod
    def _merge_order_candidates(
        order: ExecutionOrder,
        candidates: list[tuple[ResolutionSource, ExternalOrderSnapshot]],
    ) -> tuple[ExternalOrderSnapshot | None, ResolutionSource | None]:
        if not candidates:
            return None, None
        if len(candidates) == 1:
            source, snapshot = candidates[0]
            if (
                snapshot.filled_quantity == order.approved_quantity
                and snapshot.status in {OrderState.PARTIAL, OrderState.CANCELED}
            ):
                snapshot = ExternalOrderSnapshot(
                    client_order_id=snapshot.client_order_id,
                    status=OrderState.FILLED,
                    filled_quantity=snapshot.filled_quantity,
                    average_price=snapshot.average_price,
                    exchange_order_id=snapshot.exchange_order_id,
                    exchange_trade_id=snapshot.exchange_trade_id,
                    reason=snapshot.reason,
                    observed_at=snapshot.observed_at,
                )
            return snapshot, source

        max_filled = max(item.filled_quantity for _, item in candidates)
        if max_filled == order.approved_quantity and max_filled > 0:
            status = OrderState.FILLED
        elif any(item.status is OrderState.CANCELED for _, item in candidates):
            status = OrderState.CANCELED
        elif max_filled > 0:
            status = OrderState.PARTIAL
        elif any(item.status is OrderState.REJECTED for _, item in candidates):
            status = OrderState.REJECTED
        else:
            status = OrderState.ACKNOWLEDGED

        best_pair = max(
            candidates,
            key=lambda pair: (
                pair[1].filled_quantity,
                pair[1].average_price is not None,
                pair[0] is ResolutionSource.DIRECT_QUERY,
                pair[1].observed_at,
            ),
        )
        best = best_pair[1]
        average = None
        if max_filled > 0:
            priced = [
                pair
                for pair in candidates
                if pair[1].filled_quantity == max_filled
                and pair[1].average_price is not None
            ]
            if priced:
                average = max(
                    priced,
                    key=lambda pair: (
                        pair[0] is ResolutionSource.DIRECT_QUERY,
                        pair[1].observed_at,
                    ),
                )[1].average_price
        exchange_order_id = best.exchange_order_id
        if exchange_order_id is None:
            identified = [
                pair for pair in candidates if pair[1].exchange_order_id is not None
            ]
            if identified:
                exchange_order_id = max(
                    identified,
                    key=lambda pair: (
                        pair[0] is ResolutionSource.DIRECT_QUERY,
                        pair[1].observed_at,
                    ),
                )[1].exchange_order_id
        reasons: list[str] = []
        for _, item in candidates:
            if item.reason and item.reason not in reasons:
                reasons.append(item.reason)
        return (
            ExternalOrderSnapshot(
                client_order_id=order.client_order_id,
                status=status,
                filled_quantity=max_filled,
                average_price=average,
                exchange_order_id=exchange_order_id,
                exchange_trade_id=best.exchange_trade_id,
                reason=";".join(reasons) or "merged_order_facts",
                observed_at=max(item.observed_at for _, item in candidates),
            ),
            ResolutionSource.MERGED,
        )

    @classmethod
    def _aggregate_fills(
        cls,
        order: ExecutionOrder,
        values: tuple[object, ...],
        errors: list[str],
    ) -> ExternalOrderSnapshot | None:
        matching: list[ExternalOrderSnapshot] = []
        for value in values:
            candidate = cls._valid_candidate(order, value, errors, "RECENT_FILLS")
            if candidate is not None and candidate.filled_quantity > 0:
                matching.append(candidate)
        if not matching:
            return None

        unique: list[ExternalOrderSnapshot] = []
        seen_trade_ids: set[str] = set()
        seen_fingerprints: set[tuple[object, ...]] = set()
        for item in sorted(matching, key=lambda value: value.observed_at):
            if item.exchange_trade_id:
                if item.exchange_trade_id in seen_trade_ids:
                    continue
                seen_trade_ids.add(item.exchange_trade_id)
            else:
                fingerprint = (
                    item.client_order_id,
                    item.filled_quantity,
                    item.average_price,
                    item.exchange_order_id,
                    item.observed_at,
                )
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)
            unique.append(item)
        if not unique:
            return None

        raw_total = sum((item.filled_quantity for item in unique), Decimal("0"))
        total = min(raw_total, order.approved_quantity)
        if total <= 0:
            return None
        all_priced = all(item.average_price is not None for item in unique)
        average = None
        if all_priced:
            notional = sum(
                (
                    item.filled_quantity * item.average_price
                    for item in unique
                    if item.average_price is not None
                ),
                Decimal("0"),
            )
            average = notional / raw_total
        status = (
            OrderState.FILLED
            if total == order.approved_quantity
            else OrderState.PARTIAL
        )
        reason = "recovered_from_recent_fills"
        if raw_total > order.approved_quantity:
            reason += ":capped_overfill"
        return ExternalOrderSnapshot(
            client_order_id=order.client_order_id,
            status=status,
            filled_quantity=total,
            average_price=average,
            exchange_order_id=unique[-1].exchange_order_id,
            exchange_trade_id=unique[-1].exchange_trade_id,
            reason=reason,
            observed_at=max(item.observed_at for item in unique),
        )

    @staticmethod
    def _merge(
        order: ExecutionOrder,
        base: ExternalOrderSnapshot | None,
        base_source: ResolutionSource | None,
        fills: ExternalOrderSnapshot | None,
    ) -> tuple[ExternalOrderSnapshot | None, ResolutionSource]:
        if base is None and fills is None:
            return None, ResolutionSource.UNRESOLVED
        if base is None:
            return fills, ResolutionSource.RECENT_FILLS
        if fills is None:
            return base, base_source or ResolutionSource.UNRESOLVED

        filled = max(base.filled_quantity, fills.filled_quantity)
        if filled == order.approved_quantity:
            status = OrderState.FILLED
        elif filled > 0:
            status = (
                OrderState.CANCELED
                if base.status is OrderState.CANCELED
                else OrderState.PARTIAL
            )
        else:
            status = base.status
        if base.filled_quantity >= fills.filled_quantity and base.average_price:
            average = base.average_price
        else:
            average = fills.average_price or base.average_price
        observed_at = max(base.observed_at, fills.observed_at)
        reason_parts = [part for part in (base.reason, fills.reason) if part]
        return (
            ExternalOrderSnapshot(
                client_order_id=order.client_order_id,
                status=status,
                filled_quantity=filled,
                average_price=average,
                exchange_order_id=(
                    base.exchange_order_id or fills.exchange_order_id
                ),
                exchange_trade_id=fills.exchange_trade_id,
                reason=";".join(reason_parts) or "merged_recovery_fact",
                observed_at=observed_at,
            ),
            ResolutionSource.MERGED,
        )
