"""Deterministic fake exchange and permanently gated Binance write stub."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from threading import RLock
from typing import Sequence

from .idempotency import InMemoryIdempotencyStore
from .kill_switch import KillSwitch
from .service import (
    AllowAllRiskApproval,
    ApprovedOrderIntent,
    DefinitiveCancellationError,
    DefinitiveSubmissionError,
    ExecutionService,
    ExternalOrderSnapshot,
    InMemoryAuditLog,
    InMemoryExecutionStore,
    RiskApprovalPort,
)
from .state_machine import OrderState
from .unknown_resolver import UnknownOrderResolver
from freqtrade.hedge.telemetry.metrics import HedgeMetrics


@dataclass(frozen=True, slots=True)
class _TimeoutBehavior:
    recoverable_snapshot: ExternalOrderSnapshot | None = None


class FakeExchangeExecutionPort:
    def __init__(self) -> None:
        self.submit_calls: list[ApprovedOrderIntent] = []
        self.cancel_calls: list[str] = []
        self.query_calls: list[str] = []
        self.open_order_queries = 0
        self.recent_fill_queries = 0
        self._behaviors: deque[
            ExternalOrderSnapshot | Exception | _TimeoutBehavior
        ] = deque()
        self._orders: dict[str, ExternalOrderSnapshot] = {}
        self._approved: dict[str, ApprovedOrderIntent] = {}
        self._recent_fills: list[ExternalOrderSnapshot] = []
        self._trade_sequence = 0
        self._fills_by_trade_id: dict[str, tuple[str, Decimal, Decimal, Decimal, str | None]] = {}
        self._lock = RLock()

    def queue_snapshot(
        self,
        status: OrderState,
        *,
        filled_quantity: Decimal | str = Decimal("0"),
        average_price: Decimal | str | None = None,
        exchange_order_id: str | None = None,
        exchange_trade_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        snapshot = ExternalOrderSnapshot(
            client_order_id="__AUTO__",
            status=status,
            filled_quantity=Decimal(filled_quantity),
            average_price=(
                None if average_price is None else Decimal(average_price)
            ),
            exchange_order_id=exchange_order_id,
            exchange_trade_id=exchange_trade_id,
            reason=reason,
        )
        with self._lock:
            self._behaviors.append(snapshot)

    def queue_timeout(
        self,
        *,
        recover_as: ExternalOrderSnapshot | None = None,
    ) -> None:
        with self._lock:
            self._behaviors.append(_TimeoutBehavior(recover_as))

    def queue_error(self, error: Exception) -> None:
        if not isinstance(error, Exception):
            raise TypeError("error must be an exception")
        with self._lock:
            self._behaviors.append(error)

    def submit_order(
        self,
        approved: ApprovedOrderIntent,
    ) -> ExternalOrderSnapshot:
        if not isinstance(approved, ApprovedOrderIntent):
            raise TypeError("approved must be an ApprovedOrderIntent")
        with self._lock:
            self.submit_calls.append(approved)
            self._approved[approved.client_order_id] = approved
            behavior = (
                self._behaviors.popleft()
                if self._behaviors
                else ExternalOrderSnapshot(
                    client_order_id="__AUTO__",
                    status=OrderState.ACKNOWLEDGED,
                )
            )
            if isinstance(behavior, _TimeoutBehavior):
                if behavior.recoverable_snapshot is not None:
                    snapshot = self._normalize(
                        approved.client_order_id,
                        behavior.recoverable_snapshot,
                    )
                    self._orders[approved.client_order_id] = snapshot
                raise TimeoutError("fake submit timeout")
            if isinstance(behavior, Exception):
                raise behavior
            snapshot = self._normalize(approved.client_order_id, behavior)
            self._orders[approved.client_order_id] = snapshot
            return snapshot

    def query_order(
        self,
        *,
        client_order_id: str,
    ) -> ExternalOrderSnapshot | None:
        with self._lock:
            self.query_calls.append(client_order_id)
            return self._orders.get(client_order_id)

    def list_open_orders(
        self,
        *,
        account_id: str,
        symbol: str,
    ) -> Sequence[ExternalOrderSnapshot]:
        with self._lock:
            self.open_order_queries += 1
            return tuple(
                order
                for client_id, order in self._orders.items()
                if self._matches_scope(client_id, account_id, symbol)
                and order.status
                in {
                    OrderState.ACKNOWLEDGED,
                    OrderState.PARTIAL,
                    OrderState.UNKNOWN,
                }
            )

    def list_recent_fills(
        self,
        *,
        account_id: str,
        symbol: str,
    ) -> Sequence[ExternalOrderSnapshot]:
        with self._lock:
            self.recent_fill_queries += 1
            return tuple(
                fill
                for fill in self._recent_fills
                if self._matches_scope(
                    fill.client_order_id,
                    account_id,
                    symbol,
                )
            )

    def cancel_order(
        self,
        *,
        client_order_id: str,
    ) -> ExternalOrderSnapshot:
        with self._lock:
            self.cancel_calls.append(client_order_id)
            current = self._orders.get(client_order_id)
            if current is None:
                return ExternalOrderSnapshot(
                    client_order_id=client_order_id,
                    status=OrderState.UNKNOWN,
                    reason="not_found_during_cancel",
                )
            canceled = ExternalOrderSnapshot(
                client_order_id=client_order_id,
                status=OrderState.CANCELED,
                filled_quantity=current.filled_quantity,
                average_price=current.average_price,
                exchange_order_id=current.exchange_order_id,
                observed_at=datetime.now(UTC),
            )
            self._orders[client_order_id] = canceled
            return canceled

    def restore_order(
        self,
        approved: ApprovedOrderIntent,
        snapshot: ExternalOrderSnapshot,
    ) -> None:
        """Restore one durable fake order without performing a new submission."""

        if not isinstance(approved, ApprovedOrderIntent):
            raise TypeError("approved must be an ApprovedOrderIntent")
        if not isinstance(snapshot, ExternalOrderSnapshot):
            raise TypeError("snapshot must be an ExternalOrderSnapshot")
        if approved.client_order_id != snapshot.client_order_id:
            raise ValueError("restored approved intent and snapshot ids differ")
        with self._lock:
            self._approved[approved.client_order_id] = approved
            self._orders[approved.client_order_id] = snapshot

    def remember_fill_identity(
        self,
        *,
        trade_id: str,
        client_order_id: str,
        quantity: Decimal,
        price: Decimal,
        fee: Decimal = Decimal("0"),
        fee_currency: str | None = "USDT",
    ) -> None:
        """Seed an immutable trade identity during SQL recovery."""
        normalized_currency = None if fee_currency is None else str(fee_currency).strip().upper()
        normalized_quantity = Decimal(quantity)
        normalized_price = Decimal(price)
        normalized_fee = Decimal(fee)
        if (
            not normalized_quantity.is_finite()
            or normalized_quantity <= 0
            or not normalized_price.is_finite()
            or normalized_price <= 0
            or not normalized_fee.is_finite()
            or normalized_fee < 0
        ):
            raise ValueError("recovered fill identity contains invalid numeric values")
        identity = (
            str(client_order_id),
            normalized_quantity,
            normalized_price,
            normalized_fee,
            normalized_currency,
        )
        with self._lock:
            current = self._fills_by_trade_id.get(trade_id)
            if current is not None and current != identity:
                raise ValueError("recovered trade id has conflicting fill data")
            self._fills_by_trade_id[trade_id] = identity

    def set_order(self, snapshot: ExternalOrderSnapshot) -> None:
        if not isinstance(snapshot, ExternalOrderSnapshot):
            raise TypeError("snapshot must be an ExternalOrderSnapshot")
        with self._lock:
            self._orders[snapshot.client_order_id] = snapshot

    def acknowledge_order(
        self,
        client_order_id: str,
        *,
        exchange_order_id: str | None = None,
    ) -> ExternalOrderSnapshot:
        """Move a submitted fake order to ACKNOWLEDGED."""
        with self._lock:
            approved = self._require_approved(client_order_id)
            current = self._orders.get(client_order_id)
            if current is not None and current.filled_quantity > 0:
                raise ValueError("filled fake order cannot return to ACKNOWLEDGED")
            snapshot = ExternalOrderSnapshot(
                client_order_id=client_order_id,
                status=OrderState.ACKNOWLEDGED,
                exchange_order_id=(
                    exchange_order_id
                    or (current.exchange_order_id if current else None)
                    or f"fake-{approved.client_order_id}"
                ),
                observed_at=datetime.now(UTC),
            )
            self._orders[client_order_id] = snapshot
            return snapshot

    def fill_order(
        self,
        client_order_id: str,
        *,
        quantity: Decimal | str,
        price: Decimal | str,
        exchange_trade_id: str | None = None,
        fee: Decimal | str | None = None,
        fee_currency: str | None = "USDT",
    ) -> ExternalOrderSnapshot:
        """Apply one incremental fake fill and return the cumulative order snapshot."""
        fill_quantity = Decimal(quantity)
        fill_price = Decimal(price)
        fill_fee = Decimal("0") if fee is None else Decimal(fee)
        fee_currency = None if fee_currency is None else str(fee_currency).strip().upper()
        if not fill_quantity.is_finite() or fill_quantity <= 0:
            raise ValueError("fill quantity must be positive and finite")
        if not fill_price.is_finite() or fill_price <= 0:
            raise ValueError("fill price must be positive and finite")
        if not fill_fee.is_finite() or fill_fee < 0:
            raise ValueError("fill fee must be finite and non-negative")
        with self._lock:
            if exchange_trade_id is not None:
                existing_fill = self._fills_by_trade_id.get(exchange_trade_id)
                identity = (client_order_id, fill_quantity, fill_price, fill_fee, fee_currency)
                if existing_fill is not None:
                    if existing_fill != identity:
                        raise ValueError("duplicate exchange trade id has conflicting fill data")
                    existing_order = self._orders.get(client_order_id)
                    if existing_order is None:
                        raise RuntimeError("replayed fake fill has no cumulative order")
                    return existing_order
            approved = self._require_approved(client_order_id)
            current = self._orders.get(client_order_id)
            if current is None:
                current = ExternalOrderSnapshot(
                    client_order_id=client_order_id,
                    status=OrderState.ACKNOWLEDGED,
                    exchange_order_id=f"fake-{client_order_id}",
                )
            if current.status is OrderState.REJECTED:
                raise ValueError("rejected fake order cannot be filled")
            cumulative = current.filled_quantity + fill_quantity
            if cumulative > approved.approved_quantity:
                raise ValueError("fake fill exceeds approved quantity")
            previous_notional = (
                current.filled_quantity * current.average_price
                if current.average_price is not None
                else Decimal("0")
            )
            average_price = (
                previous_notional + fill_quantity * fill_price
            ) / cumulative
            if cumulative == approved.approved_quantity:
                status = OrderState.FILLED
            elif current.status is OrderState.CANCELED:
                status = OrderState.CANCELED
            else:
                status = OrderState.PARTIAL
            now = datetime.now(UTC)
            self._trade_sequence += 1
            trade_id = exchange_trade_id or (
                f"fake-trade-{self._trade_sequence}-{client_order_id}"
            )
            cumulative_snapshot = ExternalOrderSnapshot(
                client_order_id=client_order_id,
                status=status,
                filled_quantity=cumulative,
                average_price=average_price,
                exchange_order_id=(
                    current.exchange_order_id or f"fake-{client_order_id}"
                ),
                exchange_trade_id=trade_id,
                last_fill_fee=fill_fee,
                fee_currency=fee_currency,
                reason="fake_fill",
                observed_at=now,
            )
            incremental_fill = ExternalOrderSnapshot(
                client_order_id=client_order_id,
                status=OrderState.PARTIAL,
                filled_quantity=fill_quantity,
                average_price=fill_price,
                exchange_order_id=cumulative_snapshot.exchange_order_id,
                exchange_trade_id=trade_id,
                last_fill_fee=fill_fee,
                fee_currency=fee_currency,
                reason="fake_incremental_fill",
                observed_at=now,
            )
            self._orders[client_order_id] = cumulative_snapshot
            self._fills_by_trade_id[trade_id] = (
                client_order_id, fill_quantity, fill_price, fill_fee, fee_currency
            )
            self._recent_fills.append(incremental_fill)
            return cumulative_snapshot

    def reject_order(
        self,
        client_order_id: str,
        *,
        reason: str = "fake_rejected",
    ) -> ExternalOrderSnapshot:
        with self._lock:
            self._require_approved(client_order_id)
            current = self._orders.get(client_order_id)
            if current is not None and current.filled_quantity > 0:
                raise ValueError("partially filled fake order cannot be rejected")
            snapshot = ExternalOrderSnapshot(
                client_order_id=client_order_id,
                status=OrderState.REJECTED,
                exchange_order_id=(current.exchange_order_id if current else None),
                reason=reason,
                observed_at=datetime.now(UTC),
            )
            self._orders[client_order_id] = snapshot
            return snapshot

    def mark_unknown(
        self,
        client_order_id: str,
        *,
        reason: str = "fake_unknown",
    ) -> ExternalOrderSnapshot:
        with self._lock:
            self._require_approved(client_order_id)
            current = self._orders.get(client_order_id)
            snapshot = ExternalOrderSnapshot(
                client_order_id=client_order_id,
                status=OrderState.UNKNOWN,
                filled_quantity=(
                    current.filled_quantity if current is not None else Decimal("0")
                ),
                average_price=(current.average_price if current is not None else None),
                exchange_order_id=(
                    current.exchange_order_id if current is not None else None
                ),
                reason=reason,
                observed_at=datetime.now(UTC),
            )
            self._orders[client_order_id] = snapshot
            return snapshot

    def list_orders(self) -> tuple[ExternalOrderSnapshot, ...]:
        with self._lock:
            return tuple(
                self._orders[key]
                for key in sorted(self._orders)
            )

    def add_recent_fill(self, snapshot: ExternalOrderSnapshot) -> None:
        if not isinstance(snapshot, ExternalOrderSnapshot):
            raise TypeError("snapshot must be an ExternalOrderSnapshot")
        with self._lock:
            self._recent_fills.append(snapshot)

    def _require_approved(self, client_order_id: str) -> ApprovedOrderIntent:
        approved = self._approved.get(client_order_id)
        if approved is None:
            raise KeyError(client_order_id)
        return approved

    def _matches_scope(
        self,
        client_order_id: str,
        account_id: str,
        symbol: str,
    ) -> bool:
        approved = self._approved.get(client_order_id)
        if approved is None:
            return True
        return (
            approved.intent.account_id == account_id
            and approved.intent.symbol == symbol
        )

    @staticmethod
    def _normalize(
        client_order_id: str,
        snapshot: ExternalOrderSnapshot,
    ) -> ExternalOrderSnapshot:
        return ExternalOrderSnapshot(
            client_order_id=client_order_id,
            status=snapshot.status,
            filled_quantity=snapshot.filled_quantity,
            average_price=snapshot.average_price,
            exchange_order_id=(
                snapshot.exchange_order_id or f"fake-{client_order_id}"
            ),
            exchange_trade_id=snapshot.exchange_trade_id,
            last_fill_fee=snapshot.last_fill_fee,
            fee_currency=snapshot.fee_currency,
            reason=snapshot.reason,
            observed_at=datetime.now(UTC),
        )


@dataclass(frozen=True, slots=True)
class FakeExecutionHarness:
    """Ready-to-use in-memory execution runtime for tests and dry-run planning."""

    service: ExecutionService
    exchange: FakeExchangeExecutionPort
    store: InMemoryExecutionStore
    idempotency: InMemoryIdempotencyStore
    resolver: UnknownOrderResolver
    kill_switch: KillSwitch
    audit: InMemoryAuditLog
    metrics: HedgeMetrics


def build_fake_execution_harness(
    *,
    risk: RiskApprovalPort | None = None,
) -> FakeExecutionHarness:
    """Construct all direction-five ports around one deterministic fake exchange."""
    exchange = FakeExchangeExecutionPort()
    store = InMemoryExecutionStore()
    idempotency = InMemoryIdempotencyStore()
    resolver = UnknownOrderResolver(exchange)
    kill_switch = KillSwitch()
    audit = InMemoryAuditLog()
    metrics = HedgeMetrics()
    service = ExecutionService(
        risk=risk if risk is not None else AllowAllRiskApproval(),
        exchange=exchange,
        store=store,
        idempotency=idempotency,
        unknown_resolver=resolver,
        kill_switch=kill_switch,
        audit=audit,
        metrics=metrics,
    )
    return FakeExecutionHarness(
        service=service,
        exchange=exchange,
        store=store,
        idempotency=idempotency,
        resolver=resolver,
        kill_switch=kill_switch,
        audit=audit,
        metrics=metrics,
    )


class WriteFeatureDisabledError(
    DefinitiveSubmissionError,
    DefinitiveCancellationError,
):
    pass


class BinanceExecutionAdapterStub:
    """A deliberate non-implementation. It performs no network or SDK calls."""

    def __init__(self, *, live_trading_enabled: bool = False) -> None:
        if not isinstance(live_trading_enabled, bool):
            raise TypeError("live_trading_enabled must be a boolean")
        self.live_trading_enabled = live_trading_enabled

    def _disabled(self) -> WriteFeatureDisabledError:
        if self.live_trading_enabled:
            return WriteFeatureDisabledError(
                "Binance write adapter is a gated stub; integration is disabled"
            )
        return WriteFeatureDisabledError("hedge live trading is disabled")

    def submit_order(
        self,
        approved: ApprovedOrderIntent,
    ) -> ExternalOrderSnapshot:
        raise self._disabled()

    def query_order(
        self,
        *,
        client_order_id: str,
    ) -> ExternalOrderSnapshot | None:
        raise self._disabled()

    def list_open_orders(
        self,
        *,
        account_id: str,
        symbol: str,
    ) -> Sequence[ExternalOrderSnapshot]:
        raise self._disabled()

    def list_recent_fills(
        self,
        *,
        account_id: str,
        symbol: str,
    ) -> Sequence[ExternalOrderSnapshot]:
        raise self._disabled()

    def cancel_order(
        self,
        *,
        client_order_id: str,
    ) -> ExternalOrderSnapshot:
        raise self._disabled()
