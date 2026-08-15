"""Repository adapters used by the integrated hedge composition root.

The in-memory implementation is intentionally complete enough for the real
Binance read-only runtime, local smoke tests, Dry-run and integration tests.  It
can be replaced by the direction-one SQLAlchemy repository without changing the
exchange service.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from threading import RLock
from uuid import uuid4

from freqtrade.hedge.exchange.base import (
    AccountEventFact,
    AccountSnapshotFact,
    AtomicReadonlyFactRepository,
    BalanceFact,
    CalibrationKind,
    ExchangeFactBatch,
    FillFact,
    OrderFact,
    PositionFact,
    ReadonlyHistoryCursorRepository,
    ReconciliationDiffFact,
)


class InMemoryReadonlyRepository(AtomicReadonlyFactRepository, ReadonlyHistoryCursorRepository):
    """Deterministic latest-fact repository with append-only event evidence."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._positions: dict[tuple[str, str, str], PositionFact] = {}
        self._orders: dict[tuple[str, str, str], OrderFact] = {}
        self._fills: dict[tuple[str, str, str], FillFact] = {}
        self._account_events: dict[tuple[str, str], AccountEventFact] = {}
        self._account_snapshots: dict[str, AccountSnapshotFact] = {}
        self._balances: dict[tuple[str, str], BalanceFact] = {}
        self._runs: dict[str, dict[str, object]] = {}
        self._diffs: dict[str, list[ReconciliationDiffFact]] = {}
        self._history_cursors: dict[tuple[str, str], int] = {}
        self._batches: list[ExchangeFactBatch] = []

    @staticmethod
    def _newer(incoming: object, current: object | None) -> bool:
        if current is None:
            return True
        incoming_ms = int(getattr(incoming, "update_time_ms", getattr(incoming, "event_time_ms", 0)))
        current_ms = int(getattr(current, "update_time_ms", getattr(current, "event_time_ms", 0)))
        incoming_observed = getattr(incoming, "observed_at", datetime.min)
        current_observed = getattr(current, "observed_at", datetime.min)
        return (incoming_ms, incoming_observed) >= (current_ms, current_observed)

    async def append_exchange_fact_batch(self, batch: ExchangeFactBatch) -> None:
        with self._lock:
            self._batches.append(batch)
        if batch.account_snapshot is not None:
            await self.append_account_snapshot(
                batch.account_snapshot,
                reconciliation_run_id=batch.reconciliation_run_id,
            )
        await self.append_balance_snapshots(
            batch.balances,
            reconciliation_run_id=batch.reconciliation_run_id,
        )
        await self.append_position_snapshots(
            batch.positions,
            reconciliation_run_id=batch.reconciliation_run_id,
        )
        await self.append_order_snapshots(
            batch.orders,
            reconciliation_run_id=batch.reconciliation_run_id,
        )
        await self.append_fill_events(
            batch.fills,
            reconciliation_run_id=batch.reconciliation_run_id,
        )
        await self.append_account_events(batch.account_events)
        if batch.reconciliation_run_id is not None:
            await self.append_reconciliation_diffs(
                batch.reconciliation_run_id,
                batch.reconciliation_diffs,
            )

    async def append_position_snapshots(
        self,
        facts: Sequence[PositionFact],
        *,
        reconciliation_run_id: str | None = None,
    ) -> None:
        del reconciliation_run_id
        with self._lock:
            for fact in facts:
                current = self._positions.get(fact.key)
                if self._newer(fact, current):
                    if fact.quantity == 0:
                        self._positions.pop(fact.key, None)
                    else:
                        self._positions[fact.key] = fact

    async def append_order_snapshots(
        self,
        facts: Sequence[OrderFact],
        *,
        reconciliation_run_id: str | None = None,
    ) -> None:
        del reconciliation_run_id
        with self._lock:
            for fact in facts:
                current = self._orders.get(fact.key)
                if self._newer(fact, current):
                    self._orders[fact.key] = fact

    async def append_fill_events(
        self,
        facts: Sequence[FillFact],
        *,
        reconciliation_run_id: str | None = None,
    ) -> None:
        del reconciliation_run_id
        with self._lock:
            for fact in facts:
                self._fills.setdefault(fact.key, fact)

    async def append_account_snapshot(
        self,
        fact: AccountSnapshotFact,
        *,
        reconciliation_run_id: str | None = None,
    ) -> None:
        del reconciliation_run_id
        with self._lock:
            current = self._account_snapshots.get(fact.account_id)
            if current is None or fact.observed_at >= current.observed_at:
                self._account_snapshots[fact.account_id] = fact

    async def append_balance_snapshots(
        self,
        facts: Sequence[BalanceFact],
        *,
        reconciliation_run_id: str | None = None,
    ) -> None:
        del reconciliation_run_id
        with self._lock:
            for fact in facts:
                key = (fact.account_id, fact.asset)
                current = self._balances.get(key)
                if current is None or fact.observed_at >= current.observed_at:
                    self._balances[key] = fact

    async def append_account_events(self, facts: Sequence[AccountEventFact]) -> None:
        with self._lock:
            for fact in facts:
                self._account_events.setdefault((fact.account_id, fact.identity), fact)

    async def begin_reconciliation(
        self,
        *,
        account_id: str,
        kind: CalibrationKind,
        started_at: datetime,
    ) -> str:
        run_id = uuid4().hex
        with self._lock:
            self._runs[run_id] = {
                "account_id": account_id,
                "kind": kind.value,
                "started_at": started_at,
                "completed_at": None,
                "status": "RUNNING",
                "reason": "",
            }
            self._diffs[run_id] = []
        return run_id

    async def append_reconciliation_diffs(
        self,
        run_id: str,
        diffs: Sequence[ReconciliationDiffFact],
    ) -> None:
        with self._lock:
            self._diffs.setdefault(run_id, []).extend(diffs)

    async def complete_reconciliation(
        self,
        run_id: str,
        *,
        completed_at: datetime,
        status: str,
        reason: str,
    ) -> None:
        with self._lock:
            run = self._runs.setdefault(run_id, {})
            run.update(completed_at=completed_at, status=status, reason=reason)

    async def load_active_positions(self, account_id: str) -> tuple[PositionFact, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        fact
                        for fact in self._positions.values()
                        if fact.account_id == account_id and fact.quantity != 0
                    ),
                    key=lambda item: (item.symbol, item.position_side),
                )
            )

    async def load_active_orders(self, account_id: str) -> tuple[OrderFact, ...]:
        with self._lock:
            return tuple(
                sorted(
                    (
                        fact
                        for fact in self._orders.values()
                        if fact.account_id == account_id and fact.active
                    ),
                    key=lambda item: (item.symbol, item.position_side, item.exchange_order_id),
                )
            )

    async def has_fill(self, account_id: str, symbol: str, exchange_trade_id: str) -> bool:
        with self._lock:
            return (account_id, symbol, exchange_trade_id) in self._fills

    async def load_history_cursor(self, account_id: str, cursor_name: str) -> int | None:
        with self._lock:
            return self._history_cursors.get((account_id, cursor_name))

    async def save_history_cursor(self, account_id: str, cursor_name: str, cursor_ms: int) -> None:
        with self._lock:
            self._history_cursors[(account_id, cursor_name)] = int(cursor_ms)

    def account_snapshot(self, account_id: str) -> AccountSnapshotFact | None:
        with self._lock:
            return self._account_snapshots.get(account_id)

    def reconciliation_runs(self, account_id: str) -> tuple[dict[str, object], ...]:
        with self._lock:
            rows = [dict(value, run_id=key) for key, value in self._runs.items() if value.get("account_id") == account_id]
            return tuple(sorted(rows, key=lambda item: str(item.get("started_at"))))

    def reconciliation_diffs(self, run_id: str) -> tuple[ReconciliationDiffFact, ...]:
        with self._lock:
            return tuple(self._diffs.get(run_id, ()))

    @property
    def batches(self) -> tuple[ExchangeFactBatch, ...]:
        with self._lock:
            return tuple(self._batches)

    @property
    def account_events(self) -> tuple[AccountEventFact, ...]:
        """Return an immutable event-evidence snapshot for diagnostics/acceptance."""
        with self._lock:
            return tuple(
                sorted(
                    self._account_events.values(),
                    key=lambda item: (item.event_time_ms, item.event_type, item.identity),
                )
            )


class PersistenceMirroringReadonlyRepository(InMemoryReadonlyRepository):
    """Mirror direction-two exchange facts into the direction-one ledger.

    The in-memory projection remains the fast runtime view.  Each atomic batch is
    additionally written through ``HedgePersistenceService`` so restart recovery,
    current projections and the transactional outbox are populated by the same
    REST/User-Stream facts.
    """

    def __init__(self, persistence_service: object) -> None:
        super().__init__()
        self._persistence_service = persistence_service
        self._persistence_run_ids: dict[str, str] = {}

    @staticmethod
    def _event_time(milliseconds: int, fallback: datetime) -> datetime:
        if milliseconds > 0:
            from datetime import UTC

            return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
        return fallback

    @staticmethod
    def _order_action(order: OrderFact) -> str:
        side = order.position_side.upper()
        exchange_side = order.side.upper()
        increases = (side == "LONG" and exchange_side == "BUY") or (
            side == "SHORT" and exchange_side == "SELL"
        )
        return "INCREASE" if increases else "REDUCE"

    @staticmethod
    def _fact_source(value: str) -> str:
        normalized = value.upper()
        if "REST" in normalized:
            return "REST"
        if "STREAM" in normalized or "WEBSOCKET" in normalized or normalized == "WS":
            return "WEBSOCKET"
        if "RECOVER" in normalized:
            return "RECOVERY"
        if "MIGRAT" in normalized:
            return "MIGRATION"
        return "LOCAL"

    @staticmethod
    def _account_event_type(value: str) -> str:
        normalized = value.upper()
        if "FUND" in normalized:
            return "FUNDING"
        if "FEE" in normalized or "COMMISSION" in normalized:
            return "FEE"
        if "TRANSFER" in normalized:
            return "TRANSFER"
        return "BALANCE"

    @staticmethod
    def _fill_action(fill: FillFact) -> str:
        side = fill.position_side.upper()
        exchange_side = fill.side.upper()
        increases = (side == "LONG" and exchange_side == "BUY") or (
            side == "SHORT" and exchange_side == "SELL"
        )
        return "INCREASE" if increases else "REDUCE"

    @staticmethod
    def _observed_key(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")

    async def begin_reconciliation(
        self, *, account_id: str, kind: CalibrationKind, started_at: datetime
    ) -> str:
        service = self._persistence_service
        persistent_id = service.transaction(
            lambda repository: repository.start_reconciliation(
                account_id=account_id,
                exchange="binance",
                trigger=kind.value,
                scope="FULL_ACCOUNT",
            ).run_id
        )
        run_id = await super().begin_reconciliation(
            account_id=account_id, kind=kind, started_at=started_at
        )
        self._persistence_run_ids[run_id] = str(persistent_id)
        return run_id

    async def append_reconciliation_diffs(
        self, run_id: str, diffs: Sequence[ReconciliationDiffFact]
    ) -> None:
        persistent_id = self._persistence_run_ids.get(run_id)
        if persistent_id is not None and diffs:
            def persist(repository: object) -> None:
                for index, diff in enumerate(diffs):
                    repository.add_reconciliation_diff(
                        run_id=persistent_id,
                        diff_key=f"d2:{run_id}:{index}:{diff.entity_type}:{diff.entity_key}:{diff.reason_code}",
                        entity_type=diff.entity_type,
                        entity_key=diff.entity_key,
                        severity=diff.severity,
                        field_name=diff.reason_code,
                        local_value=diff.expected,
                        exchange_value=diff.observed,
                        resolution=diff.resolution.value,
                        repair_action=diff.resolution_detail,
                    )
            self._persistence_service.transaction(persist)
        await super().append_reconciliation_diffs(run_id, diffs)

    async def complete_reconciliation(
        self, run_id: str, *, completed_at: datetime, status: str, reason: str
    ) -> None:
        persistent_id = self._persistence_run_ids.get(run_id)
        if persistent_id is not None:
            self._persistence_service.transaction(
                lambda repository: repository.complete_reconciliation(
                    run_id=persistent_id,
                    status=status,
                    summary={"direction2_run_id": run_id, "completed_at": completed_at.isoformat()},
                    error_message=None if status.upper() in {"COMPLETED", "HEALTHY", "CONSISTENT"} else reason,
                )
            )
        await super().complete_reconciliation(
            run_id, completed_at=completed_at, status=status, reason=reason
        )

    async def append_exchange_fact_batch(self, batch: ExchangeFactBatch) -> None:
        service = self._persistence_service

        def persist(repository: object) -> None:
            for position in batch.positions:
                # REST mark/PnL can change without Binance updateTime changing.
                # Use observation time for immutable snapshot identity.
                event_time = position.observed_at
                observed_key = self._observed_key(position.observed_at)
                snapshot_key = (
                    f"d2-position:{position.account_id}:{position.symbol}:"
                    f"{position.position_side}:{observed_key}:{position.source}"
                )
                repository.append_position_snapshot(
                    snapshot_key=snapshot_key,
                    account_id=position.account_id,
                    exchange="binance",
                    symbol=position.symbol,
                    venue_symbol=position.symbol,
                    position_side=position.position_side,
                    quantity=abs(position.quantity),
                    entry_price=position.entry_price,
                    mark_price=position.mark_price,
                    notional=abs(position.quantity) * position.mark_price,
                    unrealized_pnl=position.unrealized_pnl,
                    liquidation_price=position.liquidation_price,
                    leverage=position.leverage,
                    margin_mode=position.margin_mode,
                    source=self._fact_source(position.source),
                    source_event_time=event_time,
                    sequence_number=max(position.update_time_ms, 0),
                    source_version=f"{position.update_time_ms}:{observed_key}",
                    raw_payload=dict(position.raw),
                )

            for order in batch.orders:
                event_time = self._event_time(order.update_time_ms, order.observed_at)
                cumulative_quote = order.cumulative_filled_quantity * order.average_price
                snapshot_key = (
                    f"d2-order:{order.account_id}:{order.exchange_order_id}:"
                    f"{order.update_time_ms}:{order.source}"
                )
                repository.append_order_snapshot(
                    snapshot_key=snapshot_key,
                    account_id=order.account_id,
                    exchange="binance",
                    symbol=order.symbol,
                    position_side=order.position_side,
                    exchange_order_id=order.exchange_order_id,
                    client_order_id=order.client_order_id or None,
                    side=order.side,
                    action=self._order_action(order),
                    order_type=order.order_type,
                    status=order.status,
                    original_quantity=order.original_quantity,
                    executed_quantity=order.cumulative_filled_quantity,
                    cumulative_quote=cumulative_quote,
                    average_price=order.average_price if order.average_price > 0 else None,
                    source=self._fact_source(order.source),
                    source_event_time=event_time,
                    sequence_number=max(order.update_time_ms, 0),
                    source_version=str(order.update_time_ms),
                    correlation_id=order.correlation_id,
                    payload_version=order.event_version,
                    raw_payload=dict(order.raw),
                )

            for fill in batch.fills:
                repository.apply_fill(
                    exchange="binance",
                    account_id=fill.account_id,
                    symbol=fill.symbol,
                    position_side=fill.position_side,
                    exchange_trade_id=fill.exchange_trade_id,
                    exchange_order_id=fill.exchange_order_id,
                    side=fill.side,
                    quantity=fill.quantity,
                    price=fill.price,
                    source=self._fact_source(fill.source),
                    event_time=self._event_time(fill.event_time_ms, fill.observed_at),
                    action=self._fill_action(fill),
                    sequence_number=max(fill.event_time_ms, 0),
                    client_order_id=None,
                    fee_amount=fill.commission,
                    fee_currency=fill.commission_asset,
                    realized_pnl=fill.realized_pnl,
                    raw_payload=dict(fill.raw),
                )

            for event in batch.account_events:
                event_time = self._event_time(event.event_time_ms, event.observed_at)
                repository.record_account_event(
                    event_key=event.identity,
                    account_id=event.account_id,
                    exchange="binance",
                    event_type=self._account_event_type(event.event_type),
                    asset=event.currency or "USDT",
                    amount=event.amount if event.amount is not None else 0,
                    source=self._fact_source(event.source),
                    event_time=event_time,
                    raw_payload=dict(event.payload),
                )

            for balance in batch.balances:
                repository.record_audit_event(
                    account_id=balance.account_id,
                    exchange="binance",
                    event_type="BALANCE_SNAPSHOT",
                    correlation_id=f"d2-balance:{balance.account_id}:{balance.asset}:{self._observed_key(balance.observed_at)}",
                    entity_type="BALANCE",
                    entity_id=balance.asset,
                    severity="INFO",
                    reason_code=None,
                    actor="BINANCE_READONLY",
                    payload={
                        "asset": balance.asset,
                        "wallet_balance": str(balance.wallet_balance),
                        "available_balance": str(balance.available_balance),
                        "cross_wallet_balance": str(balance.cross_wallet_balance),
                        "unrealized_pnl": str(balance.unrealized_pnl),
                        "observed_at": balance.observed_at.isoformat(),
                    },
                )

            account = batch.account_snapshot
            if account is not None:
                gross_long = sum(
                    (abs(item.quantity) * item.mark_price for item in batch.positions if item.position_side.upper() == "LONG"),
                    0,
                )
                gross_short = sum(
                    (abs(item.quantity) * item.mark_price for item in batch.positions if item.position_side.upper() == "SHORT"),
                    0,
                )
                pending_risk = sum(
                    (
                        max(item.original_quantity - item.cumulative_filled_quantity, 0)
                        * (item.average_price if item.average_price > 0 else 0)
                        for item in batch.orders
                        if item.active and self._order_action(item) == "INCREASE"
                    ),
                    0,
                )
                equity = account.total_margin_balance
                margin_utilization = (
                    account.total_initial_margin / equity if equity > 0 else 0
                )
                repository.append_account_risk_snapshot(
                    snapshot_key=f"d2-account:{account.account_id}:{self._observed_key(account.observed_at)}",
                    account_id=account.account_id,
                    exchange="binance",
                    source="REST",
                    source_event_time=account.observed_at,
                    equity=equity,
                    wallet_balance=account.total_wallet_balance,
                    available_balance=account.total_available_balance,
                    margin_balance=account.total_margin_balance,
                    total_initial_margin=account.total_initial_margin,
                    total_maintenance_margin=account.total_maintenance_margin,
                    gross_long_notional=gross_long,
                    gross_short_notional=gross_short,
                    gross_exposure=gross_long + gross_short,
                    net_exposure=gross_long - gross_short,
                    pending_risk=pending_risk,
                    margin_utilization=margin_utilization,
                    liquidation_buffer=(
                        (equity - account.total_maintenance_margin) / equity
                        if equity > 0
                        else 0
                    ),
                    risk_state="OBSERVED",
                    risk_data_valid=equity > 0,
                    source_snapshot_id=batch.reconciliation_run_id,
                    source_version=self._observed_key(account.observed_at),
                    reason_codes=(),
                    raw_payload=dict(account.raw),
                )

        # Persistence is committed first. The fast in-memory projection advances
        # only after the durable transaction succeeds.
        service.transaction(persist)
        await super().append_exchange_fact_batch(batch)

