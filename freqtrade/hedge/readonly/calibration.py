from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from datetime import datetime, timedelta
from decimal import Decimal

from freqtrade.hedge.exchange.base import (
    AccountEventFact,
    AccountSnapshotFact,
    BalanceFact,
    CalibrationKind,
    ExchangeFactBatch,
    CalibrationResult,
    Clock,
    OrderFact,
    PositionFact,
    ReadonlyFactRepository,
    ReconciliationDiffFact,
    ReconciliationResolution,
    ReadonlyReasonCode,
    SystemClock,
    maybe_await,
    stable_fingerprint,
    to_primitive,
)
from freqtrade.hedge.exchange.binance_readonly import BinanceAccountBundle, BinanceReadonlyClient
from freqtrade.hedge.exchange.rate_limit import BinanceDataError
from freqtrade.hedge.exchange.symbol_codec import normalize_exchange_symbols


logger = logging.getLogger(__name__)

DEFAULT_QUANTITY_TOLERANCE = Decimal("0")
DEFAULT_FINANCIAL_TOLERANCE = Decimal("0.00000001")
DEFAULT_FILL_LOOKBACK = timedelta(hours=72)
DEFAULT_HISTORY_OVERLAP = timedelta(minutes=5)
DEFAULT_MAX_HISTORY_BACKFILL = timedelta(days=30)
HISTORY_CURSOR_NAME = "BINANCE_FUTURES_HISTORY_MS"


class HistoryBackfillRequired(RuntimeError):
    def __init__(self, start_ms: int, end_ms: int, maximum_ms: int) -> None:
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.maximum_ms = maximum_ms
        super().__init__(
            f"{ReadonlyReasonCode.HISTORY_GAP_REQUIRES_BACKFILL.value}:"
            f"start={start_ms}:end={end_ms}:max_span={maximum_ms}"
        )


class ReadonlySafetyHalt(RuntimeError):
    def __init__(self, reason: str, *, result: CalibrationResult | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.result = result


def _position_map(items: Sequence[PositionFact]) -> dict[tuple[str, str], PositionFact]:
    result: dict[tuple[str, str], PositionFact] = {}
    for item in items:
        if item.quantity == 0:
            continue
        key = (item.symbol, item.position_side)
        if key in result:
            raise BinanceDataError(f"Duplicate active position fact: {key[0]}:{key[1]}")
        result[key] = item
    return result


def _order_map(items: Sequence[OrderFact]) -> dict[tuple[str, str], OrderFact]:
    result: dict[tuple[str, str], OrderFact] = {}
    for item in items:
        if not item.active:
            continue
        key = (item.symbol, item.exchange_order_id)
        previous = result.get(key)
        if previous is not None:
            raise BinanceDataError(
                f"Duplicate active order fact: {item.symbol}:{item.exchange_order_id}"
            )
        result[key] = item
    return result


class ReadonlyCalibration:
    def __init__(
        self,
        *,
        client: BinanceReadonlyClient,
        repository: ReadonlyFactRepository,
        managed_symbols: Sequence[str],
        clock: Clock | None = None,
        quantity_tolerance: Decimal = DEFAULT_QUANTITY_TOLERANCE,
        financial_tolerance: Decimal = DEFAULT_FINANCIAL_TOLERANCE,
        fill_lookback: timedelta = DEFAULT_FILL_LOOKBACK,
        history_overlap: timedelta = DEFAULT_HISTORY_OVERLAP,
        max_history_backfill: timedelta | None = DEFAULT_MAX_HISTORY_BACKFILL,
    ) -> None:
        if quantity_tolerance < 0 or not quantity_tolerance.is_finite():
            raise ValueError("quantity_tolerance must be finite and nonnegative")
        if financial_tolerance < 0 or not financial_tolerance.is_finite():
            raise ValueError("financial_tolerance must be finite and nonnegative")
        if isinstance(managed_symbols, (str, bytes)):
            raise ValueError("managed_symbols must be a sequence, not a string")
        normalized_symbols = set(normalize_exchange_symbols(list(managed_symbols)))
        if fill_lookback.total_seconds() <= 0:
            raise ValueError("fill_lookback must be positive")
        if history_overlap.total_seconds() < 0:
            raise ValueError("history_overlap must be nonnegative")
        if history_overlap >= fill_lookback:
            raise ValueError("history_overlap must be smaller than fill_lookback")
        if max_history_backfill is not None and max_history_backfill.total_seconds() <= 0:
            raise ValueError("max_history_backfill must be positive or None")
        self.client = client
        self.repository = repository
        self.managed_symbols = frozenset(normalized_symbols)
        self.clock = clock or SystemClock()
        self.quantity_tolerance = quantity_tolerance
        self.financial_tolerance = financial_tolerance
        self.fill_lookback = fill_lookback
        self.history_overlap = history_overlap
        self.max_history_backfill = max_history_backfill
        self._last_persist_was_atomic = False
        self.last_bundle: BinanceAccountBundle | None = None
        self._run_lock = asyncio.Lock()
        self._history_cursor_ms: int | None = None
        self._history_cursor_loaded = False

    async def run(self, kind: CalibrationKind) -> CalibrationResult:
        async with self._run_lock:
            return await self._run_once(kind)

    async def _load_history_cursor(self) -> None:
        if self._history_cursor_loaded:
            return
        self._history_cursor_loaded = True
        loader = getattr(self.repository, "load_history_cursor", None)
        if not callable(loader):
            return
        value = await maybe_await(
            loader(self.client.account_id, HISTORY_CURSOR_NAME)
        )
        if value is None:
            return
        parsed = int(value)
        if parsed < 0:
            raise ValueError("history cursor must be nonnegative")
        self._history_cursor_ms = parsed

    async def _save_history_cursor(self, cursor_ms: int) -> None:
        if cursor_ms < 0:
            raise ValueError("history cursor must be nonnegative")
        self._history_cursor_ms = max(self._history_cursor_ms or 0, cursor_ms)
        saver = getattr(self.repository, "save_history_cursor", None)
        if callable(saver):
            await maybe_await(
                saver(
                    self.client.account_id,
                    HISTORY_CURSOR_NAME,
                    self._history_cursor_ms,
                )
            )

    async def _history_start_time_ms(self, started_at: datetime) -> int:
        await self._load_history_cursor()
        end_ms = int(started_at.timestamp() * 1000)
        lookback_start = max(
            0, int((started_at - self.fill_lookback).timestamp() * 1000)
        )
        if self._history_cursor_ms is None:
            return lookback_start
        overlap_ms = int(self.history_overlap.total_seconds() * 1000)
        cursor_start = max(0, min(self._history_cursor_ms, end_ms) - overlap_ms)
        if self.max_history_backfill is not None:
            maximum_ms = int(self.max_history_backfill.total_seconds() * 1000)
            if end_ms - cursor_start > maximum_ms:
                raise HistoryBackfillRequired(cursor_start, end_ms, maximum_ms)
        return cursor_start

    @staticmethod
    def _includes_history(kind: CalibrationKind) -> bool:
        return kind in {
            CalibrationKind.STARTUP,
            CalibrationKind.RECONNECT,
            CalibrationKind.FULL,
        }

    @staticmethod
    def _requires_clock_sync(kind: CalibrationKind) -> bool:
        return kind in {CalibrationKind.FULL, CalibrationKind.RECONNECT}

    async def _complete_failed_run(self, run_id: str, exc: Exception) -> None:
        if isinstance(exc, ReadonlySafetyHalt):
            return
        try:
            await self.repository.complete_reconciliation(
                run_id,
                completed_at=self.clock.now(),
                status="FAILED",
                reason=f"{type(exc).__name__}:{exc}",
            )
        except Exception:
            logger.exception(
                "Failed to persist reconciliation failure for run %s",
                run_id,
            )

    async def _complete_successful_run(
        self,
        run_id: str,
        result: CalibrationResult,
        *,
        include_history: bool,
        cursor_ms: int,
        diffs: Sequence[ReconciliationDiffFact],
    ) -> CalibrationResult:
        if diffs and not self._last_persist_was_atomic:
            await self.repository.append_reconciliation_diffs(run_id, diffs)
        status = "CONSISTENT" if result.consistent else "DRIFT"
        if result.unmanaged_positions or result.unmanaged_orders:
            status = "HALT"
        await self.repository.complete_reconciliation(
            run_id,
            completed_at=result.completed_at,
            status=status,
            reason=result.reason,
        )
        if status == "HALT":
            raise ReadonlySafetyHalt(result.reason, result=result)
        if include_history:
            await self._save_history_cursor(cursor_ms)
        return result

    async def _run_once(self, kind: CalibrationKind) -> CalibrationResult:
        started_at = self.clock.now()
        cursor_ms = int(started_at.timestamp() * 1000)
        run_id = await self.repository.begin_reconciliation(
            account_id=self.client.account_id, kind=kind, started_at=started_at
        )
        include_history = self._includes_history(kind)
        start_time_ms = (
            await self._history_start_time_ms(started_at)
            if include_history
            else None
        )
        try:
            if self._requires_clock_sync(kind):
                await self.client.synchronize_clock()
            bundle = await self.client.fetch_bundle(
                include_fills=include_history,
                fill_start_time_ms=start_time_ms,
            )
            self.last_bundle = bundle
            result, diffs = await self._apply_bundle(
                run_id, kind, started_at, bundle
            )
            return await self._complete_successful_run(
                run_id,
                result,
                include_history=include_history,
                cursor_ms=cursor_ms,
                diffs=diffs,
            )
        except Exception as exc:
            await self._complete_failed_run(run_id, exc)
            raise

    def _unmanaged_facts(
        self, bundle: BinanceAccountBundle
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        unmanaged_positions = tuple(
            sorted(
                {
                    f"{item.symbol}:{item.position_side}:{item.quantity}"
                    for item in bundle.positions
                    if item.quantity != 0 and item.symbol not in self.managed_symbols
                }
            )
        )
        unmanaged_orders = tuple(
            sorted(
                {
                    f"{item.symbol}:{item.exchange_order_id}"
                    for item in bundle.open_orders
                    if item.active and item.symbol not in self.managed_symbols
                }
            )
        )
        return unmanaged_positions, unmanaged_orders

    def _unmanaged_diffs(
        self,
        unmanaged_positions: Sequence[str],
        unmanaged_orders: Sequence[str],
    ) -> list[ReconciliationDiffFact]:
        diffs = [
            ReconciliationDiffFact(
                account_id=self.client.account_id,
                entity_type="POSITION",
                entity_key=value,
                reason_code=ReadonlyReasonCode.UNMANAGED_POSITION.value,
                expected=None,
                observed={"value": value},
                severity="CRITICAL",
            )
            for value in unmanaged_positions
        ]
        diffs.extend(
            ReconciliationDiffFact(
                account_id=self.client.account_id,
                entity_type="ORDER",
                entity_key=value,
                reason_code=ReadonlyReasonCode.UNMANAGED_ORDER.value,
                expected=None,
                observed={"value": value},
                severity="CRITICAL",
            )
            for value in unmanaged_orders
        )
        return diffs

    def _external_order_diffs(
        self, bundle: BinanceAccountBundle
    ) -> list[ReconciliationDiffFact]:
        return [
            ReconciliationDiffFact(
                account_id=self.client.account_id,
                entity_type="ORDER",
                entity_key=f"{item.symbol}:{item.exchange_order_id}",
                reason_code=ReadonlyReasonCode.EXTERNAL_ORDER.value,
                expected={"origin": "SYSTEM"},
                observed=to_primitive(item),
                severity="WARNING",
                resolution=ReconciliationResolution.QUARANTINED,
                resolution_detail="External order retained as account risk but not managed",
            )
            for item in bundle.open_orders
            if item.active
            and item.symbol in self.managed_symbols
            and item.quarantined
        ]

    @staticmethod
    def _blocking_diffs(
        diffs: Sequence[ReconciliationDiffFact],
    ) -> tuple[ReconciliationDiffFact, ...]:
        return tuple(
            item for item in diffs if item.severity.upper() in {"ERROR", "CRITICAL"}
        )

    @staticmethod
    def _latest_orders(bundle: BinanceAccountBundle) -> tuple[OrderFact, ...]:
        latest: dict[tuple[str, str], OrderFact] = {}
        for item in (*bundle.order_history, *bundle.open_orders):
            key = (item.symbol, item.exchange_order_id)
            previous = latest.get(key)
            if previous is None or item.update_time_ms >= previous.update_time_ms:
                latest[key] = item
        return tuple(latest.values())

    def _configuration_event(
        self, bundle: BinanceAccountBundle, *, completed_at: datetime
    ) -> AccountEventFact:
        event_time_ms = int(completed_at.timestamp() * 1000)
        return AccountEventFact(
            account_id=self.client.account_id,
            event_type="ACCOUNT_CONFIGURATION",
            event_key=stable_fingerprint(bundle.configuration),
            event_time_ms=event_time_ms,
            transaction_time_ms=event_time_ms,
            payload=to_primitive(bundle.configuration),
            observed_at=completed_at,
            source="BINANCE_REST",
        )

    def _income_events(
        self, bundle: BinanceAccountBundle, *, completed_at: datetime
    ) -> tuple[AccountEventFact, ...]:
        facts: list[AccountEventFact] = []
        for item in bundle.income_events:
            event_type = str(item.get("incomeType") or "INCOME").upper()
            event_time_ms = int(item.get("time") or 0)
            currency = str(item.get("asset") or "").upper() or None
            amount = Decimal(str(item.get("income") or "0"))
            transaction_identity = str(
                item.get("tranId") or item.get("tradeId") or ""
            )
            economic_event_id = stable_fingerprint(
                {
                    "account_id": self.client.account_id,
                    "type": event_type,
                    "transaction_identity": transaction_identity,
                    "symbol": str(item.get("symbol") or "").upper(),
                    "currency": currency,
                    "amount": str(amount),
                    "time": event_time_ms,
                }
            )
            facts.append(
                AccountEventFact(
                    account_id=self.client.account_id,
                    event_type=event_type,
                    event_key=economic_event_id,
                    event_time_ms=event_time_ms,
                    transaction_time_ms=event_time_ms,
                    payload=item,
                    observed_at=completed_at,
                    source="BINANCE_REST",
                    currency=currency,
                    amount=amount,
                    economic_event_id=economic_event_id,
                )
            )
        return tuple(facts)

    async def _persist_bundle(
        self,
        run_id: str,
        bundle: BinanceAccountBundle,
        *,
        completed_at: datetime,
        diffs: Sequence[ReconciliationDiffFact],
    ) -> None:
        events = (self._configuration_event(bundle, completed_at=completed_at),) + (
            self._income_events(bundle, completed_at=completed_at)
        )
        batch_writer = getattr(self.repository, "append_exchange_fact_batch", None)
        if callable(batch_writer):
            batch = ExchangeFactBatch(
                account_id=self.client.account_id,
                source="BINANCE_REST",
                observed_at=completed_at,
                reconciliation_run_id=run_id,
                account_snapshot=bundle.account_snapshot,
                balances=tuple(bundle.balances),
                positions=tuple(bundle.positions),
                orders=self._latest_orders(bundle),
                fills=tuple(bundle.fills),
                account_events=events,
                reconciliation_diffs=tuple(diffs),
                correlation_id=run_id,
            )
            await maybe_await(batch_writer(batch))
            self._last_persist_was_atomic = True
            return
        self._last_persist_was_atomic = False
        await self.repository.append_account_snapshot(
            bundle.account_snapshot, reconciliation_run_id=run_id
        )
        await self.repository.append_balance_snapshots(
            bundle.balances, reconciliation_run_id=run_id
        )
        await self.repository.append_position_snapshots(
            bundle.positions, reconciliation_run_id=run_id
        )
        await self.repository.append_order_snapshots(
            self._latest_orders(bundle), reconciliation_run_id=run_id
        )
        if bundle.fills:
            await self.repository.append_fill_events(
                bundle.fills, reconciliation_run_id=run_id
            )
        await self.repository.append_account_events(events)

    @staticmethod
    def _result_reason(
        unmanaged_positions: Sequence[str],
        unmanaged_orders: Sequence[str],
        diffs: Sequence[ReconciliationDiffFact],
    ) -> str:
        if unmanaged_positions:
            return ReadonlyReasonCode.UNMANAGED_POSITION.value
        if unmanaged_orders:
            return ReadonlyReasonCode.UNMANAGED_ORDER.value
        if diffs:
            return ReadonlyReasonCode.RECONCILIATION_DRIFT.value
        return ReadonlyReasonCode.CONSISTENT.value

    async def _compare_account_snapshot(
        self, remote: AccountSnapshotFact
    ) -> list[ReconciliationDiffFact]:
        loader = getattr(self.repository, "load_latest_account_snapshot", None)
        if not callable(loader):
            return []
        local = await maybe_await(loader(self.client.account_id))
        if local is None:
            return []
        fields = (
            "total_wallet_balance",
            "total_available_balance",
            "total_margin_balance",
            "total_initial_margin",
            "total_maintenance_margin",
        )
        diffs: list[ReconciliationDiffFact] = []
        for field_name in fields:
            expected_value = getattr(local, field_name)
            observed_value = getattr(remote, field_name)
            if abs(expected_value - observed_value) <= self.financial_tolerance:
                continue
            diffs.append(
                ReconciliationDiffFact(
                    account_id=self.client.account_id,
                    entity_type="ACCOUNT",
                    entity_key=field_name,
                    reason_code="ACCOUNT_VALUE_MISMATCH",
                    expected={"value": str(expected_value)},
                    observed={"value": str(observed_value)},
                )
            )
        return diffs

    async def _compare_balances(
        self, remote: Sequence[BalanceFact]
    ) -> list[ReconciliationDiffFact]:
        loader = getattr(self.repository, "load_latest_balance_snapshots", None)
        if not callable(loader):
            return []
        local_items = tuple(await maybe_await(loader(self.client.account_id)))
        local = {item.asset: item for item in local_items}
        observed = {item.asset: item for item in remote}
        diffs: list[ReconciliationDiffFact] = []
        for asset in sorted(set(local) | set(observed)):
            expected = local.get(asset)
            current = observed.get(asset)
            if expected is None or current is None:
                diffs.append(
                    ReconciliationDiffFact(
                        account_id=self.client.account_id,
                        entity_type="BALANCE",
                        entity_key=asset,
                        reason_code=(
                            "BALANCE_MISSING_LOCAL"
                            if expected is None
                            else "BALANCE_MISSING_REMOTE"
                        ),
                        expected=None if expected is None else to_primitive(expected),
                        observed=None if current is None else to_primitive(current),
                    )
                )
                continue
            for field_name in (
                "wallet_balance",
                "available_balance",
                "cross_wallet_balance",
            ):
                expected_value = getattr(expected, field_name)
                observed_value = getattr(current, field_name)
                if abs(expected_value - observed_value) <= self.financial_tolerance:
                    continue
                diffs.append(
                    ReconciliationDiffFact(
                        account_id=self.client.account_id,
                        entity_type="BALANCE",
                        entity_key=f"{asset}:{field_name}",
                        reason_code="BALANCE_VALUE_MISMATCH",
                        expected={"value": str(expected_value)},
                        observed={"value": str(observed_value)},
                    )
                )
        return diffs

    async def _apply_bundle(
        self,
        run_id: str,
        kind: CalibrationKind,
        started_at: datetime,
        bundle: BinanceAccountBundle,
    ) -> tuple[CalibrationResult, tuple[ReconciliationDiffFact, ...]]:
        unmanaged_positions, unmanaged_orders = self._unmanaged_facts(bundle)
        local_positions = tuple(
            await self.repository.load_active_positions(self.client.account_id)
        )
        local_orders = tuple(
            await self.repository.load_active_orders(self.client.account_id)
        )
        diffs = self._unmanaged_diffs(unmanaged_positions, unmanaged_orders)
        diffs.extend(self._external_order_diffs(bundle))
        diffs.extend(self._compare_positions(local_positions, bundle.positions))
        diffs.extend(self._compare_orders(local_orders, bundle.open_orders))
        diffs.extend(await self._compare_account_snapshot(bundle.account_snapshot))
        diffs.extend(await self._compare_balances(bundle.balances))
        blocking_diffs = self._blocking_diffs(diffs)
        completed_at = self.clock.now()
        resolved_diffs = tuple(
            ReconciliationDiffFact(
                account_id=item.account_id,
                entity_type=item.entity_type,
                entity_key=item.entity_key,
                reason_code=item.reason_code,
                expected=item.expected,
                observed=item.observed,
                severity=item.severity,
                resolution=(
                    ReconciliationResolution.QUARANTINED
                    if item.reason_code in {
                        ReadonlyReasonCode.UNMANAGED_POSITION.value,
                        ReadonlyReasonCode.UNMANAGED_ORDER.value,
                        ReadonlyReasonCode.EXTERNAL_ORDER.value,
                    }
                    else ReconciliationResolution.OBSERVED
                ),
            )
            for item in diffs
        )
        await self._persist_bundle(
            run_id, bundle, completed_at=completed_at, diffs=resolved_diffs
        )
        diffs = list(resolved_diffs)
        result = CalibrationResult(
            run_id=run_id,
            kind=kind,
            started_at=started_at,
            completed_at=completed_at,
            position_count=len(bundle.positions),
            active_order_count=sum(
                1 for item in bundle.open_orders if item.active
            ),
            fill_count=len(bundle.fills),
            diff_count=len(diffs),
            unmanaged_positions=unmanaged_positions,
            unmanaged_orders=unmanaged_orders,
            consistent=not blocking_diffs,
            reason=self._result_reason(
                unmanaged_positions, unmanaged_orders, blocking_diffs
            ),
        )
        return result, tuple(diffs)

    def _compare_positions(
        self,
        local: Sequence[PositionFact],
        remote: Sequence[PositionFact],
    ) -> list[ReconciliationDiffFact]:
        local_map = _position_map(local)
        remote_map = _position_map(remote)
        diffs: list[ReconciliationDiffFact] = []
        for key in sorted(set(local_map) | set(remote_map)):
            expected = local_map.get(key)
            observed = remote_map.get(key)
            if expected is None:
                code = "POSITION_MISSING_LOCAL"
            elif observed is None:
                code = "POSITION_MISSING_REMOTE"
            elif abs(expected.quantity - observed.quantity) > self.quantity_tolerance:
                code = "POSITION_QUANTITY_MISMATCH"
            elif abs(expected.entry_price - observed.entry_price) > self.financial_tolerance:
                code = "POSITION_ENTRY_PRICE_MISMATCH"
            elif expected.leverage != observed.leverage:
                code = "POSITION_LEVERAGE_MISMATCH"
            elif expected.margin_mode != observed.margin_mode:
                code = "POSITION_MARGIN_MODE_MISMATCH"
            else:
                continue
            diffs.append(ReconciliationDiffFact(
                account_id=self.client.account_id,
                entity_type="POSITION",
                entity_key=f"{key[0]}:{key[1]}",
                reason_code=code,
                expected=None if expected is None else to_primitive(expected),
                observed=None if observed is None else to_primitive(observed),
            ))
        return diffs

    def _compare_orders(
        self,
        local: Sequence[OrderFact],
        remote: Sequence[OrderFact],
    ) -> list[ReconciliationDiffFact]:
        local_map = _order_map(local)
        remote_map = _order_map(remote)
        diffs: list[ReconciliationDiffFact] = []
        for key in sorted(set(local_map) | set(remote_map)):
            expected = local_map.get(key)
            observed = remote_map.get(key)
            if expected is None:
                code = "ORDER_MISSING_LOCAL"
            elif observed is None:
                code = "ORDER_MISSING_REMOTE"
            elif (
                expected.client_order_id,
                expected.position_side,
                expected.side,
            ) != (
                observed.client_order_id,
                observed.position_side,
                observed.side,
            ):
                code = "ORDER_IDENTITY_MISMATCH"
            elif (
                expected.status,
                expected.cumulative_filled_quantity,
            ) != (
                observed.status,
                observed.cumulative_filled_quantity,
            ):
                code = "ORDER_STATE_MISMATCH"
            else:
                continue
            diffs.append(ReconciliationDiffFact(
                account_id=self.client.account_id,
                entity_type="ORDER",
                entity_key=f"{key[0]}:{key[1]}",
                reason_code=code,
                expected=None if expected is None else to_primitive(expected),
                observed=None if observed is None else to_primitive(observed),
            ))
        return diffs
