"""Paper funding observations and durable account-event publication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import logging
from typing import Any, Mapping, Protocol, Sequence

from sqlalchemy import select

from freqtrade.hedge.simulation.exchange import (
    AccountEvent,
    AccountEventType,
    BarEvent,
    FundingEvent,
)
from freqtrade.persistence.hedge_models import (
    AccountEvent as AccountEventRow,
    ExecutionOrderStateRow,
    FillEvent as FillEventRow,
)
from freqtrade.persistence.hedge_service import HedgePersistenceService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PaperAccountEventRecovery:
    event_ids: frozenset[str]
    funding_balance_delta: Decimal
    last_funding_event_time: datetime | None


class PaperAccountEventSink(Protocol):
    def record(self, event: AccountEvent) -> bool: ...

    def recover(self) -> PaperAccountEventRecovery | None: ...


class NullPaperAccountEventSink:
    def record(self, event: AccountEvent) -> bool:
        del event
        return True

    def recover(self) -> PaperAccountEventRecovery | None:
        return None




@dataclass(frozen=True, slots=True)
class RecoveredPaperFill:
    trade_id: str
    client_order_id: str
    position_side: str
    action: str
    quantity: Decimal
    price: Decimal
    fee: Decimal
    fee_currency: str | None
    bucket: str
    event_time: datetime

    def __post_init__(self) -> None:
        trade_id = str(self.trade_id).strip()
        client_order_id = str(self.client_order_id).strip()
        if not trade_id or not client_order_id:
            raise ValueError("recovered Paper fill requires trade and client order ids")
        position_side = str(self.position_side).upper()
        action = str(self.action).upper()
        bucket = str(self.bucket).upper()
        if position_side not in {"LONG", "SHORT"}:
            raise ValueError("recovered Paper fill position_side is invalid")
        if action not in {"OPEN", "INCREASE", "REDUCE", "CLOSE"}:
            raise ValueError("recovered Paper fill action is invalid")
        if bucket not in {"CORE", "TACTICAL"}:
            raise ValueError("recovered Paper fill bucket is invalid")
        for field_name in ("quantity", "price", "fee"):
            value = getattr(self, field_name)
            if not isinstance(value, Decimal) or not value.is_finite():
                raise ValueError(f"recovered Paper fill {field_name} must be finite")
        if self.quantity <= 0 or self.price <= 0 or self.fee < 0:
            raise ValueError("recovered Paper fill quantity/price/fee is invalid")
        observed = self.event_time
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=UTC)
        else:
            observed = observed.astimezone(UTC)
        currency = None if self.fee_currency is None else str(self.fee_currency).strip().upper()
        if currency == "":
            currency = None
        object.__setattr__(self, "trade_id", trade_id)
        object.__setattr__(self, "client_order_id", client_order_id)
        object.__setattr__(self, "position_side", position_side)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "fee_currency", currency)
        object.__setattr__(self, "event_time", observed)


class PaperExecutionRecoveryPort(Protocol):
    def recover_fills(self) -> tuple[RecoveredPaperFill, ...] | None: ...


class NullPaperExecutionRecovery:
    def recover_fills(self) -> tuple[RecoveredPaperFill, ...] | None:
        return None


class SqlPaperExecutionRecovery:
    """Rebuild Paper account/bucket state from immutable SQL fill facts."""

    def __init__(self, session_factory: object, *, account_id: str, symbol: str) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory
        self._account_id = account_id
        self._symbol = symbol

    def recover_fills(self) -> tuple[RecoveredPaperFill, ...]:
        with self._session_factory() as session:  # type: ignore[operator]
            order_rows = tuple(
                session.scalars(
                    select(ExecutionOrderStateRow).where(
                        ExecutionOrderStateRow.account_id == self._account_id,
                        ExecutionOrderStateRow.exchange == "paper",
                        ExecutionOrderStateRow.symbol == self._symbol,
                    )
                )
            )
            metadata_by_client: dict[str, Mapping[str, object]] = {}
            for row in order_rows:
                try:
                    value = json.loads(row.metadata_json or "{}")
                except (TypeError, ValueError) as exc:
                    raise ValueError("invalid execution order metadata JSON") from exc
                metadata_by_client[row.client_order_id] = value if isinstance(value, Mapping) else {}
            rows = tuple(
                session.scalars(
                    select(FillEventRow)
                    .where(
                        FillEventRow.account_id == self._account_id,
                        FillEventRow.exchange == "paper",
                        FillEventRow.symbol == self._symbol,
                    )
                    .order_by(FillEventRow.event_time, FillEventRow.id)
                )
            )
        recovered: list[RecoveredPaperFill] = []
        observed_trade_ids: set[str] = set()
        for row in rows:
            if not row.client_order_id or not row.action:
                raise ValueError("Paper fill ledger row is missing order/action identity")
            if row.exchange_trade_id in observed_trade_ids:
                raise ValueError("duplicate Paper fill trade id in SQL recovery ledger")
            observed_trade_ids.add(row.exchange_trade_id)
            metadata = metadata_by_client.get(row.client_order_id, {})
            bucket = metadata.get("bucket")
            if bucket not in {"CORE", "TACTICAL"}:
                raise ValueError("Paper fill recovery requires canonical order bucket metadata")
            observed = row.event_time
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=UTC)
            else:
                observed = observed.astimezone(UTC)
            recovered.append(
                RecoveredPaperFill(
                    trade_id=row.exchange_trade_id,
                    client_order_id=row.client_order_id,
                    position_side=row.position_side,
                    action=row.action,
                    quantity=Decimal(row.quantity),
                    price=Decimal(row.price),
                    fee=Decimal(row.fee_amount),
                    fee_currency=row.fee_currency,
                    bucket=str(bucket),
                    event_time=observed,
                )
            )
        return tuple(recovered)


class SqlPaperAccountEventSink:
    """Write fee/funding/balance events through the H3 transactional repository."""

    def __init__(
        self,
        persistence: HedgePersistenceService,
        *,
        account_id: str,
        exchange: str,
        symbol: str,
        asset: str = "USDT",
        venue_exchange: str | None = None,
    ) -> None:
        self._persistence = persistence
        self._account_id = account_id
        self._exchange = exchange.strip().lower()
        self._symbol = symbol
        self._asset = asset
        self._venue_exchange = (venue_exchange or "").strip().lower() or None

    def record(self, event: AccountEvent) -> bool:
        if event.event_type not in {AccountEventType.FEE, AccountEventType.FUNDING}:
            raise ValueError(
                "SQL Paper account-event sink only accepts fee and funding events"
            )
        _, created = self._persistence.record_account_event(
            event_key=event.event_id,
            account_id=self._account_id,
            exchange=self._exchange,
            event_type=event.event_type.value,
            asset=self._asset,
            amount=event.amount,
            source="LOCAL",
            event_time=event.timestamp,
            symbol=event.symbol,
            position_side=(
                None if event.position_side is None else event.position_side.value
            ),
            raw_payload={
                "source_event_id": event.source_event_id,
                "description": event.description,
                "source_kind": "PAPER_EXECUTION",
                "venue_exchange": self._venue_exchange,
            },
        )
        return bool(created)

    def recover(self) -> PaperAccountEventRecovery:
        """Rebuild applied event ids and funding cash delta from the SQL fact ledger.

        This closes the crash window where the account event transaction commits
        before the auxiliary Paper checkpoint.  SQL is authoritative whenever
        this adapter is configured; the checkpoint is only a cache.
        """

        with self._persistence.session_factory() as session:
            rows = tuple(
                session.scalars(
                    select(AccountEventRow)
                    .where(
                        AccountEventRow.account_id == self._account_id,
                        AccountEventRow.exchange == self._exchange,
                        AccountEventRow.symbol == self._symbol,
                        AccountEventRow.event_type.in_(("FEE", "FUNDING")),
                    )
                    .order_by(AccountEventRow.event_time, AccountEventRow.id)
                )
            )
        funding_delta = Decimal("0")
        last_funding_time: datetime | None = None
        event_ids: set[str] = set()
        for row in rows:
            event_ids.add(row.event_key)
            if row.event_type != "FUNDING":
                continue
            funding_delta += Decimal(row.amount)
            observed = row.event_time
            if observed.tzinfo is None:
                observed = observed.replace(tzinfo=UTC)
            else:
                observed = observed.astimezone(UTC)
            if last_funding_time is None or observed > last_funding_time:
                last_funding_time = observed
        return PaperAccountEventRecovery(
            event_ids=frozenset(event_ids),
            funding_balance_delta=funding_delta,
            last_funding_event_time=last_funding_time,
        )


@dataclass(frozen=True, slots=True)
class FundingCollection:
    events: tuple[FundingEvent, ...]
    source: str
    observed_at: datetime


@dataclass(frozen=True, slots=True)
class FundingProviderState:
    last_seen_ms: int
    bootstrapped: bool
    pending: tuple[tuple[int, FundingEvent], ...]
    last_success_at: datetime | None
    last_poll_at: datetime | None


def _decimal(value: object) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() else None


def _timestamp_ms(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return int(aware.timestamp() * 1000)
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    if result <= 0:
        return None
    # Some adapters expose seconds while CCXT uses milliseconds.
    return result * 1000 if result < 10_000_000_000 else result


def _event_id(symbol: str, timestamp_ms: int) -> str:
    # A settlement is identified by venue symbol and settlement timestamp.
    # The announced rate may be revised before settlement and must never create
    # a second debit for the same funding interval.
    digest = sha256(f"{symbol}|{timestamp_ms}".encode()).hexdigest()[:24]
    return f"funding-{digest}"


class ExchangeFundingEventProvider:
    """Collect actual or scheduled funding events from the exchange adapter.

    Preferred evidence is ``fetch_funding_rate_history``.  ``fetch_funding_rate``
    is used to retain the next announced settlement and rate, then emit it only
    after a closed DataProvider candle crosses that settlement timestamp.  A
    predicted rate is never charged immediately on every process loop.
    """

    def __init__(
        self,
        exchange: Any,
        *,
        symbol: str,
        max_age_seconds: int = 3600,
        poll_interval_seconds: int = 60,
        initial_since_ms: int | None = None,
    ) -> None:
        if max_age_seconds <= 0:
            raise ValueError("funding max age must be positive")
        if poll_interval_seconds < 0:
            raise ValueError("funding poll interval cannot be negative")
        self._exchange = exchange
        self._symbol = symbol
        self._max_age_seconds = max_age_seconds
        self._poll_interval_seconds = poll_interval_seconds
        raw_api = getattr(exchange, "_api", None)
        self._history_fetch = getattr(exchange, "fetch_funding_rate_history", None)
        if not callable(self._history_fetch) and raw_api is not None:
            self._history_fetch = getattr(raw_api, "fetch_funding_rate_history", None)
        self._current_fetch = getattr(exchange, "fetch_funding_rate", None)
        if not callable(self._current_fetch) and raw_api is not None:
            self._current_fetch = getattr(raw_api, "fetch_funding_rate", None)
        if not callable(self._history_fetch) and not callable(self._current_fetch):
            raise RuntimeError(
                "exchange adapter exposes neither funding-rate history nor current funding rate"
            )
        if initial_since_ms is not None and initial_since_ms < 0:
            raise ValueError("initial funding cursor cannot be negative")
        self._last_seen_ms = int(initial_since_ms or 0)
        self._bootstrapped = initial_since_ms is not None
        self._pending: dict[int, FundingEvent] = {}
        self._last_success_at: datetime | None = None
        self._last_poll_at: datetime | None = None

    @property
    def last_success_at(self) -> datetime | None:
        return self._last_success_at

    def snapshot_state(self) -> FundingProviderState:
        return FundingProviderState(
            last_seen_ms=self._last_seen_ms,
            bootstrapped=self._bootstrapped,
            pending=tuple(sorted(self._pending.items())),
            last_success_at=self._last_success_at,
            last_poll_at=self._last_poll_at,
        )

    def restore_state(self, state: FundingProviderState) -> None:
        if not isinstance(state, FundingProviderState):
            raise TypeError("funding provider state token is invalid")
        self._last_seen_ms = state.last_seen_ms
        self._bootstrapped = state.bootstrapped
        self._pending = dict(state.pending)
        self._last_success_at = state.last_success_at
        self._last_poll_at = state.last_poll_at

    def _from_payload(
        self,
        payload: Mapping[str, Any],
        *,
        fallback_mark: Decimal,
        timestamp_keys: Sequence[str],
        rate_keys: Sequence[str] = ("fundingRate", "rate"),
    ) -> FundingEvent | None:
        timestamp_ms = next(
            (
                value
                for key in timestamp_keys
                if (value := _timestamp_ms(payload.get(key))) is not None
            ),
            None,
        )
        rate = next(
            (
                value
                for key in rate_keys
                if (value := _decimal(payload.get(key))) is not None
            ),
            None,
        )
        if timestamp_ms is None or rate is None:
            return None
        mark = _decimal(payload.get("markPrice")) or _decimal(payload.get("mark_price"))
        mark = mark or fallback_mark
        return FundingEvent(
            timestamp=datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC),
            symbol=self._symbol,
            rate=rate,
            mark_price=mark,
        )

    def _history(self, bar: BarEvent) -> tuple[list[FundingEvent], bool]:
        fetch = self._history_fetch
        if not callable(fetch):
            return [], False
        try:
            rows = fetch(
                self._symbol,
                since=(self._last_seen_ms + 1 if self._last_seen_ms else None),
                limit=32,
            )
        except NotImplementedError:
            return [], False
        if rows is None:
            return [], True
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            raise RuntimeError("exchange funding history must be a sequence")
        result: list[FundingEvent] = []
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            event = self._from_payload(
                raw,
                fallback_mark=bar.close,
                timestamp_keys=("timestamp", "fundingTimestamp", "fundingTime"),
            )
            if event is None:
                continue
            timestamp_ms = int(event.timestamp.timestamp() * 1000)
            if event.timestamp <= bar.timestamp:
                result.append(event)
            else:
                # Some adapters include the announced next settlement in a
                # history-shaped response. Keep it pending so REST throttling
                # cannot make the closed-candle boundary miss the settlement.
                self._pending[timestamp_ms] = event
        return result, True

    def _current(self, bar: BarEvent) -> tuple[list[FundingEvent], bool]:
        fetch = self._current_fetch
        if not callable(fetch):
            return [], False
        payload = fetch(self._symbol)
        if not isinstance(payload, Mapping):
            raise RuntimeError("exchange funding-rate response must be a mapping")
        result: list[FundingEvent] = []
        actual = self._from_payload(
            payload,
            fallback_mark=bar.close,
            timestamp_keys=("fundingTimestamp", "fundingTime"),
        )
        if actual is not None and actual.timestamp <= bar.timestamp:
            result.append(actual)
        pending = self._from_payload(
            payload,
            fallback_mark=bar.close,
            timestamp_keys=("nextFundingTimestamp", "nextFundingTime"),
            rate_keys=("nextFundingRate", "fundingRate", "rate"),
        )
        if pending is not None:
            self._pending[int(pending.timestamp.timestamp() * 1000)] = pending
        return result, True

    def collect(self, bar: BarEvent) -> FundingCollection:
        # A new Paper run must never charge arbitrary historical settlements
        # returned by an exchange's default history window.  Bootstrap the
        # cursor to the configured freshness window; recovered runs pass their
        # durable last-applied settlement explicitly.
        if not self._bootstrapped:
            cutoff = bar.timestamp.timestamp() * 1000 - self._max_age_seconds * 1000
            self._last_seen_ms = max(0, int(cutoff))
            self._bootstrapped = True
        observed_at = datetime.now(UTC)
        should_poll = (
            self._last_poll_at is None
            or self._poll_interval_seconds == 0
            or (observed_at - self._last_poll_at).total_seconds()
            >= self._poll_interval_seconds
        )
        candidates: list[FundingEvent] = []
        if should_poll:
            history_events, history_ok = self._history(bar)
            current_events, current_ok = self._current(bar)
            if not (history_ok or current_ok):
                raise RuntimeError(
                    "exchange funding capabilities are present but unsupported at runtime"
                )
            candidates.extend(history_events)
            candidates.extend(current_events)
            self._last_poll_at = observed_at
            self._last_success_at = observed_at
        # Pending settlements are evaluated on every closed candle even when the
        # network poll is throttled.  This preserves settlement timing without
        # issuing REST requests on every Freqtrade process loop.
        for timestamp_ms, pending in tuple(self._pending.items()):
            if pending.timestamp <= bar.timestamp:
                candidates.append(pending)
                self._pending.pop(timestamp_ms, None)

        unique: dict[str, FundingEvent] = {}
        for event in candidates:
            timestamp_ms = int(event.timestamp.timestamp() * 1000)
            if timestamp_ms <= self._last_seen_ms:
                continue
            unique.setdefault(_event_id(self._symbol, timestamp_ms), event)
        events = tuple(sorted(unique.values(), key=lambda item: item.timestamp))
        if events:
            self._last_seen_ms = max(
                self._last_seen_ms,
                max(int(item.timestamp.timestamp() * 1000) for item in events),
            )
        if self._last_success_at is None:
            # Constructor guarantees at least one exchange capability, so the
            # first call always polls.  Keep this defensive guard fail-closed.
            raise RuntimeError("funding source has not completed a successful poll")
        return FundingCollection(
            events=events,
            source=(
                "EXCHANGE_FUNDING_HISTORY_OR_SCHEDULE"
                if should_poll
                else "EXCHANGE_FUNDING_CACHE"
            ),
            observed_at=self._last_success_at,
        )

    def healthy(self, *, now: datetime | None = None) -> bool:
        if self._last_success_at is None:
            return False
        current = now or datetime.now(UTC)
        return (current - self._last_success_at).total_seconds() <= self._max_age_seconds


def fee_account_event(
    *,
    fill_event_id: str,
    timestamp: datetime,
    symbol: str,
    amount: Decimal,
    position_side: object | None,
) -> AccountEvent:
    return AccountEvent(
        event_id=f"paper-fee:{fill_event_id}",
        timestamp=timestamp,
        symbol=symbol,
        event_type=AccountEventType.FEE,
        amount=-amount,
        position_side=position_side,
        source_event_id=fill_event_id,
        description="Paper maker/taker fee",
    )


def funding_account_event(
    *,
    funding: FundingEvent,
    amount: Decimal,
) -> AccountEvent:
    timestamp_ms = int(funding.timestamp.timestamp() * 1000)
    return AccountEvent(
        event_id=_event_id(funding.symbol, timestamp_ms),
        timestamp=funding.timestamp,
        symbol=funding.symbol,
        event_type=AccountEventType.FUNDING,
        amount=amount,
        source_event_id=str(timestamp_ms),
        description=f"Paper funding at rate {funding.rate}",
    )
