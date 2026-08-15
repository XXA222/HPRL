from __future__ import annotations

import asyncio
import json
import logging
import math
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol
from urllib.parse import urlsplit

from .base import (
    AccountEventFact,
    BalanceFact,
    AsyncCallback,
    Clock,
    EventDisposition,
    ExchangeFactBatch,
    FillFact,
    OrderFact,
    PositionFact,
    ReadonlyFactRepository,
    StreamHealth,
    SystemClock,
    maybe_await,
    stable_fingerprint,
)
from .binance_normalizer import (
    finite_decimal,
    normalize_account_update_balances,
    normalize_fill,
    normalize_order,
    normalize_position,
    normalize_position_side,
)
from .listen_key import ListenKeyManager
from .rate_limit import BinanceDataError
from .symbol_codec import normalize_exchange_symbols


logger = logging.getLogger(__name__)


class WebSocketConnection(Protocol):
    def __aiter__(self) -> AsyncIterator[str | bytes]: ...

    async def close(self) -> None: ...


class WebSocketConnector(Protocol):
    async def connect(self, url: str) -> WebSocketConnection: ...


def _proxy_url(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().rstrip("/")
    if not normalized:
        return None
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("websocket proxy must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError("websocket proxy must not embed credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError("websocket proxy must not contain path, query, or fragment")
    return normalized


class DefaultWebSocketConnector:
    def __init__(
        self,
        *,
        proxy_url: str | None = None,
        trust_env_proxy: bool = False,
    ) -> None:
        self._proxy_url = _proxy_url(proxy_url)
        if not isinstance(trust_env_proxy, bool):
            raise ValueError("trust_env_proxy must be a boolean")
        self._trust_env_proxy = trust_env_proxy

    async def connect(self, url: str) -> WebSocketConnection:
        import websockets

        kwargs: dict[str, Any] = {
            "open_timeout": 15,
            "close_timeout": 10,
            "ping_interval": 20,
            "ping_timeout": 20,
            "max_queue": 2048,
        }
        if self._proxy_url is not None:
            kwargs["proxy"] = self._proxy_url
        elif not self._trust_env_proxy:
            # websockets 15+ discovers OS/environment proxies by default.
            # Explicitly disable this unless the runtime opts in.
            kwargs["proxy"] = None
        return await websockets.connect(url, **kwargs)


def _managed_symbol_set(values: Sequence[str]) -> frozenset[str]:
    return frozenset(normalize_exchange_symbols(list(values)))


def _reconnect_delays(minimum: float, maximum: float) -> tuple[float, float]:
    if not math.isfinite(minimum) or minimum <= 0:
        raise ValueError("min_reconnect_delay_seconds must be finite and positive")
    if not math.isfinite(maximum):
        raise ValueError("max_reconnect_delay_seconds must be finite")
    if maximum < minimum:
        raise ValueError(
            "max_reconnect_delay_seconds must be >= min_reconnect_delay_seconds"
        )
    return float(minimum), float(maximum)


def _websocket_base_url(value: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if not parsed.hostname:
        raise ValueError("websocket_base_url must be an absolute URL")
    if parsed.username or parsed.password:
        raise ValueError("websocket_base_url must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("websocket_base_url must not contain query or fragment")
    if parsed.scheme != "wss" and not (
        parsed.scheme == "ws" and parsed.hostname in local_hosts
    ):
        raise ValueError(
            "websocket_base_url must use WSS except for a local test server"
        )
    return normalized


@dataclass(frozen=True, slots=True)
class SequencingDecision:
    disposition: EventDisposition
    reason: str
    entity_key: str


@dataclass(frozen=True, slots=True)
class _PreparedPositionUpdate:
    position: PositionFact
    entity_key: str
    transaction_time_ms: int
    event_time_ms: int
    state_signature: str


class _StreamIntegrityError(BinanceDataError):
    def __init__(
        self, reason: str, *, disposition: EventDisposition = EventDisposition.GAP
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.disposition = disposition


class EventDeduplicator:
    """Bounded in-memory dedupe cache.

    Membership and commit are deliberately separate. A fingerprint is committed
    only after all Repository writes for the event have succeeded.
    """

    def __init__(
        self,
        *,
        max_entries: int = 100_000,
        ttl_seconds: float = 72 * 3600,
        clock: Clock | None = None,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")
        if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be finite and positive")
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock or SystemClock()
        self._seen: OrderedDict[str, float] = OrderedDict()

    def _prune(self, now: float) -> None:
        cutoff = now - self._ttl_seconds
        while self._seen:
            first_key = next(iter(self._seen))
            if self._seen[first_key] >= cutoff:
                break
            self._seen.popitem(last=False)

    def contains(self, fingerprint: str) -> bool:
        now = self._clock.monotonic()
        self._prune(now)
        if fingerprint not in self._seen:
            return False
        self._seen.move_to_end(fingerprint)
        return True

    def add(self, fingerprint: str) -> None:
        now = self._clock.monotonic()
        self._prune(now)
        self._seen[fingerprint] = now
        self._seen.move_to_end(fingerprint)
        while len(self._seen) > self._max_entries:
            self._seen.popitem(last=False)

    def check_and_add(self, fingerprint: str) -> bool:
        """Compatibility helper. New stream code uses contains/add transactionally."""
        if self.contains(fingerprint):
            return True
        self.add(fingerprint)
        return False


class BinanceEventSequencer:
    """Detect regressions, same-time conflicts and cumulative-fill gaps."""

    def __init__(self) -> None:
        self._last_sequence: dict[str, tuple[int, int]] = {}
        self._last_state_signature: dict[str, str] = {}
        self._last_cumulative_fill: dict[str, Decimal] = {}
        self._rest_baseline_entities: set[str] = set()

    def reset(self) -> None:
        self._last_sequence.clear()
        self._last_state_signature.clear()
        self._last_cumulative_fill.clear()
        self._rest_baseline_entities.clear()

    def seed_position(
        self,
        entity_key: str,
        transaction_time_ms: int,
        *,
        state_signature: str = "",
    ) -> None:
        sequence = int(transaction_time_ms)
        self._last_sequence[entity_key] = (sequence, sequence)
        self._rest_baseline_entities.add(entity_key)
        if state_signature:
            self._last_state_signature[entity_key] = state_signature

    def seed_order(
        self,
        entity_key: str,
        transaction_time_ms: int,
        cumulative_fill: Decimal,
        *,
        state_signature: str = "",
    ) -> None:
        sequence = int(transaction_time_ms)
        self._last_sequence[entity_key] = (sequence, sequence)
        self._rest_baseline_entities.add(entity_key)
        self._last_cumulative_fill[entity_key] = cumulative_fill
        if state_signature:
            self._last_state_signature[entity_key] = state_signature

    def _inspect_sequence(
        self,
        entity_key: str,
        *,
        transaction_time_ms: int,
        event_time_ms: int,
        state_signature: str,
        regression_reason: str,
        conflict_reason: str,
        allow_same_time_state_change: bool = False,
    ) -> SequencingDecision | None:
        current = (int(transaction_time_ms), int(event_time_ms))
        previous = self._last_sequence.get(entity_key)
        if previous is not None and current < previous:
            if entity_key in self._rest_baseline_entities:
                return SequencingDecision(
                    EventDisposition.DUPLICATE,
                    "EVENT_COVERED_BY_REST_BASELINE",
                    entity_key,
                )
            return SequencingDecision(
                EventDisposition.OUT_OF_ORDER,
                regression_reason,
                entity_key,
            )
        previous_signature = self._last_state_signature.get(entity_key)
        if (
            previous is not None
            and current == previous
            and previous_signature
            and previous_signature != state_signature
            and not allow_same_time_state_change
        ):
            if entity_key in self._rest_baseline_entities:
                return SequencingDecision(
                    EventDisposition.DUPLICATE,
                    "EVENT_COVERED_BY_REST_BASELINE",
                    entity_key,
                )
            return SequencingDecision(
                EventDisposition.GAP,
                conflict_reason,
                entity_key,
            )
        return None

    def inspect_account_position(
        self,
        entity_key: str,
        *,
        transaction_time_ms: int,
        event_time_ms: int,
        state_signature: str,
    ) -> SequencingDecision:
        failure = self._inspect_sequence(
            entity_key,
            transaction_time_ms=transaction_time_ms,
            event_time_ms=event_time_ms,
            state_signature=state_signature,
            regression_reason="POSITION_EVENT_SEQUENCE_REGRESSION",
            conflict_reason="POSITION_SAME_TIME_CONFLICT",
        )
        return failure or SequencingDecision(
            EventDisposition.APPLY, "OK", entity_key
        )

    def commit_position(
        self,
        entity_key: str,
        *,
        transaction_time_ms: int,
        event_time_ms: int,
        state_signature: str,
    ) -> None:
        self._last_sequence[entity_key] = (
            int(transaction_time_ms),
            int(event_time_ms),
        )
        self._last_state_signature[entity_key] = state_signature
        self._rest_baseline_entities.discard(entity_key)

    def inspect_order(
        self,
        entity_key: str,
        *,
        transaction_time_ms: int,
        event_time_ms: int,
        cumulative_fill: Decimal,
        last_fill_quantity: Decimal,
        state_signature: str,
    ) -> SequencingDecision:
        failure = self._inspect_sequence(
            entity_key,
            transaction_time_ms=transaction_time_ms,
            event_time_ms=event_time_ms,
            state_signature=state_signature,
            regression_reason="ORDER_EVENT_SEQUENCE_REGRESSION",
            conflict_reason="ORDER_SAME_TIME_CONFLICT",
            allow_same_time_state_change=True,
        )
        if failure is not None:
            return failure
        previous_fill = self._last_cumulative_fill.get(entity_key)
        previous_sequence = self._last_sequence.get(entity_key)
        previous_signature = self._last_state_signature.get(entity_key)
        same_time_changed = (
            previous_sequence is not None
            and previous_sequence == (int(transaction_time_ms), int(event_time_ms))
            and bool(previous_signature)
            and previous_signature != state_signature
        )
        if entity_key in self._rest_baseline_entities and previous_fill is not None:
            previous_signature = self._last_state_signature.get(entity_key)
            if cumulative_fill < previous_fill or (
                cumulative_fill == previous_fill
                and previous_signature == state_signature
            ):
                return SequencingDecision(
                    EventDisposition.DUPLICATE,
                    "EVENT_COVERED_BY_REST_BASELINE",
                    entity_key,
                )
        if previous_fill is not None:
            if cumulative_fill < previous_fill:
                return SequencingDecision(
                    EventDisposition.OUT_OF_ORDER,
                    "ORDER_CUMULATIVE_FILL_REGRESSION",
                    entity_key,
                )
            observed_delta = cumulative_fill - previous_fill
            if observed_delta > last_fill_quantity and observed_delta > 0:
                return SequencingDecision(
                    EventDisposition.GAP,
                    "ORDER_CUMULATIVE_FILL_GAP",
                    entity_key,
                )
            if same_time_changed and observed_delta <= 0:
                if entity_key in self._rest_baseline_entities:
                    return SequencingDecision(
                        EventDisposition.DUPLICATE,
                        "EVENT_COVERED_BY_REST_BASELINE",
                        entity_key,
                    )
                return SequencingDecision(
                    EventDisposition.GAP,
                    "ORDER_SAME_TIME_CONFLICT",
                    entity_key,
                )
        elif same_time_changed:
            return SequencingDecision(
                EventDisposition.GAP,
                "ORDER_SAME_TIME_CONFLICT",
                entity_key,
            )
        return SequencingDecision(EventDisposition.APPLY, "OK", entity_key)

    def commit_order(
        self,
        entity_key: str,
        *,
        transaction_time_ms: int,
        event_time_ms: int,
        cumulative_fill: Decimal,
        state_signature: str,
    ) -> None:
        self._last_sequence[entity_key] = (
            int(transaction_time_ms),
            int(event_time_ms),
        )
        self._last_cumulative_fill[entity_key] = cumulative_fill
        self._last_state_signature[entity_key] = state_signature
        self._rest_baseline_entities.discard(entity_key)


class ListenKeyExpired(RuntimeError):
    pass


def _event_milliseconds(payload: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        value = payload.get(key)
        if value in (None, ""):
            continue
        try:
            parsed = Decimal(str(value))
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise BinanceDataError(f"{key} must be an integer millisecond timestamp") from exc
        if not parsed.is_finite() or parsed != parsed.to_integral_value() or parsed < 0:
            raise BinanceDataError(f"{key} must be a nonnegative integer millisecond timestamp")
        return int(parsed)
    return 0


def _position_state_signature(item: PositionFact) -> str:
    return stable_fingerprint(
        {
            "symbol": item.symbol,
            "position_side": item.position_side,
            "quantity": item.quantity,
            "entry_price": item.entry_price,
            "mark_price": item.mark_price,
            "unrealized_pnl": item.unrealized_pnl,
            "liquidation_price": item.liquidation_price,
            "leverage": item.leverage,
            "margin_mode": item.margin_mode,
        }
    )


def _order_state_signature(item: OrderFact) -> str:
    return stable_fingerprint(
        {
            "exchange_order_id": item.exchange_order_id,
            "status": item.status,
            "original_quantity": item.original_quantity,
            "cumulative_filled_quantity": item.cumulative_filled_quantity,
            "average_price": item.average_price,
            "reduce_only": item.reduce_only,
        }
    )


class BinanceUserStream:
    """Real Binance Futures user stream with fail-closed reconnect semantics."""

    def __init__(
        self,
        *,
        account_id: str,
        managed_symbols: tuple[str, ...] | list[str],
        repository: ReadonlyFactRepository,
        listen_keys: ListenKeyManager,
        connector: WebSocketConnector | None = None,
        websocket_proxy_url: str | None = None,
        trust_env_proxy: bool = False,
        websocket_base_url: str = "wss://fstream.binance.com/ws",
        clock: Clock | None = None,
        on_connected: AsyncCallback | None = None,
        on_disconnected: AsyncCallback | None = None,
        on_integrity_fault: AsyncCallback | None = None,
        on_recalibration_required: AsyncCallback | None = None,
        on_event: AsyncCallback | None = None,
        on_execution_order_event: AsyncCallback | None = None,
        min_reconnect_delay_seconds: float = 1.0,
        max_reconnect_delay_seconds: float = 60.0,
        reconnect_reset_after_seconds: float = 30.0,
        system_client_order_prefixes: Sequence[str] = ("fthedge-",),
    ) -> None:
        normalized_account_id = account_id.strip()
        if not normalized_account_id:
            raise ValueError("account_id is required")
        normalized_symbols = _managed_symbol_set(managed_symbols)
        minimum_delay, maximum_delay = _reconnect_delays(
            min_reconnect_delay_seconds, max_reconnect_delay_seconds
        )
        normalized_ws_url = _websocket_base_url(websocket_base_url)
        if (
            not math.isfinite(reconnect_reset_after_seconds)
            or reconnect_reset_after_seconds < 0
        ):
            raise ValueError(
                "reconnect_reset_after_seconds must be finite and nonnegative"
            )

        self._account_id = normalized_account_id
        self._managed_symbols = normalized_symbols
        self._system_client_order_prefixes = tuple(
            prefix.strip() for prefix in system_client_order_prefixes if prefix.strip()
        )
        self._repository = repository
        self._listen_keys = listen_keys
        self._connector = connector or DefaultWebSocketConnector(
            proxy_url=websocket_proxy_url,
            trust_env_proxy=trust_env_proxy,
        )
        self._base_url = normalized_ws_url
        self._clock = clock or SystemClock()
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._on_integrity_fault = on_integrity_fault
        self._on_recalibration_required = on_recalibration_required
        self._on_event = on_event
        self._on_execution_order_event = on_execution_order_event
        self._min_reconnect = minimum_delay
        self._max_reconnect = maximum_delay
        self._reconnect_reset_after = float(reconnect_reset_after_seconds)
        self._dedupe = EventDeduplicator(clock=self._clock)
        self._sequencer = BinanceEventSequencer()
        self._rest_position_context: dict[tuple[str, str], PositionFact] = {}
        self._rest_balance_context: dict[str, BalanceFact] = {}
        self._rest_order_context: dict[tuple[str, str], OrderFact] = {}
        self._state_revision = 0
        self._state_observed_at: datetime | None = None
        self._connected = False
        self._last_connected_at: datetime | None = None
        self._last_event_at: datetime | None = None
        self._last_calibration_at: datetime | None = None
        self._reconnect_count = 0
        self._duplicate_count = 0
        self._out_of_order_count = 0
        self._gap_count = 0
        self._stop_event = asyncio.Event()
        self._connection: WebSocketConnection | None = None
        self._event_state_lock = asyncio.Lock()
        self._current_connection_message_count = 0
        setter = getattr(self._listen_keys, "set_on_rebuilt", None)
        if callable(setter):
            setter(self._on_listen_key_rebuilt)

    @property
    def listen_key_lease(self):
        return self._listen_keys.lease

    @property
    def health(self) -> StreamHealth:
        return StreamHealth(
            connected=self._connected,
            last_connected_at=self._last_connected_at,
            last_event_at=self._last_event_at,
            last_calibration_at=self._last_calibration_at,
            reconnect_count=self._reconnect_count,
            duplicate_count=self._duplicate_count,
            out_of_order_count=self._out_of_order_count,
            gap_count=self._gap_count,
        )

    @property
    def state_revision(self) -> int:
        return self._state_revision

    @property
    def state_observed_at(self) -> datetime | None:
        return self._state_observed_at

    @property
    def current_positions(self) -> tuple[PositionFact, ...]:
        return tuple(
            sorted(
                self._rest_position_context.values(),
                key=lambda item: (item.symbol, item.position_side),
            )
        )

    @property
    def current_balances(self) -> tuple[BalanceFact, ...]:
        return tuple(
            sorted(self._rest_balance_context.values(), key=lambda item: item.asset)
        )

    @property
    def current_active_orders(self) -> tuple[OrderFact, ...]:
        return tuple(
            sorted(
                (item for item in self._rest_order_context.values() if item.active),
                key=lambda item: (item.symbol, item.exchange_order_id),
            )
        )

    def _touch_state(self) -> None:
        self._state_revision += 1
        self._state_observed_at = self._clock.now()

    def set_callbacks(
        self,
        *,
        on_connected: AsyncCallback | None = None,
        on_disconnected: AsyncCallback | None = None,
        on_integrity_fault: AsyncCallback | None = None,
        on_recalibration_required: AsyncCallback | None = None,
        on_event: AsyncCallback | None = None,
        on_execution_order_event: AsyncCallback | None = None,
    ) -> None:
        self._on_connected = on_connected
        self._on_disconnected = on_disconnected
        self._on_integrity_fault = on_integrity_fault
        self._on_recalibration_required = on_recalibration_required
        self._on_event = on_event
        self._on_execution_order_event = on_execution_order_event

    def set_execution_order_event_callback(
        self, callback: AsyncCallback | None
    ) -> None:
        """Bind the execution lifecycle bridge without replacing health callbacks."""

        self._on_execution_order_event = callback

    def _rest_position_rows(
        self, positions: Sequence[PositionFact]
    ) -> dict[tuple[str, str], PositionFact]:
        rows: dict[tuple[str, str], PositionFact] = {}
        for item in positions:
            if item.symbol not in self._managed_symbols:
                if item.quantity != 0:
                    raise BinanceDataError(
                        f"REST seed contains unmanaged position: {item.symbol}"
                    )
                continue
            key = (item.symbol, item.position_side)
            if key in rows:
                raise BinanceDataError(
                    "REST seed contains duplicate position: "
                    f"{item.symbol}:{item.position_side}"
                )
            rows[key] = item
        return rows

    def _validate_rest_orders(self, orders: Sequence[OrderFact]) -> None:
        rows: set[tuple[str, str]] = set()
        for item in orders:
            if item.active and item.symbol not in self._managed_symbols:
                raise BinanceDataError(
                    "REST seed contains unmanaged active order: "
                    f"{item.symbol}:{item.exchange_order_id}"
                )
            key = (item.symbol, item.exchange_order_id)
            if key in rows:
                raise BinanceDataError(
                    "REST seed contains duplicate order: "
                    f"{item.symbol}:{item.exchange_order_id}"
                )
            rows.add(key)

    def _rest_balance_rows(
        self, balances: Sequence[BalanceFact]
    ) -> dict[str, BalanceFact]:
        rows: dict[str, BalanceFact] = {}
        for item in balances:
            if item.account_id != self._account_id:
                raise BinanceDataError(
                    f"REST seed balance belongs to another account: {item.asset}"
                )
            if item.asset in rows:
                raise BinanceDataError(
                    f"REST seed contains duplicate balance asset: {item.asset}"
                )
            rows[item.asset] = item
        return rows

    def _seed_from_rest_unlocked(
        self,
        positions: Sequence[PositionFact],
        orders: Sequence[OrderFact],
        balances: Sequence[BalanceFact] = (),
    ) -> None:
        position_rows = self._rest_position_rows(positions)
        self._validate_rest_orders(orders)
        balance_rows = self._rest_balance_rows(balances)

        self._sequencer.reset()
        self._rest_position_context = dict(position_rows)
        self._rest_balance_context = balance_rows
        self._rest_order_context = {
            (item.symbol, item.exchange_order_id): item
            for item in orders
            if item.active
        }
        for item in position_rows.values():
            self._sequencer.seed_position(
                f"POSITION:{item.symbol}:{item.position_side}",
                item.update_time_ms,
                state_signature=_position_state_signature(item),
            )
        for item in orders:
            self._sequencer.seed_order(
                f"ORDER:{item.symbol}:{item.exchange_order_id}",
                item.update_time_ms,
                item.cumulative_filled_quantity,
                state_signature=_order_state_signature(item),
            )
        self._touch_state()

    def seed_from_rest(
        self,
        positions: Sequence[PositionFact],
        orders: Sequence[OrderFact],
        balances: Sequence[BalanceFact] = (),
    ) -> None:
        """Seed before the stream task starts. Use ``reseed_from_rest`` at runtime."""
        if self._event_state_lock.locked():
            raise RuntimeError("Cannot synchronously seed while an event is being processed")
        self._seed_from_rest_unlocked(positions, orders, balances)

    async def reseed_from_rest(
        self,
        positions: Sequence[PositionFact],
        orders: Sequence[OrderFact],
        balances: Sequence[BalanceFact] = (),
    ) -> None:
        async with self._event_state_lock:
            self._seed_from_rest_unlocked(positions, orders, balances)

    def mark_calibrated(self, at: datetime | None = None) -> None:
        self._last_calibration_at = at or self._clock.now()

    async def _on_listen_key_rebuilt(self, _lease: Any) -> None:
        await self._close_current_connection()

    async def run_listen_key_renewal(self, stop_event: asyncio.Event) -> None:
        await self._listen_keys.run_renewal_loop(stop_event)

    async def request_reconnect(self) -> None:
        """Close the current socket so the normal reconnect path can recalibrate."""
        await self._close_current_connection()

    async def stop(self) -> None:
        self._stop_event.set()
        await self._close_current_connection()
        await self._listen_keys.close()

    async def _safe_close(self, connection: WebSocketConnection | None) -> None:
        if connection is None:
            return
        try:
            await connection.close()
        except Exception:
            logger.exception("Failed to close Binance user stream connection")

    async def _close_current_connection(self) -> None:
        await self._safe_close(self._connection)

    async def _notify_callback_safely(
        self, callback: AsyncCallback | None, *args: Any
    ) -> None:
        if callback is None:
            return
        try:
            await maybe_await(callback(*args))
        except Exception:
            logger.exception("Binance user stream callback failed")

    async def _consume_connection(
        self, connection: WebSocketConnection, *, generation: int
    ) -> None:
        self._connection = connection
        self._current_connection_message_count = 0
        self._connected = True
        self._last_connected_at = self._clock.now()
        if self._on_connected is not None:
            await maybe_await(self._on_connected(generation))
        async for raw_message in connection:
            if self._stop_event.is_set():
                break
            await self.process_message(raw_message)
            self._current_connection_message_count += 1

    async def _disconnect(self, connection: WebSocketConnection | None) -> None:
        was_connected = self._connected
        self._connected = False
        self._connection = None
        await self._safe_close(connection)
        if was_connected and not self._stop_event.is_set():
            self._reconnect_count += 1
            await self._notify_callback_safely(self._on_disconnected)

    async def run(self) -> None:
        delay = self._min_reconnect
        while not self._stop_event.is_set():
            connection: WebSocketConnection | None = None
            connected_started: float | None = None
            self._current_connection_message_count = 0
            try:
                lease = await self._listen_keys.ensure()
                connection = await self._connector.connect(
                    f"{self._base_url}/{lease.listen_key}"
                )
                connected_started = self._clock.monotonic()
                await self._consume_connection(
                    connection, generation=lease.generation
                )
            except ListenKeyExpired:
                await self._listen_keys.force_rebuild()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self._notify_callback_safely(
                    self._on_integrity_fault, "STREAM_TRANSPORT_ERROR", exc
                )
            finally:
                await self._disconnect(connection)
            if self._stop_event.is_set():
                break
            connected_duration = 0.0
            if connected_started is not None:
                connected_duration = max(
                    0.0, self._clock.monotonic() - connected_started
                )
            stable_connection = (
                self._current_connection_message_count > 0
                or connected_duration >= self._reconnect_reset_after
            )
            if stable_connection:
                delay = self._min_reconnect
            await self._clock.sleep(delay)
            if not stable_connection:
                delay = min(
                    self._max_reconnect,
                    max(self._min_reconnect, delay * 2),
                )

    async def _mark_event_observed(self) -> None:
        self._last_event_at = self._clock.now()
        if self._on_event is not None:
            await maybe_await(self._on_event(self._last_event_at))

    async def process_message(
        self, raw_message: str | bytes | Mapping[str, Any]
    ) -> EventDisposition:
        payload = await self._parse_message(raw_message)
        async with self._event_state_lock:
            return await self._process_payload(payload)

    async def _persist_generic_event(
        self,
        payload: Mapping[str, Any],
        *,
        event_type: str,
        fingerprint: str,
    ) -> bool:
        event_time = _event_milliseconds(payload, "E")
        transaction_time = _event_milliseconds(payload, "T", "E")
        if event_time <= 0 or transaction_time <= 0:
            await self._fault("EVENT_TIMESTAMP_MISSING", payload)
            return False
        await self._repository.append_account_events(
            (
                AccountEventFact(
                    account_id=self._account_id,
                    event_type=event_type or "UNKNOWN",
                    event_key=fingerprint,
                    event_time_ms=event_time,
                    transaction_time_ms=transaction_time,
                    payload=payload,
                    observed_at=self._clock.now(),
                    economic_event_id=fingerprint,
                ),
            )
        )
        return True

    async def _request_recalibration(
        self, reason: str, payload: Mapping[str, Any]
    ) -> None:
        try:
            if self._on_recalibration_required is not None:
                await maybe_await(
                    self._on_recalibration_required(reason, payload)
                )
        finally:
            await self._close_current_connection()

    async def _publish_execution_order_event(
        self, payload: Mapping[str, Any]
    ) -> None:
        callback = self._on_execution_order_event
        if callback is None:
            return
        try:
            await maybe_await(callback(payload))
        except Exception as exc:
            await self._fault("EXECUTION_ORDER_EVENT_BRIDGE_FAILED", payload)
            raise BinanceDataError(
                "Execution order event bridge failed"
            ) from exc

    async def _process_payload(
        self, payload: Mapping[str, Any]
    ) -> EventDisposition:
        event_type = str(payload.get("e") or "").strip()
        if event_type == "listenKeyExpired":
            await self._fault("LISTEN_KEY_EXPIRED", payload)
            raise ListenKeyExpired("Binance listenKey expired")

        fingerprint = stable_fingerprint(payload)
        if self._dedupe.contains(fingerprint):
            self._duplicate_count += 1
            await self._mark_event_observed()
            return EventDisposition.DUPLICATE

        if event_type == "ACCOUNT_UPDATE":
            disposition = await self._process_account_update(payload, fingerprint)
        elif event_type == "ORDER_TRADE_UPDATE":
            disposition = await self._process_order_update(payload, fingerprint)
        elif event_type in {"ACCOUNT_CONFIG_UPDATE", "MARGIN_CALL"}:
            persisted = await self._persist_generic_event(
                payload, event_type=event_type, fingerprint=fingerprint
            )
            if not persisted:
                return EventDisposition.GAP
            disposition = EventDisposition.RECALIBRATE
        else:
            persisted = await self._persist_generic_event(
                payload, event_type=event_type, fingerprint=fingerprint
            )
            if not persisted:
                return EventDisposition.GAP
            disposition = EventDisposition.UNKNOWN

        if (
            event_type == "ORDER_TRADE_UPDATE"
            and disposition is EventDisposition.APPLY
        ):
            await self._publish_execution_order_event(payload)

        if disposition in {
            EventDisposition.APPLY,
            EventDisposition.UNKNOWN,
            EventDisposition.RECALIBRATE,
            EventDisposition.DUPLICATE,
        }:
            self._dedupe.add(fingerprint)
            await self._mark_event_observed()
        if disposition is EventDisposition.RECALIBRATE:
            await self._request_recalibration(event_type, payload)
        return disposition

    async def _parse_message(
        self, raw_message: str | bytes | Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if isinstance(raw_message, Mapping):
            return raw_message
        if isinstance(raw_message, bytes):
            try:
                raw_message = raw_message.decode("utf-8")
            except UnicodeDecodeError as exc:
                await self._fault("INVALID_UTF8", {"raw_hex": raw_message[:256].hex()})
                raise BinanceDataError("Invalid UTF-8 user stream message") from exc
        try:
            parsed = json.loads(raw_message)
        except (TypeError, ValueError) as exc:
            await self._fault("INVALID_JSON", {"raw": str(raw_message)[:1000]})
            raise BinanceDataError("Invalid user stream JSON") from exc
        if not isinstance(parsed, Mapping):
            await self._fault("INVALID_MESSAGE_SHAPE", {"raw": parsed})
            raise BinanceDataError("User stream message must be an object")
        return parsed

    def _account_update_times(self, payload: Mapping[str, Any]) -> tuple[int, int]:
        transaction_time = _event_milliseconds(payload, "T", "E")
        event_time = _event_milliseconds(payload, "E")
        if transaction_time <= 0 or event_time <= 0:
            raise BinanceDataError("ACCOUNT_UPDATE timestamps are required")
        return transaction_time, event_time

    def _prepare_account_balances(
        self,
        raw_balances: Sequence[Any],
    ) -> tuple[BalanceFact, ...]:
        mappings: list[Mapping[str, Any]] = []
        for item in raw_balances:
            if not isinstance(item, Mapping):
                raise BinanceDataError(
                    "ACCOUNT_UPDATE balance row must be an object"
                )
            mappings.append(item)
        return normalize_account_update_balances(
            mappings,
            account_id=self._account_id,
            previous_by_asset=self._rest_balance_context,
            observed_at=self._clock.now(),
        )

    def _enrich_position_payload(
        self,
        raw_position: Mapping[str, Any],
        *,
        symbol: str,
        side: str,
        transaction_time: int,
    ) -> Mapping[str, Any]:
        context = self._rest_position_context.get((symbol, side))
        enriched = {**raw_position, "T": transaction_time}
        required_context_fields = ("leverage", "markPrice", "liquidationPrice")
        if context is None and not all(field in enriched for field in required_context_fields):
            raise _StreamIntegrityError("POSITION_CONTEXT_MISSING")
        if context is None:
            return enriched
        enriched.setdefault("leverage", context.leverage)
        enriched.setdefault("markPrice", str(context.mark_price))
        liquidation = "0" if context.liquidation_price is None else str(context.liquidation_price)
        enriched.setdefault("liquidationPrice", liquidation)
        return enriched

    def _prepare_account_position(
        self,
        raw_position: Mapping[str, Any],
        *,
        transaction_time: int,
        event_time: int,
    ) -> _PreparedPositionUpdate | None:
        raw_symbol = str(
            raw_position.get("s") or raw_position.get("symbol") or ""
        ).strip().upper()
        if not raw_symbol:
            raise BinanceDataError("ACCOUNT_UPDATE position symbol is required")
        raw_side = normalize_position_side(
            raw_position.get("ps") or raw_position.get("positionSide")
        )
        raw_amount = finite_decimal(
            raw_position.get("pa", raw_position.get("positionAmt", "0")),
            field=f"{raw_symbol}.positionAmt",
        )
        if raw_symbol not in self._managed_symbols:
            if raw_amount != 0:
                raise _StreamIntegrityError("UNMANAGED_NONZERO_POSITION")
            return None
        enriched = self._enrich_position_payload(
            raw_position,
            symbol=raw_symbol,
            side=raw_side,
            transaction_time=transaction_time,
        )
        position = normalize_position(
            enriched,
            account_id=self._account_id,
            observed_at=self._clock.now(),
            source="BINANCE_USER_STREAM",
        )
        entity_key = f"POSITION:{position.symbol}:{position.position_side}"
        return _PreparedPositionUpdate(
            position=position,
            entity_key=entity_key,
            transaction_time_ms=transaction_time,
            event_time_ms=event_time,
            state_signature=_position_state_signature(position),
        )

    async def _prepare_account_positions(
        self,
        raw_positions: Sequence[Any],
        *,
        transaction_time: int,
        event_time: int,
        payload: Mapping[str, Any],
    ) -> tuple[_PreparedPositionUpdate, ...] | EventDisposition:
        prepared: list[_PreparedPositionUpdate] = []
        entity_keys: set[str] = set()
        try:
            for raw_position in raw_positions:
                if not isinstance(raw_position, Mapping):
                    raise BinanceDataError("ACCOUNT_UPDATE position must be an object")
                item = self._prepare_account_position(
                    raw_position,
                    transaction_time=transaction_time,
                    event_time=event_time,
                )
                if item is None:
                    continue
                if item.entity_key in entity_keys:
                    raise _StreamIntegrityError("DUPLICATE_POSITION_ENTITY_IN_EVENT")
                entity_keys.add(item.entity_key)
                decision = self._sequencer.inspect_account_position(
                    item.entity_key,
                    transaction_time_ms=item.transaction_time_ms,
                    event_time_ms=item.event_time_ms,
                    state_signature=item.state_signature,
                )
                if decision.disposition is EventDisposition.DUPLICATE:
                    self._duplicate_count += 1
                    continue
                if decision.disposition is not EventDisposition.APPLY:
                    await self._handle_sequence_failure(decision, payload)
                    return decision.disposition
                prepared.append(item)
        except _StreamIntegrityError as exc:
            self._gap_count += 1
            await self._fault(exc.reason, payload)
            return exc.disposition
        return tuple(prepared)

    @staticmethod
    def _classify_account_reason(reason: str) -> str:
        return {
            "FUNDING_FEE": "FUNDING",
            "DEPOSIT": "TRANSFER",
            "WITHDRAW": "TRANSFER",
            "WITHDRAW_REJECT": "TRANSFER",
            "MARGIN_TRANSFER": "TRANSFER",
            "ASSET_TRANSFER": "TRANSFER",
            "ORDER": "BALANCE",
        }.get(reason, "BALANCE")

    @staticmethod
    def _single_balance_change(
        raw_balances: Sequence[Mapping[str, Any]],
    ) -> tuple[str | None, Decimal | None]:
        changes: list[tuple[str, Decimal]] = []
        for item in raw_balances:
            asset = str(item.get("a") or item.get("asset") or "").strip().upper()
            if not asset:
                raise BinanceDataError("ACCOUNT_UPDATE balance asset is required")
            amount = finite_decimal(
                item.get("bc", item.get("balanceChange", "0")),
                field=f"{asset}.balanceChange",
                default="0",
            )
            if amount != 0:
                changes.append((asset, amount))
        if len(changes) != 1:
            return None, None
        return changes[0]

    async def _persist_account_update(
        self,
        prepared: Sequence[_PreparedPositionUpdate],
        balances: Sequence[BalanceFact],
        raw_balances: Sequence[Mapping[str, Any]],
        *,
        event_type: str,
        event_key: str,
        event_time: int,
        transaction_time: int,
        reason: str,
        payload: Mapping[str, Any],
    ) -> None:
        positions = tuple(item.position for item in prepared)
        currency, amount = self._single_balance_change(raw_balances)
        economic_event_id = stable_fingerprint(
            {
                "account_id": self._account_id,
                "reason": reason,
                "currency": currency,
                "amount": amount,
                "transaction_time": transaction_time,
            }
        ) if amount is not None else event_key
        event = AccountEventFact(
            account_id=self._account_id,
            event_type=event_type,
            event_key=event_key,
            event_time_ms=event_time,
            transaction_time_ms=transaction_time,
            payload={"reason": reason, "event": payload},
            observed_at=self._clock.now(),
            currency=currency,
            amount=amount,
            economic_event_id=economic_event_id,
        )
        batch_writer = getattr(self._repository, "append_exchange_fact_batch", None)
        if callable(batch_writer):
            await maybe_await(
                batch_writer(
                    ExchangeFactBatch(
                        account_id=self._account_id,
                        source="BINANCE_USER_STREAM",
                        observed_at=self._clock.now(),
                        positions=positions,
                        balances=tuple(balances),
                        account_events=(event,),
                        correlation_id=event_key,
                    )
                )
            )
            return
        if positions:
            await self._repository.append_position_snapshots(positions)
        if balances:
            await self._repository.append_balance_snapshots(balances)
        await self._repository.append_account_events((event,))

    def _commit_account_update(
        self,
        prepared: Sequence[_PreparedPositionUpdate],
        balances: Sequence[BalanceFact],
    ) -> None:
        for item in prepared:
            self._sequencer.commit_position(
                item.entity_key,
                transaction_time_ms=item.transaction_time_ms,
                event_time_ms=item.event_time_ms,
                state_signature=item.state_signature,
            )
            position = item.position
            self._rest_position_context[(position.symbol, position.position_side)] = position
        for balance in balances:
            self._rest_balance_context[balance.asset] = balance
        if prepared or balances:
            self._touch_state()

    async def _process_account_update(
        self, payload: Mapping[str, Any], fingerprint: str
    ) -> EventDisposition:
        account = payload.get("a")
        if not isinstance(account, Mapping):
            raise BinanceDataError("ACCOUNT_UPDATE.a must be an object")
        transaction_time, event_time = self._account_update_times(payload)
        raw_positions = account.get("P")
        if raw_positions is None:
            raw_positions = []
        if not isinstance(raw_positions, list):
            raise BinanceDataError("ACCOUNT_UPDATE.a.P must be a list")
        raw_balances = account.get("B")
        if raw_balances is None:
            raw_balances = []
        if not isinstance(raw_balances, list):
            raise BinanceDataError("ACCOUNT_UPDATE.a.B must be a list")
        balances = self._prepare_account_balances(raw_balances)
        prepared = await self._prepare_account_positions(
            raw_positions,
            transaction_time=transaction_time,
            event_time=event_time,
            payload=payload,
        )
        if isinstance(prepared, EventDisposition):
            return prepared
        reason = str(account.get("m") or "UNKNOWN").upper()
        await self._persist_account_update(
            prepared,
            balances,
            tuple(raw_balances),
            event_type=self._classify_account_reason(reason),
            event_key=fingerprint,
            event_time=event_time,
            transaction_time=transaction_time,
            reason=reason,
            payload=payload,
        )
        self._commit_account_update(prepared, balances)
        return EventDisposition.APPLY

    @staticmethod
    def _order_update_times(payload: Mapping[str, Any]) -> tuple[int, int]:
        transaction_time = _event_milliseconds(payload, "T", "E")
        event_time = _event_milliseconds(payload, "E")
        if transaction_time <= 0 or event_time <= 0:
            raise BinanceDataError("ORDER_TRADE_UPDATE timestamps are required")
        return transaction_time, event_time

    @staticmethod
    def _last_fill_quantity(order_payload: Mapping[str, Any]) -> Decimal:
        quantity = finite_decimal(
            order_payload.get("l") or "0",
            field="ORDER_TRADE_UPDATE.lastFillQuantity",
        )
        if quantity < 0:
            raise BinanceDataError(
                "ORDER_TRADE_UPDATE last fill quantity is invalid"
            )
        return quantity

    async def _optional_order_fill(
        self,
        order_payload: Mapping[str, Any],
        normalized_payload: Mapping[str, Any],
        order: OrderFact,
    ) -> tuple[FillFact | None, bool]:
        execution_type = str(order_payload.get("x") or "").upper()
        trade_id = str(order_payload.get("t") or "")
        if execution_type != "TRADE" or trade_id in {"", "0", "-1"}:
            return None, False
        fill = normalize_fill(
            normalized_payload,
            account_id=self._account_id,
            observed_at=self._clock.now(),
            source="BINANCE_USER_STREAM",
        )
        exists = await self._repository.has_fill(
            self._account_id, order.symbol, trade_id
        )
        return fill, not exists

    def _order_account_events(
        self,
        *,
        payload: Mapping[str, Any],
        fingerprint: str,
        event_time: int,
        transaction_time: int,
        fill: FillFact | None,
    ) -> tuple[AccountEventFact, ...]:
        events = [
            AccountEventFact(
                account_id=self._account_id,
                event_type="ORDER_TRADE_UPDATE",
                event_key=fingerprint,
                event_time_ms=event_time,
                transaction_time_ms=transaction_time,
                payload=payload,
                observed_at=self._clock.now(),
            )
        ]
        if fill is None or fill.commission <= 0:
            return tuple(events)
        events.append(
            AccountEventFact(
                account_id=self._account_id,
                event_type="FEE",
                event_key=stable_fingerprint(
                    {
                        "symbol": fill.symbol,
                        "trade_id": fill.exchange_trade_id,
                        "commission": fill.commission,
                    }
                ),
                event_time_ms=fill.event_time_ms,
                transaction_time_ms=transaction_time,
                payload={
                    "symbol": fill.symbol,
                    "trade_id": fill.exchange_trade_id,
                    "commission": str(fill.commission),
                    "commission_asset": fill.commission_asset,
                },
                observed_at=self._clock.now(),
                currency=fill.commission_asset,
                amount=-fill.commission,
                economic_event_id=stable_fingerprint(
                    {
                        "account_id": fill.account_id,
                        "type": "COMMISSION",
                        "trade_id": fill.exchange_trade_id,
                        "currency": fill.commission_asset,
                        "amount": str(-fill.commission),
                    }
                ),
            )
        )
        return tuple(events)

    async def _process_order_update(
        self, payload: Mapping[str, Any], fingerprint: str
    ) -> EventDisposition:
        order_payload = payload.get("o")
        if not isinstance(order_payload, Mapping):
            raise BinanceDataError("ORDER_TRADE_UPDATE.o must be an object")
        transaction_time, event_time = self._order_update_times(payload)
        normalized_payload = {**order_payload, "T": transaction_time}
        order = normalize_order(
            normalized_payload,
            account_id=self._account_id,
            observed_at=self._clock.now(),
            source="BINANCE_USER_STREAM",
            system_client_order_prefixes=self._system_client_order_prefixes,
        )
        if order.active and order.symbol not in self._managed_symbols:
            self._gap_count += 1
            await self._fault("UNMANAGED_ACTIVE_ORDER", payload)
            return EventDisposition.GAP
        last_fill_quantity = self._last_fill_quantity(order_payload)
        entity_key = f"ORDER:{order.symbol}:{order.exchange_order_id}"
        state_signature = _order_state_signature(order)
        decision = self._sequencer.inspect_order(
            entity_key,
            transaction_time_ms=transaction_time,
            event_time_ms=event_time,
            cumulative_fill=order.cumulative_filled_quantity,
            last_fill_quantity=last_fill_quantity,
            state_signature=state_signature,
        )
        if decision.disposition is not EventDisposition.APPLY:
            await self._handle_sequence_failure(decision, payload)
            return decision.disposition
        fill, should_append_fill = await self._optional_order_fill(
            order_payload, normalized_payload, order
        )
        account_events = self._order_account_events(
            payload=payload,
            fingerprint=fingerprint,
            event_time=event_time,
            transaction_time=transaction_time,
            fill=fill,
        )
        persisted_fills = (fill,) if fill is not None and should_append_fill else ()
        batch_writer = getattr(self._repository, "append_exchange_fact_batch", None)
        if callable(batch_writer):
            await maybe_await(
                batch_writer(
                    ExchangeFactBatch(
                        account_id=self._account_id,
                        source="BINANCE_USER_STREAM",
                        observed_at=self._clock.now(),
                        orders=(order,),
                        fills=persisted_fills,
                        account_events=account_events,
                        correlation_id=fingerprint,
                    )
                )
            )
        else:
            await self._repository.append_order_snapshots((order,))
            if persisted_fills:
                await self._repository.append_fill_events(persisted_fills)
            await self._repository.append_account_events(account_events)
        self._sequencer.commit_order(
            entity_key,
            transaction_time_ms=transaction_time,
            event_time_ms=event_time,
            cumulative_fill=order.cumulative_filled_quantity,
            state_signature=state_signature,
        )
        order_key = (order.symbol, order.exchange_order_id)
        if order.active:
            self._rest_order_context[order_key] = order
        else:
            self._rest_order_context.pop(order_key, None)
        self._touch_state()
        return EventDisposition.APPLY

    async def _handle_sequence_failure(
        self, decision: SequencingDecision, payload: Mapping[str, Any]
    ) -> None:
        if decision.disposition is EventDisposition.DUPLICATE:
            self._duplicate_count += 1
            return
        if decision.disposition is EventDisposition.OUT_OF_ORDER:
            self._out_of_order_count += 1
        elif decision.disposition is EventDisposition.GAP:
            self._gap_count += 1
        await self._fault(decision.reason, payload)

    async def _fault(self, reason: str, payload: Mapping[str, Any]) -> None:
        callback_error: Exception | None = None
        try:
            await self._repository.append_account_events(
                (
                    AccountEventFact(
                        account_id=self._account_id,
                        event_type="STREAM_INTEGRITY_FAULT",
                        event_key=stable_fingerprint(
                            {"reason": reason, "payload": payload}
                        ),
                        event_time_ms=_event_milliseconds(payload, "E"),
                        transaction_time_ms=_event_milliseconds(payload, "T", "E"),
                        payload={"reason": reason, "event": payload},
                        observed_at=self._clock.now(),
                    ),
                )
            )
        finally:
            if self._on_integrity_fault is not None:
                try:
                    await maybe_await(self._on_integrity_fault(reason, payload))
                except Exception as exc:
                    callback_error = exc
            # Do not continue consuming a stream after a proven gap/regression.
            await self._close_current_connection()
        if callback_error is not None:
            raise callback_error
