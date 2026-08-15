from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum, StrEnum
from hashlib import sha256
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


def utc_now() -> datetime:
    return datetime.now(UTC)


async def maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await value
    return value


class ReadonlyState(StrEnum):
    STARTING = "STARTING"
    PREFLIGHT = "PREFLIGHT"
    CALIBRATING = "CALIBRATING"
    READY = "READY"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    HALT = "HALT"
    STOPPED = "STOPPED"


class CalibrationKind(StrEnum):
    FAST = "FAST"
    FULL = "FULL"
    RECONNECT = "RECONNECT"
    STARTUP = "STARTUP"


class EventDisposition(StrEnum):
    APPLY = "APPLY"
    DUPLICATE = "DUPLICATE"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    GAP = "GAP"
    RECALIBRATE = "RECALIBRATE"
    UNKNOWN = "UNKNOWN"


class ReadonlyReasonCode(StrEnum):
    CONSISTENT = "CONSISTENT"
    UNMANAGED_POSITION = "UNMANAGED_POSITION"
    UNMANAGED_ORDER = "UNMANAGED_ORDER"
    EXTERNAL_ORDER = "EXTERNAL_ORDER"
    STALE_REST_SNAPSHOT = "STALE_REST_SNAPSHOT"
    STALE_USER_STREAM = "STALE_USER_STREAM"
    CLOCK_SKEW_EXCEEDED = "CLOCK_SKEW_EXCEEDED"
    OUT_OF_ORDER_EVENT = "OUT_OF_ORDER_EVENT"
    EVENT_GAP = "EVENT_GAP"
    RECONCILIATION_DRIFT = "RECONCILIATION_DRIFT"
    HISTORY_GAP_REQUIRES_BACKFILL = "HISTORY_GAP_REQUIRES_BACKFILL"
    POSITION_MODE_MISMATCH = "POSITION_MODE_MISMATCH"
    MARGIN_MODE_MISMATCH = "MARGIN_MODE_MISMATCH"
    LEVERAGE_MISMATCH = "LEVERAGE_MISMATCH"


class OrderOrigin(StrEnum):
    SYSTEM = "SYSTEM"
    EXTERNAL = "EXTERNAL"
    UNKNOWN = "UNKNOWN"


class ReconciliationResolution(StrEnum):
    OPEN = "OPEN"
    OBSERVED = "OBSERVED"
    RESOLVED_BY_REST = "RESOLVED_BY_REST"
    QUARANTINED = "QUARANTINED"
    ACKNOWLEDGED = "ACKNOWLEDGED"


@dataclass(frozen=True, slots=True)
class BinanceHttpResponse:
    data: Any
    status: int
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TransportTelemetry:
    logical_request_count: int
    attempt_count: int
    retry_count: int
    error_count: int
    rate_limit_count: int
    server_error_count: int
    used_weight: int
    last_latency_ms: float | None
    last_status: int | None
    last_error: str | None
    last_response_at: datetime | None


@dataclass(frozen=True, slots=True)
class ApiPermissionReport:
    read_enabled: bool
    futures_enabled: bool | None
    withdrawals_enabled: bool | None
    internal_transfer_enabled: bool | None
    universal_transfer_enabled: bool | None
    spot_margin_trading_enabled: bool | None
    strict_readonly_verified: bool
    warnings: tuple[str, ...] = ()
    runtime_readonly_enforced: bool = True
    reasons: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class BalanceFact:
    account_id: str
    asset: str
    wallet_balance: Decimal
    available_balance: Decimal
    cross_wallet_balance: Decimal
    unrealized_pnl: Decimal
    observed_at: datetime
    source: str
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class PositionFact:
    account_id: str
    symbol: str
    position_side: str
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    liquidation_price: Decimal | None
    leverage: int
    margin_mode: str
    update_time_ms: int
    observed_at: datetime
    source: str
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.account_id, self.symbol, self.position_side


@dataclass(frozen=True, slots=True)
class OrderFact:
    account_id: str
    symbol: str
    position_side: str
    exchange_order_id: str
    client_order_id: str
    side: str
    order_type: str
    status: str
    original_quantity: Decimal
    cumulative_filled_quantity: Decimal
    average_price: Decimal
    reduce_only: bool
    update_time_ms: int
    observed_at: datetime
    source: str
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)
    origin: OrderOrigin = OrderOrigin.UNKNOWN
    quarantined: bool = False
    contract_version: str = "2.0"
    event_version: int = 1
    correlation_id: str | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return self.account_id, self.symbol, self.exchange_order_id

    @property
    def active(self) -> bool:
        # Fail closed for newly introduced/unknown Binance statuses. Only
        # explicitly terminal states are treated as inactive.
        return self.status not in {
            "FILLED",
            "CANCELED",
            "CANCELLED",
            "REJECTED",
            "EXPIRED",
            "EXPIRED_IN_MATCH",
        }


@dataclass(frozen=True, slots=True)
class FillFact:
    account_id: str
    symbol: str
    position_side: str
    exchange_trade_id: str
    exchange_order_id: str
    side: str
    quantity: Decimal
    price: Decimal
    commission: Decimal
    commission_asset: str | None
    realized_pnl: Decimal
    event_time_ms: int
    observed_at: datetime
    source: str
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.account_id, self.symbol, self.exchange_trade_id


@dataclass(frozen=True, slots=True)
class AccountConfigurationFact:
    account_id: str
    hedge_mode: bool
    active_margin_modes: tuple[str, ...]
    leverage_by_symbol_side: Mapping[str, int]
    observed_at: datetime
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class AccountSnapshotFact:
    account_id: str
    total_wallet_balance: Decimal
    total_available_balance: Decimal
    total_margin_balance: Decimal
    total_initial_margin: Decimal
    total_maintenance_margin: Decimal
    total_unrealized_pnl: Decimal
    observed_at: datetime
    collection_started_at: datetime
    collection_completed_at: datetime
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class AccountEventFact:
    account_id: str
    event_type: str
    event_key: str
    event_time_ms: int
    transaction_time_ms: int
    payload: Mapping[str, Any]
    observed_at: datetime
    source: str = "BINANCE_USER_STREAM"
    currency: str | None = None
    amount: Decimal | None = None
    economic_event_id: str | None = None
    contract_version: str = "2.0"
    event_version: int = 1
    correlation_id: str | None = None

    @property
    def identity(self) -> str:
        return self.economic_event_id or self.event_key


@dataclass(frozen=True, slots=True)
class ReconciliationDiffFact:
    account_id: str
    entity_type: str
    entity_key: str
    reason_code: str
    expected: Mapping[str, Any] | None
    observed: Mapping[str, Any] | None
    severity: str = "ERROR"
    resolution: ReconciliationResolution = ReconciliationResolution.OPEN
    resolution_detail: str | None = None
    resolved_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ExchangeFactBatch:
    """One atomic exchange fact ingestion unit for ledger adapters."""

    account_id: str
    source: str
    observed_at: datetime
    reconciliation_run_id: str | None = None
    account_snapshot: AccountSnapshotFact | None = None
    balances: tuple[BalanceFact, ...] = ()
    positions: tuple[PositionFact, ...] = ()
    orders: tuple[OrderFact, ...] = ()
    fills: tuple[FillFact, ...] = ()
    account_events: tuple[AccountEventFact, ...] = ()
    reconciliation_diffs: tuple[ReconciliationDiffFact, ...] = ()
    correlation_id: str | None = None


@dataclass(frozen=True, slots=True)
class Direction2HealthFact:
    """Health output consumed by the central ReadinessGate."""

    account_id: str
    rest_fresh: bool
    stream_connected: bool
    stream_fresh: bool
    clock_synchronized: bool
    configuration_valid: bool
    reconciliation_consistent: bool
    unmanaged_position_count: int
    unmanaged_order_count: int
    external_order_count: int
    reason_codes: tuple[str, ...]
    observed_at: datetime
    last_rest_at: datetime | None = None
    last_stream_event_at: datetime | None = None
    last_stream_connected_at: datetime | None = None
    latest_reconciliation_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class ReadonlyAccountView:
    """Deterministic latest account projection for the central hedge runtime.

    The full account snapshot and configuration are REST-authoritative. Position,
    balance and active-order tuples are updated by REST reseeds and user-stream
    events so callers do not need to reach into service internals.
    """

    account_id: str
    observed_at: datetime
    account_snapshot: AccountSnapshotFact | None
    balances: tuple[BalanceFact, ...]
    positions: tuple[PositionFact, ...]
    active_orders: tuple[OrderFact, ...]
    configuration: AccountConfigurationFact | None
    revision: int
    last_calibration_run_id: str | None = None


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    run_id: str
    kind: CalibrationKind
    started_at: datetime
    completed_at: datetime
    position_count: int
    active_order_count: int
    fill_count: int
    diff_count: int
    unmanaged_positions: tuple[str, ...]
    unmanaged_orders: tuple[str, ...]
    consistent: bool
    reason: str


@dataclass(frozen=True, slots=True)
class StreamHealth:
    connected: bool
    last_connected_at: datetime | None
    last_event_at: datetime | None
    last_calibration_at: datetime | None
    reconnect_count: int
    duplicate_count: int
    out_of_order_count: int
    gap_count: int


class Clock(Protocol):
    def now(self) -> datetime: ...

    def monotonic(self) -> float: ...

    async def sleep(self, seconds: float) -> None: ...


class SystemClock:
    def now(self) -> datetime:
        return utc_now()

    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class ReadonlyExchangePort(Protocol):
    account_id: str
    managed_symbols: tuple[str, ...]

    async def synchronize_clock(self) -> None: ...

    async def preflight_permissions(self, policy: Any = None) -> ApiPermissionReport: ...

    async def fetch_positions(
        self, symbols: Sequence[str] | None = None
    ) -> Sequence[PositionFact]: ...

    async def fetch_open_orders(
        self, symbol: str | None = None
    ) -> Sequence[OrderFact]: ...

    async def fetch_bundle(
        self, *, include_fills: bool, fill_start_time_ms: int | None = None
    ) -> Any: ...


class BinanceRestTransport(Protocol):
    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        signed: bool = False,
        api_group: str = "futures",
        weight: int = 1,
    ) -> BinanceHttpResponse: ...


class ReadonlyFactRepository(Protocol):
    """Public persistence port. Implementations own transactions and sessions."""

    async def append_position_snapshots(
        self, facts: Sequence[PositionFact], *, reconciliation_run_id: str | None = None
    ) -> None: ...

    async def append_order_snapshots(
        self, facts: Sequence[OrderFact], *, reconciliation_run_id: str | None = None
    ) -> None: ...

    async def append_fill_events(
        self, facts: Sequence[FillFact], *, reconciliation_run_id: str | None = None
    ) -> None: ...

    async def append_account_snapshot(
        self, fact: AccountSnapshotFact, *, reconciliation_run_id: str | None = None
    ) -> None: ...

    async def append_balance_snapshots(
        self, facts: Sequence[BalanceFact], *, reconciliation_run_id: str | None = None
    ) -> None: ...

    async def append_account_events(self, facts: Sequence[AccountEventFact]) -> None: ...

    async def begin_reconciliation(
        self, *, account_id: str, kind: CalibrationKind, started_at: datetime
    ) -> str: ...

    async def append_reconciliation_diffs(
        self, run_id: str, diffs: Sequence[ReconciliationDiffFact]
    ) -> None: ...

    async def complete_reconciliation(
        self,
        run_id: str,
        *,
        completed_at: datetime,
        status: str,
        reason: str,
    ) -> None: ...

    async def load_active_positions(self, account_id: str) -> Sequence[PositionFact]: ...

    async def load_active_orders(self, account_id: str) -> Sequence[OrderFact]: ...

    async def has_fill(
        self, account_id: str, symbol: str, exchange_trade_id: str
    ) -> bool: ...



class AtomicReadonlyFactRepository(ReadonlyFactRepository, Protocol):
    async def append_exchange_fact_batch(self, batch: ExchangeFactBatch) -> None: ...


class ReadonlyHistoryCursorRepository(Protocol):
    async def load_history_cursor(
        self, account_id: str, cursor_name: str
    ) -> int | None: ...

    async def save_history_cursor(
        self, account_id: str, cursor_name: str, cursor_ms: int
    ) -> None: ...

def to_primitive(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_primitive(item) for key, item in asdict(value).items()}
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        items = [to_primitive(item) for item in value]
        return sorted(
            items,
            key=lambda item: json.dumps(
                item, sort_keys=True, separators=(",", ":")
            ),
        )
    if isinstance(value, (list, tuple)):
        return [to_primitive(item) for item in value]
    return value


def stable_fingerprint(value: Any) -> str:
    payload = json.dumps(to_primitive(value), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


AsyncCallback = Callable[..., Awaitable[None] | None]
