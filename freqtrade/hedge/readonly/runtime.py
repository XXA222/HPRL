from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from freqtrade.hedge.exchange.base import (
    CalibrationKind,
    CalibrationResult,
    Clock,
    ReadonlyAccountView,
    ReadonlyFactRepository,
    ReadonlyState,
    SystemClock,
)
from freqtrade.hedge.exchange.binance_readonly import (
    AiohttpBinanceRestTransport,
    BinanceReadonlyClient,
    PermissionPolicy,
)
from freqtrade.hedge.exchange.clock_sync import ClockSynchronizer
from freqtrade.hedge.exchange.binance_user_stream import (
    BinanceUserStream,
    WebSocketConnector,
)
from freqtrade.hedge.exchange.symbol_codec import normalize_exchange_symbols
from freqtrade.hedge.exchange.listen_key import (
    BinanceListenKeyClient,
    ListenKeyManager,
)
from freqtrade.hedge.readonly.calibration import ReadonlyCalibration
from freqtrade.hedge.readonly.freshness import FreshnessPolicy, UserStreamFreshness
from freqtrade.hedge.readonly.scheduler import (
    CalibrationSchedule,
    ReconciliationScheduler,
)
from freqtrade.hedge.readonly.service import (
    BinanceReadonlyService,
    ReadonlyRuntimeSnapshot,
)
from freqtrade.hedge.readonly.soak_monitor import (
    ReadonlyServiceSoakProvider,
    SoakAccountingSource,
    SoakMonitor,
    SoakRunner,
)


def _normalized_symbols(values: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return normalize_exchange_symbols(list(values))


def _positive_finite(value: float, *, field_name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return parsed


def _optional_proxy_url(value: str | None, *, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().rstrip("/")
    if not normalized:
        return None
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{field_name} must not embed credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(f"{field_name} must not contain path, query, or fragment")
    return normalized


@dataclass(frozen=True, slots=True)
class BinanceReadonlyRuntimeConfig:
    """Single source of truth for the live Binance read-only runtime."""

    account_id: str
    managed_symbols: tuple[str, ...] | list[str]
    api_key: str = field(repr=False)
    api_secret: str = field(repr=False)
    futures_base_url: str = "https://fapi.binance.com"
    spot_base_url: str = "https://api.binance.com"
    websocket_base_url: str = "wss://fstream.binance.com/ws"
    rest_proxy_url: str | None = None
    websocket_proxy_url: str | None = None
    trust_env_proxy: bool = False
    request_timeout_seconds: float = 10.0
    recv_window_ms: int = 5000
    max_clock_skew_ms: float = 1000.0
    target_leverage: int | None = None
    max_collection_span_seconds: float = 15.0
    fast_calibration_interval: timedelta = timedelta(minutes=1)
    full_calibration_interval: timedelta = timedelta(minutes=15)
    calibration_error_retry_interval: timedelta = timedelta(seconds=30)
    event_stale_after: timedelta | None = None
    calibration_stale_after: timedelta = timedelta(minutes=20)
    fill_lookback: timedelta = timedelta(hours=72)
    history_overlap: timedelta = timedelta(minutes=5)
    max_history_backfill: timedelta | None = timedelta(days=30)
    quantity_tolerance: Decimal = Decimal(0)
    financial_tolerance: Decimal = Decimal("0.00000001")
    listen_key_ttl: timedelta = timedelta(minutes=60)
    listen_key_renew_interval: timedelta = timedelta(minutes=30)
    listen_key_error_retry_interval: timedelta = timedelta(seconds=30)
    min_reconnect_delay_seconds: float = 1.0
    max_reconnect_delay_seconds: float = 60.0
    reconnect_reset_after_seconds: float = 30.0
    drift_verification_attempts: int = 1
    permission_policy: PermissionPolicy = field(default_factory=PermissionPolicy)
    system_client_order_prefixes: tuple[str, ...] = ("fthedge-",)

    def __post_init__(self) -> None:
        account_id = self.account_id.strip()
        if not account_id:
            raise ValueError("account_id is required")
        api_key = self.api_key.strip()
        api_secret = self.api_secret.strip()
        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret are required")
        object.__setattr__(self, "account_id", account_id)
        object.__setattr__(
            self,
            "managed_symbols",
            _normalized_symbols(self.managed_symbols),
        )
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "api_secret", api_secret)
        object.__setattr__(
            self,
            "rest_proxy_url",
            _optional_proxy_url(self.rest_proxy_url, field_name="rest_proxy_url"),
        )
        object.__setattr__(
            self,
            "websocket_proxy_url",
            _optional_proxy_url(
                self.websocket_proxy_url,
                field_name="websocket_proxy_url",
            ),
        )
        if not isinstance(self.trust_env_proxy, bool):
            raise ValueError("trust_env_proxy must be a boolean")
        _positive_finite(
            self.request_timeout_seconds,
            field_name="request_timeout_seconds",
        )
        _positive_finite(
            self.max_collection_span_seconds,
            field_name="max_collection_span_seconds",
        )
        _positive_finite(
            self.min_reconnect_delay_seconds,
            field_name="min_reconnect_delay_seconds",
        )
        _positive_finite(
            self.max_reconnect_delay_seconds,
            field_name="max_reconnect_delay_seconds",
        )
        if self.max_reconnect_delay_seconds < self.min_reconnect_delay_seconds:
            raise ValueError(
                "max_reconnect_delay_seconds must be >= min_reconnect_delay_seconds"
            )
        if not math.isfinite(self.reconnect_reset_after_seconds):
            raise ValueError("reconnect_reset_after_seconds must be finite")
        if self.reconnect_reset_after_seconds < 0:
            raise ValueError("reconnect_reset_after_seconds must be nonnegative")
        if self.recv_window_ms < 1:
            raise ValueError("recv_window_ms must be positive")
        _positive_finite(
            self.max_clock_skew_ms,
            field_name="max_clock_skew_ms",
        )
        if self.target_leverage is not None and int(self.target_leverage) <= 0:
            raise ValueError("target_leverage must be positive when configured")
        if self.target_leverage is not None:
            object.__setattr__(self, "target_leverage", int(self.target_leverage))
        if self.drift_verification_attempts < 0:
            raise ValueError("drift_verification_attempts must be nonnegative")


@dataclass(slots=True)
class BinanceReadonlyRuntime:
    """Own all live direction-two components and their lifecycle."""

    config: BinanceReadonlyRuntimeConfig
    transport: AiohttpBinanceRestTransport
    client: BinanceReadonlyClient
    listen_keys: ListenKeyManager
    stream: BinanceUserStream
    calibration: ReadonlyCalibration
    scheduler: ReconciliationScheduler
    service: BinanceReadonlyService
    clock: Clock

    async def start(self) -> None:
        await self.service.start()

    async def stop(self) -> None:
        await self.service.stop()

    async def __aenter__(self) -> BinanceReadonlyRuntime:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        await self.stop()

    def snapshot(self) -> ReadonlyRuntimeSnapshot:
        return self.service.runtime_snapshot()

    def account_view(self) -> ReadonlyAccountView:
        return self.service.account_view()

    async def wait_until_ready(
        self,
        *,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 0.1,
    ) -> ReadonlyRuntimeSnapshot:
        timeout = _positive_finite(
            timeout_seconds,
            field_name="timeout_seconds",
        )
        poll_interval = _positive_finite(
            poll_interval_seconds,
            field_name="poll_interval_seconds",
        )
        deadline = self.clock.monotonic() + timeout
        while True:
            snapshot = self.snapshot()
            if snapshot.status.state is ReadonlyState.READY:
                return snapshot
            if snapshot.status.state in {
                ReadonlyState.HALT,
                ReadonlyState.STOPPED,
            }:
                raise RuntimeError(
                    "Binance readonly runtime cannot become READY: "
                    f"{snapshot.status.state.value}:{snapshot.status.reason}"
                )
            remaining = deadline - self.clock.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    "Binance readonly runtime did not become READY within "
                    f"{timeout_seconds} seconds; current state="
                    f"{snapshot.status.state.value}:{snapshot.status.reason}"
                )
            await self.clock.sleep(min(poll_interval, remaining))

    async def calibrate_now(
        self, kind: CalibrationKind = CalibrationKind.FULL
    ) -> CalibrationResult:
        return await self.service.calibrate_now(kind)

    async def request_reconnect(self) -> None:
        await self.stream.request_reconnect()

    async def run_until(self, stop_event: asyncio.Event) -> None:
        await self.start()
        try:
            await stop_event.wait()
        finally:
            await self.stop()

    def build_soak_runner(
        self,
        *,
        accounting_source: SoakAccountingSource,
        observation_path: str | Path,
        interval_seconds: float = 60.0,
    ) -> SoakRunner:
        observation_file = Path(observation_path)
        provider = ReadonlyServiceSoakProvider(
            account_id=self.config.account_id,
            service=self.service,
            accounting_source=accounting_source,
            clock=self.clock,
            baseline_path=observation_file.with_suffix(
                observation_file.suffix + ".baseline.json"
            ),
            run_id=f"{self.config.account_id}:{int(self.clock.now().timestamp())}",
        )
        monitor = SoakMonitor(
            observation_path,
            expected_interval_seconds=interval_seconds,
        )
        return SoakRunner(
            provider=provider,
            monitor=monitor,
            interval_seconds=interval_seconds,
            clock=self.clock,
        )


def build_binance_readonly_runtime(
    *,
    config: BinanceReadonlyRuntimeConfig,
    repository: ReadonlyFactRepository,
    clock: Clock | None = None,
    websocket_connector: WebSocketConnector | None = None,
    http_session: Any = None,
) -> BinanceReadonlyRuntime:
    """Build the complete REST + stream + reconciliation runtime."""

    runtime_clock = clock or SystemClock()
    transport = AiohttpBinanceRestTransport(
        api_key=config.api_key,
        api_secret=config.api_secret,
        futures_base_url=config.futures_base_url,
        spot_base_url=config.spot_base_url,
        timeout_seconds=config.request_timeout_seconds,
        recv_window_ms=config.recv_window_ms,
        session=http_session,
        proxy_url=config.rest_proxy_url,
        trust_env_proxy=config.trust_env_proxy,
        clock=runtime_clock,
    )
    client = BinanceReadonlyClient(
        transport=transport,
        account_id=config.account_id,
        managed_symbols=config.managed_symbols,
        clock_sync=ClockSynchronizer(
            clock=runtime_clock,
            max_abs_skew_ms=config.max_clock_skew_ms,
        ),
        clock=runtime_clock,
        max_collection_span_seconds=config.max_collection_span_seconds,
        system_client_order_prefixes=config.system_client_order_prefixes,
    )
    listen_keys = ListenKeyManager(
        client=BinanceListenKeyClient(transport),
        clock=runtime_clock,
        ttl=config.listen_key_ttl,
        renew_interval=config.listen_key_renew_interval,
        error_retry_interval=config.listen_key_error_retry_interval,
    )
    stream = BinanceUserStream(
        account_id=config.account_id,
        managed_symbols=config.managed_symbols,
        repository=repository,
        listen_keys=listen_keys,
        connector=websocket_connector,
        websocket_proxy_url=config.websocket_proxy_url,
        trust_env_proxy=config.trust_env_proxy,
        websocket_base_url=config.websocket_base_url,
        clock=runtime_clock,
        min_reconnect_delay_seconds=config.min_reconnect_delay_seconds,
        max_reconnect_delay_seconds=config.max_reconnect_delay_seconds,
        reconnect_reset_after_seconds=config.reconnect_reset_after_seconds,
        system_client_order_prefixes=config.system_client_order_prefixes,
    )
    calibration = ReadonlyCalibration(
        client=client,
        repository=repository,
        managed_symbols=config.managed_symbols,
        clock=runtime_clock,
        quantity_tolerance=config.quantity_tolerance,
        financial_tolerance=config.financial_tolerance,
        fill_lookback=config.fill_lookback,
        history_overlap=config.history_overlap,
        max_history_backfill=config.max_history_backfill,
    )
    scheduler = ReconciliationScheduler(
        calibration=calibration,
        schedule=CalibrationSchedule(
            fast_interval=config.fast_calibration_interval,
            full_interval=config.full_calibration_interval,
            error_retry_interval=config.calibration_error_retry_interval,
        ),
        clock=runtime_clock,
    )
    freshness = UserStreamFreshness(
        FreshnessPolicy(
            event_stale_after=config.event_stale_after,
            calibration_stale_after=config.calibration_stale_after,
        )
    )
    service = BinanceReadonlyService(
        client=client,
        calibration=calibration,
        stream=stream,
        scheduler=scheduler,
        freshness=freshness,
        permission_policy=config.permission_policy,
        clock=runtime_clock,
        drift_verification_attempts=config.drift_verification_attempts,
        target_leverage=config.target_leverage,
    )
    return BinanceReadonlyRuntime(
        config=config,
        transport=transport,
        client=client,
        listen_keys=listen_keys,
        stream=stream,
        calibration=calibration,
        scheduler=scheduler,
        service=service,
        clock=runtime_clock,
    )
