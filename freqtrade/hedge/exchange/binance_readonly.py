from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any
from urllib.parse import urlencode, urlsplit

import aiohttp

from .base import (
    AccountConfigurationFact,
    AccountSnapshotFact,
    ApiPermissionReport,
    BalanceFact,
    BinanceHttpResponse,
    BinanceRestTransport,
    Clock,
    FillFact,
    OrderFact,
    PositionFact,
    SystemClock,
    TransportTelemetry,
)
from .binance_normalizer import (
    finite_decimal,
    int_value,
    normalize_account_snapshot,
    normalize_configuration,
    normalize_fills,
    normalize_order,
    normalize_orders,
    normalize_positions,
    strict_bool,
)
from .clock_sync import ClockSynchronizer
from .rate_limit import (
    AdaptiveWeightLimiter,
    BinanceDataError,
    BinancePermissionError,
    BinanceRateLimitError,
    BinanceTransportError,
    RetryPolicy,
    parse_retry_after,
    run_with_retry,
)
from .symbol_codec import normalize_exchange_symbols, to_binance_symbol


def _positive_finite_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field} must be a finite positive number")
    return parsed


def _validated_base_url(value: str, *, field: str) -> str:
    normalized = value.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if not normalized or not parsed.hostname:
        raise ValueError(f"{field} must be an absolute URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(f"{field} must not contain credentials, query, or fragment")
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in local_hosts
    ):
        raise ValueError(f"{field} must use HTTPS except for a local test server")
    return normalized


def _validated_proxy_url(value: str | None, *, field: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip().rstrip("/")
    if not normalized:
        return None
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"{field} must be an absolute HTTP(S) proxy URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{field} must not embed proxy credentials")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ValueError(f"{field} must not contain path, query, or fragment")
    return normalized


def _positive_int(value: Any, *, field: str, maximum: int | None = None) -> int:
    parsed = int_value(value, field=field)
    if parsed < 1:
        raise ValueError(f"{field} must be at least 1")
    if maximum is not None and parsed > maximum:
        return maximum
    return parsed


def _nonnegative_int(value: Any, *, field: str) -> int:
    parsed = int_value(value, field=field)
    if parsed < 0:
        raise ValueError(f"{field} must be nonnegative")
    return parsed


def _time_windows(
    start_time_ms: int,
    end_time_ms: int,
    *,
    max_span_ms: int = 7 * 24 * 60 * 60 * 1000 - 1,
) -> tuple[tuple[int, int], ...]:
    """Return non-overlapping inclusive windows within Binance's 7-day limit."""
    if start_time_ms < 0 or end_time_ms < 0:
        raise ValueError("time bounds must be nonnegative")
    if end_time_ms < start_time_ms:
        raise ValueError("end_time_ms must be >= start_time_ms")
    if max_span_ms < 0:
        raise ValueError("max_span_ms must be nonnegative")
    windows: list[tuple[int, int]] = []
    cursor = start_time_ms
    while cursor <= end_time_ms:
        window_end = min(end_time_ms, cursor + max_span_ms)
        windows.append((cursor, window_end))
        cursor = window_end + 1
    return tuple(windows)


def _normalize_symbols(symbols: Sequence[str]) -> tuple[str, ...]:
    return normalize_exchange_symbols(list(symbols))


def _fill_core(item: FillFact) -> tuple[Any, ...]:
    return (
        item.symbol,
        item.position_side,
        item.exchange_order_id,
        item.side,
        item.quantity,
        item.price,
        item.commission,
        item.commission_asset,
        item.realized_pnl,
        item.event_time_ms,
    )


def _order_core(item: OrderFact) -> tuple[Any, ...]:
    return (
        item.symbol,
        item.position_side,
        item.client_order_id,
        item.side,
        item.order_type,
        item.status,
        item.original_quantity,
        item.cumulative_filled_quantity,
        item.average_price,
        item.reduce_only,
    )


def _deduplicate_fills(facts: Sequence[FillFact]) -> tuple[FillFact, ...]:
    deduplicated: dict[tuple[str, str, str], FillFact] = {}
    for item in facts:
        key = (item.account_id, item.symbol, item.exchange_trade_id)
        previous = deduplicated.get(key)
        if previous is not None and _fill_core(previous) != _fill_core(item):
            raise BinanceDataError(
                f"Conflicting duplicate fill: {item.exchange_trade_id}"
            )
        deduplicated[key] = item
    return tuple(
        sorted(
            deduplicated.values(),
            key=lambda item: (item.event_time_ms, item.exchange_trade_id),
        )
    )


def _deduplicate_orders(facts: Sequence[OrderFact]) -> tuple[OrderFact, ...]:
    deduplicated: dict[tuple[str, str, str], OrderFact] = {}
    for item in facts:
        key = (item.account_id, item.symbol, item.exchange_order_id)
        previous = deduplicated.get(key)
        if previous is None or item.update_time_ms > previous.update_time_ms:
            deduplicated[key] = item
        elif (
            item.update_time_ms == previous.update_time_ms
            and _order_core(previous) != _order_core(item)
        ):
            raise BinanceDataError(
                f"Conflicting duplicate order: {item.exchange_order_id}"
            )
    return tuple(
        sorted(
            deduplicated.values(),
            key=lambda item: (item.update_time_ms, item.exchange_order_id),
        )
    )


@dataclass(frozen=True, slots=True)
class PermissionPolicy:
    require_restriction_endpoint: bool = True
    require_read_enabled: bool = True
    forbid_withdrawals: bool = True
    forbid_internal_transfers: bool = True
    forbid_universal_transfers: bool = True
    forbid_spot_margin_trading: bool = True
    require_futures_enabled: bool = True
    forbid_futures_trading_permission: bool = False
    allow_testnet_restriction_endpoint_unavailable: bool = False

    def __post_init__(self) -> None:
        if self.require_futures_enabled and self.forbid_futures_trading_permission:
            raise ValueError(
                "require_futures_enabled and forbid_futures_trading_permission are contradictory"
            )
        if (
            self.allow_testnet_restriction_endpoint_unavailable
            and self.require_restriction_endpoint
        ):
            raise ValueError(
                "testnet restriction fallback requires require_restriction_endpoint=False"
            )


@dataclass(frozen=True, slots=True)
class BinanceAccountBundle:
    account_snapshot: AccountSnapshotFact
    balances: tuple[BalanceFact, ...]
    positions: tuple[PositionFact, ...]
    open_orders: tuple[OrderFact, ...]
    fills: tuple[FillFact, ...]
    configuration: AccountConfigurationFact
    income_events: tuple[Mapping[str, Any], ...]
    collection_started_at: datetime
    collection_completed_at: datetime
    order_history: tuple[OrderFact, ...] = ()
    order_history_query_recovered_ids: tuple[str, ...] = ()
    order_history_snapshot_fallback_ids: tuple[str, ...] = ()


class AiohttpBinanceRestTransport:
    """Signed Binance transport with an explicit read-only endpoint allowlist."""

    FUTURES_BASE = "https://fapi.binance.com"
    SPOT_BASE = "https://api.binance.com"

    _ALLOWED = {
        ("GET", "/fapi/v1/time"),
        ("GET", "/fapi/v3/account"),
        ("GET", "/fapi/v2/account"),
        ("GET", "/fapi/v3/positionRisk"),
        ("GET", "/fapi/v2/positionRisk"),
        ("GET", "/fapi/v1/openOrders"),
        ("GET", "/fapi/v1/order"),
        ("GET", "/fapi/v1/allOrders"),
        ("GET", "/fapi/v1/userTrades"),
        ("GET", "/fapi/v1/positionSide/dual"),
        ("GET", "/fapi/v1/symbolConfig"),
        ("GET", "/fapi/v1/income"),
        ("GET", "/fapi/v1/exchangeInfo"),
        ("GET", "/fapi/v1/premiumIndex"),
        ("GET", "/fapi/v1/ticker/bookTicker"),
        ("GET", "/sapi/v1/account/apiRestrictions"),
        ("POST", "/fapi/v1/listenKey"),
        ("PUT", "/fapi/v1/listenKey"),
        ("DELETE", "/fapi/v1/listenKey"),
    }

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        timestamp_provider: Callable[[], int] | None = None,
        futures_base_url: str | None = None,
        spot_base_url: str | None = None,
        timeout_seconds: float = 10.0,
        recv_window_ms: int = 5000,
        session: Any = None,
        proxy_url: str | None = None,
        trust_env_proxy: bool = False,
        limiter: AdaptiveWeightLimiter | None = None,
        retry_policy: RetryPolicy | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not api_key.strip() or not api_secret.strip():
            raise ValueError("api_key and api_secret are required")
        timeout_value = _positive_finite_float(
            timeout_seconds, field="timeout_seconds"
        )
        recv_window_value = _positive_int(
            recv_window_ms, field="recv_window_ms"
        )
        if recv_window_value > 60_000:
            raise ValueError("recv_window_ms must not exceed 60000")
        self._api_key = api_key
        self._api_secret = api_secret.encode("utf-8")
        self._timestamp_provider = timestamp_provider
        self._futures_base_url = _validated_base_url(
            futures_base_url or self.FUTURES_BASE,
            field="futures_base_url",
        )
        self._spot_base_url = _validated_base_url(
            spot_base_url or self.SPOT_BASE,
            field="spot_base_url",
        )
        self._timeout_seconds = timeout_value
        self._recv_window_ms = recv_window_value
        self._proxy_url = _validated_proxy_url(proxy_url, field="proxy_url")
        if not isinstance(trust_env_proxy, bool):
            raise TypeError("trust_env_proxy must be a boolean")
        self._trust_env_proxy = trust_env_proxy
        self._http_session = session
        self._owns_http_session = session is None
        self._limiter = limiter or AdaptiveWeightLimiter()
        self._retry_policy = retry_policy or RetryPolicy()
        self._clock = clock or SystemClock()
        self._logical_request_count = 0
        self._attempt_count = 0
        self._retry_count = 0
        self._error_count = 0
        self._rate_limit_count = 0
        self._server_error_count = 0
        self._last_latency_ms: float | None = None
        self._last_status: int | None = None
        self._last_error: str | None = None
        self._last_response_at: datetime | None = None

    @property
    def telemetry(self) -> TransportTelemetry:
        return TransportTelemetry(
            logical_request_count=self._logical_request_count,
            attempt_count=self._attempt_count,
            retry_count=self._retry_count,
            error_count=self._error_count,
            rate_limit_count=self._rate_limit_count,
            server_error_count=self._server_error_count,
            used_weight=self._limiter.used_weight,
            last_latency_ms=self._last_latency_ms,
            last_status=self._last_status,
            last_error=self._last_error,
            last_response_at=self._last_response_at,
        )

    def set_timestamp_provider(self, provider: Callable[[], int]) -> None:
        self._timestamp_provider = provider

    async def close(self) -> None:
        if self._owns_http_session and self._http_session is not None:
            await self._http_session.close()
            self._http_session = None

    def _ensure_allowed(self, method: str, path: str) -> None:
        if (method.upper(), path) not in self._ALLOWED:
            raise BinancePermissionError(
                f"Endpoint is not in the read-only allowlist: {method.upper()} {path}"
            )

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        signed: bool = False,
        api_group: str = "futures",
        weight: int = 1,
    ) -> BinanceHttpResponse:
        method = method.upper()
        self._ensure_allowed(method, path)
        if api_group not in {"futures", "spot"}:
            raise ValueError(f"Unsupported Binance API group: {api_group!r}")
        expected_group = "spot" if path.startswith("/sapi/") else "futures"
        if api_group != expected_group:
            raise BinancePermissionError(
                f"Endpoint {path} belongs to {expected_group}, not {api_group}"
            )

        self._logical_request_count += 1
        attempts = 0

        async def operation() -> BinanceHttpResponse:
            nonlocal attempts
            attempts += 1
            self._attempt_count += 1
            started = self._clock.monotonic()
            await self._limiter.acquire(weight)
            try:
                response = await self._request_once(
                    method,
                    path,
                    params=params,
                    signed=signed,
                    api_group=api_group,
                )
            except Exception as exc:
                self._last_latency_ms = max(
                    0.0, (self._clock.monotonic() - started) * 1000.0
                )
                self._last_error = f"{type(exc).__name__}:{exc}"
                if isinstance(exc, BinanceRateLimitError):
                    self._rate_limit_count += 1
                if isinstance(exc, BinanceTransportError) and (exc.status or 0) >= 500:
                    self._server_error_count += 1
                raise
            self._last_latency_ms = max(
                0.0, (self._clock.monotonic() - started) * 1000.0
            )
            self._last_status = response.status
            self._last_error = None
            self._last_response_at = self._clock.now()
            return response

        try:
            return await run_with_retry(
                operation,
                policy=self._retry_policy,
                clock=self._clock,
            )
        except Exception:
            self._error_count += 1
            raise
        finally:
            self._retry_count += max(0, attempts - 1)

    def _request_parameters(
        self, params: Mapping[str, Any] | None, *, signed: bool
    ) -> dict[str, Any]:
        request_params = dict(params or {})
        protected = {"signature", "timestamp", "recvWindow"}.intersection(request_params)
        if protected:
            names = ", ".join(sorted(protected))
            raise BinanceDataError(
                f"Caller-supplied signed transport fields are not allowed: {names}"
            )
        if not signed:
            return request_params
        if self._timestamp_provider is None:
            raise BinanceDataError("Signed request attempted before clock synchronization")
        try:
            timestamp = _nonnegative_int(
                self._timestamp_provider(), field="timestamp"
            )
        except (ValueError, BinanceDataError) as exc:
            raise BinanceDataError("Timestamp provider returned an invalid value") from exc
        request_params["timestamp"] = timestamp
        request_params["recvWindow"] = self._recv_window_ms
        query = urlencode(request_params, doseq=True)
        request_params["signature"] = hmac.new(
            self._api_secret, query.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        return request_params

    def _http_client(self) -> aiohttp.ClientSession:
        if self._http_session is None:
            timeout = aiohttp.ClientTimeout(total=self._timeout_seconds)
            self._http_session = aiohttp.ClientSession(
                timeout=timeout,
                trust_env=self._trust_env_proxy,
            )
        return self._http_session

    @staticmethod
    async def _decode_response(response: aiohttp.ClientResponse) -> Any:
        try:
            return await response.json(content_type=None)
        except (aiohttp.ClientPayloadError, json.JSONDecodeError, UnicodeDecodeError):
            return {"msg": await response.text()}

    @staticmethod
    def _normalized_error_code(data: Any) -> int | str | None:
        code = data.get("code") if isinstance(data, Mapping) else None
        if code is None:
            return None
        try:
            return int(code)
        except (TypeError, ValueError):
            return str(code)

    @staticmethod
    def _error_message(data: Any) -> str:
        if isinstance(data, Mapping):
            return str(data.get("msg") or data)
        return str(data)

    @staticmethod
    def _raise_response_error(
        *,
        status: int,
        data: Any,
        headers: Mapping[str, str],
    ) -> None:
        code = AiohttpBinanceRestTransport._normalized_error_code(data)
        message = AiohttpBinanceRestTransport._error_message(data)
        retry_after = parse_retry_after(headers)
        if status in {418, 429}:
            raise BinanceRateLimitError(
                message,
                status=status,
                code=code,
                retry_after=retry_after,
                retryable=True,
                payload=data,
            )
        if status in {401, 403} or code in {-2014, -2015}:
            raise BinancePermissionError(
                message, status=status, code=code, retryable=False, payload=data
            )
        raise BinanceTransportError(
            message,
            status=status,
            code=code,
            retry_after=retry_after,
            retryable=status >= 500,
            payload=data,
        )

    async def _request_once(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None,
        signed: bool,
        api_group: str,
    ) -> BinanceHttpResponse:
        request_params = self._request_parameters(params, signed=signed)
        base = self._spot_base_url if api_group == "spot" else self._futures_base_url
        url = f"{base}{path}"
        requires_api_key = signed or path.startswith("/sapi/") or path.endswith(
            "/listenKey"
        )
        headers = {"X-MBX-APIKEY": self._api_key} if requires_api_key else {}
        try:
            request_kwargs: dict[str, Any] = {
                "params": request_params,
                "headers": headers,
            }
            if self._proxy_url is not None:
                request_kwargs["proxy"] = self._proxy_url
            async with self._http_client().request(
                method,
                url,
                **request_kwargs,
            ) as response:
                response_headers = {
                    str(key): str(value) for key, value in response.headers.items()
                }
                self._limiter.observe_headers(response_headers)
                data = await self._decode_response(response)
                if 200 <= response.status < 300:
                    return BinanceHttpResponse(
                        data=data, status=response.status, headers=response_headers
                    )
                self._raise_response_error(
                    status=response.status, data=data, headers=response_headers
                )
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise BinanceTransportError(str(exc), retryable=True) from exc
        raise RuntimeError("Binance response error handling returned unexpectedly")


class BinanceReadonlyClient:
    """USDⓈ-M account reader. No order, leverage or mode write method exists."""

    def __init__(
        self,
        *,
        transport: BinanceRestTransport,
        account_id: str,
        managed_symbols: Sequence[str],
        clock_sync: ClockSynchronizer | None = None,
        clock: Clock | None = None,
        max_collection_span_seconds: float = 15.0,
        system_client_order_prefixes: Sequence[str] = ("fthedge-",),
    ) -> None:
        if not account_id.strip():
            raise ValueError("account_id is required")
        self.transport = transport
        self.account_id = account_id.strip()
        if isinstance(managed_symbols, (str, bytes)):
            raise TypeError("managed_symbols must be a sequence of symbols, not a string")
        self.managed_symbols = normalize_exchange_symbols(list(managed_symbols))
        if not self.managed_symbols:
            raise ValueError("managed_symbols must not be empty")
        collection_span = _positive_finite_float(
            max_collection_span_seconds, field="max_collection_span_seconds"
        )
        self._managed_symbol_set = frozenset(self.managed_symbols)
        self.clock = clock or SystemClock()
        self.clock_sync = clock_sync or ClockSynchronizer(clock=self.clock)
        self.max_collection_span_seconds = collection_span
        prefixes = tuple(
            prefix.strip() for prefix in system_client_order_prefixes if prefix.strip()
        )
        self.system_client_order_prefixes = prefixes
        setter = getattr(self.transport, "set_timestamp_provider", None)
        if callable(setter):
            setter(self.clock_sync.timestamp_ms)

    async def fetch_server_time(self) -> int:
        response = await self.transport.request("GET", "/fapi/v1/time", signed=False)
        if not isinstance(response.data, Mapping) or "serverTime" not in response.data:
            raise BinanceDataError("Invalid /fapi/v1/time response")
        try:
            return _nonnegative_int(response.data["serverTime"], field="serverTime")
        except (ValueError, BinanceDataError) as exc:
            raise BinanceDataError("Invalid /fapi/v1/time response") from exc

    async def synchronize_clock(self) -> None:
        await self.clock_sync.sync(self.fetch_server_time)

    async def _fetch_api_restrictions(
        self, policy: PermissionPolicy
    ) -> tuple[Mapping[str, Any], tuple[str, ...]]:
        try:
            response = await self.transport.request(
                "GET",
                "/sapi/v1/account/apiRestrictions",
                signed=True,
                api_group="spot",
                weight=1,
            )
        except BinancePermissionError:
            if policy.require_restriction_endpoint:
                raise
            return {}, ("API_RESTRICTIONS_UNAVAILABLE",)
        except BinanceTransportError as exc:
            if (
                policy.allow_testnet_restriction_endpoint_unavailable
                and exc.status in {400, 404}
            ):
                return {}, ("API_RESTRICTIONS_UNAVAILABLE",)
            raise
        if not isinstance(response.data, Mapping):
            raise BinanceDataError("apiRestrictions returned invalid data")
        return response.data, ()

    @staticmethod
    def _restriction_flag(
        restrictions: Mapping[str, Any], name: str
    ) -> bool | None:
        value = restrictions.get(name)
        if value is None:
            return None
        return strict_bool(value, field=f"apiRestrictions.{name}")

    @staticmethod
    def _permission_reasons(
        policy: PermissionPolicy,
        *,
        read_enabled: bool | None,
        futures_enabled: bool | None,
        withdrawals: bool | None,
        internal_transfer: bool | None,
        universal_transfer: bool | None,
        spot_margin: bool | None,
    ) -> tuple[str, ...]:
        checks = (
            (policy.require_read_enabled and read_enabled is not True, "READ_PERMISSION_MISSING"),
            (policy.forbid_withdrawals and withdrawals is True, "WITHDRAWAL_PERMISSION_ENABLED"),
            (
                policy.forbid_internal_transfers and internal_transfer is True,
                "INTERNAL_TRANSFER_PERMISSION_ENABLED",
            ),
            (
                policy.forbid_universal_transfers and universal_transfer is True,
                "UNIVERSAL_TRANSFER_PERMISSION_ENABLED",
            ),
            (
                policy.forbid_spot_margin_trading and spot_margin is True,
                "SPOT_MARGIN_TRADING_PERMISSION_ENABLED",
            ),
            (
                policy.require_futures_enabled and futures_enabled is not True,
                "FUTURES_PERMISSION_MISSING",
            ),
            (
                policy.forbid_futures_trading_permission and futures_enabled is True,
                "FUTURES_TRADING_PERMISSION_ENABLED",
            ),
        )
        return tuple(reason for failed, reason in checks if failed)

    async def preflight_permissions(
        self, policy: PermissionPolicy | None = None
    ) -> ApiPermissionReport:
        effective = policy or PermissionPolicy()
        restrictions, initial_reasons = await self._fetch_api_restrictions(effective)
        read_enabled = self._restriction_flag(restrictions, "enableReading")
        futures_enabled = self._restriction_flag(restrictions, "enableFutures")
        withdrawals = self._restriction_flag(restrictions, "enableWithdrawals")
        internal_transfer = self._restriction_flag(
            restrictions, "enableInternalTransfer"
        )
        universal_transfer = self._restriction_flag(
            restrictions, "permitsUniversalTransfer"
        )
        spot_margin = self._restriction_flag(
            restrictions, "enableSpotAndMarginTrading"
        )
        permission_reasons = self._permission_reasons(
            effective,
            read_enabled=read_enabled,
            futures_enabled=futures_enabled,
            withdrawals=withdrawals,
            internal_transfer=internal_transfer,
            universal_transfer=universal_transfer,
            spot_margin=spot_margin,
        )
        warnings: tuple[str, ...] = ()
        if effective.require_futures_enabled and futures_enabled is not True:
            try:
                await self.fetch_account_info()
            except BinanceTransportError as exc:
                if exc.status not in {401, 403} and exc.code not in {-2014, -2015}:
                    raise
                warnings = warnings + (
                    (
                        "BINANCE_ENABLE_FUTURES_FALSE_OR_MISSING;"
                        "AUTHENTICATED_USDM_USER_DATA_READ_DENIED"
                    ),
                )
            else:
                permission_reasons = tuple(
                    reason
                    for reason in permission_reasons
                    if reason != "FUTURES_PERMISSION_MISSING"
                )
                warnings = warnings + (
                    (
                        "BINANCE_ENABLE_FUTURES_FALSE_OR_MISSING;"
                        "AUTHENTICATED_USDM_USER_DATA_READ_CONFIRMED"
                    ),
                )
        if (
            initial_reasons == ("API_RESTRICTIONS_UNAVAILABLE",)
            and effective.allow_testnet_restriction_endpoint_unavailable
        ):
            # Binance Demo/Testnet credentials are not accepted by the production
            # SAPI restriction endpoint.  Authenticated USD-M account/bootstrap calls
            # immediately following this preflight remain the authority for access.
            read_enabled = True
            futures_enabled = True
            permission_reasons = tuple(
                reason
                for reason in permission_reasons
                if reason not in {"READ_PERMISSION_MISSING", "FUTURES_PERMISSION_MISSING"}
            )
            initial_reasons = ()
            warnings = (
                (
                    "TESTNET_API_RESTRICTIONS_UNAVAILABLE;"
                    "FUTURES_ACCESS_MUST_BE_PROVEN_BY_AUTHENTICATED_USDM_BOOTSTRAP"
                ),
            )
        reasons = initial_reasons + permission_reasons
        if futures_enabled is True:
            warnings = warnings + (
                (
                    "BINANCE_FUTURES_CREDENTIAL_WRITE_PERMISSION_NOT_SEPARATELY_VERIFIABLE;"
                    "APPLICATION_ENDPOINT_ALLOWLIST_REQUIRED"
                ),
            )
        report = ApiPermissionReport(
            read_enabled=read_enabled is True,
            futures_enabled=futures_enabled,
            withdrawals_enabled=withdrawals,
            internal_transfer_enabled=internal_transfer,
            universal_transfer_enabled=universal_transfer,
            spot_margin_trading_enabled=spot_margin,
            strict_readonly_verified=not reasons,
            warnings=warnings,
            runtime_readonly_enforced=True,
            reasons=reasons,
            raw=restrictions,
        )
        if report.strict_readonly_verified:
            return report
        raise BinancePermissionError(
            "Binance API key failed read-only permission preflight: "
            + ",".join(report.reasons),
            payload=restrictions,
        )

    async def fetch_real_market_prices(self, symbol: str) -> tuple[Decimal, Decimal, Decimal]:
        """Return real Binance best bid/ask and mark using public read-only endpoints."""
        normalized = to_binance_symbol(symbol)
        book, premium = await asyncio.gather(
            self.transport.request(
                "GET", "/fapi/v1/ticker/bookTicker",
                params={"symbol": normalized}, signed=False, weight=1,
            ),
            self.transport.request(
                "GET", "/fapi/v1/premiumIndex",
                params={"symbol": normalized}, signed=False, weight=1,
            ),
        )
        if not isinstance(book.data, Mapping) or not isinstance(premium.data, Mapping):
            raise BinanceDataError("public market endpoint returned invalid data")
        bid = finite_decimal(book.data.get("bidPrice"), field="bookTicker.bidPrice")
        ask = finite_decimal(book.data.get("askPrice"), field="bookTicker.askPrice")
        mark = finite_decimal(premium.data.get("markPrice"), field="premiumIndex.markPrice")
        if bid <= 0 or ask <= 0 or mark <= 0 or bid > ask:
            raise BinanceDataError("public market endpoint returned invalid prices")
        return bid, ask, mark

    async def fetch_account_info(self) -> Mapping[str, Any]:
        for path in ("/fapi/v3/account", "/fapi/v2/account"):
            try:
                response = await self.transport.request("GET", path, signed=True, weight=5)
                if isinstance(response.data, Mapping):
                    return response.data
                raise BinanceDataError(f"{path} returned invalid data")
            except BinanceTransportError as exc:
                if exc.status == 404 or exc.code in {-4046, -1000}:
                    continue
                raise
        raise BinanceDataError("No supported Binance account endpoint succeeded")

    async def fetch_positions(
        self,
        symbols: Sequence[str] | None = None,
        *,
        symbol_configurations: Sequence[Mapping[str, Any]] | None = None,
    ) -> tuple[PositionFact, ...]:
        params: dict[str, Any] = {}
        if symbols is not None:
            if len(symbols) != 1:
                raise ValueError("Binance positionRisk requires exactly one symbol or None")
            params["symbol"] = to_binance_symbol(str(symbols[0]))
        for path in ("/fapi/v3/positionRisk", "/fapi/v2/positionRisk"):
            try:
                response = await self.transport.request(
                    "GET", path, params=params, signed=True, weight=5
                )
                data = response.data
                if not isinstance(data, Sequence) or isinstance(data, (str, bytes)):
                    raise BinanceDataError(f"{path} returned invalid data")
                position_configurations = symbol_configurations
                if path == "/fapi/v3/positionRisk" and position_configurations is None:
                    requires_configuration = any(
                        isinstance(item, Mapping)
                        and (
                            (
                                item.get("marginType", item.get("mt")) in (None, "")
                                and "isolated" not in item
                            )
                            or item.get("leverage") in (None, "", 0, "0")
                        )
                        for item in data
                    )
                    if requires_configuration:
                        position_configurations = await self.fetch_symbol_configuration(None)
                return normalize_positions(
                    data,
                    account_id=self.account_id,
                    observed_at=self.clock.now(),
                    symbol_configurations=position_configurations,
                )
            except BinanceTransportError as exc:
                if exc.status == 404:
                    continue
                raise
        raise BinanceDataError("No supported Binance position endpoint succeeded")

    async def fetch_open_orders(self, symbol: str | None = None) -> tuple[OrderFact, ...]:
        if symbol is None:
            params: dict[str, Any] = {}
        else:
            params = {"symbol": to_binance_symbol(symbol)}
        response = await self.transport.request(
            "GET", "/fapi/v1/openOrders", params=params, signed=True, weight=40 if not symbol else 1
        )
        if not isinstance(response.data, Sequence) or isinstance(response.data, (str, bytes)):
            raise BinanceDataError("openOrders returned invalid data")
        return normalize_orders(
            response.data,
            account_id=self.account_id,
            observed_at=self.clock.now(),
            system_client_order_prefixes=self.system_client_order_prefixes,
        )

    async def fetch_order_status(
        self, symbol: str, *, order_id: str
    ) -> OrderFact:
        """Read one order by exchange id without changing exchange state."""
        normalized_order_id = str(order_id).strip()
        if not normalized_order_id:
            raise ValueError("order_id is required")
        response = await self.transport.request(
            "GET",
            "/fapi/v1/order",
            params={"symbol": to_binance_symbol(symbol), "orderId": normalized_order_id},
            signed=True,
            weight=1,
        )
        if not isinstance(response.data, Mapping):
            raise BinanceDataError("query order returned invalid data")
        return normalize_order(
            response.data,
            account_id=self.account_id,
            observed_at=self.clock.now(),
            source="BINANCE_REST_QUERY_ORDER",
            system_client_order_prefixes=self.system_client_order_prefixes,
        )

    async def _cover_open_orders_in_history(
        self,
        open_orders: Sequence[OrderFact],
        order_history: Sequence[OrderFact],
    ) -> tuple[tuple[OrderFact, ...], tuple[str, ...], tuple[str, ...]]:
        """Ensure current open orders remain represented outside the bounded history window.

        Binance ``allOrders`` is time-windowed while ``openOrders`` is a current-state
        surface.  An old still-open GTC order can therefore be absent from a short
        history lookback.  Recover missing identities through the read-only Query
        Order endpoint.  If Binance's documented retention prevents that lookup,
        retain the current ``openOrders`` fact as explicitly-labelled authoritative
        snapshot evidence instead of producing a false acceptance failure.
        """
        combined = list(order_history)
        known = {item.key for item in order_history}
        query_recovered: list[str] = []
        snapshot_fallback: list[str] = []
        for item in open_orders:
            if not item.active or item.key in known:
                continue
            try:
                recovered = await self.fetch_order_status(
                    item.symbol, order_id=item.exchange_order_id
                )
            except BinanceTransportError as exc:
                if exc.code not in {-2011, -2013} and exc.status != 404:
                    raise
                recovered = replace(
                    item,
                    source="BINANCE_REST_OPEN_ORDERS_RETENTION_FALLBACK",
                )
                snapshot_fallback.append(item.exchange_order_id)
            if recovered.key != item.key:
                raise BinanceDataError(
                    "Query Order identity mismatch for current open order "
                    f"{item.symbol}:{item.exchange_order_id}"
                )
            combined.append(recovered)
            known.add(recovered.key)
            if recovered.source == "BINANCE_REST_QUERY_ORDER":
                query_recovered.append(item.exchange_order_id)
        return (
            _deduplicate_orders(combined),
            tuple(sorted(query_recovered)),
            tuple(sorted(snapshot_fallback)),
        )

    def _history_windows(
        self,
        *,
        start_time_ms: int | None,
        end_time_ms: int | None,
    ) -> tuple[tuple[int, int], ...]:
        now_ms = int(self.clock.now().timestamp() * 1000)
        query_end = (
            _nonnegative_int(end_time_ms, field="end_time_ms")
            if end_time_ms is not None
            else now_ms
        )
        default_start = max(0, query_end - (7 * 24 * 60 * 60 * 1000 - 1))
        query_start = (
            _nonnegative_int(start_time_ms, field="start_time_ms")
            if start_time_ms is not None
            else default_start
        )
        return _time_windows(query_start, query_end)

    @staticmethod
    def _next_numeric_id(
        values: Sequence[str], *, endpoint: str, symbol: str
    ) -> int:
        numeric_ids = [int(value) for value in values if value.lstrip("-").isdigit()]
        if not numeric_ids:
            raise BinanceDataError(
                f"Cannot advance {endpoint} pagination for {symbol}"
            )
        return max(numeric_ids) + 1

    async def _fetch_fill_window(
        self,
        symbol: str,
        *,
        window_start: int,
        window_end: int,
        page_limit: int,
        request_budget: int,
    ) -> tuple[tuple[FillFact, ...], int]:
        facts: list[FillFact] = []
        from_id: int | None = None
        requests = 0
        while requests < request_budget:
            params: dict[str, Any] = {"symbol": symbol, "limit": page_limit}
            if from_id is None:
                params.update({"startTime": window_start, "endTime": window_end})
            else:
                params["fromId"] = from_id
            response = await self.transport.request(
                "GET",
                "/fapi/v1/userTrades",
                params=params,
                signed=True,
                weight=5,
            )
            requests += 1
            if not isinstance(response.data, Sequence) or isinstance(
                response.data, (str, bytes)
            ):
                raise BinanceDataError("userTrades returned invalid data")
            page = normalize_fills(
                response.data,
                account_id=self.account_id,
                observed_at=self.clock.now(),
            )
            facts.extend(
                item for item in page if window_start <= item.event_time_ms <= window_end
            )
            crossed_window = any(item.event_time_ms > window_end for item in page)
            if len(response.data) < page_limit or crossed_window:
                return tuple(facts), requests
            next_from_id = self._next_numeric_id(
                [item.exchange_trade_id for item in page],
                endpoint="userTrades",
                symbol=symbol,
            )
            if from_id is not None and next_from_id <= from_id:
                raise BinanceDataError(
                    f"userTrades pagination did not advance for {symbol}"
                )
            from_id = next_from_id
        raise BinanceDataError(
            f"userTrades exceeded max_pages={request_budget} for {symbol}"
        )

    async def fetch_fills(
        self,
        symbols: Sequence[str],
        *,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
        max_pages: int = 100,
    ) -> tuple[FillFact, ...]:
        request_budget = _positive_int(max_pages, field="max_pages")
        page_limit = _positive_int(limit, field="limit", maximum=1000)
        windows = self._history_windows(
            start_time_ms=start_time_ms, end_time_ms=end_time_ms
        )
        facts: list[FillFact] = []
        for symbol in _normalize_symbols(symbols):
            requests = 0
            for window_start, window_end in windows:
                page, used = await self._fetch_fill_window(
                    symbol,
                    window_start=window_start,
                    window_end=window_end,
                    page_limit=page_limit,
                    request_budget=request_budget - requests,
                )
                facts.extend(page)
                requests += used
        return _deduplicate_fills(facts)

    async def _fetch_order_window(
        self,
        symbol: str,
        *,
        window_start: int,
        window_end: int,
        page_limit: int,
        request_budget: int,
    ) -> tuple[tuple[OrderFact, ...], int]:
        facts: list[OrderFact] = []
        next_order_id: int | None = None
        requests = 0
        while requests < request_budget:
            params: dict[str, Any] = {
                "symbol": symbol,
                "limit": page_limit,
                "startTime": window_start,
                "endTime": window_end,
            }
            if next_order_id is not None:
                params["orderId"] = next_order_id
            response = await self.transport.request(
                "GET",
                "/fapi/v1/allOrders",
                params=params,
                signed=True,
                weight=5,
            )
            requests += 1
            if not isinstance(response.data, Sequence) or isinstance(
                response.data, (str, bytes)
            ):
                raise BinanceDataError("allOrders returned invalid data")
            page = normalize_orders(
                response.data,
                account_id=self.account_id,
                observed_at=self.clock.now(),
                system_client_order_prefixes=self.system_client_order_prefixes,
            )
            facts.extend(page)
            if len(response.data) < page_limit:
                return tuple(facts), requests
            candidate = self._next_numeric_id(
                [item.exchange_order_id for item in page],
                endpoint="allOrders",
                symbol=symbol,
            )
            if next_order_id is not None and candidate <= next_order_id:
                raise BinanceDataError(
                    f"allOrders pagination did not advance for {symbol}"
                )
            next_order_id = candidate
        raise BinanceDataError(
            f"allOrders exceeded max_pages={request_budget} for {symbol}"
        )

    async def fetch_all_orders(
        self,
        symbols: Sequence[str],
        *,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
        max_pages: int = 100,
    ) -> tuple[OrderFact, ...]:
        request_budget = _positive_int(max_pages, field="max_pages")
        page_limit = _positive_int(limit, field="limit", maximum=1000)
        windows = self._history_windows(
            start_time_ms=start_time_ms, end_time_ms=end_time_ms
        )
        facts: list[OrderFact] = []
        for symbol in _normalize_symbols(symbols):
            requests = 0
            for window_start, window_end in windows:
                page, used = await self._fetch_order_window(
                    symbol,
                    window_start=window_start,
                    window_end=window_end,
                    page_limit=page_limit,
                    request_budget=request_budget - requests,
                )
                facts.extend(page)
                requests += used
        return _deduplicate_orders(facts)

    async def fetch_position_mode(self) -> Mapping[str, Any]:
        response = await self.transport.request(
            "GET", "/fapi/v1/positionSide/dual", signed=True, weight=30
        )
        if not isinstance(response.data, Mapping):
            raise BinanceDataError("positionSide/dual returned invalid data")
        return response.data

    @staticmethod
    def _income_parameters(
        *,
        start_time_ms: int | None,
        end_time_ms: int | None,
        page_limit: int,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": page_limit}
        if start_time_ms is not None:
            params["startTime"] = _nonnegative_int(
                start_time_ms, field="start_time_ms"
            )
        if end_time_ms is not None:
            params["endTime"] = _nonnegative_int(end_time_ms, field="end_time_ms")
        if (
            "startTime" in params
            and "endTime" in params
            and params["endTime"] < params["startTime"]
        ):
            raise ValueError("end_time_ms must be >= start_time_ms")
        return params

    async def _fetch_income_pages(
        self,
        *,
        base_params: Mapping[str, Any],
        page_limit: int,
        page_budget: int,
    ) -> tuple[Mapping[str, Any], ...]:
        items: list[Mapping[str, Any]] = []
        for page_number in range(1, page_budget + 1):
            response = await self.transport.request(
                "GET",
                "/fapi/v1/income",
                params={**base_params, "page": page_number},
                signed=True,
                weight=30,
            )
            if not isinstance(response.data, Sequence) or isinstance(
                response.data, (str, bytes)
            ):
                raise BinanceDataError("income returned invalid data")
            page = tuple(item for item in response.data if isinstance(item, Mapping))
            if len(page) != len(response.data):
                raise BinanceDataError("income contains a non-object item")
            items.extend(page)
            if len(page) < page_limit:
                return tuple(items)
        raise BinanceDataError(f"income history exceeded max_pages={page_budget}")

    @staticmethod
    def _income_identity(item: Mapping[str, Any]) -> tuple[tuple[Any, ...], int]:
        income_type = str(item.get("incomeType") or "").strip().upper()
        if not income_type:
            raise BinanceDataError("incomeType is required")
        event_time = _nonnegative_int(item.get("time"), field="income.time")
        finite_decimal(item.get("income"), field="income.income", default="0")
        transaction_id = item.get("tranId")
        if transaction_id not in (None, ""):
            return (income_type, str(transaction_id)), event_time
        return (
            income_type,
            str(item.get("tradeId") or ""),
            event_time,
            str(item.get("symbol") or ""),
            str(item.get("asset") or ""),
            str(item.get("income") or ""),
            str(item.get("info") or ""),
        ), event_time

    @classmethod
    def _deduplicate_income(
        cls, items: Sequence[Mapping[str, Any]]
    ) -> tuple[Mapping[str, Any], ...]:
        deduplicated: dict[tuple[Any, ...], tuple[int, Mapping[str, Any]]] = {}
        for item in items:
            key, event_time = cls._income_identity(item)
            previous = deduplicated.get(key)
            if previous is not None and dict(previous[1]) != dict(item):
                raise BinanceDataError(
                    f"Conflicting duplicate income record: {key!r}"
                )
            deduplicated[key] = event_time, item
        ordered = sorted(
            deduplicated.values(),
            key=lambda pair: (
                pair[0],
                str(pair[1].get("incomeType") or ""),
                str(pair[1].get("tranId") or ""),
            ),
        )
        return tuple(item for _event_time, item in ordered)

    async def fetch_income_history(
        self,
        *,
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
        limit: int = 1000,
        max_pages: int = 100,
    ) -> tuple[Mapping[str, Any], ...]:
        page_budget = _positive_int(max_pages, field="max_pages")
        page_limit = _positive_int(limit, field="limit", maximum=1000)
        params = self._income_parameters(
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            page_limit=page_limit,
        )
        items = await self._fetch_income_pages(
            base_params=params, page_limit=page_limit, page_budget=page_budget
        )
        return self._deduplicate_income(items)

    async def fetch_symbol_configuration(
        self, symbol: str | None = None
    ) -> tuple[Mapping[str, Any], ...]:
        params: dict[str, Any] = {}
        if symbol is not None:
            params["symbol"] = to_binance_symbol(symbol)
        response = await self.transport.request(
            "GET", "/fapi/v1/symbolConfig", params=params, signed=True, weight=5
        )
        if not isinstance(response.data, Sequence) or isinstance(
            response.data, (str, bytes)
        ):
            raise BinanceDataError("symbolConfig returned invalid data")
        rows = tuple(item for item in response.data if isinstance(item, Mapping))
        if len(rows) != len(response.data):
            raise BinanceDataError("symbolConfig contains a non-object item")
        return rows

    async def confirm_configuration(
        self, symbol_configurations: Sequence[Mapping[str, Any]] | None = None
    ) -> AccountConfigurationFact:
        if symbol_configurations is None:
            mode, fetched_rows = await asyncio.gather(
                self.fetch_position_mode(),
                self.fetch_symbol_configuration(None),
            )
            rows = fetched_rows
        else:
            mode = await self.fetch_position_mode()
            rows = tuple(symbol_configurations)
        configuration = normalize_configuration(
            account_id=self.account_id,
            dual_side_payload=mode,
            symbol_configurations=rows,
            managed_symbols=self.managed_symbols,
            observed_at=self.clock.now(),
        )
        if not configuration.hedge_mode:
            raise BinancePermissionError("Binance account is not in Hedge Mode")
        if configuration.active_margin_modes != ("cross",):
            raise BinancePermissionError(
                "Managed symbols are not uniformly configured for Cross margin"
            )
        invalid_leverage = sorted(
            key
            for key, value in configuration.leverage_by_symbol_side.items()
            if value <= 0
        )
        if invalid_leverage:
            raise BinancePermissionError(
                "Managed symbol leverage could not be confirmed: "
                + ",".join(invalid_leverage)
            )
        return configuration

    @staticmethod
    def _contains_unmanaged_facts(
        positions: Sequence[PositionFact],
        orders: Sequence[OrderFact],
        *,
        managed_symbols: frozenset[str],
    ) -> bool:
        unmanaged_position = any(
            item.quantity > 0 and item.symbol not in managed_symbols
            for item in positions
        )
        unmanaged_order = any(
            item.active and item.symbol not in managed_symbols for item in orders
        )
        return unmanaged_position or unmanaged_order

    @staticmethod
    def _account_payload_history_symbols(
        account_payload: Mapping[str, Any],
    ) -> set[str]:
        result: set[str] = set()
        raw_positions = account_payload.get("positions", ())
        if not isinstance(raw_positions, Sequence) or isinstance(
            raw_positions, (str, bytes)
        ):
            return result
        for item in raw_positions:
            if not isinstance(item, Mapping):
                continue
            symbol = str(item.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            position_amount = finite_decimal(
                item.get("positionAmt"),
                field=f"{symbol}.positionAmt",
                default="0",
            )
            order_margin = finite_decimal(
                item.get("openOrderInitialMargin"),
                field=f"{symbol}.openOrderInitialMargin",
                default="0",
            )
            if position_amount != 0 or order_margin != 0:
                try:
                    result.add(to_binance_symbol(symbol))
                except ValueError as exc:
                    raise BinanceDataError(
                        "Unsupported non-perpetual account exposure: " + symbol
                    ) from exc
        return result

    @classmethod
    def _discovered_history_symbols(
        cls,
        account_payload: Mapping[str, Any],
        positions: Sequence[PositionFact],
        orders: Sequence[OrderFact],
        income: Sequence[Mapping[str, Any]],
        managed_symbols: Sequence[str],
    ) -> tuple[str, ...]:
        symbols = set(managed_symbols)
        symbols.update(item.symbol for item in positions if item.quantity != 0)
        symbols.update(item.symbol for item in orders if item.active)
        for item in income:
            raw_symbol = str(item.get("symbol") or "").strip().upper()
            if not raw_symbol:
                continue
            try:
                symbols.add(to_binance_symbol(raw_symbol))
            except ValueError:
                # Historical delivery-contract income is outside the managed
                # perpetual universe.  Current exposure is checked separately
                # through positions/open orders and fails closed there.
                continue
        symbols.update(cls._account_payload_history_symbols(account_payload))
        return tuple(sorted(symbols))

    async def _fetch_history_bundle(
        self,
        *,
        include_fills: bool,
        fill_start_time_ms: int | None,
        account_payload: Mapping[str, Any],
        positions: Sequence[PositionFact],
        orders: Sequence[OrderFact],
    ) -> tuple[
        tuple[OrderFact, ...],
        tuple[FillFact, ...],
        tuple[Mapping[str, Any], ...],
        tuple[str, ...],
        tuple[str, ...],
    ]:
        if not include_fills:
            return (), (), (), (), ()
        income = await self.fetch_income_history(start_time_ms=fill_start_time_ms)
        symbols = self._discovered_history_symbols(
            account_payload, positions, orders, income, self.managed_symbols
        )
        order_history, fills = await asyncio.gather(
            self.fetch_all_orders(symbols, start_time_ms=fill_start_time_ms),
            self.fetch_fills(symbols, start_time_ms=fill_start_time_ms),
        )
        order_history, _query_recovered, _snapshot_fallback = (
            await self._cover_open_orders_in_history(orders, order_history)
        )
        return order_history, fills, income, _query_recovered, _snapshot_fallback

    async def fetch_bundle(
        self,
        *,
        include_fills: bool,
        fill_start_time_ms: int | None = None,
    ) -> BinanceAccountBundle:
        started = self.clock.now()
        state_started_mono = self.clock.monotonic()
        account_payload, positions, orders, configuration = await asyncio.gather(
            self.fetch_account_info(),
            self.fetch_positions(None),
            self.fetch_open_orders(None),
            self.confirm_configuration(),
        )
        state_completed = self.clock.now()
        state_span = self.clock.monotonic() - state_started_mono
        if state_span > self.max_collection_span_seconds:
            raise BinanceDataError(
                f"Account state collection span {state_span:.3f}s exceeds "
                f"{self.max_collection_span_seconds:.3f}s"
            )
        snapshot, balances = normalize_account_snapshot(
            account_payload,
            account_id=self.account_id,
            collection_started_at=started,
            collection_completed_at=state_completed,
        )
        history = await self._fetch_history_bundle(
            include_fills=include_fills,
            fill_start_time_ms=fill_start_time_ms,
            account_payload=account_payload,
            positions=positions,
            orders=orders,
        )
        (
            order_history,
            fills,
            income,
            query_recovered_ids,
            snapshot_fallback_ids,
        ) = history
        return BinanceAccountBundle(
            account_snapshot=snapshot,
            balances=balances,
            positions=positions,
            open_orders=orders,
            fills=fills,
            configuration=configuration,
            income_events=income,
            collection_started_at=started,
            collection_completed_at=self.clock.now(),
            order_history=order_history,
            order_history_query_recovered_ids=query_recovered_ids,
            order_history_snapshot_fallback_ids=snapshot_fallback_ids,
        )
