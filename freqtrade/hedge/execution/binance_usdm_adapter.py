"""Concrete synchronous Binance USD-M execution adapter.

The adapter deliberately performs no automatic retry for POST/DELETE requests. Network
and 5xx failures are surfaced as ambiguous outcomes so ExecutionService transitions the
order to UNKNOWN and queries before any possible retry.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from threading import RLock
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlencode, urlsplit

from .binance_environment import (
    profile_for_environment,
    validate_rest_base_url,
)
from .production_gate import (
    ExecutionEnvironment,
    ExecutionWriteLockedError,
    ProductionExecutionGate,
)
from .service import (
    ApprovedOrderIntent,
    DefinitiveCancellationError,
    DefinitiveSubmissionError,
    ExchangeExecutionPort,
    ExecutionOrder,
    ExecutionStorePort,
    ExternalOrderSnapshot,
    IntentAction,
    OrderType,
    PositionSide,
)
from .state_machine import OrderState
from freqtrade.hedge.exchange.shared_rate_limit import SqliteSharedWeightBudget


_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_STATUS_MAP = {
    "NEW": OrderState.ACKNOWLEDGED,
    "PARTIALLY_FILLED": OrderState.PARTIAL,
    "FILLED": OrderState.FILLED,
    "CANCELED": OrderState.CANCELED,
    "REJECTED": OrderState.REJECTED,
    "EXPIRED": OrderState.CANCELED,
    "EXPIRED_IN_MATCH": OrderState.CANCELED,
}


@dataclass(frozen=True, slots=True)
class BinanceExecutionCredentials:
    api_key: str
    api_secret: str

    def __post_init__(self) -> None:
        key = str(self.api_key).strip()
        secret = str(self.api_secret).strip()
        if not key or not secret:
            raise ValueError("Binance API key and secret are required")
        if len(key) > 512 or len(secret) > 512:
            raise ValueError("Binance credentials are invalid")
        object.__setattr__(self, "api_key", key)
        object.__setattr__(self, "api_secret", secret)

    def __repr__(self) -> str:
        return "BinanceExecutionCredentials(api_key=<redacted>, api_secret=<redacted>)"


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    payload: bytes


class HttpTransport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse: ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects so signed URLs and API-key headers never change origin."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


class UrlLibHttpTransport:
    def __init__(self, *, proxy_url: str | None = None) -> None:
        handlers: list[urllib.request.BaseHandler] = [_NoRedirectHandler()]
        if proxy_url:
            parsed = urlsplit(proxy_url.strip())
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
                raise ValueError("proxy_url must be http(s)://host:port")
            handlers.append(
                urllib.request.ProxyHandler({"http": proxy_url, "https": proxy_url})
            )
        self._opener = urllib.request.build_opener(*handlers)

    def request(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HttpResponse:
        request = urllib.request.Request(url, headers=dict(headers), method=method)
        try:
            with self._opener.open(request, timeout=timeout) as response:
                payload = response.read(_MAX_RESPONSE_BYTES + 1)
                if len(payload) > _MAX_RESPONSE_BYTES:
                    raise RuntimeError("Binance response exceeded safe size")
                return HttpResponse(int(response.status), dict(response.headers.items()), payload)
        except urllib.error.HTTPError as exc:
            return HttpResponse(
                int(exc.code),
                dict(exc.headers.items()),
                exc.read(_MAX_RESPONSE_BYTES + 1),
            )
        except (TimeoutError, OSError) as exc:
            raise TimeoutError("Binance transport outcome is unknown") from exc


@dataclass(frozen=True, slots=True)
class BinanceExecutionTelemetry:
    logical_requests: int
    write_requests: int
    read_requests: int
    ambiguous_write_errors: int
    definitive_write_errors: int
    last_status: int | None
    used_weight: int | None
    order_count: int | None




@dataclass(frozen=True, slots=True)
class BinanceTestOrderValidation:
    client_order_id: str
    environment: ExecutionEnvironment
    symbol: str
    accepted: bool
    validated_at: datetime

class BinanceExecutionApiError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: int | None = None,
        ambiguous: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.ambiguous = ambiguous


class BinanceUSDMExecutionAdapter(ExchangeExecutionPort):
    """Production adapter constrained by ProductionExecutionGate."""

    def __init__(
        self,
        *,
        credentials: BinanceExecutionCredentials,
        gate: ProductionExecutionGate,
        store: ExecutionStorePort,
        proxy_url: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 10.0,
        recv_window_ms: int = 5000,
        transport: HttpTransport | None = None,
        now_ms: Callable[[], int] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        shared_weight_budget: SqliteSharedWeightBudget | None = None,
    ) -> None:
        if not isinstance(credentials, BinanceExecutionCredentials):
            raise TypeError("credentials must be BinanceExecutionCredentials")
        if not isinstance(gate, ProductionExecutionGate):
            raise TypeError("gate must be ProductionExecutionGate")
        if not callable(getattr(store, "get_by_client_order_id", None)):
            raise TypeError("store must implement ExecutionStorePort")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not isinstance(recv_window_ms, int) or not 1000 <= recv_window_ms <= 5000:
            raise ValueError("recv_window_ms must be in [1000, 5000]")
        evidence = gate.evidence
        selected_base = base_url or profile_for_environment(evidence.environment).rest_base_url
        selected_base = validate_rest_base_url(
            evidence.environment,
            selected_base,
            allow_test_override=transport is not None,
        )
        self._credentials = credentials
        self._gate = gate
        self._store = store
        self._base_url = selected_base
        self._timeout = float(timeout_seconds)
        self._recv_window_ms = recv_window_ms
        self._transport = transport or UrlLibHttpTransport(proxy_url=proxy_url)
        self._now_ms = now_ms or (lambda: time.time_ns() // 1_000_000)
        self._sleep = sleep
        self._shared_weight_budget = shared_weight_budget
        self._clock_offset_ms = 0
        self._lock = RLock()
        self._logical_requests = 0
        self._write_requests = 0
        self._read_requests = 0
        self._ambiguous_write_errors = 0
        self._definitive_write_errors = 0
        self._last_status: int | None = None
        self._used_weight: int | None = None
        self._order_count: int | None = None

    def synchronize_clock(self) -> int:
        before = self._now_ms()
        response = self._request_public("GET", "/fapi/v1/time", {})
        after = self._now_ms()
        if not isinstance(response, Mapping) or not isinstance(response.get("serverTime"), int):
            raise BinanceExecutionApiError("invalid Binance server time response")
        midpoint = (before + after) // 2
        self._clock_offset_ms = int(response["serverTime"]) - midpoint
        return self._clock_offset_ms

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def environment(self) -> ExecutionEnvironment:
        return self._gate.evidence.environment

    def validate_order(self, approved: ApprovedOrderIntent) -> BinanceTestOrderValidation:
        """Validate one order through Binance ``/order/test`` without matching-engine state.

        The same gate, symbol allowlist, account binding, notional limit and two-step
        arming used for a real order are applied before the validation request.
        """

        self._gate.assert_order_allowed(approved)
        params = self._new_order_params(approved)
        try:
            self._request_signed("POST", "/fapi/v1/order/test", params, write=True)
        except ExecutionWriteLockedError:
            raise
        except BinanceExecutionApiError as exc:
            if exc.ambiguous:
                with self._lock:
                    self._ambiguous_write_errors += 1
                raise TimeoutError("Binance test-order outcome is unknown") from exc
            with self._lock:
                self._definitive_write_errors += 1
            raise DefinitiveSubmissionError(_safe_error(exc)) from exc
        return BinanceTestOrderValidation(
            client_order_id=approved.client_order_id,
            environment=self.environment,
            symbol=approved.intent.symbol,
            accepted=True,
            validated_at=datetime.now(UTC),
        )

    def telemetry(self) -> BinanceExecutionTelemetry:
        with self._lock:
            return BinanceExecutionTelemetry(
                logical_requests=self._logical_requests,
                write_requests=self._write_requests,
                read_requests=self._read_requests,
                ambiguous_write_errors=self._ambiguous_write_errors,
                definitive_write_errors=self._definitive_write_errors,
                last_status=self._last_status,
                used_weight=self._used_weight,
                order_count=self._order_count,
            )

    def submit_order(self, approved: ApprovedOrderIntent) -> ExternalOrderSnapshot:
        self._gate.assert_order_allowed(approved)
        params = self._new_order_params(approved)
        try:
            payload = self._request_signed("POST", "/fapi/v1/order", params, write=True)
        except ExecutionWriteLockedError:
            raise
        except BinanceExecutionApiError as exc:
            if exc.ambiguous:
                with self._lock:
                    self._ambiguous_write_errors += 1
                raise TimeoutError("Binance submit outcome is unknown") from exc
            with self._lock:
                self._definitive_write_errors += 1
            raise DefinitiveSubmissionError(_safe_error(exc)) from exc
        return self._snapshot_from_order(payload, expected_client_id=approved.client_order_id)

    def query_order(self, *, client_order_id: str) -> ExternalOrderSnapshot | None:
        order = self._require_order(client_order_id)
        try:
            payload = self._request_signed(
                "GET",
                "/fapi/v1/order",
                {"symbol": order.intent.symbol, "origClientOrderId": client_order_id},
                write=False,
            )
        except BinanceExecutionApiError as exc:
            if exc.code in {-2013, -2011}:
                return None
            raise
        return self._snapshot_from_order(payload, expected_client_id=client_order_id)

    def list_open_orders(
        self,
        *,
        account_id: str,
        symbol: str,
    ) -> Sequence[ExternalOrderSnapshot]:
        self._assert_account(account_id)
        payload = self._request_signed(
            "GET", "/fapi/v1/openOrders", {"symbol": _symbol(symbol)}, write=False
        )
        if not isinstance(payload, list):
            raise BinanceExecutionApiError("openOrders returned invalid payload")
        result: list[ExternalOrderSnapshot] = []
        for row in payload:
            try:
                result.append(self._snapshot_from_order(row))
            except (TypeError, ValueError, BinanceExecutionApiError):
                continue
        return tuple(result)

    def list_recent_fills(
        self,
        *,
        account_id: str,
        symbol: str,
    ) -> Sequence[ExternalOrderSnapshot]:
        self._assert_account(account_id)
        normalized_symbol = _symbol(symbol)
        payload = self._request_signed(
            "GET",
            "/fapi/v1/userTrades",
            {"symbol": normalized_symbol, "limit": 1000},
            write=False,
        )
        if not isinstance(payload, list):
            raise BinanceExecutionApiError("userTrades returned invalid payload")
        order_id_map = self._exchange_order_id_map(normalized_symbol)
        result: list[ExternalOrderSnapshot] = []
        for row in payload:
            if not isinstance(row, Mapping):
                continue
            order_id = str(row.get("orderId", "")).strip()
            client_id = order_id_map.get(order_id)
            if client_id is None:
                continue
            quantity = _decimal(row.get("qty", "0"), "trade qty")
            price = _decimal(row.get("price", "0"), "trade price")
            if quantity <= 0 or price <= 0:
                continue
            commission = _decimal(row.get("commission", "0"), "commission")
            if commission < 0:
                commission = abs(commission)
            result.append(
                ExternalOrderSnapshot(
                    client_order_id=client_id,
                    status=OrderState.PARTIAL,
                    filled_quantity=quantity,
                    average_price=price,
                    exchange_order_id=order_id or None,
                    exchange_trade_id=str(row.get("id", "")).strip() or None,
                    last_fill_fee=commission,
                    fee_currency=str(row.get("commissionAsset", "USDT")),
                    reason="BINANCE_USER_TRADE",
                    observed_at=_timestamp(row.get("time")),
                )
            )
        return tuple(result)

    def cancel_order(self, *, client_order_id: str) -> ExternalOrderSnapshot:
        order = self._require_order(client_order_id)
        self._gate.assert_cancel_allowed(order)
        try:
            payload = self._request_signed(
                "DELETE",
                "/fapi/v1/order",
                {"symbol": order.intent.symbol, "origClientOrderId": client_order_id},
                write=True,
            )
        except BinanceExecutionApiError as exc:
            if exc.ambiguous:
                with self._lock:
                    self._ambiguous_write_errors += 1
                raise TimeoutError("Binance cancel outcome is unknown") from exc
            if exc.code in {-2011, -2013}:
                current = self.query_order(client_order_id=client_order_id)
                if current is not None:
                    return current
            with self._lock:
                self._definitive_write_errors += 1
            raise DefinitiveCancellationError(_safe_error(exc)) from exc
        return self._snapshot_from_order(payload, expected_client_id=client_order_id)

    def _new_order_params(self, approved: ApprovedOrderIntent) -> dict[str, Any]:
        intent = approved.intent
        side = _binance_side(intent.position_side, intent.action)
        params: dict[str, Any] = {
            "symbol": intent.symbol,
            "side": side,
            "positionSide": intent.position_side.value,
            "type": intent.order_type.value,
            "quantity": _decimal_text(approved.approved_quantity),
            "newClientOrderId": approved.client_order_id,
            "newOrderRespType": "RESULT",
        }
        if intent.order_type is OrderType.LIMIT:
            tif = str(intent.metadata.get("time_in_force", "GTC")).strip().upper()
            if tif not in {"GTC", "IOC", "FOK", "GTX"}:
                raise DefinitiveSubmissionError("unsupported Binance timeInForce")
            params["timeInForce"] = tif
            params["price"] = _decimal_text(intent.limit_price)
            stp = intent.metadata.get("self_trade_prevention_mode")
            if stp is not None:
                stp_value = str(stp).strip().upper()
                if stp_value not in {"EXPIRE_TAKER", "EXPIRE_MAKER", "EXPIRE_BOTH"}:
                    raise DefinitiveSubmissionError(
                        "unsupported Binance selfTradePreventionMode"
                    )
                params["selfTradePreventionMode"] = stp_value
        elif intent.order_type is not OrderType.MARKET:
            raise DefinitiveSubmissionError("unsupported Binance order type")
        return params

    def _request_public(self, method: str, path: str, params: Mapping[str, Any]) -> Any:
        query = urlencode(_clean_params(params))
        return self._perform(method, path, query, authenticated=False, write=False)

    def _request_signed(
        self,
        method: str,
        path: str,
        params: Mapping[str, Any],
        *,
        write: bool,
    ) -> Any:
        values = _clean_params(params)
        values["recvWindow"] = self._recv_window_ms
        values["timestamp"] = self._now_ms() + self._clock_offset_ms
        unsigned = urlencode(values)
        signature = hmac.new(
            self._credentials.api_secret.encode("utf-8"),
            unsigned.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return self._perform(
            method,
            path,
            f"{unsigned}&signature={signature}",
            authenticated=True,
            write=write,
        )

    @staticmethod
    def _endpoint_weight(path: str) -> int:
        return {
            "/fapi/v1/time": 1,
            "/fapi/v1/order": 2,
            "/fapi/v1/order/test": 2,
            "/fapi/v1/openOrders": 10,
            "/fapi/v1/userTrades": 10,
        }.get(path, 10)

    def _reserve_shared_weight(self, path: str) -> None:
        if self._shared_weight_budget is None:
            return
        weight = self._endpoint_weight(path)
        while True:
            decision = self._shared_weight_budget.reserve_weight(weight)
            if decision.granted:
                return
            self._sleep(decision.retry_after_seconds)

    def _perform(
        self,
        method: str,
        path: str,
        query: str,
        *,
        authenticated: bool,
        write: bool,
    ) -> Any:
        method = method.upper()
        allowed = {
            ("GET", "/fapi/v1/time"),
            ("POST", "/fapi/v1/order"),
            ("POST", "/fapi/v1/order/test"),
            ("DELETE", "/fapi/v1/order"),
            ("GET", "/fapi/v1/order"),
            ("GET", "/fapi/v1/openOrders"),
            ("GET", "/fapi/v1/userTrades"),
        }
        if (method, path) not in allowed:
            raise ExecutionWriteLockedError("BINANCE_ENDPOINT_NOT_ALLOWLISTED")
        url = f"{self._base_url}{path}"
        if query:
            url = f"{url}?{query}"
        headers = {
            "Accept": "application/json",
            "User-Agent": "freqtrade-hedge-r30-execution/3.0.0",
        }
        if authenticated:
            headers["X-MBX-APIKEY"] = self._credentials.api_key
        attempts = 1 if write else 3
        for attempt in range(attempts):
            self._reserve_shared_weight(path)
            with self._lock:
                self._logical_requests += 1
                if write:
                    self._write_requests += 1
                else:
                    self._read_requests += 1
            try:
                response = self._transport.request(method, url, headers, self._timeout)
            except TimeoutError as exc:
                if write:
                    raise BinanceExecutionApiError(
                        "Binance write transport failed with unknown outcome",
                        ambiguous=True,
                    ) from exc
                if attempt + 1 >= attempts:
                    raise BinanceExecutionApiError("Binance read transport failed") from exc
                self._sleep(0.2 * (2**attempt))
                continue
            self._record_headers(response)
            payload = _decode_json(response.payload)
            if response.status < 400:
                return payload
            code = payload.get("code") if isinstance(payload, Mapping) else None
            message = payload.get("msg") if isinstance(payload, Mapping) else "request failed"
            code_value = int(code) if isinstance(code, int) else None
            ambiguous = write and (response.status >= 500 or response.status in {408, 418, 429})
            if not write and (response.status >= 500 or response.status in {418, 429}):
                if attempt + 1 < attempts:
                    self._sleep(0.2 * (2**attempt))
                    continue
            raise BinanceExecutionApiError(
                f"Binance {method} {path} failed: HTTP {response.status}, code={code_value}, {message}",
                status=response.status,
                code=code_value,
                ambiguous=ambiguous,
            )
        raise BinanceExecutionApiError("unreachable Binance request state")

    def _record_headers(self, response: HttpResponse) -> None:
        with self._lock:
            self._last_status = response.status
            for name, value in response.headers.items():
                lowered = name.lower()
                try:
                    parsed = int(value)
                except (TypeError, ValueError):
                    continue
                if lowered.startswith("x-mbx-used-weight"):
                    self._used_weight = parsed
                    if self._shared_weight_budget is not None:
                        self._shared_weight_budget.observe_remote_weight(parsed)
                elif lowered.startswith("x-mbx-order-count"):
                    self._order_count = parsed

    def _snapshot_from_order(
        self,
        payload: Any,
        *,
        expected_client_id: str | None = None,
    ) -> ExternalOrderSnapshot:
        if not isinstance(payload, Mapping):
            raise BinanceExecutionApiError("Binance order response is not an object")
        client_id = str(payload.get("clientOrderId", "")).strip()
        if expected_client_id is not None and client_id != expected_client_id:
            raise BinanceExecutionApiError("Binance clientOrderId mismatch", ambiguous=True)
        if not client_id:
            raise BinanceExecutionApiError("Binance response lacks clientOrderId", ambiguous=True)
        raw_status = str(payload.get("status", "")).upper()
        state = _STATUS_MAP.get(raw_status)
        if state is None:
            state = OrderState.UNKNOWN
        filled = _decimal(payload.get("executedQty", "0"), "executedQty")
        average = _average_price(payload, filled)
        if state is OrderState.PARTIAL and filled <= 0:
            state = OrderState.UNKNOWN
        if state is OrderState.FILLED and filled <= 0:
            state = OrderState.UNKNOWN
        return ExternalOrderSnapshot(
            client_order_id=client_id,
            status=state,
            filled_quantity=filled,
            average_price=average,
            exchange_order_id=str(payload.get("orderId", "")).strip() or None,
            reason=raw_status or "UNKNOWN_BINANCE_STATUS",
            observed_at=_timestamp(payload.get("updateTime") or payload.get("time")),
        )

    def _require_order(self, client_order_id: str) -> ExecutionOrder:
        value = str(client_order_id).strip()
        if not value:
            raise ValueError("client_order_id is required")
        order = self._store.get_by_client_order_id(value)
        if order is None:
            raise KeyError(value)
        return order

    def _assert_account(self, account_id: str) -> None:
        expected = (
            f"{self._gate.evidence.account_id_prefix}:"
            f"{self._gate.evidence.account_fingerprint}"
        )
        if str(account_id).strip() != expected:
            raise PermissionError("account_id does not match production evidence")

    def _exchange_order_id_map(self, symbol: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for order in self._store.list_orders():
            if order.intent.symbol != symbol:
                continue
            exchange_id = order.lifecycle.exchange_order_id
            if exchange_id:
                result[str(exchange_id)] = order.client_order_id
        return result


def _binance_side(position_side: PositionSide, action: IntentAction) -> str:
    increasing = action in {IntentAction.OPEN, IntentAction.INCREASE}
    if position_side is PositionSide.LONG:
        return "BUY" if increasing else "SELL"
    if position_side is PositionSide.SHORT:
        return "SELL" if increasing else "BUY"
    raise ValueError("position_side must be LONG or SHORT")


def _clean_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in params.items() if value is not None}


def _symbol(value: str) -> str:
    normalized = str(value).strip().upper().replace("/", "").split(":", 1)[0]
    if not normalized or not normalized.isascii() or not normalized.isalnum():
        raise ValueError("symbol must be ASCII alphanumeric")
    return normalized


def _decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, (bool, float)):
        raise ValueError(f"{field_name} must be exact")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} is invalid") from exc
    if not result.is_finite() or result < 0:
        raise ValueError(f"{field_name} is invalid")
    return result


def _decimal_text(value: object) -> str:
    result = _decimal(value, "decimal")
    text = format(result, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _average_price(payload: Mapping[str, Any], filled: Decimal) -> Decimal | None:
    if filled <= 0:
        return None
    average = _decimal(payload.get("avgPrice", "0"), "avgPrice")
    if average > 0:
        return average
    cumulative_quote = _decimal(payload.get("cumQuote", "0"), "cumQuote")
    if cumulative_quote > 0:
        return cumulative_quote / filled
    price = _decimal(payload.get("price", "0"), "price")
    return price if price > 0 else None


def _timestamp(value: object) -> datetime:
    try:
        millis = int(value)
    except (TypeError, ValueError):
        return datetime.now(UTC)
    if millis <= 0:
        return datetime.now(UTC)
    return datetime.fromtimestamp(millis / 1000, tz=UTC)


def _decode_json(payload: bytes) -> Any:
    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BinanceExecutionApiError("Binance returned invalid JSON") from exc


def _safe_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}:{exc}"[:1000]
