from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from decimal import Decimal
from typing import Any

from freqtrade.hedge.exchange.base import Clock, ReadonlyFactRepository
from freqtrade.hedge.exchange.binance_user_stream import WebSocketConnector
from freqtrade.hedge.readonly.runtime import (
    BinanceReadonlyRuntime,
    BinanceReadonlyRuntimeConfig,
    build_binance_readonly_runtime,
)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _sequence(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, str):
        normalized = value.strip()
        return (normalized,) if normalized else ()
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        raise ValueError(f"{field} must be a string or sequence")
    result = tuple(str(item).strip() for item in value if str(item).strip())
    if not result:
        raise ValueError(f"{field} must not be empty")
    return result


def _seconds(value: Any, *, field: str) -> timedelta:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    return timedelta(seconds=seconds)


def _milliseconds(value: Any, *, field: str) -> timedelta:
    try:
        milliseconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    return timedelta(milliseconds=milliseconds)


def _hours(value: Any, *, field: str) -> timedelta:
    try:
        hours = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    return timedelta(hours=hours)


def _positive_integer(value: Any, *, field: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def _days_or_none(value: Any, *, field: str) -> timedelta | None:
    if value is None:
        return None
    try:
        days = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric or null") from exc
    return timedelta(days=days)


def _managed_symbols_from_config(
    config: Mapping[str, Any],
    hedge: Mapping[str, Any],
    exchange: Mapping[str, Any],
) -> tuple[str, ...]:
    candidates = (
        hedge.get("managed_symbols"),
        hedge.get("managed_pair"),
        config.get("managed_pair"),
        exchange.get("pair_whitelist"),
    )
    symbol_value = next((value for value in candidates if value is not None), None)
    return _sequence(symbol_value, field="managed symbols")


def _credentials(
    exchange: Mapping[str, Any],
    *,
    api_key: str | None,
    api_secret: str | None,
) -> tuple[str, str]:
    resolved_key = api_key if api_key is not None else exchange.get("key", "")
    resolved_secret = (
        api_secret if api_secret is not None else exchange.get("secret", "")
    )
    return str(resolved_key), str(resolved_secret)


def _proxy_from_exchange(exchange: Mapping[str, Any]) -> str | None:
    for section_name in ("ccxt_async_config", "ccxt_config"):
        section = _mapping(exchange.get(section_name), field=f"exchange.{section_name}")
        for key in ("httpsProxy", "httpProxy", "proxy"):
            value = section.get(key)
            if value is not None and str(value).strip():
                return str(value).strip()
    return None


def _copy_proxy_fields(
    hedge: Mapping[str, Any],
    exchange: Mapping[str, Any],
    kwargs: dict[str, Any],
) -> None:
    exchange_proxy = _proxy_from_exchange(exchange)
    rest_proxy = hedge.get("rest_proxy_url", exchange_proxy)
    websocket_proxy = hedge.get("websocket_proxy_url", rest_proxy)
    if rest_proxy is not None:
        kwargs["rest_proxy_url"] = str(rest_proxy)
    if websocket_proxy is not None:
        kwargs["websocket_proxy_url"] = str(websocket_proxy)
    if "trust_env_proxy" in hedge:
        trust_env_proxy = hedge["trust_env_proxy"]
        if not isinstance(trust_env_proxy, bool):
            raise ValueError("hedge.trust_env_proxy must be a boolean")
        kwargs["trust_env_proxy"] = trust_env_proxy


def _copy_direct_fields(
    hedge: Mapping[str, Any],
    kwargs: dict[str, Any],
) -> None:
    direct_fields = (
        "futures_base_url",
        "spot_base_url",
        "websocket_base_url",
        "request_timeout_seconds",
        "recv_window_ms",
        "max_collection_span_seconds",
        "min_reconnect_delay_seconds",
        "max_reconnect_delay_seconds",
        "reconnect_reset_after_seconds",
        "drift_verification_attempts",
        "max_clock_skew_ms",
    )
    kwargs.update(
        {field_name: hedge[field_name] for field_name in direct_fields if field_name in hedge}
    )


def _copy_duration_fields(
    hedge: Mapping[str, Any],
    kwargs: dict[str, Any],
) -> None:
    duration_fields = {
        "rest_reconcile_interval_seconds": ("fast_calibration_interval", _seconds),
        "fast_calibration_interval_seconds": ("fast_calibration_interval", _seconds),
        "full_reconcile_interval_seconds": ("full_calibration_interval", _seconds),
        "full_calibration_interval_seconds": ("full_calibration_interval", _seconds),
        "calibration_error_retry_interval_seconds": (
            "calibration_error_retry_interval",
            _seconds,
        ),
        "user_stream_max_age_ms": ("event_stale_after", _milliseconds),
        "calibration_stale_after_seconds": ("calibration_stale_after", _seconds),
        "fill_lookback_hours": ("fill_lookback", _hours),
        "history_overlap_seconds": ("history_overlap", _seconds),
        "listen_key_ttl_seconds": ("listen_key_ttl", _seconds),
        "listen_key_renew_interval_seconds": ("listen_key_renew_interval", _seconds),
        "listen_key_error_retry_interval_seconds": (
            "listen_key_error_retry_interval",
            _seconds,
        ),
    }
    for source_name, (target_name, converter) in duration_fields.items():
        if source_name in hedge:
            kwargs[target_name] = converter(hedge[source_name], field=source_name)


def _copy_optional_fields(
    hedge: Mapping[str, Any],
    kwargs: dict[str, Any],
) -> None:
    if "max_history_backfill_days" in hedge:
        kwargs["max_history_backfill"] = _days_or_none(
            hedge["max_history_backfill_days"],
            field="max_history_backfill_days",
        )
    for field_name in ("quantity_tolerance", "financial_tolerance"):
        if field_name in hedge:
            kwargs[field_name] = Decimal(str(hedge[field_name]))
    if "system_client_order_prefixes" in hedge:
        kwargs["system_client_order_prefixes"] = _sequence(
            hedge["system_client_order_prefixes"],
            field="system_client_order_prefixes",
        )
    if "target_leverage" in hedge:
        kwargs["target_leverage"] = _positive_integer(
            hedge["target_leverage"],
            field="target_leverage",
        )


def runtime_config_from_freqtrade(
    config: Mapping[str, Any],
    *,
    api_key: str | None = None,
    api_secret: str | None = None,
) -> BinanceReadonlyRuntimeConfig:
    """Translate standard project configuration into the direction-two runtime."""

    if not isinstance(config, Mapping):
        raise ValueError("config must be an object")
    hedge = _mapping(config.get("hedge"), field="hedge")
    exchange = _mapping(config.get("exchange"), field="exchange")
    resolved_key, resolved_secret = _credentials(
        exchange,
        api_key=api_key,
        api_secret=api_secret,
    )
    kwargs: dict[str, Any] = {
        "account_id": str(hedge.get("account_id") or "hedge-main"),
        "managed_symbols": _managed_symbols_from_config(config, hedge, exchange),
        "api_key": resolved_key,
        "api_secret": resolved_secret,
    }
    _copy_direct_fields(hedge, kwargs)
    _copy_duration_fields(hedge, kwargs)
    _copy_optional_fields(hedge, kwargs)
    _copy_proxy_fields(hedge, exchange, kwargs)
    return BinanceReadonlyRuntimeConfig(**kwargs)


def build_binance_readonly_runtime_from_freqtrade_config(
    *,
    config: Mapping[str, Any],
    repository: ReadonlyFactRepository,
    api_key: str | None = None,
    api_secret: str | None = None,
    clock: Clock | None = None,
    websocket_connector: WebSocketConnector | None = None,
    http_session: Any = None,
) -> BinanceReadonlyRuntime:
    runtime_config = runtime_config_from_freqtrade(
        config,
        api_key=api_key,
        api_secret=api_secret,
    )
    return build_binance_readonly_runtime(
        config=runtime_config,
        repository=repository,
        clock=clock,
        websocket_connector=websocket_connector,
        http_session=http_session,
    )
