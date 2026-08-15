"""Environment profiles for Binance USD-M execution.

The module provides one authoritative binding between an execution environment,
its REST/WebSocket endpoints and the account-id namespace.  Production and testnet
credentials must never share the same namespace or endpoint profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import urlsplit


class ExecutionEnvironment(StrEnum):
    DISABLED = "DISABLED"
    TESTNET = "TESTNET"
    LIVE = "LIVE"


@dataclass(frozen=True, slots=True)
class BinanceUSDMEnvironmentProfile:
    environment: ExecutionEnvironment
    rest_base_url: str
    websocket_base_url: str
    account_prefix: str
    rest_host: str
    websocket_host: str


LIVE_PROFILE = BinanceUSDMEnvironmentProfile(
    environment=ExecutionEnvironment.LIVE,
    rest_base_url="https://fapi.binance.com",
    websocket_base_url="wss://fstream.binance.com/ws",
    account_prefix="binance-usdm",
    rest_host="fapi.binance.com",
    websocket_host="fstream.binance.com",
)

TESTNET_PROFILE = BinanceUSDMEnvironmentProfile(
    environment=ExecutionEnvironment.TESTNET,
    rest_base_url="https://demo-fapi.binance.com",
    websocket_base_url="wss://demo-fstream.binance.com/ws",
    account_prefix="binance-usdm-testnet",
    rest_host="demo-fapi.binance.com",
    websocket_host="demo-fstream.binance.com",
)

DISABLED_PROFILE = BinanceUSDMEnvironmentProfile(
    environment=ExecutionEnvironment.DISABLED,
    rest_base_url=LIVE_PROFILE.rest_base_url,
    websocket_base_url=LIVE_PROFILE.websocket_base_url,
    account_prefix="binance-usdm-disabled",
    rest_host=LIVE_PROFILE.rest_host,
    websocket_host=LIVE_PROFILE.websocket_host,
)


def profile_for_environment(
    environment: ExecutionEnvironment | str,
) -> BinanceUSDMEnvironmentProfile:
    value = environment if isinstance(environment, ExecutionEnvironment) else ExecutionEnvironment(environment)
    if value is ExecutionEnvironment.TESTNET:
        return TESTNET_PROFILE
    if value is ExecutionEnvironment.LIVE:
        return LIVE_PROFILE
    return DISABLED_PROFILE


def execution_account_id(
    environment: ExecutionEnvironment | str,
    account_fingerprint: str,
) -> str:
    fingerprint = str(account_fingerprint).strip()
    if not fingerprint or len(fingerprint) > 128:
        raise ValueError("account_fingerprint is required")
    return f"{profile_for_environment(environment).account_prefix}:{fingerprint}"


def validate_rest_base_url(
    environment: ExecutionEnvironment | str,
    value: str,
    *,
    allow_test_override: bool = False,
) -> str:
    profile = profile_for_environment(environment)
    normalized = _validate_url(value, schemes={"https"}, field_name="base_url")
    parsed = urlsplit(normalized)
    host = str(parsed.hostname or "").lower()
    if parsed.path not in {"", "/"}:
        raise ValueError("Binance USD-M REST base URL path must be empty")
    if host != profile.rest_host and not _allowed_test_host(host, allow_test_override):
        raise ValueError(
            f"{profile.environment.value} execution requires REST host {profile.rest_host}"
        )
    return normalized.rstrip("/")


def validate_websocket_base_url(
    environment: ExecutionEnvironment | str,
    value: str,
    *,
    allow_test_override: bool = False,
) -> str:
    profile = profile_for_environment(environment)
    normalized = _validate_url(value, schemes={"wss"}, field_name="websocket_base_url")
    parsed = urlsplit(normalized)
    host = str(parsed.hostname or "").lower()
    if host != profile.websocket_host and not _allowed_test_host(host, allow_test_override):
        raise ValueError(
            f"{profile.environment.value} execution requires WebSocket host "
            f"{profile.websocket_host}"
        )
    if parsed.path.rstrip("/") != "/ws" and not _allowed_test_host(host, allow_test_override):
        raise ValueError("Binance USD-M user stream base path must be /ws")
    return normalized.rstrip("/")


def _validate_url(value: str, *, schemes: set[str], field_name: str) -> str:
    normalized = str(value).strip()
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in schemes
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        expected = "/".join(sorted(schemes))
        raise ValueError(f"{field_name} must be an absolute {expected} URL without credentials")
    return normalized


def _allowed_test_host(host: str, enabled: bool) -> bool:
    return enabled and (
        host.endswith((".test", ".example", ".invalid"))
        or host in {"localhost", "127.0.0.1", "::1"}
    )
