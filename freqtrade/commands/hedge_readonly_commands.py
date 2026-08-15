"""Fail-closed Binance Hedge read-only account preflight command."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from freqtrade.exceptions import OperationalException


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item) for item in value]
    if is_dataclass(value):
        return _json_value(asdict(value))
    return value


def _assert_readonly_config(config: Mapping[str, Any]) -> None:
    exchange = config.get("exchange")
    if not isinstance(exchange, Mapping) or str(exchange.get("name", "")).lower() != "binance":
        raise OperationalException("hedge-readonly-check requires exchange.name='binance'")
    if config.get("hedge_mode_enabled") is not True:
        raise OperationalException("hedge-readonly-check requires hedge_mode_enabled=true")
    hedge = config.get("hedge")
    if not isinstance(hedge, Mapping):
        raise OperationalException("hedge-readonly-check requires a hedge configuration object")
    if hedge.get("read_only", True) is not True:
        raise OperationalException("hedge-readonly-check requires hedge.read_only=true")
    if hedge.get("live_trading_enabled", False) is not False:
        raise OperationalException(
            "hedge-readonly-check requires hedge.live_trading_enabled=false"
        )
    operation_mode = str(hedge.get("operation_mode", "readonly")).lower()
    if operation_mode not in {"readonly", "shadow"}:
        raise OperationalException(
            "hedge-readonly-check requires hedge.operation_mode readonly or shadow"
        )


def _managed_set(symbols: Sequence[str]) -> frozenset[str]:
    return frozenset(str(item).upper() for item in symbols)


def _unmanaged_positions(bundle: Any, managed: frozenset[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{item.symbol}:{item.position_side}:{item.quantity}"
            for item in bundle.positions
            if item.quantity > 0 and item.symbol.upper() not in managed
        )
    )


def _unmanaged_orders(bundle: Any, managed: frozenset[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{item.symbol}:{item.exchange_order_id}:{item.status}"
            for item in bundle.open_orders
            if item.active and item.symbol.upper() not in managed
        )
    )


def _leverage_mismatches(configuration: Any, target: int | None) -> tuple[str, ...]:
    if target is None:
        return ()
    return tuple(
        sorted(
            f"{key}={value}"
            for key, value in configuration.leverage_by_symbol_side.items()
            if int(value) != target
        )
    )


async def _run_preflight(
    config: Mapping[str, Any],
    *,
    include_history: bool,
) -> dict[str, Any]:
    from freqtrade.hedge.integration.repository import InMemoryReadonlyRepository
    from freqtrade.hedge.readonly import build_binance_readonly_runtime_from_freqtrade_config

    repository = InMemoryReadonlyRepository()
    runtime = build_binance_readonly_runtime_from_freqtrade_config(
        config=config,
        repository=repository,
    )
    try:
        await runtime.client.synchronize_clock()
        permission = await runtime.client.preflight_permissions(
            runtime.config.permission_policy
        )
        fill_start_time_ms = None
        if include_history:
            fill_start_time_ms = int(
                (
                    runtime.clock.now() - runtime.config.fill_lookback
                ).timestamp()
                * 1000
            )
        bundle = await runtime.client.fetch_bundle(
            include_fills=include_history,
            fill_start_time_ms=fill_start_time_ms,
        )
        managed = _managed_set(runtime.config.managed_symbols)
        unmanaged_positions = _unmanaged_positions(bundle, managed)
        unmanaged_orders = _unmanaged_orders(bundle, managed)
        leverage_mismatches = _leverage_mismatches(
            bundle.configuration,
            runtime.config.target_leverage,
        )
        failures: list[str] = []
        if not permission.strict_readonly_verified:
            failures.extend(permission.reasons or ("API_PERMISSION_POLICY_FAILED",))
        if unmanaged_positions:
            failures.append("UNMANAGED_POSITIONS")
        if unmanaged_orders:
            failures.append("UNMANAGED_ORDERS")
        if not bundle.configuration.hedge_mode:
            failures.append("POSITION_MODE_NOT_HEDGE")
        if bundle.configuration.active_margin_modes != ("cross",):
            failures.append("MARGIN_MODE_NOT_CROSS")
        if leverage_mismatches:
            failures.append("TARGET_LEVERAGE_MISMATCH")

        clock_status = runtime.client.clock_sync.status
        telemetry = runtime.transport.telemetry
        result = {
            "status": "PASS" if not failures else "HALT",
            "mode": "BINANCE_READONLY_PREFLIGHT",
            "exchange_writes": "LOCKED",
            "runtime_started": False,
            "account_id": runtime.config.account_id,
            "managed_symbols": list(runtime.config.managed_symbols),
            "proxy": {
                "rest_enabled": runtime.config.rest_proxy_url is not None,
                "websocket_enabled": runtime.config.websocket_proxy_url is not None,
                "trust_env": runtime.config.trust_env_proxy,
            },
            "clock": {
                "synchronized": clock_status.synchronized,
                "offset_ms": clock_status.offset_ms,
                "round_trip_ms": clock_status.round_trip_ms,
                "sample_count": clock_status.sample_count,
                "max_abs_skew_ms": clock_status.max_abs_skew_ms,
            },
            "permissions": {
                "read_enabled": permission.read_enabled,
                "futures_enabled": permission.futures_enabled,
                "withdrawals_enabled": permission.withdrawals_enabled,
                "internal_transfer_enabled": permission.internal_transfer_enabled,
                "universal_transfer_enabled": permission.universal_transfer_enabled,
                "spot_margin_trading_enabled": permission.spot_margin_trading_enabled,
                "strict_readonly_verified": permission.strict_readonly_verified,
                "runtime_readonly_enforced": permission.runtime_readonly_enforced,
                "warnings": list(permission.warnings),
                "reasons": list(permission.reasons),
            },
            "configuration": {
                "hedge_mode": bundle.configuration.hedge_mode,
                "active_margin_modes": list(bundle.configuration.active_margin_modes),
                "leverage_by_symbol_side": dict(
                    sorted(bundle.configuration.leverage_by_symbol_side.items())
                ),
                "target_leverage": runtime.config.target_leverage,
                "leverage_mismatches": list(leverage_mismatches),
            },
            "account": {
                "total_wallet_balance": bundle.account_snapshot.total_wallet_balance,
                "total_available_balance": bundle.account_snapshot.total_available_balance,
                "total_margin_balance": bundle.account_snapshot.total_margin_balance,
                "total_initial_margin": bundle.account_snapshot.total_initial_margin,
                "total_maintenance_margin": bundle.account_snapshot.total_maintenance_margin,
                "total_unrealized_pnl": bundle.account_snapshot.total_unrealized_pnl,
                "balance_count": len(bundle.balances),
                "position_count": len(bundle.positions),
                "active_order_count": sum(1 for item in bundle.open_orders if item.active),
                "fill_count": len(bundle.fills),
                "collection_started_at": bundle.collection_started_at,
                "collection_completed_at": bundle.collection_completed_at,
            },
            "unmanaged": {
                "positions": list(unmanaged_positions),
                "orders": list(unmanaged_orders),
            },
            "transport": {
                "logical_request_count": telemetry.logical_request_count,
                "attempt_count": telemetry.attempt_count,
                "retry_count": telemetry.retry_count,
                "error_count": telemetry.error_count,
                "rate_limit_count": telemetry.rate_limit_count,
                "server_error_count": telemetry.server_error_count,
                "last_status": telemetry.last_status,
                "last_latency_ms": telemetry.last_latency_ms,
            },
            "failure_reasons": failures,
        }
        return _json_value(result)
    finally:
        await runtime.transport.close()


def start_hedge_readonly_check(args: dict[str, Any]) -> int:
    """Run a REST-only Binance account preflight without creating a listen key."""

    from freqtrade.configuration import Configuration

    config = Configuration(args, None).get_config()
    _assert_readonly_config(config)
    include_history = bool(args.get("hedge_readonly_include_history", False))
    result = asyncio.run(_run_preflight(config, include_history=include_history))
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    output = args.get("hedge_readonly_output")
    if output:
        destination = Path(str(output)).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if result["status"] != "PASS":
        raise OperationalException(
            "Binance Hedge read-only preflight halted: "
            + ",".join(result["failure_reasons"])
        )
    return 0
