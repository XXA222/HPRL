"""Fail-closed safety boundary for native Freqtrade write paths in Hedge mode."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from freqtrade.exceptions import OperationalException


NATIVE_WRITE_BLOCKED_REASON = "HEDGE_NATIVE_EXCHANGE_WRITE_BLOCKED"


def hedge_mode_enabled(config: Any) -> bool:
    return isinstance(config, MutableMapping) and config.get("hedge_mode_enabled") is True


def enforce_hedge_native_write_lock(config: MutableMapping[str, Any]) -> None:
    """Normalize all native Freqtrade write switches to their safe values."""

    if not hedge_mode_enabled(config):
        return
    config["force_entry_enable"] = False
    config["cancel_open_orders_on_exit"] = False


def assert_native_exchange_write_allowed(config: Any, *, operation: str) -> None:
    """Reject every native Freqtrade exchange write while Hedge mode is enabled."""

    if hedge_mode_enabled(config):
        normalized = str(operation).strip() or "UNKNOWN_NATIVE_WRITE"
        raise OperationalException(f"{NATIVE_WRITE_BLOCKED_REASON}:{normalized}")


def assert_supported_operation_mode(value: object) -> str:
    """Validate the currently supported isolated runtime modes.

    ``combined`` remains deliberately unavailable until paper and real-account
    projections use separate runtime views and a real-account HALT can never be
    overwritten by simulated state.
    """

    normalized = str(value).strip().lower()
    if normalized == "combined":
        raise OperationalException(
            "hedge.operation_mode='combined' was removed. Use 'shadow' for isolated "
            "exchange and paper projections."
        )
    if normalized not in {"paper", "readonly", "shadow"}:
        raise OperationalException(
            "hedge.operation_mode must be paper, readonly, or shadow."
        )
    return normalized


_NATIVE_EXCHANGE_WRITE_METHODS = (
    "create_order",
    "create_stoploss",
    "cancel_order",
    "cancel_stoploss_order",
    "cancel_order_with_result",
    "cancel_stoploss_order_with_result",
    "_lev_prep",
    "_set_leverage",
    "set_margin_mode",
)

_CCXT_WRITE_METHODS = (
    "create_order",
    "create_orders",
    "edit_order",
    "cancel_order",
    "cancel_orders",
    "cancel_all_orders",
    "set_leverage",
    "set_margin_mode",
    "set_position_mode",
    "transfer",
    "withdraw",
)


def _install_method_barrier(
    target: object,
    *,
    method_names: tuple[str, ...],
    config: Any,
    prefix: str,
) -> list[str]:
    blocked: list[str] = []
    for method_name in method_names:
        current = getattr(target, method_name, None)
        if not callable(current):
            continue

        def reject(*args: object, _method_name: str = method_name, **kwargs: object):
            del args, kwargs
            assert_native_exchange_write_allowed(
                config,
                operation=f"{prefix}:{_method_name}",
            )

        try:
            setattr(target, method_name, reject)
        except Exception as exc:
            raise OperationalException(
                f"Unable to install Hedge write barrier for {prefix}:{method_name}"
            ) from exc
        blocked.append(f"{prefix}:{method_name}")
    return blocked


def install_native_exchange_write_barrier(exchange: object, config: Any) -> tuple[str, ...]:
    """Install fail-closed barriers on native Freqtrade and CCXT write surfaces.

    Service/RPC guards are the first line of defence.  The Exchange-instance and
    underlying synchronous/asynchronous CCXT barriers protect against future or
    indirect native paths that bypass those services.  The dedicated Hedge Fake
    Exchange is a separate port graph and is not modified.
    """

    if not hedge_mode_enabled(config):
        return ()
    if getattr(exchange, "_hedge_native_write_barrier_installed", False):
        return tuple(getattr(exchange, "_hedge_native_write_barrier_methods", ()))

    blocked = _install_method_barrier(
        exchange,
        method_names=_NATIVE_EXCHANGE_WRITE_METHODS,
        config=config,
        prefix="EXCHANGE_METHOD",
    )
    for attribute in ("_api", "_api_async"):
        client = getattr(exchange, attribute, None)
        if client is None:
            continue
        blocked.extend(
            _install_method_barrier(
                client,
                method_names=_CCXT_WRITE_METHODS,
                config=config,
                prefix=f"CCXT_{attribute.strip('_').upper()}",
            )
        )

    setattr(exchange, "_hedge_native_write_barrier_installed", True)
    setattr(exchange, "_hedge_native_write_barrier_methods", tuple(blocked))
    return tuple(blocked)
