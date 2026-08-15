from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
import logging
from decimal import Decimal
from typing import Any

from freqtrade.enums.hedge import PositionMode
from freqtrade.exceptions import OperationalException
from freqtrade.hedge.config_migration import migrate_legacy_operations_alias
from freqtrade.hedge.paper_config import PaperSimulationConfig
from freqtrade.hedge.numeric import (
    require_nonnegative_int,
    require_positive,
    require_unit_interval,
)
from freqtrade.hedge.symbols import canonicalize_symbol
from freqtrade.hedge.safety import (
    assert_supported_operation_mode,
    enforce_hedge_native_write_lock,
)


logger = logging.getLogger(__name__)

SUPPORTED_HEDGE_ADAPTERS = frozenset({"binance"})

_HEDGE_ALLOWED_KEYS = frozenset({
    "exchange_adapter",
    "read_only",
    "live_trading_enabled",
    "account_id",
    "operation_mode",
    "target_leverage",
    "max_clock_skew_ms",
    "user_stream_max_age_ms",
    "rest_reconcile_interval_seconds",
    "fast_calibration_interval_seconds",
    "full_reconcile_interval_seconds",
    "full_calibration_interval_seconds",
    "calibration_error_retry_interval_seconds",
    "calibration_stale_after_seconds",
    "fill_lookback_hours",
    "history_overlap_seconds",
    "listen_key_ttl_seconds",
    "listen_key_renew_interval_seconds",
    "listen_key_error_retry_interval_seconds",
    "futures_base_url",
    "spot_base_url",
    "websocket_base_url",
    "rest_proxy_url",
    "websocket_proxy_url",
    "trust_env_proxy",
    "request_timeout_seconds",
    "recv_window_ms",
    "max_collection_span_seconds",
    "min_reconnect_delay_seconds",
    "max_reconnect_delay_seconds",
    "reconnect_reset_after_seconds",
    "drift_verification_attempts",
    "max_history_backfill_days",
    "quantity_tolerance",
    "financial_tolerance",
    "system_client_order_prefixes",
    "max_gross_notional",
    "max_gross_exposure_ratio",
    "max_margin_utilization",
    "min_liquidation_buffer_ratio",
    "max_single_order_notional",
    "main_loop",
    "control_plane",
    "paper",
    "planner",
    "optimization",
    "simulation",
    "dashboard",
    "native_convergence",
    "universe",
    "freqai_native",
    "producer_consumer",
    "hyperopt_native",
    "rl_native",
    "operations",
})

_CONTROL_PLANE_ALLOWED_KEYS = frozenset({
    "enabled", "confirmation_ttl_seconds", "max_pending_confirmations",
})

_MAIN_LOOP_ALLOWED_KEYS = frozenset({
    "enabled", "mode", "allowed_symbols", "state_backend", "state_path",
    "max_submissions_per_cycle", "max_cancellations_per_cycle",
    "block_new_risk_on_external_side", "require_stream_fresh",
    "require_rest_fresh", "require_reconciliation_consistent",
    "recover_on_start",
})


_PAPER_ALLOWED_KEYS = frozenset({
    "initial_balance", "leverage", "auto_fill", "fill_model",
    "long_signal", "short_signal", "tick_size", "qty_step",
    "min_qty", "min_notional", "ephemeral", "state_backend", "state_path",
    "fee_rate", "maker_fee_rate", "taker_fee_rate",
    "volume_participation", "market_slippage_bps",
    "max_entry_layers_per_bar", "max_reduce_layers_per_bar",
    "max_fill_ratio_per_order", "max_fills_per_bar", "bar_volume",
    "ohlcv_source", "require_closed_candle", "candle_max_age_seconds",
    "max_catchup_candles", "max_missing_candles", "reject_revised_candle",
    "funding_source", "funding_max_age_seconds", "funding_poll_interval_seconds",
    "account_events_enabled",
    "idempotency_lease_seconds",
})

_PLANNER_ALLOWED_KEYS = frozenset({
    "long_enabled", "short_enabled",
    "core_wallet_exposure_long", "core_wallet_exposure_short",
    "tactical_wallet_exposure_long", "tactical_wallet_exposure_short",
    "max_wallet_exposure_long", "max_wallet_exposure_short",
    "max_gross_wallet_exposure", "initial_entry_fraction",
    "max_grid_layers", "grid_spacing", "grid_spacing_growth",
    "grid_qty_growth", "qty_scale", "trailing_rebound",
    "take_profit_spacing", "take_profit_layers",
    "tactical_reduce_fraction", "core_min_fraction",
    "cooldown_seconds", "replace_price_tolerance_ticks",
    "replace_qty_tolerance_steps", "replace_min_age_seconds",
    "unstuck_trigger_gross_exposure", "unstuck_reduce_fraction",
    "unstuck_limit_only",
    "maintenance_margin_rate", "target_net_wallet_exposure",
    "net_repair_threshold", "trailing_trigger_distance",
    "grid_initial_distance", "trailing_timeout_seconds",
    "max_pending_entries", "max_single_order_notional",
    "unstuck_max_holding_seconds", "unstuck_daily_loss_budget",
    "unstuck_weekly_loss_budget", "unstuck_min_cooldown_seconds",
    "unstuck_min_risk_improvement", "liquidation_fee_rate",
    "liquidation_buffer_warning_ratio",
})

_SIMULATION_ALLOWED_KEYS = frozenset({
    "maker_fee", "taker_fee", "partial_fill_ratio",
    "max_fills_per_bar", "volume_participation",
    "market_slippage_bps", "bar_volume",
})

_OPERATIONS_ALLOWED_KEYS = frozenset({
    "enabled", "state_path",
    "market_max_age_seconds", "market_max_gap_candles",
    "mark_index_max_divergence_bps",
    "warmup_candles", "informative_warmup",
    "max_gross_ratio", "max_margin_ratio", "max_net_ratio",
    "drawdown_warning", "drawdown_pause", "drawdown_kill",
    "drawdown_recovery",
})


_DASHBOARD_ALLOWED_KEYS = frozenset({
    "enabled",
    "local_only",
    "refresh_seconds",
    "telemetry_capacity",
    "telemetry_backend",
    "telemetry_path",
    "control_state_path",
})


_NATIVE_CONVERGENCE_ALLOWED_KEYS = frozenset({
    "enabled", "callback_mode", "fail_closed_confirmations",
    "exits", "rpc_projection", "notifications", "capital_policy",
})

_UNIVERSE_ALLOWED_KEYS = frozenset({
    "enabled", "pairs", "weights", "blacklist_policy", "drain_removed_pairs",
    "max_pairs", "minimum_pair_weight",
})

_FREQAI_NATIVE_ALLOWED_KEYS = frozenset({
    "enabled", "required", "maximum_signal_age_seconds",
    "expected_feature_schema", "model_manifest_path",
    "fail_closed_on_expiry", "compatible_pairs",
})

_PRODUCER_CONSUMER_ALLOWED_KEYS = frozenset({
    "enabled", "required", "maximum_age_seconds", "conflict_tolerance",
    "fail_on_conflict", "producers",
})

_HYPEROPT_NATIVE_ALLOWED_KEYS = frozenset({
    "enabled", "epochs", "jobs", "seed", "loss_weights", "result_path",
})

_RL_NATIVE_ALLOWED_KEYS = frozenset({
    "enabled", "max_step_ratio", "min_liquidation_buffer",
    "max_gross_exposure", "require_confidence", "reward_weights",
    "vector_envs", "device",
})


def _reject_unknown_keys(
    value: MutableMapping[str, Any],
    *,
    field: str,
    allowed: frozenset[str],
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise OperationalException(
            f"Unknown {field} configuration key(s): {', '.join(unknown)}"
        )


def _normalize_legacy_managed_scope(
    config: MutableMapping[str, Any],
    raw_hedge: MutableMapping[str, Any],
    managed_pair: str | None,
) -> None:
    nested_pair = raw_hedge.pop("managed_pair", None)
    nested_symbols = raw_hedge.pop("managed_symbols", None)
    legacy_values: list[str] = []
    if nested_pair is not None:
        legacy_values.append(str(nested_pair))
    if nested_symbols is not None:
        if not isinstance(nested_symbols, list) or len(nested_symbols) != 1:
            raise OperationalException(
                "hedge.managed_symbols is deprecated and must contain exactly one symbol; "
                "use top-level managed_pair instead."
            )
        legacy_values.append(str(nested_symbols[0]))
    for value in legacy_values:
        normalized = canonicalize_symbol(value, managed_pair=managed_pair)
        if managed_pair is None:
            managed_pair = normalized
            config["managed_pair"] = normalized
        elif normalized != managed_pair:
            raise OperationalException(
                "Deprecated nested managed scope conflicts with top-level managed_pair."
            )
    if legacy_values:
        logger.warning(
            "hedge.managed_pair/managed_symbols are deprecated; use top-level managed_pair only."
        )


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def _coerce_position_mode(value: Any) -> PositionMode:
    if isinstance(value, PositionMode):
        return value
    try:
        return PositionMode(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(mode.value for mode in PositionMode)
        raise OperationalException(
            f"Invalid position_mode {value!r}. Allowed values: {allowed}."
        ) from exc


@dataclass(frozen=True, slots=True)
class HedgeRuntimeConfig:
    position_mode: PositionMode
    enabled: bool
    managed_pair: str | None
    account_id: str = "default"
    exchange_adapter: str | None = None
    read_only: bool = True
    live_trading_enabled: bool = False
    target_leverage: Decimal | None = None
    max_clock_skew_ms: int = 5000
    user_stream_max_age_ms: int = 60000
    rest_reconcile_interval_seconds: int = 60
    max_gross_notional: Decimal | None = None
    max_gross_exposure_ratio: Decimal | None = None
    max_margin_utilization: Decimal = Decimal("0.55")
    min_liquidation_buffer_ratio: Decimal = Decimal("0.20")
    operation_mode: str = "readonly"
    paper: PaperSimulationConfig | None = None


def normalize_hedge_config(
    config: MutableMapping[str, Any],
) -> HedgeRuntimeConfig:
    position_mode = _coerce_position_mode(config.get("position_mode", PositionMode.ONEWAY))
    config["position_mode"] = position_mode

    enabled = config.get("hedge_mode_enabled", False)
    if not isinstance(enabled, bool):
        raise OperationalException("hedge_mode_enabled must be a boolean.")
    config["hedge_mode_enabled"] = enabled

    managed_pair = config.get("managed_pair")
    if managed_pair is not None:
        if not isinstance(managed_pair, str) or not managed_pair.strip():
            raise OperationalException("managed_pair must be a non-empty string or null.")
        managed_pair = canonicalize_symbol(managed_pair)
    config["managed_pair"] = managed_pair

    raw_hedge = config.setdefault("hedge", {})
    if not isinstance(raw_hedge, MutableMapping):
        raise OperationalException("hedge must be a JSON object.")
    if migrate_legacy_operations_alias(raw_hedge):
        logger.warning(
            "Migrated retired Hedge operations configuration to hedge.operations. "
            "Update the persisted config to the canonical key."
        )

    _normalize_legacy_managed_scope(config, raw_hedge, managed_pair)
    managed_pair = config.get("managed_pair")
    _reject_unknown_keys(raw_hedge, field="hedge", allowed=_HEDGE_ALLOWED_KEYS)
    for section_name, allowed in (
        ("main_loop", _MAIN_LOOP_ALLOWED_KEYS),
        ("control_plane", _CONTROL_PLANE_ALLOWED_KEYS),
        ("paper", _PAPER_ALLOWED_KEYS),
        ("planner", _PLANNER_ALLOWED_KEYS),
        ("simulation", _SIMULATION_ALLOWED_KEYS),
        ("dashboard", _DASHBOARD_ALLOWED_KEYS),
        ("native_convergence", _NATIVE_CONVERGENCE_ALLOWED_KEYS),
        ("universe", _UNIVERSE_ALLOWED_KEYS),
        ("freqai_native", _FREQAI_NATIVE_ALLOWED_KEYS),
        ("producer_consumer", _PRODUCER_CONSUMER_ALLOWED_KEYS),
        ("hyperopt_native", _HYPEROPT_NATIVE_ALLOWED_KEYS),
        ("rl_native", _RL_NATIVE_ALLOWED_KEYS),
        ("operations", _OPERATIONS_ALLOWED_KEYS),
    ):
        section = raw_hedge.get(section_name, {})
        if not isinstance(section, MutableMapping):
            raise OperationalException(f"hedge.{section_name} must be a JSON object.")
        _reject_unknown_keys(section, field=f"hedge.{section_name}", allowed=allowed)

    paper_config = PaperSimulationConfig.from_hedge_mapping(raw_hedge)

    exchange = config.get("exchange")
    exchange_name = exchange.get("name") if isinstance(exchange, MutableMapping) else None
    explicit_adapter = "exchange_adapter" in raw_hedge
    adapter = raw_hedge.get("exchange_adapter", exchange_name)
    if not explicit_adapter and adapter not in SUPPORTED_HEDGE_ADAPTERS:
        adapter = None
    if adapter is not None:
        if not isinstance(adapter, str) or not adapter.strip():
            raise OperationalException("hedge.exchange_adapter must be a non-empty string or null.")
        adapter = adapter.strip().lower()
    raw_hedge["exchange_adapter"] = adapter

    account_id = raw_hedge.get("account_id", "default")
    if not isinstance(account_id, str) or not account_id.strip():
        raise OperationalException("hedge.account_id must be a non-empty string.")
    account_id = account_id.strip()
    raw_hedge["account_id"] = account_id

    read_only = raw_hedge.get("read_only", True)
    live_enabled = raw_hedge.get("live_trading_enabled", False)
    if not isinstance(read_only, bool) or not isinstance(live_enabled, bool):
        raise OperationalException(
            "hedge.read_only and hedge.live_trading_enabled must be booleans."
        )
    raw_hedge["read_only"] = read_only
    raw_hedge["live_trading_enabled"] = live_enabled

    default_operation_mode = "paper" if bool(config.get("dry_run", False)) else "readonly"
    operation_mode = assert_supported_operation_mode(
        raw_hedge.get("operation_mode", default_operation_mode)
    )
    raw_hedge["operation_mode"] = operation_mode

    target_leverage_raw = raw_hedge.get("target_leverage")
    target_leverage = (
        None
        if target_leverage_raw is None
        else require_positive(target_leverage_raw, field="hedge.target_leverage")
    )
    if target_leverage is None:
        raw_hedge.pop("target_leverage", None)
    else:
        raw_hedge["target_leverage"] = str(target_leverage)

    max_clock_skew_ms = require_nonnegative_int(
        raw_hedge.get("max_clock_skew_ms", 5000),
        field="hedge.max_clock_skew_ms",
    )
    user_stream_max_age_ms = require_nonnegative_int(
        raw_hedge.get("user_stream_max_age_ms", 60000),
        field="hedge.user_stream_max_age_ms",
    )
    rest_interval = require_nonnegative_int(
        raw_hedge.get("rest_reconcile_interval_seconds", 60),
        field="hedge.rest_reconcile_interval_seconds",
    )
    if user_stream_max_age_ms == 0 or rest_interval == 0:
        raise OperationalException("stream age and reconcile interval must be positive.")

    max_gross_notional_raw = raw_hedge.get("max_gross_notional")
    max_gross_ratio_raw = raw_hedge.get("max_gross_exposure_ratio")
    max_gross_notional = (
        None
        if max_gross_notional_raw is None
        else require_positive(
            max_gross_notional_raw,
            field="hedge.max_gross_notional",
        )
    )
    max_gross_ratio = (
        None
        if max_gross_ratio_raw is None
        else require_positive(
            max_gross_ratio_raw,
            field="hedge.max_gross_exposure_ratio",
        )
    )
    max_margin = require_unit_interval(
        raw_hedge.get("max_margin_utilization", "0.55"),
        field="hedge.max_margin_utilization",
    )
    min_liq = require_unit_interval(
        raw_hedge.get("min_liquidation_buffer_ratio", "0.20"),
        field="hedge.min_liquidation_buffer_ratio",
    )

    for key, value in {
        "max_clock_skew_ms": max_clock_skew_ms,
        "user_stream_max_age_ms": user_stream_max_age_ms,
        "rest_reconcile_interval_seconds": rest_interval,
        "max_margin_utilization": str(max_margin),
        "min_liquidation_buffer_ratio": str(min_liq),
    }.items():
        raw_hedge[key] = value

    for key, value in {
        "max_gross_notional": max_gross_notional,
        "max_gross_exposure_ratio": max_gross_ratio,
    }.items():
        if value is None:
            raw_hedge.pop(key, None)
        else:
            raw_hedge[key] = str(value)

    return HedgeRuntimeConfig(
        position_mode=position_mode,
        enabled=enabled,
        managed_pair=managed_pair,
        account_id=account_id,
        exchange_adapter=adapter,
        read_only=read_only,
        live_trading_enabled=live_enabled,
        target_leverage=target_leverage,
        max_clock_skew_ms=max_clock_skew_ms,
        user_stream_max_age_ms=user_stream_max_age_ms,
        rest_reconcile_interval_seconds=rest_interval,
        max_gross_notional=max_gross_notional,
        max_gross_exposure_ratio=max_gross_ratio,
        max_margin_utilization=max_margin,
        min_liquidation_buffer_ratio=min_liq,
        operation_mode=operation_mode,
        paper=paper_config,
    )


def validate_hedge_config(
    config: MutableMapping[str, Any],
) -> HedgeRuntimeConfig:
    hedge = normalize_hedge_config(config)
    trading_mode = str(_enum_value(config.get("trading_mode", "spot"))).lower()
    margin_mode = str(_enum_value(config.get("margin_mode", ""))).lower()

    if hedge.position_mode is PositionMode.HEDGE and not hedge.enabled:
        raise OperationalException(
            "position_mode='hedge' requires trading_mode='futures' and hedge_mode_enabled=true. "
            "The legacy engine must not receive a hedge-mode account."
        )

    if hedge.enabled:
        enforce_hedge_native_write_lock(config)
        if hedge.position_mode is not PositionMode.HEDGE:
            raise OperationalException("hedge_mode_enabled=true requires position_mode='hedge'.")
        if trading_mode != "futures":
            raise OperationalException("Hedge mode requires trading_mode='futures'.")
        if margin_mode != "cross":
            raise OperationalException("The hedge MVP requires margin_mode='cross'.")
        if hedge.exchange_adapter not in SUPPORTED_HEDGE_ADAPTERS:
            supported = ", ".join(sorted(SUPPORTED_HEDGE_ADAPTERS))
            raise OperationalException(f"hedge.exchange_adapter must be one of: {supported}.")
        if hedge.managed_pair is None:
            raise OperationalException("managed_pair is required when hedge_mode_enabled=true.")
        exchange = config.get("exchange")
        whitelist = (
            exchange.get("pair_whitelist", []) if isinstance(exchange, MutableMapping) else []
        )
        normalized_whitelist = [
            canonicalize_symbol(str(item), managed_pair=hedge.managed_pair) for item in whitelist
        ]
        if normalized_whitelist != [hedge.managed_pair]:
            raise OperationalException(
                "Hedge MVP requires exactly one whitelisted pair equal to managed_pair."
            )
        if hedge.operation_mode in {"readonly", "shadow"}:
            if not isinstance(exchange, MutableMapping) or not str(exchange.get("key", "")).strip() or not str(exchange.get("secret", "")).strip():
                raise OperationalException(
                    f"hedge.operation_mode={hedge.operation_mode!r} requires exchange key and secret."
                )
        if not hedge.read_only or hedge.live_trading_enabled:
            raise OperationalException(
                "P2-H2 execution is locked; execution is intentionally locked. "
                "Keep hedge.read_only=true and "
                "hedge.live_trading_enabled=false until P2-H3 passes."
            )
    return hedge
