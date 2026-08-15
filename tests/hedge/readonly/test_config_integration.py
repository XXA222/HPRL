from datetime import timedelta
from decimal import Decimal

import pytest

from freqtrade.hedge.readonly import runtime_config_from_freqtrade


def test_runtime_config_from_freqtrade_maps_documented_hedge_settings():
    config = {
        "exchange": {
            "key": "key",
            "secret": "secret",
            "pair_whitelist": ["ETH/USDT:USDT"],
        },
        "hedge": {
            "account_id": "hedge-main",
            "rest_reconcile_interval_seconds": 30,
            "full_reconcile_interval_seconds": 300,
            "user_stream_max_age_ms": 45_000,
            "fill_lookback_hours": 24,
            "history_overlap_seconds": 60,
            "max_history_backfill_days": 14,
            "quantity_tolerance": "0.0001",
            "financial_tolerance": "0.01",
            "system_client_order_prefixes": ["fthedge-", "manual-"],
        },
    }

    result = runtime_config_from_freqtrade(config)

    assert result.account_id == "hedge-main"
    assert result.managed_symbols == ("ETHUSDT",)
    assert result.fast_calibration_interval == timedelta(seconds=30)
    assert result.full_calibration_interval == timedelta(seconds=300)
    assert result.event_stale_after == timedelta(seconds=45)
    assert result.fill_lookback == timedelta(hours=24)
    assert result.history_overlap == timedelta(seconds=60)
    assert result.max_history_backfill == timedelta(days=14)
    assert result.quantity_tolerance == Decimal("0.0001")
    assert result.financial_tolerance == Decimal("0.01")
    assert result.system_client_order_prefixes == ("fthedge-", "manual-")


def test_explicit_credentials_override_config_and_managed_pair_is_supported():
    result = runtime_config_from_freqtrade(
        {
            "exchange": {"key": "old", "secret": "old"},
            "hedge": {"managed_pair": "BTC/USDT:USDT"},
        },
        api_key="new-key",
        api_secret="new-secret",
    )

    assert result.api_key == "new-key"
    assert result.api_secret == "new-secret"
    assert result.managed_symbols == ("BTCUSDT",)


def test_runtime_config_requires_a_managed_symbol_source():
    with pytest.raises(ValueError, match="managed symbols"):
        runtime_config_from_freqtrade(
            {"exchange": {"key": "key", "secret": "secret"}}
        )


def test_runtime_config_maps_clock_skew_and_target_leverage():
    result = runtime_config_from_freqtrade(
        {
            "exchange": {
                "key": "key",
                "secret": "secret",
                "pair_whitelist": ["ETH/USDT:USDT"],
            },
            "hedge": {
                "max_clock_skew_ms": 5000,
                "target_leverage": "3",
            },
        }
    )

    assert result.max_clock_skew_ms == 5000
    assert result.target_leverage == 3


def test_runtime_config_rejects_invalid_target_leverage():
    with pytest.raises(ValueError, match="target_leverage"):
        runtime_config_from_freqtrade(
            {
                "exchange": {
                    "key": "key",
                    "secret": "secret",
                    "pair_whitelist": ["ETH/USDT:USDT"],
                },
                "hedge": {"target_leverage": 0},
            }
        )


def test_runtime_config_inherits_ccxt_proxy_for_rest_and_websocket():
    result = runtime_config_from_freqtrade(
        {
            "exchange": {
                "key": "key",
                "secret": "secret",
                "pair_whitelist": ["BTC/USDT:USDT"],
                "ccxt_config": {
                    "httpsProxy": "http://127.0.0.1:7897",
                },
                "ccxt_async_config": {
                    "httpsProxy": "http://127.0.0.1:7897",
                },
            },
            "hedge": {},
        }
    )

    assert result.rest_proxy_url == "http://127.0.0.1:7897"
    assert result.websocket_proxy_url == "http://127.0.0.1:7897"
    assert result.trust_env_proxy is False


def test_explicit_hedge_proxy_overrides_ccxt_proxy_and_validates_trust_env():
    result = runtime_config_from_freqtrade(
        {
            "exchange": {
                "key": "key",
                "secret": "secret",
                "pair_whitelist": ["BTC/USDT:USDT"],
                "ccxt_config": {"httpsProxy": "http://127.0.0.1:7897"},
            },
            "hedge": {
                "rest_proxy_url": "http://127.0.0.1:8899",
                "websocket_proxy_url": "http://127.0.0.1:9900",
                "trust_env_proxy": True,
            },
        }
    )

    assert result.rest_proxy_url == "http://127.0.0.1:8899"
    assert result.websocket_proxy_url == "http://127.0.0.1:9900"
    assert result.trust_env_proxy is True

    with pytest.raises(ValueError, match="trust_env_proxy"):
        runtime_config_from_freqtrade(
            {
                "exchange": {
                    "key": "key",
                    "secret": "secret",
                    "pair_whitelist": ["BTC/USDT:USDT"],
                },
                "hedge": {"trust_env_proxy": "yes"},
            }
        )
