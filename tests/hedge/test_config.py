
from copy import deepcopy
from unittest import TestCase

from freqtrade.config_schema.config_schema import CONF_SCHEMA
from freqtrade.enums.hedge import PositionMode
from freqtrade.exceptions import OperationalException
from freqtrade.hedge.config import (
    normalize_hedge_config,
    validate_hedge_config,
)


def base_config() -> dict:
    return {
        "dry_run": True,
        "trading_mode": "futures",
        "margin_mode": "cross",
        "exchange": {
            "name": "binance",
            "pair_whitelist": ["ETH/USDT:USDT"],
        },
    }


class TestHedgeConfig(TestCase):
    def test_schema_contains_p1_fields(self) -> None:
        properties = CONF_SCHEMA["properties"]

        self.assertIn("position_mode", properties)
        self.assertIn("hedge_mode_enabled", properties)
        self.assertIn("managed_pair", properties)
        self.assertIn("hedge", properties)
        hedge_properties = properties["hedge"]["properties"]
        self.assertIn("rest_proxy_url", hedge_properties)
        self.assertIn("websocket_proxy_url", hedge_properties)
        self.assertIn("trust_env_proxy", hedge_properties)

    def test_missing_fields_default_to_safe_oneway(self) -> None:
        config = base_config()

        result = normalize_hedge_config(config)

        self.assertIs(result.position_mode, PositionMode.ONEWAY)
        self.assertFalse(result.enabled)
        self.assertTrue(result.read_only)


    def test_optional_numeric_fields_are_not_materialized_as_null(self) -> None:
        config = base_config()

        result = normalize_hedge_config(config)

        self.assertIsNone(result.target_leverage)
        self.assertIsNone(result.max_gross_notional)
        self.assertIsNone(result.max_gross_exposure_ratio)
        self.assertNotIn("target_leverage", config["hedge"])
        self.assertNotIn("max_gross_notional", config["hedge"])
        self.assertNotIn("max_gross_exposure_ratio", config["hedge"])

    def test_empty_utility_exchange_name_does_not_become_adapter(self) -> None:
        config = {"exchange": {"name": ""}}

        result = normalize_hedge_config(config)

        self.assertIsNone(result.exchange_adapter)
        self.assertIsNone(config["hedge"]["exchange_adapter"])

    def test_unsupported_oneway_exchange_does_not_become_adapter(self) -> None:
        config = {"exchange": {"name": "kraken"}}

        result = normalize_hedge_config(config)

        self.assertIsNone(result.exchange_adapter)
        self.assertIs(config["position_mode"], PositionMode.ONEWAY)

    def test_valid_read_only_single_pair_hedge_config(self) -> None:
        config = base_config()
        config.update(
            {
                "position_mode": "hedge",
                "hedge_mode_enabled": True,
                "managed_pair": "ETH/USDT:USDT",
                "hedge": {
                    "exchange_adapter": "binance",
                    "read_only": True,
                },
            }
        )

        result = validate_hedge_config(config)

        self.assertIs(result.position_mode, PositionMode.HEDGE)
        self.assertTrue(result.enabled)
        self.assertEqual(result.managed_pair, "ETH/USDT:USDT")

    def test_hedge_mode_rejects_spot(self) -> None:
        config = base_config()
        config.update(
            {
                "trading_mode": "spot",
                "position_mode": "hedge",
            }
        )

        with self.assertRaisesRegex(
            OperationalException,
            "requires trading_mode='futures'",
        ):
            validate_hedge_config(config)

    def test_enabled_hedge_rejects_multiple_pairs(self) -> None:
        config = base_config()
        config.update(
            {
                "position_mode": "hedge",
                "hedge_mode_enabled": True,
                "managed_pair": "ETH/USDT:USDT",
                "hedge": {
                    "exchange_adapter": "binance",
                    "read_only": True,
                },
            }
        )
        config["exchange"]["pair_whitelist"].append("BTC/USDT:USDT")

        with self.assertRaisesRegex(
            OperationalException,
            "exactly one whitelisted pair",
        ):
            validate_hedge_config(config)

    def test_enabled_hedge_rejects_execution_during_p1(self) -> None:
        config = base_config()
        config.update(
            {
                "position_mode": "hedge",
                "hedge_mode_enabled": True,
                "managed_pair": "ETH/USDT:USDT",
                "hedge": {
                    "exchange_adapter": "binance",
                    "read_only": False,
                },
            }
        )

        with self.assertRaisesRegex(
            OperationalException,
            "execution is intentionally locked",
        ):
            validate_hedge_config(config)

    def test_normalization_is_idempotent(self) -> None:
        config = base_config()
        first = normalize_hedge_config(config)
        snapshot = deepcopy(config)
        second = normalize_hedge_config(config)

        self.assertEqual(first, second)
        self.assertEqual(snapshot, config)
