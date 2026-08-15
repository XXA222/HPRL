from __future__ import annotations

from copy import deepcopy

import pytest
from jsonschema import Draft4Validator, validators

from freqtrade.config_schema.config_schema import CONF_SCHEMA
from freqtrade.exceptions import OperationalException
from freqtrade.hedge.config import normalize_hedge_config


def _validator_with_defaults():
    validate_properties = Draft4Validator.VALIDATORS["properties"]

    def set_defaults(validator, properties, instance, schema):
        for prop, subschema in properties.items():
            if "default" in subschema:
                instance.setdefault(prop, deepcopy(subschema["default"]))
        yield from validate_properties(validator, properties, instance, schema)

    return validators.extend(Draft4Validator, {"properties": set_defaults})


def _base_config() -> dict:
    return {
        "dry_run": True,
        "trading_mode": "spot",
        "margin_mode": "",
        "exchange": {"name": "binance", "pair_whitelist": ["ETH/BTC"]},
    }


def test_clean_mainline_schema_has_one_operations_key() -> None:
    hedge_properties = CONF_SCHEMA["properties"]["hedge"]["properties"]

    assert "operations" in hedge_properties
    assert "r56" not in hedge_properties
    assert "default" not in hedge_properties["operations"]


def test_schema_defaults_do_not_materialize_operations_for_ordinary_config() -> None:
    validator_cls = _validator_with_defaults()
    hedge = {}
    hedge_schema = deepcopy(CONF_SCHEMA["properties"]["hedge"])

    validator_cls(hedge_schema).validate(hedge)

    assert "operations" not in hedge
    assert "r56" not in hedge


def test_normalize_schema_normalize_is_a_fixed_point_for_ordinary_config() -> None:
    config = _base_config()
    first = normalize_hedge_config(config)

    hedge_schema = deepcopy(CONF_SCHEMA["properties"]["hedge"])
    validator_cls = _validator_with_defaults()
    validator_cls(hedge_schema).validate(config["hedge"])

    second = normalize_hedge_config(config)

    assert first == second
    assert "r56" not in config["hedge"]


def test_retired_operations_input_is_canonicalized_once() -> None:
    config = _base_config()
    config["hedge"] = {
        "r56": {
            "enabled": False,
            "state_path": "user_data/hedge/operations/runtime-state.json",
        }
    }

    first = normalize_hedge_config(config)
    snapshot = deepcopy(config)
    second = normalize_hedge_config(config)

    assert first == second
    assert snapshot == config
    assert "r56" not in config["hedge"]
    assert config["hedge"]["operations"]["enabled"] is False


def test_ambiguous_operations_input_fails_before_schema_validation() -> None:
    config = _base_config()
    config["hedge"] = {
        "operations": {"enabled": False},
        "r56": {"enabled": False},
    }

    with pytest.raises(OperationalException, match="cannot both be configured"):
        normalize_hedge_config(config)


def test_canonical_operations_survives_hedge_schema_validation() -> None:
    config = _base_config()
    config["hedge"] = {
        "operations": {
            "enabled": False,
            "state_path": "user_data/hedge/operations/runtime-state.json",
        }
    }
    normalize_hedge_config(config)

    hedge_schema = deepcopy(CONF_SCHEMA["properties"]["hedge"])
    _validator_with_defaults()(hedge_schema).validate(config["hedge"])

    assert config["hedge"]["operations"]["enabled"] is False
    assert "r56" not in config["hedge"]


def test_full_config_consistency_does_not_materialize_retired_operations_key(
    default_conf: dict,
) -> None:
    from freqtrade.configuration.config_validation import validate_config_consistency

    config = deepcopy(default_conf)
    validate_config_consistency(config)

    hedge = config.get("hedge", {})
    assert "r56" not in hedge
    assert not (
        "operations" in hedge
        and any(key == "r56" for key in hedge)
    )


def test_full_config_consistency_migrates_retired_operations_input_before_schema(
    default_conf: dict,
) -> None:
    from freqtrade.configuration.config_validation import validate_config_consistency

    config = deepcopy(default_conf)
    config["hedge"] = {"r56": {"enabled": False}}
    validate_config_consistency(config)

    hedge = config["hedge"]
    assert "r56" not in hedge
    assert hedge["operations"]["enabled"] is False
