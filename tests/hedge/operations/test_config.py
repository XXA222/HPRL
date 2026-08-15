from __future__ import annotations

import ast
import json
from pathlib import Path

from freqtrade.exceptions import OperationalException
from freqtrade.hedge.config import normalize_hedge_config
from freqtrade.hedge.operations.config import operations_config, validate_operations_config

ROOT = Path(__file__).resolve().parents[3]


def test_operations_example_is_safe_and_valid() -> None:
    config = json.loads(
        (ROOT / "config_examples/config_hedge_paper.example.json").read_text(encoding="utf-8")
    )
    assert validate_operations_config(config) == ()
    assert config["hedge"]["read_only"] is True
    assert config["hedge"]["live_trading_enabled"] is False


def test_project_integration_surfaces_are_present_and_parseable() -> None:
    paths = (
        ROOT / "freqtrade/hedge/integration/paper_runtime.py",
        ROOT / "freqtrade/hedge/config.py",
        ROOT / "freqtrade/rpc/api_server/hedge_dashboard.py",
        ROOT / "freqtrade/rpc/api_server/hedge_dashboard_schemas.py",
    )
    for path in paths:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    paper = paths[0].read_text(encoding="utf-8")
    assert "DryRunOperationsRuntime" in paper
    assert "operations_error" in paper
    assert "self.operations.latest.new_risk_enabled" in paper
    cfg = paths[1].read_text(encoding="utf-8")
    assert '"operations"' in cfg
    assert "_OPERATIONS_ALLOWED_KEYS" in cfg
    dashboard = paths[2].read_text(encoding="utf-8")
    assert "operations_snapshot" in dashboard
    js = (ROOT / "freqtrade/rpc/api_server/hedge_ui/app.js").read_text(encoding="utf-8")
    assert "operations" in js


def test_retired_operations_key_is_migrated_before_runtime_use() -> None:
    config = {
        "dry_run": True,
        "hedge": {
            "read_only": True,
            "live_trading_enabled": False,
            "r56": {
                "enabled": True,
                "state_path": "user_data/hedge/operations/runtime-state.json",
            },
        },
    }

    assert validate_operations_config(config) == ("OPERATIONS_CONFIG_NOT_NORMALIZED",)

    normalize_hedge_config(config)

    assert "r56" not in config["hedge"]
    assert config["hedge"]["operations"]["enabled"] is True
    assert operations_config(config) == config["hedge"]["operations"]
    assert validate_operations_config(config) == ()


def test_current_and_retired_operations_keys_fail_closed() -> None:
    config = {
        "dry_run": True,
        "hedge": {
            "read_only": True,
            "live_trading_enabled": False,
            "operations": {"enabled": False},
            "r56": {"enabled": False},
        },
    }

    try:
        normalize_hedge_config(config)
    except OperationalException as exc:
        assert "cannot both be configured" in str(exc)
    else:
        raise AssertionError("ambiguous current/retired operations config must fail closed")


def test_operations_runtime_never_consumes_retired_key_directly() -> None:
    config = {
        "dry_run": False,
        "hedge": {
            "read_only": False,
            "live_trading_enabled": True,
            "r56": {
                "enabled": True,
                "state_path": "../x",
            },
        },
    }
    assert validate_operations_config(config) == ("OPERATIONS_CONFIG_NOT_NORMALIZED",)
