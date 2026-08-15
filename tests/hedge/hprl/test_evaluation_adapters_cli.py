from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch

from freqtrade.hedge.hprl.adapters import NoExchangeWriteGuard, ReadonlyTargetAdapter
from freqtrade.hedge.hprl.evaluation import evaluate_trading, walk_forward_folds
from freqtrade.hedge.hprl.registry import available_algorithms


def test_trading_metrics_cover_tail_risk() -> None:
    metrics = evaluate_trading(
        [1000, 1010, 1005, 1025, 1020, 1030],
        periods_per_year=365,
        turnover=3.0,
        fees=2.0,
        funding=-0.5,
    )
    assert metrics.net_return == pytest.approx(0.03)
    assert metrics.max_drawdown > 0
    assert metrics.cvar >= 0
    assert metrics.turnover == 3.0


def test_walk_forward_has_strict_boundaries() -> None:
    folds = walk_forward_folds(100, train=40, validation=10, test=10)
    assert folds[0].train_start == 0
    assert folds[0].train_end == 40
    assert folds[0].validation_start == 41
    assert folds[0].validation_end == 51
    assert folds[0].test_start == 52
    assert folds[0].test_end == 62
    assert folds[1].train_start == 10


def test_readonly_adapter_decodes_dual_legs() -> None:
    adapter = ReadonlyTargetAdapter(("BTC", "ETH"), "xqc-test")
    intents = adapter.decode(torch.tensor([0.6, 0.1, 0.2, 0.4]))
    assert len(intents) == 2
    assert intents[0].target_long_exposure == pytest.approx(0.6)
    assert intents[0].target_short_exposure == pytest.approx(0.1)
    assert intents[0].metadata["exchange_write"] == "forbidden"


def test_exchange_write_guard_is_hard_false() -> None:
    assert NoExchangeWriteGuard.live_order_write is False
    assert NoExchangeWriteGuard.exchange_api_access is False
    assert NoExchangeWriteGuard.assert_safe()


def test_registry_contains_five_modern_agents() -> None:
    assert available_algorithms() == (
        "fast_dsac",
        "fast_td3",
        "rebrac_v2",
        "simba_sac",
        "xqc",
    )


def test_standalone_cli_inspect() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "freqtrade.hedge.hprl", "inspect"],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["existing_rl_modified"] is False
    assert payload["device_policy"] == "cpu-cuda-auto"
    assert payload["live_order_write"] is False
    assert payload["default_device"] == "auto"
    assert payload["gpu_acceleration"]["resident_replay"] is True


def test_standalone_cli_train_smoke_cpu() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "freqtrade.hedge.hprl",
            "train-smoke",
            "--device",
            "cpu",
            "--algorithm",
            "fast_td3",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["status"] == "PASS"
    assert payload["resolved_device"] == "cpu"
    assert payload["environment_device"] == "cpu"
    assert payload["agent_device"] == "cpu"
    assert payload["updates"] > 0
