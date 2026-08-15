from __future__ import annotations

import pytest

from freqtrade.hedge.hprl.config import (
    HPRLActionConfig,
    HPRLConfig,
    HPRLEnvironmentConfig,
    HPRLTrainingConfig,
)
from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent, TargetExposure, TensorShape
from freqtrade.hedge.hprl.errors import HPRLConfigError


def test_tensor_shape_action_width() -> None:
    shape = TensorShape(environments=32, symbols=3, features=40)
    assert shape.action_width == 6


def test_target_exposure_allows_true_hedge() -> None:
    target = TargetExposure("BTC/USDT:USDT", long=0.7, short=0.2)
    assert target.gross == pytest.approx(0.9)
    assert target.net == pytest.approx(0.5)


def test_invalid_target_exposure_rejected() -> None:
    with pytest.raises(ValueError):
        TargetExposure("BTC", long=-0.1, short=0.2)


def test_config_from_mapping_nested() -> None:
    config = HPRLConfig.from_mapping({
        "environment": {
            "parallel_envs": 8,
            "action": {"max_gross_exposure": 1.2},
            "costs": {"taker_fee_bps": 4.0},
        },
        "training": {"algorithm": "fast_td3", "batch_size": 64, "replay_capacity": 256},
    })
    assert config.environment.parallel_envs == 8
    assert config.environment.action.max_gross_exposure == pytest.approx(1.2)
    assert config.environment.costs.taker_fee_bps == pytest.approx(4.0)
    assert config.training.algorithm == "fast_td3"


def test_invalid_training_capacity_rejected() -> None:
    with pytest.raises(HPRLConfigError):
        HPRLTrainingConfig(batch_size=100, replay_capacity=99)


def test_invalid_action_envelope_rejected() -> None:
    with pytest.raises(HPRLConfigError):
        HPRLActionConfig(max_gross_exposure=0.5, max_abs_net_exposure=0.6)


def test_environment_defaults_are_positive() -> None:
    config = HPRLEnvironmentConfig()
    assert config.initial_equity > 0
    assert config.parallel_envs > 0


def test_planned_intent_is_not_order_contract() -> None:
    intent = PlannedExecutionIntent(
        symbol="BTC/USDT:USDT",
        target_long_exposure=0.6,
        target_short_exposure=0.1,
        confidence=0.8,
        model_id="xqc-a",
    )
    assert intent.target_long_exposure == pytest.approx(0.6)
    assert not hasattr(intent, "order_type")
    assert not hasattr(intent, "exchange")
