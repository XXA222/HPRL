from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from freqtrade.hedge.hprl.adapters import CleanMainlineSignalAdapter, HPRLPlannerSignal
from freqtrade.hedge.hprl.compatibility import assert_clean_mainline_compatible
from freqtrade.hedge.hprl.config import HPRLTrainingConfig
from freqtrade.hedge.hprl.env import VectorizedHedgeEnv
from freqtrade.hedge.hprl.data import TensorMarketDataset
from freqtrade.hedge.hprl.registry import create_agent
from freqtrade.hedge.hprl.errors import HPRLDependencyError


def test_clean_mainline_compatibility_probe_passes() -> None:
    report = assert_clean_mainline_compatible(Path.cwd())
    assert report.compatible
    assert "clean-mainline-v1.2.1" in report.mainline_version


def test_signal_adapter_maps_projected_dual_leg_targets() -> None:
    adapter = CleanMainlineSignalAdapter(("BTC/USDT:USDT",), "hprl-xqc", max_leg_exposure=1.0)
    (signal,) = adapter.decode(torch.tensor([0.7, 0.2]), confidence=0.8, risk_scale=0.6)
    assert isinstance(signal, HPRLPlannerSignal)
    assert signal.long_score == pytest.approx(0.7)
    assert signal.short_score == pytest.approx(0.2)
    assert signal.target_net_ratio == pytest.approx(0.5)


def test_signal_adapter_builds_canonical_snapshot_payload() -> None:
    adapter = CleanMainlineSignalAdapter(("BTC/USDT:USDT",), "hprl-xqc")
    (signal,) = adapter.decode(torch.tensor([0.6, 0.1]))
    now = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
    payload = adapter.signal_snapshot_kwargs(
        signal,
        timeframe="1m",
        candle_close_time=now,
        feature_timestamp=now,
    )
    assert payload["symbol"] == "BTC/USDT:USDT"
    assert payload["model_version"] == "hprl-xqc"
    assert payload["regime"] == "HPRL"
    assert payload["target_net_ratio"] is not None
    assert set(payload) >= {"long_score", "short_score", "allow_new_risk", "risk_scale"}


def test_signal_adapter_target_net_ratio_is_equity_ratio_not_leg_normalized() -> None:
    adapter = CleanMainlineSignalAdapter(
        ("BTC/USDT:USDT",), "hprl-xqc", max_leg_exposure=0.5
    )
    (signal,) = adapter.decode(torch.tensor([0.5, 0.0]))
    assert signal.long_score == pytest.approx(1.0)
    assert signal.target_net_ratio == pytest.approx(0.5)


def test_signal_adapter_rejects_nonfinite_target() -> None:
    adapter = CleanMainlineSignalAdapter(("BTC/USDT:USDT",), "hprl-xqc")
    with pytest.raises(ValueError):
        adapter.decode(torch.tensor([float("nan"), 0.1]))


def test_hprl_public_config_accepts_cpu_cuda_auto_policy() -> None:
    assert HPRLTrainingConfig(device="auto").device == "auto"
    assert HPRLTrainingConfig(device="cpu").device == "cpu"
    assert HPRLTrainingConfig(device="cuda").device == "cuda"
    assert HPRLTrainingConfig(device="cuda:0").device == "cuda:0"
    with pytest.raises(ValueError):
        HPRLTrainingConfig(device="tpu")


def test_vector_environment_uses_auto_and_supports_cuda_when_available() -> None:
    features = torch.zeros((3, 1, 2))
    returns = torch.zeros((3, 1))
    dataset = TensorMarketDataset(features, returns, symbols=("BTC",))
    env = VectorizedHedgeEnv(dataset, device="auto")
    assert env.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        cuda_env = VectorizedHedgeEnv(dataset, device="cuda")
        assert cuda_env.device.type == "cuda"
    else:
        with pytest.raises(HPRLDependencyError):
            VectorizedHedgeEnv(dataset, device="cuda")


def test_registry_resolves_configured_accelerator() -> None:
    config = HPRLTrainingConfig(
        device="auto",
        batch_size=8,
        replay_capacity=32,
        warmup_steps=0,
        hidden_dim=32,
        hidden_depth=1,
    )
    agent = create_agent("fast_td3", 6, 2, config)
    assert agent.device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_dataset_transfer_supports_selected_device() -> None:
    features = torch.zeros((3, 1, 2))
    returns = torch.zeros((3, 1))
    dataset = TensorMarketDataset(features, returns, symbols=("BTC",))
    auto = dataset.to("auto")
    assert auto.features.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(HPRLDependencyError):
            dataset.to("cuda")


def test_replay_buffer_supports_selected_device() -> None:
    from freqtrade.hedge.hprl.replay import TensorReplayBuffer

    replay = TensorReplayBuffer(16, 4, 2, device="auto")
    assert replay.device.type == ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        with pytest.raises(HPRLDependencyError):
            TensorReplayBuffer(16, 4, 2, device="cuda")


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    [
        ("freqtrade.hedge.hprl.algorithms.fast_td3", "FastTD3Agent"),
        ("freqtrade.hedge.hprl.algorithms.fast_dsac", "FastDSACAgent"),
        ("freqtrade.hedge.hprl.algorithms.simba_sac", "SimbaSACAgent"),
        ("freqtrade.hedge.hprl.algorithms.xqc", "XQCAgent"),
        ("freqtrade.hedge.hprl.algorithms.rebrac_v2", "ReBRACv2Agent"),
    ],
)
def test_direct_agent_construction_supports_runtime_device(
    module_name: str, class_name: str
) -> None:
    import importlib

    config = HPRLTrainingConfig(
        device="cpu",
        batch_size=8,
        replay_capacity=32,
        warmup_steps=0,
        hidden_dim=32,
        hidden_depth=1,
    )
    cls = getattr(importlib.import_module(module_name), class_name)
    agent = cls(6, 2, config, device="cpu")
    assert agent.device.type == "cpu"

