"""Algorithm registry for HPRL without touching the legacy RL resolver."""

from __future__ import annotations

from collections.abc import Callable

from .config import HPRLTrainingConfig
from .device import configure_acceleration, resolve_device, seed_everything
from .performance import compile_agent_hotpaths, configure_training_runtime


AgentFactory = Callable[[int, int, HPRLTrainingConfig, str], object]


def _factory(module: str, class_name: str) -> AgentFactory:
    def create(obs_dim: int, action_dim: int, config: HPRLTrainingConfig, device: str):
        imported = __import__(module, fromlist=[class_name])
        cls = getattr(imported, class_name)
        return cls(obs_dim, action_dim, config, device=device)

    return create


_FACTORIES: dict[str, AgentFactory] = {
    "xqc": _factory("freqtrade.hedge.hprl.algorithms.xqc", "XQCAgent"),
    "simba_sac": _factory("freqtrade.hedge.hprl.algorithms.simba_sac", "SimbaSACAgent"),
    "fast_dsac": _factory("freqtrade.hedge.hprl.algorithms.fast_dsac", "FastDSACAgent"),
    "fast_td3": _factory("freqtrade.hedge.hprl.algorithms.fast_td3", "FastTD3Agent"),
    "rebrac_v2": _factory("freqtrade.hedge.hprl.algorithms.rebrac_v2", "ReBRACv2Agent"),
}


def available_algorithms() -> tuple[str, ...]:
    return tuple(sorted(_FACTORIES))


def create_agent(
    name: str,
    obs_dim: int,
    action_dim: int,
    config: HPRLTrainingConfig,
    *,
    device: str | None = None,
):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("HPRL algorithm name must be a non-empty string")
    key = name.strip().lower().replace("-", "_")
    config_device = resolve_device(config.device)
    resolved = resolve_device(config.device if device is None else device)
    if resolved.resolved != config_device.resolved:
        raise ValueError(
            "agent device must resolve to the same device as HPRL training configuration"
        )
    configure_acceleration(
        resolved.resolved,
        deterministic=config.deterministic,
        allow_tf32=config.allow_tf32,
        matmul_precision=config.matmul_precision,
        cudnn_benchmark=config.cudnn_benchmark,
        cuda_memory_fraction=config.cuda_memory_fraction,
    )
    seed_everything(config.seed, deterministic=config.deterministic, device=resolved.resolved)
    performance_info = configure_training_runtime(config, resolved.resolved)
    try:
        factory = _FACTORIES[key]
    except KeyError as exc:
        message = f"unknown HPRL algorithm {name!r}; available={available_algorithms()}"
        raise ValueError(message) from exc
    agent = factory(obs_dim, action_dim, config, resolved.resolved)
    setattr(agent, "performance_info", performance_info)
    compiled = compile_agent_hotpaths(agent, config, resolved.resolved)
    setattr(agent, "compiled_hotpaths", compiled)
    return agent
