"""Config-driven HPRL runtime construction without modifying legacy RL entry points."""

from __future__ import annotations

from dataclasses import dataclass

from .action_space import configure_agent_action_levels
from .config import HPRLConfig
from .env import VectorizedHedgeEnv
from .registry import create_agent
from .trainer import OfflineTrainer, OnlineTrainer


@dataclass(frozen=True, slots=True)
class OnlineRuntime:
    env: VectorizedHedgeEnv
    agent: object
    trainer: OnlineTrainer

    def close(self, *, aggressive: bool = False) -> None:
        self.trainer.close(close_environment=True, aggressive=aggressive)

    def __enter__(self) -> "OnlineRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(aggressive=True)


@dataclass(frozen=True, slots=True)
class OfflineRuntime:
    agent: object
    trainer: OfflineTrainer

    def close(self, *, release_source: bool = False, aggressive: bool = False) -> None:
        self.trainer.close(release_source=release_source, aggressive=aggressive)

    def __enter__(self) -> "OfflineRuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(release_source=True, aggressive=True)


def build_online_runtime(dataset, config: HPRLConfig) -> OnlineRuntime:
    """Build environment, agent and trainer from one device authority in ``config.training``."""
    device = config.training.device
    env = VectorizedHedgeEnv(
        dataset,
        config.environment,
        device=device,
        memory_config=config.memory,
    )
    agent = create_agent(
        config.training.algorithm,
        env.observation_dim,
        env.action_dim,
        config.training,
        device=device,
    )
    if env.action_level_count >= 2:
        configure_agent_action_levels(agent, env.action_level_count)
    trainer = OnlineTrainer(env, agent, config.training, config.memory)
    return OnlineRuntime(env=env, agent=agent, trainer=trainer)


def build_offline_runtime(dataset, config: HPRLConfig) -> OfflineRuntime:
    """Build an offline agent/trainer from the same configured CPU/CUDA device authority."""
    agent = create_agent(
        config.training.algorithm,
        dataset.observation_dim,
        dataset.action_dim,
        config.training,
        device=config.training.device,
    )
    if config.environment.action.mode.strip().lower() == "tiered":
        configure_agent_action_levels(agent, config.environment.action.level_count)
    trainer = OfflineTrainer(
        dataset,
        agent,
        config.training,
        device=config.training.device,
        memory_config=config.memory,
        action_config=config.environment.action,
    )
    return OfflineRuntime(agent=agent, trainer=trainer)
