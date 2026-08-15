"""High-throughput TD3 variant with a categorical distributional critic."""

from __future__ import annotations

import copy

from ..action_space import action_for_critic
from ..config import HPRLTrainingConfig
from ..device import PrecisionManager, configured_torch_device, require_torch
from ..networks import CategoricalTwinCritic, DeterministicActor
from ..performance import (
    make_adam,
    maybe_cudagraph_mark_step_begin,
    resolve_grad_clip_foreach,
    resolve_polyak_foreach,
)
from .base import FrozenModulePlan, PolyakUpdatePlan, UpdateMetrics, hard_update, make_metrics


torch = require_torch()


class FastTD3Agent:
    def __init__(
        self, obs_dim: int, action_dim: int, config: HPRLTrainingConfig, *, device: str
    ) -> None:
        self.config = config
        self.device = configured_torch_device(config.device, device)
        self._foreach_polyak = resolve_polyak_foreach(config.polyak_backend, self.device)
        self.precision = PrecisionManager(
            self.device,
            enabled=config.mixed_precision,
            dtype=config.amp_dtype,
            grad_clip_foreach=resolve_grad_clip_foreach(
                config.grad_clip_backend, self.device
            ),
        )
        kwargs = dict(hidden_dim=config.hidden_dim, depth=config.hidden_depth)
        self.actor = DeterministicActor(obs_dim, action_dim, **kwargs).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = CategoricalTwinCritic(obs_dim, action_dim, **kwargs).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)
        self._actor_params = tuple(self.actor.parameters())
        self._critic_params = tuple(self.critic.parameters())
        self._critic_freezer = FrozenModulePlan(self.critic)
        self._actor_polyak = PolyakUpdatePlan(
            self.actor_target, self.actor, foreach=self._foreach_polyak
        )
        self._critic_polyak = PolyakUpdatePlan(
            self.critic_target, self.critic, foreach=self._foreach_polyak
        )
        self.actor_opt = make_adam(
            self._actor_params,
            lr=config.learning_rate,
            device=self.device,
            backend=config.optimizer_backend,
        )
        self.critic_opt = make_adam(
            self._critic_params,
            lr=config.learning_rate,
            device=self.device,
            backend=config.optimizer_backend,
        )
        self.update_count = 0
        self.policy_delay = 2
        self.target_noise = 0.1
        self.noise_clip = 0.2
        self.exploration_noise = 0.1
        self.reward_normalization = "return_std"

    @torch.inference_mode()
    def act(self, obs, *, deterministic: bool = False):
        with self.precision.autocast():
            action = self.actor(obs.to(self.device, non_blocking=True))
        if not deterministic:
            action = action + torch.randn_like(action) * self.exploration_noise
        return action_for_critic(self, action.float().clamp(0.0, 1.0), straight_through=False)

    def update(self, batch, *, collect_metrics: bool = True) -> UpdateMetrics:
        cfg = self.config
        maybe_cudagraph_mark_step_begin(self)
        self.update_count += 1
        obs, action = batch.obs, batch.action
        with torch.no_grad(), self.precision.autocast():
            noise = torch.randn_like(action) * self.target_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(batch.next_obs) + noise).clamp(0.0, 1.0)
            next_action = action_for_critic(self, next_action, straight_through=False)
            next_q1, next_q2 = self.critic_target(batch.next_obs, next_action)
            next_q = torch.minimum(next_q1, next_q2)
            target = batch.reward + cfg.gamma * (1.0 - batch.done) * next_q

        with self.precision.autocast():
            logits1, logits2 = self.critic.logits(obs, action)
            critic_loss = self.critic.twin_cross_entropy(logits1, logits2, target)
        critic_grad = self.precision.backward_step(
            critic_loss,
            self.critic_opt,
            self._critic_params,
            cfg.gradient_clip_norm,
        )

        actor_loss_value = torch.zeros((), device=self.device)
        actor_grad = torch.zeros((), device=self.device)
        if self.update_count % self.policy_delay == 0:
            with self.precision.autocast():
                actor_action = action_for_critic(self, self.actor(obs), straight_through=True)
                with self._critic_freezer.frozen():
                    q1, _ = self.critic(obs, actor_action)
                    actor_loss = -q1.mean()
            actor_grad = self.precision.backward_step(
                actor_loss,
                self.actor_opt,
                self._actor_params,
                cfg.gradient_clip_norm,
            )
            actor_loss_value = actor_loss.detach()
            self._actor_polyak.step(cfg.tau)
            self._critic_polyak.step(cfg.tau)

        if not collect_metrics:
            return UpdateMetrics({})
        return make_metrics(True, {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss_value,
            "critic_grad_norm": critic_grad,
            "actor_grad_norm": actor_grad,
        })
