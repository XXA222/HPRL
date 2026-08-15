"""FastDSAC-inspired continuous distributional SAC for parallel HPRL environments."""

from __future__ import annotations

import copy
import math

from ..action_space import (
    action_for_critic,
    agent_tier_boundaries,
    gaussian_selected_tier_log_prob_from_boundaries,
    gaussian_tier_entropy_from_boundaries,
)
from ..config import HPRLTrainingConfig
from ..device import PrecisionManager, configured_torch_device, require_torch
from ..networks import GaussianActor, GaussianTwinCritic
from ..performance import (
    make_adam,
    maybe_cudagraph_mark_step_begin,
    resolve_grad_clip_foreach,
    resolve_polyak_foreach,
)
from .base import (
    FrozenModulePlan,
    OptimizerStepPlan,
    PolyakUpdatePlan,
    UpdateMetrics,
    hard_update,
    make_metrics,
)


torch = require_torch()


class FastDSACAgent:
    """Maximum-entropy agent with per-action-dimension entropy modulation."""

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
        self.actor = GaussianActor(obs_dim, action_dim, **kwargs).to(self.device)
        self.critic = GaussianTwinCritic(obs_dim, action_dim, **kwargs).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        hard_update(self.critic_target, self.critic)
        self._actor_params = tuple(self.actor.parameters())
        self._critic_params = tuple(self.critic.parameters())
        self._critic_freezer = FrozenModulePlan(self.critic)
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
        self.log_alpha = torch.nn.Parameter(torch.zeros(action_dim, device=self.device))
        self.alpha_opt = make_adam(
            [self.log_alpha],
            lr=config.learning_rate,
            device=self.device,
            backend=config.optimizer_backend,
        )
        self.target_entropy_per_dim = -1.0
        self._tier_entropy_fn = gaussian_tier_entropy_from_boundaries
        self._selected_tier_log_prob_fn = gaussian_selected_tier_log_prob_from_boundaries
        self._critic_step = OptimizerStepPlan(
            self.precision, self.critic_opt, self._critic_params, config.gradient_clip_norm
        )
        self._actor_step = OptimizerStepPlan(
            self.precision, self.actor_opt, self._actor_params, config.gradient_clip_norm
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.inference_mode()
    def act(self, obs, *, deterministic: bool = False):
        with self.precision.autocast():
            action, _, mean = self.actor.sample(
                obs.to(self.device, non_blocking=True), deterministic=deterministic,
                compute_log_prob=False, compute_mean_action=deterministic
            )
        selected = (mean if deterministic else action).float()
        return action_for_critic(self, selected, straight_through=False)

    def _critic_loss_surface(
        self, obs, action, reward, next_obs, done, alpha_detached, boundaries
    ):
        cfg = self.config
        level_count = int(getattr(self, "action_level_count", 0) or 0)
        with torch.no_grad(), self.precision.autocast():
            sampled = self.actor.sample(
                next_obs,
                per_dim_log_prob=True,
                return_params=True,
                compute_log_prob=level_count < 2,
                compute_mean_action=False,
            )
            next_action, next_log_prob_dim, _, next_mean, next_log_std = sampled
            next_action = action_for_critic(self, next_action, straight_through=False)
            if level_count >= 2:
                next_log_prob_dim = self._selected_tier_log_prob_fn(
                    next_mean, next_log_std, next_action, boundaries
                )
            (m1, _), (m2, _) = self.critic_target(next_obs, next_action)
            entropy = (alpha_detached * next_log_prob_dim).sum(dim=-1, keepdim=True)
            bootstrap = torch.minimum(m1, m2) - entropy
            target = reward + cfg.gamma * (1.0 - done) * bootstrap
        with self.precision.autocast():
            params1, params2 = self.critic(obs, action)
            return self.critic.twin_nll(params1, params2, target)

    def _actor_loss_surface(self, obs, alpha_detached, boundaries):
        level_count = int(getattr(self, "action_level_count", 0) or 0)
        with self.precision.autocast():
            sampled = self.actor.sample(
                obs,
                per_dim_log_prob=True,
                return_params=True,
                compute_log_prob=level_count < 2,
                compute_mean_action=False,
            )
            new_action, log_prob_dim, _, policy_mean, policy_log_std = sampled
            critic_action = action_for_critic(self, new_action, straight_through=True)
            (mean1, _), (mean2, _) = self.critic(obs, critic_action)
            if level_count >= 2:
                tier_entropy = self._tier_entropy_fn(
                    policy_mean, policy_log_std, boundaries
                )
                entropy_term = -(alpha_detached * tier_entropy).sum(dim=-1, keepdim=True)
            else:
                tier_entropy = torch.empty((0,), device=obs.device, dtype=obs.dtype)
                entropy_term = (alpha_detached * log_prob_dim).sum(dim=-1, keepdim=True)
            actor_loss = (entropy_term - torch.minimum(mean1, mean2)).mean()
        return actor_loss, tier_entropy, log_prob_dim

    @torch.no_grad()
    def _post_update_surface(self):
        self.log_alpha.clamp_(-20.0, 5.0)
        self._critic_polyak.step(self.config.tau)

    def update(self, batch, *, collect_metrics: bool = True) -> UpdateMetrics:
        cfg = self.config
        maybe_cudagraph_mark_step_begin(self)
        level_count = int(getattr(self, "action_level_count", 0) or 0)
        boundaries = agent_tier_boundaries(self) if level_count >= 2 else torch.empty(
            (0,), device=self.device, dtype=torch.float32
        )
        alpha_detached = self.log_alpha.detach().exp()

        critic_loss = self._critic_loss_surface(
            batch.obs, batch.action, batch.reward, batch.next_obs, batch.done,
            alpha_detached, boundaries
        )
        critic_grad = self._critic_step.step(critic_loss)

        with self._critic_freezer.frozen():
            actor_loss, tier_entropy_tensor, log_prob_dim = self._actor_loss_surface(
                batch.obs, alpha_detached, boundaries
            )
        actor_grad = self._actor_step.step(actor_loss)

        tier_entropy = tier_entropy_tensor if level_count >= 2 else None
        # Keep entropy temperature in FP32 even when the policy/critic use autocast.
        if level_count >= 2 and tier_entropy is not None:
            target_entropy = cfg.tier_entropy_target_fraction * math.log(float(level_count))
            alpha_loss = (
                self.log_alpha * (tier_entropy.detach().float() - target_entropy)
            ).mean()
        else:
            alpha_loss = -(
                self.log_alpha
                * (log_prob_dim.detach().float() + self.target_entropy_per_dim)
            ).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        self._post_update_surface()
        if not collect_metrics:
            return UpdateMetrics({})
        return make_metrics(True, {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_mean": self.log_alpha.detach().exp().mean(),
            "tier_entropy_mean": (
                torch.tensor(0.0, device=self.device)
                if tier_entropy is None
                else tier_entropy.detach().mean()
            ),
            "critic_grad_norm": critic_grad,
            "actor_grad_norm": actor_grad,
        })
