"""XQC-style well-conditioned soft actor-critic for HPRL.

The critic combines batch normalization, weight normalization and categorical cross-entropy value
learning. This is a clean-room HPRL implementation of the published ingredients and API contract.
"""

from __future__ import annotations

import copy
import math

from ..action_space import action_for_critic
from ..config import HPRLTrainingConfig
from ..device import PrecisionManager, configured_torch_device, require_torch
from ..networks import CategoricalTwinCritic, GaussianActor
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


class XQCAgent:
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
        self.critic = CategoricalTwinCritic(
            obs_dim,
            action_dim,
            **kwargs,
            bins=101,
            value_min=-5.0,
            value_max=5.0,
            batch_norm=True,
            weight_norm=True,
        ).to(self.device)
        self.critic.project_weight_norm_to_unit_sphere()
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        hard_update(self.critic_target, self.critic)
        self.critic_target.eval()
        self._actor_params = tuple(self.actor.parameters())
        self._critic_params = tuple(self.critic.parameters())
        self._critic_freezer = FrozenModulePlan(self.critic, eval_mode=True)
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
        self.log_alpha = torch.nn.Parameter(
            torch.tensor(math.log(0.01), device=self.device, dtype=torch.float32)
        )
        self.alpha_opt = make_adam(
            [self.log_alpha],
            lr=config.learning_rate,
            device=self.device,
            backend=config.optimizer_backend,
        )
        self.target_entropy = -0.5 * float(action_dim)
        self._critic_step = OptimizerStepPlan(
            self.precision, self.critic_opt, self._critic_params, config.gradient_clip_norm
        )
        self._actor_step = OptimizerStepPlan(
            self.precision, self.actor_opt, self._actor_params, config.gradient_clip_norm
        )
        self.reward_normalization = "return_std"
        self._xqc_fused_compute = str(config.compile_scope).strip().lower() == "xqc_fused"
        self.update_count = 0
        self.policy_delay = 3


    def _xqc_target_value_surface(self, next_obs, next_action):
        if self._xqc_fused_compute:
            logits1, logits2 = self.critic_target.logits(next_obs, next_action)
            q1, q2 = self.critic_target.twin_expectation_stacked(logits1, logits2)
        else:
            q1, q2 = self.critic_target(next_obs, next_action)
        return torch.minimum(q1, q2)

    def _xqc_critic_loss_surface(self, joined_obs, joined_action, target, rows: int):
        logits1, logits2 = self.critic.logits(joined_obs, joined_action)
        if self._xqc_fused_compute:
            return self.critic.twin_cross_entropy_stacked(logits1[:rows], logits2[:rows], target)
        return self.critic.twin_cross_entropy(logits1[:rows], logits2[:rows], target)

    def _xqc_actor_q_surface(self, obs, critic_action):
        if self._xqc_fused_compute:
            logits1, logits2 = self.critic.logits(obs, critic_action)
            q1, q2 = self.critic.twin_expectation_stacked(logits1, logits2)
        else:
            q1, q2 = self.critic(obs, critic_action)
        return torch.minimum(q1, q2)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.inference_mode()
    def act(self, obs, *, deterministic: bool = False):
        with self.precision.autocast():
            action, _, mean_action = self.actor.sample(
                obs.to(self.device, non_blocking=True), deterministic=deterministic,
                compute_log_prob=False
            )
        selected = (mean_action if deterministic else action).float()
        return action_for_critic(self, selected, straight_through=False)

    def profile_update_stages(self, batch, recorder, *, collect_metrics: bool = True):
        """Execute one semantically identical XQC update with diagnostic stage attribution."""
        cfg = self.config
        maybe_cudagraph_mark_step_begin(self)
        self.update_count += 1
        alpha_detached = self.log_alpha.detach().exp()
        with torch.no_grad():
            self.log_alpha.clamp_(-20.0, 5.0)

        with recorder.record("forward_backward.critic_forward"):
            self.critic.train()
            with torch.no_grad(), self.precision.autocast():
                next_action, next_log_prob, _ = self.actor.sample(batch.next_obs)
                next_action = action_for_critic(self, next_action, straight_through=False)
                next_q = self._xqc_target_value_surface(batch.next_obs, next_action)
                next_q = next_q - alpha_detached * next_log_prob
                target = batch.reward + cfg.gamma * (1.0 - batch.done) * next_q
            with self.precision.autocast():
                joined_obs = torch.cat((batch.obs, batch.next_obs), dim=0)
                joined_action = torch.cat((batch.action, next_action), dim=0)
                rows = batch.obs.shape[0]
                critic_loss = self._xqc_critic_loss_surface(
                    joined_obs, joined_action, target, rows
                )
        with recorder.record("forward_backward.critic_backward_clip"):
            critic_grad = self._critic_step.backward_and_clip(critic_loss)
        with recorder.record("optimizer.critic"):
            self._critic_step.optimizer_step()
            self.critic.project_weight_norm_to_unit_sphere()

        actor_loss_value = torch.zeros((), device=self.device)
        alpha_loss_value = torch.zeros((), device=self.device)
        actor_grad = torch.zeros((), device=self.device)
        if self.update_count % self.policy_delay == 0:
            with recorder.record("forward_backward.actor_forward"):
                with self.precision.autocast():
                    action, log_prob, _ = self.actor.sample(batch.obs)
                    critic_action = action_for_critic(self, action, straight_through=True)
                    with self._critic_freezer.frozen():
                        q = self._xqc_actor_q_surface(batch.obs, critic_action)
                        actor_loss = (alpha_detached * log_prob - q).mean()
            with recorder.record("forward_backward.actor_backward_clip"):
                actor_grad = self._actor_step.backward_and_clip(actor_loss)
            with recorder.record("optimizer.actor"):
                self._actor_step.optimizer_step()
            with recorder.record("optimizer.alpha"):
                alpha_loss = -(
                    self.log_alpha * (log_prob.detach().float() + self.target_entropy)
                ).mean()
                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()
                with torch.no_grad():
                    self.log_alpha.clamp_(-20.0, 5.0)
                actor_loss_value = actor_loss.detach()
                alpha_loss_value = alpha_loss.detach()

        with recorder.record("target.polyak"):
            self._critic_polyak.step(cfg.tau)
        if not collect_metrics:
            return UpdateMetrics({})
        with recorder.record("metrics.materialize"):
            return make_metrics(True, {
                "critic_loss": critic_loss,
                "actor_loss": actor_loss_value,
                "alpha_loss": alpha_loss_value,
                "alpha": self.alpha.detach(),
                "critic_grad_norm": critic_grad,
                "actor_grad_norm": actor_grad,
            })

    def update(self, batch, *, collect_metrics: bool = True) -> UpdateMetrics:
        cfg = self.config
        maybe_cudagraph_mark_step_begin(self)
        self.update_count += 1
        alpha_detached = self.log_alpha.detach().exp()
        with torch.no_grad():
            self.log_alpha.clamp_(-20.0, 5.0)
        self.critic.train()
        with torch.no_grad(), self.precision.autocast():
            next_action, next_log_prob, _ = self.actor.sample(batch.next_obs)
            next_action = action_for_critic(self, next_action, straight_through=False)
            next_q = self._xqc_target_value_surface(batch.next_obs, next_action)
            next_q = next_q - alpha_detached * next_log_prob
            target = batch.reward + cfg.gamma * (1.0 - batch.done) * next_q

        # A joined current/next forward pass makes BatchNorm running statistics represent both
        # Bellman distributions instead of only the replay-state side. The target critic remains
        # in eval mode and uses the Polyak-synchronized running statistics.
        with self.precision.autocast():
            joined_obs = torch.cat((batch.obs, batch.next_obs), dim=0)
            joined_action = torch.cat((batch.action, next_action), dim=0)
            rows = batch.obs.shape[0]
            critic_loss = self._xqc_critic_loss_surface(joined_obs, joined_action, target, rows)
        critic_grad = self._critic_step.step(critic_loss)
        self.critic.project_weight_norm_to_unit_sphere()

        actor_loss_value = torch.zeros((), device=self.device)
        alpha_loss_value = torch.zeros((), device=self.device)
        actor_grad = torch.zeros((), device=self.device)
        if self.update_count % self.policy_delay == 0:
            with self.precision.autocast():
                action, log_prob, _ = self.actor.sample(batch.obs)
                critic_action = action_for_critic(self, action, straight_through=True)
                with self._critic_freezer.frozen():
                    q = self._xqc_actor_q_surface(batch.obs, critic_action)
                    actor_loss = (alpha_detached * log_prob - q).mean()
            actor_grad = self._actor_step.step(actor_loss)

            alpha_loss = -(
                self.log_alpha * (log_prob.detach().float() + self.target_entropy)
            ).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()
            with torch.no_grad():
                self.log_alpha.clamp_(-20.0, 5.0)
            actor_loss_value = actor_loss.detach()
            alpha_loss_value = alpha_loss.detach()
        self._critic_polyak.step(cfg.tau)
        if not collect_metrics:
            return UpdateMetrics({})
        return make_metrics(True, {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "alpha": self.alpha.detach(),
            "critic_grad_norm": critic_grad,
            "actor_grad_norm": actor_grad,
        })
