"""ReBRAC-v2-inspired offline RL agent for HPRL.

This implementation preserves the main published design ideas relevant to the project: an
exact-likelihood conditional flow actor, mixed likelihood/MSE/MAE behavior regularization,
categorical residual value learning, staged optimization, and multi-sample inference.  It is a
clean-room project implementation, not the authors' reference source.
"""

from __future__ import annotations

import copy
import math

from ..action_space import action_for_critic
from ..config import HPRLTrainingConfig
from ..device import PrecisionManager, configured_torch_device, require_torch
from ..networks import CategoricalTwinCritic, mlp
from ..performance import (
    discounted_returns_scan,
    make_adam,
    maybe_cudagraph_mark_step_begin,
    resolve_grad_clip_foreach,
    resolve_polyak_foreach,
    resolve_rebrac_flow_precision,
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
nn = torch.nn


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, mask, hidden_dim: int) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.register_buffer("mask", mask)
        self.register_buffer("inv_mask", 1.0 - mask)
        self.net = mlp(obs_dim + action_dim, hidden_dim, action_dim * 2, 2, layer_norm=True)

    def obs_projection(self, obs):
        """First-layer observation contribution, reusable across forward/inverse paths."""
        first = self.net[0]
        return torch.nn.functional.linear(
            obs,
            first.weight[:, : self.obs_dim],
            first.bias,
        )

    def _params_from_obs_projection(
        self, value, obs_projection, *, stable_fp32: bool = False
    ):
        masked = value * self.mask
        first = self.net[0]
        hidden = obs_projection + torch.nn.functional.linear(
            masked,
            first.weight[:, self.obs_dim :],
            None,
        )
        for index, module in enumerate(self.net):
            if index > 0:
                hidden = module(hidden)
        # In mixed likelihood mode cross the BF16->FP32 boundary once for the
        # whole coupling output instead of independently casting masked/shift/log_scale.
        if stable_fp32:
            hidden = hidden.float()
        shift, log_scale = hidden.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * self.inv_mask
        shift = shift * self.inv_mask
        return masked, shift, log_scale

    def forward_from_obs_projection(self, z, obs_projection, *, stable_fp32: bool = False):
        masked, shift, log_scale = self._params_from_obs_projection(
            z, obs_projection, stable_fp32=stable_fp32
        )
        out = masked + self.inv_mask * (z * torch.exp(log_scale) + shift)
        return out, log_scale.sum(dim=-1, keepdim=True)

    def inverse_from_obs_projection(
        self, value, obs_projection, *, stable_fp32: bool = False
    ):
        masked, shift, log_scale = self._params_from_obs_projection(
            value, obs_projection, stable_fp32=stable_fp32
        )
        z = masked + self.inv_mask * ((value - shift) * torch.exp(-log_scale))
        return z, -log_scale.sum(dim=-1, keepdim=True)

    def _params(self, value, obs, *, stable_fp32: bool = False):
        masked = value * self.mask
        params = self.net(torch.cat((obs, masked), dim=-1))
        if stable_fp32:
            # One coarse BF16->FP32 boundary per coupling layer. ``value`` is already
            # FP32 on the exact-likelihood inverse path, so avoid three redundant casts.
            params = params.float()
        shift, log_scale = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale) * self.inv_mask
        shift = shift * self.inv_mask
        return masked, shift, log_scale

    def transform(self, z, obs):
        """Forward coupling without the unused log-determinant reduction."""
        masked, shift, log_scale = self._params(z, obs)
        return masked + self.inv_mask * (z * torch.exp(log_scale) + shift)

    def forward(self, z, obs):
        masked, shift, log_scale = self._params(z, obs)
        out = masked + self.inv_mask * (z * torch.exp(log_scale) + shift)
        return out, log_scale.sum(dim=-1, keepdim=True)

    def inverse(self, value, obs, *, stable_fp32: bool = False):
        masked, shift, log_scale = self._params(value, obs, stable_fp32=stable_fp32)
        z = masked + self.inv_mask * ((value - shift) * torch.exp(-log_scale))
        return z, -log_scale.sum(dim=-1, keepdim=True)


class ConditionalFlowActor(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = 512, flow_layers: int = 4
    ) -> None:
        super().__init__()
        if action_dim < 1 or flow_layers < 1:
            raise ValueError("flow action dimension and flow_layers must be positive")
        self.action_dim = action_dim
        layers = []
        for index in range(flow_layers):
            mask = torch.tensor(
                [float((dim + index) % 2 == 0) for dim in range(action_dim)],
                dtype=torch.float32,
            )
            # A 1D flow would otherwise be fully masked every other layer.
            if action_dim == 1:
                mask = torch.tensor([float(index % 2)], dtype=torch.float32)
            layers.append(ConditionalAffineCoupling(obs_dim, action_dim, mask, hidden_dim))
        self.layers = nn.ModuleList(layers)

    def _base_log_prob(self, z):
        return (-0.5 * (z.square() + math.log(2.0 * math.pi))).sum(dim=-1, keepdim=True)

    def sample(
        self, obs, *, samples: int = 1, deterministic: bool = False,
        compute_log_prob: bool = True,
    ):
        if samples < 1:
            raise ValueError("flow sample count must be positive")
        batch = obs.shape[0]
        expanded_obs = obs if samples == 1 else obs.repeat_interleave(samples, dim=0)
        shape = (batch * samples, self.action_dim)
        z = (
            torch.zeros(shape, device=obs.device, dtype=obs.dtype)
            if deterministic
            else torch.randn(shape, device=obs.device, dtype=obs.dtype)
        )
        value = z
        if not compute_log_prob:
            for layer in self.layers:
                value = layer.transform(value, expanded_obs)
            return torch.sigmoid(value), None, expanded_obs

        log_det = None
        for layer in self.layers:
            value, delta = layer(value, expanded_obs)
            log_det = delta if log_det is None else log_det + delta
        action = torch.sigmoid(value)
        sigmoid_log_det = (
            torch.nn.functional.logsigmoid(value) + torch.nn.functional.logsigmoid(-value)
        ).sum(dim=-1, keepdim=True)
        log_prob = self._base_log_prob(z) - log_det - sigmoid_log_det
        return action, log_prob, expanded_obs

    def sample_and_data_log_prob(
        self, obs, data_action, *, stable_fp32: bool = False
    ):
        """Share first-layer observation GEMMs across policy sample and data likelihood.

        For one actor update both flow directions use the same observation batch and the same
        coupling parameters. Linear([obs, masked_action]) is exactly separable into an observation
        projection plus an action projection, so caching the former removes duplicate GEMMs without
        changing the flow, likelihood, loss, or parameterization.
        """
        projections = [layer.obs_projection(obs) for layer in self.layers]
        z = torch.randn(
            (obs.shape[0], self.action_dim),
            device=obs.device,
            dtype=obs.dtype,
        )
        value = z
        sample_log_det = None
        for layer, projection in zip(self.layers, projections, strict=True):
            value, delta = layer.forward_from_obs_projection(
                value, projection, stable_fp32=False
            )
            sample_log_det = delta if sample_log_det is None else sample_log_det + delta
        sampled_action = torch.sigmoid(value)
        sample_sigmoid_log_det = (
            torch.nn.functional.logsigmoid(value)
            + torch.nn.functional.logsigmoid(-value)
        ).sum(dim=-1, keepdim=True)
        sample_log_prob = self._base_log_prob(z) - sample_log_det - sample_sigmoid_log_det

        if stable_fp32:
            data_action = data_action.float()
        eps = torch.finfo(data_action.dtype).eps
        clipped = data_action.clamp(eps, 1.0 - eps)
        inverse_value = torch.logit(clipped)
        inverse_log_det = None
        for layer, projection in zip(
            reversed(self.layers), reversed(projections), strict=True
        ):
            inverse_value, delta = layer.inverse_from_obs_projection(
                inverse_value, projection, stable_fp32=stable_fp32
            )
            inverse_log_det = delta if inverse_log_det is None else inverse_log_det + delta
        sigmoid_log_det = (torch.log(clipped) + torch.log1p(-clipped)).sum(
            dim=-1, keepdim=True
        )
        data_log_prob = (
            self._base_log_prob(inverse_value) + inverse_log_det - sigmoid_log_det
        )
        return sampled_action, sample_log_prob, data_log_prob

    def log_prob(self, obs, action, *, stable_fp32: bool = False):
        if stable_fp32:
            obs = obs.float()
            action = action.float()
        eps = torch.finfo(action.dtype).eps
        clipped = action.clamp(eps, 1.0 - eps)
        value = torch.logit(clipped)
        inverse_log_det = None
        for layer in reversed(self.layers):
            value, delta = layer.inverse(value, obs, stable_fp32=stable_fp32)
            inverse_log_det = delta if inverse_log_det is None else inverse_log_det + delta
        sigmoid_log_det = (torch.log(clipped) + torch.log1p(-clipped)).sum(
            dim=-1, keepdim=True
        )
        return self._base_log_prob(value) + inverse_log_det - sigmoid_log_det


class ReBRACv2Agent:
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
        self.actor = ConditionalFlowActor(obs_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = CategoricalTwinCritic(
            obs_dim, action_dim, config.hidden_dim, config.hidden_depth
        ).to(self.device)
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
        self.behavior_likelihood_weight = 0.10
        self.behavior_mse_weight = 0.25
        self.behavior_mae_weight = 0.10
        self.flow_likelihood_precision = resolve_rebrac_flow_precision(
            config.flow_likelihood_precision,
            self.device,
            getattr(config, "hardware_profile", "auto"),
            mixed_precision_enabled=self.precision.enabled,
        )
        self.warmup_updates = max(10, config.warmup_steps // max(config.batch_size, 1))
        self._critic_step = OptimizerStepPlan(
            self.precision, self.critic_opt, self._critic_params, config.gradient_clip_norm
        )
        self._actor_step = OptimizerStepPlan(
            self.precision, self.actor_opt, self._actor_params, config.gradient_clip_norm
        )
        self.update_count = 0

    @torch.no_grad()
    def calibrate_return_support(self, reward, done) -> tuple[float, float]:
        """Set categorical support from ordered offline discounted returns.

        Offline datasets can have reward scales that differ by orders of magnitude.  A fixed
        categorical range silently clips Bellman targets, so the support is calibrated once from
        Monte Carlo returns while including zero as a conservative anchor.
        """
        reward = reward.to(self.device, dtype=torch.float32).reshape(-1)
        done = done.to(self.device, dtype=torch.float32).reshape(-1)
        if reward.numel() < 1 or reward.shape != done.shape:
            raise ValueError("offline reward/done must be non-empty matching vectors")
        if not torch.isfinite(reward).all() or not torch.isfinite(done).all():
            raise ValueError("offline reward/done must be finite")
        if not bool(((done == 0.0) | (done == 1.0)).all()):
            raise ValueError("offline done values must be 0/1")
        returns = discounted_returns_scan(
            reward,
            done,
            self.config.gamma,
            backend=self.config.return_scan_backend,
        )
        low = min(float(returns.min().item()), 0.0)
        high = max(float(returns.max().item()), 0.0)
        span = high - low
        padding = max(span * 0.025, 1e-4)
        value_min = low - padding
        value_max = high + padding
        self.critic.set_support(value_min, value_max)
        self.critic_target.set_support(value_min, value_max)
        return value_min, value_max

    @torch.inference_mode()
    def act(self, obs, *, deterministic: bool = True, samples: int = 16):
        obs = obs.to(self.device, non_blocking=True)
        sample_count = 1 if deterministic else max(1, samples)
        with self.precision.autocast():
            action, _, expanded_obs = self.actor.sample(
                obs, samples=sample_count, deterministic=deterministic, compute_log_prob=False
            )
            critic_action = action_for_critic(self, action, straight_through=False)
            q1, q2 = self.critic(expanded_obs, critic_action)
            score = torch.minimum(q1, q2).reshape(obs.shape[0], sample_count)
        candidates = critic_action.reshape(obs.shape[0], sample_count, -1)
        best = score.argmax(dim=1)
        selected = candidates[torch.arange(obs.shape[0], device=obs.device), best]
        return selected.float()

    def _critic_loss_surface(self, obs, action, reward, next_obs, done):
        cfg = self.config
        with torch.no_grad(), self.precision.autocast():
            next_action, _, _ = self.actor.sample(next_obs, compute_log_prob=False)
            next_action = action_for_critic(self, next_action, straight_through=False)
            nq1, nq2 = self.critic_target(next_obs, next_action)
            target = reward + cfg.gamma * (1.0 - done) * torch.minimum(nq1, nq2)
        with self.precision.autocast():
            l1, l2 = self.critic.logits(obs, action)
            return self.critic.twin_cross_entropy(l1, l2, target)

    def _actor_loss_surface(self, obs, data_action):
        cfg = self.config
        use_paired_flow = bool(cfg.flow_obs_projection_reuse) and (
            not self.precision.enabled or self.flow_likelihood_precision == "mixed"
        )
        with self.precision.autocast():
            if use_paired_flow:
                sampled, log_prob, data_log_prob = self.actor.sample_and_data_log_prob(
                    obs, data_action,
                    stable_fp32=self.flow_likelihood_precision == "mixed",
                )
            else:
                sampled, log_prob, _ = self.actor.sample(obs)
                data_log_prob = None
            critic_sampled = action_for_critic(self, sampled, straight_through=True)
            q1, q2 = self.critic(obs, critic_sampled)
            q_loss = -torch.minimum(q1, q2).mean()
            behavior_delta = critic_sampled - data_action
            mse = behavior_delta.square().mean()
            mae = behavior_delta.abs().mean()
        if data_log_prob is None:
            if self.flow_likelihood_precision == "mixed":
                with self.precision.autocast():
                    data_log_prob = self.actor.log_prob(
                        obs, data_action, stable_fp32=True
                    )
            else:
                data_log_prob = self.actor.log_prob(
                    obs.float(), data_action.float(), stable_fp32=False
                )
        likelihood = -data_log_prob.float().mean()
        actor_loss = (
            q_loss.float()
            + self.behavior_likelihood_weight * likelihood
            + self.behavior_mse_weight * mse.float()
            + self.behavior_mae_weight * mae.float()
            + 1e-4 * log_prob.float().square().mean()
        )
        return actor_loss

    @torch.no_grad()
    def _post_update_surface(self):
        self._critic_polyak.step(self.config.tau)

    def update(self, batch, *, collect_metrics: bool = True) -> UpdateMetrics:
        cfg = self.config
        maybe_cudagraph_mark_step_begin(self)
        self.update_count += 1

        critic_loss = self._critic_loss_surface(
            batch.obs, batch.action, batch.reward, batch.next_obs, batch.done
        )
        critic_grad = self._critic_step.step(critic_loss)

        actor_loss_value = torch.zeros((), device=self.device)
        actor_grad = torch.zeros((), device=self.device)
        if self.update_count > self.warmup_updates:
            with self._critic_freezer.frozen():
                actor_loss = self._actor_loss_surface(batch.obs, batch.action)
            actor_grad = self._actor_step.step(actor_loss)
            actor_loss_value = actor_loss.detach()
        self._post_update_surface()
        if not collect_metrics:
            return UpdateMetrics({})
        return make_metrics(True, {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss_value,
            "critic_grad_norm": critic_grad,
            "actor_grad_norm": actor_grad,
            "stage": 0.0 if self.update_count <= self.warmup_updates else 1.0,
        })
