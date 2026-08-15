"""Reusable neural building blocks for modern HPRL agents."""

from __future__ import annotations

import math

from .device import require_torch


torch = require_torch()
nn = torch.nn
F = torch.nn.functional


def mlp(input_dim: int, hidden_dim: int, output_dim: int, depth: int, *, layer_norm: bool = False):
    if min(input_dim, hidden_dim, output_dim, depth) < 1:
        raise ValueError("MLP dimensions and depth must be positive")
    layers: list[nn.Module] = []
    width = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(width, hidden_dim))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.SiLU())
        width = hidden_dim
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _reparameterized_sigmoid_gaussian(
    mean,
    log_std,
    *,
    deterministic: bool,
    compute_log_prob: bool = True,
    compute_mean_action: bool = True,
):
    """Distribution-object-free sigmoid Gaussian sample/log-prob hot path."""
    if deterministic:
        z = mean
        eps = None
    else:
        eps = torch.randn_like(mean)
        z = mean + log_std.exp() * eps
    action = torch.sigmoid(z)
    squashed_mean = torch.sigmoid(mean) if compute_mean_action else None
    if not compute_log_prob:
        return action, None, squashed_mean
    if eps is None:
        normal_log_prob = -log_std - _LOG_SQRT_2PI
    else:
        normal_log_prob = -0.5 * eps.square() - log_std - _LOG_SQRT_2PI
    log_jacobian = F.logsigmoid(z) + F.logsigmoid(-z)
    return action, normal_log_prob - log_jacobian, squashed_mean


class RunningNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1:
            raise ValueError("RunningNorm dimension must be a positive integer")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("RunningNorm eps must be positive and finite")
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(eps))
        self.eps = eps

    @torch.no_grad()
    def update(self, x) -> None:
        if x.numel() < 1 or x.shape[-1] != self.mean.shape[0] or not torch.isfinite(x).all():
            raise ValueError("RunningNorm update requires non-empty finite matching features")
        x = x.detach().reshape(-1, x.shape[-1])
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps)


class HypersphericalLinear(nn.Module):
    """Linear map with row-normalized weights and feature normalization."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.log_scale = nn.Parameter(torch.zeros(out_features))
        nn.init.orthogonal_(self.weight)

    def _linear_from_normalized_input(self, x):
        weight = F.normalize(self.weight, dim=-1)
        scale = self.log_scale.clamp(-5.0, 5.0).exp()
        return F.linear(x, weight, self.bias) * scale

    def forward(self, x):
        return self._linear_from_normalized_input(F.normalize(x, dim=-1))

    def forward_normalized(self, x):
        """Linear map for an input already normalized by the surrounding Simba graph.

        Simba residual outputs and encoder outputs are explicitly L2-normalized. Repeating the
        same input normalization inside the next hyperspherical layer adds a vector-norm/divide
        surface without changing the ideal hyperspherical transform. Weight normalization and
        learned scale remain identical.
        """
        return self._linear_from_normalized_input(x)


class SimbaResidualBlock(nn.Module):
    def __init__(self, dim: int, expansion: int = 4) -> None:
        super().__init__()
        self.fc1 = HypersphericalLinear(dim, dim * expansion)
        self.fc2 = HypersphericalLinear(dim * expansion, dim)

    def forward(self, x):
        # ``x`` is the normalized output of SimbaEncoder or the previous residual block.
        hidden = F.silu(self.fc1.forward_normalized(x))
        return F.normalize(x + self.fc2(hidden), dim=-1)


class SimbaEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.input = HypersphericalLinear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(SimbaResidualBlock(hidden_dim) for _ in range(depth))

    def forward(self, x):
        x = F.normalize(F.silu(self.input(x)), dim=-1)
        for block in self.blocks:
            x = block(x)
        return x


class GaussianActor(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = 512, depth: int = 3
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.body = mlp(obs_dim, hidden_dim, hidden_dim, depth, layer_norm=True)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def distribution(self, obs):
        h = self.body(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(-5.0, 2.0)
        return mean, log_std

    def sample(
        self,
        obs,
        *,
        deterministic: bool = False,
        per_dim_log_prob: bool = False,
        return_params: bool = False,
        compute_log_prob: bool = True,
        compute_mean_action: bool = True,
    ):
        mean, log_std = self.distribution(obs)
        action, log_prob, squashed_mean = _reparameterized_sigmoid_gaussian(
            mean,
            log_std,
            deterministic=deterministic,
            compute_log_prob=compute_log_prob,
            compute_mean_action=compute_mean_action,
        )
        if log_prob is not None and not per_dim_log_prob:
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        result = (action, log_prob, squashed_mean)
        if return_params:
            return (*result, mean, log_std)
        return result


class SimbaGaussianActor(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = 512, depth: int = 3
    ) -> None:
        super().__init__()
        self.encoder = SimbaEncoder(obs_dim, hidden_dim, depth)
        self.mean = HypersphericalLinear(hidden_dim, action_dim)
        self.log_std = HypersphericalLinear(hidden_dim, action_dim)

    def distribution(self, obs):
        h = self.encoder(obs)
        # Encoder output is already L2-normalized; both heads can consume it directly.
        mean = self.mean.forward_normalized(h)
        log_std = self.log_std.forward_normalized(h).clamp(-5.0, 2.0)
        return mean, log_std

    def sample(
        self,
        obs,
        *,
        deterministic: bool = False,
        return_params: bool = False,
        compute_log_prob: bool = True,
        compute_mean_action: bool = True,
    ):
        mean, log_std = self.distribution(obs)
        action, log_prob, squashed_mean = _reparameterized_sigmoid_gaussian(
            mean,
            log_std,
            deterministic=deterministic,
            compute_log_prob=compute_log_prob,
            compute_mean_action=compute_mean_action,
        )
        if log_prob is not None:
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        result = (action, log_prob, squashed_mean)
        if return_params:
            return (*result, mean, log_std)
        return result


class DeterministicActor(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = 512, depth: int = 3
    ) -> None:
        super().__init__()
        self.net = mlp(obs_dim, hidden_dim, action_dim, depth, layer_norm=True)

    def forward(self, obs):
        return torch.sigmoid(self.net(obs))


class ScalarTwinCritic(nn.Module):
    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = 512, depth: int = 3
    ) -> None:
        super().__init__()
        self.q1 = mlp(obs_dim + action_dim, hidden_dim, 1, depth, layer_norm=True)
        self.q2 = mlp(obs_dim + action_dim, hidden_dim, 1, depth, layer_norm=True)

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=-1)
        return self.q1(x), self.q2(x)


class CategoricalTwinCritic(nn.Module):
    """Twin categorical value critics used by XQC/FastTD3-style agents."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        depth: int = 3,
        bins: int = 101,
        value_min: float = -5.0,
        value_max: float = 5.0,
        *,
        batch_norm: bool = False,
        weight_norm: bool = False,
    ) -> None:
        super().__init__()
        if bins < 2 or not math.isfinite(value_min) or not math.isfinite(value_max):
            raise ValueError("categorical critic support must be finite with at least two bins")
        if value_min >= value_max:
            raise ValueError("categorical critic value_min must be less than value_max")
        self.bins = bins
        self.register_buffer("support", torch.linspace(value_min, value_max, bins))

        def build():
            layers: list[nn.Module] = []
            width = obs_dim + action_dim
            if batch_norm:
                layers.append(nn.BatchNorm1d(width))
            for _ in range(depth):
                layer = nn.Linear(width, hidden_dim)
                if weight_norm:
                    layer = torch.nn.utils.parametrizations.weight_norm(layer)
                layers.append(layer)
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.SiLU())
                width = hidden_dim
            out = nn.Linear(width, bins)
            if weight_norm:
                out = torch.nn.utils.parametrizations.weight_norm(out)
            layers.append(out)
            return nn.Sequential(*layers)

        self.q1 = build()
        self.q2 = build()

    def logits(self, obs, action):
        x = torch.cat((obs, action), dim=-1)
        return self.q1(x), self.q2(x)

    @property
    def value_min(self) -> float:
        return float(self.support[0].item())

    @property
    def value_max(self) -> float:
        return float(self.support[-1].item())

    @torch.no_grad()
    def set_support(self, value_min: float, value_max: float) -> None:
        if not math.isfinite(value_min) or not math.isfinite(value_max):
            raise ValueError("categorical critic support must be finite")
        if value_min >= value_max:
            raise ValueError("categorical critic value_min must be less than value_max")
        self.support.copy_(
            torch.linspace(
                value_min,
                value_max,
                self.bins,
                device=self.support.device,
                dtype=self.support.dtype,
            )
        )

    @torch.no_grad()
    def project_weight_norm_to_unit_sphere(self) -> None:
        """Make PyTorch weight-norm scales exactly unit after an optimizer step.

        XQC uses weight normalization together with an explicit post-update projection to the
        unit sphere.  In PyTorch's parametrization, ``original0`` is the learned row scale.
        """
        for name, parameter in self.named_parameters():
            if name.endswith("parametrizations.weight.original0"):
                parameter.fill_(1.0)

    def expectation_from_logits(self, logits):
        return (torch.softmax(logits, dim=-1) * self.support).sum(dim=-1, keepdim=True)

    def forward(self, obs, action):
        l1, l2 = self.logits(obs, action)
        return self.expectation_from_logits(l1), self.expectation_from_logits(l2)

    def project_scalar(self, target):
        if target.ndim == 1:
            target = target.unsqueeze(-1)
        if target.ndim < 1 or target.shape[-1] != 1:
            raise ValueError("categorical scalar target must have trailing dimension 1")
        support_min = self.support[0].to(dtype=target.dtype)
        support_max = self.support[-1].to(dtype=target.dtype)
        target = target.clamp(support_min, support_max)
        scaled = (target - support_min) / (support_max - support_min) * (self.bins - 1)
        lower = scaled.floor().long().clamp(0, self.bins - 1)
        upper = scaled.ceil().long().clamp(0, self.bins - 1)
        upper_w = scaled - lower.to(scaled.dtype)
        lower_w = 1.0 - upper_w
        probs = torch.zeros(
            (*target.shape[:-1], self.bins), device=target.device, dtype=target.dtype
        )
        probs.scatter_add_(-1, lower, lower_w)
        probs.scatter_add_(-1, upper, upper_w)
        return probs

    def cross_entropy(self, logits, scalar_target):
        target_probs = self.project_scalar(scalar_target.detach())
        return -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()

    def twin_cross_entropy(self, logits1, logits2, scalar_target):
        """Twin categorical loss with one shared scalar-target projection."""
        target_probs = self.project_scalar(scalar_target.detach())
        logp1 = torch.log_softmax(logits1, dim=-1)
        logp2 = torch.log_softmax(logits2, dim=-1)
        return -((target_probs * (logp1 + logp2)).sum(dim=-1)).mean()

    def twin_expectation_stacked(self, logits1, logits2):
        """Evaluate both categorical heads in one batched reduction surface."""
        stacked = torch.stack((logits1, logits2), dim=0)
        values = (torch.softmax(stacked, dim=-1) * self.support).sum(dim=-1, keepdim=True)
        return values[0], values[1]

    def twin_cross_entropy_stacked(self, logits1, logits2, scalar_target):
        """Twin categorical CE using one stacked log-softmax reduction surface."""
        target_probs = self.project_scalar(scalar_target.detach())
        stacked = torch.stack((logits1, logits2), dim=0)
        logp = torch.log_softmax(stacked, dim=-1)
        return -((target_probs * logp.sum(dim=0)).sum(dim=-1)).mean()


class GaussianTwinCritic(nn.Module):
    """Twin continuous Gaussian return distributions for FastDSAC-inspired learning."""

    def __init__(
        self, obs_dim: int, action_dim: int, hidden_dim: int = 512, depth: int = 3
    ) -> None:
        super().__init__()
        self.q1 = mlp(obs_dim + action_dim, hidden_dim, 2, depth, layer_norm=True)
        self.q2 = mlp(obs_dim + action_dim, hidden_dim, 2, depth, layer_norm=True)

    @staticmethod
    def _split(raw):
        mean, log_std = raw.chunk(2, dim=-1)
        return mean, log_std.clamp(-5.0, 2.0)

    def forward(self, obs, action):
        x = torch.cat((obs, action), dim=-1)
        return self._split(self.q1(x)), self._split(self.q2(x))

    @staticmethod
    def nll(params, target):
        mean, log_std = params
        var = (2.0 * log_std).exp()
        term = (target - mean).square() / var + 2.0 * log_std + math.log(2 * math.pi)
        return (0.5 * term).mean()

    @staticmethod
    def twin_nll(params1, params2, target):
        """Twin Gaussian NLL with one fused-style reduction surface."""
        mean1, log_std1 = params1
        mean2, log_std2 = params2
        const = math.log(2.0 * math.pi)
        term1 = (target - mean1).square() * torch.exp(-2.0 * log_std1)
        term2 = (target - mean2).square() * torch.exp(-2.0 * log_std2)
        term1 = term1 + 2.0 * log_std1 + const
        term2 = term2 + 2.0 * log_std2 + const
        return (0.5 * (term1 + term2)).mean()
