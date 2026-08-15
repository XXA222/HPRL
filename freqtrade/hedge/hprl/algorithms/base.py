"""Shared helpers for HPRL off-policy algorithms."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Mapping

from ..device import require_torch


torch = require_torch()


@dataclass(frozen=True, slots=True)
class UpdateMetrics:
    values: Mapping[str, float]


def make_metrics(collect: bool, values: Mapping[str, object]) -> UpdateMetrics:
    """Materialize scalar metrics only when requested.

    CUDA tensors are stacked before the host copy so one sampled metrics event causes one device
    synchronization instead of one synchronization per metric.
    """
    if not collect:
        return UpdateMetrics({})
    names = tuple(values)
    device = next(
        (value.device for value in values.values() if torch.is_tensor(value)),
        torch.device("cpu"),
    )
    tensors = [
        (
            value.detach().float().reshape(())
            if torch.is_tensor(value)
            else torch.tensor(float(value))
        ).to(device=device)
        for value in values.values()
    ]
    if not tensors:
        return UpdateMetrics({})
    materialized = torch.stack(tensors).cpu().tolist()
    return UpdateMetrics(dict(zip(names, (float(value) for value in materialized), strict=True)))



class PolyakUpdatePlan:
    """Pre-bound Polyak update plan that removes per-update module introspection.

    Parameter and buffer identities are stable for the lifetime of an HPRL agent.  Binding
    them once avoids tuple construction, named-buffer dictionaries, key comparisons and
    repeated dtype branching on every optimizer update.
    """

    __slots__ = ("target_params", "source_params", "floating_target",
                 "floating_source", "copy_target", "copy_source", "foreach")

    def __init__(self, target, source, *, foreach: bool = True) -> None:
        self.target_params = tuple(target.parameters())
        self.source_params = tuple(source.parameters())
        if len(self.target_params) != len(self.source_params):
            raise ValueError("soft-update module parameters do not match")
        target_buffers = dict(target.named_buffers())
        source_buffers = dict(source.named_buffers())
        if target_buffers.keys() != source_buffers.keys():
            raise ValueError("soft-update module buffers do not match")
        floating_target = []
        floating_source = []
        copy_target = []
        copy_source = []
        for name, target_buffer in target_buffers.items():
            source_buffer = source_buffers[name]
            if target_buffer.dtype.is_floating_point:
                floating_target.append(target_buffer)
                floating_source.append(source_buffer)
            else:
                copy_target.append(target_buffer)
                copy_source.append(source_buffer)
        self.floating_target = tuple(floating_target)
        self.floating_source = tuple(floating_source)
        self.copy_target = tuple(copy_target)
        self.copy_source = tuple(copy_source)
        self.foreach = bool(foreach)

    def step(self, tau: float) -> None:
        if not 0.0 < float(tau) <= 1.0:
            raise ValueError("soft-update tau must be in (0, 1]")
        with torch.no_grad():
            if self.target_params:
                if self.foreach:
                    try:
                        torch._foreach_lerp_(self.target_params, self.source_params, tau)
                    except (RuntimeError, TypeError):
                        for target_param, source_param in zip(
                            self.target_params, self.source_params, strict=True
                        ):
                            target_param.lerp_(source_param, tau)
                else:
                    for target_param, source_param in zip(
                        self.target_params, self.source_params, strict=True
                    ):
                        target_param.lerp_(source_param, tau)
            if self.floating_target:
                if self.foreach:
                    try:
                        torch._foreach_lerp_(self.floating_target, self.floating_source, tau)
                    except (RuntimeError, TypeError):
                        for target_buffer, source_buffer in zip(
                            self.floating_target, self.floating_source, strict=True
                        ):
                            target_buffer.lerp_(source_buffer, tau)
                else:
                    for target_buffer, source_buffer in zip(
                        self.floating_target, self.floating_source, strict=True
                    ):
                        target_buffer.lerp_(source_buffer, tau)
            if self.copy_target:
                if self.foreach:
                    try:
                        torch._foreach_copy_(self.copy_target, self.copy_source)
                    except (RuntimeError, TypeError):
                        for target_buffer, source_buffer in zip(
                            self.copy_target, self.copy_source, strict=True
                        ):
                            target_buffer.copy_(source_buffer)
                else:
                    for target_buffer, source_buffer in zip(
                        self.copy_target, self.copy_source, strict=True
                    ):
                        target_buffer.copy_(source_buffer)




class OptimizerStepPlan:
    """Pre-bound backward/clip/optimizer execution plan for one parameter group."""

    __slots__ = ("precision", "optimizer", "parameters", "max_norm")

    def __init__(self, precision, optimizer, parameters, max_norm: float) -> None:
        self.precision = precision
        self.optimizer = optimizer
        self.parameters = parameters if isinstance(parameters, tuple) else tuple(parameters)
        self.max_norm = float(max_norm)
        if self.max_norm <= 0.0:
            raise ValueError("optimizer step max_norm must be positive")

    def backward_and_clip(self, loss):
        return self.precision.backward_and_clip(
            loss, self.optimizer, self.parameters, self.max_norm
        )

    def optimizer_step(self) -> None:
        self.precision.optimizer_step(self.optimizer)

    def step(self, loss):
        norm = self.backward_and_clip(loss)
        self.optimizer_step()
        return norm


class FrozenModulePlan:
    """Reusable freeze context with a pre-bound parameter tuple."""

    __slots__ = ("module", "params", "eval_mode")

    def __init__(self, module, *, eval_mode: bool = False) -> None:
        self.module = module
        self.params = tuple(module.parameters())
        self.eval_mode = bool(eval_mode)

    @contextmanager
    def frozen(self):
        requires_grad = tuple(param.requires_grad for param in self.params)
        was_training = self.module.training
        try:
            for param in self.params:
                param.requires_grad_(False)
            if self.eval_mode:
                self.module.eval()
            yield self.module
        finally:
            for param, enabled in zip(self.params, requires_grad, strict=True):
                param.requires_grad_(enabled)
            self.module.train(was_training)

def soft_update(target, source, tau: float, *, foreach: bool = True) -> None:
    """Polyak-update parameters and stateful buffers such as BatchNorm statistics."""
    if not 0.0 < float(tau) <= 1.0:
        raise ValueError("soft-update tau must be in (0, 1]")
    with torch.no_grad():
        target_params = tuple(target.parameters())
        source_params = tuple(source.parameters())
        if len(target_params) != len(source_params):
            raise ValueError("soft-update module parameters do not match")
        if target_params:
            if foreach:
                try:
                    torch._foreach_lerp_(target_params, source_params, tau)
                except (RuntimeError, TypeError):
                    for target_param, source_param in zip(
                        target_params, source_params, strict=True
                    ):
                        target_param.lerp_(source_param, tau)
            else:
                for target_param, source_param in zip(target_params, source_params, strict=True):
                    target_param.lerp_(source_param, tau)
        target_buffers = dict(target.named_buffers())
        source_buffers = dict(source.named_buffers())
        if target_buffers.keys() != source_buffers.keys():
            raise ValueError("soft-update module buffers do not match")
        floating_target = []
        floating_source = []
        for name, target_buffer in target_buffers.items():
            source_buffer = source_buffers[name]
            if target_buffer.dtype.is_floating_point:
                floating_target.append(target_buffer)
                floating_source.append(source_buffer)
            else:
                target_buffer.copy_(source_buffer)
        if floating_target:
            if foreach:
                try:
                    torch._foreach_lerp_(floating_target, floating_source, tau)
                except (RuntimeError, TypeError):
                    for target_buffer, source_buffer in zip(
                        floating_target, floating_source, strict=True
                    ):
                        target_buffer.lerp_(source_buffer, tau)
            else:
                for target_buffer, source_buffer in zip(
                    floating_target, floating_source, strict=True
                ):
                    target_buffer.lerp_(source_buffer, tau)


def hard_update(target, source) -> None:
    target.load_state_dict(source.state_dict())


def min_q(critic, obs, action):
    q1, q2 = critic(obs, action)
    return torch.minimum(q1, q2)


@contextmanager
def frozen_module(module, *, eval_mode: bool = False):
    """Freeze module parameters while retaining gradients with respect to its inputs."""
    params = tuple(module.parameters())
    requires_grad = tuple(param.requires_grad for param in params)
    was_training = module.training
    try:
        for param in params:
            param.requires_grad_(False)
        if eval_mode:
            module.eval()
        yield module
    finally:
        for param, enabled in zip(params, requires_grad, strict=True):
            param.requires_grad_(enabled)
        module.train(was_training)
