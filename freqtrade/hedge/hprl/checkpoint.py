"""Atomic HPRL checkpoint persistence with resumable CPU/CUDA training state."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping

from .device import require_torch, torch_device


torch = require_torch()

_MODULE_NAMES = ("actor", "critic", "actor_target", "critic_target")
_OPTIMIZER_NAMES = ("actor_opt", "critic_opt", "alpha_opt")
_TENSOR_NAMES = ("log_alpha",)


def _serializable_metadata(metadata: Mapping[str, object]) -> tuple[dict[str, object], str]:
    payload = dict(metadata)
    try:
        text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint metadata must be JSON serializable") from exc
    return payload, text + "\n"


def _capture_agent_state(agent) -> dict[str, object]:
    modules = {
        name: getattr(agent, name).state_dict()
        for name in _MODULE_NAMES
        if hasattr(agent, name)
    }
    optimizers = {
        name: getattr(agent, name).state_dict()
        for name in _OPTIMIZER_NAMES
        if hasattr(agent, name)
    }
    tensors = {
        name: getattr(agent, name).detach().clone()
        for name in _TENSOR_NAMES
        if hasattr(agent, name)
    }
    scalars = {
        name: value
        for name, value in vars(agent).items()
        if isinstance(value, (bool, int, float, str)) and not name.startswith("_")
    }
    state: dict[str, object] = {
        "modules": modules,
        "optimizers": optimizers,
        "tensors": tensors,
        "scalars": scalars,
        "torch_rng_state": torch.get_rng_state(),
    }
    precision = getattr(agent, "precision", None)
    scaler = getattr(precision, "scaler", None)
    if scaler is not None:
        state["amp_scaler"] = scaler.state_dict()
    agent_device = getattr(agent, "device", torch.device("cpu"))
    if torch.device(agent_device).type == "cuda" and torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _clone_state_tree_to_cpu(value):
    """Deep-clone a checkpoint state tree onto CPU for race-free asynchronous writes."""
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, dict):
        return {key: _clone_state_tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_state_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_tree_to_cpu(item) for item in value)
    return value


def capture_checkpoint_payload(
    agent, metadata: Mapping[str, object], *, cpu_snapshot: bool = True
) -> tuple[dict[str, object], str]:
    """Capture one self-consistent checkpoint payload in the training thread.

    Async writers must never traverse a live optimizer/module state while training mutates it.
    The default CPU snapshot deliberately pays the state-copy cost on the producer side and
    moves JSON/torch serialization plus filesystem I/O to the background worker.
    """
    metadata_payload, metadata_text = _serializable_metadata(metadata)
    agent_state = _capture_agent_state(agent)
    if cpu_snapshot:
        agent_state = _clone_state_tree_to_cpu(agent_state)
    state = {
        "schema": 4,
        "agent_class": f"{type(agent).__module__}.{type(agent).__qualname__}",
        "metadata": metadata_payload,
        "agent_state": agent_state,
    }
    return state, metadata_text


def write_checkpoint_payload(
    path: str | Path, state: Mapping[str, object], metadata_text: str
) -> Path:
    """Atomically persist a previously captured checkpoint payload."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    sidecar = target.with_suffix(target.suffix + ".json")
    sidecar_tmp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    try:
        torch.save(dict(state), tmp)
        sidecar_tmp.write_text(metadata_text, encoding="utf-8")
        os.replace(tmp, target)
        os.replace(sidecar_tmp, sidecar)
    finally:
        tmp.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)
    return target


def save_checkpoint(path: str | Path, agent, metadata: Mapping[str, object]) -> Path:
    state, metadata_text = capture_checkpoint_payload(agent, metadata, cpu_snapshot=False)
    return write_checkpoint_payload(path, state, metadata_text)


def _load_schema1(state: Mapping[str, object], agent) -> None:
    agent.actor.load_state_dict(state["actor"])
    agent.critic.load_state_dict(state["critic"])
    if hasattr(agent, "actor_target"):
        agent.actor_target.load_state_dict(agent.actor.state_dict())
    if hasattr(agent, "critic_target"):
        agent.critic_target.load_state_dict(agent.critic.state_dict())


def _load_schema2(agent_state: Mapping[str, object], agent, *, restore_rng: bool) -> None:
    modules = dict(agent_state.get("modules", {}))
    for name in _MODULE_NAMES:
        if name in modules and hasattr(agent, name):
            getattr(agent, name).load_state_dict(modules[name])
    if "actor_target" not in modules and hasattr(agent, "actor_target"):
        agent.actor_target.load_state_dict(agent.actor.state_dict())
    if "critic_target" not in modules and hasattr(agent, "critic_target"):
        agent.critic_target.load_state_dict(agent.critic.state_dict())

    tensors = dict(agent_state.get("tensors", {}))
    for name, value in tensors.items():
        if hasattr(agent, name):
            target = getattr(agent, name)
            if torch.is_tensor(target):
                target.data.copy_(value.to(device=target.device, dtype=target.dtype))

    scalars = dict(agent_state.get("scalars", {}))
    for name, value in scalars.items():
        if hasattr(agent, name) and isinstance(getattr(agent, name), (bool, int, float, str)):
            setattr(agent, name, value)

    optimizers = dict(agent_state.get("optimizers", {}))
    for name, value in optimizers.items():
        if hasattr(agent, name):
            getattr(agent, name).load_state_dict(value)

    precision = getattr(agent, "precision", None)
    scaler = getattr(precision, "scaler", None)
    if scaler is not None and "amp_scaler" in agent_state:
        scaler.load_state_dict(agent_state["amp_scaler"])

    if restore_rng and "torch_rng_state" in agent_state:
        torch.set_rng_state(agent_state["torch_rng_state"].cpu())
    agent_device = getattr(agent, "device", torch.device("cpu"))
    if (
        restore_rng
        and torch.device(agent_device).type == "cuda"
        and "cuda_rng_state_all" in agent_state
        and torch.cuda.is_available()
    ):
        states = [state.cpu() for state in agent_state["cuda_rng_state_all"]]
        if len(states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(states)


def load_checkpoint(
    path: str | Path,
    agent,
    *,
    map_location: str = "agent",
    restore_rng: bool = False,
) -> dict[str, object]:
    target_request = str(agent.device) if map_location == "agent" else map_location
    target = torch_device(target_request)
    state = torch.load(Path(path), map_location=target, weights_only=True)
    saved_class = state.get("agent_class")
    current_class = f"{type(agent).__module__}.{type(agent).__qualname__}"
    if saved_class is not None and saved_class != current_class:
        raise ValueError(
            f"checkpoint agent class mismatch: saved={saved_class!r}, current={current_class!r}"
        )
    if int(state.get("schema", 1)) >= 2 and "agent_state" in state:
        _load_schema2(state["agent_state"], agent, restore_rng=restore_rng)
    else:
        _load_schema1(state, agent)
    return dict(state.get("metadata", {}))
