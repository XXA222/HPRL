"""FreqAI experiment configuration materialization for local research jobs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import ResearchKind, ResearchRequest
from .workspace import ResearchWorkspace

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_FREQAI_SCALAR_KEYS = (
    "train_period_days",
    "backtest_period_days",
    "live_retrain_hours",
    "expiration_hours",
    "data_kitchen_thread_count",
    "purge_old_models",
    "activate_tensorboard",
    "save_backtest_models",
    "write_metrics_to_disk",
)
_FREQAI_DICT_KEYS = (
    "model_training_parameters",
    "rl_config",
    "feature_parameters",
    "data_split_parameters",
)
_TRAINING_DEVICES = frozenset({"auto", "cpu", "cuda"})


@dataclass(frozen=True, slots=True)
class TrainingMaterialization:
    identifier: str
    relative_path: str
    path: Path
    continual_learning: bool


def _safe_identifier(value: str) -> str:
    normalized = value.strip()
    if not _IDENTIFIER.fullmatch(normalized):
        raise ValueError(
            "freqai identifier must contain only letters, digits, "
            "dot, dash, underscore"
        )
    return normalized


def default_identifier(job_id: str, kind: ResearchKind) -> str:
    family = "rl" if kind in {ResearchKind.RL_TRAIN, ResearchKind.RL_EVAL} else "ml"
    return f"hedge-research-{family}-{job_id[:16]}"


def normalize_training_device(value: object) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in _TRAINING_DEVICES:
        raise ValueError("training_device must be auto, cpu, or cuda")
    return normalized


def normalize_cpu_threads(value: object, *, default: int = 1) -> int:
    try:
        threads = int(default if value in (None, "") else value)
    except (TypeError, ValueError) as exc:
        raise ValueError("cpu_threads must be an integer") from exc
    if threads < 1:
        raise ValueError("cpu_threads must be positive")
    return threads


def _validate_training_kind(kind: ResearchKind) -> None:
    if kind not in {
        ResearchKind.ML_TRAIN,
        ResearchKind.ML_EVAL,
        ResearchKind.RL_TRAIN,
        ResearchKind.RL_EVAL,
    }:
        raise ValueError(
            "FreqAI override is only valid for ML/RL research jobs"
        )


def _training_identifier(
    request: ResearchRequest,
    *,
    job_id: str,
) -> tuple[str, bool]:
    parameters = request.parameters
    resume_identifier = str(
        parameters.get("resume_identifier", "")
    ).strip()
    requested_identifier = str(
        parameters.get("freqai_identifier", "")
    ).strip()
    identifier = _safe_identifier(
        resume_identifier
        or requested_identifier
        or default_identifier(job_id, request.kind)
    )
    continual = bool(
        parameters.get("continual_learning", False)
        or resume_identifier
    )
    return identifier, continual


def _base_freqai_override(
    request: ResearchRequest,
    *,
    identifier: str,
    continual: bool,
) -> dict[str, Any]:
    parameters = request.parameters
    freqai: dict[str, Any] = {
        "enabled": True,
        "identifier": identifier,
        "save_backtest_models": True,
        "write_metrics_to_disk": True,
        "activate_tensorboard": bool(
            parameters.get("activate_tensorboard", True)
        ),
        "continual_learning": continual,
    }
    for key in _FREQAI_SCALAR_KEYS:
        if key in parameters:
            freqai[key] = parameters[key]
    return freqai


def _copy_freqai_dict_parameters(
    freqai: dict[str, Any],
    parameters: dict[str, Any],
) -> None:
    for key in _FREQAI_DICT_KEYS:
        value = parameters.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise TypeError(f"parameters.{key} must be an object")
        freqai[key] = dict(value)


def _apply_training_device(
    freqai: dict[str, Any],
    request: ResearchRequest,
    *,
    resolved_device: str | None,
    cpu_threads: int | None,
) -> None:
    parameters = request.parameters
    device = normalize_training_device(
        resolved_device
        if resolved_device is not None
        else parameters.get("training_device", "auto")
    )
    model_training = dict(
        freqai.get("model_training_parameters", {})
    )
    model_training["device"] = device

    if request.kind in {
        ResearchKind.ML_TRAIN,
        ResearchKind.ML_EVAL,
    }:
        if cpu_threads is not None:
            model_training["cpu_threads"] = normalize_cpu_threads(
                cpu_threads
            )
    else:
        model_training.pop("cpu_threads", None)
        rl_config = dict(freqai.get("rl_config", {}))
        if cpu_threads is not None:
            rl_config["cpu_count"] = normalize_cpu_threads(cpu_threads)
        freqai["rl_config"] = rl_config

    freqai["model_training_parameters"] = model_training


def build_freqai_override(
    request: ResearchRequest,
    *,
    job_id: str,
    resolved_device: str | None = None,
    cpu_threads: int | None = None,
) -> tuple[dict[str, Any], str]:
    """Build a narrow FreqAI-only overlay for one research experiment.

    The overlay intentionally cannot modify exchange credentials, live order
    settings, pairlists, or strategy code. It only controls FreqAI
    experiment/training fields.
    """

    _validate_training_kind(request.kind)
    identifier, continual = _training_identifier(
        request,
        job_id=job_id,
    )
    freqai = _base_freqai_override(
        request,
        identifier=identifier,
        continual=continual,
    )
    _copy_freqai_dict_parameters(freqai, request.parameters)
    _apply_training_device(
        freqai,
        request,
        resolved_device=resolved_device,
        cpu_threads=cpu_threads,
    )

    # Backtesting is the only executable command produced for ML/RL research.
    # The explicit dry_run marker makes the generated overlay safe to reuse for
    # later dry-run evaluation and impossible to mistake for live-write config.
    return {"dry_run": True, "freqai": freqai}, identifier


def materialize_training_override(
    workspace: ResearchWorkspace,
    request: ResearchRequest,
    *,
    job_id: str,
    resolved_device: str | None = None,
    cpu_threads: int | None = None,
) -> TrainingMaterialization:
    payload, identifier = build_freqai_override(
        request,
        job_id=job_id,
        resolved_device=resolved_device,
        cpu_threads=cpu_threads,
    )
    relative = f"jobs/{job_id}/freqai-override.json"
    workspace.write_bytes(
        relative,
        (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        media_type="application/json",
        max_bytes=1024 * 1024,
    )
    return TrainingMaterialization(
        identifier=identifier,
        relative_path=relative,
        path=workspace.resolve(relative),
        continual_learning=bool(payload["freqai"]["continual_learning"]),
    )
