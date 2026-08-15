"""Replay Hedge optimization candidates through independent backtests.

Generated overlays contain only allow-listed ``hedge.planner`` / ``hedge.paper``
values. Credentials and live-trading controls from the base configuration are
never copied into the research workspace.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from freqtrade.hedge.optimization.config import parse_optimization_config
from freqtrade.hedge.optimization.config_patch import apply_parameters

from .workspace import ResearchWorkspace


@dataclass(frozen=True, slots=True)
class OptimizationReplayMaterialization:
    optimization_job_id: str
    trial_id: int | str | None
    relative_path: str
    path: Path
    parameters: dict[str, object]
    source_artifact: Path

    def snapshot(self) -> dict[str, object]:
        return {
            "optimization_job_id": self.optimization_job_id,
            "trial_id": self.trial_id,
            "relative_path": self.relative_path,
            "parameters": self.parameters,
            "source_artifact": str(self.source_artifact),
        }


def _load_config(config_path: Path) -> dict[str, Any]:
    path = config_path.expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"config file does not exist: {path}")
    try:
        loader = importlib.import_module("freqtrade.configuration.load_config")
        load_from_files = getattr(loader, "load_from_files")
        payload = load_from_files([str(path)])
    except (ImportError, AttributeError):
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Freqtrade configuration must be an object")
    return payload


def _artifact_path(workspace: ResearchWorkspace, optimization_job_id: str, name: str) -> Path:
    path = workspace.resolve(f"jobs/{optimization_job_id}/outputs/optimization/{name}")
    if not path.is_file():
        raise FileNotFoundError(f"optimization artifact is not available: {name}")
    return path


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"optimization artifact must contain an object: {path.name}")
    return payload


def _safe_trial_token(trial_id: int | str | None) -> str:
    token = "best" if trial_id is None else str(trial_id)
    return "".join(char for char in token if char.isalnum() or char in "-_")[:64] or "candidate"


def materialize_parameter_overlay(
    workspace: ResearchWorkspace,
    *,
    optimization_job_id: str,
    config_path: Path,
    parameters: Mapping[str, object],
    trial_id: int | str | None,
    source_artifact: Path,
) -> OptimizationReplayMaterialization:
    if not parameters:
        raise ValueError("optimization candidate does not contain parameters")
    base_config = _load_config(config_path)
    optimization_config = parse_optimization_config(base_config)
    values = {str(key): value for key, value in parameters.items()}

    # Apply to an empty mapping so only allow-listed optimized paths can enter
    # the overlay. This intentionally excludes exchange credentials and all
    # runtime/live-trading controls from the materialized artifact.
    overlay = apply_parameters({}, optimization_config.parameters, values)
    token = _safe_trial_token(trial_id)
    relative_path = f"jobs/{optimization_job_id}/candidate-{token}-overlay.json"
    workspace.write_json(relative_path, overlay, max_bytes=2 * 1024 * 1024)
    return OptimizationReplayMaterialization(
        optimization_job_id=optimization_job_id,
        trial_id=trial_id,
        relative_path=relative_path,
        path=workspace.resolve(relative_path),
        parameters=values,
        source_artifact=source_artifact,
    )


def materialize_best_parameter_overlay(
    workspace: ResearchWorkspace,
    *,
    optimization_job_id: str,
    config_path: Path,
) -> OptimizationReplayMaterialization:
    best_path = _artifact_path(
        workspace,
        optimization_job_id,
        "optimization-best-parameters.json",
    )
    payload = _load_object(best_path)
    raw_values = payload.get("parameters")
    if not isinstance(raw_values, Mapping):
        raise ValueError("best-parameters artifact does not contain parameters")
    return materialize_parameter_overlay(
        workspace,
        optimization_job_id=optimization_job_id,
        config_path=config_path,
        parameters=raw_values,
        trial_id=payload.get("trial_id"),
        source_artifact=best_path,
    )


def ranked_candidates(
    workspace: ResearchWorkspace,
    *,
    optimization_job_id: str,
    limit: int = 5,
) -> tuple[dict[str, Any], ...]:
    """Return highest scalar-score COMPLETE candidates from optimization summary."""

    bounded = max(1, min(int(limit), 100))
    summary_path = _artifact_path(
        workspace,
        optimization_job_id,
        "optimization-summary.json",
    )
    payload = _load_object(summary_path)
    raw_trials = payload.get("trials", ())
    if not isinstance(raw_trials, Sequence) or isinstance(raw_trials, (str, bytes)):
        raise ValueError("optimization summary trials must be an array")
    rows: list[dict[str, Any]] = []
    for raw in raw_trials:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("status", "")).lower() != "complete":
            continue
        parameters = raw.get("parameters")
        if not isinstance(parameters, Mapping) or not parameters:
            continue
        score = raw.get("scalar_score")
        try:
            numeric_score = float(score)
        except (TypeError, ValueError):
            continue
        row = dict(raw)
        row["scalar_score"] = numeric_score
        rows.append(row)
    rows.sort(
        key=lambda item: (
            -float(item["scalar_score"]),
            int(item.get("trial_id", 2**31 - 1)),
        )
    )
    return tuple(rows[:bounded])
