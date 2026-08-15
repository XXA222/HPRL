"""Configuration helpers for the Hedge research control plane."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from .execution import ResearchExecutorConfig
from .service import HedgeResearchService


def _research_config(config: dict[str, Any]) -> dict[str, Any]:
    hedge = config.get("hedge", {})
    research = hedge.get("research", {}) if isinstance(hedge, dict) else {}
    if not isinstance(research, dict):
        raise TypeError("hedge.research must be an object")
    return research


def _workspace_root(config: dict[str, Any], research: dict[str, Any]) -> Path:
    user_data_dir = Path(str(config.get("user_data_dir", "user_data")))
    raw_workspace = research.get("workspace")
    workspace = (
        user_data_dir / "hedge_research"
        if raw_workspace is None
        else Path(str(raw_workspace))
    )
    if raw_workspace is not None and not workspace.is_absolute():
        workspace = user_data_dir / workspace
    return workspace


def _executor_config(research: dict[str, Any]) -> ResearchExecutorConfig:
    raw = research.get("execution", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError("hedge.research.execution must be an object")
    return ResearchExecutorConfig(
        enabled=bool(raw.get("enabled", True)),
        max_concurrent=int(raw.get("max_concurrent", 2)),
        max_gpu_jobs=int(raw.get("max_gpu_jobs", 1)),
        max_cpu_training_jobs=int(raw.get("max_cpu_training_jobs", 1)),
        cpu_threads_per_job=int(raw.get("cpu_threads_per_job", 4)),
        max_cpu_threads_total=int(raw.get("max_cpu_threads_total", 0)),
        default_training_device=str(raw.get("default_training_device", "auto")),
        poll_seconds=float(raw.get("poll_seconds", 1.0)),
        cancel_grace_seconds=float(raw.get("cancel_grace_seconds", 5.0)),
        max_log_bytes=int(raw.get("max_log_bytes", 128 * 1024 * 1024)),
        gpu_device=int(raw.get("gpu_device", 0)),
        min_gpu_free_mb=int(raw.get("min_gpu_free_mb", 1024)),
        max_memory_percent=float(raw.get("max_memory_percent", 92.0)),
        max_cpu_percent=float(raw.get("max_cpu_percent", 99.0)),
    )


def build_research_service(config: dict[str, Any]) -> HedgeResearchService:
    research = _research_config(config)
    user_data_dir = Path(str(config.get("user_data_dir", "user_data"))).expanduser().resolve()
    service = HedgeResearchService(
        _workspace_root(config, research),
        job_capacity=int(research.get("job_capacity", 1000)),
        user_data_dir=user_data_dir,
    )
    executor_config = _executor_config(research)
    if executor_config.enabled:
        raw_project_root = research.get("project_root")
        project_root = (
            Path(__file__).resolve().parents[3]
            if raw_project_root is None
            else Path(str(raw_project_root)).expanduser().resolve()
        )
        service.configure_executor(
            project_root=project_root,
            python_executable=sys.executable,
            executor_config=executor_config,
        )
    return service
