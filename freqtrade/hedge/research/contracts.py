"""Contracts for the local-only Hedge research control plane."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ResearchKind(StrEnum):
    BACKTEST = "BACKTEST"
    OPTIMIZATION = "OPTIMIZATION"
    ML_TRAIN = "ML_TRAIN"
    ML_EVAL = "ML_EVAL"
    RL_TRAIN = "RL_TRAIN"
    RL_EVAL = "RL_EVAL"


class ResearchState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


_TERMINAL_STATES = frozenset(
    {ResearchState.SUCCEEDED, ResearchState.FAILED, ResearchState.CANCELED}
)


@dataclass(frozen=True, slots=True)
class ResearchBudget:
    max_seconds: int = 3600
    max_trials: int = 100
    max_workers: int = 1
    max_artifact_bytes: int = 256 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_seconds < 1:
            raise ValueError("max_seconds must be positive")
        if self.max_trials < 1:
            raise ValueError("max_trials must be positive")
        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")
        if self.max_artifact_bytes < 1:
            raise ValueError("max_artifact_bytes must be positive")


@dataclass(frozen=True, slots=True)
class ResearchRequest:
    kind: ResearchKind
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    priority: int = 50
    budget: ResearchBudget = field(default_factory=ResearchBudget)

    def __post_init__(self) -> None:
        normalized = self.name.strip()
        if not normalized:
            raise ValueError("research request name cannot be empty")
        if len(normalized) > 96:
            raise ValueError("research request name is too long")
        if any(not tag.strip() for tag in self.tags):
            raise ValueError("research tags cannot be empty")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("research tags must be unique")
        if not 0 <= int(self.priority) <= 100:
            raise ValueError("research priority must be within [0, 100]")


@dataclass(frozen=True, slots=True)
class ResearchMetric:
    name: str
    value: float
    step: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("metric name cannot be empty")
        if self.step is not None and self.step < 0:
            raise ValueError("metric step cannot be negative")


@dataclass(frozen=True, slots=True)
class ResearchArtifact:
    name: str
    relative_path: str
    media_type: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.relative_path.strip():
            raise ValueError("artifact name/path cannot be empty")
        if self.size < 0:
            raise ValueError("artifact size cannot be negative")
        digest = self.sha256.strip().lower()
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("artifact sha256 must be empty or a hexadecimal digest")


@dataclass(slots=True)
class ResearchJob:
    job_id: str
    request: ResearchRequest
    state: ResearchState = ResearchState.QUEUED
    progress: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    finished_at: datetime | None = None
    message: str = ""
    metrics: list[ResearchMetric] = field(default_factory=list)
    artifacts: list[ResearchArtifact] = field(default_factory=list)
    revision: int = 0

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "kind": self.request.kind.value,
            "name": self.request.name,
            "state": self.state.value,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "started_at": None if self.started_at is None else self.started_at.isoformat(),
            "finished_at": None if self.finished_at is None else self.finished_at.isoformat(),
            "metrics": [
                {"name": item.name, "value": item.value, "step": item.step}
                for item in self.metrics
            ],
            "artifacts": [
                {
                    "name": item.name,
                    "relative_path": item.relative_path,
                    "media_type": item.media_type,
                    "size": item.size,
                }
                for item in self.artifacts
            ],
            "tags": list(self.request.tags),
            "priority": self.request.priority,
            "parameters": dict(self.request.parameters),
            "budget": {
                "max_seconds": self.request.budget.max_seconds,
                "max_trials": self.request.budget.max_trials,
                "max_workers": self.request.budget.max_workers,
                "max_artifact_bytes": self.request.budget.max_artifact_bytes,
            },
            "revision": self.revision,
        }
