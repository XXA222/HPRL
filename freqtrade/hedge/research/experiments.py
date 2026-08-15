"""Persistent experiment and FreqAI model catalog for Hedge research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .contracts import ResearchJob, ResearchKind
from .workspace import ResearchWorkspace

_MODEL_SUFFIXES = frozenset({".zip", ".pt", ".pth", ".ckpt", ".h5", ".joblib", ".pkl", ".json"})


@dataclass(frozen=True, slots=True)
class ModelFile:
    path: str
    size: int
    modified_at: str
    role: str

    def snapshot(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "modified_at": self.modified_at,
            "role": self.role,
        }


def _file_role(path: Path) -> str:
    name = path.name.lower()
    if name == "best_model.zip" or "best_model" in name:
        return "best"
    if "tfevents" in name:
        return "tensorboard"
    if "checkpoint" in name or path.suffix.lower() == ".ckpt":
        return "checkpoint"
    if "_model" in name or name.endswith("model.zip"):
        return "model"
    if "metadata" in name or "manifest" in name:
        return "metadata"
    return "support"


def scan_model_files(model_root: Path, *, limit: int = 2000) -> tuple[ModelFile, ...]:
    root = model_root.expanduser().resolve()
    if not root.is_dir():
        return ()
    rows: list[ModelFile] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        lower_name = path.name.lower()
        if (
            path.suffix.lower() not in _MODEL_SUFFIXES
            and "best_model" not in lower_name
            and "tfevents" not in lower_name
        ):
            continue
        stat = path.stat()
        rows.append(
            ModelFile(
                path=path.relative_to(root).as_posix(),
                size=int(stat.st_size),
                modified_at=datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
                role=_file_role(path),
            )
        )
        if len(rows) >= limit:
            break
    return tuple(rows)


def latest_metrics(job: ResearchJob) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in job.metrics:
        result[metric.name] = float(metric.value)
    return result


class ExperimentRegistry:
    """One JSON record per completed ML/RL experiment; no database dependency."""

    def __init__(self, workspace: ResearchWorkspace, *, user_data_dir: Path) -> None:
        self.workspace = workspace
        self.user_data_dir = user_data_dir.expanduser().resolve()

    def _record_path(self, job_id: str) -> str:
        return f"experiments/{job_id}/experiment.json"

    def model_root(self, identifier: str) -> Path:
        return (self.user_data_dir / "models" / identifier).resolve()

    def record(self, job: ResearchJob, *, identifier: str) -> dict[str, Any]:
        model_root = self.model_root(identifier)
        files = scan_model_files(model_root)
        payload = {
            "experiment_id": job.job_id,
            "job_id": job.job_id,
            "kind": job.request.kind.value,
            "name": job.request.name,
            "state": job.state.value,
            "identifier": identifier,
            "strategy": str(job.request.parameters.get("strategy", "")),
            "config_path": str(job.request.parameters.get("config_path", "")),
            "timerange": str(job.request.parameters.get("timerange", "")),
            "tags": list(job.request.tags),
            "metrics": latest_metrics(job),
            "model_root": str(model_root),
            "model_files": [item.snapshot() for item in files],
            "model_bytes": sum(item.size for item in files),
            "created_at": job.created_at.isoformat(),
            "finished_at": None if job.finished_at is None else job.finished_at.isoformat(),
            "continual_learning": bool(job.request.parameters.get("continual_learning", False))
            or bool(job.request.parameters.get("resume_identifier")),
            "resume_identifier": str(job.request.parameters.get("resume_identifier", "")),
            "walk_forward_group": str(job.request.parameters.get("walk_forward_group", "")),
            "fold_index": job.request.parameters.get("fold_index"),
        }
        self.workspace.write_json(self._record_path(job.job_id), payload, max_bytes=4 * 1024 * 1024)
        return payload

    def get(self, experiment_id: str) -> dict[str, Any]:
        payload = self.workspace.read_json(self._record_path(experiment_id))
        if not isinstance(payload, dict):
            raise ValueError("experiment record is invalid")
        return payload

    def list(self, *, limit: int = 500) -> tuple[dict[str, Any], ...]:
        rows: list[dict[str, Any]] = []
        root = self.workspace.root / "experiments"
        if not root.is_dir():
            return ()
        for path in root.glob("*/experiment.json"):
            try:
                payload = self.workspace.read_json(path.relative_to(self.workspace.root).as_posix())
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        rows.sort(
            key=lambda item: str(
                item.get("finished_at")
                or item.get("created_at")
                or ""
            ),
            reverse=True,
        )
        return tuple(rows[: max(1, int(limit))])

    def refresh(self, experiment_id: str) -> dict[str, Any]:
        payload = self.get(experiment_id)
        identifier = str(payload.get("identifier", ""))
        files = scan_model_files(self.model_root(identifier)) if identifier else ()
        payload["model_files"] = [item.snapshot() for item in files]
        payload["model_bytes"] = sum(item.size for item in files)
        payload["refreshed_at"] = datetime.now(UTC).isoformat()
        self.workspace.write_json(
            self._record_path(experiment_id),
            payload,
            max_bytes=4 * 1024 * 1024,
        )
        return payload

    def leaderboard(
        self,
        *,
        metric: str,
        maximize: bool = True,
        limit: int = 50,
        kinds: tuple[ResearchKind, ...] = (),
    ) -> tuple[dict[str, Any], ...]:
        rows = []
        allowed = {item.value for item in kinds}
        for item in self.list(limit=5000):
            if allowed and str(item.get("kind")) not in allowed:
                continue
            metrics = item.get("metrics", {})
            if not isinstance(metrics, dict) or metric not in metrics:
                continue
            try:
                score = float(metrics[metric])
            except (TypeError, ValueError):
                continue
            row = dict(item)
            row["leaderboard_metric"] = metric
            row["leaderboard_score"] = score
            rows.append(row)
        rows.sort(key=lambda item: float(item["leaderboard_score"]), reverse=maximize)
        return tuple(rows[: max(1, int(limit))])

    def model_catalog(self, *, limit_identifiers: int = 200) -> tuple[dict[str, Any], ...]:
        models_root = self.user_data_dir / "models"
        if not models_root.is_dir():
            return ()
        rows: list[dict[str, Any]] = []
        for directory in sorted((item for item in models_root.iterdir() if item.is_dir())):
            files = scan_model_files(directory)
            if not files:
                continue
            latest = max((item.modified_at for item in files), default="")
            rows.append(
                {
                    "identifier": directory.name,
                    "root": str(directory.resolve()),
                    "files": [item.snapshot() for item in files],
                    "file_count": len(files),
                    "bytes": sum(item.size for item in files),
                    "latest_modified_at": latest,
                    "best_model_count": sum(item.role == "best" for item in files),
                    "checkpoint_count": sum(item.role == "checkpoint" for item in files),
                    "tensorboard_count": sum(item.role == "tensorboard" for item in files),
                }
            )
        rows.sort(key=lambda item: str(item["latest_modified_at"]), reverse=True)
        return tuple(rows[: max(1, int(limit_identifiers))])
