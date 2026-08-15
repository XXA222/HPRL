"""Integrity-checked model registry for Hedge ML/RL artifacts."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .actions import DEFAULT_ACTION_CATALOG


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def action_catalog_signature() -> str:
    payload = [
        {
            "id": int(spec.action),
            "name": spec.action.name,
            "long_command": spec.long_command.value,
            "short_command": spec.short_command.value,
            "long_fraction": spec.long_fraction,
            "short_fraction": spec.short_fraction,
            "urgency": spec.urgency.value,
        }
        for spec in DEFAULT_ACTION_CATALOG.specs()
    ]
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class ModelManifest:
    model_version: str
    model_kind: str
    observation_schema_signature: str
    source_version: str
    framework: str = "pytorch"
    action_catalog_signature: str = field(default_factory=action_catalog_signature)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metrics: dict[str, float] = field(default_factory=dict)
    artifact_sha256: str = ""

    def __post_init__(self) -> None:
        for name in (
            "model_version",
            "model_kind",
            "observation_schema_signature",
            "source_version",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} cannot be empty")
        if self.framework != "pytorch":
            raise ValueError("only pytorch artifacts are supported")
        if self.artifact_sha256 and len(self.artifact_sha256) != 64:
            raise ValueError("artifact_sha256 must be a SHA-256 hex digest")

    def with_checksum(self, checksum: str) -> ModelManifest:
        data = asdict(self)
        data["artifact_sha256"] = checksum
        return ModelManifest(**data)


class HedgeModelRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_name(name: str) -> str:
        if not _SAFE_NAME.fullmatch(name):
            raise ValueError("model name contains unsupported characters")
        return name

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def save(
        self,
        name: str,
        model: nn.Module,
        manifest: ModelManifest,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        training_state: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        safe_name = self._validate_name(name)
        artifact_path = self.root / f"{safe_name}.pt"
        manifest_path = self.root / f"{safe_name}.manifest.json"
        artifact_tmp = self.root / f".{safe_name}.{os.getpid()}.pt.tmp"
        manifest_tmp = self.root / f".{safe_name}.{os.getpid()}.manifest.tmp"
        payload: dict[str, Any] = {
            "model_state_dict": model.state_dict(),
            "training_state": training_state or {},
        }
        if optimizer is not None:
            payload["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(payload, artifact_tmp)
        checksum = self._sha256(artifact_tmp)
        completed = manifest.with_checksum(checksum)
        manifest_tmp.write_text(
            json.dumps(asdict(completed), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifact_tmp.replace(artifact_path)
        manifest_tmp.replace(manifest_path)
        return artifact_path, manifest_path

    def read_manifest(self, name: str) -> ModelManifest:
        safe_name = self._validate_name(name)
        path = self.root / f"{safe_name}.manifest.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"unable to read model manifest {path}") from exc
        return ModelManifest(**data)

    def load_state(
        self,
        name: str,
        *,
        expected_observation_schema: str,
        expected_action_catalog: str | None = None,
        map_location: str = "cpu",
    ) -> dict[str, Any]:
        safe_name = self._validate_name(name)
        artifact_path = self.root / f"{safe_name}.pt"
        manifest = self.read_manifest(safe_name)
        checksum = self._sha256(artifact_path)
        if checksum != manifest.artifact_sha256:
            raise ValueError("model artifact checksum mismatch")
        if manifest.observation_schema_signature != expected_observation_schema:
            raise ValueError("observation schema is incompatible with model artifact")
        expected_action = expected_action_catalog or action_catalog_signature()
        if manifest.action_catalog_signature != expected_action:
            raise ValueError("action catalogue is incompatible with model artifact")
        payload = torch.load(artifact_path, map_location=map_location, weights_only=True)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError("model artifact payload is invalid")
        payload["manifest"] = manifest
        return payload
