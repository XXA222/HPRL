"""Safe local artifact workspace for Hedge research jobs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .contracts import ResearchArtifact


def normalize_relative_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    pure = PurePosixPath(normalized)
    if not normalized or pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError("unsafe research artifact path")
    if ":" in pure.parts[0] or "\x00" in normalized:
        raise ValueError("unsafe research artifact path")
    return pure.as_posix()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class ResearchWorkspace:
    def __init__(self, root: Path, *, max_artifact_bytes: int = 256 * 1024 * 1024) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if max_artifact_bytes < 1:
            raise ValueError("max_artifact_bytes must be positive")
        self.max_artifact_bytes = max_artifact_bytes

    def resolve(self, relative: str) -> Path:
        normalized = normalize_relative_path(relative)
        target = self.root.joinpath(*PurePosixPath(normalized).parts).resolve()
        if target != self.root and self.root not in target.parents:
            raise ValueError("research artifact escapes workspace")
        return target

    def write_bytes(
        self,
        relative: str,
        payload: bytes,
        *,
        media_type: str,
        max_bytes: int | None = None,
    ) -> ResearchArtifact:
        limit = (
            self.max_artifact_bytes
            if max_bytes is None
            else min(self.max_artifact_bytes, max_bytes)
        )
        if limit < 1:
            raise ValueError("artifact byte limit must be positive")
        if len(payload) > limit:
            raise ValueError("research artifact exceeds configured size limit")
        target = self.resolve(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(target)
        return ResearchArtifact(
            name=target.name,
            relative_path=normalize_relative_path(relative),
            media_type=media_type,
            size=len(payload),
            sha256=sha256_bytes(payload),
        )

    def write_json(
        self,
        relative: str,
        payload: Any,
        *,
        max_bytes: int | None = None,
    ) -> ResearchArtifact:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode()
        return self.write_bytes(
            relative,
            encoded,
            media_type="application/json",
            max_bytes=max_bytes,
        )

    def read_json(self, relative: str) -> Any:
        return json.loads(self.resolve(relative).read_text(encoding="utf-8"))

    def describe_file(self, relative: str, *, media_type: str) -> ResearchArtifact:
        """Describe an existing workspace file without re-reading or hashing it."""
        target = self.resolve(relative)
        if not target.is_file():
            raise FileNotFoundError(relative)
        return ResearchArtifact(
            name=target.name,
            relative_path=normalize_relative_path(relative),
            media_type=media_type,
            size=target.stat().st_size,
            sha256="",
        )

    def list_artifacts(self) -> tuple[str, ...]:
        return tuple(
            path.relative_to(self.root).as_posix()
            for path in sorted(self.root.rglob("*"))
            if path.is_file()
        )
