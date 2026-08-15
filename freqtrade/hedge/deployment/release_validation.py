"""Release-package and local-overlay validation used by R80 delivery tooling."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile
from zipfile import ZipInfo


@dataclass(frozen=True, slots=True)
class OverlayAction:
    relative_path: str
    action: str


def safe_zip_member(name: str) -> bool:
    path = PurePosixPath(name.replace("\\", "/"))
    return (
        bool(name)
        and not path.is_absolute()
        and ".." not in path.parts
        and not (path.parts and ":" in path.parts[0])
    )


def validate_zip_members(infos: Sequence[ZipInfo]) -> tuple[str, ...]:
    names: set[str] = set()
    validated: list[str] = []
    for info in infos:
        normalized = info.filename.replace("\\", "/")
        if not safe_zip_member(normalized):
            raise ValueError(f"unsafe ZIP member: {info.filename}")
        key = normalized.rstrip("/").casefold()
        if key in names:
            raise ValueError(f"duplicate ZIP member: {info.filename}")
        names.add(key)
        validated.append(normalized)
    return tuple(validated)



def _safe_path(root: Path, relative: str) -> Path:
    if not safe_zip_member(relative):
        raise ValueError(f"unsafe payload path: {relative}")
    resolved_root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"payload path escapes root: {relative}")
    return candidate

def payload_manifest(root: Path, relative_paths: Iterable[str]) -> dict[str, str]:
    output: dict[str, str] = {}
    for relative in sorted(set(relative_paths)):
        path = _safe_path(root, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        output[relative] = sha256(path.read_bytes()).hexdigest()
    return output


def verify_manifest(root: Path, manifest: Mapping[str, str]) -> tuple[str, ...]:
    failures: list[str] = []
    for relative, expected in sorted(manifest.items()):
        path = _safe_path(root, relative)
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif sha256(path.read_bytes()).hexdigest().lower() != str(expected).lower():
            failures.append(f"hash:{relative}")
    return tuple(failures)


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def backup_plan(target_root: Path, relative_paths: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        relative
        for relative in sorted(set(relative_paths))
        if _safe_path(target_root, relative).is_file()
    )


def overlay_plan(
    source_root: Path,
    target_root: Path,
    relative_paths: Iterable[str],
) -> tuple[OverlayAction, ...]:
    output: list[OverlayAction] = []
    for relative in sorted(set(relative_paths)):
        source = _safe_path(source_root, relative)
        target = _safe_path(target_root, relative)
        if not source.is_file():
            raise FileNotFoundError(source)
        if not target.exists():
            action = "create"
        elif (
            target.is_file()
            and sha256(source.read_bytes()).digest() == sha256(target.read_bytes()).digest()
        ):
            action = "unchanged"
        else:
            action = "replace"
        output.append(OverlayAction(relative, action))
    return tuple(output)


def rollback_plan(actions: Sequence[OverlayAction]) -> tuple[OverlayAction, ...]:
    output = []
    for action in reversed(actions):
        if action.action == "create":
            output.append(OverlayAction(action.relative_path, "delete"))
        elif action.action == "replace":
            output.append(OverlayAction(action.relative_path, "restore"))
    return tuple(output)


def validation_level_plan(level: str, *, full_dependencies_available: bool) -> tuple[str, ...]:
    normalized = level.strip().lower()
    core = ("zip", "manifest", "compile", "r80-tests", "existing-optimization-tests")
    full = core + ("existing-parity-tests", "cli-smoke", "integration-import")
    if normalized == "core":
        return core
    if normalized == "full":
        if not full_dependencies_available:
            raise ValueError("full validation dependencies are unavailable")
        return full
    if normalized == "auto":
        return full if full_dependencies_available else core
    raise ValueError("validation level must be core, auto, or full")


def install_report(
    *,
    version: str,
    actions: Sequence[OverlayAction],
    gates: Sequence[str],
    success: bool,
) -> dict[str, object]:
    if not version.strip() or not gates:
        raise ValueError("version and validation gates are required")
    counts = {
        name: sum(action.action == name for action in actions)
        for name in ("create", "replace", "unchanged")
    }
    report = {"version": version, "success": success, "counts": counts, "gates": list(gates)}
    json.dumps(report, sort_keys=True)
    return report
