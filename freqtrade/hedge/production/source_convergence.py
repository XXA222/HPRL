"""Canonical source-snapshot identity for HPRL V3 Production Integration."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CanonicalSourceSnapshot:
    manifest_schema: str
    manifest_version: str
    manifest_file_count: int
    manifest_sha256: str
    tree_sha256: str
    hprl_api_version: str
    hprl_release: str
    production_api_version: str
    production_release: str
    closed_loop_api_version: str
    closed_loop_release: str
    github_baseline_repository: str
    github_baseline_commit: str
    required_paths_present: bool
    manifest_matches_workspace: bool
    missing_paths: tuple[str, ...]
    manifest_missing_files: tuple[str, ...]
    manifest_unexpected_files: tuple[str, ...]
    manifest_mismatched_files: tuple[str, ...]
    validation_policy: str
    managed_attestation_verified: bool
    managed_attestation_path: str
    managed_attestation_count: int
    managed_attestation_overlay_sha256: str
    managed_attestation_target_release: str
    workspace_missing_files: tuple[str, ...]
    workspace_unexpected_files: tuple[str, ...]
    workspace_mismatched_files: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.required_paths_present
            and self.manifest_file_count > 0
            and self.manifest_matches_workspace
            and (self.validation_policy != "live" or self.managed_attestation_verified)
        )


_REQUIRED = (
    "CLEAN-MAINLINE-MANIFEST.json",
    "freqtrade/hedge/hprl/__init__.py",
    "freqtrade/hedge/production/__init__.py",
    "freqtrade/hedge/execution/service.py",
    "freqtrade/persistence/hedge_execution_adapters.py",
    "freqtrade/hedge/simulation/replay.py",
    "freqtrade/hedge/production/hprl_hedge_adapter.py",
    "freqtrade/hedge/production/hprl_replay_backtest.py",
    "freqtrade/hedge/production/recovery_checkpoint.py",
    "freqtrade/hedge/production/postgres_acceptance.py",
    "freqtrade/hedge/production/binance_dryrun.py",
    "freqtrade/hedge/production/acceptance_r2.py",
    "freqtrade/hedge/production/closed_loop.py",
    "freqtrade/hedge/production/closed_loop_recovery.py",
    "freqtrade/hedge/production/closed_loop_dryrun.py",
    "freqtrade/hedge/production/closed_loop_sql.py",
    "freqtrade/hedge/production/acceptance_closed_loop.py",
)

_IGNORED_DIRS = {
    ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv",
    "artifacts", "dist", "site", "venv", "__pycache__",
}

_LIVE_CRITICAL_PREFIXES = (
    "freqtrade/hedge/production/",
    "freqtrade/hedge/execution/",
    "freqtrade/hedge/hprl/",
    "freqtrade/hedge/planning/",
    "freqtrade/hedge/risk/",
    "freqtrade/hedge/integration/",
    "freqtrade/hedge/simulation/",
)

_LIVE_CRITICAL_EXACT = {
    "freqtrade/persistence/hedge_execution_adapters.py",
}

# CLEAN-MAINLINE-VERSION.txt is package-governance metadata.  It is still
# byte-exact under package policy, but it is deliberately not a live blocking
# path: deployed workspaces may retain local historical metadata while all
# executable Hedge/HPRL sources are attested independently.

_ATTESTATION_SCHEMA = "hprl-v3-closed-loop-managed-attestation-v1"


def _is_live_critical(path: str) -> bool:
    return path in _LIVE_CRITICAL_EXACT or any(path.startswith(prefix) for prefix in _LIVE_CRITICAL_PREFIXES)


def _verify_managed_attestation(
    root: Path,
    attestation_path: str,
    *,
    expected_overlay_sha256: str = "",
    expected_target_release: str = "",
) -> tuple[bool, int, str, str]:
    if not attestation_path:
        return False, 0, "", ""
    path = Path(attestation_path)
    if not path.is_file():
        return False, 0, "", ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False, 0, "", ""
    if payload.get("schema") != _ATTESTATION_SCHEMA:
        return False, 0, "", ""
    if str(payload.get("project_root", "")) != str(root):
        return False, 0, "", ""
    overlay_sha256 = str(payload.get("overlay_sha256", "")).strip().lower()
    target_release = str(payload.get("target_release", "")).strip()
    expected_overlay_sha256 = expected_overlay_sha256.strip().lower()
    expected_target_release = expected_target_release.strip()
    if expected_overlay_sha256 and overlay_sha256 != expected_overlay_sha256:
        return False, 0, overlay_sha256, target_release
    if expected_target_release and target_release != expected_target_release:
        return False, 0, overlay_sha256, target_release
    rows = payload.get("files")
    if not isinstance(rows, list) or not rows:
        return False, 0, overlay_sha256, target_release
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            return False, 0, overlay_sha256, target_release
        rel = str(row.get("path", ""))
        expected = str(row.get("sha256", "")).lower()
        if not rel or rel in seen or len(expected) != 64:
            return False, 0, overlay_sha256, target_release
        seen.add(rel)
        target = (root / rel).resolve()
        if root != target and root not in target.parents:
            return False, 0, overlay_sha256, target_release
        if not target.is_file() or _sha_file(target) != expected:
            return False, 0, overlay_sha256, target_release
    declared = payload.get("managed_count")
    if declared != len(rows):
        return False, 0, overlay_sha256, target_release
    return True, len(rows), overlay_sha256, target_release



def _workspace_rows(root: Path) -> dict[str, tuple[int, str]]:
    rows: dict[str, tuple[int, str]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if not relative.parts:
            continue
        if relative.parts[0] == "user_data":
            continue
        if any(part in _IGNORED_DIRS or part.startswith(".pytest-") or part.startswith(".merge-") or part.endswith(".egg-info") for part in relative.parts):
            continue
        rel = relative.as_posix()
        if rel == "CLEAN-MAINLINE-MANIFEST.json":
            continue
        rows[rel] = (path.stat().st_size, _sha_file(path))
    return rows


def _sha_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_constant(path: Path, name: str) -> str:
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id == name:
                    value = node.value
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        return value.value
    raise ValueError(f"{name} not found in {path}")


def build_canonical_source_snapshot(
    root: str | Path,
    *,
    github_baseline_repository: str = "XXA222/HPRL",
    github_baseline_commit: str = "c7411179744a38b3af91a11a91985db2327c77a4",
) -> CanonicalSourceSnapshot:
    root_path = Path(root).resolve()
    missing = tuple(path for path in _REQUIRED if not (root_path / path).is_file())
    manifest_path = root_path / "CLEAN-MAINLINE-MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise ValueError("clean-mainline manifest files must be a list")
    canonical_rows = []
    manifest_map: dict[str, tuple[int, str]] = {}
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("manifest row must be an object")
        row = {
            "path": str(item.get("path", "")),
            "size": int(item.get("size", -1)),
            "sha256": str(item.get("sha256", "")).lower(),
        }
        if not row["path"] or row["path"] in manifest_map:
            raise ValueError("manifest paths must be unique non-empty strings")
        canonical_rows.append(row)
        manifest_map[row["path"]] = (row["size"], row["sha256"])
    canonical_rows.sort(key=lambda item: item["path"])
    tree_raw = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode()
    workspace_map = _workspace_rows(root_path)
    manifest_paths = set(manifest_map)
    workspace_paths = set(workspace_map)
    workspace_missing = tuple(sorted(manifest_paths - workspace_paths))
    workspace_unexpected = tuple(sorted(workspace_paths - manifest_paths))
    workspace_mismatched = tuple(sorted(
        path for path in manifest_paths & workspace_paths
        if manifest_map[path] != workspace_map[path]
    ))
    policy = os.environ.get("HPRL_SOURCE_VALIDATION_POLICY", "package").strip().lower()
    if policy not in {"package", "live"}:
        raise ValueError("HPRL_SOURCE_VALIDATION_POLICY must be package or live")
    attestation_path = os.environ.get("HPRL_MANAGED_ATTESTATION", "").strip()
    expected_overlay_sha256 = os.environ.get("HPRL_EXPECTED_MANAGED_OVERLAY_SHA256", "").strip()
    expected_target_release = os.environ.get("HPRL_EXPECTED_MANAGED_TARGET_RELEASE", "").strip()
    (
        attestation_verified,
        attestation_count,
        attestation_overlay_sha256,
        attestation_target_release,
    ) = _verify_managed_attestation(
        root_path,
        attestation_path,
        expected_overlay_sha256=expected_overlay_sha256,
        expected_target_release=expected_target_release,
    )
    if policy == "live":
        missing_from_workspace = tuple(path for path in workspace_missing if _is_live_critical(path))
        unexpected_in_workspace = tuple(path for path in workspace_unexpected if _is_live_critical(path))
        mismatched = tuple(path for path in workspace_mismatched if _is_live_critical(path))
    else:
        missing_from_workspace = workspace_missing
        unexpected_in_workspace = workspace_unexpected
        mismatched = workspace_mismatched
    manifest_matches = not (missing_from_workspace or unexpected_in_workspace or mismatched)
    hprl_init = root_path / "freqtrade/hedge/hprl/__init__.py"
    prod_init = root_path / "freqtrade/hedge/production/__init__.py"
    return CanonicalSourceSnapshot(
        manifest_schema=str(payload.get("schema", "")),
        manifest_version=str(payload.get("version", "")),
        manifest_file_count=len(rows),
        manifest_sha256=_sha_file(manifest_path),
        tree_sha256=sha256(tree_raw).hexdigest(),
        hprl_api_version=_module_constant(hprl_init, "HPRL_API_VERSION"),
        hprl_release=_module_constant(hprl_init, "HPRL_RELEASE"),
        production_api_version=_module_constant(prod_init, "PRODUCTION_READINESS_API_VERSION"),
        production_release=_module_constant(prod_init, "PRODUCTION_READINESS_RELEASE"),
        closed_loop_api_version=_module_constant(prod_init, "HPRL_V3_CLOSED_LOOP_API_VERSION"),
        closed_loop_release=_module_constant(prod_init, "HPRL_V3_CLOSED_LOOP_RELEASE"),
        github_baseline_repository=github_baseline_repository,
        github_baseline_commit=github_baseline_commit,
        required_paths_present=not missing,
        manifest_matches_workspace=manifest_matches,
        missing_paths=missing,
        manifest_missing_files=missing_from_workspace,
        manifest_unexpected_files=unexpected_in_workspace,
        manifest_mismatched_files=mismatched,
        validation_policy=policy,
        managed_attestation_verified=attestation_verified,
        managed_attestation_path=attestation_path,
        managed_attestation_count=attestation_count,
        managed_attestation_overlay_sha256=attestation_overlay_sha256,
        managed_attestation_target_release=attestation_target_release,
        workspace_missing_files=workspace_missing,
        workspace_unexpected_files=workspace_unexpected,
        workspace_mismatched_files=workspace_mismatched,
    )
