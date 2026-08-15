from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


TOOLS = Path(__file__).resolve().parents[2] / "tools"


def _load_validator():
    sys.path.insert(0, str(TOOLS))
    try:
        spec = importlib.util.spec_from_file_location(
            "_clean_mainline_200_workspace_test",
            TOOLS / "validate_clean_mainline_200.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(TOOLS))


def _write_manifest(root: Path) -> None:
    source = root / "main.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    payload = {
        "schema": "test",
        "version": "test",
        "file_count": 1,
        "total_bytes": source.stat().st_size,
        "files": [
            {
                "path": "main.py",
                "size": source.stat().st_size,
                "sha256": "unused-by-path-set-check",
            }
        ],
    }
    (root / "CLEAN-MAINLINE-MANIFEST.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _add_workspace_artifacts(root: Path) -> None:
    (root / ".venv/Lib/site-packages").mkdir(parents=True)
    (root / ".venv/Lib/site-packages/bad.py").write_text(
        "this is intentionally not valid Python !!!",
        encoding="utf-8",
    )
    (root / "artifacts/run").mkdir(parents=True)
    (root / "artifacts/run/state.json").write_text("{broken", encoding="utf-8")
    (root / "user_data/audit/run").mkdir(parents=True)
    (root / "user_data/audit/run/log.txt").write_text("runtime", encoding="utf-8")
    (root / "freqtrade.egg-info").mkdir()
    (root / "freqtrade.egg-info/PKG-INFO").write_text("metadata", encoding="utf-8")
    (root / ".pytest-install").mkdir()
    (root / ".pytest-install/control.json").write_text("{broken", encoding="utf-8")


def test_workspace_mode_ignores_only_canonical_workspace_artifacts(tmp_path: Path) -> None:
    validator = _load_validator()
    _write_manifest(tmp_path)
    _add_workspace_artifacts(tmp_path)

    assert "workspace" in validator._no_generated_payload(tmp_path, True)
    assert validator._manifest_exact(tmp_path, True) == "1 manifest rows"
    assert validator._python_compile(tmp_path, True) == "1 Python files compiled"


def test_package_mode_remains_strict_for_workspace_artifacts(tmp_path: Path) -> None:
    validator = _load_validator()
    _write_manifest(tmp_path)
    _add_workspace_artifacts(tmp_path)

    with pytest.raises((AssertionError, SyntaxError)):
        validator._no_generated_payload(tmp_path, False)
    with pytest.raises(AssertionError):
        validator._manifest_exact(tmp_path, False)
    with pytest.raises(SyntaxError):
        validator._python_compile(tmp_path, False)
