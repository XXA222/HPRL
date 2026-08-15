#!/usr/bin/env python3
"""Validate the Freqtrade-Hedge clean-mainline source and installed workspace."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
import tomllib
from typing import Any

SCHEMA = "freqtrade-hedge-clean-mainline-validation-v1"
MANIFEST_NAME = "CLEAN-MAINLINE-MANIFEST.json"
VERSION_NAME = "CLEAN-MAINLINE-VERSION.txt"

SOURCE_TOP_LEVEL_DIRS = {
    ".devcontainer",
    ".github",
    ".vscode",
    "build_helpers",
    "config_examples",
    "docker",
    "docs",
    "freqtrade",
    "ft_client",
    "scripts",
    "tests",
    "tools",
    "user_data",
}
WORKSPACE_IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "artifacts",
    "dist",
    "site",
    "venv",
}
GENERATED_DIR_PATTERNS = (
    re.compile(r"^\.pytest[-_].*"),
    re.compile(r"^\.merge[-_].*"),
    re.compile(r"^.*\.egg-info$"),
    re.compile(r"^__pycache__$"),
)
FORBIDDEN_MAINLINE_DIRS = {
    "merge_history",
    "release",
    "hedge_port",
    "project_docs",
    "verification",
}
FORBIDDEN_ROOT_PREFIXES = (
    "MERGE",
    "UNIFIED",
    "REMEDIATION",
    "SOURCE-PACKAGE",
    "SOURCE-FILE",
    "R4",
    "R5",
    "R6",
    "README-R",
    "Install-Freqtrade-Hedge-R",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "freqtrade.hedge.r4",
    "freqtrade.hedge.r5",
    "freqtrade.hedge.r54",
    "freqtrade.hedge.r55",
    "freqtrade.hedge.r56",
    "freqtrade.hedge.r561",
    "freqtrade.hedge.r58",
    "freqtrade.hedge.binance_readonly",
    "freqtrade.hedge.binance_user_stream",
    "freqtrade.hedge.target_position",
    "freqtrade.hedge.user_stream_lifecycle",
    "freqtrade.hedge.persistence",
    "freqtrade.hedge.faults",
    "integrate_h3_full",
)
VERSIONED_PATH_RE = re.compile(r"(^|/)(r\d+|p\d+_h\d+)(/|$)", re.IGNORECASE)
VERSIONED_TEST_FILE_RE = re.compile(r"(^|/).*_(?:r\d+|v\d+)(?:_|\.)", re.IGNORECASE)
ALLOWED_LEGACY_CONFIG_ALIAS_FILES = {
    "freqtrade/hedge/config_migration.py",
    "tests/hedge/test_clean_mainline_config_isolation.py",
    "tests/hedge/operations/test_config.py",
    "tools/validate_clean_mainline.py",
    "tools/validate_clean_mainline_200.py",
}
CANONICAL_FILES = (
    "freqtrade/hedge/strategies/contract.py",
    "freqtrade/hedge/control/dryrun.py",
    "freqtrade/hedge/telemetry/dryrun.py",
    "freqtrade/hedge/operations/runtime.py",
    "freqtrade/hedge/acceptance/acceptance.py",
    "freqtrade/commands/hedge_acceptance_commands.py",
    "freqtrade/hedge/exchange/binance_readonly.py",
    "freqtrade/hedge/exchange/binance_user_stream.py",
    "freqtrade/hedge/execution/service.py",
    "freqtrade/hedge/planning/ideal_orders.py",
    "freqtrade/hedge/risk/engine.py",
    "freqtrade/hedge/integration/paper_runtime.py",
    "freqtrade/hedge/integration/central_source.py",
    "freqtrade/hedge/config_migration.py",
    "freqtrade/hedge/research/validation_matrix.py",
    "freqtrade/freqai/hedge_rl/environment.py",
    "scripts/Configure-Freqtrade-Hedge-LocalSource.ps1",
    "tools/migrate_clean_mainline_config.py",
    "tools/validate_clean_mainline_200.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def is_generated_dir(name: str) -> bool:
    return any(pattern.match(name) for pattern in GENERATED_DIR_PATTERNS)


def should_ignore_workspace_path(root: Path, path: Path) -> bool:
    relative = path.relative_to(root)
    if not relative.parts:
        return False
    if relative.parts[0] == "user_data":
        return True
    return any(part in WORKSPACE_IGNORED_DIRS or is_generated_dir(part) for part in relative.parts)


def source_files(root: Path, workspace_mode: bool) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if workspace_mode and should_ignore_workspace_path(root, path):
            continue
        if any(part in {"__pycache__"} or is_generated_dir(part) for part in relative.parts):
            continue
        files.append(path)
    return sorted(files)


def manifest_files(root: Path, workspace_mode: bool) -> list[Path]:
    result = []
    for path in source_files(root, workspace_mode):
        relative = rel(root, path)
        if relative == MANIFEST_NAME or relative.startswith("user_data/"):
            continue
        result.append(path)
    return result


def gate(name: str, passed: bool, detail: Any) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "detail": detail}


def check_layout(root: Path, workspace_mode: bool) -> dict[str, Any]:
    findings: list[str] = []
    for directory in FORBIDDEN_MAINLINE_DIRS:
        if (root / directory).exists():
            findings.append(f"forbidden mainline directory: {directory}")
    for path in root.iterdir():
        if path.is_file() and path.name.startswith(FORBIDDEN_ROOT_PREFIXES):
            findings.append(f"historical root file: {path.name}")
    for path in root.rglob("*"):
        if workspace_mode and should_ignore_workspace_path(root, path):
            continue
        relative = rel(root, path)
        if VERSIONED_PATH_RE.search(relative):
            findings.append(f"versioned path: {relative}")
        if path.is_file() and relative.startswith("tests/hedge/") and VERSIONED_TEST_FILE_RE.search(relative):
            findings.append(f"versioned Hedge test filename: {relative}")
    for item in CANONICAL_FILES:
        if not (root / item).is_file():
            findings.append(f"missing canonical file: {item}")
    return gate("clean-mainline-layout", not findings, findings[:200])


def imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imports.append((node.lineno, node.module))
    return imports


def check_python(root: Path, workspace_mode: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    syntax: list[str] = []
    forbidden: list[str] = []
    count = 0
    for path in source_files(root, workspace_mode):
        if path.suffix != ".py":
            continue
        relative = rel(root, path)
        if relative.startswith("user_data/"):
            continue
        count += 1
        try:
            source = path.read_text(encoding="utf-8-sig")
            tree = ast.parse(source, filename=str(path))
            compile(source, str(path), "exec")
        except Exception as exc:
            syntax.append(f"{relative}: {type(exc).__name__}: {exc}")
            continue
        for lineno, module in imported_modules(tree):
            if any(module == prefix or module.startswith(prefix + ".") for prefix in FORBIDDEN_IMPORT_PREFIXES):
                forbidden.append(f"{relative}:{lineno}: {module}")
        if relative not in ALLOWED_LEGACY_CONFIG_ALIAS_FILES and re.search(r"[\"']r56[\"']", source):
            forbidden.append(f"{relative}: legacy r56 config token outside migration boundary")
    return (
        gate("python-parse-compile", not syntax, {"files": count, "failures": syntax[:100]}),
        gate("no-versioned-or-removed-imports", not forbidden, forbidden[:200]),
    )



def parse_jsonc(text: str) -> Any:
    out: list[str] = []
    index = 0
    in_string = False
    escape = False
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            out.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            out.append(char)
            index += 1
            continue
        if char == "/" and next_char == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                index += 1
            index += 2
            continue
        out.append(char)
        index += 1
    cleaned = re.sub(r",\s*([}\]])", r"\1", "".join(out))
    return json.loads(cleaned)

def check_configs(root: Path, workspace_mode: bool) -> dict[str, Any]:
    failures: list[str] = []
    counts = {"json": 0, "toml": 0, "yaml": 0}
    yaml_loader = None
    try:
        import yaml  # type: ignore
        yaml_loader = yaml.compose
    except Exception:
        pass
    for path in source_files(root, workspace_mode):
        relative = rel(root, path)
        if relative.startswith("user_data/"):
            continue
        suffix = path.suffix.lower()
        try:
            if suffix in {".json", ".jsonc"}:
                counts["json"] += 1
                text = path.read_text(encoding="utf-8-sig")
                parse_jsonc(text)
            elif suffix == ".toml":
                counts["toml"] += 1
                tomllib.loads(path.read_text(encoding="utf-8-sig"))
            elif suffix in {".yaml", ".yml"}:
                counts["yaml"] += 1
                if yaml_loader is not None:
                    yaml_loader(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            failures.append(f"{relative}: {type(exc).__name__}: {exc}")
    return gate("config-syntax", not failures, {"counts": counts, "failures": failures[:100]})



def check_config_single_authority(root: Path) -> dict[str, Any]:
    findings: list[str] = []
    extension_path = root / "freqtrade/hedge/config_schema_extension.py"
    migration_path = root / "freqtrade/hedge/config_migration.py"
    runtime_path = root / "freqtrade/hedge/operations/config.py"
    main_config_path = root / "freqtrade/hedge/config.py"

    if not migration_path.is_file():
        findings.append("missing one-way config migration module")
    if not extension_path.is_file():
        findings.append("missing Hedge schema extension")
    else:
        spec = importlib.util.spec_from_file_location(
            "_clean_mainline_schema_extension",
            extension_path,
        )
        if spec is None or spec.loader is None:
            findings.append("unable to load Hedge schema extension")
        else:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            schema: dict[str, Any] = {"properties": {}}
            module.extend_config_schema(schema)
            hedge = schema.get("properties", {}).get("hedge", {})
            properties = hedge.get("properties", {}) if isinstance(hedge, dict) else {}
            if "operations" not in properties:
                findings.append("canonical hedge.operations schema is missing")
            else:
                operations = properties["operations"]
                if isinstance(operations, dict) and "default" in operations:
                    findings.append("hedge.operations container must not be auto-materialized")
            if "r56" in properties:
                findings.append("retired operations alias leaked into the current JSON schema")

    for path, label in (
        (main_config_path, "main Hedge config"),
        (runtime_path, "operations runtime config"),
    ):
        if path.is_file() and re.search(r"[\"']r56[\"']", path.read_text(encoding="utf-8-sig")):
            findings.append(f"retired operations key leaked into {label}")

    return gate("single-authority-operations-config", not findings, findings)

def check_pyproject(root: Path) -> dict[str, Any]:
    findings: list[str] = []
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload.get("project", {}).get("scripts", {})
    if scripts != {"freqtrade": "freqtrade.main:main"}:
        findings.append(f"unexpected console scripts: {scripts}")
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    if "freqtrade-hedge-r" in text or "verification/r58" in text:
        findings.append("historical entry point or verifier ignore remains in pyproject")
    return gate("pyproject-current-entrypoints", not findings, findings)


def check_package_hygiene(root: Path, workspace_mode: bool) -> dict[str, Any]:
    if workspace_mode:
        return gate("package-hygiene", True, "workspace mode ignores local environment/runtime state")
    findings: list[str] = []
    for path in root.rglob("*"):
        relative = rel(root, path)
        parts = path.relative_to(root).parts
        if any(part in WORKSPACE_IGNORED_DIRS or is_generated_dir(part) for part in parts):
            findings.append(relative)
        if path.is_file() and (path.suffix == ".pyc" or path.name == "network_diag.txt"):
            findings.append(relative)
        if relative.startswith("user_data/") and path.is_file() and path.name != ".gitkeep":
            findings.append(relative)
    return gate("package-hygiene", not findings, sorted(set(findings))[:200])


def check_manifest(root: Path, workspace_mode: bool) -> dict[str, Any]:
    path = root / MANIFEST_NAME
    if not path.is_file():
        return gate("manifest-exhaustive-sha256", False, "manifest missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("files", [])
    by_path = {row.get("path"): row for row in rows if isinstance(row, dict)}
    expected = {rel(root, item): item for item in manifest_files(root, workspace_mode)}
    missing = sorted(set(expected) - set(by_path))
    extra = sorted(set(by_path) - set(expected))
    bad: list[str] = []
    for relative, file_path in expected.items():
        row = by_path.get(relative)
        if not row:
            continue
        if row.get("size") != file_path.stat().st_size or row.get("sha256") != sha256_file(file_path):
            bad.append(relative)
    passed = not missing and not extra and not bad
    return gate(
        "manifest-exhaustive-sha256",
        passed,
        {"expected": len(expected), "rows": len(by_path), "missing": missing[:50], "extra": extra[:50], "bad": bad[:50]},
    )


def validate(root: Path, workspace_mode: bool) -> dict[str, Any]:
    root = root.resolve()
    python_gate, import_gate = check_python(root, workspace_mode)
    gates = [
        check_layout(root, workspace_mode),
        check_package_hygiene(root, workspace_mode),
        python_gate,
        import_gate,
        check_configs(root, workspace_mode),
        check_config_single_authority(root),
        check_pyproject(root),
        check_manifest(root, workspace_mode),
    ]
    failures = [item for item in gates if item["status"] != "PASS"]
    return {
        "schema": SCHEMA,
        "project_root": str(root),
        "mode": "workspace" if workspace_mode else "package",
        "status": "PASS" if not failures else "FAIL",
        "blocking_failures": len(failures),
        "gates": gates,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--workspace-mode", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    payload = validate(args.project_root, args.workspace_mode)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
