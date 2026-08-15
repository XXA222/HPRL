from __future__ import annotations

import ast
import sys
from collections import deque
from pathlib import Path

RUNTIME_PACKAGES = ("exchange", "readonly")
FORBIDDEN_IMPORT_PREFIXES = (
    "sqlalchemy",
    "freqtrade.persistence",
)
FORBIDDEN_METHOD_NAMES = {
    "cancel_order",
    "change_leverage",
    "change_margin_type",
    "create_order",
    "edit_order",
    "set_leverage",
    "set_margin_mode",
    "set_position_mode",
}
FORBIDDEN_RUNTIME_PATHS = {
    "/fapi/v1/batchOrders",
    "/fapi/v1/leverage",
    "/fapi/v1/marginType",
    "/fapi/v1/order",
    "/fapi/v1/positionMargin",
}
ALLOWED_NON_GET_PATHS = {"/fapi/v1/listenKey"}
WRITE_HTTP_METHODS = {"DELETE", "PATCH", "POST", "PUT"}


def _module_name(hedge_root: Path, path: Path) -> str:
    relative = path.relative_to(hedge_root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    suffix = ".".join(parts)
    return "freqtrade.hedge" if not suffix else f"freqtrade.hedge.{suffix}"


def _module_index(hedge_root: Path) -> dict[str, Path]:
    return {
        _module_name(hedge_root, path): path
        for path in hedge_root.rglob("*.py")
        if "__pycache__" not in path.parts
    }


def _resolve_from_import(
    current_module: str,
    node: ast.ImportFrom,
    *,
    is_package: bool,
) -> str:
    if node.level == 0:
        return node.module or ""
    package_parts = current_module.split(".")
    if not is_package:
        package_parts = package_parts[:-1]
    keep = max(0, len(package_parts) - node.level + 1)
    base = package_parts[:keep]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _is_forbidden_import(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.")
        for prefix in FORBIDDEN_IMPORT_PREFIXES
    )


def _imported_modules(
    current_module: str,
    node: ast.Import | ast.ImportFrom,
    *,
    is_package: bool,
) -> tuple[str, ...]:
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)
    base = _resolve_from_import(current_module, node, is_package=is_package)
    return (base,) if base else ()


def _local_import_targets(
    current_module: str,
    node: ast.Import | ast.ImportFrom,
    modules: dict[str, Path],
    *,
    is_package: bool,
) -> tuple[Path, ...]:
    candidates = list(
        _imported_modules(current_module, node, is_package=is_package)
    )
    if isinstance(node, ast.ImportFrom):
        base = _resolve_from_import(
            current_module, node, is_package=is_package
        )
        candidates.extend(
            f"{base}.{alias.name}"
            for alias in node.names
            if base and alias.name != "*"
        )
    return tuple(modules[name] for name in candidates if name in modules)


def _runtime_entrypoints(hedge_root: Path) -> tuple[Path, ...]:
    return tuple(
        path
        for package in RUNTIME_PACKAGES
        for path in sorted((hedge_root / package).rglob("*.py"))
        if "__pycache__" not in path.parts
    )


def _forbidden_import_findings(
    path: Path,
    current_module: str,
    node: ast.Import | ast.ImportFrom,
    *,
    is_package: bool,
) -> list[str]:
    return [
        f"Forbidden persistence import {module!r} in {path}:{node.lineno}"
        for module in _imported_modules(
            current_module, node, is_package=is_package
        )
        if _is_forbidden_import(module)
    ]


def _dynamic_import_name(node: ast.Call) -> str | None:
    builtins_import = (
        isinstance(node.func, ast.Name) and node.func.id == "__import__"
    )
    importlib_import = (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "importlib"
        and node.func.attr == "import_module"
    )
    if not builtins_import and not importlib_import:
        return None
    if not node.args or not isinstance(node.args[0], ast.Constant):
        return None
    return str(node.args[0].value)


def _request_call_entry(node: ast.Call) -> tuple[str, str] | None:
    if (
        not isinstance(node.func, ast.Attribute)
        or node.func.attr not in {"request", "_request_once"}
    ):
        return None
    if len(node.args) < 2:
        return None
    method_node, path_node = node.args[:2]
    if not isinstance(method_node, ast.Constant) or not isinstance(path_node, ast.Constant):
        return None
    method = method_node.value
    path = path_node.value
    if not isinstance(method, str) or not isinstance(path, str):
        return None
    normalized_method = method.upper()
    if normalized_method not in WRITE_HTTP_METHODS or path not in FORBIDDEN_RUNTIME_PATHS:
        return None
    return normalized_method, path


def _call_findings(path: Path, node: ast.Call) -> list[str]:
    findings: list[str] = []
    module = _dynamic_import_name(node)
    if module is not None and _is_forbidden_import(module):
        findings.append(f"Forbidden dynamic import {module!r} in {path}:{node.lineno}")
    write_entry = _request_call_entry(node)
    if write_entry is not None:
        findings.append(
            f"Forbidden exchange write request {write_entry!r} in {path}:{node.lineno}"
        )
    return findings


def _definition_findings(
    path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    if node.name not in FORBIDDEN_METHOD_NAMES:
        return []
    return [f"Forbidden exchange write method {node.name} in {path}:{node.lineno}"]


def _http_allowlist_entry(node: ast.Tuple) -> tuple[str, str] | None:
    if len(node.elts) != 2:
        return None
    method_node, path_node = node.elts
    if not isinstance(method_node, ast.Constant) or not isinstance(path_node, ast.Constant):
        return None
    method = method_node.value
    path = path_node.value
    if not isinstance(method, str) or not isinstance(path, str):
        return None
    if method not in WRITE_HTTP_METHODS or path in ALLOWED_NON_GET_PATHS:
        return None
    return method, path


def _tuple_findings(path: Path, node: ast.Tuple) -> list[str]:
    entry = _http_allowlist_entry(node)
    if entry is None:
        return []
    return [f"Non-read HTTP allowlist entry {entry!r} in {path}:{node.lineno}"]


def _node_findings(
    path: Path,
    current_module: str,
    node: ast.AST,
    *,
    is_package: bool,
) -> list[str]:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return _forbidden_import_findings(
            path, current_module, node, is_package=is_package
        )
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return _definition_findings(path, node)
    if isinstance(node, ast.Call):
        return _call_findings(path, node)
    if isinstance(node, ast.Tuple):
        return _tuple_findings(path, node)
    return []


def _inspect_module(
    path: Path,
    current_module: str,
    modules: dict[str, Path],
) -> tuple[list[str], tuple[Path, ...]]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"Syntax error in {path}: {exc}"], ()
    findings: list[str] = []
    targets: list[Path] = []
    is_package = path.name == "__init__.py"
    for node in ast.walk(tree):
        findings.extend(
            _node_findings(
                path, current_module, node, is_package=is_package
            )
        )
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            targets.extend(
                _local_import_targets(
                    current_module,
                    node,
                    modules,
                    is_package=is_package,
                )
            )
    return findings, tuple(targets)


def verify(hedge_root: Path) -> list[str]:
    modules = _module_index(hedge_root)
    path_to_module = {path: module for module, path in modules.items()}
    queue = deque(_runtime_entrypoints(hedge_root))
    visited: set[Path] = set()
    findings: list[str] = []
    while queue:
        path = queue.popleft()
        if path in visited:
            continue
        visited.add(path)
        module_findings, targets = _inspect_module(
            path,
            path_to_module[path],
            modules,
        )
        findings.extend(module_findings)
        queue.extend(targets)
    return findings


def main() -> int:
    hedge_root = Path(
        sys.argv[1] if len(sys.argv) > 1 else "overlay/freqtrade/hedge"
    ).resolve()
    findings = verify(hedge_root)
    if findings:
        print("READONLY SURFACE: FAIL")
        for finding in findings:
            print(f"- {finding}")
        return 1
    print(f"READONLY SURFACE: PASS ({hedge_root})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
