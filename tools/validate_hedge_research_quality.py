"""Dependency-light source-quality gate for the clean-mainline research control plane."""

from __future__ import annotations

import argparse
import ast
import importlib
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_rounds = importlib.import_module("freqtrade.hedge.research.validation_matrix")
ROUND_SPECS = _rounds.ROUND_SPECS
validate_registry = _rounds.validate_registry


@dataclass(frozen=True, slots=True)
class Issue:
    code: str
    path: str
    line: int
    message: str


class _Complexity(ast.NodeVisitor):
    def __init__(self) -> None:
        self.score = 1

    def visit_If(self, node: ast.If) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.score += 1
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_While(self, node: ast.While) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:
        self.score += len(node.handlers)
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp) -> None:
        self.score += max(0, len(node.values) - 1)
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.score += 1
        self.generic_visit(node)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.score += 1 + len(node.ifs)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.generic_visit(node)


def _python_files(root: Path) -> tuple[Path, ...]:
    paths = list((root / "freqtrade/hedge/research").rglob("*.py"))
    paths.extend(
        [
            root / "freqtrade/rpc/api_server/hedge_plugin.py",
            root / "freqtrade/rpc/api_server/hedge_research.py",
            root / "freqtrade/rpc/api_server/hedge_research_schemas.py",
            root / "freqtrade/commands/hedge_research_commands.py",
            root / "tools/run_hedge_research_validation.py",
            root / "tools/validate_hedge_research_quality.py",
        ]
    )
    paths.extend((root / "tests/hedge/research").rglob("*.py"))
    return tuple(sorted(set(paths)))


def _source_files(root: Path) -> tuple[Path, ...]:
    python = list(_python_files(root))
    python.extend((root / "freqtrade/rpc/api_server/hedge_research_ui").rglob("*"))
    python.extend(
        [
            root / "freqtrade/rpc/api_server/hedge_plugin.py",
            root / "freqtrade/rpc/api_server/hedge_ui/index.html",
            root / "config_examples/config_hedge_research.example.json",
        ]
    )
    return tuple(path for path in sorted(set(python)) if path.is_file())


def _import_section(node: ast.Import | ast.ImportFrom) -> int:
    if isinstance(node, ast.ImportFrom) and node.module == "__future__":
        return 0
    if isinstance(node, ast.ImportFrom) and node.level:
        return 4
    if isinstance(node, ast.Import):
        module = node.names[0].name
    else:
        module = node.module or ""
    root = module.split(".", 1)[0]
    if root in sys.stdlib_module_names:
        return 1
    if root == "freqtrade":
        return 3
    return 2


def _import_name_key(name: str) -> tuple[int, str]:
    if name.isupper():
        group = 0
    elif name[:1].isupper():
        group = 1
    else:
        group = 2
    return group, name.casefold()


def _section_nodes(
    imports: list[ast.Import | ast.ImportFrom],
    sections: list[int],
    section: int,
) -> list[ast.Import | ast.ImportFrom]:
    return [node for node, value in zip(imports, sections, strict=True) if value == section]


def _style_order_issue(relative: str, nodes: list[ast.Import | ast.ImportFrom]) -> Issue | None:
    style_keys = [0 if isinstance(node, ast.Import) else 1 for node in nodes]
    if style_keys == sorted(style_keys):
        return None
    return Issue("I001-PROXY", relative, nodes[0].lineno, "import styles unsorted")


def _module_order_issues(
    relative: str,
    nodes: list[ast.Import | ast.ImportFrom],
) -> list[Issue]:
    issues: list[Issue] = []
    for style in (ast.Import, ast.ImportFrom):
        styled = [node for node in nodes if isinstance(node, style)]
        modules = [
            (node.names[0].name if isinstance(node, ast.Import) else (node.module or "")).casefold()
            for node in styled
        ]
        if modules != sorted(modules):
            issues.append(
                Issue("I001-PROXY", relative, styled[0].lineno, "import modules unsorted")
            )
    return issues


def _member_order_issues(
    relative: str,
    nodes: list[ast.Import | ast.ImportFrom],
) -> list[Issue]:
    issues: list[Issue] = []
    for node in nodes:
        if not isinstance(node, ast.ImportFrom) or len(node.names) < 2:
            continue
        names = [alias.name for alias in node.names]
        if names != sorted(names, key=_import_name_key):
            issues.append(
                Issue("I001-PROXY", relative, node.lineno, "from-import members unsorted")
            )
    return issues


def _import_order_checks(relative: str, tree: ast.Module) -> list[Issue]:
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    if not imports:
        return []
    sections = [_import_section(node) for node in imports]
    if sections != sorted(sections):
        return [Issue("I001-PROXY", relative, imports[0].lineno, "import sections unsorted")]
    issues: list[Issue] = []
    for section in sorted(set(sections)):
        if section == 0:
            continue
        nodes = _section_nodes(imports, sections, section)
        style_issue = _style_order_issue(relative, nodes)
        if style_issue is not None:
            issues.append(style_issue)
            continue
        issues.extend(_module_order_issues(relative, nodes))
        issues.extend(_member_order_issues(relative, nodes))
    return issues


def _module_level_import_checks(relative: str, tree: ast.Module) -> list[Issue]:
    issues: list[Issue] = []
    seen_runtime_statement = False
    for index, node in enumerate(tree.body):
        is_docstring = (
            index == 0
            and isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
        if is_docstring:
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if seen_runtime_statement:
                issues.append(
                    Issue("E402-PROXY", relative, node.lineno, "module import follows runtime code")
                )
            continue
        seen_runtime_statement = True
    return issues


def _compile_and_line_checks(root: Path, files: Iterable[Path]) -> list[Issue]:
    issues: list[Issue] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=relative)
        except SyntaxError as exc:
            issues.append(Issue("PY-COMPILE", relative, exc.lineno or 0, str(exc)))
            continue
        for line_no, line in enumerate(source.splitlines(), start=1):
            if len(line) > 100:
                issues.append(Issue("E501-PROXY", relative, line_no, "line exceeds 100 chars"))
        issues.extend(_module_level_import_checks(relative, tree))
        issues.extend(_import_order_checks(relative, tree))
        issues.extend(_unused_imports(relative, tree))
        issues.extend(_complexity(relative, tree))
    return issues


def _import_usage(tree: ast.AST) -> tuple[dict[str, int], set[str]]:
    imports: dict[str, int] = {}
    loaded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports[alias.asname or alias.name.split(".")[0]] = node.lineno
            continue
        if isinstance(node, ast.ImportFrom) and node.module != "__future__":
            for alias in node.names:
                if alias.name != "*":
                    imports[alias.asname or alias.name] = node.lineno
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loaded.add(node.id)
    return imports, loaded


def _unused_imports(relative: str, tree: ast.AST) -> list[Issue]:
    if relative.endswith("/__init__.py"):
        return []
    imports, loaded = _import_usage(tree)
    return [
        Issue("F401-PROXY", relative, line, f"unused import: {name}")
        for name, line in imports.items()
        if name not in loaded
    ]


def _complexity(relative: str, tree: ast.AST) -> list[Issue]:
    issues: list[Issue] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        visitor = _Complexity()
        for statement in node.body:
            visitor.visit(statement)
        if visitor.score > 12:
            issues.append(
                Issue(
                    "C901-PROXY",
                    relative,
                    node.lineno,
                    f"{node.name} complexity {visitor.score} > 12",
                )
            )
    return issues


def _security_checks(root: Path) -> list[Issue]:
    issues: list[Issue] = []
    forbidden = {
        "create_order(": "exchange write create_order is forbidden",
        "cancel_order(": "exchange write cancel_order is forbidden",
        "edit_order(": "exchange write edit_order is forbidden",
        "shell=True": "shell execution is forbidden",
        "requests.post(": "direct external HTTP writes are forbidden",
        "ccxt.": "direct exchange client use is forbidden",
    }
    production = list((root / "freqtrade/hedge/research").rglob("*.py"))
    production.extend(
        [
            root / "freqtrade/rpc/api_server/hedge_plugin.py",
            root / "freqtrade/rpc/api_server/hedge_research.py",
            root / "freqtrade/rpc/api_server/hedge_research_schemas.py",
            root / "freqtrade/commands/hedge_research_commands.py",
        ]
    )
    for path in production:
        relative = path.relative_to(root).as_posix()
        text = path.read_text(encoding="utf-8")
        for token, message in forbidden.items():
            line = text.find(token)
            if line >= 0:
                line_no = text[:line].count("\n") + 1
                issues.append(Issue("SAFETY", relative, line_no, message))
    plugin_path = root / "freqtrade/rpc/api_server/hedge_plugin.py"
    plugin = plugin_path.read_text(encoding="utf-8")
    if "Hedge research dashboard requires a loopback API listen address" not in plugin:
        issues.append(
            Issue(
                "LOCAL-ONLY",
                "freqtrade/rpc/api_server/hedge_plugin.py",
                0,
                "loopback guard missing",
            )
        )
    api_path = root / "freqtrade/rpc/api_server/hedge_research.py"
    api = api_path.read_text(encoding="utf-8")
    if "exchange" in api.lower() and "write" in api.lower():
        issues.append(
            Issue(
                "API-SURFACE",
                "freqtrade/rpc/api_server/hedge_research.py",
                0,
                "review exchange write wording",
            )
        )
    return issues


def _ui_checks(root: Path) -> list[Issue]:
    issues: list[Issue] = []
    html_path = root / "freqtrade/rpc/api_server/hedge_research_ui/index.html"
    js_path = root / "freqtrade/rpc/api_server/hedge_research_ui/app.js"
    api_path = root / "freqtrade/rpc/api_server/hedge_research.py"
    html = html_path.read_text(encoding="utf-8")
    js = js_path.read_text(encoding="utf-8")
    api = api_path.read_text(encoding="utf-8")
    if "<script>" in html.lower():
        issues.append(
            Issue("CSP-INLINE", "hedge_research_ui/index.html", 0, "inline script forbidden")
        )
    if 'fetch("/api/v1"' not in js:
        issues.append(
            Issue("UI-API", "hedge_research_ui/app.js", 0, "same-origin API prefix missing")
        )
    if "frame-ancestors 'none'" not in api or "object-src 'none'" not in api:
        issues.append(Issue("CSP", "hedge_research.py", 0, "strict CSP directives missing"))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=_PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.project_root.resolve()
    validate_registry()
    python_files = _python_files(root)
    issues = _compile_and_line_checks(root, python_files)
    issues.extend(_security_checks(root))
    issues.extend(_ui_checks(root))
    payload = {
        "schema": "freqtrade-hedge-research-quality-v1",
        "status": "PASS" if not issues else "FAIL",
        "round_count": len(ROUND_SPECS),
        "python_files_scanned": len(python_files),
        "source_files_scanned": len(_source_files(root)),
        "issue_count": len(issues),
        "issues": [asdict(issue) for issue in issues],
    }
    output = args.output or root / "research-quality-result.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
