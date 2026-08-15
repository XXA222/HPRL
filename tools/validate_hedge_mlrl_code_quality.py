#!/usr/bin/env python3
"""Dependency-light code-quality proxy for the current Hedge ML/RL mainline."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


RELEASE = "CLEAN-MAINLINE"
MAX_LINE_LENGTH = 100
TYPING_ABCS = {
    "AsyncIterable",
    "AsyncIterator",
    "Callable",
    "Collection",
    "Generator",
    "Iterable",
    "Iterator",
    "Mapping",
    "MutableMapping",
    "MutableSet",
    "Sequence",
    "Set",
}


@dataclass(frozen=True, slots=True)
class Issue:
    path: str
    line: int
    code: str
    message: str


def iter_scope(source: Path) -> list[Path]:
    roots = (
        source / "freqtrade/freqai/hedge_rl",
        source / "freqtrade/freqai/prediction_models/HedgeReinforcementLearner.py",
        source / "freqtrade/freqai/prediction_models/HedgePyTorchMultiTaskRegressor.py",
        source / "tests/hedge/mlrl",
        source / "tools/run_hedge_mlrl_validation.py",
        source / "tools/validate_hedge_mlrl_sb3.py",
        source / "tools/validate_hedge_mlrl_code_quality.py",
    )
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(sorted(root.rglob("*.py")))
        elif root.is_file():
            files.append(root)
    return sorted(set(files))


def add_issue(
    issues: list[Issue],
    source: Path,
    path: Path,
    line: int,
    code: str,
    message: str,
) -> None:
    issues.append(Issue(path.relative_to(source).as_posix(), line, code, message))


def complexity(function: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    value = 1
    for node in ast.walk(function):
        if isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp)):
            value += 1
        elif isinstance(node, ast.Try):
            value += len(node.handlers) + int(bool(node.orelse))
    return value


def scan_annotations(
    issues: list[Issue],
    source: Path,
    path: Path,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> None:
    annotations = [node.returns]
    annotations.extend(
        argument.annotation
        for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
    )
    for annotation in annotations:
        if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
            add_issue(
                issues,
                source,
                path,
                annotation.lineno,
                "UP037",
                "quoted annotation is unnecessary with future annotations",
            )


def scan_basic_call_rules(
    issues: list[Issue],
    source: Path,
    path: Path,
    node: ast.Call,
) -> None:
    if (
        isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and not node.args
        and node.keywords
    ):
        add_issue(issues, source, path, node.lineno, "C408", "use a dict literal")
    if (
        isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) >= 2
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    ):
        add_issue(
            issues,
            source,
            path,
            node.lineno,
            "B009",
            "constant getattr attribute",
        )
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "os"
        and node.func.attr == "replace"
    ):
        add_issue(issues, source, path, node.lineno, "PTH105", "prefer Path.replace")


def scan_decimal_call(
    issues: list[Issue],
    source: Path,
    path: Path,
    node: ast.Call,
) -> None:
    if not (isinstance(node.func, ast.Name) and node.func.id == "Decimal" and node.args):
        return
    argument = node.args[0]
    if (
        isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and re.fullmatch(r"[+-]?\d+", argument.value)
    ):
        add_issue(
            issues,
            source,
            path,
            node.lineno,
            "FURB157",
            "construct integral Decimal from int",
        )


def scan_successive_zip(
    issues: list[Issue],
    source: Path,
    path: Path,
    node: ast.Call,
) -> None:
    if not (isinstance(node.func, ast.Name) and node.func.id == "zip" and len(node.args) >= 2):
        return
    left, right = node.args[:2]
    if not (isinstance(left, ast.Name) and isinstance(right, ast.Subscript)):
        return
    same_name = isinstance(right.value, ast.Name) and right.value.id == left.id
    sliced_from_one = (
        isinstance(right.slice, ast.Slice)
        and isinstance(right.slice.lower, ast.Constant)
        and right.slice.lower.value == 1
    )
    if same_name and sliced_from_one:
        add_issue(
            issues,
            source,
            path,
            node.lineno,
            "RUF007",
            "prefer itertools.pairwise for successive pairs",
        )


def scan_call(
    issues: list[Issue],
    source: Path,
    path: Path,
    node: ast.Call,
) -> None:
    scan_basic_call_rules(issues, source, path, node)
    scan_decimal_call(issues, source, path, node)
    scan_successive_zip(issues, source, path, node)




def exported_names(tree: ast.Module) -> set[str]:
    exported: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, (ast.List, ast.Tuple, ast.Set)):
            for item in node.value.elts:
                if isinstance(item, ast.Constant) and isinstance(item.value, str):
                    exported.add(item.value)
    return exported


def scan_unused_imports(
    issues: list[Issue],
    source: Path,
    path: Path,
    tree: ast.Module,
) -> None:
    used = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    used.update(exported_names(tree))
    used.update(assigned)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                if bound not in used:
                    add_issue(
                        issues,
                        source,
                        path,
                        node.lineno,
                        "F401-proxy",
                        f"unused import {bound}",
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module == "__future__":
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name
                if bound not in used:
                    add_issue(
                        issues,
                        source,
                        path,
                        node.lineno,
                        "F401-proxy",
                        f"unused import {bound}",
                    )


def scan_dataclass_call_defaults(
    issues: list[Issue],
    source: Path,
    path: Path,
    tree: ast.Module,
) -> None:
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        is_dataclass = any(
            (isinstance(decorator, ast.Name) and decorator.id == "dataclass")
            or (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "dataclass"
            )
            for decorator in node.decorator_list
        )
        if not is_dataclass:
            continue
        for statement in node.body:
            if isinstance(statement, ast.AnnAssign) and isinstance(statement.value, ast.Call):
                is_field = (
                    isinstance(statement.value.func, ast.Name)
                    and statement.value.func.id == "field"
                )
                if not is_field:
                    add_issue(
                        issues,
                        source,
                        path,
                        statement.lineno,
                        "RUF009-proxy",
                        "dataclass field default performs a function call",
                    )

def scan_module_docstring_directives(
    issues: list[Issue],
    source: Path,
    path: Path,
    tree: ast.Module,
) -> None:
    if not tree.body:
        return
    first = tree.body[0]
    is_docstring = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    if is_docstring and "# ruff:" in first.value.value:
        add_issue(
            issues,
            source,
            path,
            first.lineno,
            "RUF-DIRECTIVE-PROXY",
            "Ruff directive is inside the module docstring and is ineffective",
        )


def scan_tree(source: Path, path: Path, text: str) -> list[Issue]:
    issues: list[Issue] = []
    tree = ast.parse(text, filename=str(path))
    scan_module_docstring_directives(issues, source, path, tree)
    scan_unused_imports(issues, source, path, tree)
    scan_dataclass_call_defaults(issues, source, path, tree)
    is_test = path.is_relative_to(source / "tests")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            bad = sorted(alias.name for alias in node.names if alias.name in TYPING_ABCS)
            if bad:
                add_issue(
                    issues,
                    source,
                    path,
                    node.lineno,
                    "UP035",
                    f"typing ABC imports: {bad}",
                )
        if isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            add_issue(issues, source, path, node.lineno, "F403", "wildcard import")
        if isinstance(node, ast.Assert) and not is_test:
            add_issue(issues, source, path, node.lineno, "S101", "assert in non-test code")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            scan_annotations(issues, source, path, node)
            defaults = list(node.args.defaults)
            defaults.extend(item for item in node.args.kw_defaults if item is not None)
            for default in defaults:
                if isinstance(default, (ast.Call, ast.List, ast.Dict, ast.Set)):
                    add_issue(
                        issues,
                        source,
                        path,
                        default.lineno,
                        "B008/B006",
                        f"mutable or called default in {node.name}",
                    )
            score = complexity(node)
            if score > 12:
                add_issue(
                    issues,
                    source,
                    path,
                    node.lineno,
                    "C901-proxy",
                    f"complexity proxy {score} > 12 in {node.name}",
                )
        if isinstance(node, ast.Call):
            scan_call(issues, source, path, node)
            if isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"}:
                add_issue(
                    issues,
                    source,
                    path,
                    node.lineno,
                    "S307",
                    f"forbidden dynamic call {node.func.id}",
                )
    return issues


def scan_file(source: Path, path: Path) -> list[Issue]:
    text = path.read_text(encoding="utf-8")
    issues: list[Issue] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if len(line) > MAX_LINE_LENGTH:
            add_issue(
                issues,
                source,
                path,
                line_number,
                "E501",
                f"line length {len(line)} > {MAX_LINE_LENGTH}",
            )
    try:
        issues.extend(scan_tree(source, path, text))
    except SyntaxError as exc:
        add_issue(
            issues,
            source,
            path,
            exc.lineno or 0,
            "E999",
            str(exc),
        )
    return issues


def run(source: Path) -> dict[str, object]:
    source = source.resolve()
    files = iter_scope(source)
    issues: list[Issue] = []
    for path in files:
        issues.extend(scan_file(source, path))
    payload = {
        "schema": "freqtrade-hedge-mlrl-code-quality-v1",
        "release": RELEASE,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if not issues else "FAIL",
        "python_files_scanned": len(files),
        "issue_count": len(issues),
        "issues": [asdict(issue) for issue in issues],
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = run(args.source)
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
