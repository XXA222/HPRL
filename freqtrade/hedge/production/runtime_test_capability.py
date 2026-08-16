"""Container test capability bootstrap and source-safe pytest execution.

The production venv intentionally does not need development tools at runtime.  This
module derives the test toolchain from the repository's ``develop`` optional dependency
set, installs only those dependency specs into the *current Python environment* when
explicitly requested, and runs pytest with all cache/temp artifacts redirected outside
of the source tree.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Iterable
import xml.etree.ElementTree as ET

_TEST_DISTRIBUTION_PREFIXES = (
    "pytest",
    "time-machine",
)
_RUNTIME_CRITICAL_DISTRIBUTIONS = ("ccxt", "sqlalchemy", "humanize", "aiohttp")
_IMPORT_BY_DISTRIBUTION = {
    "pytest": "pytest",
    "pytest-asyncio": "pytest_asyncio",
    "pytest-cov": "pytest_cov",
    "pytest-mock": "pytest_mock",
    "pytest-random-order": "random_order",
    "pytest-timeout": "pytest_timeout",
    "pytest-xdist": "xdist",
    "time-machine": "time_machine",
}
_SPEC_NAME = re.compile(r"^\s*([A-Za-z0-9_.-]+)")


def _aware_now() -> datetime:
    return datetime.now(UTC)


def _distribution_name(spec: str) -> str:
    match = _SPEC_NAME.match(spec)
    if match is None:
        raise ValueError(f"invalid dependency spec: {spec!r}")
    return match.group(1).lower().replace("_", "-")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_digest(root: Path, paths: Iterable[Path]) -> str:
    digest = sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        digest.update(_file_sha256(path).encode("ascii") + b"\0")
    return digest.hexdigest()


def source_python_files(root: str | Path) -> tuple[Path, ...]:
    base = Path(root).resolve()
    selected: list[Path] = []
    for prefix in ("freqtrade", "tests/hedge", "tools"):
        folder = base / prefix
        if not folder.exists():
            continue
        selected.extend(
            path
            for path in folder.rglob("*.py")
            if "__pycache__" not in path.parts
        )
    return tuple(sorted(set(selected)))


@dataclass(frozen=True, slots=True)
class TestDependencyStatus:
    distribution: str
    spec: str
    import_name: str
    available: bool


@dataclass(frozen=True, slots=True)
class RuntimeTestCapabilityReport:
    python_executable: str
    source_root: str
    develop_specs: tuple[str, ...]
    test_specs: tuple[str, ...]
    dependencies: tuple[TestDependencyStatus, ...]
    runtime_dependencies: tuple[TestDependencyStatus, ...]
    pytest_available: bool
    xdist_available: bool
    postgres_driver: str
    source_tree_sha256: str
    observed_at: datetime

    @property
    def missing_specs(self) -> tuple[str, ...]:
        return tuple(item.spec for item in self.dependencies if not item.available)

    @property
    def missing_runtime_specs(self) -> tuple[str, ...]:
        return tuple(item.spec for item in self.runtime_dependencies if not item.available)

    @property
    def ready_for_serial_pytest(self) -> bool:
        return self.pytest_available and not self.missing_runtime_specs

    @property
    def ready_for_full_pytest(self) -> bool:
        return self.ready_for_serial_pytest and not self.missing_specs

    @property
    def ready_for_postgres_acceptance(self) -> bool:
        return self.postgres_driver in {"psycopg", "psycopg2"}


def read_test_dependency_specs(root: str | Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    base = Path(root).resolve()
    payload = tomllib.loads((base / "pyproject.toml").read_text(encoding="utf-8"))
    optional = payload.get("project", {}).get("optional-dependencies", {})
    develop = tuple(str(item) for item in optional.get("develop", ()))
    test_specs = tuple(
        item
        for item in develop
        if _distribution_name(item).startswith(_TEST_DISTRIBUTION_PREFIXES)
    )
    if "pytest" not in {_distribution_name(item) for item in test_specs}:
        raise ValueError("pyproject develop dependencies do not declare pytest")
    return develop, test_specs


def read_runtime_dependency_specs(root: str | Path) -> tuple[str, ...]:
    base = Path(root).resolve()
    payload = tomllib.loads((base / "pyproject.toml").read_text(encoding="utf-8"))
    declared = tuple(str(item) for item in payload.get("project", {}).get("dependencies", ()))
    by_name = {_distribution_name(item): item for item in declared}
    missing = [name for name in _RUNTIME_CRITICAL_DISTRIBUTIONS if name not in by_name]
    if missing:
        raise ValueError("pyproject is missing critical runtime dependencies: " + ",".join(missing))
    return tuple(by_name[name] for name in _RUNTIME_CRITICAL_DISTRIBUTIONS)


def read_postgres_dependency_specs(root: str | Path) -> tuple[str, ...]:
    """Return the source-controlled PostgreSQL acceptance dependency set.

    PostgreSQL support is intentionally kept outside the protected core ``pyproject.toml``
    authority.  Runtime Closure uses a dedicated requirements file so normal Freqtrade/HPRL
    installs do not acquire a database driver they do not need.
    """
    base = Path(root).resolve()
    path = base / "requirements-hprl-postgres.txt"
    if not path.is_file():
        raise ValueError("requirements-hprl-postgres.txt is missing")
    specs = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if not specs:
        raise ValueError("requirements-hprl-postgres.txt is empty")
    if any(line.startswith(("-", ".", "/")) or "://" in line for line in specs):
        raise ValueError("PostgreSQL requirements must contain package specs only")
    names = {_distribution_name(item) for item in specs}
    if "psycopg" not in names and "psycopg2" not in names and "psycopg2-binary" not in names:
        raise ValueError("PostgreSQL requirements must declare psycopg/psycopg2")
    return specs


def probe_runtime_test_capability(root: str | Path) -> RuntimeTestCapabilityReport:
    base = Path(root).resolve()
    develop, test_specs = read_test_dependency_specs(base)
    runtime_specs = read_runtime_dependency_specs(base)
    rows: list[TestDependencyStatus] = []
    for spec in test_specs:
        distribution = _distribution_name(spec)
        import_name = _IMPORT_BY_DISTRIBUTION.get(distribution, distribution.replace("-", "_"))
        rows.append(
            TestDependencyStatus(
                distribution=distribution,
                spec=spec,
                import_name=import_name,
                available=importlib.util.find_spec(import_name) is not None,
            )
        )
    runtime_rows: list[TestDependencyStatus] = []
    for spec in runtime_specs:
        distribution = _distribution_name(spec)
        import_name = "sqlalchemy" if distribution == "sqlalchemy" else distribution.replace("-", "_")
        runtime_rows.append(TestDependencyStatus(
            distribution=distribution, spec=spec, import_name=import_name,
            available=importlib.util.find_spec(import_name) is not None,
        ))
    postgres_driver = ""
    if importlib.util.find_spec("psycopg") is not None:
        postgres_driver = "psycopg"
    elif importlib.util.find_spec("psycopg2") is not None:
        postgres_driver = "psycopg2"
    files = source_python_files(base)
    return RuntimeTestCapabilityReport(
        python_executable=sys.executable,
        source_root=str(base),
        develop_specs=develop,
        test_specs=test_specs,
        dependencies=tuple(rows),
        runtime_dependencies=tuple(runtime_rows),
        pytest_available=importlib.util.find_spec("pytest") is not None,
        xdist_available=importlib.util.find_spec("xdist") is not None,
        postgres_driver=postgres_driver,
        source_tree_sha256=_tree_digest(base, files),
        observed_at=_aware_now(),
    )


def bootstrap_runtime_test_dependencies(
    root: str | Path,
    *,
    include_all_test_plugins: bool = True,
    timeout_seconds: int = 900,
) -> RuntimeTestCapabilityReport:
    """Install missing test dependencies into the current interpreter's environment.

    No project path is passed to pip, no editable install is performed, and no files are
    written below ``root``.  The caller is expected to invoke this with the production
    venv Python executable, e.g. ``/opt/hedge-venv/bin/python``.
    """
    base = Path(root).resolve()
    before = probe_runtime_test_capability(base)
    if before.missing_runtime_specs:
        raise RuntimeError(
            "declared runtime dependencies are missing from the selected Python environment: "
            + ", ".join(before.missing_runtime_specs)
        )
    missing = list(before.missing_specs)
    if not include_all_test_plugins:
        missing = [spec for spec in missing if _distribution_name(spec) == "pytest"]
    if missing:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            *missing,
        ]
        env = dict(os.environ)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            command,
            cwd=tempfile.gettempdir(),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-40:])
            raise RuntimeError(f"test dependency bootstrap failed rc={completed.returncode}\n{tail}")
    after = probe_runtime_test_capability(base)
    if before.source_tree_sha256 != after.source_tree_sha256:
        raise RuntimeError("test dependency bootstrap modified source Python files")
    return after


def bootstrap_postgres_driver(
    root: str | Path,
    *,
    timeout_seconds: int = 900,
) -> RuntimeTestCapabilityReport:
    """Install the declared optional PostgreSQL driver into the current interpreter.

    The project itself is never installed and source Python bytes are attested before and
    after the operation.  Existing psycopg/psycopg2 installations are left untouched.
    """
    base = Path(root).resolve()
    before = probe_runtime_test_capability(base)
    if before.ready_for_postgres_acceptance:
        return before
    specs = read_postgres_dependency_specs(base)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        *specs,
    ]
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=tempfile.gettempdir(),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise RuntimeError(f"PostgreSQL driver bootstrap failed rc={completed.returncode}\n{tail}")
    after = probe_runtime_test_capability(base)
    if before.source_tree_sha256 != after.source_tree_sha256:
        raise RuntimeError("PostgreSQL driver bootstrap modified source Python files")
    if not after.ready_for_postgres_acceptance:
        raise RuntimeError("PostgreSQL driver bootstrap completed but psycopg/psycopg2 is unavailable")
    return after


@dataclass(frozen=True, slots=True)
class PytestSuiteResult:
    name: str
    returncode: int
    tests: int
    minimum_tests: int
    failures: int
    errors: int
    skipped: int
    duration_seconds: float
    junit_sha256: str
    stdout_tail: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.returncode == 0
            and self.failures == 0
            and self.errors == 0
            and self.tests >= self.minimum_tests > 0
        )


@dataclass(frozen=True, slots=True)
class RuntimeTestSuiteReport:
    suites: tuple[PytestSuiteResult, ...]
    source_sha256_before: str
    source_sha256_after: str
    source_unchanged: bool
    pytest_available: bool
    observed_at: datetime

    @property
    def passed(self) -> bool:
        return self.pytest_available and self.source_unchanged and all(item.passed for item in self.suites)

    @property
    def tests(self) -> int:
        return sum(item.tests for item in self.suites)


def _junit_counts(path: Path) -> tuple[int, int, int, int, float]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = failures = errors = skipped = 0
    duration = 0.0
    for suite in suites:
        tests += int(suite.attrib.get("tests", "0"))
        failures += int(suite.attrib.get("failures", "0"))
        errors += int(suite.attrib.get("errors", "0"))
        skipped += int(suite.attrib.get("skipped", "0"))
        try:
            duration += float(suite.attrib.get("time", "0"))
        except ValueError:
            pass
    return tests, failures, errors, skipped, duration


def run_runtime_pytest_suites(
    root: str | Path,
    *,
    suites: tuple[tuple[str, tuple[str, ...], int], ...] | None = None,
    timeout_seconds: int = 3600,
    use_xdist: bool = False,
) -> RuntimeTestSuiteReport:
    base = Path(root).resolve()
    capability = probe_runtime_test_capability(base)
    if not capability.pytest_available:
        return RuntimeTestSuiteReport((), capability.source_tree_sha256, capability.source_tree_sha256, True, False, _aware_now())
    selected = suites or (
        ("hprl", ("tests/hedge/hprl",), 530),
        ("execution", ("tests/hedge/execution",), 108),
        ("production", ("tests/hedge/production",), 116),
    )
    files = source_python_files(base)
    before = _tree_digest(base, files)
    results: list[PytestSuiteResult] = []
    temp_root = Path(tempfile.mkdtemp(prefix="hprl-runtime-pytest-"))
    try:
        for index, (name, targets, minimum_tests) in enumerate(selected):
            junit = temp_root / f"{index:02d}-{name}.xml"
            basetemp = temp_root / f"{index:02d}-{name}-tmp"
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                "-o",
                "addopts=",
                "--basetemp",
                str(basetemp),
                "--junitxml",
                str(junit),
            ]
            if use_xdist and capability.xdist_available:
                command.extend(["-n", "auto"])
            command.extend(targets)
            env = dict(os.environ)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["PYTHONPATH"] = str(base) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            completed = subprocess.run(
                command,
                cwd=base,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
            )
            if junit.is_file():
                counts = _junit_counts(junit)
                junit_hash = _file_sha256(junit)
            else:
                counts = (0, 0, 1, 0, 0.0)
                junit_hash = "0" * 64
            results.append(
                PytestSuiteResult(
                    name=name,
                    returncode=completed.returncode,
                    tests=counts[0], minimum_tests=minimum_tests, failures=counts[1], errors=counts[2], skipped=counts[3],
                    duration_seconds=counts[4], junit_sha256=junit_hash,
                    stdout_tail=tuple(completed.stdout.splitlines()[-80:]),
                )
            )
            if completed.returncode != 0:
                break
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)
    after = _tree_digest(base, files)
    return RuntimeTestSuiteReport(
        suites=tuple(results),
        source_sha256_before=before,
        source_sha256_after=after,
        source_unchanged=(before == after),
        pytest_available=True,
        observed_at=_aware_now(),
    )


def report_json(report: object) -> str:
    from dataclasses import asdict

    return json.dumps(asdict(report), sort_keys=True, indent=2, default=str) + "\n"
