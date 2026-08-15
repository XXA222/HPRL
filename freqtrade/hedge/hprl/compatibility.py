"""Fail-closed compatibility probe for the Clean Mainline contracts consumed by HPRL."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


_REQUIRED_SIGNAL_FIELDS = frozenset(
    {
        "symbol",
        "timeframe",
        "candle_close_time",
        "feature_timestamp",
        "long_score",
        "short_score",
        "target_net",
        "target_net_ratio",
        "confidence",
        "risk_scale",
        "allow_new_risk",
        "model_version",
    }
)
_REQUIRED_CANONICAL_FILES = (
    "freqtrade/hedge/integration/signal_provider.py",
    "freqtrade/hedge/planning/target.py",
    "freqtrade/hedge/risk/engine.py",
    "freqtrade/hedge/simulation/exchange.py",
)


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    compatible: bool
    mainline_version: str
    missing_files: tuple[str, ...]
    missing_signal_fields: tuple[str, ...]


def _signal_snapshot_fields(path: Path) -> set[str]:
    """Read the dataclass contract without importing the full Freqtrade runtime."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "SignalSnapshot":
            return {
                item.target.id
                for item in node.body
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
            }
    return set()


def inspect_clean_mainline(project_root: str | Path) -> CompatibilityReport:
    root = Path(project_root).resolve()
    version_path = root / "CLEAN-MAINLINE-VERSION.txt"
    version = version_path.read_text(encoding="utf-8").strip() if version_path.is_file() else ""
    missing_files = tuple(
        relative for relative in _REQUIRED_CANONICAL_FILES if not (root / relative).is_file()
    )

    signal_path = root / "freqtrade/hedge/integration/signal_provider.py"
    try:
        names = _signal_snapshot_fields(signal_path) if signal_path.is_file() else set()
    except (OSError, SyntaxError, UnicodeError):
        names = set()
    missing_signal_fields = tuple(sorted(_REQUIRED_SIGNAL_FIELDS - names))

    compatible = bool(version) and not missing_files and not missing_signal_fields
    return CompatibilityReport(
        compatible=compatible,
        mainline_version=version,
        missing_files=missing_files,
        missing_signal_fields=missing_signal_fields,
    )


def assert_clean_mainline_compatible(project_root: str | Path) -> CompatibilityReport:
    report = inspect_clean_mainline(project_root)
    if not report.compatible:
        raise RuntimeError(f"HPRL Clean Mainline compatibility check failed: {report}")
    return report
