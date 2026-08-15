"""Safe argv-only launch plans for existing Hedge/FreqAI research entry points."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .contracts import ResearchKind


_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,127}$")
_MODEL_BY_KIND = {
    ResearchKind.ML_TRAIN: "HedgePyTorchMultiTaskRegressor",
    ResearchKind.ML_EVAL: "HedgePyTorchMultiTaskRegressor",
    ResearchKind.RL_TRAIN: "HedgeReinforcementLearner",
    ResearchKind.RL_EVAL: "HedgeReinforcementLearner",
}


@dataclass(frozen=True, slots=True)
class ResearchCommandPlan:
    kind: ResearchKind
    argv: tuple[str, ...]
    exchange_write_enabled: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "argv": list(self.argv),
            "exchange_write_enabled": self.exchange_write_enabled,
        }


def _safe_name(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not _NAME.fullmatch(normalized):
        raise ValueError(f"{field} must be a Python-style identifier")
    return normalized


def _base_argv(
    config_path: Path,
    *,
    python_executable: str | None,
    extra_config_paths: tuple[Path, ...] = (),
) -> list[str]:
    paths = (config_path, *extra_config_paths)
    argv = [python_executable or sys.executable, "-m", "freqtrade"]
    for raw_path in paths:
        config = raw_path.expanduser().resolve()
        if not config.is_file():
            raise ValueError(f"config file does not exist: {config}")
        argv.extend(["--config", str(config)])
    return argv


def _command_for_kind(kind: ResearchKind) -> str:
    if kind in _MODEL_BY_KIND:
        return "backtesting"
    if kind is ResearchKind.OPTIMIZATION:
        return "hedge-research-optimize"
    return "hedge-backtesting"


def _append_optimization_args(
    argv: list[str],
    *,
    trials: int | None,
    workers: int | None,
) -> None:
    if trials is not None:
        if trials < 1:
            raise ValueError("optimization trials must be positive")
        argv.extend(["--hedge-trials", str(trials)])
    if workers is not None:
        if workers < 1:
            raise ValueError("optimization workers must be positive")
        argv.extend(["--hedge-workers", str(workers)])


def _append_output_args(
    argv: list[str],
    *,
    kind: ResearchKind,
    output_directory: Path,
) -> None:
    output_root = output_directory.expanduser().resolve()
    if kind is ResearchKind.BACKTEST:
        argv.extend(
            [
                "--hedge-export-filename",
                str(output_root / "backtest-result.json"),
            ]
        )
        return
    if kind is ResearchKind.OPTIMIZATION:
        argv.extend(
            [
                "--hedge-optimization-output",
                str(output_root / "optimization"),
            ]
        )
        return
    if kind in _MODEL_BY_KIND:
        argv.extend(
            [
                "--export",
                "trades",
                "--backtest-directory",
                str(output_root / "freqai-backtest"),
            ]
        )


def build_command_plan(
    kind: ResearchKind,
    *,
    config_path: Path,
    strategy: str,
    timerange: str = "",
    trials: int | None = None,
    workers: int | None = None,
    python_executable: str | None = None,
    output_directory: Path | None = None,
    extra_config_paths: tuple[Path, ...] = (),
) -> ResearchCommandPlan:
    argv = _base_argv(
        config_path,
        python_executable=python_executable,
        extra_config_paths=extra_config_paths,
    )
    argv.insert(3, _command_for_kind(kind))
    argv.extend(["--strategy", _safe_name(strategy, field="strategy")])

    normalized_timerange = timerange.strip()
    if normalized_timerange:
        argv.extend(["--timerange", normalized_timerange])

    if kind is ResearchKind.OPTIMIZATION:
        _append_optimization_args(argv, trials=trials, workers=workers)

    model = _MODEL_BY_KIND.get(kind)
    if model is not None:
        argv.extend(["--freqaimodel", model])

    if output_directory is not None:
        _append_output_args(
            argv,
            kind=kind,
            output_directory=output_directory,
        )

    return ResearchCommandPlan(kind=kind, argv=tuple(argv))
