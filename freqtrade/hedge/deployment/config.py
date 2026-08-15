"""Strict deployment configuration for the Hedge process supervisor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping, Sequence


class DeploymentConfigError(ValueError):
    """Raised when deployment configuration is unsafe or incomplete."""


class DeploymentMode(StrEnum):
    PAPER = "HEDGE_SIMULATED"
    PRODUCTION_LOCKED = "HEDGE_PRODUCTION_LOCKED"
    TESTNET_VALIDATION = "HEDGE_TESTNET_VALIDATION"


_ALLOWED_SYMBOLS = ("BTCUSDT", "ETHUSDT")


@dataclass(frozen=True, slots=True)
class RestartPolicy:
    max_restarts: int = 3
    window_seconds: int = 300
    base_backoff_seconds: float = 2.0
    max_backoff_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not 0 <= self.max_restarts <= 20:
            raise DeploymentConfigError("max_restarts must be between 0 and 20")
        if not 30 <= self.window_seconds <= 86400:
            raise DeploymentConfigError("window_seconds must be between 30 and 86400")
        if not 0.1 <= self.base_backoff_seconds <= 300:
            raise DeploymentConfigError("base_backoff_seconds must be between 0.1 and 300")
        if not self.base_backoff_seconds <= self.max_backoff_seconds <= 3600:
            raise DeploymentConfigError(
                "max_backoff_seconds must be >= base_backoff_seconds and <= 3600"
            )


@dataclass(frozen=True, slots=True)
class HedgeDeploymentConfig:
    project_root: Path
    freqtrade_config: Path
    mode: DeploymentMode
    symbols: tuple[str, ...]
    state_dir: Path
    log_dir: Path
    backup_dir: Path
    database_path: Path | None
    python_executable: Path
    additional_args: tuple[str, ...]
    heartbeat_interval_seconds: float
    heartbeat_stale_seconds: float
    graceful_shutdown_seconds: float
    restart_policy: RestartPolicy
    security_readiness_report: Path

    @classmethod
    def from_file(cls, path: Path) -> "HedgeDeploymentConfig":
        config_path = path.expanduser().resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise DeploymentConfigError("deployment config must be a JSON object")
        return cls.from_mapping(payload, source_path=config_path)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        source_path: Path | None = None,
    ) -> "HedgeDeploymentConfig":
        allowed_keys = {
            "project_root",
            "freqtrade_config",
            "mode",
            "symbols",
            "state_dir",
            "log_dir",
            "backup_dir",
            "database_path",
            "python_executable",
            "additional_args",
            "heartbeat_interval_seconds",
            "heartbeat_stale_seconds",
            "graceful_shutdown_seconds",
            "restart_policy",
            "security_readiness_report",
        }
        unknown = sorted(set(payload) - allowed_keys)
        if unknown:
            raise DeploymentConfigError(f"unknown deployment config keys: {unknown}")

        base = source_path.parent if source_path is not None else Path.cwd()

        def resolve_path(name: str, *, required: bool = True) -> Path | None:
            raw = payload.get(name)
            if raw is None:
                if required:
                    raise DeploymentConfigError(f"missing required path: {name}")
                return None
            if not isinstance(raw, str) or not raw.strip():
                raise DeploymentConfigError(f"{name} must be a non-empty string")
            candidate = Path(raw.strip()).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            return candidate.resolve()

        project_root = resolve_path("project_root")
        assert project_root is not None
        if not project_root.is_dir():
            raise DeploymentConfigError(f"project_root does not exist: {project_root}")

        freqtrade_config = resolve_path("freqtrade_config")
        assert freqtrade_config is not None
        if not freqtrade_config.is_file():
            raise DeploymentConfigError(f"freqtrade_config does not exist: {freqtrade_config}")

        python_raw = payload.get("python_executable")
        if isinstance(python_raw, str) and python_raw.strip().upper() == "AUTO":
            candidates = (
                project_root / ".venv" / "Scripts" / "python.exe",
                project_root / ".venv" / "bin" / "python",
            )
            python_executable = next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)
            if python_executable is None:
                raise DeploymentConfigError("python_executable=AUTO did not find a project virtual environment")
        else:
            python_executable = resolve_path("python_executable")
            assert python_executable is not None
            if not python_executable.is_file():
                raise DeploymentConfigError(f"python_executable does not exist: {python_executable}")

        readiness_raw = payload.get("security_readiness_report")
        if isinstance(readiness_raw, str) and readiness_raw.strip().upper() == "AUTO":
            report_root = project_root / "user_data" / "audit" / "security-readiness"
            candidates = sorted(
                report_root.glob("**/SECURITY-DEPLOYMENT-READINESS.json"),
                key=lambda candidate: candidate.stat().st_mtime_ns,
                reverse=True,
            )
            security_readiness_report = candidates[0].resolve() if candidates else None
            if security_readiness_report is None:
                raise DeploymentConfigError(
                    "security_readiness_report=AUTO did not find a current readiness report"
                )
        else:
            security_readiness_report = resolve_path("security_readiness_report")
            assert security_readiness_report is not None
            if not security_readiness_report.is_file():
                raise DeploymentConfigError(
                    f"security_readiness_report does not exist: {security_readiness_report}"
                )

        try:
            mode = DeploymentMode(str(payload.get("mode", "")).strip())
        except ValueError as exc:
            raise DeploymentConfigError(
                "mode must be HEDGE_SIMULATED, HEDGE_PRODUCTION_LOCKED, "
                "or HEDGE_TESTNET_VALIDATION"
            ) from exc

        symbols_raw = payload.get("symbols", list(_ALLOWED_SYMBOLS))
        if not isinstance(symbols_raw, Sequence) or isinstance(symbols_raw, (str, bytes)):
            raise DeploymentConfigError("symbols must be a JSON array")
        symbols = tuple(dict.fromkeys(str(item).strip().upper() for item in symbols_raw))
        if not symbols or any(symbol not in _ALLOWED_SYMBOLS for symbol in symbols):
            raise DeploymentConfigError("symbols must be a non-empty subset of BTCUSDT and ETHUSDT")

        additional_raw = payload.get("additional_args", [])
        if not isinstance(additional_raw, Sequence) or isinstance(additional_raw, (str, bytes)):
            raise DeploymentConfigError("additional_args must be a JSON array")
        additional_args = tuple(str(item) for item in additional_raw)
        forbidden_fragments = (
            "--live",
            "enable-live-trading",
            "production-armed",
            "hedge_production_armed",
        )
        if any(
            any(fragment in arg.lower() for fragment in forbidden_fragments)
            for arg in additional_args
        ):
            raise DeploymentConfigError("additional_args contains a forbidden live-trading flag")

        restart_raw = payload.get("restart_policy", {})
        if not isinstance(restart_raw, Mapping):
            raise DeploymentConfigError("restart_policy must be a JSON object")
        restart_policy = RestartPolicy(
            max_restarts=int(restart_raw.get("max_restarts", 3)),
            window_seconds=int(restart_raw.get("window_seconds", 300)),
            base_backoff_seconds=float(restart_raw.get("base_backoff_seconds", 2.0)),
            max_backoff_seconds=float(restart_raw.get("max_backoff_seconds", 30.0)),
        )

        heartbeat_interval = float(payload.get("heartbeat_interval_seconds", 10.0))
        heartbeat_stale = float(payload.get("heartbeat_stale_seconds", 45.0))
        graceful_shutdown = float(payload.get("graceful_shutdown_seconds", 30.0))
        if not 1 <= heartbeat_interval <= 300:
            raise DeploymentConfigError("heartbeat_interval_seconds must be between 1 and 300")
        if not heartbeat_interval * 2 <= heartbeat_stale <= 3600:
            raise DeploymentConfigError(
                "heartbeat_stale_seconds must be at least twice heartbeat_interval_seconds"
            )
        if not 1 <= graceful_shutdown <= 600:
            raise DeploymentConfigError("graceful_shutdown_seconds must be between 1 and 600")

        def under_project(name: str, default: str) -> Path:
            raw = payload.get(name, default)
            if not isinstance(raw, str) or not raw.strip():
                raise DeploymentConfigError(f"{name} must be a non-empty string")
            candidate = Path(raw.strip()).expanduser()
            if not candidate.is_absolute():
                candidate = project_root / candidate
            resolved = candidate.resolve()
            try:
                resolved.relative_to(project_root)
            except ValueError as exc:
                raise DeploymentConfigError(f"{name} must remain under project_root") from exc
            return resolved

        state_dir = under_project("state_dir", "user_data/hedge/runtime")
        log_dir = under_project("log_dir", "user_data/hedge/runtime/logs")
        backup_dir = under_project("backup_dir", "user_data/hedge/runtime/backups")

        database_path = resolve_path("database_path", required=False)
        if database_path is not None:
            try:
                database_path.relative_to(project_root)
            except ValueError as exc:
                raise DeploymentConfigError("database_path must remain under project_root") from exc

        return cls(
            project_root=project_root,
            freqtrade_config=freqtrade_config,
            mode=mode,
            symbols=symbols,
            state_dir=state_dir,
            log_dir=log_dir,
            backup_dir=backup_dir,
            database_path=database_path,
            python_executable=python_executable,
            additional_args=additional_args,
            heartbeat_interval_seconds=heartbeat_interval,
            heartbeat_stale_seconds=heartbeat_stale,
            graceful_shutdown_seconds=graceful_shutdown,
            restart_policy=restart_policy,
            security_readiness_report=security_readiness_report,
        )

    def child_command(self) -> tuple[str, ...]:
        return (
            str(self.python_executable),
            "-m",
            "freqtrade",
            "trade",
            "--config",
            str(self.freqtrade_config),
            *self.additional_args,
        )
