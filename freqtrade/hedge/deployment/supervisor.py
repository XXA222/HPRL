"""Single-instance subprocess supervisor with bounded crash recovery."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Protocol

from .backup import BackupEvidence, SQLiteBackupManager
from .config import HedgeDeploymentConfig
from .events import JsonlEventWriter
from .readiness import DeploymentReadiness, validate_security_readiness_report
from .single_instance import SingleInstanceLock
from .state import RuntimePhase, RuntimeState, RuntimeStateStore


class ChildProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SupervisorResult:
    phase: RuntimePhase
    exit_code: int | None
    restart_count: int
    stopped_by_request: bool
    readiness: DeploymentReadiness
    backup: BackupEvidence | None


class HedgeProcessSupervisor:
    def __init__(
        self,
        config: HedgeDeploymentConfig,
        *,
        process_factory: Callable[..., ChildProcess] = subprocess.Popen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.process_factory = process_factory
        self.sleep = sleep
        self.monotonic = monotonic
        self.lock = SingleInstanceLock(config.state_dir / "supervisor.lock")
        self.state_store = RuntimeStateStore(config.state_dir / "runtime-state.json")
        self.stop_request_path = config.state_dir / "stop.request"
        self.events = JsonlEventWriter(config.log_dir / "supervisor-events.jsonl")
        self._stop_requested = False
        self._child: ChildProcess | None = None

    def run(self) -> SupervisorResult:
        with self.lock:
            readiness = validate_security_readiness_report(self.config)
            backup = self._backup_database()
            identity = self.lock.identity
            assert identity is not None
            now = datetime.now(UTC).isoformat()
            state = RuntimeState(
                phase=RuntimePhase.PREFLIGHT,
                supervisor_pid=os.getpid(),
                instance_token=identity.token,
                child_pid=None,
                restart_count=0,
                started_at_utc=now,
                updated_at_utc=now,
                heartbeat_at_utc=now,
            )
            self.state_store.write(state)
            self.stop_request_path.unlink(missing_ok=True)
            self._install_signal_handlers()
            self.events.write(
                "SUPERVISOR_PREFLIGHT_PASS",
                readiness=readiness.status,
                security_readiness_report_sha256=readiness.report_sha256,
                mode=self.config.mode.value,
                symbols=self.config.symbols,
            )
            restarts: deque[float] = deque()
            last_exit: int | None = None
            while True:
                if self._should_stop():
                    final = state.evolve(
                        phase=RuntimePhase.STOPPED,
                        child_pid=None,
                        last_exit_code=last_exit,
                        stop_requested=True,
                    )
                    self.state_store.write(final)
                    self.events.write("SUPERVISOR_STOPPED", requested=True)
                    return SupervisorResult(
                        phase=final.phase,
                        exit_code=last_exit,
                        restart_count=final.restart_count,
                        stopped_by_request=True,
                        readiness=readiness,
                        backup=backup,
                    )

                state = state.evolve(phase=RuntimePhase.STARTING, child_pid=None)
                self.state_store.write(state)
                self._child = self._start_child()
                state = state.evolve(phase=RuntimePhase.RUNNING, child_pid=self._child.pid)
                self.state_store.write(state)
                self.events.write("CHILD_STARTED", child_pid=self._child.pid, restart_count=state.restart_count)
                last_exit = self._monitor_child(state)
                self._child = None
                if self._should_stop():
                    state = state.evolve(
                        phase=RuntimePhase.STOPPED,
                        child_pid=None,
                        last_exit_code=last_exit,
                        stop_requested=True,
                    )
                    self.state_store.write(state)
                    self.events.write("SUPERVISOR_STOPPED", requested=True, child_exit_code=last_exit)
                    return SupervisorResult(
                        phase=state.phase,
                        exit_code=last_exit,
                        restart_count=state.restart_count,
                        stopped_by_request=True,
                        readiness=readiness,
                        backup=backup,
                    )
                if last_exit == 0:
                    state = state.evolve(
                        phase=RuntimePhase.STOPPED,
                        child_pid=None,
                        last_exit_code=last_exit,
                    )
                    self.state_store.write(state)
                    self.events.write("CHILD_EXITED_CLEANLY")
                    return SupervisorResult(
                        phase=state.phase,
                        exit_code=last_exit,
                        restart_count=state.restart_count,
                        stopped_by_request=False,
                        readiness=readiness,
                        backup=backup,
                    )

                now_tick = self.monotonic()
                while restarts and now_tick - restarts[0] > self.config.restart_policy.window_seconds:
                    restarts.popleft()
                if len(restarts) >= self.config.restart_policy.max_restarts:
                    state = state.evolve(
                        phase=RuntimePhase.FAILED,
                        child_pid=None,
                        last_exit_code=last_exit,
                        last_error="RESTART_BUDGET_EXHAUSTED",
                    )
                    self.state_store.write(state)
                    self.events.write(
                        "RESTART_BUDGET_EXHAUSTED",
                        child_exit_code=last_exit,
                        restart_count=state.restart_count,
                    )
                    return SupervisorResult(
                        phase=state.phase,
                        exit_code=last_exit,
                        restart_count=state.restart_count,
                        stopped_by_request=False,
                        readiness=readiness,
                        backup=backup,
                    )
                restarts.append(now_tick)
                restart_count = state.restart_count + 1
                backoff = min(
                    self.config.restart_policy.base_backoff_seconds * (2 ** (restart_count - 1)),
                    self.config.restart_policy.max_backoff_seconds,
                )
                state = state.evolve(
                    phase=RuntimePhase.BACKOFF,
                    child_pid=None,
                    restart_count=restart_count,
                    last_exit_code=last_exit,
                )
                self.state_store.write(state)
                self.events.write(
                    "CHILD_RESTART_SCHEDULED",
                    child_exit_code=last_exit,
                    restart_count=restart_count,
                    backoff_seconds=backoff,
                )
                self._sleep_interruptibly(backoff)

    def request_stop(self) -> None:
        self.config.state_dir.mkdir(parents=True, exist_ok=True)
        self.stop_request_path.write_text("STOP\n", encoding="ascii", newline="\n")

    def _backup_database(self) -> BackupEvidence | None:
        if self.config.database_path is None or not self.config.database_path.is_file():
            return None
        manager = SQLiteBackupManager(self.config.backup_dir)
        evidence = manager.create(self.config.database_path, label="prestart")
        self.events.write("DATABASE_BACKUP_CREATED", evidence=asdict(evidence))
        return evidence

    def _start_child(self) -> ChildProcess:
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = self.config.log_dir / "freqtrade-stdout.log"
        stderr_path = self.config.log_dir / "freqtrade-stderr.log"
        stdout = stdout_path.open("ab", buffering=0)
        stderr = stderr_path.open("ab", buffering=0)
        safe_environment_keys = {
            "SYSTEMROOT",
            "WINDIR",
            "COMSPEC",
            "PATH",
            "PATHEXT",
            "TEMP",
            "TMP",
            "USERPROFILE",
            "HOMEDRIVE",
            "HOMEPATH",
            "APPDATA",
            "LOCALAPPDATA",
            "PROGRAMDATA",
            "PROGRAMFILES",
            "PROGRAMFILES(X86)",
            "NUMBER_OF_PROCESSORS",
            "PROCESSOR_ARCHITECTURE",
            "SSL_CERT_FILE",
            "REQUESTS_CA_BUNDLE",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
        }
        environment = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in safe_environment_keys
        }
        environment["FREQTRADE_HEDGE_DEPLOYMENT_MODE"] = self.config.mode.value
        environment["FREQTRADE_HEDGE_ALLOWED_SYMBOLS"] = ",".join(self.config.symbols)
        environment["PYTHONUNBUFFERED"] = "1"
        try:
            child = self.process_factory(
                list(self.config.child_command()),
                cwd=str(self.config.project_root),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                shell=False,
                creationflags=(
                    subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                ),
            )
            return child
        finally:
            stdout.close()
            stderr.close()

    def _monitor_child(self, state: RuntimeState) -> int:
        assert self._child is not None
        current = state
        while True:
            exit_code = self._child.poll()
            if exit_code is not None:
                self.events.write("CHILD_EXITED", child_pid=self._child.pid, exit_code=exit_code)
                return int(exit_code)
            if self._should_stop():
                self._stop_child()
                exit_code = self._child.poll()
                if exit_code is None:
                    exit_code = self._child.wait(timeout=5)
                return int(exit_code)
            current = current.evolve(phase=RuntimePhase.RUNNING, child_pid=self._child.pid)
            self.state_store.write(current)
            self.sleep(self.config.heartbeat_interval_seconds)

    def _stop_child(self) -> None:
        child = self._child
        if child is None or child.poll() is not None:
            return
        self.events.write("CHILD_GRACEFUL_STOP_REQUESTED", child_pid=child.pid)
        try:
            if (
                os.name == "nt"
                and isinstance(child, subprocess.Popen)
                and hasattr(signal, "CTRL_BREAK_EVENT")
            ):
                os.kill(child.pid, signal.CTRL_BREAK_EVENT)
            else:
                child.terminate()
            child.wait(timeout=self.config.graceful_shutdown_seconds)
        except Exception:
            self.events.write("CHILD_FORCE_KILL", child_pid=child.pid)
            child.kill()
            child.wait(timeout=10)

    def _should_stop(self) -> bool:
        return self._stop_requested or self.stop_request_path.is_file()

    def _sleep_interruptibly(self, seconds: float) -> None:
        deadline = self.monotonic() + seconds
        while not self._should_stop():
            remaining = deadline - self.monotonic()
            if remaining <= 0:
                return
            self.sleep(min(0.25, remaining))

    def _install_signal_handlers(self) -> None:
        def handler(signum, frame) -> None:  # noqa: ARG001
            self._stop_requested = True

        for candidate in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(candidate, handler)
            except (ValueError, OSError):
                pass
