"""Bounded local subprocess executor for Hedge research jobs.

Only command plans produced by :mod:`freqtrade.hedge.research.command_plan` are
accepted. Those plans target backtesting, optimization, and FreqAI research
entry points and never the live ``trade`` command.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from .command_plan import ResearchCommandPlan, build_command_plan
from .contracts import ResearchKind, ResearchRequest, ResearchState
from .progress import observe_metrics, observe_progress
from .resources import process_snapshot, system_snapshot
from .training import materialize_training_override, normalize_training_device

if TYPE_CHECKING:
    from .service import HedgeResearchService


_TRAINING_KINDS = frozenset(
    {
        ResearchKind.ML_TRAIN,
        ResearchKind.ML_EVAL,
        ResearchKind.RL_TRAIN,
        ResearchKind.RL_EVAL,
    }
)


@dataclass(frozen=True, slots=True)
class ResearchExecutorConfig:
    enabled: bool = True
    max_concurrent: int = 2
    max_gpu_jobs: int = 1
    max_cpu_training_jobs: int = 1
    cpu_threads_per_job: int = 4
    max_cpu_threads_total: int = 0
    default_training_device: str = "auto"
    poll_seconds: float = 1.0
    cancel_grace_seconds: float = 5.0
    max_log_bytes: int = 128 * 1024 * 1024
    gpu_device: int = 0
    min_gpu_free_mb: int = 1024
    max_memory_percent: float = 92.0
    max_cpu_percent: float = 99.0

    def _validate_concurrency_limits(self) -> None:
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        if self.max_gpu_jobs < 0:
            raise ValueError("max_gpu_jobs cannot be negative")
        if self.max_gpu_jobs > self.max_concurrent:
            raise ValueError("max_gpu_jobs cannot exceed max_concurrent")
        if self.max_cpu_training_jobs < 0:
            raise ValueError("max_cpu_training_jobs cannot be negative")
        if self.max_cpu_training_jobs > self.max_concurrent:
            raise ValueError("max_cpu_training_jobs cannot exceed max_concurrent")
        if self.cpu_threads_per_job < 1:
            raise ValueError("cpu_threads_per_job must be positive")
        if self.max_cpu_threads_total < 0:
            raise ValueError("max_cpu_threads_total cannot be negative")

    def _validate_polling_limits(self) -> None:
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if self.cancel_grace_seconds <= 0:
            raise ValueError("cancel_grace_seconds must be positive")
        if self.max_log_bytes < 1024:
            raise ValueError("max_log_bytes must be at least 1024")

    def _validate_resource_thresholds(self) -> None:
        if self.gpu_device < 0:
            raise ValueError("gpu_device cannot be negative")
        if self.min_gpu_free_mb < 0:
            raise ValueError("min_gpu_free_mb cannot be negative")
        if not 1.0 <= self.max_memory_percent <= 100.0:
            raise ValueError("max_memory_percent must be within [1, 100]")
        if not 1.0 <= self.max_cpu_percent <= 100.0:
            raise ValueError("max_cpu_percent must be within [1, 100]")

    def __post_init__(self) -> None:
        self._validate_concurrency_limits()
        self._validate_polling_limits()
        self._validate_resource_thresholds()
        normalize_training_device(self.default_training_device)


@dataclass(slots=True)
class _RunningProcess:
    job_id: str
    kind: ResearchKind
    plan: ResearchCommandPlan
    process: subprocess.Popen[str]
    started_monotonic: float
    log_path: Path
    cancel_requested: threading.Event = field(default_factory=threading.Event)
    timed_out: threading.Event = field(default_factory=threading.Event)
    last_progress: float = 0.0
    last_progress_update: float = 0.0
    resources: dict[str, Any] = field(default_factory=dict)
    log_limit_bytes: int = 128 * 1024 * 1024
    metric_steps: dict[str, int | None] = field(default_factory=dict)
    metric_times: dict[str, float] = field(default_factory=dict)
    paused_at: float | None = None
    paused_total_seconds: float = 0.0
    training_device: str = "none"
    cpu_threads: int = 0


class ResearchExecutionManager:
    """Small local scheduler with bounded CPU/GPU concurrency and process control."""

    def __init__(
        self,
        service: HedgeResearchService,
        *,
        project_root: Path,
        python_executable: str | None = None,
        config: ResearchExecutorConfig | None = None,
    ) -> None:
        self.service = service
        self.project_root = project_root.expanduser().resolve()
        self.python_executable = python_executable or sys.executable
        self.config = config or ResearchExecutorConfig()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._pending: deque[str] = deque()
        self._starting: set[str] = set()
        self._plans: dict[str, ResearchCommandPlan] = {}
        self._training_devices: dict[str, str] = {}
        self._cpu_threads: dict[str, int] = {}
        self._active: dict[str, _RunningProcess] = {}
        self._last_runtime: dict[str, dict[str, Any]] = {}
        self._paused = False
        self._stopping = False
        self._system: dict[str, Any] = {}
        self._last_gpu_sample = 0.0
        self._gpu_cache: list[dict[str, Any]] = []
        self._scheduler = threading.Thread(
            target=self._scheduler_loop,
            name="hedge-research-scheduler",
            daemon=True,
        )
        self._monitor = threading.Thread(
            target=self._monitor_loop,
            name="hedge-research-monitor",
            daemon=True,
        )
        if self.config.enabled:
            self._scheduler.start()
            self._monitor.start()

    def _resolve_request_inputs(
        self,
        request: ResearchRequest,
    ) -> tuple[dict[str, Any], Path, str]:
        parameters = request.parameters
        config_value = str(parameters.get("config_path", "")).strip()
        strategy = str(parameters.get("strategy", "")).strip()
        if not config_value:
            raise ValueError("executable research job requires parameters.config_path")
        if not strategy:
            raise ValueError("executable research job requires parameters.strategy")
        config_path = Path(config_value).expanduser()
        if not config_path.is_absolute():
            config_path = self.project_root / config_path
        return parameters, config_path, strategy

    @staticmethod
    def _resolve_budget_limits(
        request: ResearchRequest,
    ) -> tuple[int | None, int | None]:
        parameters = request.parameters
        trials = parameters.get("trials")
        workers = parameters.get("workers")
        trial_count = (
            None
            if trials is None
            else min(int(trials), request.budget.max_trials)
        )
        worker_count = (
            None
            if workers is None
            else min(int(workers), request.budget.max_workers)
        )
        if request.kind is ResearchKind.OPTIMIZATION:
            trial_count = (
                request.budget.max_trials
                if trial_count is None
                else trial_count
            )
            worker_count = (
                request.budget.max_workers
                if worker_count is None
                else worker_count
            )
        return trial_count, worker_count

    def _resolve_replay_overlay(
        self,
        request: ResearchRequest,
    ) -> list[Path]:
        replay_overlay = str(
            request.parameters.get("replay_overlay_relative_path", "")
        ).strip()
        if not replay_overlay:
            return []
        overlay_path = self.service.workspace.resolve(replay_overlay)
        if not overlay_path.is_file():
            raise ValueError("optimization replay overlay does not exist")
        return [overlay_path]

    def _materialize_training_config(
        self,
        job_id: str,
        request: ResearchRequest,
    ) -> Path | None:
        if request.kind not in _TRAINING_KINDS:
            return None
        materialized = materialize_training_override(
            self.service.workspace,
            request,
            job_id=job_id,
            resolved_device=self._training_devices.get(job_id, "auto"),
            cpu_threads=self._cpu_threads.get(job_id),
        )
        try:
            self.service.attach_file(
                job_id,
                materialized.relative_path,
                media_type="application/json",
            )
        except ValueError:
            pass
        return materialized.path

    @staticmethod
    def _assert_safe_plan(plan: ResearchCommandPlan) -> None:
        if plan.exchange_write_enabled:
            raise RuntimeError(
                "research command plan unexpectedly enables exchange writes"
            )
        if "trade" in plan.argv:
            raise RuntimeError(
                "live trade command is forbidden in research executor"
            )

    def _request_plan(
        self,
        job_id: str,
        request: ResearchRequest,
        *,
        output_directory: Path,
    ) -> ResearchCommandPlan:
        parameters, config_path, strategy = self._resolve_request_inputs(request)
        trial_count, worker_count = self._resolve_budget_limits(request)
        extra_config_list = self._resolve_replay_overlay(request)
        training_config = self._materialize_training_config(job_id, request)
        if training_config is not None:
            extra_config_list.append(training_config)

        plan = build_command_plan(
            request.kind,
            config_path=config_path,
            strategy=strategy,
            timerange=str(parameters.get("timerange", "")),
            trials=trial_count,
            workers=worker_count,
            python_executable=self.python_executable,
            output_directory=output_directory,
            extra_config_paths=tuple(extra_config_list),
        )
        self._assert_safe_plan(plan)
        return plan

    def enqueue(self, job_id: str) -> dict[str, Any]:
        if not self.config.enabled:
            raise RuntimeError("research executor is disabled")
        job = self.service.jobs.get(job_id)
        if job.state is not ResearchState.QUEUED:
            raise ValueError("only QUEUED research jobs can be executed")
        output_directory = self.service.workspace.resolve(f"jobs/{job_id}/outputs")
        output_directory.mkdir(parents=True, exist_ok=True)
        if job.request.kind in _TRAINING_KINDS:
            self._training_devices[job_id] = self._resolve_training_device(job.request)
            self._cpu_threads[job_id] = self._resolve_cpu_threads(job.request)
        plan = self._request_plan(
            job_id,
            job.request,
            output_directory=output_directory,
        )
        with self._condition:
            if job_id in self._pending or job_id in self._active:
                raise ValueError("research job is already queued for execution")
            self._plans[job_id] = plan
            self._pending.append(job_id)
            self.service.note(job_id, "queued for local execution")
            self._condition.notify_all()
        return self.runtime(job_id)

    def _resolve_training_device(self, request: ResearchRequest) -> str:
        requested = normalize_training_device(
            request.parameters.get("training_device", self.config.default_training_device)
        )
        if requested != "auto":
            return requested
        if self.config.max_gpu_jobs <= 0:
            return "cpu"
        snapshot = self._system or system_snapshot(include_gpu=True)
        device = int(request.parameters.get("gpu_device", self.config.gpu_device))
        for row in snapshot.get("gpus", []):
            try:
                if int(row.get("index", -1)) == device:
                    return "cuda"
            except (TypeError, ValueError):
                continue
        return "cpu"

    def _resolve_cpu_threads(self, request: ResearchRequest) -> int:
        raw = request.parameters.get("cpu_threads", self.config.cpu_threads_per_job)
        threads = int(raw)
        if threads < 1:
            raise ValueError("cpu_threads must be positive")
        logical = psutil.cpu_count(logical=True) or threads
        return min(threads, logical)

    def _cpu_thread_limit(self) -> int:
        if self.config.max_cpu_threads_total > 0:
            return self.config.max_cpu_threads_total
        logical = psutil.cpu_count(logical=True) or self.config.cpu_threads_per_job
        return max(1, logical - 1)

    def _active_gpu_count(self) -> int:
        return sum(item.training_device == "cuda" for item in self._active.values())

    def _active_cpu_training_count(self) -> int:
        return sum(item.training_device == "cpu" for item in self._active.values())

    def _active_cpu_threads(self) -> int:
        return sum(
            item.cpu_threads
            for item in self._active.values()
            if item.training_device == "cpu"
        )

    def _ordered_pending_locked(self) -> list[str]:
        positions = {job_id: index for index, job_id in enumerate(self._pending)}
        return sorted(
            self._pending,
            key=lambda job_id: (
                -self.service.jobs.get(job_id).request.priority,
                positions[job_id],
            ),
        )

    def _gpu_row(self, device: int) -> dict[str, Any] | None:
        for row in self._system.get("gpus", []):
            try:
                if int(row.get("index", -1)) == device:
                    return row
            except (TypeError, ValueError):
                continue
        return None

    def _dependency_block_reason(self, job_id: str) -> str:
        job = self.service.jobs.get(job_id)
        dependency_id = str(
            job.request.parameters.get("depends_on_job_id", "")
        ).strip()
        if not dependency_id:
            return ""
        try:
            dependency = self.service.jobs.get(dependency_id)
        except KeyError:
            return f"dependency {dependency_id} is missing"
        if dependency.state is not ResearchState.SUCCEEDED:
            return f"dependency {dependency_id} is {dependency.state.value}"
        return ""

    def _system_block_reason(self) -> str:
        if len(self._active) >= self.config.max_concurrent:
            return "concurrency limit"
        memory_percent = self._system.get("memory_percent")
        if (
            memory_percent is not None
            and float(memory_percent) >= self.config.max_memory_percent
        ):
            return (
                f"system memory {float(memory_percent):.1f}% >= limit "
                f"{self.config.max_memory_percent:.1f}%"
            )
        cpu_percent = self._system.get("cpu_percent")
        if (
            cpu_percent is not None
            and float(cpu_percent) >= self.config.max_cpu_percent
        ):
            return (
                f"system cpu {float(cpu_percent):.1f}% >= limit "
                f"{self.config.max_cpu_percent:.1f}%"
            )
        return ""

    def _cpu_training_block_reason(self, job_id: str) -> str:
        if self._active_cpu_training_count() >= self.config.max_cpu_training_jobs:
            return "cpu training slot limit"
        requested_threads = self._cpu_threads.get(
            job_id,
            self.config.cpu_threads_per_job,
        )
        thread_limit = self._cpu_thread_limit()
        if self._active_cpu_threads() + requested_threads > thread_limit:
            return (
                f"cpu thread budget {self._active_cpu_threads()}+"
                f"{requested_threads} > {thread_limit}"
            )
        return ""

    def _gpu_training_block_reason(self, job_id: str) -> str:
        if self._active_gpu_count() >= self.config.max_gpu_jobs:
            return "gpu slot limit"
        job = self.service.jobs.get(job_id)
        device = int(
            job.request.parameters.get(
                "gpu_device",
                self.config.gpu_device,
            )
        )
        row = self._gpu_row(device)
        if row is None:
            return f"cuda requested but gpu {device} is unavailable"
        free_mb = float(row.get("memory_free_mb", 0.0))
        required_mb = int(
            job.request.parameters.get(
                "min_gpu_free_mb",
                self.config.min_gpu_free_mb,
            )
        )
        if free_mb < required_mb:
            return (
                f"gpu {device} free memory {free_mb:.0f} MB "
                f"< required {required_mb} MB"
            )
        return ""

    def _resource_block_reason_locked(self, job_id: str) -> str:
        if self._paused:
            return "executor paused"
        pipeline_reason = self.service.pipeline_block_reason(job_id)
        if pipeline_reason:
            return pipeline_reason
        dependency_reason = self._dependency_block_reason(job_id)
        if dependency_reason:
            return dependency_reason
        system_reason = self._system_block_reason()
        if system_reason:
            return system_reason

        job = self.service.jobs.get(job_id)
        if job.request.kind not in _TRAINING_KINDS:
            return ""
        if self._training_devices.get(job_id, "cpu") == "cpu":
            return self._cpu_training_block_reason(job_id)
        return self._gpu_training_block_reason(job_id)

    def _next_startable_locked(self) -> str | None:
        for job_id in self._ordered_pending_locked():
            job = self.service.jobs.get(job_id)
            dependency_id = str(job.request.parameters.get("depends_on_job_id", "")).strip()
            if dependency_id:
                try:
                    dependency = self.service.jobs.get(dependency_id)
                except KeyError:
                    self._pending.remove(job_id)
                    self._plans.pop(job_id, None)
                    self.service.fail(job_id, f"dependency {dependency_id} is missing")
                    continue
                if dependency.terminal and dependency.state is not ResearchState.SUCCEEDED:
                    self._pending.remove(job_id)
                    self._plans.pop(job_id, None)
                    self.service.fail(
                        job_id,
                        f"dependency {dependency_id} ended as {dependency.state.value}",
                    )
                    continue
            if self._resource_block_reason_locked(job_id):
                continue
            self._pending.remove(job_id)
            self._starting.add(job_id)
            return job_id
        return None

    def _scheduler_loop(self) -> None:
        while True:
            with self._condition:
                while not self._stopping:
                    job_id = self._next_startable_locked()
                    if job_id is not None:
                        break
                    self._condition.wait(timeout=self.config.poll_seconds)
                if self._stopping:
                    return
            self._start(job_id)

    def _process_environment(self, job_id: str) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["HEDGE_RESEARCH_JOB_ID"] = job_id
        env["HEDGE_RESEARCH_LOCAL_ONLY"] = "1"
        env["HEDGE_RESEARCH_LIVE_WRITE"] = "0"
        job = self.service.jobs.get(job_id)
        if job.request.kind in _TRAINING_KINDS:
            training_device = self._training_devices.get(job_id, "cpu")
            env["HEDGE_RESEARCH_TRAINING_DEVICE"] = training_device
            if training_device == "cuda":
                device = int(job.request.parameters.get("gpu_device", self.config.gpu_device))
                env["CUDA_VISIBLE_DEVICES"] = str(device)
                env["HEDGE_RESEARCH_GPU_DEVICE"] = str(device)
            else:
                threads = self._cpu_threads.get(job_id, self.config.cpu_threads_per_job)
                env["CUDA_VISIBLE_DEVICES"] = ""
                env["HEDGE_RESEARCH_CPU_THREADS"] = str(threads)
                for key in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                ):
                    env[key] = str(threads)
        return env

    @staticmethod
    def _creation_options() -> dict[str, Any]:
        if os.name == "nt":
            return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        return {"start_new_session": True}

    def _start(self, job_id: str) -> None:
        plan = self._plans[job_id]
        log_path = self.service.workspace.resolve(f"jobs/{job_id}/execution.log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        job = self.service.jobs.get(job_id)
        if job.state is ResearchState.CANCELED:
            with self._condition:
                self._starting.discard(job_id)
                self._plans.pop(job_id, None)
                self._condition.notify_all()
            return
        try:
            self.service.begin(job_id)
            process = subprocess.Popen(
                list(plan.argv),
                cwd=self.project_root,
                env=self._process_environment(job_id),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **self._creation_options(),
            )
        except (OSError, ValueError) as exc:
            current = self.service.jobs.get(job_id)
            if not current.terminal:
                self.service.fail(job_id, f"process start failed: {exc}")
            with self._condition:
                self._starting.discard(job_id)
                self._plans.pop(job_id, None)
                self._condition.notify_all()
            return

        job = self.service.jobs.get(job_id)
        used_bytes = sum(item.size for item in job.artifacts)
        remaining_budget = max(1024, job.request.budget.max_artifact_bytes - used_bytes)
        log_limit = min(self.config.max_log_bytes, max(1024, remaining_budget // 2))
        running = _RunningProcess(
            job_id=job_id,
            kind=job.request.kind,
            plan=plan,
            process=process,
            started_monotonic=time.monotonic(),
            log_path=log_path,
            log_limit_bytes=log_limit,
            training_device=self._training_devices.get(job_id, "none"),
            cpu_threads=self._cpu_threads.get(job_id, 0),
        )
        with self._condition:
            self._starting.discard(job_id)
            self._active[job_id] = running
            self._last_runtime[job_id] = self._runtime_row(running, state="RUNNING")
        if self.service.jobs.get(job_id).state is ResearchState.CANCELED:
            running.cancel_requested.set()
            self._terminate_process_tree(
                process.pid,
                grace_seconds=self.config.cancel_grace_seconds,
            )
        else:
            self.service.progress(job_id, 0.0, message=f"process started pid={process.pid}")
        thread = threading.Thread(
            target=self._watch_process,
            args=(running,),
            name=f"hedge-research-{job_id[:8]}",
            daemon=True,
        )
        thread.start()

    @staticmethod
    def _append_log(
        running: _RunningProcess,
        handle: Any,
        line: str,
        *,
        bytes_written: int,
    ) -> int:
        encoded_len = len(line.encode("utf-8", errors="replace"))
        if bytes_written + encoded_len > running.log_limit_bytes:
            return bytes_written
        handle.write(line)
        handle.flush()
        return bytes_written + encoded_len

    def _observe_output_line(self, running: _RunningProcess, line: str) -> None:
        job = self.service.jobs.get(running.job_id)
        now = time.monotonic()
        observation = observe_progress(line, max_trials=job.request.budget.max_trials)
        if observation is not None and observation.progress >= running.last_progress:
            if (
                observation.progress > running.last_progress
                or now - running.last_progress_update >= 2.0
            ):
                running.last_progress = observation.progress
                running.last_progress_update = now
                try:
                    self.service.progress(
                        running.job_id,
                        observation.progress,
                        message=observation.message,
                    )
                except ValueError:
                    pass

        for metric in observe_metrics(line):
            previous_step = running.metric_steps.get(metric.name)
            previous_time = running.metric_times.get(metric.name, 0.0)
            if metric.step is not None and previous_step == metric.step:
                continue
            if metric.step is None and now - previous_time < 2.0:
                continue
            try:
                self.service.metric(
                    running.job_id,
                    metric.name,
                    metric.value,
                    step=metric.step,
                )
            except ValueError:
                continue
            running.metric_steps[metric.name] = metric.step
            running.metric_times[metric.name] = now

    def _attach_generated_outputs(self, job_id: str) -> list[str]:
        output_root = self.service.workspace.resolve(f"jobs/{job_id}/outputs")
        if not output_root.is_dir():
            return []
        attached: list[str] = []
        for path in sorted(output_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.service.workspace.root).as_posix()
            media_type = (
                "application/json"
                if path.suffix.lower() == ".json"
                else "application/octet-stream"
            )
            if path.suffix.lower() == ".csv":
                media_type = "text/csv; charset=utf-8"
            try:
                self.service.attach_file(job_id, relative, media_type=media_type)
            except ValueError:
                break
            attached.append(relative)
        return attached

    def _stream_process_log(self, running: _RunningProcess) -> None:
        bytes_written = 0
        with running.log_path.open("w", encoding="utf-8", newline="") as handle:
            stdout = running.process.stdout
            if stdout is None:
                return
            for line in stdout:
                bytes_written = self._append_log(
                    running,
                    handle,
                    line,
                    bytes_written=bytes_written,
                )
                self._observe_output_line(running, line)

    def _register_experiment_best_effort(self, job_id: str) -> None:
        try:
            self.service.register_experiment(job_id)
        except (OSError, TypeError, ValueError):
            pass

    def _write_replay_error(
        self,
        job_id: str,
        exc: Exception,
    ) -> None:
        try:
            self.service.workspace.write_json(
                f"optimization_replay/{job_id}-error.json",
                {"error": str(exc), "type": type(exc).__name__},
                max_bytes=256 * 1024,
            )
        except (OSError, ValueError):
            pass

    def _auto_replay_optimization(self, running: _RunningProcess) -> None:
        if running.kind is not ResearchKind.OPTIMIZATION:
            return
        request = self.service.jobs.get(running.job_id).request
        if not bool(request.parameters.get("auto_replay_best", False)):
            return
        try:
            replay_top_n = max(
                1,
                int(request.parameters.get("replay_top_n", 1)),
            )
            replay_timerange = str(
                request.parameters.get("replay_timerange", "")
            )
            if replay_top_n > 1:
                self.service.replay_top_optimization(
                    running.job_id,
                    limit=min(replay_top_n, 20),
                    timerange=replay_timerange,
                    auto_execute=True,
                )
            else:
                self.service.replay_best_optimization(
                    running.job_id,
                    timerange=replay_timerange,
                    auto_execute=True,
                )
        except (
            FileNotFoundError,
            KeyError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self._write_replay_error(running.job_id, exc)

    def _complete_successful_process(
        self,
        running: _RunningProcess,
        *,
        active_duration: float,
        wall_duration: float,
    ) -> None:
        outputs = self._attach_generated_outputs(running.job_id)
        output_root = self.service.workspace.resolve(
            f"jobs/{running.job_id}/outputs"
        )
        extracted_metrics = self.service.ingest_output_metrics(
            running.job_id,
            output_root,
        )
        self.service.complete(
            running.job_id,
            {
                "returncode": 0,
                "active_duration_seconds": round(active_duration, 3),
                "wall_duration_seconds": round(wall_duration, 3),
                "command": list(running.plan.argv),
                "log": f"jobs/{running.job_id}/execution.log",
                "outputs": outputs,
                "extracted_metrics": extracted_metrics,
                "training_device": running.training_device,
                "cpu_threads": running.cpu_threads,
                "exchange_write_enabled": False,
            },
        )
        self._register_experiment_best_effort(running.job_id)
        self._auto_replay_optimization(running)

    def _finalize_process_result(
        self,
        running: _RunningProcess,
        *,
        returncode: int,
        active_duration: float,
        wall_duration: float,
    ) -> None:
        if running.cancel_requested.is_set():
            self.service.cancel(running.job_id)
            return
        if running.timed_out.is_set():
            self.service.fail(
                running.job_id,
                f"execution timed out after {active_duration:.1f}s active time",
            )
            return
        if returncode == 0:
            self._complete_successful_process(
                running,
                active_duration=active_duration,
                wall_duration=wall_duration,
            )
            return
        self.service.fail(
            running.job_id,
            f"research process exited with code {returncode}",
        )

    def _record_executor_failure(
        self,
        running: _RunningProcess,
        exc: Exception,
    ) -> None:
        try:
            job = self.service.jobs.get(running.job_id)
            if not job.terminal:
                self.service.fail(
                    running.job_id,
                    f"executor failure: {type(exc).__name__}: {exc}",
                )
        except Exception:
            pass

    def _cleanup_running(self, running: _RunningProcess) -> None:
        with self._condition:
            self._last_runtime[running.job_id] = self._runtime_row(
                running,
                state=self.service.jobs.get(running.job_id).state.value,
            )
            self._active.pop(running.job_id, None)
            self._plans.pop(running.job_id, None)
            self._training_devices.pop(running.job_id, None)
            self._cpu_threads.pop(running.job_id, None)
            self._condition.notify_all()

    def _watch_process(self, running: _RunningProcess) -> None:
        try:
            self._stream_process_log(running)
            returncode = running.process.wait()
            finished = time.monotonic()
            wall_duration = max(
                0.0,
                finished - running.started_monotonic,
            )
            active_duration = self._active_elapsed(running, now=finished)
            self.service.attach_file(
                running.job_id,
                f"jobs/{running.job_id}/execution.log",
                media_type="text/plain; charset=utf-8",
            )
            self._finalize_process_result(
                running,
                returncode=returncode,
                active_duration=active_duration,
                wall_duration=wall_duration,
            )
        except Exception as exc:
            self._record_executor_failure(running, exc)
        finally:
            self._cleanup_running(running)

    @staticmethod
    def _active_elapsed(running: _RunningProcess, *, now: float | None = None) -> float:
        current = time.monotonic() if now is None else now
        paused = running.paused_total_seconds
        if running.paused_at is not None:
            paused += max(0.0, current - running.paused_at)
        return max(0.0, current - running.started_monotonic - paused)

    @staticmethod
    def _set_process_tree_suspended(pid: int, *, suspended: bool) -> None:
        try:
            root = psutil.Process(pid)
        except psutil.NoSuchProcess as exc:
            raise RuntimeError("research process is no longer running") from exc
        processes = root.children(recursive=True)
        processes.append(root)
        ordered = list(reversed(processes)) if suspended else processes
        changed: list[psutil.Process] = []
        try:
            for process in ordered:
                if suspended:
                    process.suspend()
                else:
                    process.resume()
                changed.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            if suspended:
                for process in reversed(changed):
                    try:
                        process.resume()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
            action = "pause" if suspended else "resume"
            raise RuntimeError(
                f"unable to {action} research process tree"
            ) from exc

    @staticmethod
    def _terminate_process_tree(pid: int, *, grace_seconds: float) -> None:
        try:
            root = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return
        processes = root.children(recursive=True)
        processes.append(root)
        for process in reversed(processes):
            try:
                process.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        _, alive = psutil.wait_procs(processes, timeout=grace_seconds)
        for process in alive:
            try:
                process.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

    def pause_job(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            running = self._active.get(job_id)
        if running is None:
            raise ValueError("only an active research job can be paused")
        if self.service.jobs.get(job_id).state is ResearchState.PAUSED:
            return self.runtime(job_id)
        self._set_process_tree_suspended(running.process.pid, suspended=True)
        running.paused_at = time.monotonic()
        self.service.mark_paused(job_id, message="process suspended")
        with self._condition:
            self._last_runtime[job_id] = self._runtime_row(
                running,
                state=ResearchState.PAUSED.value,
            )
        return self.runtime(job_id)

    def resume_job(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            running = self._active.get(job_id)
        if running is None:
            raise ValueError("only an active research job can be resumed")
        if self.service.jobs.get(job_id).state is not ResearchState.PAUSED:
            raise ValueError("research job is not paused")
        self._set_process_tree_suspended(running.process.pid, suspended=False)
        now = time.monotonic()
        if running.paused_at is not None:
            running.paused_total_seconds += max(0.0, now - running.paused_at)
        running.paused_at = None
        self.service.mark_resumed(job_id, message="process resumed")
        with self._condition:
            self._last_runtime[job_id] = self._runtime_row(
                running,
                state=ResearchState.RUNNING.value,
            )
        return self.runtime(job_id)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            if job_id in self._pending:
                self._pending.remove(job_id)
                self._plans.pop(job_id, None)
                self._training_devices.pop(job_id, None)
                self._cpu_threads.pop(job_id, None)
                self.service.cancel(job_id)
                self._last_runtime[job_id] = {
                    "job_id": job_id,
                    "state": ResearchState.CANCELED.value,
                    "queued": False,
                    "active": False,
                }
                self._condition.notify_all()
                return self.runtime(job_id)
            if job_id in self._starting:
                self.service.cancel(job_id)
                return {
                    "job_id": job_id,
                    "state": ResearchState.CANCELED.value,
                    "queued": False,
                    "active": False,
                    "starting": True,
                }
            running = self._active.get(job_id)
        if running is None:
            job = self.service.jobs.get(job_id)
            if job.terminal:
                return self.runtime(job_id)
            return {"job_id": job_id, "state": job.state.value, "active": False, "queued": False}
        running.cancel_requested.set()
        try:
            self.service.note(job_id, "cancel requested")
        except ValueError:
            pass
        self._terminate_process_tree(
            running.process.pid,
            grace_seconds=self.config.cancel_grace_seconds,
        )
        return self.runtime(job_id)

    def pause(self) -> dict[str, Any]:
        with self._condition:
            self._paused = True
        return self.snapshot()

    def resume(self) -> dict[str, Any]:
        with self._condition:
            self._paused = False
            self._condition.notify_all()
        return self.snapshot()

    def _monitor_loop(self) -> None:
        while True:
            with self._condition:
                if self._stopping:
                    return
                active = tuple(self._active.values())
            now = time.monotonic()
            include_gpu = now - self._last_gpu_sample >= 5.0
            sampled = system_snapshot(include_gpu=include_gpu)
            if include_gpu:
                self._gpu_cache = list(sampled.get("gpus", []))
                self._last_gpu_sample = now
            else:
                sampled["gpus"] = list(self._gpu_cache)
            self._system = sampled
            for running in active:
                process_row = process_snapshot(running.process.pid)
                if process_row is not None:
                    running.resources = process_row.to_dict()
                job = self.service.jobs.get(running.job_id)
                active_elapsed = self._active_elapsed(running, now=now)
                if active_elapsed > job.request.budget.max_seconds:
                    if not running.timed_out.is_set():
                        running.timed_out.set()
                        self.service.note(running.job_id, "time budget exceeded; stopping process")
                        self._terminate_process_tree(
                            running.process.pid,
                            grace_seconds=self.config.cancel_grace_seconds,
                        )
                with self._condition:
                    self._last_runtime[running.job_id] = self._runtime_row(
                        running,
                        state=self.service.jobs.get(running.job_id).state.value,
                    )
            time.sleep(self.config.poll_seconds)

    def _runtime_row(self, running: _RunningProcess, *, state: str) -> dict[str, Any]:
        return {
            "job_id": running.job_id,
            "state": state,
            "queued": False,
            "active": running.process.poll() is None,
            "pid": running.process.pid,
            "kind": running.kind.value,
            "training_device": running.training_device,
            "cpu_threads": running.cpu_threads,
            "elapsed_seconds": round(self._active_elapsed(running), 3),
            "paused": running.paused_at is not None,
            "paused_seconds": round(running.paused_total_seconds + (
                0.0 if running.paused_at is None else max(0.0, time.monotonic() - running.paused_at)
            ), 3),
            "resources": dict(running.resources),
            "command": list(running.plan.argv),
            "cancel_requested": running.cancel_requested.is_set(),
            "timed_out": running.timed_out.is_set(),
        }

    def runtime(self, job_id: str) -> dict[str, Any]:
        with self._condition:
            if job_id in self._pending:
                return {
                    "job_id": job_id,
                    "state": ResearchState.QUEUED.value,
                    "queued": True,
                    "active": False,
                    "queue_position": self._ordered_pending_locked().index(job_id) + 1,
                    "blocked_reason": self._resource_block_reason_locked(job_id),
                    "training_device": self._training_devices.get(job_id, "none"),
                    "cpu_threads": self._cpu_threads.get(job_id, 0),
                    "command": list(self._plans[job_id].argv),
                }
            if job_id in self._starting:
                job = self.service.jobs.get(job_id)
                return {
                    "job_id": job_id,
                    "state": job.state.value,
                    "queued": False,
                    "active": False,
                    "starting": True,
                    "training_device": self._training_devices.get(job_id, "none"),
                    "cpu_threads": self._cpu_threads.get(job_id, 0),
                    "command": list(self._plans[job_id].argv),
                }
            running = self._active.get(job_id)
            if running is not None:
                state = self.service.jobs.get(job_id).state.value
                return self._runtime_row(running, state=state)
            if job_id in self._last_runtime:
                return dict(self._last_runtime[job_id])
        job = self.service.jobs.get(job_id)
        return {
            "job_id": job_id,
            "state": job.state.value,
            "queued": False,
            "active": False,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            pending = list(self._pending)
            active = [
                self._runtime_row(item, state=self.service.jobs.get(item.job_id).state.value)
                for item in self._active.values()
            ]
            return {
                "enabled": self.config.enabled,
                "paused": self._paused,
                "max_concurrent": self.config.max_concurrent,
                "max_gpu_jobs": self.config.max_gpu_jobs,
                "max_cpu_training_jobs": self.config.max_cpu_training_jobs,
                "cpu_threads_per_job": self.config.cpu_threads_per_job,
                "max_cpu_threads_total": self.config.max_cpu_threads_total,
                "effective_cpu_thread_limit": self._cpu_thread_limit(),
                "default_training_device": self.config.default_training_device,
                "gpu_device": self.config.gpu_device,
                "min_gpu_free_mb": self.config.min_gpu_free_mb,
                "max_memory_percent": self.config.max_memory_percent,
                "max_cpu_percent": self.config.max_cpu_percent,
                "queued": self._ordered_pending_locked(),
                "queued_detail": [
                    {
                        "job_id": job_id,
                        "priority": self.service.jobs.get(job_id).request.priority,
                        "training_device": self._training_devices.get(job_id, "none"),
                        "cpu_threads": self._cpu_threads.get(job_id, 0),
                        "blocked_reason": self._resource_block_reason_locked(job_id),
                    }
                    for job_id in self._ordered_pending_locked()
                ],
                "queued_count": len(pending),
                "active": active,
                "active_count": len(active),
                "system": dict(self._system),
                "live_order_write": False,
            }

    def stop(self, *, terminate_running: bool = False) -> None:
        with self._condition:
            self._stopping = True
            active = tuple(self._active.values())
            self._condition.notify_all()
        if terminate_running:
            for running in active:
                running.cancel_requested.set()
                self._terminate_process_tree(
                    running.process.pid,
                    grace_seconds=self.config.cancel_grace_seconds,
                )
