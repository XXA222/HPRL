"""Unified local research control plane for backtest, optimization, ML, and RL."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from statistics import pstdev
from typing import Any

from .contracts import (
    ResearchArtifact,
    ResearchBudget,
    ResearchJob,
    ResearchKind,
    ResearchMetric,
    ResearchRequest,
    ResearchState,
)
from .dashboard import dashboard_summary
from .experiments import ExperimentRegistry
from .jobs import ResearchJobStore
from .optimization_replay import (
    materialize_best_parameter_overlay,
    materialize_parameter_overlay,
    ranked_candidates,
)
from .pipeline import ResearchPipelineManager, ResearchPipelineSpec
from .promotion import PromotionPolicy, build_promotion_record, evaluate_promotion
from .results import extract_metrics_from_directory
from .tensorboard import read_tensorboard_scalars
from .walkforward import WalkForwardFold, build_walk_forward_folds, new_group_id
from .workspace import ResearchWorkspace


class HedgeResearchService:
    """Fail-closed orchestration facade. It never submits exchange orders."""

    def __init__(
        self,
        workspace_root: Path,
        *,
        job_capacity: int = 1000,
        user_data_dir: Path | None = None,
    ) -> None:
        self.workspace = ResearchWorkspace(workspace_root)
        self.jobs = ResearchJobStore(capacity=job_capacity)
        resolved_user_data = (user_data_dir or workspace_root.parent).expanduser().resolve()
        self.experiments = ExperimentRegistry(self.workspace, user_data_dir=resolved_user_data)
        self.recovery_errors: list[str] = []
        self.executor: Any | None = None
        self.pipeline_manager: ResearchPipelineManager | None = None
        self._recover_jobs()

    @staticmethod
    def _parse_datetime(value: object) -> datetime | None:
        if value in {None, ""}:
            return None
        parsed = datetime.fromisoformat(str(value))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)

    def _request_from_payload(self, payload: dict[str, Any]) -> ResearchRequest:
        budget = payload.get("budget", {})
        if not isinstance(budget, dict):
            raise TypeError("recovered research budget must be an object")
        return ResearchRequest(
            kind=ResearchKind(str(payload["kind"])),
            name=str(payload["name"]),
            parameters=dict(payload.get("parameters", {})),
            tags=tuple(str(item) for item in payload.get("tags", ())),
            priority=int(payload.get("priority", 50)),
            budget=ResearchBudget(
                max_seconds=int(budget.get("max_seconds", 3600)),
                max_trials=int(budget.get("max_trials", 100)),
                max_workers=int(budget.get("max_workers", 1)),
                max_artifact_bytes=int(budget.get("max_artifact_bytes", 256 * 1024 * 1024)),
            ),
        )

    @staticmethod
    def _artifact_from_payload(payload: dict[str, Any]) -> ResearchArtifact:
        return ResearchArtifact(
            name=str(payload["name"]),
            relative_path=str(payload["relative_path"]),
            media_type=str(payload["media_type"]),
            size=int(payload["size"]),
            sha256=str(payload.get("sha256", "")),
        )

    def _job_from_payloads(
        self,
        request_payload: dict[str, Any],
        state_payload: dict[str, Any],
    ) -> ResearchJob:
        request = self._request_from_payload(request_payload)
        state = ResearchState(str(state_payload.get("state", "QUEUED")))
        message = str(state_payload.get("message", ""))
        updated_at = self._parse_datetime(state_payload.get("updated_at")) or datetime.now(UTC)
        finished_at = self._parse_datetime(state_payload.get("finished_at"))
        revision = int(state_payload.get("revision", 0))
        if state in {ResearchState.RUNNING, ResearchState.PAUSED}:
            now = datetime.now(UTC)
            state = ResearchState.FAILED
            message = "interrupted by research service restart"
            updated_at = now
            finished_at = now
            revision += 1
        return ResearchJob(
            job_id=str(state_payload["job_id"]),
            request=request,
            state=state,
            progress=float(state_payload.get("progress", 0.0)),
            created_at=self._parse_datetime(state_payload.get("created_at")) or datetime.now(UTC),
            updated_at=updated_at,
            started_at=self._parse_datetime(state_payload.get("started_at")),
            finished_at=finished_at,
            message=message,
            metrics=[
                ResearchMetric(str(item["name"]), float(item["value"]), item.get("step"))
                for item in state_payload.get("metrics", ())
            ],
            artifacts=[
                self._artifact_from_payload(item)
                for item in state_payload.get("artifacts", ())
            ],
            revision=revision,
        )

    def _recover_jobs(self) -> None:
        jobs_root = self.workspace.root / "jobs"
        if not jobs_root.is_dir():
            return
        for request_path in sorted(jobs_root.glob("*/request.json")):
            job_id = request_path.parent.name
            state_path = request_path.parent / "state.json"
            if not state_path.is_file():
                continue
            try:
                request_payload = self.workspace.read_json(f"jobs/{job_id}/request.json")
                state_payload = self.workspace.read_json(f"jobs/{job_id}/state.json")
                if not isinstance(request_payload, dict) or not isinstance(state_payload, dict):
                    raise TypeError("recovered research metadata must be objects")
                job = self._job_from_payloads(request_payload, state_payload)
                if job.job_id != job_id:
                    raise ValueError("recovered research job id does not match directory")
                self.jobs.restore(job)
                if str(state_payload.get("state")) in {
                    ResearchState.RUNNING.value,
                    ResearchState.PAUSED.value,
                }:
                    self._persist_state(job_id)
            except (KeyError, TypeError, ValueError, OSError) as exc:
                self.recovery_errors.append(f"{job_id}: {exc}")

    def _persist_state(self, job_id: str) -> None:
        snapshot = self.jobs.get(job_id).snapshot()
        self.workspace.write_json(f"jobs/{job_id}/state.json", snapshot, max_bytes=1024 * 1024)

    def submit(self, request: ResearchRequest) -> dict[str, Any]:
        job = self.jobs.create(request)
        try:
            artifact = self.workspace.write_json(
                f"jobs/{job.job_id}/request.json",
                {
                    "job_id": job.job_id,
                    "kind": request.kind.value,
                    "name": request.name,
                    "parameters": request.parameters,
                    "tags": list(request.tags),
                    "priority": request.priority,
                    "budget": {
                        "max_seconds": request.budget.max_seconds,
                        "max_trials": request.budget.max_trials,
                        "max_workers": request.budget.max_workers,
                        "max_artifact_bytes": request.budget.max_artifact_bytes,
                    },
                },
                max_bytes=request.budget.max_artifact_bytes,
            )
            self.jobs.add_artifact(job.job_id, artifact)
            self._persist_state(job.job_id)
            return self.jobs.get(job.job_id).snapshot()
        except Exception:
            try:
                self.jobs.discard_queued(job.job_id)
            except (KeyError, ValueError):
                pass
            raise

    def configure_executor(
        self,
        *,
        project_root: Path,
        python_executable: str | None = None,
        executor_config: Any | None = None,
    ) -> None:
        """Attach the bounded local research executor to this service."""
        from .execution import ResearchExecutionManager

        if self.executor is not None:
            raise RuntimeError("research executor is already configured")
        self.executor = ResearchExecutionManager(
            self,
            project_root=project_root,
            python_executable=python_executable,
            config=executor_config,
        )
        self.pipeline_manager = ResearchPipelineManager(self)

    def submit_pipeline(
        self,
        spec: ResearchPipelineSpec,
        *,
        auto_start: bool = True,
    ) -> dict[str, Any]:
        if self.pipeline_manager is None:
            raise RuntimeError("research pipeline manager is not configured")
        return self.pipeline_manager.create(spec, auto_start=auto_start)

    def list_pipelines(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        if self.pipeline_manager is None:
            return ()
        return self.pipeline_manager.list(limit=limit)

    def pipeline(self, pipeline_id: str) -> dict[str, Any]:
        if self.pipeline_manager is None:
            raise RuntimeError("research pipeline manager is not configured")
        return self.pipeline_manager.snapshot(pipeline_id)

    def start_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        if self.pipeline_manager is None:
            raise RuntimeError("research pipeline manager is not configured")
        return self.pipeline_manager.start(pipeline_id)

    def pause_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        if self.pipeline_manager is None:
            raise RuntimeError("research pipeline manager is not configured")
        return self.pipeline_manager.pause(pipeline_id)

    def resume_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        if self.pipeline_manager is None:
            raise RuntimeError("research pipeline manager is not configured")
        return self.pipeline_manager.resume(pipeline_id)

    def approve_pipeline_training(self, pipeline_id: str) -> dict[str, Any]:
        if self.pipeline_manager is None:
            raise RuntimeError("research pipeline manager is not configured")
        return self.pipeline_manager.approve_training(pipeline_id)

    def cancel_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        if self.pipeline_manager is None:
            raise RuntimeError("research pipeline manager is not configured")
        return self.pipeline_manager.cancel(pipeline_id)

    def reconsider_pipeline_promotion(
        self,
        pipeline_id: str,
        policy: PromotionPolicy,
    ) -> dict[str, Any]:
        if self.pipeline_manager is None:
            raise RuntimeError("research pipeline manager is not configured")
        return self.pipeline_manager.reconsider_promotion(pipeline_id, policy)

    def retry_pipeline(self, pipeline_id: str) -> dict[str, Any]:
        if self.pipeline_manager is None:
            raise RuntimeError("research pipeline manager is not configured")
        return self.pipeline_manager.retry(pipeline_id)

    def pipeline_block_reason(self, job_id: str) -> str:
        if self.pipeline_manager is None:
            return ""
        job = self.jobs.get(job_id)
        pipeline_id = str(job.request.parameters.get("pipeline_id", "")).strip()
        if not pipeline_id:
            return ""
        return self.pipeline_manager.block_reason(pipeline_id)

    def stop_orchestration(self) -> None:
        manager = self.pipeline_manager
        if manager is not None:
            manager.stop()

    def note(self, job_id: str, message: str) -> dict[str, Any]:
        self.jobs.note(job_id, message)
        self._persist_state(job_id)
        return self.jobs.get(job_id).snapshot()

    def attach_file(
        self,
        job_id: str,
        relative_path: str,
        *,
        media_type: str,
    ) -> dict[str, Any]:
        artifact = self.workspace.describe_file(relative_path, media_type=media_type)
        job = self.jobs.get(job_id)
        used_bytes = sum(item.size for item in job.artifacts)
        if used_bytes + artifact.size > job.request.budget.max_artifact_bytes:
            raise ValueError("research job artifact budget would be exceeded")
        self.jobs.add_artifact(job_id, artifact)
        self._persist_state(job_id)
        return self.jobs.get(job_id).snapshot()

    def mark_paused(self, job_id: str, *, message: str = "paused") -> dict[str, Any]:
        self.jobs.pause(job_id, message=message)
        self._persist_state(job_id)
        return self.jobs.get(job_id).snapshot()

    def mark_resumed(self, job_id: str, *, message: str = "resumed") -> dict[str, Any]:
        self.jobs.resume(job_id, message=message)
        self._persist_state(job_id)
        return self.jobs.get(job_id).snapshot()

    def pause_execution(self, job_id: str) -> dict[str, Any]:
        if self.executor is None:
            raise RuntimeError("research executor is not configured")
        return self.executor.pause_job(job_id)

    def resume_execution(self, job_id: str) -> dict[str, Any]:
        if self.executor is None:
            raise RuntimeError("research executor is not configured")
        return self.executor.resume_job(job_id)

    def retry(self, job_id: str, *, auto_execute: bool = True) -> dict[str, Any]:
        original = self.jobs.get(job_id)
        if not original.terminal:
            raise ValueError("only terminal research jobs can be retried")
        payload = self.submit(original.request)
        if auto_execute:
            payload["runtime"] = self.execute(str(payload["job_id"]))
        return payload

    def execute(self, job_id: str) -> dict[str, Any]:
        if self.executor is None:
            raise RuntimeError("research executor is not configured")
        return self.executor.enqueue(job_id)

    def cancel_execution(self, job_id: str) -> dict[str, Any]:
        if self.executor is None:
            return self.cancel(job_id)
        return self.executor.cancel(job_id)

    def runtime(self, job_id: str) -> dict[str, Any]:
        if self.executor is None:
            job = self.jobs.get(job_id)
            return {
                "job_id": job_id,
                "state": job.state.value,
                "queued": False,
                "active": False,
                "executor_enabled": False,
            }
        return self.executor.runtime(job_id)

    def executor_status(self) -> dict[str, Any]:
        if self.executor is None:
            return {
                "enabled": False,
                "paused": False,
                "queued": [],
                "queued_count": 0,
                "active": [],
                "active_count": 0,
                "live_order_write": False,
            }
        return self.executor.snapshot()

    def pause_executor(self) -> dict[str, Any]:
        if self.executor is None:
            raise RuntimeError("research executor is not configured")
        return self.executor.pause()

    def resume_executor(self) -> dict[str, Any]:
        if self.executor is None:
            raise RuntimeError("research executor is not configured")
        return self.executor.resume()

    def log_tail(self, job_id: str, *, lines: int = 300) -> dict[str, Any]:
        self.jobs.get(job_id)
        bounded = max(1, min(int(lines), 5000))
        path = self.workspace.resolve(f"jobs/{job_id}/execution.log")
        if not path.is_file():
            return {"job_id": job_id, "lines": [], "line_count": 0}
        with path.open("rb") as handle:
            size = path.stat().st_size
            handle.seek(max(0, size - 512 * 1024))
            text = handle.read().decode("utf-8", errors="replace")
        rows = text.splitlines()[-bounded:]
        return {"job_id": job_id, "lines": rows, "line_count": len(rows)}

    def ingest_output_metrics(self, job_id: str, output_directory: Path) -> dict[str, float]:
        job = self.jobs.get(job_id)
        metrics = extract_metrics_from_directory(
            output_directory,
            strategy=str(job.request.parameters.get("strategy", "")),
        )
        for name, value in metrics.items():
            try:
                self.metric(job_id, name, value)
            except ValueError:
                continue
        return metrics

    def register_experiment(self, job_id: str) -> dict[str, Any] | None:
        job = self.jobs.get(job_id)
        if job.request.kind not in {
            ResearchKind.ML_TRAIN,
            ResearchKind.ML_EVAL,
            ResearchKind.RL_TRAIN,
            ResearchKind.RL_EVAL,
        }:
            return None
        override_path = self.workspace.resolve(f"jobs/{job_id}/freqai-override.json")
        if not override_path.is_file():
            return None
        payload = self.workspace.read_json(f"jobs/{job_id}/freqai-override.json")
        if not isinstance(payload, dict) or not isinstance(payload.get("freqai"), dict):
            return None
        identifier = str(payload["freqai"].get("identifier", "")).strip()
        if not identifier:
            return None
        return self.experiments.record(job, identifier=identifier)

    def list_experiments(self, *, limit: int = 500) -> tuple[dict[str, Any], ...]:
        return self.experiments.list(limit=limit)

    def experiment(self, experiment_id: str, *, refresh: bool = False) -> dict[str, Any]:
        return (
            self.experiments.refresh(experiment_id)
            if refresh
            else self.experiments.get(experiment_id)
        )

    def experiment_tensorboard(
        self,
        experiment_id: str,
        *,
        max_points_per_tag: int = 1000,
        max_tags: int = 100,
    ) -> dict[str, Any]:
        experiment = self.experiments.refresh(experiment_id)
        identifier = str(experiment.get("identifier", "")).strip()
        if not identifier:
            raise ValueError("experiment has no FreqAI identifier")
        return read_tensorboard_scalars(
            self.experiments.model_root(identifier),
            max_points_per_tag=max_points_per_tag,
            max_tags=max_tags,
        )

    def leaderboard(
        self,
        *,
        metric: str,
        maximize: bool = True,
        limit: int = 50,
    ) -> tuple[dict[str, Any], ...]:
        return self.experiments.leaderboard(metric=metric, maximize=maximize, limit=limit)

    def model_catalog(self, *, limit: int = 200) -> tuple[dict[str, Any], ...]:
        return self.experiments.model_catalog(limit_identifiers=limit)

    def resume_training(
        self,
        experiment_id: str,
        *,
        auto_execute: bool = True,
        name: str | None = None,
    ) -> dict[str, Any]:
        experiment = self.experiments.get(experiment_id)
        job_id = str(experiment.get("job_id", ""))
        original = self.jobs.get(job_id)
        if original.request.kind not in {ResearchKind.ML_TRAIN, ResearchKind.RL_TRAIN}:
            raise ValueError("only ML_TRAIN or RL_TRAIN experiments can be resumed")
        identifier = str(experiment.get("identifier", "")).strip()
        if not identifier:
            raise ValueError("experiment has no FreqAI identifier")
        parameters = dict(original.request.parameters)
        parameters["resume_identifier"] = identifier
        parameters["freqai_identifier"] = identifier
        parameters["continual_learning"] = True
        request = ResearchRequest(
            kind=original.request.kind,
            name=name or f"{original.request.name}-resume",
            parameters=parameters,
            tags=tuple(dict.fromkeys((*original.request.tags, "resume"))),
            priority=original.request.priority,
            budget=original.request.budget,
        )
        payload = self.submit(request)
        if auto_execute:
            payload["runtime"] = self.execute(str(payload["job_id"]))
        return payload

    def plan_walk_forward(
        self,
        *,
        start: str,
        end: str,
        train_days: int,
        eval_days: int,
        step_days: int | None = None,
        expanding: bool = False,
        max_folds: int = 100,
    ) -> dict[str, Any]:
        folds = build_walk_forward_folds(
            start=start,
            end=end,
            train_days=train_days,
            eval_days=eval_days,
            step_days=step_days,
            expanding=expanding,
            max_folds=max_folds,
        )
        return {
            "start": start,
            "end": end,
            "train_days": train_days,
            "eval_days": eval_days,
            "step_days": eval_days if step_days is None else step_days,
            "expanding": expanding,
            "fold_count": len(folds),
            "folds": [item.snapshot() for item in folds],
        }

    @staticmethod
    def _walk_forward_shared_identifier(
        request: ResearchRequest,
        *,
        group_id: str,
        continual: bool,
    ) -> str:
        shared_identifier = str(
            request.parameters.get("freqai_identifier", "")
        ).strip()
        if not continual or request.kind not in {
            ResearchKind.ML_TRAIN,
            ResearchKind.RL_TRAIN,
        }:
            return shared_identifier
        if shared_identifier:
            return shared_identifier
        prefix = "ml" if request.kind is ResearchKind.ML_TRAIN else "rl"
        return f"hedge-wf-{prefix}-{group_id[3:]}"

    @staticmethod
    def _walk_forward_fold_parameters(
        request: ResearchRequest,
        *,
        fold: WalkForwardFold,
        fold_total: int,
        group_id: str,
        train_days: int,
        eval_days: int,
        continual: bool,
        shared_identifier: str,
        previous_job_id: str,
    ) -> dict[str, Any]:
        parameters = dict(request.parameters)
        parameters.update(
            {
                "timerange": fold.timerange,
                "train_period_days": train_days,
                "backtest_period_days": eval_days,
                "walk_forward_group": group_id,
                "fold_index": fold.index,
                "fold_total": fold_total,
                "fold_train_start": fold.train_start.date().isoformat(),
                "fold_train_end": fold.train_end.date().isoformat(),
                "fold_eval_start": fold.eval_start.date().isoformat(),
                "fold_eval_end": fold.eval_end.date().isoformat(),
            }
        )
        if request.kind not in {
            ResearchKind.ML_TRAIN,
            ResearchKind.RL_TRAIN,
        }:
            return parameters

        parameters.pop("resume_identifier", None)
        parameters.pop("depends_on_job_id", None)
        if continual:
            parameters["freqai_identifier"] = shared_identifier
            parameters["continual_learning"] = fold.index > 0
            if fold.index > 0:
                parameters["resume_identifier"] = shared_identifier
                parameters["depends_on_job_id"] = previous_job_id
            return parameters

        parameters["continual_learning"] = False
        base_identifier = str(
            parameters.get("freqai_identifier", "")
        ).strip()
        if base_identifier:
            parameters["freqai_identifier"] = (
                f"{base_identifier[:72]}-wf-{fold.index + 1:03d}"
            )
        return parameters

    def _submit_walk_forward_fold(
        self,
        request: ResearchRequest,
        *,
        fold: WalkForwardFold,
        fold_total: int,
        group_id: str,
        train_days: int,
        eval_days: int,
        continual: bool,
        shared_identifier: str,
        previous_job_id: str,
        auto_execute: bool,
    ) -> dict[str, Any]:
        parameters = self._walk_forward_fold_parameters(
            request,
            fold=fold,
            fold_total=fold_total,
            group_id=group_id,
            train_days=train_days,
            eval_days=eval_days,
            continual=continual,
            shared_identifier=shared_identifier,
            previous_job_id=previous_job_id,
        )
        child = ResearchRequest(
            kind=request.kind,
            name=f"{request.name}-fold-{fold.index + 1:02d}",
            parameters=parameters,
            tags=tuple(
                dict.fromkeys(
                    (*request.tags, "walk-forward", group_id)
                )
            ),
            priority=request.priority,
            budget=request.budget,
        )
        payload = self.submit(child)
        if auto_execute:
            payload["runtime"] = self.execute(str(payload["job_id"]))
        return payload

    @staticmethod
    def _walk_forward_group_payload(
        request: ResearchRequest,
        *,
        group_id: str,
        rows: list[dict[str, Any]],
        folds: tuple[WalkForwardFold, ...],
        train_days: int,
        eval_days: int,
        step_days: int | None,
        expanding: bool,
        continual: bool,
        shared_identifier: str,
    ) -> dict[str, Any]:
        return {
            "group_id": group_id,
            "created_at": datetime.now(UTC).isoformat(),
            "kind": request.kind.value,
            "name": request.name,
            "train_days": train_days,
            "eval_days": eval_days,
            "step_days": eval_days if step_days is None else step_days,
            "expanding": expanding,
            "continual_learning": continual,
            "shared_freqai_identifier": (
                shared_identifier if continual else ""
            ),
            "jobs": [str(item["job_id"]) for item in rows],
            "folds": [item.snapshot() for item in folds],
        }

    def submit_walk_forward(
        self,
        request: ResearchRequest,
        *,
        start: str,
        end: str,
        train_days: int,
        eval_days: int,
        step_days: int | None = None,
        expanding: bool = False,
        max_folds: int = 100,
        auto_execute: bool = True,
    ) -> dict[str, Any]:
        folds = build_walk_forward_folds(
            start=start,
            end=end,
            train_days=train_days,
            eval_days=eval_days,
            step_days=step_days,
            expanding=expanding,
            max_folds=max_folds,
        )
        group_id = new_group_id()
        continual = bool(
            request.parameters.get("walk_forward_continual", False)
        )
        shared_identifier = self._walk_forward_shared_identifier(
            request,
            group_id=group_id,
            continual=continual,
        )

        rows: list[dict[str, Any]] = []
        previous_job_id = ""
        for fold in folds:
            payload = self._submit_walk_forward_fold(
                request,
                fold=fold,
                fold_total=len(folds),
                group_id=group_id,
                train_days=train_days,
                eval_days=eval_days,
                continual=continual,
                shared_identifier=shared_identifier,
                previous_job_id=previous_job_id,
                auto_execute=auto_execute,
            )
            rows.append(payload)
            previous_job_id = str(payload["job_id"])

        group = self._walk_forward_group_payload(
            request,
            group_id=group_id,
            rows=rows,
            folds=folds,
            train_days=train_days,
            eval_days=eval_days,
            step_days=step_days,
            expanding=expanding,
            continual=continual,
            shared_identifier=shared_identifier,
        )
        self.workspace.write_json(
            f"walk_forward/{group_id}.json",
            group,
            max_bytes=2 * 1024 * 1024,
        )
        return {
            "group": self.walk_forward_group(group_id),
            "jobs": rows,
        }

    def list_walk_forward_groups(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        root = self.workspace.root / "walk_forward"
        if not root.is_dir():
            return ()
        rows: list[dict[str, Any]] = []
        for path in root.glob("wf-*.json"):
            try:
                payload = self.workspace.read_json(path.relative_to(self.workspace.root).as_posix())
                if isinstance(payload, dict):
                    rows.append(self.walk_forward_group(str(payload.get("group_id", ""))))
            except (FileNotFoundError, KeyError, OSError, TypeError, ValueError):
                continue
        rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return tuple(rows[: max(1, int(limit))])

    def walk_forward_group(self, group_id: str) -> dict[str, Any]:
        payload = self.workspace.read_json(f"walk_forward/{group_id}.json")
        if not isinstance(payload, dict):
            raise ValueError("walk-forward group is invalid")
        jobs = []
        metric_values: dict[str, list[float]] = {}
        for job_id in payload.get("jobs", []):
            job = self.jobs.get(str(job_id))
            snapshot = job.snapshot()
            jobs.append(snapshot)
            latest: dict[str, float] = {}
            for metric in job.metrics:
                latest[metric.name] = float(metric.value)
            for name, value in latest.items():
                metric_values.setdefault(name, []).append(value)
        aggregate = {
            name: {
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "stdev": pstdev(values) if len(values) > 1 else 0.0,
                "count": len(values),
            }
            for name, values in metric_values.items()
            if values
        }
        states: dict[str, int] = {}
        for item in jobs:
            state = str(item["state"])
            states[state] = states.get(state, 0) + 1
        result = dict(payload)
        result["job_states"] = states
        result["jobs_detail"] = jobs
        result["metrics"] = aggregate
        result["success_ratio"] = (
            states.get("SUCCEEDED", 0) / len(jobs) if jobs else 0.0
        )
        result["complete"] = bool(jobs) and all(
            item["state"] in {"SUCCEEDED", "FAILED", "CANCELED"} for item in jobs
        )
        return result

    def _config_path_for_job(self, job: ResearchJob) -> Path:
        raw = str(job.request.parameters.get("config_path", "")).strip()
        if not raw:
            raise ValueError("research job does not define config_path")
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path.resolve()
        if self.executor is None:
            raise RuntimeError("relative config_path requires a configured research executor")
        return (self.executor.project_root / path).resolve()

    def _submit_optimization_replay_child(
        self,
        parent: ResearchJob,
        materialized: Any,
        *,
        timerange: str,
        auto_execute: bool,
        name: str,
    ) -> dict[str, Any]:
        job_id = parent.job_id
        try:
            self.attach_file(job_id, materialized.relative_path, media_type="application/json")
        except ValueError:
            pass
        parameters = dict(parent.request.parameters)
        for key in (
            "trials",
            "workers",
            "auto_replay_best",
            "replay_timerange",
            "replay_top_n",
        ):
            parameters.pop(key, None)
        if timerange.strip():
            parameters["timerange"] = timerange.strip()
        parameters.update(
            {
                "optimization_parent_job_id": job_id,
                "optimization_trial_id": materialized.trial_id,
                "replay_overlay_relative_path": materialized.relative_path,
            }
        )
        request = ResearchRequest(
            kind=ResearchKind.BACKTEST,
            name=name,
            parameters=parameters,
            tags=tuple(dict.fromkeys((*parent.request.tags, "optimization-replay", job_id))),
            priority=parent.request.priority,
            budget=parent.request.budget,
        )
        child = self.submit(request)
        child_id = str(child["job_id"])
        try:
            self.attach_file(child_id, materialized.relative_path, media_type="application/json")
        except ValueError:
            pass
        self._append_optimization_replay(
            job_id,
            child_id=child_id,
            timerange=str(parameters.get("timerange", "")),
            trial_id=materialized.trial_id,
            overlay_path=materialized.relative_path,
        )
        child = self.get(child_id)
        if auto_execute:
            child["runtime"] = self.execute(child_id)
        return child

    def replay_best_optimization(
        self,
        job_id: str,
        *,
        timerange: str = "",
        auto_execute: bool = True,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Backtest the winning optimization candidate with a safe config overlay."""

        parent = self.jobs.get(job_id)
        if parent.request.kind is not ResearchKind.OPTIMIZATION:
            raise ValueError("best-parameter replay requires an OPTIMIZATION job")
        if parent.state is not ResearchState.SUCCEEDED:
            raise ValueError(
                "optimization job must succeed before its "
                "best candidate can be replayed"
            )
        materialized = materialize_best_parameter_overlay(
            self.workspace,
            optimization_job_id=job_id,
            config_path=self._config_path_for_job(parent),
        )
        selected_timerange = timerange.strip() or str(
            parent.request.parameters.get("replay_timerange", "")
        ).strip()
        child = self._submit_optimization_replay_child(
            parent,
            materialized,
            timerange=selected_timerange,
            auto_execute=auto_execute,
            name=name or f"{parent.request.name}-best-replay",
        )
        return {
            "optimization_job_id": job_id,
            "materialization": materialized.snapshot(),
            "replay_job": child,
            "replays": self.optimization_replays(job_id),
        }

    def replay_top_optimization(
        self,
        job_id: str,
        *,
        limit: int = 5,
        timerange: str = "",
        auto_execute: bool = True,
    ) -> dict[str, Any]:
        """Backtest the strongest scalar-score candidates on one independent window."""

        parent = self.jobs.get(job_id)
        if parent.request.kind is not ResearchKind.OPTIMIZATION:
            raise ValueError("candidate replay requires an OPTIMIZATION job")
        if parent.state is not ResearchState.SUCCEEDED:
            raise ValueError("optimization job must succeed before candidates can be replayed")
        candidates = ranked_candidates(
            self.workspace,
            optimization_job_id=job_id,
            limit=limit,
        )
        if not candidates:
            raise ValueError("optimization summary contains no COMPLETE candidates")
        config_path = self._config_path_for_job(parent)
        selected_timerange = timerange.strip() or str(
            parent.request.parameters.get("replay_timerange", "")
        ).strip()
        jobs: list[dict[str, Any]] = []
        materializations: list[dict[str, object]] = []
        summary_path = self.workspace.resolve(
            f"jobs/{job_id}/outputs/optimization/optimization-summary.json"
        )
        for rank, candidate in enumerate(candidates, start=1):
            values = candidate.get("parameters")
            if not isinstance(values, dict):
                continue
            materialized = materialize_parameter_overlay(
                self.workspace,
                optimization_job_id=job_id,
                config_path=config_path,
                parameters=values,
                trial_id=candidate.get("trial_id"),
                source_artifact=summary_path,
            )
            child = self._submit_optimization_replay_child(
                parent,
                materialized,
                timerange=selected_timerange,
                auto_execute=auto_execute,
                name=f"{parent.request.name}-candidate-{rank:02d}",
            )
            child["candidate_rank"] = rank
            child["optimization_scalar_score"] = candidate.get("scalar_score")
            jobs.append(child)
            materializations.append(materialized.snapshot())
        return {
            "optimization_job_id": job_id,
            "candidate_count": len(jobs),
            "timerange": selected_timerange,
            "jobs": jobs,
            "materializations": materializations,
            "replays": self.optimization_replays(job_id),
        }

    def _append_optimization_replay(
        self,
        optimization_job_id: str,
        *,
        child_id: str,
        timerange: str,
        trial_id: object,
        overlay_path: str,
    ) -> None:
        relative = f"optimization_replay/{optimization_job_id}.json"
        try:
            payload = self.workspace.read_json(relative)
        except FileNotFoundError:
            payload = {
                "optimization_job_id": optimization_job_id,
                "created_at": datetime.now(UTC).isoformat(),
                "replays": [],
            }
        if not isinstance(payload, dict):
            raise ValueError("optimization replay registry is invalid")
        rows = payload.setdefault("replays", [])
        if not isinstance(rows, list):
            raise ValueError("optimization replay registry rows are invalid")
        rows.append(
            {
                "job_id": child_id,
                "trial_id": trial_id,
                "timerange": timerange,
                "overlay_path": overlay_path,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )
        self.workspace.write_json(relative, payload, max_bytes=2 * 1024 * 1024)

    def _optimization_replay_row(
        self,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        row = dict(item)
        child_id = str(row.get("job_id", ""))
        try:
            child = self.jobs.get(child_id)
        except KeyError:
            row["state"] = "MISSING"
            row["metrics"] = {}
            return row

        row["state"] = child.state.value
        row["progress"] = child.progress
        latest: dict[str, float] = {}
        for metric in child.metrics:
            latest[metric.name] = float(metric.value)
        row["metrics"] = latest
        return row

    @staticmethod
    def _replay_ranking_metric(
        rows: list[dict[str, Any]],
    ) -> str:
        available_names = {
            name
            for row in rows
            for name in row.get("metrics", {})
            if isinstance(row.get("metrics"), dict)
        }
        return next(
            (
                name
                for name in (
                    "sharpe",
                    "profit",
                    "reward",
                    "sortino",
                    "win_rate",
                )
                if name in available_names
            ),
            "",
        )

    @staticmethod
    def _replay_leaderboard(
        rows: list[dict[str, Any]],
        ranking_metric: str,
    ) -> list[dict[str, Any]]:
        ranked = [
            row
            for row in rows
            if ranking_metric in row.get("metrics", {})
        ]
        ranked.sort(
            key=lambda row: float(row["metrics"][ranking_metric]),
            reverse=True,
        )
        return [
            {
                "rank": index,
                "job_id": row.get("job_id"),
                "trial_id": row.get("trial_id"),
                "state": row.get("state"),
                "value": row["metrics"][ranking_metric],
                "metrics": row["metrics"],
            }
            for index, row in enumerate(ranked, start=1)
        ]

    def optimization_replays(
        self,
        optimization_job_id: str,
    ) -> dict[str, Any]:
        relative = f"optimization_replay/{optimization_job_id}.json"
        try:
            payload = self.workspace.read_json(relative)
        except FileNotFoundError:
            payload = {
                "optimization_job_id": optimization_job_id,
                "replays": [],
            }
        if not isinstance(payload, dict):
            raise ValueError("optimization replay registry is invalid")

        rows = [
            self._optimization_replay_row(item)
            for item in payload.get("replays", [])
            if isinstance(item, dict)
        ]
        result = dict(payload)
        result["replays"] = rows
        ranking_metric = self._replay_ranking_metric(rows)
        result["ranking_metric"] = ranking_metric
        result["oos_leaderboard"] = (
            self._replay_leaderboard(rows, ranking_metric)
            if ranking_metric
            else []
        )
        return result

    def list_promotions(self, *, limit: int = 200) -> tuple[dict[str, Any], ...]:
        root = self.workspace.root / "promotions"
        if not root.is_dir():
            return ()
        rows: list[dict[str, Any]] = []
        for path in root.glob("promotion-*.json"):
            if path.name.endswith("-dryrun-override.json"):
                continue
            try:
                payload = self.workspace.read_json(path.relative_to(self.workspace.root).as_posix())
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return tuple(rows[: max(1, int(limit))])

    def promotion(self, promotion_id: str) -> dict[str, Any]:
        record_path = f"promotions/{promotion_id}.json"
        override_path = f"promotions/{promotion_id}-dryrun-override.json"
        record = self.workspace.read_json(record_path)
        override = self.workspace.read_json(override_path)
        if not isinstance(record, dict) or not isinstance(override, dict):
            raise ValueError("promotion record is invalid")
        result = dict(record)
        result["dry_run_override"] = override
        result["record_path"] = record_path
        result["dry_run_override_path"] = override_path
        return result

    def evaluate_promotion(self, experiment_id: str, policy: PromotionPolicy) -> dict[str, Any]:
        experiment = self.experiments.refresh(experiment_id)
        return evaluate_promotion(experiment, policy)

    def promote(self, experiment_id: str, policy: PromotionPolicy) -> dict[str, Any]:
        experiment = self.experiments.refresh(experiment_id)
        record, override = build_promotion_record(experiment, policy)
        promotion_id = str(record["promotion_id"])
        record_path = f"promotions/{promotion_id}.json"
        override_path = f"promotions/{promotion_id}-dryrun-override.json"
        self.workspace.write_json(record_path, record, max_bytes=1024 * 1024)
        self.workspace.write_json(override_path, override, max_bytes=1024 * 1024)
        result = dict(record)
        result["record_path"] = record_path
        result["dry_run_override_path"] = override_path
        return result

    def begin(self, job_id: str) -> dict[str, Any]:
        self.jobs.transition(job_id, ResearchState.RUNNING, message="started")
        self._persist_state(job_id)
        return self.jobs.get(job_id).snapshot()

    def progress(self, job_id: str, value: float, *, message: str = "") -> dict[str, Any]:
        self.jobs.progress(job_id, value, message=message)
        self._persist_state(job_id)
        return self.jobs.get(job_id).snapshot()

    def metric(
        self,
        job_id: str,
        name: str,
        value: float,
        *,
        step: int | None = None,
    ) -> dict[str, Any]:
        self.jobs.add_metric(job_id, ResearchMetric(name, float(value), step))
        self._persist_state(job_id)
        return self.jobs.get(job_id).snapshot()

    def complete(self, job_id: str, result: dict[str, Any]) -> dict[str, Any]:
        job = self.jobs.get(job_id)
        if job.state is not ResearchState.RUNNING:
            raise ValueError("research completion requires RUNNING state")
        used_bytes = sum(item.size for item in job.artifacts)
        remaining_bytes = job.request.budget.max_artifact_bytes - used_bytes
        if remaining_bytes < 1:
            raise ValueError("research job artifact budget is exhausted")
        artifact = self.workspace.write_json(
            f"jobs/{job_id}/result.json",
            result,
            max_bytes=remaining_bytes,
        )
        self.jobs.add_artifact(job_id, artifact)
        self.jobs.transition(
            job_id,
            ResearchState.SUCCEEDED,
            message="completed",
            progress=1.0,
        )
        self._persist_state(job_id)
        return self.jobs.get(job_id).snapshot()

    def fail(self, job_id: str, message: str) -> dict[str, Any]:
        self.jobs.transition(job_id, ResearchState.FAILED, message=message)
        self._persist_state(job_id)
        return self.jobs.get(job_id).snapshot()

    def cancel(self, job_id: str) -> dict[str, Any]:
        self.jobs.cancel(job_id)
        self._persist_state(job_id)
        return self.jobs.get(job_id).snapshot()

    def get(self, job_id: str) -> dict[str, Any]:
        return self.jobs.get(job_id).snapshot()

    def list_jobs(self, *, limit: int = 200) -> tuple[dict[str, Any], ...]:
        return tuple(item.snapshot() for item in self.jobs.list_jobs(limit=limit))

    def dashboard(self) -> dict[str, Any]:
        return dashboard_summary(self.jobs.list_jobs(limit=self.jobs.capacity))

    def artifact_path(self, job_id: str, relative_path: str) -> Path:
        job = self.jobs.get(job_id)
        allowed = {item.relative_path for item in job.artifacts}
        if relative_path not in allowed:
            raise KeyError(f"unknown research artifact for job {job_id}: {relative_path}")
        path = self.workspace.resolve(relative_path)
        if not path.is_file():
            raise FileNotFoundError(relative_path)
        return path

    def capabilities(self) -> dict[str, Any]:
        from .validation_matrix import ROUND_SPECS

        by_domain: dict[str, int] = {}
        for item in ROUND_SPECS:
            by_domain[item.domain] = by_domain.get(item.domain, 0) + 1
        executor = self.executor_status()
        return {
            "read_only_exchange": True,
            "live_order_write": False,
            "research_kinds": [item.value for item in ResearchKind],
            "rounds": len(ROUND_SPECS),
            "rounds_by_domain": by_domain,
            "recovered_jobs": len(self.jobs.list_jobs(limit=self.jobs.capacity)),
            "recovery_errors": tuple(self.recovery_errors),
            "experiment_registry": True,
            "checkpoint_catalog": True,
            "continual_learning_resume": True,
            "walk_forward_batch": True,
            "walk_forward_continual_dependencies": True,
            "optimization_best_replay": True,
            "optimization_top_n_oos_replay": True,
            "dry_run_promotion_gate": True,
            "experiment_pipeline_dag": self.pipeline_manager is not None,
            "pipeline_recovery": self.pipeline_manager is not None,
            "pipeline_pause_resume": self.pipeline_manager is not None,
            "pipeline_top_n_oos": self.pipeline_manager is not None,
            "pipeline_walk_forward_promotion": self.pipeline_manager is not None,
            "cpu_training": True,
            "cuda_training": True,
            "training_device_modes": ["auto", "cpu", "cuda"],
            "executor": {
                "enabled": bool(executor.get("enabled", False)),
                "max_concurrent": executor.get("max_concurrent", 0),
                "max_gpu_jobs": executor.get("max_gpu_jobs", 0),
                "max_cpu_training_jobs": executor.get("max_cpu_training_jobs", 0),
                "cpu_threads_per_job": executor.get("cpu_threads_per_job", 0),
                "effective_cpu_thread_limit": executor.get("effective_cpu_thread_limit", 0),
                "default_training_device": executor.get("default_training_device", "auto"),
                "paused": bool(executor.get("paused", False)),
            },
        }
