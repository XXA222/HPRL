"""Persistent research pipeline orchestration for Hedge experiments.

The pipeline composes existing research primitives instead of introducing a
second execution engine.  Every expensive step is still represented by a
normal :class:`ResearchJob`, so CPU/GPU admission, cancellation, logging,
artifacts, and fail-closed live-write protection remain centralized in the
research executor.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from .contracts import ResearchBudget, ResearchKind, ResearchRequest, ResearchState
from .promotion import PromotionPolicy, build_promotion_record, evaluate_promotion
from .training import normalize_training_device

if TYPE_CHECKING:
    from .service import HedgeResearchService


class PipelineState(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    PAUSED = "PAUSED"
    SUCCEEDED = "SUCCEEDED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"
    CANCELED = "CANCELED"


class PipelineStage(StrEnum):
    CREATED = "CREATED"
    OPTIMIZATION = "OPTIMIZATION"
    OOS_REPLAY = "OOS_REPLAY"
    WALK_FORWARD = "WALK_FORWARD"
    PROMOTION = "PROMOTION"
    DONE = "DONE"


_TERMINAL_PIPELINE_STATES = frozenset(
    {
        PipelineState.SUCCEEDED,
        PipelineState.REJECTED,
        PipelineState.FAILED,
        PipelineState.CANCELED,
    }
)


@dataclass(frozen=True, slots=True)
class ResearchPipelineSpec:
    name: str
    config_path: str
    strategy: str
    optimization_timerange: str
    oos_timerange: str
    training_kind: ResearchKind
    training_device: str = "auto"
    cpu_threads: int = 4
    walk_forward_start: str = ""
    walk_forward_end: str = ""
    train_days: int = 60
    eval_days: int = 15
    step_days: int = 15
    trials: int = 100
    workers: int = 1
    top_n: int = 5
    max_folds: int = 50
    expanding: bool = False
    continual_learning: bool = True
    require_training_approval: bool = False
    priority: int = 50
    max_seconds: int = 14_400
    max_artifact_bytes: int = 512 * 1024 * 1024
    oos_metric: str = "auto"
    min_oos_success_ratio: float = 1.0
    walk_forward_metric: str = "sharpe"
    min_walk_forward_success_ratio: float = 1.0
    stability_penalty: float = 0.5
    max_stage_retries: int = 1
    training_parameters: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    promotion_policy: PromotionPolicy = field(default_factory=PromotionPolicy)

    def _validate_required_fields(self) -> None:
        required = {
            "name": self.name,
            "config_path": self.config_path,
            "strategy": self.strategy,
            "optimization_timerange": self.optimization_timerange,
            "oos_timerange": self.oos_timerange,
        }
        for field_name, value in required.items():
            if not value.strip():
                raise ValueError(f"pipeline {field_name} is required")
        if (
            not self.walk_forward_start.strip()
            or not self.walk_forward_end.strip()
        ):
            raise ValueError("walk-forward start/end are required")

    def _validate_training_settings(self) -> None:
        if self.training_kind not in {
            ResearchKind.ML_TRAIN,
            ResearchKind.RL_TRAIN,
        }:
            raise ValueError("training_kind must be ML_TRAIN or RL_TRAIN")
        normalize_training_device(self.training_device)
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")

    def _validate_count_limits(self) -> None:
        if min(self.trials, self.workers, self.top_n) < 1:
            raise ValueError("trials, workers, and top_n must be positive")
        if self.top_n > 20:
            raise ValueError("top_n cannot exceed 20")
        if min(self.train_days, self.eval_days, self.step_days) < 1:
            raise ValueError("walk-forward day counts must be positive")
        if self.max_folds < 1:
            raise ValueError("max_folds must be positive")
        if not 0 <= self.priority <= 100:
            raise ValueError("priority must be within [0, 100]")

    def _validate_quality_thresholds(self) -> None:
        if not 0.0 < self.min_oos_success_ratio <= 1.0:
            raise ValueError("min_oos_success_ratio must be within (0, 1]")
        if not 0.0 < self.min_walk_forward_success_ratio <= 1.0:
            raise ValueError(
                "min_walk_forward_success_ratio must be within (0, 1]"
            )
        if self.stability_penalty < 0:
            raise ValueError("stability_penalty cannot be negative")
        if not 0 <= self.max_stage_retries <= 10:
            raise ValueError("max_stage_retries must be within [0, 10]")

    def __post_init__(self) -> None:
        self._validate_required_fields()
        self._validate_training_settings()
        self._validate_count_limits()
        self._validate_quality_thresholds()

    def snapshot(self) -> dict[str, Any]:
        policy = self.promotion_policy
        return {
            "name": self.name,
            "config_path": self.config_path,
            "strategy": self.strategy,
            "optimization_timerange": self.optimization_timerange,
            "oos_timerange": self.oos_timerange,
            "training_kind": self.training_kind.value,
            "training_device": self.training_device,
            "cpu_threads": self.cpu_threads,
            "walk_forward_start": self.walk_forward_start,
            "walk_forward_end": self.walk_forward_end,
            "train_days": self.train_days,
            "eval_days": self.eval_days,
            "step_days": self.step_days,
            "trials": self.trials,
            "workers": self.workers,
            "top_n": self.top_n,
            "max_folds": self.max_folds,
            "expanding": self.expanding,
            "continual_learning": self.continual_learning,
            "require_training_approval": self.require_training_approval,
            "priority": self.priority,
            "max_seconds": self.max_seconds,
            "max_artifact_bytes": self.max_artifact_bytes,
            "oos_metric": self.oos_metric,
            "min_oos_success_ratio": self.min_oos_success_ratio,
            "walk_forward_metric": self.walk_forward_metric,
            "min_walk_forward_success_ratio": self.min_walk_forward_success_ratio,
            "stability_penalty": self.stability_penalty,
            "max_stage_retries": self.max_stage_retries,
            "training_parameters": dict(self.training_parameters),
            "tags": list(self.tags),
            "promotion_policy": {
                "min_sharpe": policy.min_sharpe,
                "max_drawdown": policy.max_drawdown,
                "min_reward": policy.min_reward,
                "max_loss": policy.max_loss,
                "min_profit": policy.min_profit,
                "require_model_files": policy.require_model_files,
            },
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ResearchPipelineSpec:
        policy = payload.get("promotion_policy", {})
        if not isinstance(policy, dict):
            raise TypeError("pipeline promotion_policy must be an object")
        training_parameters = payload.get("training_parameters", {})
        if not isinstance(training_parameters, dict):
            raise TypeError("pipeline training_parameters must be an object")
        return cls(
            name=str(payload["name"]),
            config_path=str(payload["config_path"]),
            strategy=str(payload["strategy"]),
            optimization_timerange=str(payload["optimization_timerange"]),
            oos_timerange=str(payload["oos_timerange"]),
            training_kind=ResearchKind(str(payload["training_kind"])),
            training_device=str(payload.get("training_device", "auto")),
            cpu_threads=int(payload.get("cpu_threads", 4)),
            walk_forward_start=str(payload["walk_forward_start"]),
            walk_forward_end=str(payload["walk_forward_end"]),
            train_days=int(payload.get("train_days", 60)),
            eval_days=int(payload.get("eval_days", 15)),
            step_days=int(payload.get("step_days", 15)),
            trials=int(payload.get("trials", 100)),
            workers=int(payload.get("workers", 1)),
            top_n=int(payload.get("top_n", 5)),
            max_folds=int(payload.get("max_folds", 50)),
            expanding=bool(payload.get("expanding", False)),
            continual_learning=bool(payload.get("continual_learning", True)),
            require_training_approval=bool(payload.get("require_training_approval", False)),
            priority=int(payload.get("priority", 50)),
            max_seconds=int(payload.get("max_seconds", 14_400)),
            max_artifact_bytes=int(payload.get("max_artifact_bytes", 512 * 1024 * 1024)),
            oos_metric=str(payload.get("oos_metric", "auto")),
            min_oos_success_ratio=float(payload.get("min_oos_success_ratio", 1.0)),
            walk_forward_metric=str(payload.get("walk_forward_metric", "sharpe")),
            min_walk_forward_success_ratio=float(
                payload.get("min_walk_forward_success_ratio", 1.0)
            ),
            stability_penalty=float(payload.get("stability_penalty", 0.5)),
            max_stage_retries=int(payload.get("max_stage_retries", 1)),
            training_parameters=dict(training_parameters),
            tags=tuple(str(item) for item in payload.get("tags", ())),
            promotion_policy=PromotionPolicy(**policy),
        )


@dataclass(slots=True)
class PipelineRecord:
    pipeline_id: str
    spec: ResearchPipelineSpec
    state: PipelineState
    stage: PipelineStage
    created_at: str
    updated_at: str
    message: str = ""
    optimization: dict[str, Any] = field(default_factory=dict)
    oos: dict[str, Any] = field(default_factory=dict)
    walk_forward: dict[str, Any] = field(default_factory=dict)
    promotion: dict[str, Any] = field(default_factory=dict)
    retries: dict[str, int] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL_PIPELINE_STATES

    def snapshot(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "spec": self.spec.snapshot(),
            "state": self.state.value,
            "stage": self.stage.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message": self.message,
            "optimization": self.optimization,
            "oos": self.oos,
            "walk_forward": self.walk_forward,
            "promotion": self.promotion,
            "retries": dict(self.retries),
            "events": list(self.events[-200:]),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> PipelineRecord:
        return cls(
            pipeline_id=str(payload["pipeline_id"]),
            spec=ResearchPipelineSpec.from_payload(dict(payload["spec"])),
            state=PipelineState(str(payload["state"])),
            stage=PipelineStage(str(payload["stage"])),
            created_at=str(payload["created_at"]),
            updated_at=str(payload.get("updated_at", payload["created_at"])),
            message=str(payload.get("message", "")),
            optimization=dict(payload.get("optimization", {})),
            oos=dict(payload.get("oos", {})),
            walk_forward=dict(payload.get("walk_forward", {})),
            promotion=dict(payload.get("promotion", {})),
            retries={str(k): int(v) for k, v in dict(payload.get("retries", {})).items()},
            events=[dict(item) for item in payload.get("events", ()) if isinstance(item, dict)],
        )


class ResearchPipelineManager:
    """Durable stage orchestrator driven entirely by real ResearchJob state."""

    def __init__(
        self,
        service: HedgeResearchService,
        *,
        poll_seconds: float = 1.0,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("pipeline poll_seconds must be positive")
        self.service = service
        self.poll_seconds = poll_seconds
        self._lock = threading.RLock()
        self._records: dict[str, PipelineRecord] = {}
        self._stopping = threading.Event()
        self._load()
        self._thread = threading.Thread(
            target=self._loop,
            name="hedge-research-pipeline",
            daemon=True,
        )
        self._thread.start()

    def _path(self, pipeline_id: str) -> str:
        return f"pipelines/{pipeline_id}/pipeline.json"

    def _load(self) -> None:
        root = self.service.workspace.root / "pipelines"
        if not root.is_dir():
            return
        for path in sorted(root.glob("*/pipeline.json")):
            try:
                payload = self.service.workspace.read_json(
                    path.relative_to(self.service.workspace.root).as_posix()
                )
                if isinstance(payload, dict):
                    record = PipelineRecord.from_payload(payload)
                    self._records[record.pipeline_id] = record
            except (KeyError, OSError, TypeError, ValueError):
                continue

    def _persist(self, record: PipelineRecord) -> None:
        record.updated_at = datetime.now(UTC).isoformat()
        self.service.workspace.write_json(
            self._path(record.pipeline_id),
            record.snapshot(),
            max_bytes=8 * 1024 * 1024,
        )

    def _event(self, record: PipelineRecord, event: str, message: str, **fields: Any) -> None:
        row = {
            "at": datetime.now(UTC).isoformat(),
            "event": event,
            "message": message,
        }
        row.update(fields)
        record.events.append(row)
        if len(record.events) > 500:
            del record.events[:-500]
        record.message = message
        self._persist(record)

    def create(self, spec: ResearchPipelineSpec, *, auto_start: bool = True) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        pipeline_id = f"pipeline-{uuid.uuid4().hex[:20]}"
        record = PipelineRecord(
            pipeline_id=pipeline_id,
            spec=spec,
            state=PipelineState.QUEUED,
            stage=PipelineStage.CREATED,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._records[pipeline_id] = record
            self._event(record, "created", "pipeline created")
        if auto_start:
            return self.start(pipeline_id)
        return self.snapshot(pipeline_id)

    def start(self, pipeline_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._get(pipeline_id)
            if record.terminal:
                raise ValueError("terminal pipeline cannot be started")
            if record.state is PipelineState.RUNNING:
                return self._enriched_snapshot(record)
            record.state = PipelineState.RUNNING
            self._event(record, "started", "pipeline running")
            if record.stage is PipelineStage.CREATED:
                self._start_optimization(record)
            return self._enriched_snapshot(record)

    def _get(self, pipeline_id: str) -> PipelineRecord:
        try:
            return self._records[pipeline_id]
        except KeyError as exc:
            raise KeyError(f"unknown research pipeline: {pipeline_id}") from exc

    def list(self, *, limit: int = 100) -> tuple[dict[str, Any], ...]:
        with self._lock:
            rows = sorted(
                self._records.values(),
                key=lambda item: item.updated_at,
                reverse=True,
            )
            return tuple(self._enriched_snapshot(item) for item in rows[: max(1, int(limit))])

    def snapshot(self, pipeline_id: str) -> dict[str, Any]:
        with self._lock:
            return self._enriched_snapshot(self._get(pipeline_id))

    def is_paused(self, pipeline_id: str) -> bool:
        # Called from the executor while its condition lock is held.  Keep this
        # lookup lock-free to avoid an executor-lock -> pipeline-lock inversion.
        record = self._records.get(pipeline_id)
        return record is not None and record.state is PipelineState.PAUSED

    def block_reason(self, pipeline_id: str) -> str:
        # See is_paused(): state replacement is atomic and this fast read avoids
        # deadlocks with pipeline operations that enqueue/cancel executor work.
        record = self._records.get(pipeline_id)
        if record is None:
            return f"pipeline {pipeline_id} is missing"
        if record.state is PipelineState.PAUSED:
            return f"pipeline {pipeline_id} paused"
        if record.state in _TERMINAL_PIPELINE_STATES:
            return f"pipeline {pipeline_id} is {record.state.value}"
        return ""

    def _owned_job_ids(self, record: PipelineRecord) -> tuple[str, ...]:
        ids: list[str] = []
        opt = str(record.optimization.get("job_id", ""))
        if opt:
            ids.append(opt)
        for row in record.oos.get("jobs", ()):
            if isinstance(row, dict) and row.get("job_id"):
                ids.append(str(row["job_id"]))
        for job_id in record.walk_forward.get("jobs", ()):
            if job_id:
                ids.append(str(job_id))
        return tuple(dict.fromkeys(ids))

    def pause(self, pipeline_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._get(pipeline_id)
            if record.terminal:
                raise ValueError("terminal pipeline cannot be paused")
            record.state = PipelineState.PAUSED
            job_ids = self._owned_job_ids(record)
            self._event(record, "paused", "pipeline paused")
        for job_id in job_ids:
            try:
                job = self.service.jobs.get(job_id)
                if job.state is ResearchState.RUNNING:
                    self.service.pause_execution(job_id)
            except (KeyError, RuntimeError, ValueError):
                continue
        return self.snapshot(pipeline_id)

    def resume(self, pipeline_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._get(pipeline_id)
            if record.state is not PipelineState.PAUSED:
                raise ValueError("pipeline is not paused")
            record.state = PipelineState.RUNNING
            job_ids = self._owned_job_ids(record)
            self._event(record, "resumed", "pipeline resumed")
        for job_id in job_ids:
            try:
                job = self.service.jobs.get(job_id)
                if job.state is ResearchState.PAUSED:
                    self.service.resume_execution(job_id)
            except (KeyError, RuntimeError, ValueError):
                continue
        return self.snapshot(pipeline_id)

    def approve_training(self, pipeline_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._get(pipeline_id)
            if record.state is not PipelineState.AWAITING_APPROVAL:
                raise ValueError("pipeline is not awaiting training approval")
            selected = record.oos.get("selected", {})
            overlay_path = (
                str(selected.get("overlay_path", ""))
                if isinstance(selected, dict)
                else ""
            )
            if not overlay_path:
                raise ValueError("pipeline has no approved OOS parameter overlay")
            record.state = PipelineState.RUNNING
            self._event(record, "approved", "training approved after OOS review")
            self._start_walk_forward(record, overlay_path=overlay_path)
            return self._enriched_snapshot(record)

    def cancel(self, pipeline_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._get(pipeline_id)
            if record.terminal:
                return self._enriched_snapshot(record)
            job_ids = self._owned_job_ids(record)
            record.state = PipelineState.CANCELED
            record.stage = PipelineStage.DONE
            self._event(record, "canceled", "pipeline canceled")
        for job_id in job_ids:
            try:
                job = self.service.jobs.get(job_id)
                if not job.terminal:
                    self.service.cancel_execution(job_id)
            except (KeyError, RuntimeError, ValueError):
                continue
        return self.snapshot(pipeline_id)

    def retry(self, pipeline_id: str) -> dict[str, Any]:
        with self._lock:
            record = self._get(pipeline_id)
            if record.state is not PipelineState.FAILED:
                raise ValueError(
                    "only FAILED pipelines can be retried; "
                    "REJECTED needs a new gate policy"
                )
            record.state = PipelineState.RUNNING
            if record.walk_forward:
                record.stage = PipelineStage.WALK_FORWARD
                record.retries["walk_forward"] = 0
            elif record.oos:
                record.stage = PipelineStage.OOS_REPLAY
                for row in record.oos.get("jobs", ()):
                    if isinstance(row, dict):
                        row["attempt"] = 1
            else:
                record.stage = PipelineStage.OPTIMIZATION
                record.retries["optimization"] = 0
            self._event(record, "retry", f"manual pipeline retry from {record.stage.value}")
            return self._enriched_snapshot(record)

    def reconsider_promotion(
        self,
        pipeline_id: str,
        policy: PromotionPolicy,
    ) -> dict[str, Any]:
        with self._lock:
            record = self._get(pipeline_id)
            if record.state not in {PipelineState.REJECTED, PipelineState.FAILED}:
                raise ValueError(
                    "promotion can only be reconsidered for "
                    "REJECTED or FAILED pipeline"
                )
            candidate = record.promotion.get("candidate")
            if not isinstance(candidate, dict):
                raise ValueError("pipeline has no completed promotion candidate")
            gate = evaluate_promotion(candidate, policy)
            record.promotion["reconsidered_gate"] = gate
            record.promotion["reconsidered_policy"] = {
                "min_sharpe": policy.min_sharpe,
                "max_drawdown": policy.max_drawdown,
                "min_reward": policy.min_reward,
                "max_loss": policy.max_loss,
                "min_profit": policy.min_profit,
                "require_model_files": policy.require_model_files,
            }
            if not gate["passed"]:
                record.state = PipelineState.REJECTED
                record.stage = PipelineStage.DONE
                self._event(
                    record,
                    "promotion-rejected",
                    "candidate still does not satisfy reconsidered promotion gate",
                )
                return self._enriched_snapshot(record)
        promotion, override = build_promotion_record(candidate, policy)
        promotion["pipeline_id"] = record.pipeline_id
        promotion["walk_forward_group"] = record.walk_forward.get("group_id")
        promotion["selected_oos_trial_id"] = record.oos.get("selected", {}).get("trial_id")
        promotion["gate_reconsidered"] = True
        promotion_id = str(promotion["promotion_id"])
        record_path = f"promotions/{promotion_id}.json"
        override_path = f"promotions/{promotion_id}-dryrun-override.json"
        self.service.workspace.write_json(record_path, promotion, max_bytes=1024 * 1024)
        self.service.workspace.write_json(override_path, override, max_bytes=1024 * 1024)
        with self._lock:
            record.promotion.update(
                {
                    "promotion_id": promotion_id,
                    "record_path": record_path,
                    "dry_run_override_path": override_path,
                    "target": "DRY_RUN_CANDIDATE",
                }
            )
            record.state = PipelineState.SUCCEEDED
            record.stage = PipelineStage.DONE
            self._event(
                record,
                "promotion-reconsidered",
                "reconsidered gate produced DRY_RUN_CANDIDATE",
            )
            return self._enriched_snapshot(record)

    def _budget(self, spec: ResearchPipelineSpec) -> ResearchBudget:
        return ResearchBudget(
            max_seconds=spec.max_seconds,
            max_trials=spec.trials,
            max_workers=spec.workers,
            max_artifact_bytes=spec.max_artifact_bytes,
        )

    def _base_tags(self, record: PipelineRecord, stage: str) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys((*record.spec.tags, "pipeline", record.pipeline_id, f"stage:{stage}"))
        )

    def _start_optimization(self, record: PipelineRecord) -> None:
        spec = record.spec
        request = ResearchRequest(
            kind=ResearchKind.OPTIMIZATION,
            name=f"{spec.name}-optimization",
            parameters={
                "config_path": spec.config_path,
                "strategy": spec.strategy,
                "timerange": spec.optimization_timerange,
                "trials": spec.trials,
                "workers": spec.workers,
                "pipeline_id": record.pipeline_id,
                "pipeline_stage": PipelineStage.OPTIMIZATION.value,
            },
            tags=self._base_tags(record, "optimization"),
            priority=spec.priority,
            budget=self._budget(spec),
        )
        payload = self.service.submit(request)
        job_id = str(payload["job_id"])
        self.service.execute(job_id)
        record.optimization = {
            "job_id": job_id,
            "attempt": record.retries.get("optimization", 0) + 1,
        }
        record.stage = PipelineStage.OPTIMIZATION
        self._event(record, "stage-start", "optimization started", job_id=job_id)

    def _job_state(self, job_id: str) -> ResearchState:
        return self.service.jobs.get(job_id).state

    def _retry_optimization(self, record: PipelineRecord) -> bool:
        retries = record.retries.get("optimization", 0)
        if retries >= record.spec.max_stage_retries:
            return False
        record.retries["optimization"] = retries + 1
        self._start_optimization(record)
        return True

    def _advance_optimization(self, record: PipelineRecord) -> None:
        job_id = str(record.optimization.get("job_id", ""))
        if not job_id:
            self._start_optimization(record)
            return
        state = self._job_state(job_id)
        if state is ResearchState.SUCCEEDED:
            replay = self.service.replay_top_optimization(
                job_id,
                limit=record.spec.top_n,
                timerange=record.spec.oos_timerange,
                auto_execute=True,
            )
            rows: list[dict[str, Any]] = []
            materializations = {
                str(item.get("trial_id")): item
                for item in replay.get("materializations", ())
                if isinstance(item, dict)
            }
            for item in replay.get("jobs", ()):
                if not isinstance(item, dict):
                    continue
                child_job_id = str(item["job_id"])
                child_job = self.service.jobs.get(child_job_id)
                trial_id = child_job.request.parameters.get("optimization_trial_id")
                materialized = materializations.get(str(trial_id), {})
                rows.append(
                    {
                        "job_id": child_job_id,
                        "trial_id": trial_id,
                        "is_rank": item.get("candidate_rank"),
                        "is_score": item.get("optimization_scalar_score"),
                        "overlay_path": materialized.get("relative_path")
                        or child_job.request.parameters.get("replay_overlay_relative_path", ""),
                        "attempt": 1,
                    }
                )
            record.oos = {"jobs": rows, "timerange": record.spec.oos_timerange}
            record.stage = PipelineStage.OOS_REPLAY
            self._event(record, "stage-complete", "optimization succeeded; OOS replay started")
        elif state in {ResearchState.FAILED, ResearchState.CANCELED}:
            if not self._retry_optimization(record):
                self._fail(record, f"optimization ended as {state.value}")

    @staticmethod
    def _latest_metrics(job: Any) -> dict[str, float]:
        latest: dict[str, float] = {}
        for metric in job.metrics:
            latest[metric.name] = float(metric.value)
        return latest

    @staticmethod
    def _metric_direction(metric: str) -> int:
        return -1 if metric.lower() in {"loss", "drawdown", "mae", "rmse"} else 1

    def _choose_metric(self, rows: list[dict[str, Any]], requested: str) -> str:
        available = {
            name
            for row in rows
            for name in row.get("metrics", {})
            if isinstance(row.get("metrics"), dict)
        }
        if requested and requested.lower() != "auto":
            if requested not in available:
                raise ValueError(f"requested OOS metric is unavailable: {requested}")
            return requested
        for name in ("sharpe", "profit", "reward", "sortino", "win_rate", "drawdown", "loss"):
            if name in available:
                return name
        raise ValueError("OOS replay produced no rankable metrics")

    def _retry_oos_failures(self, record: PipelineRecord) -> bool:
        changed = False
        for row in record.oos.get("jobs", ()):
            if not isinstance(row, dict):
                continue
            job_id = str(row.get("job_id", ""))
            if not job_id:
                continue
            state = self._job_state(job_id)
            if state not in {ResearchState.FAILED, ResearchState.CANCELED}:
                continue
            attempt = int(row.get("attempt", 1))
            if attempt > record.spec.max_stage_retries:
                continue
            payload = self.service.retry(job_id, auto_execute=True)
            row["job_id"] = str(payload["job_id"])
            row["attempt"] = attempt + 1
            changed = True
        if changed:
            self._event(record, "retry", "retrying failed OOS candidate jobs")
        return changed

    @staticmethod
    def _oos_states_terminal(states: list[ResearchState]) -> bool:
        terminal = {
            ResearchState.SUCCEEDED,
            ResearchState.FAILED,
            ResearchState.CANCELED,
        }
        return all(state in terminal for state in states)

    def _oos_success_ratio(self, rows: list[dict[str, Any]]) -> float:
        succeeded = sum(
            self._job_state(str(row["job_id"])) is ResearchState.SUCCEEDED
            for row in rows
        )
        return succeeded / len(rows)

    def _scored_oos_rows(
        self,
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        scored: list[dict[str, Any]] = []
        for row in rows:
            job = self.service.jobs.get(str(row["job_id"]))
            if job.state is ResearchState.SUCCEEDED:
                scored.append({**row, "metrics": self._latest_metrics(job)})
        return scored

    def _rank_oos_rows(
        self,
        rows: list[dict[str, Any]],
        requested_metric: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        metric = self._choose_metric(rows, requested_metric)
        direction = self._metric_direction(metric)
        ranked = [row for row in rows if metric in row["metrics"]]
        ranked.sort(
            key=lambda row: direction * float(row["metrics"][metric]),
            reverse=True,
        )
        return metric, ranked

    @staticmethod
    def _oos_leaderboard(
        rows: list[dict[str, Any]],
        metric: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                "rank": index,
                "job_id": row["job_id"],
                "trial_id": row.get("trial_id"),
                "value": row["metrics"][metric],
                "is_score": row.get("is_score"),
                "metrics": row["metrics"],
                "overlay_path": row.get("overlay_path", ""),
            }
            for index, row in enumerate(rows, start=1)
        ]

    def _oos_jobs_ready(
        self,
        record: PipelineRecord,
        rows: list[dict[str, Any]],
    ) -> bool:
        states = [
            self._job_state(str(row["job_id"]))
            for row in rows
        ]
        if not self._oos_states_terminal(states):
            return False
        if any(
            state is not ResearchState.SUCCEEDED
            for state in states
        ):
            return not self._retry_oos_failures(record)
        return True

    def _validate_oos_success_ratio(
        self,
        record: PipelineRecord,
        rows: list[dict[str, Any]],
    ) -> bool:
        success_ratio = self._oos_success_ratio(rows)
        record.oos["success_ratio"] = success_ratio
        if success_ratio >= record.spec.min_oos_success_ratio:
            return True
        self._fail(
            record,
            (
                f"OOS success ratio {success_ratio:.3f} below required "
                f"{record.spec.min_oos_success_ratio:.3f}"
            ),
        )
        return False

    def _select_oos_winner(
        self,
        record: PipelineRecord,
        rows: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]] | None:
        scored = self._scored_oos_rows(rows)
        if not scored:
            self._fail(record, "all OOS candidate jobs failed")
            return None
        try:
            metric, ranked = self._rank_oos_rows(
                scored,
                record.spec.oos_metric,
            )
        except ValueError as exc:
            self._fail(record, str(exc))
            return None
        if not ranked:
            self._fail(
                record,
                f"no successful OOS candidate has metric {metric}",
            )
            return None
        return metric, ranked, ranked[0]

    def _advance_oos(self, record: PipelineRecord) -> None:
        raw_rows = [
            item
            for item in record.oos.get("jobs", ())
            if isinstance(item, dict)
        ]
        if not raw_rows:
            self._fail(record, "OOS stage has no jobs")
            return
        if not self._oos_jobs_ready(record, raw_rows):
            return
        if not self._validate_oos_success_ratio(record, raw_rows):
            return

        selection = self._select_oos_winner(record, raw_rows)
        if selection is None:
            return
        metric, ranked, winner = selection
        leaderboard = self._oos_leaderboard(ranked, metric)
        record.oos["ranking_metric"] = metric
        record.oos["leaderboard"] = leaderboard
        record.oos["selected"] = dict(leaderboard[0])

        if record.spec.require_training_approval:
            record.state = PipelineState.AWAITING_APPROVAL
            self._event(
                record,
                "approval-required",
                "OOS selection complete; waiting for explicit training approval",
                selected_trial=winner.get("trial_id"),
            )
            return
        self._start_walk_forward(
            record,
            overlay_path=str(winner.get("overlay_path", "")),
        )

    def _training_request(
        self,
        record: PipelineRecord,
        *,
        overlay_path: str,
        attempt: int,
    ) -> ResearchRequest:
        spec = record.spec
        parameters = dict(spec.training_parameters)
        parameters["training_device"] = spec.training_device
        parameters["cpu_threads"] = spec.cpu_threads
        parameters.update(
            {
                "config_path": spec.config_path,
                "strategy": spec.strategy,
                "pipeline_id": record.pipeline_id,
                "pipeline_stage": PipelineStage.WALK_FORWARD.value,
                "replay_overlay_relative_path": overlay_path,
                "pipeline_attempt": attempt,
                "walk_forward_continual": spec.continual_learning,
            }
        )
        base_identifier = str(parameters.get("freqai_identifier", "")).strip()
        if not base_identifier:
            base_identifier = f"pipe-{record.pipeline_id[-10:]}-{spec.training_kind.value.lower()}"
        if attempt > 1:
            base_identifier = f"{base_identifier[:72]}-retry-{attempt}"
        parameters["freqai_identifier"] = base_identifier
        return ResearchRequest(
            kind=spec.training_kind,
            name=f"{spec.name}-walk-forward",
            parameters=parameters,
            tags=self._base_tags(record, "walk-forward"),
            priority=spec.priority,
            budget=ResearchBudget(
                max_seconds=spec.max_seconds,
                max_trials=max(1, spec.trials),
                max_workers=max(1, spec.workers),
                max_artifact_bytes=spec.max_artifact_bytes,
            ),
        )

    def _start_walk_forward(self, record: PipelineRecord, *, overlay_path: str) -> None:
        if not overlay_path:
            self._fail(record, "selected OOS candidate has no parameter overlay")
            return
        attempt = int(record.walk_forward.get("attempt", 0)) + 1
        request = self._training_request(record, overlay_path=overlay_path, attempt=attempt)
        payload = self.service.submit_walk_forward(
            request,
            start=record.spec.walk_forward_start,
            end=record.spec.walk_forward_end,
            train_days=record.spec.train_days,
            eval_days=record.spec.eval_days,
            step_days=record.spec.step_days,
            expanding=record.spec.expanding,
            max_folds=record.spec.max_folds,
            auto_execute=True,
        )
        group = payload["group"]
        record.walk_forward = {
            "group_id": str(group["group_id"]),
            "jobs": [str(item["job_id"]) for item in payload.get("jobs", ())],
            "attempt": attempt,
            "overlay_path": overlay_path,
        }
        record.stage = PipelineStage.WALK_FORWARD
        self._event(
            record,
            "stage-complete",
            "OOS candidate selected; walk-forward training started",
            selected_trial=record.oos.get("selected", {}).get("trial_id"),
        )

    def _restart_walk_forward(self, record: PipelineRecord) -> bool:
        retries = record.retries.get("walk_forward", 0)
        if retries >= record.spec.max_stage_retries:
            return False
        record.retries["walk_forward"] = retries + 1
        self._start_walk_forward(
            record,
            overlay_path=str(record.walk_forward.get("overlay_path", "")),
        )
        return True

    def _aggregate_walk_forward(
        self,
        record: PipelineRecord,
        group: dict[str, Any],
    ) -> dict[str, float]:
        aggregate = group.get("metrics", {})
        result: dict[str, float] = {}
        for name, stats in aggregate.items():
            if not isinstance(stats, dict):
                continue
            try:
                if name == "drawdown":
                    result[name] = float(stats["max"])
                else:
                    result[name] = float(stats["mean"])
            except (KeyError, TypeError, ValueError):
                continue
        result["walk_forward_success_ratio"] = float(group.get("success_ratio", 0.0))
        metric = record.spec.walk_forward_metric
        stats = aggregate.get(metric, {})
        if isinstance(stats, dict):
            try:
                mean = float(stats["mean"])
                stdev = float(stats.get("stdev", 0.0))
                direction = self._metric_direction(metric)
                robust = (
                    mean - record.spec.stability_penalty * stdev
                    if direction > 0
                    else mean + record.spec.stability_penalty * stdev
                )
                result["walk_forward_robustness_score"] = robust
            except (KeyError, TypeError, ValueError):
                pass
        return result

    def _final_experiment(self, record: PipelineRecord, group: dict[str, Any]) -> dict[str, Any]:
        jobs = [item for item in group.get("jobs_detail", ()) if isinstance(item, dict)]
        succeeded = [item for item in jobs if item.get("state") == ResearchState.SUCCEEDED.value]
        if not succeeded:
            raise ValueError("walk-forward has no successful training experiment")
        if record.spec.continual_learning:
            succeeded.sort(key=lambda item: int(item.get("parameters", {}).get("fold_index", -1)))
            selected_job_id = str(succeeded[-1]["job_id"])
        else:
            metric = record.spec.walk_forward_metric
            direction = self._metric_direction(metric)
            ranked: list[tuple[float, str]] = []
            for item in succeeded:
                job = self.service.jobs.get(str(item["job_id"]))
                metrics = self._latest_metrics(job)
                if metric in metrics:
                    ranked.append((direction * float(metrics[metric]), job.job_id))
            selected_job_id = max(ranked)[1] if ranked else str(succeeded[-1]["job_id"])
        try:
            return self.service.experiment(selected_job_id, refresh=True)
        except (FileNotFoundError, KeyError, ValueError):
            registered = self.service.register_experiment(selected_job_id)
            if registered is None:
                raise ValueError("selected walk-forward job did not register an experiment")
            return self.service.experiment(selected_job_id, refresh=True)

    def _advance_walk_forward(self, record: PipelineRecord) -> None:
        group_id = str(record.walk_forward.get("group_id", ""))
        if not group_id:
            self._fail(record, "walk-forward group is missing")
            return
        group = self.service.walk_forward_group(group_id)
        if not bool(group.get("complete", False)):
            return
        success_ratio = float(group.get("success_ratio", 0.0))
        if success_ratio < record.spec.min_walk_forward_success_ratio:
            if self._restart_walk_forward(record):
                return
            self._fail(
                record,
                (
                    f"walk-forward success ratio {success_ratio:.3f} below "
                    f"required {record.spec.min_walk_forward_success_ratio:.3f}"
                ),
            )
            return
        try:
            experiment = self._final_experiment(record, group)
        except ValueError as exc:
            if self._restart_walk_forward(record):
                return
            self._fail(record, str(exc))
            return
        aggregate_metrics = self._aggregate_walk_forward(record, group)
        candidate = dict(experiment)
        candidate["metrics"] = aggregate_metrics
        candidate["pipeline_id"] = record.pipeline_id
        candidate["walk_forward_group"] = group_id
        candidate["source_experiment_metrics"] = experiment.get("metrics", {})
        record.walk_forward["aggregate_metrics"] = aggregate_metrics
        record.walk_forward["selected_experiment_id"] = experiment.get("experiment_id")
        record.walk_forward["selected_identifier"] = experiment.get("identifier")
        record.walk_forward["success_ratio"] = success_ratio
        record.promotion = {
            "candidate": candidate,
            "gate": evaluate_promotion(candidate, record.spec.promotion_policy),
        }
        record.stage = PipelineStage.PROMOTION
        self._event(record, "stage-complete", "walk-forward complete; evaluating promotion")

    def _advance_promotion(self, record: PipelineRecord) -> None:
        candidate = record.promotion.get("candidate")
        if not isinstance(candidate, dict):
            self._fail(record, "promotion candidate is missing")
            return
        gate = evaluate_promotion(candidate, record.spec.promotion_policy)
        record.promotion["gate"] = gate
        if not gate["passed"]:
            record.state = PipelineState.REJECTED
            record.stage = PipelineStage.DONE
            self._event(
                record,
                "promotion-rejected",
                "walk-forward candidate did not satisfy promotion gate",
            )
            return
        try:
            promotion, override = build_promotion_record(candidate, record.spec.promotion_policy)
        except ValueError as exc:
            self._fail(record, str(exc))
            return
        promotion["pipeline_id"] = record.pipeline_id
        promotion["walk_forward_group"] = record.walk_forward.get("group_id")
        promotion["selected_oos_trial_id"] = record.oos.get("selected", {}).get("trial_id")
        promotion_id = str(promotion["promotion_id"])
        record_path = f"promotions/{promotion_id}.json"
        override_path = f"promotions/{promotion_id}-dryrun-override.json"
        self.service.workspace.write_json(record_path, promotion, max_bytes=1024 * 1024)
        self.service.workspace.write_json(override_path, override, max_bytes=1024 * 1024)
        record.promotion.update(
            {
                "promotion_id": promotion_id,
                "record_path": record_path,
                "dry_run_override_path": override_path,
                "target": "DRY_RUN_CANDIDATE",
            }
        )
        record.state = PipelineState.SUCCEEDED
        record.stage = PipelineStage.DONE
        self._event(record, "completed", "pipeline produced DRY_RUN_CANDIDATE")

    def _fail(self, record: PipelineRecord, message: str) -> None:
        record.state = PipelineState.FAILED
        record.stage = PipelineStage.DONE
        job_ids = self._owned_job_ids(record)
        self._event(record, "failed", message)
        for job_id in job_ids:
            try:
                job = self.service.jobs.get(job_id)
                if not job.terminal:
                    self.service.cancel_execution(job_id)
            except (KeyError, RuntimeError, ValueError):
                continue

    def _advance(self, record: PipelineRecord) -> None:
        if record.state is not PipelineState.RUNNING:
            return
        if record.stage is PipelineStage.CREATED:
            self._start_optimization(record)
        elif record.stage is PipelineStage.OPTIMIZATION:
            self._advance_optimization(record)
        elif record.stage is PipelineStage.OOS_REPLAY:
            self._advance_oos(record)
        elif record.stage is PipelineStage.WALK_FORWARD:
            self._advance_walk_forward(record)
        elif record.stage is PipelineStage.PROMOTION:
            self._advance_promotion(record)

    def tick(self) -> None:
        with self._lock:
            for record in tuple(self._records.values()):
                if record.state is PipelineState.RUNNING:
                    try:
                        self._advance(record)
                    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                        self._fail(record, f"pipeline orchestration error: {exc}")

    def _loop(self) -> None:
        while not self._stopping.is_set():
            self.tick()
            self._stopping.wait(self.poll_seconds)

    def stop(self) -> None:
        self._stopping.set()
        if self._thread.is_alive():
            self._thread.join(timeout=max(1.0, self.poll_seconds * 3))

    def _job_node(self, node_id: str, label: str, job_id: str) -> dict[str, Any]:
        try:
            job = self.service.jobs.get(job_id)
            return {
                "id": node_id,
                "label": label,
                "type": "job",
                "job_id": job_id,
                "state": job.state.value,
                "progress": job.progress,
                "kind": job.request.kind.value,
                "message": job.message,
            }
        except KeyError:
            return {
                "id": node_id,
                "label": label,
                "type": "job",
                "job_id": job_id,
                "state": "MISSING",
            }

    def _append_optimization_node(
        self,
        record: PipelineRecord,
        nodes: list[dict[str, Any]],
    ) -> None:
        job_id = str(record.optimization.get("job_id", ""))
        if job_id:
            nodes.append(
                self._job_node(
                    "optimization",
                    "Optimization / IS",
                    job_id,
                )
            )
            return
        nodes.append(
            {
                "id": "optimization",
                "label": "Optimization / IS",
                "type": "stage",
                "state": "PENDING",
            }
        )

    def _append_oos_nodes(
        self,
        record: PipelineRecord,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        rows = [
            item
            for item in record.oos.get("jobs", ())
            if isinstance(item, dict)
        ]
        if not rows:
            nodes.append(
                {
                    "id": "oos",
                    "label": "Top-N OOS",
                    "type": "fanout",
                    "state": "PENDING",
                }
            )
            edges.append({"from": "optimization", "to": "oos"})
            return rows
        for index, row in enumerate(rows, start=1):
            node_id = f"oos-{index}"
            nodes.append(
                self._job_node(
                    node_id,
                    f"OOS Candidate {index}",
                    str(row.get("job_id", "")),
                )
            )
            edges.append({"from": "optimization", "to": node_id})
        return rows

    def _append_walk_forward_nodes(
        self,
        record: PipelineRecord,
        oos_rows: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        edges: list[dict[str, str]],
    ) -> str:
        jobs = [
            str(item)
            for item in record.walk_forward.get("jobs", ())
            if item
        ]
        if not jobs:
            nodes.append(
                {
                    "id": "walk-forward",
                    "label": "ML/RL Walk-forward",
                    "type": "stage",
                    "state": "PENDING",
                }
            )
            edges.append({"from": "oos", "to": "walk-forward"})
            return "walk-forward"

        previous = ""
        for index, job_id in enumerate(jobs, start=1):
            node_id = f"wf-{index}"
            nodes.append(
                self._job_node(
                    node_id,
                    f"WF Fold {index}",
                    job_id,
                )
            )
            if index == 1:
                sources = (
                    [f"oos-{item}" for item in range(1, len(oos_rows) + 1)]
                    if oos_rows
                    else ["oos"]
                )
                edges.extend(
                    {"from": source, "to": node_id}
                    for source in sources
                )
            elif previous:
                edges.append({"from": previous, "to": node_id})
            previous = node_id
        return previous

    @staticmethod
    def _promotion_node_state(record: PipelineRecord) -> str:
        if record.state is PipelineState.SUCCEEDED:
            return "SUCCEEDED"
        if record.state is PipelineState.REJECTED:
            return "REJECTED"
        if record.stage is PipelineStage.PROMOTION:
            return "RUNNING"
        return "PENDING"

    def _dag(self, record: PipelineRecord) -> dict[str, Any]:
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, str]] = []
        self._append_optimization_node(record, nodes)
        oos_rows = self._append_oos_nodes(record, nodes, edges)
        previous = self._append_walk_forward_nodes(
            record,
            oos_rows,
            nodes,
            edges,
        )
        nodes.append(
            {
                "id": "promotion",
                "label": "Promotion → Dry-run",
                "type": "gate",
                "state": self._promotion_node_state(record),
            }
        )
        edges.append(
            {
                "from": previous or "walk-forward",
                "to": "promotion",
            }
        )
        return {"nodes": nodes, "edges": edges}

    def _optimization_progress(self, record: PipelineRecord, base: float) -> float:
        job_id = str(record.optimization.get("job_id", ""))
        if not job_id:
            return base
        try:
            progress = self.service.jobs.get(job_id).progress
        except KeyError:
            return base
        return min(0.34, base + progress * 0.24)

    def _oos_progress(self, record: PipelineRecord, base: float) -> float:
        rows = [
            item
            for item in record.oos.get("jobs", ())
            if isinstance(item, dict)
        ]
        if not rows:
            return base
        terminal = {
            ResearchState.SUCCEEDED,
            ResearchState.FAILED,
            ResearchState.CANCELED,
        }
        done = sum(
            self._job_state(str(item["job_id"])) in terminal
            for item in rows
        )
        return min(0.54, base + (done / len(rows)) * 0.19)

    def _walk_forward_progress(
        self,
        record: PipelineRecord,
        base: float,
    ) -> float:
        job_ids = [
            str(item)
            for item in record.walk_forward.get("jobs", ())
            if item
        ]
        if not job_ids:
            return base
        values: list[float] = []
        for job_id in job_ids:
            try:
                values.append(self.service.jobs.get(job_id).progress)
            except KeyError:
                values.append(0.0)
        return min(
            0.94,
            base + (sum(values) / len(values)) * 0.39,
        )

    def _progress(self, record: PipelineRecord) -> float:
        if record.state in {
            PipelineState.SUCCEEDED,
            PipelineState.REJECTED,
        }:
            return 1.0
        weights = {
            PipelineStage.CREATED: 0.0,
            PipelineStage.OPTIMIZATION: 0.10,
            PipelineStage.OOS_REPLAY: 0.35,
            PipelineStage.WALK_FORWARD: 0.55,
            PipelineStage.PROMOTION: 0.95,
            PipelineStage.DONE: 1.0,
        }
        base = weights[record.stage]
        handlers = {
            PipelineStage.OPTIMIZATION: self._optimization_progress,
            PipelineStage.OOS_REPLAY: self._oos_progress,
            PipelineStage.WALK_FORWARD: self._walk_forward_progress,
        }
        handler = handlers.get(record.stage)
        return base if handler is None else handler(record, base)

    def _enriched_snapshot(self, record: PipelineRecord) -> dict[str, Any]:
        payload = record.snapshot()
        payload["progress"] = round(self._progress(record), 6)
        payload["dag"] = self._dag(record)
        states: dict[str, int] = {}
        for job_id in self._owned_job_ids(record):
            try:
                state = self.service.jobs.get(job_id).state.value
            except KeyError:
                state = "MISSING"
            states[state] = states.get(state, 0) + 1
        aggregate = record.walk_forward.get("aggregate_metrics", {})
        payload["summary"] = {
            "job_count": sum(states.values()),
            "job_states": states,
            "selected_oos_trial_id": record.oos.get("selected", {}).get("trial_id")
            if isinstance(record.oos.get("selected"), dict) else None,
            "selected_identifier": record.walk_forward.get("selected_identifier", ""),
            "robustness_score": aggregate.get("walk_forward_robustness_score")
            if isinstance(aggregate, dict) else None,
            "dry_run_candidate": record.promotion.get("promotion_id", ""),
        }
        return payload
