"""Thread-safe state machine for local Hedge research jobs."""

from __future__ import annotations

import uuid
from copy import deepcopy
from datetime import UTC, datetime
from threading import RLock

from .contracts import ResearchArtifact, ResearchJob, ResearchMetric, ResearchRequest, ResearchState


_ALLOWED = {
    ResearchState.QUEUED: {ResearchState.RUNNING, ResearchState.FAILED, ResearchState.CANCELED},
    ResearchState.RUNNING: {
        ResearchState.PAUSED,
        ResearchState.SUCCEEDED,
        ResearchState.FAILED,
        ResearchState.CANCELED,
    },
    ResearchState.PAUSED: {
        ResearchState.RUNNING,
        ResearchState.FAILED,
        ResearchState.CANCELED,
    },
    ResearchState.SUCCEEDED: set(),
    ResearchState.FAILED: set(),
    ResearchState.CANCELED: set(),
}


class ResearchJobStore:
    def __init__(self, *, capacity: int = 1000) -> None:
        if capacity < 1:
            raise ValueError("job capacity must be positive")
        self.capacity = capacity
        self._lock = RLock()
        self._jobs: dict[str, ResearchJob] = {}

    def create(self, request: ResearchRequest) -> ResearchJob:
        with self._lock:
            if len(self._jobs) >= self.capacity:
                terminal = sorted(
                    (item for item in self._jobs.values() if item.terminal),
                    key=lambda item: item.updated_at,
                )
                if not terminal:
                    raise RuntimeError("research job store is full")
                self._jobs.pop(terminal[0].job_id)
            job = ResearchJob(job_id=uuid.uuid4().hex, request=request)
            self._jobs[job.job_id] = job
            return deepcopy(job)


    def restore(self, job: ResearchJob) -> ResearchJob:
        with self._lock:
            if job.job_id in self._jobs:
                raise ValueError(f"duplicate research job: {job.job_id}")
            if len(self._jobs) >= self.capacity:
                raise RuntimeError("research job store is full during recovery")
            self._jobs[job.job_id] = deepcopy(job)
            return deepcopy(job)

    def discard_queued(self, job_id: str) -> None:
        with self._lock:
            job = self._get_mutable(job_id)
            if job.state is not ResearchState.QUEUED or job.metrics or job.artifacts:
                raise ValueError("only an empty QUEUED research job can be discarded")
            self._jobs.pop(job_id)

    def _get_mutable(self, job_id: str) -> ResearchJob:
        try:
            return self._jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"unknown research job: {job_id}") from exc

    def get(self, job_id: str) -> ResearchJob:
        with self._lock:
            return deepcopy(self._get_mutable(job_id))

    def list_jobs(self, *, limit: int = 200) -> tuple[ResearchJob, ...]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with self._lock:
            rows = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return tuple(deepcopy(item) for item in rows[:limit])

    def transition(
        self,
        job_id: str,
        state: ResearchState,
        *,
        message: str = "",
        progress: float | None = None,
    ) -> ResearchJob:
        with self._lock:
            job = self._get_mutable(job_id)
            if state not in _ALLOWED[job.state]:
                raise ValueError(f"invalid research transition {job.state.value}->{state.value}")
            now = datetime.now(UTC)
            job.state = state
            job.message = message[:1000]
            if progress is not None:
                self._set_progress(job, progress)
            if state is ResearchState.RUNNING and job.started_at is None:
                job.started_at = now
            if state in {ResearchState.SUCCEEDED, ResearchState.FAILED, ResearchState.CANCELED}:
                job.finished_at = now
                if state is ResearchState.SUCCEEDED:
                    job.progress = 1.0
            job.updated_at = now
            job.revision += 1
            return deepcopy(job)

    @staticmethod
    def _set_progress(job: ResearchJob, progress: float) -> None:
        numeric = float(progress)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError("research progress must be within [0, 1]")
        if numeric < job.progress:
            raise ValueError("research progress cannot move backwards")
        job.progress = numeric

    def progress(self, job_id: str, progress: float, *, message: str = "") -> ResearchJob:
        with self._lock:
            job = self._get_mutable(job_id)
            if job.state is not ResearchState.RUNNING:
                raise ValueError("research progress requires RUNNING state")
            self._set_progress(job, progress)
            if message:
                job.message = message[:1000]
            job.updated_at = datetime.now(UTC)
            job.revision += 1
            return deepcopy(job)

    def pause(self, job_id: str, *, message: str = "paused") -> ResearchJob:
        return self.transition(job_id, ResearchState.PAUSED, message=message)

    def resume(self, job_id: str, *, message: str = "resumed") -> ResearchJob:
        return self.transition(job_id, ResearchState.RUNNING, message=message)

    def note(self, job_id: str, message: str) -> ResearchJob:
        """Update a non-terminal job message without changing progress."""
        with self._lock:
            job = self._get_mutable(job_id)
            if job.terminal:
                raise ValueError("terminal research job cannot be updated")
            job.message = message[:1000]
            job.updated_at = datetime.now(UTC)
            job.revision += 1
            return deepcopy(job)

    def add_metric(self, job_id: str, metric: ResearchMetric) -> ResearchJob:
        with self._lock:
            job = self._get_mutable(job_id)
            if job.state is not ResearchState.RUNNING:
                raise ValueError("research metrics require RUNNING state")
            job.metrics.append(metric)
            job.updated_at = datetime.now(UTC)
            job.revision += 1
            return deepcopy(job)

    def add_artifact(self, job_id: str, artifact: ResearchArtifact) -> ResearchJob:
        with self._lock:
            job = self._get_mutable(job_id)
            if any(item.relative_path == artifact.relative_path for item in job.artifacts):
                raise ValueError("duplicate research artifact path")
            job.artifacts.append(artifact)
            job.updated_at = datetime.now(UTC)
            job.revision += 1
            return deepcopy(job)

    def cancel(self, job_id: str, *, message: str = "canceled") -> ResearchJob:
        with self._lock:
            state = self._get_mutable(job_id).state
        if state is ResearchState.CANCELED:
            return self.get(job_id)
        return self.transition(job_id, ResearchState.CANCELED, message=message)
