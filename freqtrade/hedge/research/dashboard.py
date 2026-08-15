"""Dashboard projections for research jobs and model/backtest artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from .contracts import ResearchJob, ResearchState


def dashboard_summary(jobs: Sequence[ResearchJob]) -> dict[str, Any]:
    states = Counter(item.state.value for item in jobs)
    kinds = Counter(item.request.kind.value for item in jobs)
    running = [item for item in jobs if item.state is ResearchState.RUNNING]
    failed = [item for item in jobs if item.state is ResearchState.FAILED]
    recent = sorted(jobs, key=lambda item: item.updated_at, reverse=True)[:20]
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "total": len(jobs),
        "states": dict(sorted(states.items())),
        "kinds": dict(sorted(kinds.items())),
        "running": len(running),
        "failed": len(failed),
        "mean_running_progress": (
            0.0 if not running else sum(item.progress for item in running) / len(running)
        ),
        "recent": [item.snapshot() for item in recent],
    }


def metric_series(job: ResearchJob, name: str) -> tuple[dict[str, float | int | None], ...]:
    return tuple(
        {"value": item.value, "step": item.step}
        for item in job.metrics
        if item.name == name
    )


def compare_job_metrics(
    jobs: Sequence[ResearchJob],
    metric: str,
    *,
    descending: bool = True,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        matches = [item for item in job.metrics if item.name == metric]
        if not matches:
            continue
        rows.append(
            {
                "job_id": job.job_id,
                "name": job.request.name,
                "kind": job.request.kind.value,
                "value": matches[-1].value,
            }
        )
    return tuple(sorted(rows, key=lambda row: float(row["value"]), reverse=descending))
