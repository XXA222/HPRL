"""Optional TensorBoard scalar reader for completed/running FreqAI experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _downsample(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if len(rows) <= limit:
        return rows
    if limit <= 1:
        return [rows[-1]]
    last = len(rows) - 1
    indices = sorted({round(index * last / (limit - 1)) for index in range(limit)})
    return [rows[index] for index in indices]


def _event_directories(root: Path) -> list[Path]:
    event_files = sorted(
        path
        for path in root.rglob("events.out.tfevents*")
        if path.is_file()
    )
    return sorted({path.parent for path in event_files})


def _load_event_accumulator(
    event_accumulator_type: Any,
    directory: Path,
    *,
    max_points_per_tag: int,
) -> tuple[Any | None, str]:
    try:
        accumulator = event_accumulator_type(
            str(directory),
            size_guidance={
                "scalars": max(1, int(max_points_per_tag) * 4)
            },
        )
        accumulator.Reload()
    except (OSError, RuntimeError, ValueError) as exc:
        return None, f"{directory}: {type(exc).__name__}: {exc}"
    return accumulator, ""


def _append_scalar_events(
    series: dict[str, list[dict[str, Any]]],
    *,
    accumulator: Any,
    directory: Path,
    root: Path,
    max_tags: int,
) -> None:
    source_name = (
        directory.relative_to(root).as_posix()
        if directory != root
        else "."
    )
    tags = accumulator.Tags().get("scalars", [])
    for tag in tags[: max(1, int(max_tags))]:
        try:
            events = accumulator.Scalars(tag)
        except (KeyError, RuntimeError, ValueError):
            continue
        target = series.setdefault(tag, [])
        target.extend(
            {
                "step": int(event.step),
                "value": float(event.value),
                "wall_time": float(event.wall_time),
                "source": source_name,
            }
            for event in events
        )


def _normalize_scalar_series(
    series: dict[str, list[dict[str, Any]]],
    *,
    max_points_per_tag: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for tag, rows in series.items():
        rows.sort(
            key=lambda item: (
                int(item["step"]),
                float(item["wall_time"]),
            )
        )
        result[tag] = _downsample(
            rows,
            max(1, int(max_points_per_tag)),
        )
    return result


def read_tensorboard_scalars(
    model_root: Path,
    *,
    max_points_per_tag: int = 1000,
    max_tags: int = 100,
) -> dict[str, Any]:
    root = model_root.expanduser().resolve()
    if not root.is_dir():
        return {
            "available": True,
            "tags": {},
            "sources": [],
            "message": "model directory missing",
        }
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return {
            "available": False,
            "tags": {},
            "sources": [],
            "message": "tensorboard event reader is not installed",
        }

    series: dict[str, list[dict[str, Any]]] = {}
    sources: list[str] = []
    errors: list[str] = []
    for directory in _event_directories(root):
        accumulator, error = _load_event_accumulator(
            EventAccumulator,
            directory,
            max_points_per_tag=max_points_per_tag,
        )
        if accumulator is None:
            errors.append(error)
            continue
        sources.append(str(directory))
        _append_scalar_events(
            series,
            accumulator=accumulator,
            directory=directory,
            root=root,
            max_tags=max_tags,
        )

    result = _normalize_scalar_series(
        series,
        max_points_per_tag=max_points_per_tag,
    )
    return {
        "available": True,
        "tags": result,
        "tag_count": len(result),
        "sources": sources,
        "errors": errors,
    }
