"""Walk-forward fold planning for backtest, optimization, ML, and RL jobs."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

_DATE_FORMAT = "%Y%m%d"


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    index: int
    train_start: datetime
    train_end: datetime
    eval_start: datetime
    eval_end: datetime

    @property
    def timerange(self) -> str:
        return f"{self.eval_start.strftime(_DATE_FORMAT)}-{self.eval_end.strftime(_DATE_FORMAT)}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "train_start": self.train_start.date().isoformat(),
            "train_end": self.train_end.date().isoformat(),
            "eval_start": self.eval_start.date().isoformat(),
            "eval_end": self.eval_end.date().isoformat(),
            "timerange": self.timerange,
        }


def _parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value.strip(), _DATE_FORMAT).replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("walk-forward dates must use YYYYMMDD") from exc


def build_walk_forward_folds(
    *,
    start: str,
    end: str,
    train_days: int,
    eval_days: int,
    step_days: int | None = None,
    expanding: bool = False,
    max_folds: int = 1000,
) -> tuple[WalkForwardFold, ...]:
    if train_days < 1 or eval_days < 1:
        raise ValueError("train_days and eval_days must be positive")
    step = eval_days if step_days is None else int(step_days)
    if step < 1:
        raise ValueError("step_days must be positive")
    if max_folds < 1:
        raise ValueError("max_folds must be positive")

    overall_start = _parse_date(start)
    overall_end = _parse_date(end)
    if overall_end <= overall_start:
        raise ValueError("walk-forward end must be after start")

    first_eval = overall_start + timedelta(days=train_days)
    folds: list[WalkForwardFold] = []
    eval_start = first_eval
    while eval_start < overall_end and len(folds) < max_folds:
        eval_end = min(overall_end, eval_start + timedelta(days=eval_days))
        if eval_end <= eval_start:
            break
        train_start = overall_start if expanding else eval_start - timedelta(days=train_days)
        train_end = eval_start
        folds.append(
            WalkForwardFold(
                index=len(folds),
                train_start=train_start,
                train_end=train_end,
                eval_start=eval_start,
                eval_end=eval_end,
            )
        )
        eval_start += timedelta(days=step)
    if not folds:
        raise ValueError("walk-forward range is too short for the requested training window")
    return tuple(folds)


def new_group_id() -> str:
    return f"wf-{uuid.uuid4().hex[:20]}"
