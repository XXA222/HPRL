"""Best-effort progress extraction for local research subprocesses.

The parser is intentionally conservative: it only recognizes explicit progress
signals that can be mapped to the [0, 1] range without guessing about trading
results. Unrecognized output remains available through the execution log.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_PERCENT = re.compile(r"(?:^|[^0-9])(?P<value>100|[0-9]{1,2})\s*%(?:[^0-9]|$)")
_EPOCH = re.compile(
    r"\b(?:epoch|epochs)\s*[:#]?\s*(?P<current>[0-9]+)\s*(?:/|of)\s*(?P<total>[0-9]+)\b",
    re.IGNORECASE,
)
_STEP = re.compile(
    (
        r"\b(?:step|steps|timestep|timesteps)\s*[:#]?\s*"
        r"(?P<current>[0-9]+)\s*(?:/|of)\s*(?P<total>[0-9]+)\b"
    ),
    re.IGNORECASE,
)
_TRIAL = re.compile(r"\btrial\s+(?P<current>[0-9]+)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ProgressObservation:
    progress: float
    message: str


def _bounded_ratio(current: int, total: int) -> float | None:
    if current < 0 or total < 1:
        return None
    return min(1.0, max(0.0, current / total))


def observe_progress(line: str, *, max_trials: int | None = None) -> ProgressObservation | None:
    """Extract progress from one process output line.

    Priority is explicit epoch/step ratios, then percentages, then optimization
    trial numbers when the configured trial budget is known.
    """

    text = line.strip()
    if not text:
        return None

    for pattern, label in ((_EPOCH, "epoch"), (_STEP, "step")):
        match = pattern.search(text)
        if match is not None:
            current = int(match.group("current"))
            total = int(match.group("total"))
            ratio = _bounded_ratio(current, total)
            if ratio is not None:
                return ProgressObservation(ratio, f"{label} {current}/{total}")

    match = _PERCENT.search(text)
    if match is not None:
        percent = int(match.group("value"))
        return ProgressObservation(percent / 100.0, f"progress {percent}%")

    if max_trials is not None and max_trials > 0:
        match = _TRIAL.search(text)
        if match is not None:
            # Optuna trial numbers are zero-based, so trial 0 means one trial has run.
            current = int(match.group("current")) + 1
            ratio = _bounded_ratio(current, max_trials)
            if ratio is not None:
                return ProgressObservation(ratio, f"trial {current}/{max_trials}")
    return None

_METRIC = re.compile(
    r"\b(?P<name>loss|reward|sharpe|sortino|drawdown|profit|accuracy|f1|mae|rmse)"
    r"\s*[:=]\s*(?P<value>[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class MetricObservation:
    name: str
    value: float
    step: int | None


def _line_step(text: str) -> int | None:
    for pattern in (_EPOCH, _STEP):
        match = pattern.search(text)
        if match is not None:
            return int(match.group("current"))
    return None


def observe_metrics(line: str) -> tuple[MetricObservation, ...]:
    """Extract explicit numeric training/evaluation metrics from one log line."""

    text = line.strip()
    if not text:
        return ()
    step = _line_step(text)
    rows: list[MetricObservation] = []
    for match in _METRIC.finditer(text):
        try:
            value = float(match.group("value"))
        except ValueError:
            continue
        rows.append(MetricObservation(match.group("name").lower(), value, step))
    return tuple(rows)
