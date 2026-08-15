"""Deterministic resource and trial orchestration primitives."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import Event


@dataclass(frozen=True, slots=True)
class RetryDecision:
    retry: bool
    delay_seconds: Decimal


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    trials: int
    workers: int
    estimated_seconds: Decimal
    estimated_peak_memory_mb: Decimal


def budget_schedule(*, minimum: int, maximum: int, factor: int) -> tuple[int, ...]:
    if minimum <= 0 or maximum < minimum or factor < 2:
        raise ValueError("invalid budget schedule")
    values = []
    current = minimum
    while current < maximum:
        values.append(current)
        current *= factor
    values.append(maximum)
    return tuple(dict.fromkeys(values))


def successive_halving_plan(
    candidate_count: int,
    budgets: Sequence[int],
    *,
    reduction_factor: int,
) -> tuple[int, ...]:
    if candidate_count <= 0 or not budgets or reduction_factor < 2:
        raise ValueError("invalid successive-halving plan")
    survivors = []
    current = candidate_count
    for _ in budgets:
        survivors.append(current)
        current = max(1, current // reduction_factor)
    return tuple(survivors)


def deterministic_shards(trial_ids: Sequence[int], *, workers: int) -> tuple[tuple[int, ...], ...]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    shards: list[list[int]] = [[] for _ in range(workers)]
    for index, trial_id in enumerate(sorted(trial_ids)):
        shards[index % workers].append(trial_id)
    return tuple(tuple(shard) for shard in shards)


def retry_policy(
    attempt: int,
    *,
    maximum_attempts: int,
    base_delay_seconds: object,
    retryable: bool,
) -> RetryDecision:
    if attempt < 1 or maximum_attempts < 1 or attempt > maximum_attempts:
        raise ValueError("attempt counters are invalid")
    base = Decimal(str(base_delay_seconds))
    if base < 0:
        raise ValueError("base delay cannot be negative")
    retry = retryable and attempt < maximum_attempts
    return RetryDecision(retry, base * (Decimal(2) ** (attempt - 1)) if retry else Decimal(0))


def failure_budget_exceeded(
    failures: int,
    total: int,
    *,
    maximum_ratio: object,
    minimum_trials: int = 1,
) -> bool:
    ratio = Decimal(str(maximum_ratio))
    if (
        failures < 0
        or total < 0
        or failures > total
        or ratio < 0
        or ratio > 1
        or minimum_trials < 1
    ):
        raise ValueError("failure budget inputs are invalid")
    return total >= minimum_trials and Decimal(failures) / Decimal(total) > ratio


def timeout_deadline(started_at: datetime, timeout_seconds: object) -> datetime:
    if started_at.tzinfo is None:
        raise ValueError("started_at must be timezone-aware")
    timeout = Decimal(str(timeout_seconds))
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    return started_at + timedelta(seconds=float(timeout))


def heartbeat_expired(
    last_heartbeat: datetime,
    *,
    now: datetime | None = None,
    ttl_seconds: object,
) -> bool:
    current = now or datetime.now(UTC)
    if last_heartbeat.tzinfo is None or current.tzinfo is None:
        raise ValueError("heartbeat timestamps must be timezone-aware")
    ttl = Decimal(str(ttl_seconds))
    if ttl <= 0:
        raise ValueError("heartbeat ttl must be positive")
    return (current - last_heartbeat).total_seconds() > float(ttl)


def stable_merge_results(
    results: Iterable[Mapping[str, object]],
    *,
    id_key: str = "trial_id",
) -> tuple[Mapping[str, object], ...]:
    materialized = list(results)
    if any(id_key not in item for item in materialized):
        raise ValueError(f"all results require {id_key}")
    ids = [int(item[id_key]) for item in materialized]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trial result")
    return tuple(
        item
        for _, item in sorted(
            zip(ids, materialized, strict=True),
            key=lambda pair: pair[0],
        )
    )


class CancellationToken:
    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError("optimization cancelled")


def estimate_resources(
    *,
    trials: int,
    workers: int,
    seconds_per_trial: object,
    memory_mb_per_worker: object,
) -> ResourceEstimate:
    seconds = Decimal(str(seconds_per_trial))
    memory = Decimal(str(memory_mb_per_worker))
    if trials <= 0 or workers <= 0 or seconds < 0 or memory < 0:
        raise ValueError("resource estimate inputs are invalid")
    waves = (trials + workers - 1) // workers
    return ResourceEstimate(trials, workers, seconds * waves, memory * workers)
