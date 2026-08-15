from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from freqtrade.hedge.exchange.base import ReadonlyState, StreamHealth


@dataclass(frozen=True, slots=True)
class FreshnessPolicy:
    event_stale_after: timedelta | None = None
    calibration_stale_after: timedelta = timedelta(minutes=15)
    reconnect_calibration_required: bool = True
    future_timestamp_tolerance: timedelta = timedelta(seconds=5)

    def __post_init__(self) -> None:
        if self.event_stale_after is not None and self.event_stale_after.total_seconds() <= 0:
            raise ValueError("event_stale_after must be positive or None")
        if self.calibration_stale_after.total_seconds() <= 0:
            raise ValueError("calibration_stale_after must be positive")
        if self.future_timestamp_tolerance.total_seconds() < 0:
            raise ValueError("future_timestamp_tolerance must be nonnegative")


@dataclass(frozen=True, slots=True)
class FreshnessAssessment:
    state: ReadonlyState
    fresh: bool
    reason: str
    event_age_seconds: float | None
    calibration_age_seconds: float | None


def _is_timezone_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _age_seconds(now: datetime, value: datetime | None) -> float | None:
    if not _is_timezone_aware(now):
        raise ValueError("Freshness timestamps must be timezone-aware")
    if value is None:
        return None
    if not _is_timezone_aware(value):
        raise ValueError("Freshness timestamps must be timezone-aware")
    return (now - value).total_seconds()


class UserStreamFreshness:
    def __init__(self, policy: FreshnessPolicy | None = None) -> None:
        self.policy = policy or FreshnessPolicy()

    def _failure(
        self,
        health: StreamHealth,
        *,
        raw_event_age: float | None,
        raw_calibration_age: float | None,
        event_age: float | None,
        calibration_age: float | None,
    ) -> tuple[ReadonlyState, str] | None:
        future_tolerance = self.policy.future_timestamp_tolerance.total_seconds()
        if raw_event_age is not None and raw_event_age < -future_tolerance:
            return ReadonlyState.DEGRADED, "USER_STREAM_TIMESTAMP_IN_FUTURE"
        if raw_calibration_age is not None and raw_calibration_age < -future_tolerance:
            return ReadonlyState.DEGRADED, "REST_CALIBRATION_TIMESTAMP_IN_FUTURE"
        if not health.connected:
            return ReadonlyState.RECOVERING, "USER_STREAM_DISCONNECTED"
        if health.last_connected_at is None:
            return ReadonlyState.DEGRADED, "USER_STREAM_CONNECTION_TIME_MISSING"
        if self._calibration_missing(health):
            return ReadonlyState.CALIBRATING, "REST_CALIBRATION_REQUIRED"
        if self._calibration_predates_connection(health):
            return ReadonlyState.CALIBRATING, "REST_CALIBRATION_PREDATES_CONNECTION"
        if self._calibration_is_stale(calibration_age):
            return ReadonlyState.DEGRADED, "REST_CALIBRATION_STALE"
        if self._event_is_stale(event_age):
            return ReadonlyState.DEGRADED, "USER_STREAM_STALE"
        return None

    def _calibration_missing(self, health: StreamHealth) -> bool:
        return (
            self.policy.reconnect_calibration_required
            and health.last_calibration_at is None
        )

    def _calibration_predates_connection(self, health: StreamHealth) -> bool:
        return bool(
            self.policy.reconnect_calibration_required
            and health.last_calibration_at is not None
            and health.last_connected_at is not None
            and health.last_calibration_at < health.last_connected_at
        )

    def _calibration_is_stale(self, age_seconds: float | None) -> bool:
        return bool(
            age_seconds is not None
            and age_seconds > self.policy.calibration_stale_after.total_seconds()
        )

    def _event_is_stale(self, age_seconds: float | None) -> bool:
        return bool(
            self.policy.event_stale_after is not None
            and age_seconds is not None
            and age_seconds > self.policy.event_stale_after.total_seconds()
        )

    def assess(self, health: StreamHealth, *, now: datetime) -> FreshnessAssessment:
        event_baseline = health.last_event_at or health.last_connected_at
        raw_event_age = _age_seconds(now, event_baseline)
        raw_calibration_age = _age_seconds(now, health.last_calibration_at)
        event_age = None if raw_event_age is None else max(0.0, raw_event_age)
        calibration_age = (
            None if raw_calibration_age is None else max(0.0, raw_calibration_age)
        )
        failure = self._failure(
            health,
            raw_event_age=raw_event_age,
            raw_calibration_age=raw_calibration_age,
            event_age=event_age,
            calibration_age=calibration_age,
        )
        if failure is None:
            state, fresh, reason = ReadonlyState.READY, True, "FRESH"
        else:
            state, fresh, reason = failure[0], False, failure[1]
        return FreshnessAssessment(
            state=state,
            fresh=fresh,
            reason=reason,
            event_age_seconds=event_age,
            calibration_age_seconds=calibration_age,
        )
