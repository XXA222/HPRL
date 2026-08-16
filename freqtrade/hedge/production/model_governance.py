"""Model registry/approval/fallback contracts for ML, Risk-Level RL and HPRL."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json


class ModelStatus(StrEnum):
    CANDIDATE = "CANDIDATE"
    APPROVED = "APPROVED"
    QUARANTINED = "QUARANTINED"
    RETIRED = "RETIRED"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    model_id: str
    algorithm: str
    model_sha256: str
    feature_schema_sha256: str
    data_manifest_sha256: str
    training_config_sha256: str
    framework_fingerprint: str

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.algorithm.strip():
            raise ValueError("model_id and algorithm are required")
        for name in ("model_sha256", "feature_schema_sha256", "data_manifest_sha256", "training_config_sha256"):
            value = getattr(self, name).lower()
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError(f"{name} must be sha256")
            object.__setattr__(self, name, value)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.__dict__ if hasattr(self, "__dict__") else {
            "model_id": self.model_id,
            "algorithm": self.algorithm,
            "model_sha256": self.model_sha256,
            "feature_schema_sha256": self.feature_schema_sha256,
            "data_manifest_sha256": self.data_manifest_sha256,
            "training_config_sha256": self.training_config_sha256,
            "framework_fingerprint": self.framework_fingerprint,
        }, sort_keys=True, separators=(",", ":")).encode()
        return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    identity: ModelIdentity
    status: ModelStatus
    approved_at: datetime | None
    approved_by: str | None
    walkforward_passed: bool
    recorded_replay_passed: bool
    shadow_passed: bool
    fallback_profile: str

    @property
    def deployable(self) -> bool:
        return (
            self.status is ModelStatus.APPROVED
            and self.approved_at is not None
            and bool(self.approved_by)
            and self.walkforward_passed
            and self.recorded_replay_passed
            and self.shadow_passed
            and bool(self.fallback_profile.strip())
        )


@dataclass(frozen=True, slots=True)
class InferenceHealth:
    latency_ms: float
    finite: bool
    feature_schema_sha256: str
    model_sha256: str
    drift_score: float
    max_latency_ms: float = 100.0
    max_drift_score: float = 0.25


@dataclass(frozen=True, slots=True)
class ModelRuntimeDecision:
    use_model: bool
    fallback_profile: str
    reasons: tuple[str, ...]


def decide_model_runtime(record: ApprovalRecord, health: InferenceHealth) -> ModelRuntimeDecision:
    reasons: list[str] = []
    if not record.deployable:
        reasons.append("MODEL_NOT_APPROVED")
    if not health.finite:
        reasons.append("MODEL_NONFINITE")
    if health.latency_ms < 0 or health.latency_ms > health.max_latency_ms:
        reasons.append("MODEL_LATENCY_BUDGET")
    if health.drift_score < 0 or health.drift_score > health.max_drift_score:
        reasons.append("MODEL_DRIFT_LIMIT")
    if health.feature_schema_sha256 != record.identity.feature_schema_sha256:
        reasons.append("FEATURE_SCHEMA_MISMATCH")
    if health.model_sha256 != record.identity.model_sha256:
        reasons.append("MODEL_HASH_MISMATCH")
    return ModelRuntimeDecision(not reasons, record.fallback_profile, tuple(reasons))


@dataclass(frozen=True, slots=True)
class ModelCircuitPolicy:
    consecutive_failures_to_open: int = 3
    consecutive_successes_to_close: int = 5
    cooldown_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.consecutive_failures_to_open <= 0 or self.consecutive_successes_to_close <= 0:
            raise ValueError("model circuit thresholds must be positive")
        if self.cooldown_seconds <= 0:
            raise ValueError("model circuit cooldown must be positive")


@dataclass(frozen=True, slots=True)
class ModelCircuitSnapshot:
    open: bool
    consecutive_failures: int
    consecutive_successes: int
    opened_at: datetime | None
    reasons: tuple[str, ...]


class ModelCircuitBreaker:
    """Stateful failover guard for repeated inference health failures.

    A single transient inference problem may use the deterministic fallback without
    permanently opening the circuit.  Repeated failures open it; closing requires both
    cooldown expiry and a streak of healthy probes, preventing rapid model flapping.
    """

    def __init__(self, policy: ModelCircuitPolicy | None = None) -> None:
        self.policy = policy or ModelCircuitPolicy()
        self._open = False
        self._failures = 0
        self._successes = 0
        self._opened_at: datetime | None = None
        self._reasons: tuple[str, ...] = ()

    def observe(
        self,
        decision: ModelRuntimeDecision,
        *,
        now: datetime,
    ) -> ModelCircuitSnapshot:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now = now.astimezone(UTC)
        if decision.use_model:
            self._failures = 0
            self._successes += 1
            if self._open and self._opened_at is not None:
                elapsed = (now - self._opened_at).total_seconds()
                if (
                    elapsed >= self.policy.cooldown_seconds
                    and self._successes >= self.policy.consecutive_successes_to_close
                ):
                    self._open = False
                    self._opened_at = None
                    self._reasons = ()
        else:
            self._successes = 0
            self._failures += 1
            self._reasons = decision.reasons
            if self._failures >= self.policy.consecutive_failures_to_open:
                self._open = True
                self._opened_at = self._opened_at or now
        return self.snapshot()

    def snapshot(self) -> ModelCircuitSnapshot:
        return ModelCircuitSnapshot(
            self._open,
            self._failures,
            self._successes,
            self._opened_at,
            self._reasons,
        )


@dataclass(frozen=True, slots=True)
class FallbackProfile:
    name: str
    config_sha256: str
    approved: bool
    max_long_ratio: float
    max_short_ratio: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("fallback profile name is required")
        digest = self.config_sha256.lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("fallback config_sha256 must be sha256")
        if not (0 <= self.max_long_ratio <= 1) or not (0 <= self.max_short_ratio <= 1):
            raise ValueError("fallback exposure ratios must be in [0,1]")
        object.__setattr__(self, "config_sha256", digest)


class FallbackProfileRegistry:
    def __init__(self, profiles: tuple[FallbackProfile, ...] = ()) -> None:
        self._profiles: dict[str, FallbackProfile] = {}
        for profile in profiles:
            self.register(profile)

    def register(self, profile: FallbackProfile) -> None:
        if profile.name in self._profiles and self._profiles[profile.name] != profile:
            raise ValueError("fallback profile name already registered with different content")
        self._profiles[profile.name] = profile

    def approved(self, name: str) -> bool:
        profile = self._profiles.get(name)
        return bool(profile and profile.approved)

    def resolve(self, name: str) -> FallbackProfile:
        profile = self._profiles[name]
        if not profile.approved:
            raise PermissionError("fallback profile is not approved")
        return profile
