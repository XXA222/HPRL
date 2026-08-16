"""Cumulative stage evaluator and short-lived capability lease issuer."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from .contracts import (
    Capability,
    CapabilityLease,
    EvidenceKind,
    EvidenceStatus,
    GateResult,
    ProductionStage,
    cumulative_requirements,
)
from .evidence import EvidenceLedger


CAPABILITY_STAGE = {
    Capability.READ_ONLY: ProductionStage.SOURCE_READY,
    Capability.PAPER: ProductionStage.REPLAY_READY,
    Capability.SHADOW: ProductionStage.SHADOW_24H,
    Capability.TESTNET_WRITE: ProductionStage.TESTNET_READY,
    Capability.LIVE_REDUCE: ProductionStage.SOURCE_READY,
    Capability.LIVE_CANARY_RISK: ProductionStage.LIVE_CANDIDATE,
    Capability.LIVE_NEW_RISK: ProductionStage.LIVE_READY,
}


@dataclass(frozen=True, slots=True)
class ProductionPolicy:
    lease_ttl: timedelta = timedelta(minutes=5)
    max_evidence_age: timedelta = timedelta(days=7)
    max_future_skew: timedelta = timedelta(seconds=5)

    def __post_init__(self) -> None:
        if self.lease_ttl <= timedelta(0) or self.lease_ttl > timedelta(hours=1):
            raise ValueError("lease_ttl must be in (0, 1h]")
        if self.max_evidence_age <= timedelta(0):
            raise ValueError("max_evidence_age must be positive")
        if self.max_future_skew < timedelta(0) or self.max_future_skew > timedelta(minutes=5):
            raise ValueError("max_future_skew must be in [0, 5m]")


class StageEvaluator:
    def __init__(self, ledger: EvidenceLedger, policy: ProductionPolicy | None = None) -> None:
        self.ledger = ledger
        self.policy = policy or ProductionPolicy()
        self._generation = 0

    def evaluate(self, stage: ProductionStage, *, now: datetime) -> GateResult:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now = now.astimezone(UTC)
        required = cumulative_requirements(stage)
        latest = self.ledger.latest_map()
        missing: list[EvidenceKind] = []
        failed: list[EvidenceKind] = []
        stale: list[EvidenceKind] = []
        reasons: list[str] = []
        for kind in sorted(required, key=lambda x: x.value):
            record = latest.get(kind)
            if record is None:
                missing.append(kind); continue
            if record.status is not EvidenceStatus.PASS:
                failed.append(kind)
            if record.observed_at > now + self.policy.max_future_skew:
                stale.append(kind)
                reasons.append(f"EVIDENCE_FROM_FUTURE:{kind.value}")
            if not record.is_fresh(now) or now - record.observed_at > self.policy.max_evidence_age:
                stale.append(kind)
        if not self.ledger.verify_chain(): reasons.append("EVIDENCE_CHAIN_INVALID")
        passed = not missing and not failed and not stale and not reasons
        return GateResult(
            stage,
            passed,
            tuple(missing),
            tuple(failed),
            tuple(dict.fromkeys(stale)),
            tuple(dict.fromkeys(reasons)),
            self.ledger.digest(),
        )

    def issue_lease(self, capability: Capability, *, actor: str, now: datetime) -> CapabilityLease:
        actor = actor.strip()
        if not actor: raise ValueError("actor is required")
        stage = CAPABILITY_STAGE[capability]
        result = self.evaluate(stage, now=now)
        if not result.passed:
            raise PermissionError(f"capability {capability.value} blocked: {result}")
        self._generation += 1
        now = now.astimezone(UTC)
        return CapabilityLease(capability, self._generation, stage, now, now + self.policy.lease_ttl, result.evidence_digest, actor)
