"""Frozen production-readiness contracts for the canonical Hedge runtime.

This package is deliberately orchestration-focused.  It does not replace the existing
execution/risk/replay implementations; it gives them one fail-closed promotion spine.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256
import json
from typing import Mapping


class ProductionStage(StrEnum):
    SOURCE_READY = "SOURCE_READY"
    DATABASE_READY = "DATABASE_READY"
    REPLAY_READY = "REPLAY_READY"
    SHADOW_24H = "SHADOW_24H"
    SHADOW_72H = "SHADOW_72H"
    TESTNET_READY = "TESTNET_READY"
    LIVE_CANDIDATE = "LIVE_CANDIDATE"
    LIVE_READY = "LIVE_READY"


STAGE_ORDER: tuple[ProductionStage, ...] = tuple(ProductionStage)


class Capability(StrEnum):
    READ_ONLY = "READ_ONLY"
    PAPER = "PAPER"
    SHADOW = "SHADOW"
    TESTNET_WRITE = "TESTNET_WRITE"
    LIVE_REDUCE = "LIVE_REDUCE"
    LIVE_CANARY_RISK = "LIVE_CANARY_RISK"
    LIVE_NEW_RISK = "LIVE_NEW_RISK"


class EvidenceKind(StrEnum):
    SOURCE_GATES = "SOURCE_GATES"
    CLEAN_MANIFEST = "CLEAN_MANIFEST"
    LEGACY_ISOLATION = "LEGACY_ISOLATION"
    POSTGRES_MIGRATION = "POSTGRES_MIGRATION"
    POSTGRES_CONCURRENCY = "POSTGRES_CONCURRENCY"
    BACKUP_RESTORE = "BACKUP_RESTORE"
    RECOVERY_MATRIX = "RECOVERY_MATRIX"
    RECONCILIATION = "RECONCILIATION"
    RECORDED_REPLAY = "RECORDED_REPLAY"
    BACKTEST_REALISM = "BACKTEST_REALISM"
    GOLDEN_STRATEGY = "GOLDEN_STRATEGY"
    SHADOW_24H = "SHADOW_24H"
    SHADOW_72H = "SHADOW_72H"
    FAULT_INJECTION = "FAULT_INJECTION"
    TESTNET_E2E = "TESTNET_E2E"
    CONTROL_PLANE = "CONTROL_PLANE"
    OBSERVABILITY = "OBSERVABILITY"
    MODEL_GOVERNANCE = "MODEL_GOVERNANCE"
    RISK_ENVELOPE = "RISK_ENVELOPE"
    SECURITY = "SECURITY"
    LIVE_CANDIDATE_APPROVAL = "LIVE_CANDIDATE_APPROVAL"
    LIVE_CANARY = "LIVE_CANARY"


class EvidenceStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    PENDING = "PENDING"


class Decision(StrEnum):
    APPROVE = "APPROVE"
    CLIP = "CLIP"
    REJECT = "REJECT"
    HALT = "HALT"


class Severity(StrEnum):
    INFO = "INFO"
    WARNING = "WARNING"
    HALT_NEW_RISK = "HALT_NEW_RISK"
    HALT_ACCOUNT = "HALT_ACCOUNT"


@dataclass(frozen=True, slots=True)
class CapabilityLease:
    capability: Capability
    generation: int
    stage: ProductionStage
    issued_at: datetime
    expires_at: datetime
    evidence_digest: str
    actor: str

    def valid_at(self, now: datetime) -> bool:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        return self.issued_at <= now < self.expires_at


@dataclass(frozen=True, slots=True)
class GateResult:
    stage: ProductionStage
    passed: bool
    missing: tuple[EvidenceKind, ...] = ()
    failed: tuple[EvidenceKind, ...] = ()
    stale: tuple[EvidenceKind, ...] = ()
    reasons: tuple[str, ...] = ()
    evidence_digest: str = ""


REQUIRED_EVIDENCE: Mapping[ProductionStage, frozenset[EvidenceKind]] = {
    ProductionStage.SOURCE_READY: frozenset(
        {EvidenceKind.SOURCE_GATES, EvidenceKind.CLEAN_MANIFEST, EvidenceKind.LEGACY_ISOLATION}
    ),
    ProductionStage.DATABASE_READY: frozenset(
        {
            EvidenceKind.SOURCE_GATES,
            EvidenceKind.CLEAN_MANIFEST,
            EvidenceKind.POSTGRES_MIGRATION,
            EvidenceKind.POSTGRES_CONCURRENCY,
            EvidenceKind.BACKUP_RESTORE,
            EvidenceKind.RECOVERY_MATRIX,
            EvidenceKind.RECONCILIATION,
        }
    ),
    ProductionStage.REPLAY_READY: frozenset(
        {
            EvidenceKind.RECORDED_REPLAY,
            EvidenceKind.BACKTEST_REALISM,
            EvidenceKind.GOLDEN_STRATEGY,
            EvidenceKind.RISK_ENVELOPE,
        }
    ),
    ProductionStage.SHADOW_24H: frozenset(
        {
            EvidenceKind.SHADOW_24H,
            EvidenceKind.FAULT_INJECTION,
            EvidenceKind.CONTROL_PLANE,
            EvidenceKind.OBSERVABILITY,
        }
    ),
    ProductionStage.SHADOW_72H: frozenset({EvidenceKind.SHADOW_72H}),
    ProductionStage.TESTNET_READY: frozenset(
        {EvidenceKind.TESTNET_E2E, EvidenceKind.SECURITY, EvidenceKind.RECONCILIATION}
    ),
    ProductionStage.LIVE_CANDIDATE: frozenset(
        {
            EvidenceKind.MODEL_GOVERNANCE,
            EvidenceKind.SECURITY,
            EvidenceKind.LIVE_CANDIDATE_APPROVAL,
        }
    ),
    # LIVE_CANARY is deliberately unique to LIVE_READY.  A candidate may receive
    # only a tightly bounded LIVE_CANARY_RISK lease; ordinary live new-risk
    # capability is impossible until real canary evidence has been recorded.
    ProductionStage.LIVE_READY: frozenset({EvidenceKind.LIVE_CANARY}),
}


def stages_through(stage: ProductionStage) -> tuple[ProductionStage, ...]:
    index = STAGE_ORDER.index(stage)
    return STAGE_ORDER[: index + 1]


def cumulative_requirements(stage: ProductionStage) -> frozenset[EvidenceKind]:
    required: set[EvidenceKind] = set()
    for item in stages_through(stage):
        required.update(REQUIRED_EVIDENCE[item])
    return frozenset(required)


def canonical_digest(payload: Mapping[str, object]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256(raw).hexdigest()


def utc_now() -> datetime:
    return datetime.now(UTC)
