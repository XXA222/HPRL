"""Canonical Production Readiness coordinator."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Mapping

from .contracts import Capability, EvidenceKind, EvidenceStatus, ProductionStage
from .evidence import EvidenceLedger, EvidenceRecord
from .policy import ProductionPolicy, StageEvaluator


@dataclass(slots=True)
class ProductionReadinessCoordinator:
    ledger: EvidenceLedger
    evaluator: StageEvaluator

    @classmethod
    def create(cls, *, policy: ProductionPolicy | None = None) -> "ProductionReadinessCoordinator":
        ledger = EvidenceLedger()
        return cls(ledger, StageEvaluator(ledger, policy))

    def record_pass(
        self,
        kind: EvidenceKind,
        *,
        observed_at: datetime,
        ttl: timedelta,
        producer: str,
        artifact_bytes: bytes,
        metadata: Mapping[str, object] | None = None,
    ) -> EvidenceRecord:
        return self.ledger.add(
            kind=kind,
            status=EvidenceStatus.PASS,
            observed_at=observed_at,
            ttl=ttl,
            artifact_sha256=sha256(artifact_bytes).hexdigest(),
            producer=producer,
            metadata=metadata,
        )

    def record_fail(
        self,
        kind: EvidenceKind,
        *,
        observed_at: datetime,
        ttl: timedelta,
        producer: str,
        artifact_bytes: bytes,
        metadata: Mapping[str, object] | None = None,
    ) -> EvidenceRecord:
        return self.ledger.add(
            kind=kind,
            status=EvidenceStatus.FAIL,
            observed_at=observed_at,
            ttl=ttl,
            artifact_sha256=sha256(artifact_bytes).hexdigest(),
            producer=producer,
            metadata=metadata,
        )

    def evaluate(self, stage: ProductionStage, *, now: datetime):
        return self.evaluator.evaluate(stage, now=now)

    def lease(self, capability: Capability, *, actor: str, now: datetime):
        return self.evaluator.issue_lease(capability, actor=actor, now=now)
