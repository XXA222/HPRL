"""Canonical Production Readiness Spine for Freqtrade-Hedge."""
from .contracts import (
    Capability,
    CapabilityLease,
    Decision,
    EvidenceKind,
    EvidenceStatus,
    GateResult,
    ProductionStage,
    Severity,
)
from .evidence import EvidenceLedger, EvidenceRecord
from .policy import ProductionPolicy, StageEvaluator

PRODUCTION_READINESS_API_VERSION = "1.1"
PRODUCTION_READINESS_RELEASE = "freqtrade-hedge-production-readiness-r1.1-deep200"
HPRL_V3_PRODUCTION_API_VERSION = "3.0"
HPRL_V3_PRODUCTION_RELEASE = "freqtrade-hedge-hprl-v3-production-integration-r2"
HPRL_V3_CLOSED_LOOP_API_VERSION = "3.1"
HPRL_V3_CLOSED_LOOP_RELEASE = "freqtrade-hedge-hprl-v3-closed-loop-r1"
HPRL_RUNTIME_CLOSURE_API_VERSION = "2.0"
HPRL_RUNTIME_CLOSURE_RELEASE = "freqtrade-hedge-hprl-v3-runtime-closure-r2"
HPRL_REAL_ENVIRONMENT_API_VERSION = "3.0"
HPRL_REAL_ENVIRONMENT_RELEASE = "freqtrade-hedge-hprl-v3-real-environment-r3"

__all__ = [
    "Capability", "CapabilityLease", "Decision", "EvidenceKind", "EvidenceLedger",
    "EvidenceRecord", "EvidenceStatus", "GateResult", "ProductionPolicy", "ProductionStage",
    "Severity", "StageEvaluator", "PRODUCTION_READINESS_API_VERSION",
    "PRODUCTION_READINESS_RELEASE", "HPRL_V3_PRODUCTION_API_VERSION",
    "HPRL_V3_PRODUCTION_RELEASE", "HPRL_V3_CLOSED_LOOP_API_VERSION",
    "HPRL_V3_CLOSED_LOOP_RELEASE", "HPRL_RUNTIME_CLOSURE_API_VERSION",
    "HPRL_RUNTIME_CLOSURE_RELEASE", "HPRL_REAL_ENVIRONMENT_API_VERSION",
    "HPRL_REAL_ENVIRONMENT_RELEASE",
]
