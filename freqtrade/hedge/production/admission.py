"""Final production admission decision before an intent can reach Execution."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .contracts import Capability, CapabilityLease, Decision
from .control import ProductionControlPlane
from .model_governance import ModelRuntimeDecision
from .reconciliation import ReconciliationResult
from .risk_envelope import RiskEnvelopeDecision, RiskDirection


@dataclass(frozen=True, slots=True)
class AdmissionContext:
    lease: CapabilityLease
    control: ProductionControlPlane
    reconciliation: ReconciliationResult
    risk: RiskEnvelopeDecision
    model: ModelRuntimeDecision | None
    market_data_fresh: bool
    risk_data_fresh: bool
    incident_blocks_new_risk: bool
    now: datetime
    direction: RiskDirection
    fallback_profile_approved: bool = False


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    decision: Decision
    reasons: tuple[str, ...]
    approved_notional: str


def admit(ctx: AdmissionContext) -> AdmissionDecision:
    reasons: list[str] = []
    is_reduce = ctx.direction is RiskDirection.REDUCE
    if not ctx.lease.valid_at(ctx.now): reasons.append("CAPABILITY_LEASE_EXPIRED")
    allowed_capabilities = (
        {Capability.LIVE_REDUCE, Capability.TESTNET_WRITE}
        if is_reduce
        else {Capability.LIVE_CANARY_RISK, Capability.LIVE_NEW_RISK, Capability.TESTNET_WRITE}
    )
    if ctx.lease.capability not in allowed_capabilities:
        reasons.append("CAPABILITY_DIRECTION_MISMATCH")
    if not ctx.market_data_fresh: reasons.append("MARKET_DATA_STALE")
    if not ctx.risk_data_fresh: reasons.append("RISK_DATA_STALE")
    if is_reduce:
        if not ctx.control.allows_reduce: reasons.append("CONTROL_BLOCKS_REDUCE")
        if not ctx.reconciliation.allow_reduce: reasons.append("RECONCILIATION_BLOCKS_REDUCE")
    else:
        if not ctx.control.allows_new_risk: reasons.append("CONTROL_BLOCKS_NEW_RISK")
        if not ctx.reconciliation.allow_new_risk: reasons.append("RECONCILIATION_BLOCKS_NEW_RISK")
        if ctx.incident_blocks_new_risk: reasons.append("OPEN_INCIDENT_BLOCKS_NEW_RISK")
        if ctx.model is not None and not ctx.model.use_model and not ctx.fallback_profile_approved:
            reasons.append("MODEL_FALLBACK_NOT_APPROVED")
    if ctx.risk.decision in {Decision.REJECT, Decision.HALT}:
        reasons.extend(ctx.risk.reasons or ("RISK_REJECTED",))
    if reasons:
        return AdmissionDecision(Decision.REJECT if not any("HALT" in x for x in reasons) else Decision.HALT, tuple(dict.fromkeys(reasons)), "0")
    return AdmissionDecision(ctx.risk.decision, (), str(ctx.risk.approved_notional))
