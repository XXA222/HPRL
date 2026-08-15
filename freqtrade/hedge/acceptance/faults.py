from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FaultClass(StrEnum):
    NETWORK = "NETWORK"
    RATE_LIMIT = "RATE_LIMIT"
    SERVER = "SERVER"
    AUTH = "AUTH"
    DATA = "DATA"


@dataclass(frozen=True, slots=True)
class FaultDecision:
    fault_class: FaultClass
    retryable: bool
    new_risk_allowed: bool
    requires_reconciliation: bool
    minimum_backoff_seconds: float


def classify_http_fault(status: int | None, *, network_error: bool = False) -> FaultDecision:
    if network_error or status is None:
        return FaultDecision(FaultClass.NETWORK, True, False, True, 1.0)
    if status in {401, 403}:
        return FaultDecision(FaultClass.AUTH, False, False, True, 0.0)
    if status in {418, 429}:
        return FaultDecision(FaultClass.RATE_LIMIT, True, False, True, 5.0)
    if 500 <= status <= 599:
        return FaultDecision(FaultClass.SERVER, True, False, True, 1.0)
    if 400 <= status <= 499:
        return FaultDecision(FaultClass.DATA, False, False, True, 0.0)
    return FaultDecision(FaultClass.DATA, False, False, False, 0.0)
