"""Operational service-level objectives for the production Hedge runtime."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SLOSnapshot:
    loop_p99_ms: float
    reconciliation_p99_s: float
    db_p99_ms: float
    order_ack_p99_ms: float
    recovery_p99_s: float
    duplicate_effects: int
    unresolved_unknown_orders: int


@dataclass(frozen=True, slots=True)
class SLOPolicy:
    loop_p99_ms: float = 250.0
    reconciliation_p99_s: float = 5.0
    db_p99_ms: float = 100.0
    order_ack_p99_ms: float = 2000.0
    recovery_p99_s: float = 30.0


def evaluate_slo(value: SLOSnapshot, policy: SLOPolicy | None = None) -> tuple[str, ...]:
    p = policy or SLOPolicy()
    reasons: list[str] = []
    if value.loop_p99_ms > p.loop_p99_ms: reasons.append("LOOP_P99")
    if value.reconciliation_p99_s > p.reconciliation_p99_s: reasons.append("RECONCILIATION_P99")
    if value.db_p99_ms > p.db_p99_ms: reasons.append("DB_P99")
    if value.order_ack_p99_ms > p.order_ack_p99_ms: reasons.append("ORDER_ACK_P99")
    if value.recovery_p99_s > p.recovery_p99_s: reasons.append("RECOVERY_P99")
    if value.duplicate_effects: reasons.append("DUPLICATE_EFFECTS")
    if value.unresolved_unknown_orders: reasons.append("UNKNOWN_ORDERS")
    return tuple(reasons)
