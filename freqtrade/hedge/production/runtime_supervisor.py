"""Runtime safety epoch shared by production control and the final exchange write gate."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from threading import RLock

from .control import ControlMode
from .model_governance import ModelCircuitSnapshot
from .observability import AlertState
from .reconciliation_runtime import ReconciliationSupervisorSnapshot


@dataclass(frozen=True, slots=True)
class RuntimeSafetyInput:
    control_mode: ControlMode
    reconciliation: ReconciliationSupervisorSnapshot
    active_alerts: tuple[AlertState, ...] = ()
    incident_blocks_new_risk: bool = False
    incident_blocks_account: bool = False
    market_data_fresh: bool = True
    risk_data_fresh: bool = True
    model_circuit: ModelCircuitSnapshot | None = None
    deterministic_fallback_ready: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeSafetySnapshot:
    safety_epoch: int
    observed_at: datetime
    allows_new_risk: bool
    allows_reduce: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.safety_epoch < 0:
            raise ValueError("safety_epoch must be nonnegative")
        if self.observed_at.tzinfo is None:
            raise ValueError("runtime safety observed_at must be timezone-aware")
        object.__setattr__(self, "observed_at", self.observed_at.astimezone(UTC))
        object.__setattr__(self, "reasons", tuple(dict.fromkeys(str(x) for x in self.reasons)))


class ProductionRuntimeSafetySupervisor:
    """Fail-closed runtime state with monotonic epoch invalidation.

    The epoch advances whenever the runtime transitions into a blocked state or the set
    of blocking reasons changes.  An execution gate records the current epoch at arm time;
    a later fault therefore invalidates the armed session immediately, even if evidence
    leases and the operator arm token have not expired yet.
    """

    def __init__(self) -> None:
        self._lock = RLock()
        self._epoch = 0
        self._last_reasons: tuple[str, ...] = ("RUNTIME_NOT_OBSERVED",)
        self._snapshot: RuntimeSafetySnapshot | None = None

    def observe(self, value: RuntimeSafetyInput, *, now: datetime) -> RuntimeSafetySnapshot:
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now = now.astimezone(UTC)
        reasons: list[str] = []
        if value.control_mode is ControlMode.HALT:
            reasons.append("CONTROL_HALT")
        if value.control_mode is ControlMode.CLOSE_ONLY:
            reasons.append("CONTROL_CLOSE_ONLY")
        if value.control_mode is ControlMode.PAUSE_NEW_RISK:
            reasons.append("CONTROL_PAUSE_NEW_RISK")
        if not value.reconciliation.allow_new_risk:
            reasons.append("RECONCILIATION_BLOCKS_NEW_RISK")
        if not value.reconciliation.allow_reduce:
            reasons.append("RECONCILIATION_BLOCKS_REDUCE")
        if value.incident_blocks_account:
            reasons.append("ACCOUNT_INCIDENT")
        elif value.incident_blocks_new_risk:
            reasons.append("NEW_RISK_INCIDENT")
        if not value.market_data_fresh:
            reasons.append("MARKET_DATA_STALE")
        if not value.risk_data_fresh:
            reasons.append("RISK_DATA_STALE")
        if (
            value.model_circuit is not None
            and value.model_circuit.open
            and not value.deterministic_fallback_ready
        ):
            reasons.append("MODEL_CIRCUIT_OPEN")
        for alert in value.active_alerts:
            if alert.active and alert.severity is not None:
                reasons.append(f"ALERT:{alert.code}:{alert.severity.value}")
        normalized = tuple(dict.fromkeys(reasons))
        hard_reduce_block = any(
            x in normalized
            for x in ("CONTROL_HALT", "RECONCILIATION_BLOCKS_REDUCE", "ACCOUNT_INCIDENT")
        ) or any(x.endswith(":HALT_ACCOUNT") for x in normalized)
        new_risk_block = bool(normalized)
        with self._lock:
            if normalized != self._last_reasons:
                self._epoch += 1
                self._last_reasons = normalized
            snapshot = RuntimeSafetySnapshot(
                self._epoch,
                now,
                not new_risk_block,
                not hard_reduce_block,
                normalized,
            )
            self._snapshot = snapshot
            return snapshot

    def snapshot(self) -> RuntimeSafetySnapshot:
        with self._lock:
            if self._snapshot is None:
                raise RuntimeError("runtime safety has not been observed yet")
            return self._snapshot
