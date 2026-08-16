"""Production health snapshot and fail-closed alert evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import isfinite

from .contracts import Severity


class HealthDomain(StrEnum):
    ACCOUNT = "ACCOUNT"
    POSITION = "POSITION"
    EXECUTION = "EXECUTION"
    DATA = "DATA"
    SYSTEM = "SYSTEM"
    MODEL = "MODEL"
    RISK = "RISK"
    DATABASE = "DATABASE"


@dataclass(frozen=True, slots=True)
class HealthSnapshot:
    available_margin_ratio: float
    liquidation_buffer_ratio: float
    unknown_orders: int
    position_divergences: int
    market_data_age_seconds: float
    user_stream_age_seconds: float
    loop_p99_ms: float
    db_p99_ms: float
    model_p99_ms: float
    model_fallbacks_1h: int
    risk_reject_ratio_1h: float
    memory_growth_ratio_1h: float
    outbox_backlog: int = 0
    fencing_mismatches: int = 0
    database_disconnects_1h: int = 0
    disk_free_ratio: float = 1.0
    evidence_chain_valid: bool = True
    unknown_recovery_oldest_seconds: float = 0.0

    def __post_init__(self) -> None:
        integer_fields = (
            "unknown_orders", "position_divergences", "model_fallbacks_1h",
            "outbox_backlog", "fencing_mismatches", "database_disconnects_1h",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        numeric_fields = (
            "available_margin_ratio", "liquidation_buffer_ratio",
            "market_data_age_seconds", "user_stream_age_seconds", "loop_p99_ms",
            "db_p99_ms", "model_p99_ms", "risk_reject_ratio_1h",
            "memory_growth_ratio_1h", "disk_free_ratio",
            "unknown_recovery_oldest_seconds",
        )
        for name in numeric_fields:
            value = float(getattr(self, name))
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        for name in ("available_margin_ratio", "liquidation_buffer_ratio", "risk_reject_ratio_1h", "disk_free_ratio"):
            value = getattr(self, name)
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0,1]")
        for name in ("market_data_age_seconds", "user_stream_age_seconds", "loop_p99_ms", "db_p99_ms", "model_p99_ms", "unknown_recovery_oldest_seconds"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True, slots=True)
class Alert:
    domain: HealthDomain
    code: str
    severity: Severity
    value: float
    threshold: float


@dataclass(frozen=True, slots=True)
class ObservabilityPolicy:
    min_available_margin_ratio: float = 0.15
    min_liquidation_buffer_ratio: float = 0.25
    max_market_data_age_seconds: float = 10.0
    max_user_stream_age_seconds: float = 30.0
    max_loop_p99_ms: float = 250.0
    max_db_p99_ms: float = 100.0
    max_model_p99_ms: float = 100.0
    max_risk_reject_ratio_1h: float = 0.50
    max_memory_growth_ratio_1h: float = 0.10
    max_outbox_backlog: int = 100
    max_database_disconnects_1h: int = 3
    min_disk_free_ratio: float = 0.10
    max_unknown_recovery_seconds: float = 120.0

    def __post_init__(self) -> None:
        for name in (
            "min_available_margin_ratio", "min_liquidation_buffer_ratio",
            "max_market_data_age_seconds", "max_user_stream_age_seconds",
            "max_loop_p99_ms", "max_db_p99_ms", "max_model_p99_ms",
            "max_risk_reject_ratio_1h", "max_memory_growth_ratio_1h",
            "min_disk_free_ratio", "max_unknown_recovery_seconds",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)
        for name in ("min_available_margin_ratio", "min_liquidation_buffer_ratio", "max_risk_reject_ratio_1h", "min_disk_free_ratio"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1")
        for name in ("max_outbox_backlog", "max_database_disconnects_1h"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


def evaluate_health(snapshot: HealthSnapshot, policy: ObservabilityPolicy | None = None) -> tuple[Alert, ...]:
    policy = policy or ObservabilityPolicy()
    alerts: list[Alert] = []
    def low(domain: HealthDomain, code: str, value: float, threshold: float, severity: Severity) -> None:
        if value < threshold: alerts.append(Alert(domain, code, severity, value, threshold))
    def high(domain: HealthDomain, code: str, value: float, threshold: float, severity: Severity) -> None:
        if value > threshold: alerts.append(Alert(domain, code, severity, value, threshold))
    low(HealthDomain.ACCOUNT, "AVAILABLE_MARGIN_LOW", snapshot.available_margin_ratio, policy.min_available_margin_ratio, Severity.HALT_NEW_RISK)
    low(HealthDomain.ACCOUNT, "LIQUIDATION_BUFFER_LOW", snapshot.liquidation_buffer_ratio, policy.min_liquidation_buffer_ratio, Severity.HALT_ACCOUNT)
    if snapshot.unknown_orders > 0: alerts.append(Alert(HealthDomain.EXECUTION, "UNKNOWN_ORDER", Severity.HALT_ACCOUNT, float(snapshot.unknown_orders), 0.0))
    if snapshot.position_divergences > 0: alerts.append(Alert(HealthDomain.POSITION, "POSITION_DIVERGENCE", Severity.HALT_NEW_RISK, float(snapshot.position_divergences), 0.0))
    high(HealthDomain.DATA, "MARKET_DATA_STALE", snapshot.market_data_age_seconds, policy.max_market_data_age_seconds, Severity.HALT_NEW_RISK)
    high(HealthDomain.DATA, "USER_STREAM_STALE", snapshot.user_stream_age_seconds, policy.max_user_stream_age_seconds, Severity.HALT_NEW_RISK)
    high(HealthDomain.SYSTEM, "LOOP_LATENCY", snapshot.loop_p99_ms, policy.max_loop_p99_ms, Severity.WARNING)
    high(HealthDomain.DATABASE, "DB_LATENCY", snapshot.db_p99_ms, policy.max_db_p99_ms, Severity.HALT_NEW_RISK)
    high(HealthDomain.MODEL, "MODEL_LATENCY", snapshot.model_p99_ms, policy.max_model_p99_ms, Severity.WARNING)
    high(HealthDomain.RISK, "RISK_REJECT_RATE", snapshot.risk_reject_ratio_1h, policy.max_risk_reject_ratio_1h, Severity.WARNING)
    high(HealthDomain.SYSTEM, "MEMORY_GROWTH", snapshot.memory_growth_ratio_1h, policy.max_memory_growth_ratio_1h, Severity.WARNING)
    if snapshot.outbox_backlog > policy.max_outbox_backlog:
        alerts.append(Alert(HealthDomain.EXECUTION, "OUTBOX_BACKLOG", Severity.HALT_NEW_RISK, float(snapshot.outbox_backlog), float(policy.max_outbox_backlog)))
    if snapshot.fencing_mismatches > 0:
        alerts.append(Alert(HealthDomain.DATABASE, "FENCING_MISMATCH", Severity.HALT_ACCOUNT, float(snapshot.fencing_mismatches), 0.0))
    if snapshot.database_disconnects_1h > policy.max_database_disconnects_1h:
        alerts.append(Alert(HealthDomain.DATABASE, "DATABASE_DISCONNECT_STORM", Severity.HALT_NEW_RISK, float(snapshot.database_disconnects_1h), float(policy.max_database_disconnects_1h)))
    low(HealthDomain.SYSTEM, "DISK_FREE_LOW", snapshot.disk_free_ratio, policy.min_disk_free_ratio, Severity.HALT_NEW_RISK)
    if not snapshot.evidence_chain_valid:
        alerts.append(Alert(HealthDomain.SYSTEM, "EVIDENCE_CHAIN_INVALID", Severity.HALT_ACCOUNT, 1.0, 0.0))
    high(HealthDomain.EXECUTION, "UNKNOWN_RECOVERY_SLA", snapshot.unknown_recovery_oldest_seconds, policy.max_unknown_recovery_seconds, Severity.HALT_ACCOUNT)
    return tuple(alerts)


@dataclass(frozen=True, slots=True)
class AlertHysteresisPolicy:
    raise_after: int = 2
    clear_after: int = 3

    def __post_init__(self) -> None:
        if self.raise_after <= 0 or self.clear_after <= 0:
            raise ValueError("alert hysteresis thresholds must be positive")


@dataclass(frozen=True, slots=True)
class AlertState:
    code: str
    active: bool
    consecutive_bad: int
    consecutive_good: int
    severity: Severity | None


class AlertStateTracker:
    """Debounce transient metric spikes without delaying first account-halting anomalies."""

    def __init__(self, policy: AlertHysteresisPolicy | None = None) -> None:
        self.policy = policy or AlertHysteresisPolicy()
        self._states: dict[str, AlertState] = {}

    def observe(self, alerts: tuple[Alert, ...]) -> tuple[AlertState, ...]:
        incoming = {item.code: item for item in alerts}
        known = set(self._states) | set(incoming)
        out: list[AlertState] = []
        for code in sorted(known):
            previous = self._states.get(code, AlertState(code, False, 0, 0, None))
            alert = incoming.get(code)
            if alert is not None:
                bad = previous.consecutive_bad + 1
                immediate = alert.severity is Severity.HALT_ACCOUNT
                active = previous.active or immediate or bad >= self.policy.raise_after
                state = AlertState(code, active, bad, 0, alert.severity)
            else:
                good = previous.consecutive_good + 1
                active = previous.active and good < self.policy.clear_after
                state = AlertState(code, active, 0, good, previous.severity if active else None)
            self._states[code] = state
            out.append(state)
        return tuple(out)

    @property
    def active(self) -> tuple[AlertState, ...]:
        return tuple(x for x in self._states.values() if x.active)
