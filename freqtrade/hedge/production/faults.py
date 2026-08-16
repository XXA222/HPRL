"""Deterministic production fault-injection catalog and acceptance policy."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable


class FaultScenario(StrEnum):
    HTTP_TIMEOUT_AFTER_ACCEPT = "HTTP_TIMEOUT_AFTER_ACCEPT"
    QUERY_TIMEOUT = "QUERY_TIMEOUT"
    HTTP_429 = "HTTP_429"
    HTTP_5XX = "HTTP_5XX"
    WS_DISCONNECT = "WS_DISCONNECT"
    WS_DUPLICATE = "WS_DUPLICATE"
    WS_OUT_OF_ORDER = "WS_OUT_OF_ORDER"
    WS_SEQUENCE_GAP = "WS_SEQUENCE_GAP"
    DB_CONNECTION_LOSS = "DB_CONNECTION_LOSS"
    DB_DEADLOCK = "DB_DEADLOCK"
    DB_PAUSE = "DB_PAUSE"
    PROCESS_CRASH_BEFORE_COMMIT = "PROCESS_CRASH_BEFORE_COMMIT"
    PROCESS_CRASH_AFTER_COMMIT = "PROCESS_CRASH_AFTER_COMMIT"
    PROCESS_CRASH_AFTER_SUBMIT = "PROCESS_CRASH_AFTER_SUBMIT"
    PROCESS_CRASH_AFTER_FILL = "PROCESS_CRASH_AFTER_FILL"
    CANCEL_FILL_RACE = "CANCEL_FILL_RACE"
    PARTIAL_FILL = "PARTIAL_FILL"
    STALE_MARKET_DATA = "STALE_MARKET_DATA"
    CLOCK_SKEW = "CLOCK_SKEW"
    DUPLICATE_FILL = "DUPLICATE_FILL"
    MANUAL_EXTERNAL_ORDER = "MANUAL_EXTERNAL_ORDER"
    POSITION_DRIFT = "POSITION_DRIFT"
    CHECKPOINT_CORRUPTION = "CHECKPOINT_CORRUPTION"
    FENCING_TOKEN_STALE = "FENCING_TOKEN_STALE"
    DNS_FAILURE = "DNS_FAILURE"
    TLS_FAILURE = "TLS_FAILURE"
    CONNECTION_RESET = "CONNECTION_RESET"
    PARTIAL_HTTP_BODY = "PARTIAL_HTTP_BODY"
    REST_STALE_SNAPSHOT = "REST_STALE_SNAPSHOT"
    FUNDING_SPIKE = "FUNDING_SPIKE"
    MARK_PRICE_GAP = "MARK_PRICE_GAP"
    EXCHANGE_FILTER_CHANGE = "EXCHANGE_FILTER_CHANGE"
    MULTI_WRITER_RACE = "MULTI_WRITER_RACE"
    OUTBOX_PUBLISHER_DOWN = "OUTBOX_PUBLISHER_DOWN"
    BACKUP_RESTORE_INTERRUPTED = "BACKUP_RESTORE_INTERRUPTED"
    API_CLOCK_DRIFT = "API_CLOCK_DRIFT"
    LIQUIDATION_DATA_MISSING = "LIQUIDATION_DATA_MISSING"
    MODEL_TIMEOUT_STORM = "MODEL_TIMEOUT_STORM"
    MODEL_NONFINITE = "MODEL_NONFINITE"
    DISK_FULL = "DISK_FULL"
    MEMORY_PRESSURE = "MEMORY_PRESSURE"
    PROCESS_PAUSE = "PROCESS_PAUSE"
    USER_STREAM_LISTEN_KEY_EXPIRE = "USER_STREAM_LISTEN_KEY_EXPIRE"
    EXTERNAL_POSITION_CHANGE = "EXTERNAL_POSITION_CHANGE"


@dataclass(frozen=True, slots=True)
class FaultResult:
    scenario: FaultScenario
    passed: bool
    duplicate_writes: int
    final_converged: bool
    new_risk_blocked_during_fault: bool
    recovery_seconds: float
    detail: str = ""
    state_hash_match: bool = True
    outbox_drained: bool = True
    fencing_preserved: bool = True


def evaluate_fault_campaign(results: Iterable[FaultResult], *, max_recovery_seconds: float = 30.0) -> tuple[bool, tuple[str, ...]]:
    rows = tuple(results)
    by = {item.scenario: item for item in rows}
    reasons: list[str] = []
    if len(by) != len(rows):
        reasons.append("DUPLICATE_FAULT_SCENARIO_RESULT")
    for scenario in FaultScenario:
        item = by.get(scenario)
        if item is None:
            reasons.append(f"MISSING:{scenario.value}")
            continue
        if not item.passed: reasons.append(f"FAILED:{scenario.value}")
        if item.duplicate_writes != 0: reasons.append(f"DUPLICATE_WRITE:{scenario.value}")
        if not item.final_converged: reasons.append(f"NOT_CONVERGED:{scenario.value}")
        if not item.new_risk_blocked_during_fault: reasons.append(f"NEW_RISK_NOT_BLOCKED:{scenario.value}")
        if item.recovery_seconds < 0 or item.recovery_seconds > max_recovery_seconds: reasons.append(f"RECOVERY_SLA:{scenario.value}")
        if not item.state_hash_match: reasons.append(f"STATE_HASH_MISMATCH:{scenario.value}")
        if not item.outbox_drained: reasons.append(f"OUTBOX_NOT_DRAINED:{scenario.value}")
        if not item.fencing_preserved: reasons.append(f"FENCING_VIOLATION:{scenario.value}")
    return not reasons, tuple(reasons)
