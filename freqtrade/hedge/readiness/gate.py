"""Unified startup and runtime ReadinessGate with scoped action decisions."""

from __future__ import annotations

import time
from threading import RLock
from typing import Callable

from freqtrade.hedge.readiness.checks import ReadinessInputs, run_readiness_checks
from freqtrade.hedge.readiness.state import (
    ReadinessCheckResult,
    ReadinessReasonCode,
    ReadinessReport,
    ReadinessSeverity,
    ReadinessState,
    reason_policy,
)
from freqtrade.hedge.identity import RiskPositionKey


class ReadinessGate:
    def __init__(self, *, clock_ms: Callable[[], int] | None = None) -> None:
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._lock = RLock()
        try:
            initial_now = self._now_ms()
        except Exception:
            self._report = self._clock_failure(previous_timestamp=0)
        else:
            self._report = ReadinessReport(
                ReadinessState.STARTING,
                (),
                (),
                initial_now,
                False,
            )

    def _now_ms(self) -> int:
        value = self._clock_ms()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Readiness clock must return nonnegative integer milliseconds.")
        return value

    @staticmethod
    def _clock_failure(*, previous_timestamp: int) -> ReadinessReport:
        check = ReadinessCheckResult(
            "readiness_clock",
            False,
            ReadinessReasonCode.READINESS_CLOCK_INVALID,
            "Clock source did not return valid milliseconds.",
        )
        return ReadinessReport(
            ReadinessState.NOT_READY,
            (check,),
            (ReadinessReasonCode.READINESS_CLOCK_INVALID,),
            previous_timestamp,
            False,
        )

    @property
    def report(self) -> ReadinessReport:
        with self._lock:
            return self._report

    def evaluate(self, inputs: ReadinessInputs) -> ReadinessReport:
        try:
            evaluated_at_ms = self._now_ms()
        except Exception:
            with self._lock:
                self._report = self._clock_failure(
                    previous_timestamp=self._report.evaluated_at_ms
                )
                return self._report

        checks = run_readiness_checks(inputs, now_ms=evaluated_at_ms)
        reasons = tuple(
            check.reason_code
            for check in checks
            if not check.passed and check.reason_code is not None
        )
        severities = [reason_policy(code).severity for code in reasons]
        if not reasons:
            state = ReadinessState.READY
        elif any(
            severity in {ReadinessSeverity.HALT_ACCOUNT, ReadinessSeverity.HALT_WRITE}
            for severity in severities
        ):
            state = ReadinessState.HALT
        else:
            state = ReadinessState.DEGRADED

        non_positional_reasons = tuple(
            code
            for code in reasons
            if code is not ReadinessReasonCode.UNKNOWN_ORDER_PRESENT
        )
        emergency_allowed = bool(reasons) and all(
            reason_policy(code).controlled_reduce_allowed for code in non_positional_reasons
        )
        if not reasons:
            emergency_allowed = True
        blocked = tuple(dict.fromkeys(item.position_key for item in inputs.unknown_orders))
        report = ReadinessReport(
            state,
            checks,
            reasons,
            evaluated_at_ms,
            emergency_allowed,
            blocked,
        )
        with self._lock:
            self._report = report
        return report

    def assert_ready(self) -> ReadinessReport:
        report = self.report
        if not report.ready:
            joined = ",".join(code.value for code in report.reason_codes)
            raise RuntimeError(f"ReadinessGate is not READY: {joined or report.state.value}")
        return report

    def allows_new_risk(self, position_key: RiskPositionKey | None = None) -> bool:
        report = self.report
        if report.ready:
            return True
        if position_key is None or report.is_position_blocked(position_key):
            return False
        return bool(report.reason_codes) and all(
            code is ReadinessReasonCode.UNKNOWN_ORDER_PRESENT
            for code in report.reason_codes
        )

    def allows_controlled_reduce(
        self,
        position_key: RiskPositionKey | None = None,
    ) -> bool:
        report = self.report
        if position_key is not None and report.is_position_blocked(position_key):
            return False
        return report.ready or report.emergency_reduce_only_allowed
