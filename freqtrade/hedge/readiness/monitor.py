"""Mutable runtime aggregation for the unified ReadinessGate."""

from __future__ import annotations

from dataclasses import fields, replace
from threading import RLock
from typing import TYPE_CHECKING, Any

from freqtrade.hedge.readiness.checks import ReadinessInputs
from freqtrade.hedge.readiness.gate import ReadinessGate
from freqtrade.hedge.readiness.state import ReadinessReport

if TYPE_CHECKING:
    from freqtrade.hedge.concurrency.single_writer import SingleWriterGuard


_INPUT_FIELDS = frozenset(item.name for item in fields(ReadinessInputs))


class ReadinessMonitor:
    """Maintain readiness inputs and evaluate them as one atomic snapshot."""

    def __init__(
        self,
        *,
        gate: ReadinessGate,
        inputs: ReadinessInputs,
        writer: "SingleWriterGuard | None" = None,
    ) -> None:
        self._gate = gate
        self._inputs = inputs
        self._writer = writer
        self._lock = RLock()

    @property
    def gate(self) -> ReadinessGate:
        return self._gate

    @property
    def inputs(self) -> ReadinessInputs:
        with self._lock:
            return self._inputs

    @property
    def report(self) -> ReadinessReport:
        return self._gate.report

    def bind_writer(self, writer: "SingleWriterGuard | None") -> ReadinessReport:
        with self._lock:
            self._writer = writer
        return self.refresh()

    def update(self, **changes: Any) -> ReadinessReport:
        unknown = set(changes) - _INPUT_FIELDS
        if unknown:
            raise ValueError(f"Unknown readiness input fields: {sorted(unknown)!r}.")
        with self._lock:
            self._inputs = replace(self._inputs, **changes)
        return self.refresh()

    def set_halt_reason(self, reason_code: str) -> ReadinessReport:
        if not isinstance(reason_code, str) or not reason_code.strip():
            raise ValueError("reason_code must be a non-empty string.")
        reason = reason_code.strip()
        with self._lock:
            reasons = tuple(dict.fromkeys((*self._inputs.halt_reasons, reason)))
            self._inputs = replace(self._inputs, halt_reasons=reasons)
        return self.refresh()

    def clear_halt_reason(self, reason_code: str | None = None) -> ReadinessReport:
        with self._lock:
            if reason_code is None:
                reasons: tuple[str, ...] = ()
            else:
                if not isinstance(reason_code, str) or not reason_code.strip():
                    raise ValueError("reason_code must be a non-empty string.")
                normalized = reason_code.strip()
                reasons = tuple(
                    item for item in self._inputs.halt_reasons if item != normalized
                )
            self._inputs = replace(self._inputs, halt_reasons=reasons)
        return self.refresh()

    def _refresh_locked(self) -> ReadinessReport:
        current = self._inputs
        writer = self._writer
        if writer is not None:
            try:
                writer_valid = writer.status().valid
            except Exception:
                writer_valid = False
            current = replace(current, single_writer_lease_valid=writer_valid)
            self._inputs = current
        return self._gate.evaluate(current)

    def refresh(self) -> ReadinessReport:
        # Keep input replacement and report evaluation under one monitor lock so
        # a concurrent update cannot be overwritten by an older writer status.
        with self._lock:
            return self._refresh_locked()

    def snapshot(self, *, refresh: bool = True) -> dict[str, object]:
        with self._lock:
            report = self._refresh_locked() if refresh else self._gate.report
            inputs = self._inputs
            return {
                "inputs": inputs.as_dict(),
                "report": report.as_dict(),
            }
