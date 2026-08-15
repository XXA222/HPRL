"""Dependency-free, bounded-label hedge metrics facade.

The facade intentionally keeps labels bounded and exposes a stable snapshot shape so it can
be adapted to Prometheus, OpenTelemetry or Freqtrade's existing metrics layer later.
"""

from __future__ import annotations

import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from threading import RLock
from urllib.parse import quote

_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_REASON = re.compile(r"[^A-Z0-9_]+")


class HedgeMetrics:
    REQUIRED_FAMILIES = (
        "intent_total",
        "order_transition_total",
        "fill_quantity_total",
        "drift_total",
        "halt_total",
        "reconnect_total",
        "lock_state",
        "order_ack_latency_seconds",
        "unknown_order_count",
        "unknown_order_age_seconds",
        "cancel_replace_total",
        "duplicate_fill_total",
        "outbox_backlog",
        "projection_version",
        "reconciliation_diff_total",
        "rest_error_total",
        "clock_skew_milliseconds",
        "gross_long_notional",
        "gross_short_notional",
        "net_notional",
        "gross_exposure_ratio",
        "margin_utilization",
        "pending_risk_notional",
        "single_writer_leader",
        "single_writer_lease_age_seconds",
        "lock_wait_seconds",
        "lock_timeout_total",
        "deadlock_total",
    )

    def __init__(self) -> None:
        self._counters: dict[
            tuple[str, tuple[tuple[str, str], ...]], Decimal
        ] = defaultdict(Decimal)
        self._gauges: dict[
            tuple[str, tuple[tuple[str, str], ...]], Decimal
        ] = {}
        self._lock = RLock()

    @staticmethod
    def _label(value: object, *, field_name: str) -> str:
        if isinstance(value, bool):
            result = "true" if value else "false"
        elif isinstance(value, (str, int)) and not isinstance(value, bool):
            result = str(value).strip()
        else:
            raise TypeError(f"metric label {field_name} has unsupported type")
        if not result or len(result) > 256 or _CONTROL.search(result):
            raise ValueError(f"metric label {field_name} is invalid")
        return quote(result, safe="-_.~")

    @classmethod
    def _key(
        cls,
        name: str,
        labels: dict[str, object],
    ) -> tuple[str, tuple[tuple[str, str], ...]]:
        if not isinstance(name, str) or not name:
            raise ValueError("metric name is required")
        normalized: list[tuple[str, str]] = []
        for key, value in labels.items():
            normalized.append(
                (
                    cls._label(key, field_name="key"),
                    cls._label(value, field_name=str(key)),
                )
            )
        return name, tuple(sorted(normalized))

    @staticmethod
    def _amount(value: Decimal | int) -> Decimal:
        if isinstance(value, (bool, float)):
            raise ValueError("metric amount must use an exact numeric value")
        try:
            amount = Decimal(value)
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("metric amount must be a valid decimal") from exc
        if not amount.is_finite() or amount < 0:
            raise ValueError("metric amount must be finite and non-negative")
        return amount

    @staticmethod
    def _signed_amount(value: Decimal | int) -> Decimal:
        if isinstance(value, (bool, float)):
            raise ValueError("metric amount must use an exact numeric value")
        try:
            amount = Decimal(value)
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("metric amount must be a valid decimal") from exc
        if not amount.is_finite():
            raise ValueError("metric amount must be finite")
        return amount

    @staticmethod
    def _bounded_reason(reason: str) -> str:
        if not isinstance(reason, str):
            raise TypeError("reason must be a string")
        normalized = _REASON.sub("_", reason.strip().upper()).strip("_")
        return normalized[:64] or "UNSPECIFIED"

    def _inc(
        self,
        name: str,
        amount: Decimal | int = 1,
        **labels: object,
    ) -> None:
        increment = self._amount(amount)
        with self._lock:
            self._counters[self._key(name, labels)] += increment

    def _set(
        self,
        name: str,
        value: Decimal | int,
        *,
        signed: bool = False,
        **labels: object,
    ) -> None:
        amount = self._signed_amount(value) if signed else self._amount(value)
        with self._lock:
            self._gauges[self._key(name, labels)] = amount

    def intent(self, outcome: str) -> None:
        self._inc("intent_total", outcome=outcome)

    def order(self, from_state: str, to_state: str) -> None:
        self._inc(
            "order_transition_total",
            from_state=from_state,
            to_state=to_state,
        )

    def fill(self, quantity: Decimal) -> None:
        self._inc("fill_quantity_total", quantity)

    def drift(self, kind: str) -> None:
        self._inc("drift_total", kind=kind)

    def halt(self, active: bool, reason: str = "") -> None:
        if not isinstance(active, bool):
            raise TypeError("active must be a boolean")
        # Keep the mandatory family cardinality bounded. Reason is tracked in a separate,
        # normalized family so arbitrary exception messages cannot explode labels.
        self._inc("halt_total", active=active)
        if active:
            self._inc("halt_reason_total", reason=self._bounded_reason(reason))

    def reconnect(self, source: str, outcome: str) -> None:
        self._inc("reconnect_total", source=source, outcome=outcome)

    def lock(self, leg: str, locked: bool) -> None:
        if not isinstance(locked, bool):
            raise TypeError("locked must be a boolean")
        self._set("lock_state", int(locked), leg=leg)

    def order_ack_latency(self, seconds: Decimal, *, exchange: str = "unknown") -> None:
        self._set("order_ack_latency_seconds", seconds, exchange=exchange)

    def unknown_orders(self, count: int, *, leg: str = "all") -> None:
        self._set("unknown_order_count", count, leg=leg)

    def unknown_order_age(self, seconds: Decimal, *, leg: str) -> None:
        self._set("unknown_order_age_seconds", seconds, leg=leg)

    def cancel_replace(self, outcome: str) -> None:
        self._inc("cancel_replace_total", outcome=outcome)

    def duplicate_fill(self, source: str = "unknown") -> None:
        self._inc("duplicate_fill_total", source=source)

    def outbox(self, backlog: int) -> None:
        self._set("outbox_backlog", backlog)

    def projection(self, version: int, *, projection: str) -> None:
        self._set("projection_version", version, projection=projection)

    def reconciliation_diff(self, severity: str, *, resolved: bool) -> None:
        self._inc(
            "reconciliation_diff_total",
            severity=severity,
            resolved=resolved,
        )

    def rest_error(self, code: str, *, endpoint: str = "unknown") -> None:
        self._inc("rest_error_total", code=code, endpoint=endpoint)

    def clock_skew(self, milliseconds: Decimal, *, source: str = "exchange") -> None:
        self._set("clock_skew_milliseconds", abs(milliseconds), source=source)

    def account_risk(
        self,
        *,
        gross_long: Decimal,
        gross_short: Decimal,
        net: Decimal,
        gross_ratio: Decimal,
        margin_utilization: Decimal,
        pending_risk: Decimal,
        account_id: str,
    ) -> None:
        labels = {"account_id": account_id}
        self._set("gross_long_notional", gross_long, **labels)
        self._set("gross_short_notional", gross_short, **labels)
        self._set("net_notional", net, signed=True, **labels)
        self._set("gross_exposure_ratio", gross_ratio, **labels)
        self._set("margin_utilization", margin_utilization, **labels)
        self._set("pending_risk_notional", pending_risk, **labels)

    def single_writer(self, *, leader: bool, lease_age_seconds: Decimal) -> None:
        if not isinstance(leader, bool):
            raise TypeError("leader must be a boolean")
        self._set("single_writer_leader", int(leader))
        self._set("single_writer_lease_age_seconds", lease_age_seconds)

    def lock_wait(self, seconds: Decimal, *, scope: str) -> None:
        self._set("lock_wait_seconds", seconds, scope=scope)

    def lock_timeout(self, *, scope: str) -> None:
        self._inc("lock_timeout_total", scope=scope)

    def deadlock(self, *, scope: str) -> None:
        self._inc("deadlock_total", scope=scope)

    def snapshot(self) -> dict[str, dict[str, str]]:
        with self._lock:
            result: dict[str, dict[str, str]] = {
                name: {} for name in self.REQUIRED_FAMILIES
            }
            for (name, labels), value in self._counters.items():
                result.setdefault(name, {})[self._labels(labels)] = str(value)
            for (name, labels), value in self._gauges.items():
                result.setdefault(name, {})[self._labels(labels)] = str(value)
            return result

    @staticmethod
    def _labels(labels: tuple[tuple[str, str], ...]) -> str:
        return ",".join(f"{key}={value}" for key, value in labels) or "_"
