"""The complete ReadinessGate condition set and raw freshness validation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from freqtrade.hedge.numeric import require_positive
from freqtrade.hedge.readiness.state import ReadinessCheckResult, ReadinessReasonCode
from freqtrade.hedge.identity import UnknownOrderRisk


def _mode_value(value: object, *, field_name: str) -> str:
    raw = getattr(value, "value", value)
    normalized = str(raw).strip().lower()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _require_bool(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _optional_nonnegative_int(value: int | None, *, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a nonnegative integer or None.")
    return value


@dataclass(frozen=True, slots=True)
class ReadinessInputs:
    database_migration_succeeded: bool
    single_writer_lease_valid: bool
    position_mode: str
    margin_mode: str
    configured_leverage: Decimal
    observed_leverages: tuple[Decimal, ...]
    unmanaged_position_count: int
    unmanaged_order_count: int
    rest_snapshot_valid: bool
    user_stream_fresh: bool
    unknown_order_count: int
    reconciliation_converged: bool
    risk_data_valid: bool
    halt_reasons: tuple[str, ...] = ()
    database_migration_checksum_valid: bool = True
    rest_snapshot_observed_at_ms: int | None = None
    rest_snapshot_max_age_ms: int | None = None
    user_stream_observed_at_ms: int | None = None
    user_stream_max_age_ms: int | None = None
    user_stream_gap_detected: bool = False
    risk_snapshot_observed_at_ms: int | None = None
    risk_snapshot_max_age_ms: int | None = None
    unknown_orders: tuple[UnknownOrderRisk, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "database_migration_succeeded",
            "database_migration_checksum_valid",
            "single_writer_lease_valid",
            "rest_snapshot_valid",
            "user_stream_fresh",
            "user_stream_gap_detected",
            "reconciliation_converged",
            "risk_data_valid",
        ):
            object.__setattr__(
                self,
                field_name,
                _require_bool(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "position_mode",
            _mode_value(self.position_mode, field_name="position_mode"),
        )
        object.__setattr__(
            self,
            "margin_mode",
            _mode_value(self.margin_mode, field_name="margin_mode"),
        )
        configured_leverage = require_positive(
            self.configured_leverage,
            field="configured_leverage",
        )
        if configured_leverage < 1:
            raise ValueError("configured_leverage must be greater than or equal to 1.")
        object.__setattr__(self, "configured_leverage", configured_leverage)
        if isinstance(self.observed_leverages, (str, bytes)):
            raise ValueError("observed_leverages must be a tuple of leverage values.")
        try:
            observed_leverages = tuple(self.observed_leverages)
        except TypeError as exc:
            raise ValueError("observed_leverages must be an iterable of leverage values.") from exc
        normalized_leverages = tuple(
            require_positive(value, field="observed_leverage")
            for value in observed_leverages
        )
        if any(leverage < 1 for leverage in normalized_leverages):
            raise ValueError("observed_leverage must be greater than or equal to 1.")
        object.__setattr__(self, "observed_leverages", normalized_leverages)
        for field_name in (
            "unmanaged_position_count",
            "unmanaged_order_count",
            "unknown_order_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer.")
        for field_name in (
            "rest_snapshot_observed_at_ms",
            "rest_snapshot_max_age_ms",
            "user_stream_observed_at_ms",
            "user_stream_max_age_ms",
            "risk_snapshot_observed_at_ms",
            "risk_snapshot_max_age_ms",
        ):
            object.__setattr__(
                self,
                field_name,
                _optional_nonnegative_int(getattr(self, field_name), field_name=field_name),
            )
        for age_field in (
            "rest_snapshot_max_age_ms",
            "user_stream_max_age_ms",
            "risk_snapshot_max_age_ms",
        ):
            value = getattr(self, age_field)
            if value is not None and value <= 0:
                raise ValueError(f"{age_field} must be positive when provided.")
        if not isinstance(self.halt_reasons, tuple):
            raise ValueError("halt_reasons must be a tuple of non-empty strings.")
        normalized_halts = tuple(
            reason.strip()
            for reason in self.halt_reasons
            if isinstance(reason, str) and reason.strip()
        )
        if len(normalized_halts) != len(self.halt_reasons):
            raise ValueError("halt_reasons must contain non-empty strings only.")
        object.__setattr__(self, "halt_reasons", tuple(dict.fromkeys(normalized_halts)))
        if not isinstance(self.unknown_orders, tuple):
            raise ValueError("unknown_orders must be a tuple.")
        if any(not isinstance(item, UnknownOrderRisk) for item in self.unknown_orders):
            raise ValueError("unknown_orders must contain UnknownOrderRisk values only.")
        if len(self.unknown_orders) > self.unknown_order_count:
            raise ValueError("unknown_order_count must include every supplied unknown order.")
        object.__setattr__(self, "unknown_orders", tuple(self.unknown_orders))

    def as_dict(self) -> dict[str, object]:
        return {
            "database_migration_succeeded": self.database_migration_succeeded,
            "database_migration_checksum_valid": self.database_migration_checksum_valid,
            "single_writer_lease_valid": self.single_writer_lease_valid,
            "position_mode": self.position_mode,
            "margin_mode": self.margin_mode,
            "configured_leverage": str(self.configured_leverage),
            "observed_leverages": [str(value) for value in self.observed_leverages],
            "unmanaged_position_count": self.unmanaged_position_count,
            "unmanaged_order_count": self.unmanaged_order_count,
            "rest_snapshot_valid": self.rest_snapshot_valid,
            "rest_snapshot_observed_at_ms": self.rest_snapshot_observed_at_ms,
            "rest_snapshot_max_age_ms": self.rest_snapshot_max_age_ms,
            "user_stream_fresh": self.user_stream_fresh,
            "user_stream_observed_at_ms": self.user_stream_observed_at_ms,
            "user_stream_max_age_ms": self.user_stream_max_age_ms,
            "user_stream_gap_detected": self.user_stream_gap_detected,
            "unknown_order_count": self.unknown_order_count,
            "unknown_orders": [item.as_dict() for item in self.unknown_orders],
            "reconciliation_converged": self.reconciliation_converged,
            "risk_data_valid": self.risk_data_valid,
            "risk_snapshot_observed_at_ms": self.risk_snapshot_observed_at_ms,
            "risk_snapshot_max_age_ms": self.risk_snapshot_max_age_ms,
            "halt_reasons": list(self.halt_reasons),
        }


def _result(
    name: str,
    passed: bool,
    code: ReadinessReasonCode,
    detail: str = "",
) -> ReadinessCheckResult:
    return ReadinessCheckResult(name, passed, None if passed else code, detail)


def _fresh(
    *,
    valid_flag: bool,
    observed_at_ms: int | None,
    max_age_ms: int | None,
    now_ms: int,
) -> bool:
    if not valid_flag:
        return False
    if max_age_ms is None:
        return True
    if observed_at_ms is None:
        return False
    return observed_at_ms <= now_ms and now_ms - observed_at_ms <= max_age_ms


def run_readiness_checks(
    inputs: ReadinessInputs,
    *,
    now_ms: int | None = None,
) -> tuple[ReadinessCheckResult, ...]:
    evaluation_time = 0 if now_ms is None else now_ms
    leverage_matches = bool(inputs.observed_leverages) and all(
        leverage == inputs.configured_leverage for leverage in inputs.observed_leverages
    )
    rest_fresh = _fresh(
        valid_flag=inputs.rest_snapshot_valid,
        observed_at_ms=inputs.rest_snapshot_observed_at_ms,
        max_age_ms=inputs.rest_snapshot_max_age_ms,
        now_ms=evaluation_time,
    )
    stream_fresh = _fresh(
        valid_flag=inputs.user_stream_fresh,
        observed_at_ms=inputs.user_stream_observed_at_ms,
        max_age_ms=inputs.user_stream_max_age_ms,
        now_ms=evaluation_time,
    )
    risk_fresh = _fresh(
        valid_flag=inputs.risk_data_valid,
        observed_at_ms=inputs.risk_snapshot_observed_at_ms,
        max_age_ms=inputs.risk_snapshot_max_age_ms,
        now_ms=evaluation_time,
    )
    return (
        _result(
            "database_migration",
            inputs.database_migration_succeeded,
            ReadinessReasonCode.DATABASE_MIGRATION_FAILED,
        ),
        _result(
            "database_migration_checksum",
            inputs.database_migration_checksum_valid,
            ReadinessReasonCode.DATABASE_MIGRATION_CHECKSUM_INVALID,
        ),
        _result(
            "single_writer",
            inputs.single_writer_lease_valid,
            ReadinessReasonCode.SINGLE_WRITER_LEASE_INVALID,
        ),
        _result(
            "position_mode",
            inputs.position_mode == "hedge",
            ReadinessReasonCode.POSITION_MODE_NOT_HEDGE,
            inputs.position_mode,
        ),
        _result(
            "margin_mode",
            inputs.margin_mode == "cross",
            ReadinessReasonCode.MARGIN_MODE_NOT_CROSS,
            inputs.margin_mode,
        ),
        _result("leverage", leverage_matches, ReadinessReasonCode.LEVERAGE_MISMATCH),
        _result(
            "unmanaged_positions",
            inputs.unmanaged_position_count == 0,
            ReadinessReasonCode.UNMANAGED_POSITION_PRESENT,
            str(inputs.unmanaged_position_count),
        ),
        _result(
            "unmanaged_orders",
            inputs.unmanaged_order_count == 0,
            ReadinessReasonCode.UNMANAGED_ORDER_PRESENT,
            str(inputs.unmanaged_order_count),
        ),
        _result(
            "rest_snapshot",
            inputs.rest_snapshot_valid,
            ReadinessReasonCode.REST_SNAPSHOT_INVALID,
        ),
        _result(
            "rest_snapshot_freshness",
            rest_fresh,
            ReadinessReasonCode.REST_SNAPSHOT_STALE,
        ),
        _result("user_stream", stream_fresh, ReadinessReasonCode.USER_STREAM_STALE),
        _result(
            "user_stream_gap",
            not inputs.user_stream_gap_detected,
            ReadinessReasonCode.USER_STREAM_GAP,
        ),
        _result(
            "unknown_orders",
            inputs.unknown_order_count == 0,
            ReadinessReasonCode.UNKNOWN_ORDER_PRESENT,
            str(inputs.unknown_order_count),
        ),
        _result(
            "unknown_order_scope",
            inputs.unknown_order_count == len(inputs.unknown_orders),
            ReadinessReasonCode.UNKNOWN_ORDER_SCOPE_INCOMPLETE,
            f"count={inputs.unknown_order_count},scoped={len(inputs.unknown_orders)}",
        ),
        _result(
            "reconciliation",
            inputs.reconciliation_converged,
            ReadinessReasonCode.RECONCILIATION_NOT_CONVERGED,
        ),
        _result("risk_data", inputs.risk_data_valid, ReadinessReasonCode.RISK_DATA_INVALID),
        _result(
            "risk_snapshot_freshness",
            risk_fresh,
            ReadinessReasonCode.RISK_SNAPSHOT_STALE,
        ),
        _result(
            "halt_reasons",
            not inputs.halt_reasons,
            ReadinessReasonCode.HALT_REASON_PRESENT,
            ",".join(inputs.halt_reasons),
        ),
    )
