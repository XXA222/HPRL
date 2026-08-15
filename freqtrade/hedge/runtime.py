"""Source-separated, fail-closed runtime projections for Hedge control planes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from threading import RLock
from typing import Mapping

from freqtrade.hedge.config import HedgeRuntimeConfig
from freqtrade.hedge.position_book import PositionRecord
from freqtrade.hedge.risk.models import AccountRiskSnapshot


class HedgeProjectionSource(StrEnum):
    """Authoritative namespace for one runtime projection."""

    EXCHANGE = "EXCHANGE"
    PAPER = "PAPER"
    LIVE = "LIVE"
    SHADOW = "SHADOW"


_EXCHANGE_CHECKS = (
    "common.persistence_healthy",
    "exchange.readonly_service_bound",
    "exchange.rest_calibrated",
    "exchange.user_stream_fresh",
    "exchange.reconciliation_converged",
    "exchange.risk_snapshot_valid",
)
_PAPER_CHECKS = (
    "common.persistence_healthy",
    "paper.market_data_fresh",
    "paper.funding_source_healthy",
    "paper.account_events_durable",
    "paper.simulation_engine_healthy",
    "paper.ledger_durable",
    "paper.risk_snapshot_valid",
)
_LIVE_CHECKS = (
    "common.persistence_healthy",
    "exchange.readonly_service_bound",
    "exchange.rest_calibrated",
    "exchange.user_stream_fresh",
    "exchange.reconciliation_converged",
    "live.writer_lease_valid",
    "live.risk_approval_ready",
    "live.execution_adapter_ready",
)
_SHADOW_CHECKS = (
    "common.persistence_healthy",
    "shadow.comparison_healthy",
)

_REQUIRED_CHECKS: dict[HedgeProjectionSource, tuple[str, ...]] = {
    HedgeProjectionSource.EXCHANGE: _EXCHANGE_CHECKS,
    HedgeProjectionSource.PAPER: _PAPER_CHECKS,
    HedgeProjectionSource.LIVE: _LIVE_CHECKS,
    HedgeProjectionSource.SHADOW: _SHADOW_CHECKS,
}

# Backward compatibility for callers from the pre-v1.5 runtime.  The values are
# normalized at the boundary and never exposed again under the ambiguous names.
_LEGACY_EXCHANGE_CHECK_NAMES = {
    "readonly_service_bound": "exchange.readonly_service_bound",
    "rest_calibrated": "exchange.rest_calibrated",
    "user_stream_fresh": "exchange.user_stream_fresh",
    "reconciliation_converged": "exchange.reconciliation_converged",
    "risk_snapshot_valid": "exchange.risk_snapshot_valid",
}


@dataclass(frozen=True, slots=True)
class HedgeSourceView:
    source: HedgeProjectionSource
    source_version: str
    sequence: int
    positions: tuple[PositionRecord, ...]
    risk: AccountRiskSnapshot | None
    reconciliation_status: str
    reconciliation_at: datetime | None
    reconciliation_details: tuple[str, ...]
    stream_state: str
    stream_last_event_at: datetime | None
    stream_reconnect_count: int
    checks: tuple[tuple[str, bool], ...]
    reasons: tuple[str, ...]
    source_event_time: datetime | None
    published_at: datetime
    stale: bool
    ready: bool


@dataclass(frozen=True, slots=True)
class HedgeRuntimeView:
    """Backward-compatible effective view plus source metadata."""

    positions: tuple[PositionRecord, ...]
    risk: AccountRiskSnapshot | None
    reconciliation_status: str
    reconciliation_at: datetime | None
    reconciliation_details: tuple[str, ...]
    stream_state: str
    stream_last_event_at: datetime | None
    stream_reconnect_count: int
    ready: bool
    halted: bool
    checks: tuple[tuple[str, bool], ...]
    reasons: tuple[str, ...]
    observed_at: datetime
    source: HedgeProjectionSource
    source_version: str
    sequence: int
    source_event_time: datetime | None
    stale: bool
    available_sources: tuple[HedgeProjectionSource, ...]


class HedgeRuntime:
    """Thread-safe runtime that never lets simulated state overwrite facts.

    Every publisher writes to a source namespace.  The operation mode chooses an
    effective source for safety and legacy APIs:

    - readonly and shadow: EXCHANGE facts are authoritative;
    - paper: PAPER simulation is authoritative for the simulated account;
    - live: LIVE execution is authoritative, while EXCHANGE remains a fact input.

    A halt is scoped to SYSTEM or a source.  Publishing PAPER can therefore never
    clear an EXCHANGE or SYSTEM halt.
    """

    def __init__(self, config: HedgeRuntimeConfig) -> None:
        if not config.enabled:
            raise ValueError("HedgeRuntime requires hedge_mode_enabled=true")
        self.config = config
        self._lock = RLock()
        self._sources: dict[HedgeProjectionSource, HedgeSourceView] = {}
        self._next_sequence: dict[HedgeProjectionSource, int] = {
            source: 0 for source in HedgeProjectionSource
        }
        self._halts: dict[str, str] = {
            "SYSTEM": "HEDGE_RUNTIME_NOT_READY",
        }
        self._observed_at = datetime.now(UTC)

    @staticmethod
    def _source(value: HedgeProjectionSource | str) -> HedgeProjectionSource:
        if isinstance(value, HedgeProjectionSource):
            return value
        try:
            return HedgeProjectionSource(str(value).strip().upper())
        except ValueError as exc:
            raise ValueError(f"Unsupported hedge projection source: {value!r}") from exc

    def _effective_source(self) -> HedgeProjectionSource:
        mode = self.config.operation_mode
        if mode in {"readonly", "shadow"}:
            return HedgeProjectionSource.EXCHANGE
        if mode == "paper":
            return HedgeProjectionSource.PAPER
        if mode == "live":
            return HedgeProjectionSource.LIVE
        # Configuration validation should reject this first.  Keep a safe default.
        return HedgeProjectionSource.EXCHANGE

    @staticmethod
    def _normalize_status(reconciliation_status: str, stream_state: str) -> tuple[str, str]:
        reconciliation = str(reconciliation_status).strip().upper()
        if reconciliation not in {"HEALTHY", "DRIFT", "RUNNING", "UNKNOWN", "NOT_APPLICABLE"}:
            raise ValueError("Unsupported reconciliation status")
        stream = str(stream_state).strip().upper()
        if stream not in {
            "CONNECTED",
            "STALE",
            "DISCONNECTED",
            "RECONNECTING",
            "UNKNOWN",
            "NOT_APPLICABLE",
        }:
            raise ValueError("Unsupported user stream state")
        return reconciliation, stream

    @staticmethod
    def _normalize_checks(
        source: HedgeProjectionSource,
        checks: Mapping[str, bool],
    ) -> dict[str, bool]:
        normalized: dict[str, bool] = {}
        for raw_name, value in checks.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ValueError("readiness check names must be non-empty strings")
            if not isinstance(value, bool):
                raise ValueError("readiness check values must be booleans")
            name = raw_name.strip()
            if source is HedgeProjectionSource.EXCHANGE:
                name = _LEGACY_EXCHANGE_CHECK_NAMES.get(name, name)
            normalized[name] = value

        # Older EXCHANGE publishers did not have an explicit persistence check.
        if source is HedgeProjectionSource.EXCHANGE and "common.persistence_healthy" not in normalized:
            normalized["common.persistence_healthy"] = True

        required = set(_REQUIRED_CHECKS[source])
        if set(normalized) != required:
            missing = sorted(required - set(normalized))
            extra = sorted(set(normalized) - required)
            raise ValueError(
                f"checks for {source.value} must match the complete namespaced set; "
                f"missing={missing}, extra={extra}"
            )
        return normalized

    def publish(
        self,
        *,
        source: HedgeProjectionSource | str = HedgeProjectionSource.EXCHANGE,
        positions: tuple[PositionRecord, ...],
        risk: AccountRiskSnapshot | None,
        reconciliation_status: str,
        reconciliation_at: datetime | None,
        reconciliation_details: tuple[str, ...] = (),
        stream_state: str,
        stream_last_event_at: datetime | None,
        stream_reconnect_count: int,
        checks: Mapping[str, bool],
        reasons: tuple[str, ...] = (),
        source_version: str = "1",
        source_event_time: datetime | None = None,
        stale: bool = False,
    ) -> HedgeSourceView:
        """Atomically publish one immutable source projection."""

        resolved_source = self._source(source)
        reconciliation, stream = self._normalize_status(
            reconciliation_status,
            stream_state,
        )
        if stream_reconnect_count < 0:
            raise ValueError("stream_reconnect_count must be nonnegative")
        normalized_checks = self._normalize_checks(resolved_source, checks)
        normalized_version = str(source_version).strip()
        if not normalized_version:
            raise ValueError("source_version must not be empty")
        if not isinstance(stale, bool):
            raise ValueError("stale must be a boolean")
        if risk is not None and risk.account_id != self.config.account_id:
            raise ValueError("Risk snapshot account does not match runtime")
        for position in positions:
            if position.account_id != self.config.account_id:
                raise ValueError("Position account does not match runtime")

        now = datetime.now(UTC)
        ready = all(normalized_checks.values()) and not reasons and not stale
        with self._lock:
            sequence = self._next_sequence[resolved_source] + 1
            self._next_sequence[resolved_source] = sequence
            projection = HedgeSourceView(
                source=resolved_source,
                source_version=normalized_version,
                sequence=sequence,
                positions=tuple(positions),
                risk=risk,
                reconciliation_status=reconciliation,
                reconciliation_at=reconciliation_at,
                reconciliation_details=tuple(reconciliation_details),
                stream_state=stream,
                stream_last_event_at=stream_last_event_at,
                stream_reconnect_count=stream_reconnect_count,
                checks=tuple(normalized_checks.items()),
                reasons=tuple(str(item) for item in reasons),
                source_event_time=source_event_time or reconciliation_at or stream_last_event_at,
                published_at=now,
                stale=stale,
                ready=ready,
            )
            self._sources[resolved_source] = projection
            self._halts.pop(resolved_source.value, None)
            # Initial NOT_READY is cleared by the first valid source publication, but
            # other explicit SYSTEM halts remain persistent until cleared by owner code.
            if self._halts.get("SYSTEM") == "HEDGE_RUNTIME_NOT_READY":
                self._halts.pop("SYSTEM", None)
            self._observed_at = now
            return projection

    def halt(
        self,
        reason: str,
        *,
        source: HedgeProjectionSource | str | None = None,
    ) -> None:
        normalized = str(reason).strip()
        if not normalized:
            raise ValueError("Halt reason must not be empty")
        scope = "SYSTEM" if source is None else self._source(source).value
        with self._lock:
            self._halts[scope] = normalized
            self._observed_at = datetime.now(UTC)

    def clear_halt(self, *, source: HedgeProjectionSource | str | None = None) -> None:
        scope = "SYSTEM" if source is None else self._source(source).value
        with self._lock:
            self._halts.pop(scope, None)
            self._observed_at = datetime.now(UTC)

    def heartbeat(self) -> None:
        """Assert the central write-lock invariant on every bot loop."""

        if not self.config.read_only or self.config.live_trading_enabled:
            self.halt("HEDGE_WRITE_CONFIGURATION_REJECTED")
            raise RuntimeError("Hedge runtime write mode is disabled")

    def source_view(self, source: HedgeProjectionSource | str) -> HedgeSourceView | None:
        resolved = self._source(source)
        with self._lock:
            return self._sources.get(resolved)

    def _empty_source_view(self, source: HedgeProjectionSource) -> HedgeSourceView:
        now = self._observed_at
        checks = tuple((name, False) for name in _REQUIRED_CHECKS[source])
        return HedgeSourceView(
            source=source,
            source_version="0",
            sequence=0,
            positions=(),
            risk=None,
            reconciliation_status=(
                "NOT_APPLICABLE" if source is HedgeProjectionSource.PAPER else "UNKNOWN"
            ),
            reconciliation_at=None,
            reconciliation_details=(),
            stream_state=(
                "NOT_APPLICABLE" if source is HedgeProjectionSource.PAPER else "UNKNOWN"
            ),
            stream_last_event_at=None,
            stream_reconnect_count=0,
            checks=checks,
            reasons=(f"{source.value}_PROJECTION_NOT_PUBLISHED",),
            source_event_time=None,
            published_at=now,
            stale=True,
            ready=False,
        )

    def view(
        self,
        source: HedgeProjectionSource | str | None = None,
    ) -> HedgeRuntimeView:
        resolved = self._effective_source() if source is None else self._source(source)
        with self._lock:
            projection = self._sources.get(resolved) or self._empty_source_view(resolved)
            halt_reasons: list[str] = []
            system_halt = self._halts.get("SYSTEM")
            if system_halt:
                halt_reasons.append(system_halt)
            source_halt = self._halts.get(resolved.value)
            if source_halt:
                halt_reasons.append(source_halt)
            reasons = tuple(dict.fromkeys((*halt_reasons, *projection.reasons)))
            ready = projection.ready and not halt_reasons
            return HedgeRuntimeView(
                positions=projection.positions,
                risk=projection.risk,
                reconciliation_status=projection.reconciliation_status,
                reconciliation_at=projection.reconciliation_at,
                reconciliation_details=projection.reconciliation_details,
                stream_state=projection.stream_state,
                stream_last_event_at=projection.stream_last_event_at,
                stream_reconnect_count=projection.stream_reconnect_count,
                ready=ready,
                halted=not ready,
                checks=projection.checks,
                reasons=reasons,
                observed_at=projection.published_at,
                source=projection.source,
                source_version=projection.source_version,
                sequence=projection.sequence,
                source_event_time=projection.source_event_time,
                stale=projection.stale,
                available_sources=tuple(sorted(self._sources, key=lambda item: item.value)),
            )

    @staticmethod
    def empty_risk(account_id: str) -> AccountRiskSnapshot:
        """Return a valid sentinel for callers that require a domain snapshot."""

        return AccountRiskSnapshot(
            account_id=account_id,
            equity=Decimal("1"),
            wallet_balance=Decimal("0"),
            available_balance=Decimal("0"),
            initial_margin=Decimal("0"),
            maintenance_margin=Decimal("0"),
            gross_long_notional=Decimal("0"),
            gross_short_notional=Decimal("0"),
            net_notional=Decimal("0"),
            risk_data_valid=False,
            liquidation_buffer_ratio=Decimal("0"),
        )
