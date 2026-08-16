"""Binance real-market / simulated-execution dry-run acceptance for HPRL V3."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from typing import Iterable

from freqtrade.hedge.telemetry.dryrun import DryRunCycleTelemetry

ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class BinanceDryRunSafetyContext:
    exchange: str
    operation_mode: str
    real_market_data: bool
    exchange_write_capability: bool
    simulated_execution: bool
    hedge_mode_semantics: bool
    cross_margin_semantics: bool
    source_release: str
    account_namespace: str

    def __post_init__(self) -> None:
        if not self.exchange.strip() or not self.operation_mode.strip():
            raise ValueError("exchange/operation_mode are required")
        if not self.source_release.strip() or not self.account_namespace.strip():
            raise ValueError("source_release/account_namespace are required")

    @property
    def safe(self) -> bool:
        return (
            self.exchange.strip().lower() == "binance"
            and self.operation_mode.strip().lower().replace("-", "_") == "dry_run"
            and self.real_market_data
            and not self.exchange_write_capability
            and self.simulated_execution
            and self.hedge_mode_semantics
            and self.cross_margin_semantics
        )


@dataclass(frozen=True, slots=True)
class BinanceDryRunPolicy:
    minimum_cycles: int = 100
    minimum_duration: timedelta = timedelta(minutes=30)
    maximum_cycle_gap: timedelta = timedelta(minutes=5)
    require_dual_leg_target: bool = True
    maximum_risk_block_ratio: Decimal = Decimal("0.25")

    def __post_init__(self) -> None:
        if self.minimum_cycles <= 0:
            raise ValueError("minimum_cycles must be positive")
        if self.minimum_duration < timedelta(0) or self.maximum_cycle_gap <= timedelta(0):
            raise ValueError("dry-run durations are invalid")
        if not ZERO <= self.maximum_risk_block_ratio <= Decimal("1"):
            raise ValueError("maximum_risk_block_ratio must be within [0,1]")


@dataclass(frozen=True, slots=True)
class BinanceDryRunAcceptanceReport:
    passed: bool
    cycle_count: int
    duration_seconds: int
    unique_cycle_ids: bool
    monotonic_timestamps: bool
    maximum_gap_seconds: int
    dual_leg_target_observed: bool
    dual_leg_position_observed: bool
    risk_block_ratio: Decimal
    final_equity: Decimal
    telemetry_sha256: str
    reasons: tuple[str, ...]
    observed_at: datetime | None


def _telemetry_hash(items: tuple[DryRunCycleTelemetry, ...]) -> str:
    payload = []
    for item in items:
        payload.append({
            "cycle_id": item.cycle_id,
            "account_id": item.account_id,
            "symbol": item.symbol,
            "timestamp": item.timestamp.astimezone(UTC).isoformat(),
            "mark_price": str(item.mark_price),
            "equity": str(item.equity),
            "gross_notional": str(item.gross_notional),
            "long_quantity": str(item.long_quantity),
            "short_quantity": str(item.short_quantity),
            "long_target_quantity": str(item.long_target_quantity),
            "short_target_quantity": str(item.short_target_quantity),
            "fees": str(item.fees),
            "funding_pnl": str(item.funding_pnl),
            "risk_blocked": item.risk_blocked,
            "model_version": item.strategy.model_version,
            "regime": item.strategy.regime,
        })
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256(raw).hexdigest()


def evaluate_binance_dryrun(
    telemetry: Iterable[DryRunCycleTelemetry],
    *,
    safety: BinanceDryRunSafetyContext,
    policy: BinanceDryRunPolicy | None = None,
) -> BinanceDryRunAcceptanceReport:
    policy = policy or BinanceDryRunPolicy()
    items = tuple(telemetry)
    reasons: list[str] = []
    if not safety.safe:
        reasons.append("BINANCE_DRYRUN_SAFETY_CONTEXT_INVALID")
    if len(items) < policy.minimum_cycles:
        reasons.append("BINANCE_DRYRUN_TOO_FEW_CYCLES")
    if any(not isinstance(item, DryRunCycleTelemetry) for item in items):
        raise TypeError("telemetry must contain DryRunCycleTelemetry")
    ids = [item.cycle_id for item in items]
    unique = len(ids) == len(set(ids)) and all(bool(item.strip()) for item in ids)
    if not unique:
        reasons.append("BINANCE_DRYRUN_DUPLICATE_CYCLE_ID")
    timestamps = [item.timestamp.astimezone(UTC) for item in items]
    monotonic = all(right >= left for left, right in zip(timestamps, timestamps[1:], strict=False))
    if not monotonic:
        reasons.append("BINANCE_DRYRUN_TIMESTAMP_REGRESSION")
    duration = 0
    max_gap = 0
    if timestamps:
        duration = max(0, int((timestamps[-1] - timestamps[0]).total_seconds()))
        gaps = [int((right - left).total_seconds()) for left, right in zip(timestamps, timestamps[1:], strict=False)]
        max_gap = max(gaps, default=0)
        if duration < int(policy.minimum_duration.total_seconds()):
            reasons.append("BINANCE_DRYRUN_DURATION_TOO_SHORT")
        if max_gap > int(policy.maximum_cycle_gap.total_seconds()):
            reasons.append("BINANCE_DRYRUN_CYCLE_GAP")
    else:
        reasons.append("BINANCE_DRYRUN_NO_TELEMETRY")
    if any(
        item.mark_price <= ZERO
        or item.equity < ZERO
        or item.long_quantity < ZERO
        or item.short_quantity < ZERO
        or item.long_target_quantity < ZERO
        or item.short_target_quantity < ZERO
        for item in items
    ):
        reasons.append("BINANCE_DRYRUN_INVALID_NUMERIC_STATE")
    namespaces = {item.account_id.split(":", 1)[0] for item in items}
    if items and namespaces != {safety.account_namespace}:
        reasons.append("BINANCE_DRYRUN_ACCOUNT_NAMESPACE_MISMATCH")
    models = {item.strategy.model_version for item in items}
    if items and any(not model.strip() for model in models):
        reasons.append("BINANCE_DRYRUN_MODEL_ID_MISSING")
    dual_target = any(
        item.long_target_quantity > ZERO and item.short_target_quantity > ZERO for item in items
    )
    dual_position = any(item.long_quantity > ZERO and item.short_quantity > ZERO for item in items)
    if policy.require_dual_leg_target and not dual_target:
        reasons.append("BINANCE_DRYRUN_DUAL_LEG_TARGET_NOT_EXERCISED")
    blocked = sum(bool(item.risk_blocked) for item in items)
    blocked_ratio = Decimal(blocked) / Decimal(len(items)) if items else Decimal("1")
    if blocked_ratio > policy.maximum_risk_block_ratio:
        reasons.append("BINANCE_DRYRUN_EXCESSIVE_RISK_BLOCK_RATE")
    diagnostics = tuple(
        diagnostic
        for item in items
        for diagnostic in item.diagnostics
        if any(token in diagnostic.upper() for token in ("LIVE_WRITE", "REAL_ORDER", "API_WRITE"))
    )
    if diagnostics:
        reasons.append("BINANCE_DRYRUN_WRITE_DIAGNOSTIC_PRESENT")
    final_equity = items[-1].equity if items else ZERO
    digest = _telemetry_hash(items)
    return BinanceDryRunAcceptanceReport(
        passed=not reasons,
        cycle_count=len(items),
        duration_seconds=duration,
        unique_cycle_ids=unique,
        monotonic_timestamps=monotonic,
        maximum_gap_seconds=max_gap,
        dual_leg_target_observed=dual_target,
        dual_leg_position_observed=dual_position,
        risk_block_ratio=blocked_ratio,
        final_equity=final_equity,
        telemetry_sha256=digest,
        reasons=tuple(dict.fromkeys(reasons)),
        observed_at=timestamps[-1] if timestamps else None,
    )
