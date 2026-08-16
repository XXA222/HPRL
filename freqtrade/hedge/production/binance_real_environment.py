"""R3 Binance real-market acceptance with structurally zero trading writes.

R2 already separated ``BinanceReadonlyClient`` from the simulated execution engine.  R3
adds evidence eligibility: an internal deterministic acceptance probe may prove plumbing,
but only an externally supplied HPRL model-target feed may satisfy the final real-market
evidence gate used by Production Acceptance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
import json
from typing import Mapping, Sequence

from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
from freqtrade.hedge.symbols import raw_symbol

from .binance_runtime_dryrun import (
    BinanceRuntimeDryRunReport,
    collect_binance_real_market_preflight,
    run_binance_real_market_dryrun,
)
from .risk_behavior import HprlBehaviorObservation


_FORBIDDEN_TRADE_WRITE_ROUTES = {
    ("POST", "/fapi/v1/order"),
    ("DELETE", "/fapi/v1/order"),
    ("POST", "/fapi/v1/batchOrders"),
    ("DELETE", "/fapi/v1/batchOrders"),
    ("POST", "/fapi/v1/allOpenOrders"),
    ("DELETE", "/fapi/v1/allOpenOrders"),
    ("POST", "/fapi/v1/leverage"),
    ("POST", "/fapi/v1/marginType"),
    ("POST", "/fapi/v1/positionSide/dual"),
}


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(raw).hexdigest()


def _transport_allowed_routes(client: object) -> frozenset[tuple[str, str]]:
    transport = getattr(client, "transport", None)
    routes = getattr(transport, "_ALLOWED", None)
    if routes is None:
        return frozenset()
    return frozenset((str(method).upper(), str(path)) for method, path in routes)


def _telemetry_requests(client: object) -> int:
    transport = getattr(client, "transport", None)
    telemetry = getattr(transport, "telemetry", None)
    try:
        return int(getattr(telemetry, "logical_request_count", 0))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True, slots=True)
class BinanceR3SafetySurface:
    readonly_client_write_methods_absent: bool
    transport_allowlist_visible: bool
    forbidden_trade_write_routes_absent: bool
    allowed_route_count: int
    forbidden_routes: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.readonly_client_write_methods_absent
            and self.transport_allowlist_visible
            and self.forbidden_trade_write_routes_absent
        )


def inspect_binance_r3_safety_surface(client: object) -> BinanceR3SafetySurface:
    forbidden_methods = (
        "submit_order", "create_order", "cancel_order", "cancel_all_orders",
        "set_leverage", "set_margin_mode", "set_position_mode",
    )
    methods_absent = not any(callable(getattr(client, name, None)) for name in forbidden_methods)
    routes = _transport_allowed_routes(client)
    found = tuple(
        sorted(f"{method} {path}" for method, path in routes if (method, path) in _FORBIDDEN_TRADE_WRITE_ROUTES)
    )
    return BinanceR3SafetySurface(
        readonly_client_write_methods_absent=methods_absent,
        transport_allowlist_visible=bool(routes),
        forbidden_trade_write_routes_absent=not found,
        allowed_route_count=len(routes),
        forbidden_routes=found,
    )


@dataclass(frozen=True, slots=True)
class BinanceR3AccountSurface:
    long_side_fact_visible: bool
    short_side_fact_visible: bool
    real_long_quantity: Decimal
    real_short_quantity: Decimal
    active_real_order_count: int
    bundle_sha256: str

    @property
    def passed(self) -> bool:
        return self.long_side_fact_visible and self.short_side_fact_visible


async def collect_binance_r3_account_surface(client: object, *, symbol: str) -> BinanceR3AccountSurface:
    bundle = await client.fetch_bundle(include_fills=False)
    normalized = raw_symbol(symbol)
    rows = tuple(item for item in bundle.positions if raw_symbol(item.symbol) == normalized)
    sides = {str(item.position_side).upper() for item in rows}
    long_qty = sum((abs(item.quantity) for item in rows if str(item.position_side).upper() == "LONG"), Decimal("0"))
    short_qty = sum((abs(item.quantity) for item in rows if str(item.position_side).upper() == "SHORT"), Decimal("0"))
    active_orders = sum(
        1 for item in bundle.open_orders if raw_symbol(item.symbol) == normalized and item.active
    )
    digest = _sha({
        "symbol": normalized,
        "positions": [
            {
                "side": str(item.position_side).upper(),
                "quantity": str(item.quantity),
                "leverage": int(item.leverage),
            }
            for item in rows
        ],
        "active_orders": active_orders,
        "configuration": repr(bundle.configuration),
        "account": repr(bundle.account_snapshot),
        "completed_at": bundle.collection_completed_at,
    })
    return BinanceR3AccountSurface(
        long_side_fact_visible="LONG" in sides,
        short_side_fact_visible="SHORT" in sides,
        real_long_quantity=long_qty,
        real_short_quantity=short_qty,
        active_real_order_count=active_orders,
        bundle_sha256=digest,
    )


def _target_uncertainty(target: PlannedExecutionIntent) -> float:
    raw = target.metadata.get("uncertainty", "0") if isinstance(target.metadata, Mapping) else "0"
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, value))


@dataclass(frozen=True, slots=True)
class BinanceR3BehaviorRow:
    cycle_id: str
    model_id: str
    target_source: str
    observation: HprlBehaviorObservation
    target_sha256: str
    market_evidence_sha256: str


@dataclass(frozen=True, slots=True)
class BinanceR3RealMarketReport:
    dryrun: BinanceRuntimeDryRunReport
    safety_surface: BinanceR3SafetySurface
    account_surface_before: BinanceR3AccountSurface
    account_surface_after: BinanceR3AccountSurface
    model_target_feed: bool
    production_evidence_eligible: bool
    behavior_rows: tuple[BinanceR3BehaviorRow, ...]
    transport_requests_before: int
    transport_requests_after: int
    real_trade_write_count: int
    evidence_sha256: str
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.dryrun.passed and self.safety_surface.passed and self.account_surface_before.passed and self.account_surface_after.passed and self.real_trade_write_count == 0 and not self.reasons


async def run_binance_r3_real_market_acceptance(
    client: object,
    *,
    symbol: str,
    targets: Sequence[PlannedExecutionIntent],
    journal_path: str | None = None,
    checkpoint_path: str | None = None,
    cycle_interval_seconds: float = 0.0,
    require_model_target_feed: bool = True,
) -> BinanceR3RealMarketReport:
    if not targets:
        raise ValueError("R3 Binance acceptance requires a non-empty target sequence")
    safety = inspect_binance_r3_safety_surface(client)
    requests_before = _telemetry_requests(client)
    preflight = await collect_binance_real_market_preflight(client, symbol=symbol)
    account_before = await collect_binance_r3_account_surface(client, symbol=symbol)
    sources = {
        str(target.metadata.get("source", ""))
        for target in targets
        if isinstance(target.metadata, Mapping)
    }
    model_feed = sources == {"model-target-feed"}
    source_label = "model-target-feed" if model_feed else "acceptance-probe"
    dryrun = await run_binance_real_market_dryrun(
        client,
        symbol=symbol,
        targets=targets,
        journal_path=journal_path,
        checkpoint_path=checkpoint_path,
        cycle_interval_seconds=cycle_interval_seconds,
        source=source_label,
    )
    account_after = await collect_binance_r3_account_surface(client, symbol=symbol)
    requests_after = _telemetry_requests(client)
    reasons: list[str] = []
    if not safety.passed:
        reasons.append("BINANCE_READONLY_SAFETY_SURFACE_FAILED")
    if not preflight.passed:
        reasons.append("BINANCE_REAL_MARKET_PREFLIGHT_FAILED")
    if not account_before.passed or not account_after.passed:
        reasons.append("BINANCE_HEDGE_LONG_SHORT_FACTS_NOT_BOTH_VISIBLE")
    if not dryrun.passed:
        reasons.append("BINANCE_SIMULATED_EXECUTION_CHAIN_FAILED")
    if require_model_target_feed and not model_feed:
        reasons.append("BINANCE_MODEL_TARGET_FEED_REQUIRED")
    if dryrun.real_exchange_write_count != 0:
        reasons.append("BINANCE_REAL_TRADE_WRITE_DETECTED")

    rows: list[BinanceR3BehaviorRow] = []
    peak_equity: Decimal | None = None
    previous_equity: Decimal | None = None
    for target, cycle in zip(targets, dryrun.cycles, strict=True):
        equity = Decimal(cycle.telemetry.equity)
        peak_equity = equity if peak_equity is None else max(peak_equity, equity)
        if previous_equity is None or previous_equity == 0:
            equity_return = 0.0
        else:
            equity_return = float((equity / previous_equity) - Decimal("1"))
        drawdown = 0.0 if peak_equity in (None, Decimal("0")) else float(max(Decimal("0"), Decimal("1") - equity / peak_equity))
        projection = cycle.telemetry.strategy
        observation = HprlBehaviorObservation(
            timestamp=_aware(cycle.telemetry.timestamp),
            long_margin_ratio=Decimal(projection.long_score),
            short_margin_ratio=Decimal(projection.short_score),
            equity_return=equity_return,
            drawdown=drawdown,
            uncertainty=_target_uncertainty(target),
        )
        target_digest = _sha({
            "symbol": target.symbol,
            "long": target.target_long_exposure,
            "short": target.target_short_exposure,
            "confidence": target.confidence,
            "model_id": target.model_id,
            "metadata": dict(target.metadata),
        })
        rows.append(BinanceR3BehaviorRow(
            cycle_id=cycle.telemetry.cycle_id,
            model_id=target.model_id,
            target_source=str(target.metadata.get("source", "")),
            observation=observation,
            target_sha256=target_digest,
            market_evidence_sha256=cycle.market_evidence_sha256,
        ))
        previous_equity = equity
    eligible = dryrun.passed and safety.passed and account_before.passed and account_after.passed and model_feed and dryrun.real_exchange_write_count == 0
    payload = {
        "dryrun": asdict(dryrun),
        "safety": asdict(safety),
        "account_before": asdict(account_before),
        "account_after": asdict(account_after),
        "model_feed": model_feed,
        "eligible": eligible,
        "request_count": [requests_before, requests_after],
        "behavior": [asdict(row) for row in rows],
    }
    return BinanceR3RealMarketReport(
        dryrun=dryrun,
        safety_surface=safety,
        account_surface_before=account_before,
        account_surface_after=account_after,
        model_target_feed=model_feed,
        production_evidence_eligible=eligible,
        behavior_rows=tuple(rows),
        transport_requests_before=requests_before,
        transport_requests_after=requests_after,
        real_trade_write_count=0,
        evidence_sha256=_sha(payload),
        reasons=tuple(dict.fromkeys(reasons)),
    )
