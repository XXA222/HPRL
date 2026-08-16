"""Real Binance market/account input with strictly simulated HPRL execution.

The runtime deliberately composes two separate worlds:

* ``BinanceReadonlyClient`` owns authenticated/public reads and exposes no order,
  leverage, margin-mode or position-mode write operation.
* ``PositionAwareFakeExchange`` owns every simulated order/fill.  Its account id is
  namespaced away from the real Binance account id and no object implementing a
  Binance execution-write port is accepted by this module.

This makes the safety property structural instead of relying on a command-line flag.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from typing import Awaitable, Callable, Iterable, Mapping, Sequence

from freqtrade.hedge.execution.integrated_fake import build_integrated_fake_runtime
from freqtrade.hedge.execution.ownership import ExecutionOrderOwnershipRegistry
from freqtrade.hedge.execution.service import PositionSide as ExecutionPositionSide
from freqtrade.hedge.hprl.contracts import PlannedExecutionIntent
from freqtrade.hedge.integration.production_main_loop import (
    ExecutionEngineKind,
    HedgeExecutionMode,
    ProductionEquivalentHedgeMainLoop,
)
from freqtrade.hedge.planning.context import (
    LegPosition,
    MarketSnapshot,
    PlannerConfig,
    PlanningContext,
    PositionSide,
    WalletSnapshot,
)
from freqtrade.hedge.production.binance_dryrun import (
    BinanceDryRunAcceptanceReport,
    BinanceDryRunPolicy,
    BinanceDryRunSafetyContext,
    evaluate_binance_dryrun,
)
from freqtrade.hedge.production.closed_loop import (
    ClosedLoopCycleJournalStore,
    HprlProductionClosedLoop,
)
from freqtrade.hedge.production.hprl_hedge_adapter import (
    HprlHedgeAdapter,
    HprlHedgeAdapterPolicy,
    HprlTargetUnit,
)
from freqtrade.hedge.production.recovery_checkpoint import RecoveryCheckpointStore
from freqtrade.hedge.symbols import raw_symbol
from freqtrade.hedge.telemetry.dryrun import DryRunCycleTelemetry, StrategyTelemetry

ZERO = Decimal("0")


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(raw).hexdigest()


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class BinanceRealMarketPreflight:
    account_id: str
    symbol: str
    bid: Decimal
    ask: Decimal
    mark: Decimal
    equity: Decimal
    available_balance: Decimal
    hedge_mode: bool
    cross_margin: bool
    leverage: Decimal
    strict_readonly_verified: bool
    runtime_readonly_enforced: bool
    real_long_quantity: Decimal
    real_short_quantity: Decimal
    active_real_order_count: int
    observed_at: datetime
    evidence_sha256: str
    warnings: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return all((
            self.bid > ZERO,
            self.ask >= self.bid,
            self.mark > ZERO,
            self.equity > ZERO,
            self.hedge_mode,
            self.cross_margin,
            self.leverage > ZERO,
            self.strict_readonly_verified,
            self.runtime_readonly_enforced,
        ))


@dataclass(frozen=True, slots=True)
class BinanceRuntimeDryRunCycle:
    sequence: int
    telemetry: DryRunCycleTelemetry
    closed_loop_record_sha256: str
    market_evidence_sha256: str
    simulated_fill_count: int
    real_exchange_write_count: int = 0


@dataclass(frozen=True, slots=True)
class BinanceRuntimeDryRunReport:
    preflight: BinanceRealMarketPreflight
    cycles: tuple[BinanceRuntimeDryRunCycle, ...]
    acceptance: BinanceDryRunAcceptanceReport
    journal_tip_sha256: str
    journal_valid: bool
    real_exchange_write_count: int
    simulated_submit_count: int
    simulated_fill_count: int
    source: str
    evidence_sha256: str

    @property
    def passed(self) -> bool:
        return (
            self.preflight.passed
            and self.acceptance.passed
            and self.journal_valid
            and self.real_exchange_write_count == 0
            and len(self.cycles) > 0
        )


async def collect_binance_real_market_preflight(
    client: object,
    *,
    symbol: str,
    permission_policy: object | None = None,
) -> BinanceRealMarketPreflight:
    """Collect authenticated account/config plus public real-market prices.

    ``client`` must be a ``BinanceReadonlyClient``-compatible object.  The method
    checks the class surface as well as the returned permission report; an execution
    adapter is intentionally not accepted.
    """
    required = (
        "synchronize_clock",
        "preflight_permissions",
        "fetch_bundle",
        "fetch_real_market_prices",
    )
    missing = [name for name in required if not callable(getattr(client, name, None))]
    forbidden = [
        name
        for name in ("submit_order", "create_order", "cancel_order", "set_leverage", "set_margin_mode")
        if callable(getattr(client, name, None))
    ]
    if missing:
        raise TypeError("readonly Binance client missing methods: " + ",".join(missing))
    if forbidden:
        raise TypeError("Binance dry-run refuses a client with exchange-write methods: " + ",".join(forbidden))

    await client.synchronize_clock()
    permissions = await client.preflight_permissions(permission_policy)
    bundle, prices = await asyncio.gather(
        client.fetch_bundle(include_fills=False),
        client.fetch_real_market_prices(symbol),
    )
    bid, ask, mark = (Decimal(value) for value in prices)
    normalized = raw_symbol(symbol)
    positions = tuple(item for item in bundle.positions if raw_symbol(item.symbol) == normalized)
    long_qty = sum(
        (abs(item.quantity) for item in positions if str(item.position_side).upper() == "LONG"), ZERO
    )
    short_qty = sum(
        (abs(item.quantity) for item in positions if str(item.position_side).upper() == "SHORT"), ZERO
    )
    leverages = {Decimal(max(int(item.leverage), 1)) for item in positions}
    if not leverages:
        # symbolConfig is the account configuration authority for flat symbols.
        leverages = {
            Decimal(value)
            for key, value in bundle.configuration.leverage_by_symbol_side.items()
            if normalized in str(key).upper()
        }
    if len(leverages) != 1:
        raise RuntimeError(f"HPRL dry-run requires one confirmed leverage for {normalized}: {sorted(leverages)}")
    leverage = next(iter(leverages))
    active_orders = sum(1 for item in bundle.open_orders if raw_symbol(item.symbol) == normalized and item.active)
    snapshot = bundle.account_snapshot
    evidence = _sha({
        "account_id": snapshot.account_id,
        "symbol": normalized,
        "bid": str(bid), "ask": str(ask), "mark": str(mark),
        "equity": str(snapshot.total_margin_balance),
        "available": str(snapshot.total_available_balance),
        "long": str(long_qty), "short": str(short_qty),
        "configuration": repr(bundle.configuration),
        "active_orders": active_orders,
        "collection_started_at": bundle.collection_started_at,
        "collection_completed_at": bundle.collection_completed_at,
    })
    return BinanceRealMarketPreflight(
        account_id=snapshot.account_id,
        symbol=normalized,
        bid=bid,
        ask=ask,
        mark=mark,
        equity=snapshot.total_margin_balance,
        available_balance=snapshot.total_available_balance,
        hedge_mode=bool(bundle.configuration.hedge_mode),
        cross_margin=bundle.configuration.active_margin_modes == ("cross",),
        leverage=leverage,
        strict_readonly_verified=bool(permissions.strict_readonly_verified),
        runtime_readonly_enforced=bool(permissions.runtime_readonly_enforced),
        real_long_quantity=long_qty,
        real_short_quantity=short_qty,
        active_real_order_count=active_orders,
        observed_at=_aware(bundle.collection_completed_at),
        evidence_sha256=evidence,
        warnings=tuple(getattr(permissions, "warnings", ())),
    )


def acceptance_probe_targets(symbol: str, cycles: int) -> tuple[PlannedExecutionIntent, ...]:
    """Safe deterministic target schedule used only to exercise the runtime plumbing.

    This is *not* an HPRL model-quality result.  Reports label the source
    ``acceptance-probe`` so production acceptance cannot confuse it with learned-policy
    behavior evidence.
    """
    if cycles < 1:
        raise ValueError("cycles must be positive")
    levels = ((0.05, 0.05), (0.12, 0.05), (0.12, 0.12), (0.05, 0.12), (0.05, 0.05))
    return tuple(
        PlannedExecutionIntent(
            symbol=symbol,
            target_long_exposure=levels[index % len(levels)][0],
            target_short_exposure=levels[index % len(levels)][1],
            confidence=1.0,
            model_id="hprl-runtime-acceptance-probe",
            metadata={"unit": "margin/equity", "source": "acceptance-probe"},
        )
        for index in range(cycles)
    )


def _context(
    *,
    market: MarketSnapshot,
    account: object,
    account_id: str,
    equity: Decimal,
    leverage: Decimal,
) -> PlanningContext:
    long = account.leg(account_id=account_id, symbol=raw_symbol(market.symbol), position_side=ExecutionPositionSide.LONG)
    short = account.leg(account_id=account_id, symbol=raw_symbol(market.symbol), position_side=ExecutionPositionSide.SHORT)
    long_leg = LegPosition(
        PositionSide.LONG,
        quantity=long.quantity,
        average_price=long.average_price,
        core_quantity=long.quantity,
        core_average_price=long.average_price,
    ) if long.quantity > ZERO else LegPosition(PositionSide.LONG)
    short_leg = LegPosition(
        PositionSide.SHORT,
        quantity=short.quantity,
        average_price=short.average_price,
        core_quantity=short.quantity,
        core_average_price=short.average_price,
    ) if short.quantity > ZERO else LegPosition(PositionSide.SHORT)
    gross = (long.quantity + short.quantity) * market.mark
    used_margin = gross / leverage
    available = max(equity - used_margin, ZERO)
    return PlanningContext(
        market=market,
        wallet=WalletSnapshot(
            balance=equity,
            equity=equity,
            available_balance=available,
            long=long_leg,
            short=short_leg,
            leverage=leverage,
        ),
        config=PlannerConfig(),
    )


def _telemetry(
    *,
    sequence: int,
    outcome: object,
    account: object,
    account_id: str,
    market: MarketSnapshot,
    equity: Decimal,
    submitted: int,
    fills: int,
) -> DryRunCycleTelemetry:
    long = account.leg(account_id=account_id, symbol=raw_symbol(market.symbol), position_side=ExecutionPositionSide.LONG)
    short = account.leg(account_id=account_id, symbol=raw_symbol(market.symbol), position_side=ExecutionPositionSide.SHORT)
    planning = outcome.main_loop_cycle.planning if outcome.main_loop_cycle is not None else None
    long_target = ZERO if planning is None else planning.long_target_quantity
    short_target = ZERO if planning is None else planning.short_target_quantity
    gross = (long.quantity + short.quantity) * market.mark
    net = long.quantity - short.quantity
    realized = long.realized_pnl + short.realized_pnl
    fees = long.fees + short.fees
    projection = outcome.projection
    return DryRunCycleTelemetry(
        cycle_id=outcome.record.cycle_id,
        account_id=account_id,
        symbol=raw_symbol(market.symbol),
        timestamp=market.timestamp,
        mark_price=market.mark,
        equity=equity + realized - fees,
        available_balance=max(equity - gross, ZERO),
        gross_notional=gross,
        net_quantity=net,
        target_net_quantity=long_target - short_target,
        net_gap_quantity=(long_target - short_target) - net,
        long_quantity=long.quantity,
        short_quantity=short.quantity,
        long_target_quantity=long_target,
        short_target_quantity=short_target,
        long_average_price=long.average_price,
        short_average_price=short.average_price,
        realized_pnl=realized,
        fees=fees,
        ideal_order_count=0 if planning is None else len(planning.ideal_orders),
        submit_order_count=submitted,
        fill_count=fills,
        active_order_count=0,
        risk_blocked=not outcome.record.safety_allows_new_risk,
        diagnostics=("REAL_BINANCE_MARKET_INPUT", "SIMULATED_EXECUTION_ONLY", "EXCHANGE_WRITE_CAPABILITY_FALSE"),
        strategy=StrategyTelemetry(
            long_score=projection.long_margin_ratio,
            short_score=projection.short_margin_ratio,
            confidence=projection.confidence,
            long_exposure_scale=projection.long_margin_ratio / Decimal("0.40"),
            short_exposure_scale=projection.short_margin_ratio / Decimal("0.40"),
            allow_new_risk=projection.accepted,
            regime="HPRL_REAL_MARKET_DRYRUN",
            reason="HPRL_EXACT_DUAL_LEG_SIMULATED_EXECUTION",
            model_version=projection.model_id,
        ),
    )


async def run_binance_real_market_dryrun(
    client: object,
    *,
    symbol: str,
    targets: Sequence[PlannedExecutionIntent],
    journal_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    cycle_interval_seconds: float = 0.0,
    fee_rate: Decimal = Decimal("0.0004"),
    policy: BinanceDryRunPolicy | None = None,
    source: str = "model-target-feed",
) -> BinanceRuntimeDryRunReport:
    """Run real Binance input through HPRL Planner with simulated local fills only."""
    if not targets:
        raise ValueError("at least one HPRL target is required")
    if cycle_interval_seconds < 0:
        raise ValueError("cycle_interval_seconds must be nonnegative")
    preflight = await collect_binance_real_market_preflight(client, symbol=symbol)
    if not preflight.passed:
        raise RuntimeError("Binance real-market preflight did not pass")
    normalized = raw_symbol(symbol)
    sim_account_id = "hprl-dryrun:" + preflight.account_id
    fake = build_integrated_fake_runtime(fee_rate=fee_rate)
    loop = ProductionEquivalentHedgeMainLoop(
        account_id=sim_account_id,
        engine=fake.engine,
        ownership=ExecutionOrderOwnershipRegistry(fake.store),
        kill_switch=fake.kill_switch,
        mode=HedgeExecutionMode.HEDGE_SIMULATED,
        engine_kind=ExecutionEngineKind.SIMULATED,
        allowed_symbols=(normalized,),
    )
    adapter = HprlHedgeAdapter(HprlHedgeAdapterPolicy(
        leverage=preflight.leverage,
        target_unit=HprlTargetUnit.MARGIN_EQUITY_RATIO,
    ))
    owned_temp = journal_path is None or checkpoint_path is None
    temp_dir = Path(tempfile.mkdtemp(prefix="hprl-binance-dryrun-")) if owned_temp else None
    journal_file = Path(journal_path) if journal_path is not None else temp_dir / "journal.json"  # type: ignore[operator]
    checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else temp_dir / "checkpoint.json"  # type: ignore[operator]
    journal = ClosedLoopCycleJournalStore(journal_file)
    checkpoint = RecoveryCheckpointStore(checkpoint_file)
    closed = HprlProductionClosedLoop(
        adapter=adapter,
        main_loop=loop,
        source_release="freqtrade-hedge-hprl-v3-runtime-closure-r2",
        journal_store=journal,
        checkpoint_store=checkpoint,
    )
    cycles: list[BinanceRuntimeDryRunCycle] = []
    telemetry: list[DryRunCycleTelemetry] = []
    submit_cursor = 0
    total_fills = 0
    equity = preflight.equity
    try:
        for index, target in enumerate(targets, start=1):
            if raw_symbol(target.symbol) != normalized:
                raise ValueError("target symbol does not match dry-run symbol")
            bid, ask, mark = await client.fetch_real_market_prices(normalized)
            now = datetime.now(UTC)
            market = MarketSnapshot(
                symbol=symbol,
                timestamp=now,
                bid=Decimal(bid), ask=Decimal(ask), mark=Decimal(mark),
                tick_size=Decimal("0.01"), qty_step=Decimal("0.0001"),
            )
            context = _context(
                market=market,
                account=fake.account,
                account_id=sim_account_id,
                equity=equity,
                leverage=preflight.leverage,
            )
            real_evidence = _sha({
                "preflight": preflight.evidence_sha256,
                "sequence": index,
                "bid": str(bid), "ask": str(ask), "mark": str(mark),
                "target": asdict(target),
            })
            outcome = closed.run(
                target,
                projection_sequence=index,
                observed_at=now,
                now=now,
                context=context,
                evidence_digest=real_evidence,
                reconciliation_digest=_sha({"simulation": fake.account.snapshot(), "sequence": index}),
                last_market_sequence=index,
                last_user_sequence=index,
                safety_allows_reduce=True,
                safety_allows_new_risk=True,
            )
            new_submits = fake.exchange.submit_calls[submit_cursor:]
            cycle_fills = 0
            for approved in new_submits:
                fake.exchange.fill_order(
                    approved.client_order_id,
                    quantity=approved.intent.quantity,
                    price=market.mark,
                    exchange_trade_id=f"hprl-dryrun-{index}-{cycle_fills + 1}",
                )
                fake.engine.refresh_order(approved.client_order_id)
                cycle_fills += 1
            submit_cursor = len(fake.exchange.submit_calls)
            total_fills += cycle_fills
            row = _telemetry(
                sequence=index,
                outcome=outcome,
                account=fake.account,
                account_id=sim_account_id,
                market=market,
                equity=equity,
                submitted=len(new_submits),
                fills=cycle_fills,
            )
            telemetry.append(row)
            cycles.append(BinanceRuntimeDryRunCycle(
                sequence=index,
                telemetry=row,
                closed_loop_record_sha256=outcome.record.record_sha256,
                market_evidence_sha256=real_evidence,
                simulated_fill_count=cycle_fills,
            ))
            if cycle_interval_seconds:
                await asyncio.sleep(cycle_interval_seconds)
        safety = BinanceDryRunSafetyContext(
            exchange="binance",
            operation_mode="dry_run",
            real_market_data=True,
            exchange_write_capability=False,
            simulated_execution=True,
            hedge_mode_semantics=preflight.hedge_mode,
            cross_margin_semantics=preflight.cross_margin,
            source_release="freqtrade-hedge-hprl-v3-runtime-closure-r2",
            account_namespace="hprl-dryrun",
        )
        effective_policy = policy or BinanceDryRunPolicy(
            minimum_cycles=max(1, len(telemetry)),
            minimum_duration=timedelta(0),
            maximum_cycle_gap=max(
                BinanceDryRunPolicy().maximum_cycle_gap,
                timedelta(seconds=max(1, int(cycle_interval_seconds) + 5)),
            ),
        )
        acceptance = evaluate_binance_dryrun(telemetry, safety=safety, policy=effective_policy)
        loaded = journal.load()
        report_hash = _sha({
            "preflight": asdict(preflight),
            "cycles": [asdict(item) for item in cycles],
            "acceptance": asdict(acceptance),
            "journal_tip": loaded.tip_sha256,
        })
        return BinanceRuntimeDryRunReport(
            preflight=preflight,
            cycles=tuple(cycles),
            acceptance=acceptance,
            journal_tip_sha256=loaded.tip_sha256,
            journal_valid=loaded.verify(),
            real_exchange_write_count=0,
            simulated_submit_count=len(fake.exchange.submit_calls),
            simulated_fill_count=total_fills,
            source=source,
            evidence_sha256=report_hash,
        )
    finally:
        if temp_dir is not None:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
