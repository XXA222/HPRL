from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256

from freqtrade.hedge.execution.integrated_fake import build_integrated_fake_runtime
from freqtrade.hedge.execution.ownership import ExecutionOrderOwnershipRegistry
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
from freqtrade.hedge.production.closed_loop import (
    ClosedLoopCycleJournalStore,
    ClosedLoopCycleStatus,
    HprlProductionClosedLoop,
)
from freqtrade.hedge.production.hprl_hedge_adapter import (
    HprlHedgeAdapter,
    HprlHedgeAdapterPolicy,
    HprlTargetUnit,
)
from freqtrade.hedge.production.recovery_checkpoint import RecoveryCheckpointStore

D = Decimal
NOW = datetime(2026, 8, 15, 5, 0, tzinfo=UTC)


def _context(*, leverage: str = "3") -> PlanningContext:
    return PlanningContext(
        market=MarketSnapshot(
            symbol="BTC/USDT:USDT",
            timestamp=NOW,
            bid=D("99.9"),
            ask=D("100.1"),
            mark=D("100"),
            tick_size=D("0.1"),
            qty_step=D("0.0001"),
        ),
        wallet=WalletSnapshot(
            balance=D("1000"),
            equity=D("1000"),
            available_balance=D("1000"),
            long=LegPosition(PositionSide.LONG),
            short=LegPosition(PositionSide.SHORT),
            leverage=D(leverage),
        ),
        config=PlannerConfig(),
    )


def _intent(long: float, short: float) -> PlannedExecutionIntent:
    return PlannedExecutionIntent(
        symbol="BTC/USDT:USDT",
        target_long_exposure=long,
        target_short_exposure=short,
        confidence=1.0,
        model_id="hprl-closed-loop-test",
        metadata={"unit": "margin/equity"},
    )


def _runtime(tmp_path):
    fake = build_integrated_fake_runtime()
    loop = ProductionEquivalentHedgeMainLoop(
        account_id="acct",
        engine=fake.engine,
        ownership=ExecutionOrderOwnershipRegistry(fake.store),
        kill_switch=fake.kill_switch,
        mode=HedgeExecutionMode.HEDGE_SIMULATED,
        engine_kind=ExecutionEngineKind.SIMULATED,
        allowed_symbols=("BTCUSDT",),
    )
    adapter = HprlHedgeAdapter(
        HprlHedgeAdapterPolicy(
            leverage=D("3"),
            target_unit=HprlTargetUnit.MARGIN_EQUITY_RATIO,
        )
    )
    journal = ClosedLoopCycleJournalStore(tmp_path / "cycle-journal.json")
    checkpoint = RecoveryCheckpointStore(tmp_path / "recovery-checkpoint.json")
    closed = HprlProductionClosedLoop(
        adapter=adapter,
        main_loop=loop,
        source_release="hprl-v3-closed-loop-test",
        journal_store=journal,
        checkpoint_store=checkpoint,
    )
    return fake, closed, journal, checkpoint


def test_hprl_target_traverses_one_hash_chained_cycle(tmp_path) -> None:
    fake, closed, journal_store, checkpoint_store = _runtime(tmp_path)
    evidence = sha256(b"evidence").hexdigest()
    reconciliation = sha256(b"reconciliation").hexdigest()
    outcome = closed.run(
        _intent(0.25, 0.05),
        projection_sequence=1,
        observed_at=NOW,
        now=NOW,
        context=_context(),
        evidence_digest=evidence,
        reconciliation_digest=reconciliation,
        last_market_sequence=10,
        last_user_sequence=20,
        safety_allows_reduce=True,
        safety_allows_new_risk=True,
    )

    assert outcome.projection.long_margin_ratio == D("0.25")
    assert outcome.projection.short_margin_ratio == D("0.05")
    assert outcome.projection.long_notional_ratio == D("0.75")
    assert outcome.projection.short_notional_ratio == D("0.15")
    assert outcome.main_loop_cycle is not None
    assert outcome.main_loop_cycle.cycle_id == outcome.record.cycle_id
    assert outcome.main_loop_cycle.planning is not None
    assert outcome.main_loop_cycle.planning.long_target_quantity == D("7.5")
    assert outcome.main_loop_cycle.planning.short_target_quantity == D("1.5")
    assert outcome.record.status is ClosedLoopCycleStatus.COMMITTED
    assert outcome.record.writes_attempted > 0
    assert journal_store.load().verify()
    assert journal_store.load().tip_sha256 == outcome.record.record_sha256

    checkpoint = checkpoint_store.load()
    assert checkpoint is not None
    metadata = dict(checkpoint.metadata)
    assert metadata["closed_loop_cycle_id"] == outcome.record.cycle_id
    assert metadata["closed_loop_cycle_sha256"] == outcome.record.record_sha256
    assert checkpoint.projection_chain_sha256 == outcome.record.projection_chain_sha256
    assert not outcome.record.unresolved_client_order_ids
    assert fake.exchange.submit_calls


def test_account_safety_halt_records_cycle_without_exchange_write(tmp_path) -> None:
    fake, closed, journal_store, _ = _runtime(tmp_path)
    outcome = closed.run(
        _intent(0.12, 0.05),
        projection_sequence=1,
        observed_at=NOW,
        now=NOW,
        context=_context(),
        evidence_digest=sha256(b"e").hexdigest(),
        reconciliation_digest=sha256(b"r").hexdigest(),
        last_market_sequence=1,
        last_user_sequence=1,
        safety_allows_reduce=False,
        safety_allows_new_risk=False,
    )
    assert outcome.record.status is ClosedLoopCycleStatus.HALTED
    assert outcome.record.writes_attempted == 0
    assert outcome.main_loop_cycle is None
    assert not fake.exchange.submit_calls
    assert journal_store.load().tip_sha256 == outcome.record.record_sha256


def test_journal_rehydrates_previous_projection_and_allows_fast_derisk(tmp_path) -> None:
    _, closed, journal_store, _ = _runtime(tmp_path)
    common = dict(
        now=NOW,
        context=_context(),
        evidence_digest=sha256(b"e").hexdigest(),
        reconciliation_digest=sha256(b"r").hexdigest(),
        last_market_sequence=1,
        last_user_sequence=1,
        safety_allows_reduce=True,
        safety_allows_new_risk=True,
    )
    first = closed.run(
        _intent(0.25, 0.12), projection_sequence=1, observed_at=NOW, **common
    )
    second = closed.run(
        _intent(0.0, 0.0), projection_sequence=2, observed_at=NOW, **common
    )
    assert first.projection.accepted
    assert second.projection.accepted
    assert second.record.previous_record_sha256 == first.record.record_sha256
    assert second.record.projection_chain_sha256 != first.record.projection_chain_sha256
    assert len(journal_store.load().records) == 2
