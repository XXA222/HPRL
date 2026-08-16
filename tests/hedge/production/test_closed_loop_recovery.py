from __future__ import annotations

from dataclasses import replace
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
    LegPosition, MarketSnapshot, PlannerConfig, PlanningContext, PositionSide, WalletSnapshot,
)
from freqtrade.hedge.production.closed_loop import ClosedLoopCycleJournalStore, HprlProductionClosedLoop
from freqtrade.hedge.production.closed_loop_recovery import ClosedLoopRecoveryBarrier
from freqtrade.hedge.production.hprl_hedge_adapter import HprlHedgeAdapter, HprlHedgeAdapterPolicy, HprlTargetUnit
from freqtrade.hedge.production.recovery_checkpoint import RecoveryCheckpointStore

D = Decimal
NOW = datetime(2026, 8, 15, 5, 0, tzinfo=UTC)


def _build(tmp_path):
    fake = build_integrated_fake_runtime()
    loop = ProductionEquivalentHedgeMainLoop(
        account_id="acct", engine=fake.engine,
        ownership=ExecutionOrderOwnershipRegistry(fake.store), kill_switch=fake.kill_switch,
        mode=HedgeExecutionMode.HEDGE_SIMULATED, engine_kind=ExecutionEngineKind.SIMULATED,
        allowed_symbols=("BTCUSDT",),
    )
    adapter = HprlHedgeAdapter(HprlHedgeAdapterPolicy(
        leverage=D("3"), target_unit=HprlTargetUnit.MARGIN_EQUITY_RATIO,
    ))
    journal = ClosedLoopCycleJournalStore(tmp_path / "j.json")
    checkpoint = RecoveryCheckpointStore(tmp_path / "c.json")
    closed = HprlProductionClosedLoop(
        adapter=adapter, main_loop=loop, source_release="release",
        journal_store=journal, checkpoint_store=checkpoint,
    )
    context = PlanningContext(
        market=MarketSnapshot("BTC/USDT:USDT", NOW, D("100"), D("100"), D("100")),
        wallet=WalletSnapshot(D("1000"), D("1000"), D("1000"),
            LegPosition(PositionSide.LONG), LegPosition(PositionSide.SHORT), leverage=D("3")),
        config=PlannerConfig(),
    )
    intent = PlannedExecutionIntent("BTC/USDT:USDT", .12, .05, 1.0, "model")
    e = sha256(b"e").hexdigest(); r = sha256(b"r").hexdigest()
    closed.run(intent, projection_sequence=1, observed_at=NOW, now=NOW, context=context,
        evidence_digest=e, reconciliation_digest=r, last_market_sequence=1,
        last_user_sequence=2, safety_allows_reduce=True, safety_allows_new_risk=True)
    return fake, journal, checkpoint, e, r


def test_restart_barrier_requires_exact_journal_checkpoint_link(tmp_path) -> None:
    fake, journal_store, checkpoint_store, evidence, reconciliation = _build(tmp_path)
    report = ClosedLoopRecoveryBarrier().evaluate(
        checkpoint_store.load(), journal_store.load(), orders=fake.core.list_orders(), now=NOW,
        current_evidence_digest=evidence, current_reconciliation_digest=reconciliation,
    )
    assert report.passed
    assert report.allow_new_risk
    assert report.reasons == ()


def test_restart_barrier_blocks_checkpoint_cycle_hash_mismatch(tmp_path) -> None:
    fake, journal_store, checkpoint_store, evidence, reconciliation = _build(tmp_path)
    checkpoint = checkpoint_store.load(); assert checkpoint is not None
    metadata = dict(checkpoint.metadata)
    metadata["closed_loop_cycle_sha256"] = "f" * 64
    tampered = replace(checkpoint, metadata=tuple(metadata.items()))
    report = ClosedLoopRecoveryBarrier().evaluate(
        tampered, journal_store.load(), orders=fake.core.list_orders(), now=NOW,
        current_evidence_digest=evidence, current_reconciliation_digest=reconciliation,
    )
    assert not report.passed
    assert not report.allow_new_risk
    assert "CLOSED_LOOP_CYCLE_HASH_MISMATCH" in report.reasons
