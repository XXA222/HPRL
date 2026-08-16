from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from freqtrade.hedge.production.closed_loop import (
    ClosedLoopCycleRecord,
    ClosedLoopCycleStatus,
    ClosedLoopJournalConcurrencyError,
    ZERO_HASH,
)
from freqtrade.hedge.production.closed_loop_sql import SqlClosedLoopCycleJournalStore

NOW = datetime(2026, 8, 15, 5, 0, tzinfo=UTC)


def _record(sequence: int, previous: str, suffix: str) -> ClosedLoopCycleRecord:
    h = lambda value: sha256(value.encode()).hexdigest()
    return ClosedLoopCycleRecord(
        sequence=sequence, cycle_id=f"cycle-{suffix}", observed_at=NOW,
        source_release="release", model_id="model", symbol="BTCUSDT",
        projection_sequence=sequence, projection_observed_at=NOW,
        projection_source_sha256=h("source-" + suffix),
        projection_semantic_sha256=h("projection-" + suffix),
        long_margin_ratio=Decimal("0.12"), short_margin_ratio=Decimal("0.05"),
        long_notional_ratio=Decimal("0.36"), short_notional_ratio=Decimal("0.15"),
        confidence=Decimal("1"), projection_accepted=True, projection_reasons=(),
        projection_chain_sha256=h("chain-" + suffix), planner_profile_sha256=h("planner"),
        input_state_sha256=h("input-" + suffix), planning_sha256=h("planning-" + suffix),
        execution_sha256=h("execution-" + suffix), reconciliation_digest=h("recon"),
        evidence_digest=h("evidence"), safety_allows_reduce=True,
        safety_allows_new_risk=True, status=ClosedLoopCycleStatus.COMMITTED,
        writes_attempted=1, previous_record_sha256=previous,
    )


def _store():
    pytest.importorskip("humanize", reason="full Freqtrade persistence runtime not installed")
    from freqtrade.persistence.hedge_models import HedgeModelBase

    engine = create_engine("sqlite+pysqlite:///:memory:")
    HedgeModelBase.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    return SqlClosedLoopCycleJournalStore(factory, account_id="acct")


def test_sql_journal_persists_and_reloads_hash_chain() -> None:
    store = _store()
    one = _record(1, ZERO_HASH, "one")
    after_one = store.append_atomic(one, expected_previous_sha256=ZERO_HASH)
    assert after_one.tip_sha256 == one.record_sha256
    two = _record(2, one.record_sha256, "two")
    store.append_atomic(two, expected_previous_sha256=one.record_sha256)
    loaded = store.load()
    assert loaded.verify()
    assert [item.cycle_id for item in loaded.records] == ["cycle-one", "cycle-two"]
    assert loaded.tip_sha256 == two.record_sha256


def test_sql_journal_rejects_stale_compare_and_swap_tip() -> None:
    store = _store()
    one = _record(1, ZERO_HASH, "one")
    store.append_atomic(one, expected_previous_sha256=ZERO_HASH)
    two = _record(2, one.record_sha256, "two")
    with pytest.raises(ClosedLoopJournalConcurrencyError):
        store.append_atomic(two, expected_previous_sha256=ZERO_HASH)
