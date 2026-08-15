from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from freqtrade.hedge.exchange.base import (
    AccountSnapshotFact,
    FillFact,
    OrderFact,
    PositionFact,
)


class FakeClock:
    def __init__(self, value: datetime | None = None) -> None:
        self.value = value or datetime(2026, 7, 26, 0, 0, tzinfo=UTC)
        self.mono = 100.0

    def now(self) -> datetime:
        return self.value

    def monotonic(self) -> float:
        return self.mono

    async def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)
        self.mono += seconds


class FakeRepository:
    def __init__(self) -> None:
        self.positions = []
        self.orders = []
        self.fills = []
        self.account_snapshots = []
        self.balance_snapshots = []
        self.events = []
        self.runs = {}
        self.diffs = []
        self.local_positions = []
        self.local_orders = []

    async def append_position_snapshots(
        self,
        facts,
        *,
        reconciliation_run_id=None,
    ) -> None:
        self.positions.extend(facts)

    async def append_order_snapshots(
        self,
        facts,
        *,
        reconciliation_run_id=None,
    ) -> None:
        self.orders.extend(facts)

    async def append_fill_events(
        self,
        facts,
        *,
        reconciliation_run_id=None,
    ) -> None:
        known = {(item.account_id, item.exchange_trade_id) for item in self.fills}
        self.fills.extend(
            item
            for item in facts
            if (item.account_id, item.exchange_trade_id) not in known
        )

    async def append_account_snapshot(
        self,
        fact,
        *,
        reconciliation_run_id=None,
    ) -> None:
        self.account_snapshots.append(fact)

    async def append_balance_snapshots(
        self,
        facts,
        *,
        reconciliation_run_id=None,
    ) -> None:
        self.balance_snapshots.extend(facts)

    async def append_account_events(self, facts) -> None:
        self.events.extend(facts)

    async def begin_reconciliation(
        self,
        *,
        account_id,
        kind,
        started_at,
    ):
        run_id = f"run-{len(self.runs) + 1}"
        self.runs[run_id] = {
            "account_id": account_id,
            "kind": kind,
            "started_at": started_at,
            "status": "RUNNING",
        }
        return run_id

    async def append_reconciliation_diffs(self, run_id, diffs) -> None:
        self.diffs.extend(diffs)

    async def complete_reconciliation(
        self,
        run_id,
        *,
        completed_at,
        status,
        reason,
    ) -> None:
        self.runs[run_id].update(
            status=status,
            reason=reason,
            completed_at=completed_at,
        )

    async def load_active_positions(self, account_id):
        return tuple(self.local_positions)

    async def load_active_orders(self, account_id):
        return tuple(self.local_orders)

    async def has_fill(self, account_id, symbol, exchange_trade_id) -> bool:
        return any(
            item.account_id == account_id
            and item.symbol == symbol
            and item.exchange_trade_id == exchange_trade_id
            for item in self.fills
        )


def position(
    clock,
    symbol="BTCUSDT",
    side="LONG",
    qty="1",
    update=1000,
    margin="cross",
):
    return PositionFact(
        "acct",
        symbol,
        side,
        Decimal(qty),
        Decimal("100"),
        Decimal("101"),
        Decimal("1"),
        None,
        5,
        margin,
        update,
        clock.now(),
        "TEST",
        {},
    )


def order(
    clock,
    symbol="BTCUSDT",
    oid="10",
    status="NEW",
    filled="0",
    update=1000,
):
    return OrderFact(
        "acct",
        symbol,
        "LONG",
        oid,
        f"c-{oid}",
        "BUY",
        "LIMIT",
        status,
        Decimal("1"),
        Decimal(filled),
        Decimal("100"),
        False,
        update,
        clock.now(),
        "TEST",
        {},
    )


def fill(clock, tid="20", oid="10"):
    return FillFact(
        "acct",
        "BTCUSDT",
        "LONG",
        tid,
        oid,
        "BUY",
        Decimal("0.1"),
        Decimal("100"),
        Decimal("0.01"),
        "USDT",
        Decimal("0"),
        1000,
        clock.now(),
        "TEST",
        {},
    )


def snapshot(clock):
    return AccountSnapshotFact(
        "acct",
        Decimal("1000"),
        Decimal("900"),
        Decimal("1001"),
        Decimal("10"),
        Decimal("2"),
        Decimal("1"),
        clock.now(),
        clock.now(),
        clock.now(),
        {},
    )
