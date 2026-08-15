import asyncio

import pytest

from freqtrade.enums.hedge import PositionSide
from freqtrade.hedge.concurrency import (
    AccountLockKey,
    AsyncHierarchicalLockManager,
    HierarchicalLockManager,
    LockOrderViolation,
    OrderLockKey,
    PositionLockKey,
)


def test_account_position_order_ordering_is_enforced() -> None:
    manager = HierarchicalLockManager()
    account = AccountLockKey("main")
    position = PositionLockKey("main", "ETH/USDT:USDT", PositionSide.LONG)
    order = OrderLockKey("main", "ETH/USDT:USDT", PositionSide.LONG, "order-1")
    with manager.acquire(account):
        with manager.acquire(position):
            with manager.acquire(order):
                pass
    with manager.acquire(order):
        with pytest.raises(LockOrderViolation):
            with manager.acquire(account):
                pass


def test_async_hierarchy_does_not_block_event_loop_and_enforces_order() -> None:
    async def scenario() -> None:
        manager = AsyncHierarchicalLockManager(default_timeout_seconds=1)
        account = AccountLockKey("main")
        position = PositionLockKey("main", "ETH/USDT:USDT", PositionSide.SHORT)
        async with manager.acquire(account):
            await asyncio.sleep(0)
            async with manager.acquire(position):
                await asyncio.sleep(0)
        async with manager.acquire(position):
            with pytest.raises(LockOrderViolation):
                async with manager.acquire(account):
                    pass

    asyncio.run(scenario())


def test_async_acquire_many_sorts_account_position_order() -> None:
    async def scenario() -> None:
        manager = AsyncHierarchicalLockManager(default_timeout_seconds=1)
        account = AccountLockKey("main")
        position = PositionLockKey("main", "ETH/USDT:USDT", PositionSide.LONG)
        order = OrderLockKey("main", "ETH/USDT:USDT", PositionSide.LONG, "order-1")
        async with manager.acquire_many([order, account, position]) as acquired:
            assert acquired == (account, position, order)
            await asyncio.sleep(0)

    asyncio.run(scenario())


def test_async_hierarchy_rejects_reentrant_same_lock() -> None:
    async def scenario() -> None:
        manager = AsyncHierarchicalLockManager(default_timeout_seconds=1)
        account = AccountLockKey("main")
        async with manager.acquire(account):
            with pytest.raises(LockOrderViolation, match="not reentrant"):
                async with manager.acquire(account):
                    pass

    asyncio.run(scenario())
