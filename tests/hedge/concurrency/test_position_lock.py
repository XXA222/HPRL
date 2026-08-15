import threading
import time
from decimal import Decimal

import pytest

from freqtrade.enums.hedge import PositionSide
from freqtrade.hedge.concurrency import (
    LockOrderViolation,
    PositionLockKey,
    PositionLockManager,
    PositionLockTimeout,
)


def test_two_threads_cannot_over_reserve_confirmed_position() -> None:
    manager = PositionLockManager(default_timeout_seconds=1)
    barrier = threading.Barrier(3)
    reservations = []

    def worker() -> None:
        barrier.wait()
        reservations.append(
            manager.reserve_reduce(
                account_id="main",
                symbol="ETH/USDT:USDT",
                position_side=PositionSide.LONG,
                requested_quantity=Decimal("8"),
                confirmed_quantity=Decimal("10"),
            )
        )

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join()

    assert sum((item.allowed_quantity for item in reservations), Decimal("0")) == Decimal("10")
    for item in reservations:
        item.release()


def test_long_lock_does_not_block_short_operation() -> None:
    manager = PositionLockManager(default_timeout_seconds=0.2)
    entered = threading.Event()
    release = threading.Event()

    def hold_long() -> None:
        with manager.lock(account_id="main", symbol="ETH/USDT:USDT", position_side="LONG"):
            entered.set()
            release.wait(1)

    thread = threading.Thread(target=hold_long)
    thread.start()
    assert entered.wait(1)
    started = time.monotonic()
    with manager.lock(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side="SHORT",
        timeout_seconds=0.1,
    ):
        pass
    assert time.monotonic() - started < 0.1
    release.set()
    thread.join()


def test_same_side_lock_times_out() -> None:
    manager = PositionLockManager(default_timeout_seconds=0.05)
    entered = threading.Event()
    release = threading.Event()

    def hold() -> None:
        with manager.lock(account_id="main", symbol="ETH/USDT:USDT", position_side="LONG"):
            entered.set()
            release.wait(1)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(1)
    with pytest.raises(PositionLockTimeout):
        with manager.lock(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            timeout_seconds=0.02,
        ):
            pass
    release.set()
    thread.join()


def test_fixed_lock_order_is_enforced() -> None:
    manager = PositionLockManager()
    with manager.lock(account_id="main", symbol="ETH/USDT:USDT", position_side="SHORT"):
        with pytest.raises(LockOrderViolation):
            with manager.lock(account_id="main", symbol="ETH/USDT:USDT", position_side="LONG"):
                pass


def test_wait_for_graph_detects_deadlock_cycle() -> None:
    manager = PositionLockManager()
    first = PositionLockKey("main", "BTC/USDT:USDT", "LONG")
    second = PositionLockKey("main", "ETH/USDT:USDT", "LONG")
    first_entry = manager._entry(first)
    second_entry = manager._entry(second)
    with manager._state_lock:
        first_entry.owner_thread_id = 101
        second_entry.owner_thread_id = 202
        manager._waiting[101] = second
        assert manager._would_deadlock(202, 101)


def test_reduce_reservation_context_releases_on_normal_exit() -> None:
    manager = PositionLockManager()
    key = PositionLockKey("main", "ETH/USDT:USDT", "LONG")
    with manager.reserve_reduce(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side="LONG",
        requested_quantity=Decimal("3"),
        confirmed_quantity=Decimal("10"),
    ):
        assert manager.pending_reduce_quantity(key) == Decimal("3")
    assert manager.pending_reduce_quantity(key) == 0


def test_pre_reservation_check_runs_inside_position_lock() -> None:
    manager = PositionLockManager()
    key = PositionLockKey("main", "ETH/USDT:USDT", "LONG")

    def reject() -> None:
        raise RuntimeError("lease lost")

    with pytest.raises(RuntimeError, match="lease lost"):
        manager.reserve_reduce(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side="LONG",
            requested_quantity=Decimal("3"),
            confirmed_quantity=Decimal("10"),
            pre_reservation_check=reject,
        )
    assert manager.pending_reduce_quantity(key) == 0


def test_nonfinite_timeout_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        PositionLockManager(default_timeout_seconds=float("nan"))


def test_existing_pending_reduce_is_combined_with_local_reservations() -> None:
    manager = PositionLockManager()
    first = manager.reserve_reduce(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        requested_quantity=Decimal("7"),
        confirmed_quantity=Decimal("10"),
        existing_pending_reduce_quantity=Decimal("2"),
    )
    second = manager.reserve_reduce(
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side=PositionSide.LONG,
        requested_quantity=Decimal("7"),
        confirmed_quantity=Decimal("10"),
        existing_pending_reduce_quantity=Decimal("2"),
    )
    assert first.allowed_quantity == Decimal("7")
    assert second.allowed_quantity == Decimal("1")
    first.release()
    second.release()


def test_boolean_timeout_is_rejected() -> None:
    import pytest

    manager = PositionLockManager()
    with pytest.raises(ValueError, match="positive finite"):
        with manager.lock(
            account_id="main",
            symbol="ETH/USDT:USDT",
            position_side=PositionSide.LONG,
            timeout_seconds=True,
        ):
            pass


def test_position_reservation_identity_includes_exchange() -> None:
    manager = PositionLockManager()
    binance = manager.reserve_reduce(
        exchange="binance",
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side="LONG",
        requested_quantity=Decimal("1"),
        confirmed_quantity=Decimal("1"),
    )
    gateio = manager.reserve_reduce(
        exchange="gateio",
        account_id="main",
        symbol="ETH/USDT:USDT",
        position_side="LONG",
        requested_quantity=Decimal("1"),
        confirmed_quantity=Decimal("1"),
    )
    assert binance.allowed_quantity == Decimal("1")
    assert gateio.allowed_quantity == Decimal("1")
    exchanges = {row["exchange"] for row in manager.reservation_snapshot()}
    assert exchanges == {"binance", "gateio"}
    binance.release()
    gateio.release()
