"""Cross-process Binance weight coordination backed by SQLite.

The store is deliberately small and independent of the execution ledger so every
local process sharing an IP/API quota can reserve from the same minute window.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Callable


@dataclass(frozen=True, slots=True)
class SharedWeightDecision:
    granted: bool
    used_weight: int
    retry_after_seconds: float
    window_epoch_minute: int


class SqliteSharedWeightBudget:
    def __init__(
        self,
        path: str | Path,
        *,
        limit_per_minute: int,
        reserve_weight: int,
        now_epoch: Callable[[], float] | None = None,
    ) -> None:
        if limit_per_minute <= 0:
            raise ValueError("limit_per_minute must be positive")
        if reserve_weight < 0 or reserve_weight >= limit_per_minute:
            raise ValueError("reserve_weight must be in [0, limit_per_minute)")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.limit = limit_per_minute
        self.reserve = reserve_weight
        self._now = now_epoch or time.time
        self._lock = RLock()
        self._initialize()

    @property
    def usable_capacity(self) -> int:
        return self.limit - self.reserve

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS binance_shared_weight_budget (
                    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
                    window_epoch_minute INTEGER NOT NULL,
                    reserved_weight INTEGER NOT NULL,
                    observed_remote_weight INTEGER NOT NULL,
                    updated_at_epoch REAL NOT NULL
                )
                """
            )

    def _window(self) -> tuple[int, float, float]:
        now = float(self._now())
        if now < 0:
            raise ValueError("epoch time cannot be negative")
        minute = int(now // 60)
        retry = max(0.01, ((minute + 1) * 60) - now)
        return minute, now, retry

    def reserve_weight(self, weight: int) -> SharedWeightDecision:
        if weight <= 0:
            raise ValueError("weight must be positive")
        if weight > self.usable_capacity:
            raise ValueError("weight exceeds usable shared capacity")
        minute, now, retry = self._window()
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT window_epoch_minute, reserved_weight, observed_remote_weight "
                    "FROM binance_shared_weight_budget WHERE singleton_id=1"
                ).fetchone()
                if row is None or int(row[0]) != minute:
                    reserved = 0
                    remote = 0
                else:
                    reserved = int(row[1])
                    remote = int(row[2])
                used = max(reserved, remote)
                granted = used + weight <= self.usable_capacity
                if granted:
                    reserved += weight
                    used = max(reserved, remote)
                connection.execute(
                    """
                    INSERT INTO binance_shared_weight_budget(
                        singleton_id, window_epoch_minute, reserved_weight,
                        observed_remote_weight, updated_at_epoch
                    ) VALUES(1, ?, ?, ?, ?)
                    ON CONFLICT(singleton_id) DO UPDATE SET
                        window_epoch_minute=excluded.window_epoch_minute,
                        reserved_weight=excluded.reserved_weight,
                        observed_remote_weight=excluded.observed_remote_weight,
                        updated_at_epoch=excluded.updated_at_epoch
                    """,
                    (minute, reserved, remote, now),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return SharedWeightDecision(granted, used, 0.0 if granted else retry, minute)

    def observe_remote_weight(self, weight: int) -> int:
        if weight < 0:
            raise ValueError("weight cannot be negative")
        minute, now, _ = self._window()
        with self._lock, closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT window_epoch_minute, reserved_weight, observed_remote_weight "
                    "FROM binance_shared_weight_budget WHERE singleton_id=1"
                ).fetchone()
                if row is None or int(row[0]) != minute:
                    reserved = 0
                    remote = weight
                else:
                    reserved = int(row[1])
                    remote = max(int(row[2]), weight)
                connection.execute(
                    """
                    INSERT INTO binance_shared_weight_budget(
                        singleton_id, window_epoch_minute, reserved_weight,
                        observed_remote_weight, updated_at_epoch
                    ) VALUES(1, ?, ?, ?, ?)
                    ON CONFLICT(singleton_id) DO UPDATE SET
                        window_epoch_minute=excluded.window_epoch_minute,
                        reserved_weight=excluded.reserved_weight,
                        observed_remote_weight=excluded.observed_remote_weight,
                        updated_at_epoch=excluded.updated_at_epoch
                    """,
                    (minute, reserved, remote, now),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return max(reserved, remote)

    def snapshot(self) -> SharedWeightDecision:
        minute, _now, retry = self._window()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT window_epoch_minute, reserved_weight, observed_remote_weight "
                "FROM binance_shared_weight_budget WHERE singleton_id=1"
            ).fetchone()
        if row is None or int(row[0]) != minute:
            return SharedWeightDecision(True, 0, 0.0, minute)
        used = max(int(row[1]), int(row[2]))
        return SharedWeightDecision(used < self.usable_capacity, used, retry, minute)
