"""Single-writer guard backed by a database lease."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from threading import RLock
from typing import Callable

from freqtrade.hedge.concurrency.database_lease import (
    DatabaseLeaseStore,
    LeaseLost,
    LeaseRecord,
    LeaseUnavailable,
)


@dataclass(frozen=True, slots=True)
class SingleWriterStatus:
    valid: bool
    owner_id: str
    lease_name: str
    fencing_token: int | None
    expires_at_ms: int | None
    reason_code: str

    def as_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "owner_id": self.owner_id,
            "lease_name": self.lease_name,
            "fencing_token": self.fencing_token,
            "expires_at_ms": self.expires_at_ms,
            "reason_code": self.reason_code,
        }


class SingleWriterGuard:
    def __init__(
        self,
        store: DatabaseLeaseStore,
        *,
        owner_id: str,
        lease_name: str = "freqtrade-hedge-writer",
        ttl_ms: int = 15_000,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ValueError("owner_id must not be empty.")
        if not isinstance(lease_name, str) or not lease_name.strip():
            raise ValueError("lease_name must not be empty.")
        if len(lease_name.strip()) > 255:
            raise ValueError("lease_name must not exceed 255 characters.")
        if isinstance(ttl_ms, bool) or not isinstance(ttl_ms, int) or ttl_ms <= 0:
            raise ValueError("ttl_ms must be a positive integer.")
        self._store = store
        owner_label = owner_id.strip()
        if len(owner_label) > 222:
            raise ValueError(
                "owner_id label must not exceed 222 characters after fencing suffix reserve."
            )
        self._owner_id = f"{owner_label}:{uuid.uuid4().hex}"
        self._lease_name = lease_name.strip()
        self._ttl_ms = ttl_ms
        self._clock_ms = clock_ms or (lambda: time.time_ns() // 1_000_000)
        self._lease: LeaseRecord | None = None
        self._lost = False
        self._lock = RLock()

    @property
    def lease(self) -> LeaseRecord | None:
        with self._lock:
            return self._lease

    @property
    def owner_id(self) -> str:
        return self._owner_id

    @property
    def lease_name(self) -> str:
        return self._lease_name

    @property
    def ttl_ms(self) -> int:
        return self._ttl_ms

    def _mark_lost(self) -> None:
        self._lease = None
        self._lost = True

    def _now_ms(self) -> int:
        value = self._clock_ms()
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "Single-writer clock must return a nonnegative integer milliseconds value."
            )
        return value

    def acquire(self) -> LeaseRecord:
        with self._lock:
            try:
                now = self._now_ms()
            except Exception as exc:
                self._mark_lost()
                raise LeaseUnavailable("Single-writer clock is invalid.") from exc
            try:
                lease = self._store.acquire(
                    lease_name=self._lease_name,
                    owner_id=self._owner_id,
                    now_ms=now,
                    ttl_ms=self._ttl_ms,
                )
            except Exception as exc:
                self._mark_lost()
                raise LeaseUnavailable("Single-writer lease acquisition failed.") from exc
            if lease is None:
                self._mark_lost()
                raise LeaseUnavailable(
                    "Single-writer lease is held by another process: "
                    f"{self._lease_name}."
                )
            self._lease = lease
            self._lost = False
            return lease

    def renew(self) -> LeaseRecord:
        with self._lock:
            if self._lease is None:
                self._lost = True
                raise LeaseLost("Single-writer lease has not been acquired.")
            try:
                now = self._now_ms()
            except Exception as exc:
                self._mark_lost()
                raise LeaseLost("Single-writer clock is invalid.") from exc
            try:
                renewed = self._store.renew(
                    lease=self._lease,
                    now_ms=now,
                    ttl_ms=self._ttl_ms,
                )
            except Exception as exc:
                self._mark_lost()
                raise LeaseLost("Single-writer lease renewal failed.") from exc
            if renewed is None:
                self._mark_lost()
                raise LeaseLost("Single-writer lease renewal failed.")
            self._lease = renewed
            return renewed

    def status(self) -> SingleWriterStatus:
        with self._lock:
            lease = self._lease
            if lease is None or self._lost:
                return SingleWriterStatus(
                    False,
                    self._owner_id,
                    self._lease_name,
                    None,
                    None,
                    "SINGLE_WRITER_LEASE_INVALID",
                )
            try:
                now = self._now_ms()
            except Exception:
                self._mark_lost()
                return SingleWriterStatus(
                    False,
                    self._owner_id,
                    self._lease_name,
                    None,
                    None,
                    "SINGLE_WRITER_CLOCK_INVALID",
                )
            try:
                current = self._store.read(lease_name=self._lease_name)
            except Exception:
                self._mark_lost()
                return SingleWriterStatus(
                    False,
                    self._owner_id,
                    self._lease_name,
                    None,
                    None,
                    "SINGLE_WRITER_STORE_UNAVAILABLE",
                )
            valid = (
                current is not None
                and current.owner_id == lease.owner_id
                and current.fencing_token == lease.fencing_token
                and current.is_valid(now_ms=now)
            )
            if not valid:
                self._mark_lost()
                return SingleWriterStatus(
                    False,
                    self._owner_id,
                    self._lease_name,
                    None,
                    None,
                    "SINGLE_WRITER_LEASE_INVALID",
                )
            self._lease = current
            return SingleWriterStatus(
                True,
                self._owner_id,
                self._lease_name,
                current.fencing_token,
                current.expires_at_ms,
                "SINGLE_WRITER_LEASE_VALID",
            )

    def assert_valid(self) -> LeaseRecord:
        status = self.status()
        with self._lock:
            if not status.valid or self._lease is None:
                raise LeaseLost(
                    "Single-writer lease is no longer valid: "
                    f"{status.reason_code}."
                )
            return self._lease

    def can_increase_risk(self) -> bool:
        return self.status().valid

    def release(self) -> bool:
        with self._lock:
            lease = self._lease
            self._mark_lost()
            if lease is None:
                return False
            try:
                return self._store.release(lease=lease)
            except Exception:
                return False
