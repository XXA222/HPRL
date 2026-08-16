"""Immutable, hash-chained production evidence ledger."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
import os
from pathlib import Path
from contextlib import contextmanager
from typing import Iterable, Mapping

from .contracts import EvidenceKind, EvidenceStatus


def _aware(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _metadata(value: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for key, item in sorted(value.items()):
        if not str(key).strip():
            raise ValueError("evidence metadata key must not be empty")
        encoded = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        result.append((str(key), encoded))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    kind: EvidenceKind
    status: EvidenceStatus
    observed_at: datetime
    expires_at: datetime
    artifact_sha256: str
    producer: str
    metadata: tuple[tuple[str, str], ...] = ()
    previous_record_sha256: str = "0" * 64
    record_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", _aware(self.observed_at, "observed_at"))
        object.__setattr__(self, "expires_at", _aware(self.expires_at, "expires_at"))
        if self.expires_at <= self.observed_at:
            raise ValueError("evidence expires_at must be after observed_at")
        for name in ("artifact_sha256", "previous_record_sha256"):
            digest = str(getattr(self, name)).lower()
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"{name} must be a sha256 hex digest")
            object.__setattr__(self, name, digest)
        producer = str(self.producer).strip()
        if not producer or len(producer) > 128:
            raise ValueError("producer is required")
        object.__setattr__(self, "producer", producer)
        payload = self.payload(include_record_hash=False)
        digest = sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
        object.__setattr__(self, "record_sha256", digest)

    @classmethod
    def create(
        cls,
        *,
        kind: EvidenceKind,
        status: EvidenceStatus,
        observed_at: datetime,
        ttl: timedelta,
        artifact_sha256: str,
        producer: str,
        metadata: Mapping[str, object] | None = None,
        previous_record_sha256: str = "0" * 64,
    ) -> "EvidenceRecord":
        if ttl <= timedelta(0):
            raise ValueError("evidence ttl must be positive")
        return cls(
            kind=kind,
            status=status,
            observed_at=observed_at,
            expires_at=observed_at + ttl,
            artifact_sha256=artifact_sha256,
            producer=producer,
            metadata=_metadata(metadata or {}),
            previous_record_sha256=previous_record_sha256,
        )

    def is_fresh(self, now: datetime) -> bool:
        return _aware(now, "now") < self.expires_at

    def payload(self, *, include_record_hash: bool = True) -> dict[str, object]:
        body: dict[str, object] = {
            "kind": self.kind.value,
            "status": self.status.value,
            "observed_at": self.observed_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "artifact_sha256": self.artifact_sha256,
            "producer": self.producer,
            "metadata": list(self.metadata),
            "previous_record_sha256": self.previous_record_sha256,
        }
        if include_record_hash:
            body["record_sha256"] = self.record_sha256
        return body


class EvidenceLedger:
    """Append-only ledger.  A newer record never erases older evidence."""

    def __init__(self, records: Iterable[EvidenceRecord] = ()) -> None:
        self._records: list[EvidenceRecord] = []
        for record in records:
            self.append(record)

    def append(self, record: EvidenceRecord) -> EvidenceRecord:
        if not isinstance(record, EvidenceRecord):
            raise TypeError("record must be EvidenceRecord")
        expected = self._records[-1].record_sha256 if self._records else "0" * 64
        if record.previous_record_sha256 != expected:
            raise ValueError("evidence chain predecessor mismatch")
        if self._records and record.observed_at < self._records[-1].observed_at:
            raise ValueError("evidence ledger timestamp regression")
        self._records.append(record)
        return record

    def add(
        self,
        *,
        kind: EvidenceKind,
        status: EvidenceStatus,
        observed_at: datetime,
        ttl: timedelta,
        artifact_sha256: str,
        producer: str,
        metadata: Mapping[str, object] | None = None,
    ) -> EvidenceRecord:
        previous = self._records[-1].record_sha256 if self._records else "0" * 64
        return self.append(
            EvidenceRecord.create(
                kind=kind,
                status=status,
                observed_at=observed_at,
                ttl=ttl,
                artifact_sha256=artifact_sha256,
                producer=producer,
                metadata=metadata,
                previous_record_sha256=previous,
            )
        )

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records)

    def latest(self, kind: EvidenceKind) -> EvidenceRecord | None:
        return next((item for item in reversed(self._records) if item.kind is kind), None)

    def latest_map(self) -> dict[EvidenceKind, EvidenceRecord]:
        result: dict[EvidenceKind, EvidenceRecord] = {}
        for item in self._records:
            result[item.kind] = item
        return result

    def verify_chain(self) -> bool:
        previous = "0" * 64
        previous_time: datetime | None = None
        for record in self._records:
            if record.previous_record_sha256 != previous:
                return False
            rebuilt = EvidenceRecord(
                kind=record.kind,
                status=record.status,
                observed_at=record.observed_at,
                expires_at=record.expires_at,
                artifact_sha256=record.artifact_sha256,
                producer=record.producer,
                metadata=record.metadata,
                previous_record_sha256=record.previous_record_sha256,
            )
            if rebuilt.record_sha256 != record.record_sha256:
                return False
            if previous_time is not None and record.observed_at < previous_time:
                return False
            previous = record.record_sha256
            previous_time = record.observed_at
        return True


    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "freqtrade-hedge-production-evidence-ledger-v1",
            "records": [item.payload() for item in self._records],
            "digest": self.digest(),
        }

    def save_atomic(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        raw = json.dumps(self.to_payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
        temporary.write_text(raw, encoding="utf-8")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(str(target.parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return target

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "EvidenceLedger":
        if payload.get("schema") != "freqtrade-hedge-production-evidence-ledger-v1":
            raise ValueError("unsupported production evidence ledger schema")
        raw_records = payload.get("records")
        if not isinstance(raw_records, list):
            raise ValueError("evidence ledger records must be a list")
        records: list[EvidenceRecord] = []
        for item in raw_records:
            if not isinstance(item, Mapping):
                raise ValueError("evidence record payload must be an object")
            metadata_raw = item.get("metadata", [])
            metadata: list[tuple[str, str]] = []
            if not isinstance(metadata_raw, list):
                raise ValueError("evidence metadata must be a list")
            for pair in metadata_raw:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError("evidence metadata entry is invalid")
                metadata.append((str(pair[0]), str(pair[1])))
            record = EvidenceRecord(
                kind=EvidenceKind(str(item["kind"])),
                status=EvidenceStatus(str(item["status"])),
                observed_at=datetime.fromisoformat(str(item["observed_at"])),
                expires_at=datetime.fromisoformat(str(item["expires_at"])),
                artifact_sha256=str(item["artifact_sha256"]),
                producer=str(item["producer"]),
                metadata=tuple(metadata),
                previous_record_sha256=str(item["previous_record_sha256"]),
            )
            expected_hash = str(item.get("record_sha256", ""))
            if expected_hash and record.record_sha256 != expected_hash:
                raise ValueError("evidence record hash mismatch")
            records.append(record)
        ledger = cls(records)
        expected_digest = str(payload.get("digest", ""))
        if expected_digest and ledger.digest() != expected_digest:
            raise ValueError("evidence ledger digest mismatch")
        return ledger

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceLedger":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("evidence ledger document must be an object")
        return cls.from_payload(payload)

    def digest(self) -> str:
        payload = [item.payload() for item in self._records]
        return sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()


class EvidenceConcurrencyError(RuntimeError):
    """Raised when an evidence ledger compare-and-swap detects another writer."""


class EvidenceLedgerStore:
    """Optimistic atomic persistence wrapper for the hash-chained ledger.

    The caller supplies the digest it loaded.  A later append refuses to overwrite a
    different on-disk digest, closing the lost-update window between concurrent operator
    or automation processes.  The existing SingleWriter/fencing layer should still be
    used; this is a final fail-closed guard rather than a distributed lock service.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> tuple[EvidenceLedger, str]:
        ledger = EvidenceLedger.load(self.path) if self.path.exists() else EvidenceLedger()
        return ledger, ledger.digest()

    @contextmanager
    def _exclusive_lock(self):
        lock_path = self.path.with_name(self.path.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            try:
                import fcntl
            except ImportError as exc:  # pragma: no cover - production container is POSIX
                raise RuntimeError("evidence ledger locking requires POSIX fcntl") from exc
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            try:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()

    def save_if_unchanged(
        self,
        ledger: EvidenceLedger,
        *,
        expected_digest: str,
    ) -> Path:
        with self._exclusive_lock():
            if self.path.exists():
                current = EvidenceLedger.load(self.path)
                actual_digest = current.digest()
            else:
                actual_digest = EvidenceLedger().digest()
            if actual_digest != expected_digest:
                raise EvidenceConcurrencyError(
                    f"evidence ledger changed concurrently: expected={expected_digest} actual={actual_digest}"
                )
            return ledger.save_atomic(self.path)

    def append_record(
        self,
        *,
        kind: EvidenceKind,
        status: EvidenceStatus,
        observed_at: datetime,
        ttl: timedelta,
        artifact_sha256: str,
        producer: str,
        metadata: Mapping[str, object] | None = None,
        expected_digest: str,
    ) -> EvidenceRecord:
        with self._exclusive_lock():
            ledger, loaded_digest = self.load()
            if loaded_digest != expected_digest:
                raise EvidenceConcurrencyError("evidence ledger changed before append")
            record = ledger.add(
                kind=kind,
                status=status,
                observed_at=observed_at,
                ttl=ttl,
                artifact_sha256=artifact_sha256,
                producer=producer,
                metadata=metadata,
            )
            ledger.save_atomic(self.path)
            return record
