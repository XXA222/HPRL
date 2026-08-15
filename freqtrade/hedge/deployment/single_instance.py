"""Cross-platform operating-system single-instance lock."""

from __future__ import annotations

import json
import os
import secrets
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO


class InstanceLockError(RuntimeError):
    """Raised when another supervisor already owns the instance lock."""


@dataclass(frozen=True, slots=True)
class InstanceIdentity:
    pid: int
    token: str
    acquired_at_utc: str


class SingleInstanceLock:
    """Hold byte zero of a file under an OS lock until explicit release.

    The lock byte is initialized before any contender tries to acquire it.
    Contenders never read the protected byte before locking it. This matters
    on Windows, where reading a byte already protected by ``msvcrt.locking``
    may raise ``PermissionError`` before the non-blocking lock call runs.
    """

    _LOCK_BYTE_COUNT = 1

    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: BinaryIO | None = None
        self.identity: InstanceIdentity | None = None

    def acquire(self) -> InstanceIdentity:
        if self._handle is not None:
            raise InstanceLockError("instance lock is already held by this object")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            handle = self._open_handle()
        except PermissionError as exc:
            raise InstanceLockError("another Hedge supervisor already owns the lock") from exc
        try:
            handle.seek(0)
            self._lock_handle(handle)
            identity = InstanceIdentity(
                pid=os.getpid(),
                token=secrets.token_hex(16),
                acquired_at_utc=datetime.now(UTC).isoformat(),
            )
            encoded = json.dumps(
                asdict(identity),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            handle.seek(0)
            handle.write(encoded)
            handle.truncate()
            handle.flush()
            os.fsync(handle.fileno())
            self._handle = handle
            self.identity = identity
            return identity
        except Exception:
            handle.close()
            raise

    def _open_handle(self) -> BinaryIO:
        """Open a seeded binary lock file without reading the locked region."""

        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        descriptor = os.open(self.path, flags, 0o600)
        try:
            if os.fstat(descriptor).st_size == 0:
                os.lseek(descriptor, 0, os.SEEK_SET)
                os.write(descriptor, b"0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            return os.fdopen(descriptor, "r+b", buffering=0)
        except Exception:
            os.close(descriptor)
            raise

    def release(self) -> None:
        handle = self._handle
        if handle is None:
            return
        try:
            self._unlock_handle(handle)
        finally:
            handle.close()
            self._handle = None
            self.identity = None

    def __enter__(self) -> "SingleInstanceLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    @classmethod
    def _lock_handle(cls, handle: BinaryIO) -> None:
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, cls._LOCK_BYTE_COUNT)
            except OSError as exc:
                raise InstanceLockError("another Hedge supervisor already owns the lock") from exc
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise InstanceLockError("another Hedge supervisor already owns the lock") from exc

    @classmethod
    def _unlock_handle(cls, handle: BinaryIO) -> None:
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, cls._LOCK_BYTE_COUNT)
            except OSError:
                pass
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
