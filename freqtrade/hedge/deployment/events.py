"""Append-only structured supervisor events with conservative redaction."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Mapping

_SECRET_KEYS = {"api_key", "apikey", "secret", "token", "password", "signature"}


def _redact(value: Any, *, key: str = "") -> Any:
    normalized = key.lower().replace("-", "_")
    if any(part in normalized for part in _SECRET_KEYS):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(child_key): _redact(child, key=str(child_key)) for child_key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class JsonlEventWriter:
    def __init__(self, path: Path, *, max_bytes: int = 10 * 1024 * 1024) -> None:
        if max_bytes < 1024:
            raise ValueError("max_bytes must be at least 1024")
        self.path = path
        self.max_bytes = max_bytes
        self._lock = Lock()

    def write(self, event: str, **fields: Any) -> None:
        payload = {
            "at_utc": datetime.now(UTC).isoformat(),
            "event": str(event),
            **_redact(fields),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_if_needed(len(encoded) + 1)
            with self.path.open("a", encoding="ascii", newline="\n") as handle:
                handle.write(encoded + "\n")
                handle.flush()
                os.fsync(handle.fileno())

    def _rotate_if_needed(self, incoming_bytes: int) -> None:
        if not self.path.exists() or self.path.stat().st_size + incoming_bytes <= self.max_bytes:
            return
        rotated = self.path.with_suffix(self.path.suffix + ".1")
        rotated.unlink(missing_ok=True)
        os.replace(self.path, rotated)
