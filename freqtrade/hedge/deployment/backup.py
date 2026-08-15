"""Deterministic SQLite backup with independent manifest hashing."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BackupEvidence:
    source: str
    backup: str
    created_at_utc: str
    source_size: int
    backup_size: int
    backup_sha256: str
    integrity_check: str


class SQLiteBackupManager:
    def __init__(self, backup_dir: Path) -> None:
        self.backup_dir = backup_dir

    def create(self, source: Path, *, label: str) -> BackupEvidence:
        if not source.is_file():
            raise FileNotFoundError(source)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        safe_label = "".join(ch for ch in label if ch.isalnum() or ch in "-_") or "backup"
        target = self.backup_dir / f"{safe_label}-{stamp}.sqlite"
        source_connection = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
        target_connection = sqlite3.connect(target)
        try:
            source_connection.backup(target_connection)
            target_connection.commit()
            integrity = str(target_connection.execute("PRAGMA integrity_check").fetchone()[0])
        finally:
            target_connection.close()
            source_connection.close()
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        evidence = BackupEvidence(
            source=str(source),
            backup=str(target),
            created_at_utc=datetime.now(UTC).isoformat(),
            source_size=source.stat().st_size,
            backup_size=target.stat().st_size,
            backup_sha256=digest,
            integrity_check=integrity,
        )
        manifest = target.with_suffix(".manifest.json")
        encoded = json.dumps(asdict(evidence), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        with manifest.open("w", encoding="ascii", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if integrity.lower() != "ok":
            raise RuntimeError(f"backup integrity_check failed: {integrity}")
        return evidence
