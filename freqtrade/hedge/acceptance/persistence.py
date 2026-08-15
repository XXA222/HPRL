from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from freqtrade.hedge.acceptance.models import FactPlane, _json_ready


class RuntimeAcceptanceStore:
    """Small SQLite evidence store with durable exactly-once effect keys."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        self._initialize()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS acceptance_snapshots (
                plane TEXT NOT NULL,
                account_id TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                payload TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                PRIMARY KEY (plane, account_id)
            );
            CREATE TABLE IF NOT EXISTS acceptance_effects (
                effect_key TEXT PRIMARY KEY,
                effect_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                applied_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS acceptance_unknown_orders (
                order_key TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                detail TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS acceptance_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                state_hash TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS acceptance_evidence (
                round_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        self._connection.commit()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield self._connection
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def close(self) -> None:
        self._connection.close()

    def save_plane(self, plane_name: str, plane: FactPlane) -> None:
        payload = json.dumps(_json_ready(asdict(plane)), ensure_ascii=False, sort_keys=True)
        self._connection.execute(
            """
            INSERT INTO acceptance_snapshots(plane, account_id, fingerprint, payload, observed_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(plane, account_id) DO UPDATE SET
                fingerprint=excluded.fingerprint,
                payload=excluded.payload,
                observed_at=excluded.observed_at
            """,
            (
                plane_name.upper(),
                plane.account_id,
                plane.fingerprint(),
                payload,
                plane.observed_at.isoformat(),
            ),
        )
        self._connection.commit()

    def plane_fingerprint(self, plane_name: str, account_id: str) -> str | None:
        row = self._connection.execute(
            "SELECT fingerprint FROM acceptance_snapshots WHERE plane=? AND account_id=?",
            (plane_name.upper(), account_id),
        ).fetchone()
        return None if row is None else str(row[0])

    def apply_effect_once(
        self, effect_key: str, effect_type: str, payload: Mapping[str, Any]
    ) -> bool:
        rendered = json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True)
        try:
            with self.transaction() as connection:
                connection.execute(
                    "INSERT INTO acceptance_effects"
                    "(effect_key,effect_type,payload,applied_at) VALUES(?,?,?,?)",
                    (effect_key, effect_type, rendered, datetime.now(UTC).isoformat()),
                )
        except sqlite3.IntegrityError:
            return False
        return True

    def effect_count(self, effect_type: str | None = None) -> int:
        if effect_type is None:
            row = self._connection.execute("SELECT COUNT(*) FROM acceptance_effects").fetchone()
        else:
            row = self._connection.execute(
                "SELECT COUNT(*) FROM acceptance_effects WHERE effect_type=?", (effect_type,)
            ).fetchone()
        return int(row[0])

    def record_unknown_order(self, order_key: str, detail: str) -> None:
        self._connection.execute(
            """
            INSERT INTO acceptance_unknown_orders(order_key,state,detail,updated_at)
            VALUES(?, 'UNKNOWN', ?, ?)
            ON CONFLICT(order_key) DO UPDATE SET state='UNKNOWN', detail=excluded.detail,
                updated_at=excluded.updated_at
            """,
            (order_key, detail, datetime.now(UTC).isoformat()),
        )
        self._connection.commit()

    def resolve_unknown_order(self, order_key: str, detail: str) -> None:
        self._connection.execute(
            "UPDATE acceptance_unknown_orders "
            "SET state='RESOLVED', detail=?, updated_at=? WHERE order_key=?",
            (detail, datetime.now(UTC).isoformat(), order_key),
        )
        self._connection.commit()

    def unknown_order_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) FROM acceptance_unknown_orders WHERE state='UNKNOWN'"
        ).fetchone()
        return int(row[0])

    def save_checkpoint(
        self, checkpoint_id: str, state_hash: str, payload: Mapping[str, Any]
    ) -> None:
        rendered = json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True)
        self._connection.execute(
            """
            INSERT OR REPLACE INTO acceptance_checkpoints(checkpoint_id,state_hash,payload,created_at)
            VALUES(?,?,?,?)
            """,
            (checkpoint_id, state_hash, rendered, datetime.now(UTC).isoformat()),
        )
        self._connection.commit()

    def checkpoint_hash(self, checkpoint_id: str) -> str | None:
        row = self._connection.execute(
            "SELECT state_hash FROM acceptance_checkpoints WHERE checkpoint_id=?", (checkpoint_id,)
        ).fetchone()
        return None if row is None else str(row[0])

    def checkpoint_hash_fresh_connection(self, checkpoint_id: str) -> str | None:
        connection = sqlite3.connect(self.path)
        try:
            row = connection.execute(
                "SELECT state_hash FROM acceptance_checkpoints WHERE checkpoint_id=?",
                (checkpoint_id,),
            ).fetchone()
            return None if row is None else str(row[0])
        finally:
            connection.close()

    def save_evidence(self, round_id: str, payload: Mapping[str, Any]) -> None:
        rendered = json.dumps(_json_ready(payload), ensure_ascii=False, sort_keys=True)
        self._connection.execute(
            "INSERT OR REPLACE INTO acceptance_evidence(round_id,payload,created_at) VALUES(?,?,?)",
            (round_id, rendered, datetime.now(UTC).isoformat()),
        )
        self._connection.commit()
