"""Crash-safe persistence for the integrated paper hedge application.

The SQL checkpoint includes confirmed account/planner state and recoverable active
Paper orders.  Rehydration never resubmits an order: it restores the local lifecycle,
idempotency result and fake-exchange snapshot before the next matcher cycle.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Protocol
from uuid import uuid4

from sqlalchemy import func, select

from freqtrade.persistence.hedge_models import (
    EventOutbox,
    PaperRuntimeCheckpointRow,
    StrategySideState,
)


PAPER_STATE_SCHEMA_VERSION = 2


class PaperStateStore(Protocol):
    def load(self) -> Mapping[str, object] | None: ...

    def save(self, state: Mapping[str, object]) -> None: ...


class NullPaperStateStore:
    """Explicit ephemeral store used only when ``hedge.paper.ephemeral=true``."""

    def load(self) -> Mapping[str, object] | None:
        return None

    def save(self, state: Mapping[str, object]) -> None:
        del state


class JsonPaperStateStore:
    """Atomic UTF-8 JSON state store for local Paper/Dry-run continuity."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        if not str(self._path).strip():
            raise ValueError("paper state path must not be empty")
        self._lock = RLock()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> Mapping[str, object] | None:
        with self._lock:
            if not self._path.exists():
                return None
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise ValueError("paper state document must be a JSON object")
            version = payload.get("schema_version")
            if version not in {1, PAPER_STATE_SCHEMA_VERSION}:
                raise ValueError(
                    "unsupported paper state schema version: "
                    f"{version!r}; expected 1 or {PAPER_STATE_SCHEMA_VERSION}"
                )
            return payload

    def save(self, state: Mapping[str, object]) -> None:
        payload = dict(state)
        payload["schema_version"] = PAPER_STATE_SCHEMA_VERSION
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_name(f".{self._path.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                    handle.write(encoded)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self._path)
            finally:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


class SqlPaperStateStore:
    """Transactional SQL checkpoint with an Outbox event on every revision.

    The dedicated checkpoint table is authoritative. A one-time legacy read from
    ``StrategySideState`` keeps v1.4 JSON/SQL checkpoints recoverable and is
    migrated automatically on the next save.
    """

    _LEGACY_STRATEGY_NAME = "__paper_runtime_v2__"

    def __init__(
        self,
        session_factory: object,
        *,
        exchange: str,
        account_id: str,
        symbol: str,
        source: str = "PAPER",
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._session_factory = session_factory
        self._exchange = str(exchange).strip().lower()
        self._account_id = str(account_id).strip()
        self._symbol = str(symbol).strip()
        self._source = str(source).strip().upper()
        if self._source not in {"PAPER", "SHADOW"}:
            raise ValueError("SQL paper state source must be PAPER or SHADOW")
        if not all((self._exchange, self._account_id, self._symbol)):
            raise ValueError("SQL paper state identity fields must be non-empty")

    def _statement(self):
        return select(PaperRuntimeCheckpointRow).where(
            PaperRuntimeCheckpointRow.exchange == self._exchange,
            PaperRuntimeCheckpointRow.account_id == self._account_id,
            PaperRuntimeCheckpointRow.symbol == self._symbol,
            PaperRuntimeCheckpointRow.source == self._source,
        )

    def _legacy_statement(self):
        return select(StrategySideState).where(
            StrategySideState.exchange == self._exchange,
            StrategySideState.account_id == self._account_id,
            StrategySideState.symbol == self._symbol,
            StrategySideState.position_side == "LONG",
            StrategySideState.strategy_name == self._LEGACY_STRATEGY_NAME,
        )

    @staticmethod
    def _decode(encoded: str) -> Mapping[str, object]:
        payload = json.loads(encoded)
        if not isinstance(payload, dict):
            raise ValueError("SQL paper state must decode to a JSON object")
        version = payload.get("schema_version")
        if version not in {1, PAPER_STATE_SCHEMA_VERSION}:
            raise ValueError(
                f"unsupported SQL paper state schema version: {version!r}; "
                f"expected 1 or {PAPER_STATE_SCHEMA_VERSION}"
            )
        return payload

    def load(self) -> Mapping[str, object] | None:
        with self._session_factory() as session:  # type: ignore[operator]
            row = session.scalar(self._statement())
            if row is not None:
                return self._decode(row.state_json)
            legacy = session.scalar(self._legacy_statement())
            if legacy is None:
                return None
            return self._decode(legacy.state_json)

    def save(self, state: Mapping[str, object]) -> None:
        payload = dict(state)
        payload["schema_version"] = PAPER_STATE_SCHEMA_VERSION
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        aggregate_id = f"{self._source}:{self._account_id}:{self._symbol}"
        with self._session_factory.begin() as session:  # type: ignore[operator]
            row = session.scalar(self._statement())
            if row is None:
                previous_sequence = int(
                    session.scalar(
                        select(func.coalesce(func.max(EventOutbox.aggregate_sequence), 0)).where(
                            EventOutbox.aggregate_type == "PAPER_RUNTIME_STATE",
                            EventOutbox.aggregate_id == aggregate_id,
                        )
                    )
                    or 0
                )
                revision = previous_sequence + 1
                row = PaperRuntimeCheckpointRow(
                    exchange=self._exchange,
                    account_id=self._account_id,
                    symbol=self._symbol,
                    source=self._source,
                    schema_version=PAPER_STATE_SCHEMA_VERSION,
                    revision=revision,
                    state_json=encoded,
                )
                session.add(row)
            else:
                revision = int(row.revision or 0) + 1
                row.state_json = encoded
                row.schema_version = PAPER_STATE_SCHEMA_VERSION
                row.revision = revision

            event_id = str(uuid4())
            session.add(
                EventOutbox(
                    event_id=event_id,
                    aggregate_type="PAPER_RUNTIME_STATE",
                    aggregate_id=aggregate_id,
                    event_type="PAPER_RUNTIME_STATE_SAVED",
                    aggregate_sequence=revision,
                    correlation_id=event_id,
                    payload_json=json.dumps(
                        {
                            "source": self._source,
                            "account_id": self._account_id,
                            "symbol": self._symbol,
                            "revision": revision,
                            "schema_version": PAPER_STATE_SCHEMA_VERSION,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    headers_json=json.dumps(
                        {"durability": "SQL", "source": self._source},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )
