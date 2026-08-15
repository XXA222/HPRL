"""Durable planner-side state adapters for the production-equivalent main loop."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from threading import RLock
from typing import Mapping, Protocol

from freqtrade.hedge.planning.context import PositionSide, StrategyLegState, TrailingPhase
from freqtrade.hedge.symbols import canonicalize_symbol


class StrategyStateStorePort(Protocol):
    def load(self, side: PositionSide) -> StrategyLegState | None: ...

    def save(self, state: StrategyLegState, *, decision_id: str) -> None: ...


class InMemoryStrategyStateStore:
    def __init__(self) -> None:
        self._values: dict[PositionSide, StrategyLegState] = {}
        self._lock = RLock()

    def load(self, side: PositionSide) -> StrategyLegState | None:
        with self._lock:
            return self._values.get(PositionSide(side))

    def save(self, state: StrategyLegState, *, decision_id: str) -> None:
        if not str(decision_id).strip():
            raise ValueError("decision_id is required")
        with self._lock:
            self._values[state.side] = state


class JsonStrategyStateStore:
    """Atomic local state store used by simulated and development runtimes."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).expanduser()
        self._lock = RLock()

    def load(self, side: PositionSide) -> StrategyLegState | None:
        with self._lock:
            if not self._path.exists():
                return None
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or payload.get("schema_version") != 1:
                raise ValueError("unsupported strategy state document")
            row = payload.get(PositionSide(side).value)
            return None if row is None else decode_strategy_state(row)

    def save(self, state: StrategyLegState, *, decision_id: str) -> None:
        if not str(decision_id).strip():
            raise ValueError("decision_id is required")
        with self._lock:
            payload: dict[str, object] = {"schema_version": 1}
            if self._path.exists():
                current = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(current, dict) and current.get("schema_version") == 1:
                    payload.update(current)
            payload[state.side.value] = encode_strategy_state(state)
            payload["last_decision_id"] = str(decision_id)
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._path.with_name(f".{self._path.name}.{os.getpid()}.tmp")
            try:
                with temporary.open("w", encoding="ascii", newline="\n") as handle:
                    handle.write(encoded)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, self._path)
            finally:
                temporary.unlink(missing_ok=True)


class SqlStrategyStateStore:
    """SQL-backed state using the existing StrategySideState and outbox path."""

    def __init__(
        self,
        session_factory: object,
        *,
        exchange: str,
        account_id: str,
        symbol: str,
        strategy_name: str,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session_factory must be callable")
        self._sessions = session_factory
        from freqtrade.persistence.hedge_service import HedgePersistenceService

        self._service = HedgePersistenceService(session_factory)  # type: ignore[arg-type]
        self._exchange = str(exchange).strip().lower()
        self._account_id = str(account_id).strip()
        self._symbol = canonicalize_symbol(str(symbol).strip())
        self._strategy = str(strategy_name).strip()
        if not all((self._exchange, self._account_id, self._symbol, self._strategy)):
            raise ValueError("SQL strategy state identity fields are required")

    def _statement(self, side: PositionSide):
        from sqlalchemy import select
        from freqtrade.persistence.hedge_models import StrategySideState

        return select(StrategySideState).where(
            StrategySideState.exchange == self._exchange,
            StrategySideState.account_id == self._account_id,
            StrategySideState.symbol == self._symbol,
            StrategySideState.position_side == side.value,
            StrategySideState.strategy_name == self._strategy,
        )

    def load(self, side: PositionSide) -> StrategyLegState | None:
        normalized = PositionSide(side)
        with self._sessions() as session:  # type: ignore[operator]
            row = session.scalar(self._statement(normalized))
            if row is None:
                return None
            payload = json.loads(str(row.state_json))
            return decode_strategy_state(payload)

    def save(self, state: StrategyLegState, *, decision_id: str) -> None:
        self._service.update_strategy_side_state(
            exchange=self._exchange,
            account_id=self._account_id,
            symbol=self._symbol,
            position_side=state.side.value,
            strategy_name=self._strategy,
            state_name=state.trailing_phase.value,
            state=encode_strategy_state(state),
            last_decision_id=str(decision_id),
            cooldown_until=state.trailing_cooldown_until,
        )


def encode_strategy_state(state: StrategyLegState) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in asdict(state).items():
        if isinstance(value, Decimal):
            result[key] = str(value)
        elif isinstance(value, datetime):
            result[key] = value.astimezone(UTC).isoformat()
        elif isinstance(value, (PositionSide, TrailingPhase)):
            result[key] = value.value
        else:
            result[key] = value
    return result


def decode_strategy_state(value: object) -> StrategyLegState:
    if not isinstance(value, Mapping):
        raise TypeError("strategy state must be a mapping")
    data = dict(value)
    data["side"] = PositionSide(str(data["side"]))
    data["trailing_phase"] = TrailingPhase(str(data.get("trailing_phase", "IDLE")))
    for key in (
        "trailing_extreme",
        "trailing_trigger_price",
        "unstuck_daily_loss",
        "unstuck_weekly_loss",
    ):
        raw = data.get(key)
        if raw is not None:
            data[key] = Decimal(str(raw))
    for key in (
        "last_entry_at",
        "last_reduce_at",
        "trailing_started_at",
        "trailing_confirmed_at",
        "trailing_cooldown_until",
        "last_unstuck_at",
    ):
        raw = data.get(key)
        if raw:
            data[key] = datetime.fromisoformat(str(raw))
        else:
            data[key] = None
    return StrategyLegState(**data)
