from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from freqtrade.hedge.acceptance.persistence import RuntimeAcceptanceStore


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    account_id: str
    event_type: str
    event_time_ms: int
    transaction_time_ms: int
    symbol: str
    position_side: str
    order_id: str
    trade_id: str
    payload: Mapping[str, Any]

    @property
    def entity_key(self) -> str:
        parts = (self.account_id, self.event_type, self.symbol, self.position_side, self.order_id)
        return ":".join(parts)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        source = "|".join(
            (
                self.account_id,
                self.event_type,
                str(self.event_time_ms),
                str(self.transaction_time_ms),
                self.symbol,
                self.position_side,
                self.order_id,
                self.trade_id,
                payload,
            )
        )
        return sha256(source.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SequenceDecision:
    apply: bool
    duplicate: bool
    out_of_order: bool
    gap: bool
    reason: str


class EventSequenceTracker:
    def __init__(self) -> None:
        self._last_transaction_time: dict[str, int] = {}
        self._last_event_time: dict[str, int] = {}
        self._fingerprints: set[str] = set()

    def inspect(self, event: EventEnvelope) -> SequenceDecision:
        if event.fingerprint in self._fingerprints:
            return SequenceDecision(False, True, False, False, "DUPLICATE")
        last_transaction = self._last_transaction_time.get(event.entity_key)
        last_event = self._last_event_time.get(event.entity_key)
        if last_transaction is not None and event.transaction_time_ms < last_transaction:
            return SequenceDecision(False, False, True, False, "TRANSACTION_TIME_REGRESSION")
        if last_event is not None and event.event_time_ms < last_event:
            return SequenceDecision(False, False, True, False, "EVENT_TIME_REGRESSION")
        return SequenceDecision(True, False, False, False, "APPLY")

    def commit(self, event: EventEnvelope) -> None:
        self._fingerprints.add(event.fingerprint)
        self._last_transaction_time[event.entity_key] = event.transaction_time_ms
        self._last_event_time[event.entity_key] = event.event_time_ms


class ExactlyOnceEffectJournal:
    def __init__(self, store: RuntimeAcceptanceStore) -> None:
        self.store = store

    def apply_fill(self, event: EventEnvelope) -> bool:
        trade_id = event.trade_id.strip()
        if not trade_id:
            raise ValueError("fill effect requires trade_id")
        effect_key = f"FILL:{event.account_id}:{event.symbol}:{trade_id}"
        return self.store.apply_effect_once(effect_key, "FILL", event.payload)

    def apply_funding(self, event: EventEnvelope) -> bool:
        transaction_id = str(event.payload.get("tranId") or event.trade_id or "").strip()
        if not transaction_id:
            transaction_id = event.fingerprint
        effect_key = f"FUNDING:{event.account_id}:{transaction_id}"
        return self.store.apply_effect_once(effect_key, "FUNDING", event.payload)
