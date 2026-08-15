"""Fail-closed Producer/Consumer signal convergence for Hedge."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from json import dumps
from threading import RLock
from typing import Any, Mapping

from .freqai import HedgeSignalEnvelope
from .models import AdmissionCode, AdmissionDecision, NativeOrderIntent, utc_datetime


@dataclass(frozen=True, slots=True)
class ProducerIdentity:
    producer_id: str
    priority: int = 100
    allowed_pairs: tuple[str, ...] = ()
    expected_feature_schema: str | None = None

    def __post_init__(self) -> None:
        if not str(self.producer_id).strip():
            raise ValueError("producer_id is required")
        if isinstance(self.priority, bool):
            raise TypeError("producer priority must be an integer")
        object.__setattr__(self, "allowed_pairs", tuple(str(item).upper() for item in self.allowed_pairs))


@dataclass(frozen=True, slots=True)
class ProducerObservation:
    identity: ProducerIdentity
    signal: HedgeSignalEnvelope
    received_at: datetime
    message_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "received_at", utc_datetime(self.received_at))
        if not self.message_id:
            payload = f"{self.identity.producer_id}:{self.signal.evidence_hash()}:{self.received_at.isoformat()}"
            object.__setattr__(self, "message_id", sha256(payload.encode()).hexdigest())


@dataclass(frozen=True, slots=True)
class ProducerSelection:
    selected: ProducerObservation | None
    candidates: tuple[ProducerObservation, ...]
    rejected: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    conflict: bool = False


class HedgeProducerConsumerGate:
    """Validate identity, freshness, candle evidence and deterministic conflict policy."""

    def __init__(
        self,
        identities: Mapping[str, ProducerIdentity],
        *,
        maximum_age: timedelta = timedelta(minutes=2),
        conflict_tolerance: Decimal = Decimal("0.10"),
        fail_on_conflict: bool = True,
        seen_message_capacity: int = 4096,
    ) -> None:
        if maximum_age <= timedelta(0):
            raise ValueError("maximum_age must be positive")
        self.identities = dict(identities)
        self.maximum_age = maximum_age
        self.conflict_tolerance = Decimal(str(conflict_tolerance))
        if self.conflict_tolerance < 0:
            raise ValueError("conflict_tolerance cannot be negative")
        self.fail_on_conflict = bool(fail_on_conflict)
        if (
            not isinstance(seen_message_capacity, int)
            or isinstance(seen_message_capacity, bool)
            or seen_message_capacity < 1
        ):
            raise ValueError("seen_message_capacity must be a positive integer")
        self.seen_message_capacity = seen_message_capacity
        self._observations: dict[tuple[str, str], ProducerObservation] = {}
        # Message dedupe is a recent-observation cache, not a durable ledger.
        # Bound it so a long-running producer stream cannot grow process memory forever.
        self._seen_messages: OrderedDict[str, None] = OrderedDict()
        self._lock = RLock()
        self._last_selection: ProducerSelection | None = None

    def observe(
        self,
        signal: HedgeSignalEnvelope,
        *,
        received_at: datetime | None = None,
        message_id: str = "",
    ) -> bool:
        identity = self.identities.get(signal.producer_id)
        if identity is None:
            raise PermissionError(f"unknown Hedge producer: {signal.producer_id}")
        if identity.allowed_pairs and signal.pair not in identity.allowed_pairs:
            raise PermissionError("producer is not authorized for pair")
        if identity.expected_feature_schema and signal.feature_schema != identity.expected_feature_schema:
            raise ValueError("producer feature schema mismatch")
        observation = ProducerObservation(identity, signal, utc_datetime(received_at), message_id)
        with self._lock:
            if observation.message_id in self._seen_messages:
                self._seen_messages.move_to_end(observation.message_id)
                return False
            self._seen_messages[observation.message_id] = None
            while len(self._seen_messages) > self.seen_message_capacity:
                self._seen_messages.popitem(last=False)
            self._observations[(signal.pair, identity.producer_id)] = observation
        return True

    def select(
        self,
        pair: str,
        *,
        at: datetime | None = None,
        candle_fingerprint: str | None = None,
    ) -> ProducerSelection:
        now = utc_datetime(at)
        pair = str(pair).upper()
        rejected: dict[str, tuple[str, ...]] = {}
        accepted: list[ProducerObservation] = []
        with self._lock:
            rows = [item for (item_pair, _), item in self._observations.items() if item_pair == pair]
        for row in rows:
            reasons: list[str] = []
            if now - row.received_at > self.maximum_age:
                reasons.append("RECEIPT_STALE")
            if now - row.signal.timestamp > self.maximum_age:
                reasons.append("SIGNAL_STALE")
            if candle_fingerprint and row.signal.candle_fingerprint != candle_fingerprint:
                reasons.append("CANDLE_FINGERPRINT_MISMATCH")
            if reasons:
                rejected[row.identity.producer_id] = tuple(reasons)
            else:
                accepted.append(row)
        accepted.sort(key=lambda item: (item.identity.priority, -item.received_at.timestamp(), item.identity.producer_id))
        conflict = False
        if len(accepted) > 1:
            first = accepted[0].signal
            for row in accepted[1:]:
                if (
                    abs(first.long_score - row.signal.long_score) > self.conflict_tolerance
                    or abs(first.short_score - row.signal.short_score) > self.conflict_tolerance
                ):
                    conflict = True
                    break
        selected = None if conflict and self.fail_on_conflict else (accepted[0] if accepted else None)
        result = ProducerSelection(selected, tuple(accepted), rejected, conflict)
        with self._lock:
            self._last_selection = result
        return result

    def admit(self, intent: NativeOrderIntent) -> AdmissionDecision:
        if intent.reduce_only:
            return AdmissionDecision.allow(reason="PRODUCER_GATE_REDUCE_ONLY", reduce_only_exempt=True)
        selection = self.select(intent.pair)
        if selection.selected is None:
            reason = "PRODUCER_CONFLICT" if selection.conflict else "PRODUCER_SIGNAL_MISSING_OR_STALE"
            return AdmissionDecision.block(AdmissionCode.PRODUCER_STALE, reason)
        return AdmissionDecision.allow(reason=f"PRODUCER_SELECTED:{selection.selected.identity.producer_id}")

    def status(self) -> dict[str, Any]:
        with self._lock:
            selection = self._last_selection
            seen = len(self._seen_messages)
        return {
            "known_producers": sorted(self.identities),
            "seen_messages": seen,
            "last_selection": None
            if selection is None
            else {
                "selected": None if selection.selected is None else selection.selected.identity.producer_id,
                "candidates": [item.identity.producer_id for item in selection.candidates],
                "rejected": {key: list(value) for key, value in selection.rejected.items()},
                "conflict": selection.conflict,
            },
        }


def producer_message_payload(signal: HedgeSignalEnvelope) -> dict[str, Any]:
    payload = {
        "schema": "hedge-producer-signal-v1",
        "pair": signal.pair,
        "timestamp": signal.timestamp.isoformat(),
        "long_score": str(signal.long_score),
        "short_score": str(signal.short_score),
        "target_net_ratio": None if signal.target_net_ratio is None else str(signal.target_net_ratio),
        "target_gross_ratio": None if signal.target_gross_ratio is None else str(signal.target_gross_ratio),
        "confidence": str(signal.confidence),
        "risk_scale": str(signal.risk_scale),
        "model_version": signal.model_version,
        "feature_schema": signal.feature_schema,
        "candle_fingerprint": signal.candle_fingerprint,
        "producer_id": signal.producer_id,
        "evidence_hash": signal.evidence_hash(),
    }
    canonical = dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["message_sha256"] = sha256(canonical.encode()).hexdigest()
    return payload
