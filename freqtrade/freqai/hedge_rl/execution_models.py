"""Execution realism, latency, liquidity, and audit primitives (rounds 61-70)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from hashlib import sha256

import numpy as np


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


# Round 61 -------------------------------------------------------------------------------
def volatility_adjusted_slippage_bps(
    *,
    base_bps: float,
    realized_volatility: float,
    reference_volatility: float,
    sensitivity: float = 1.0,
    maximum_bps: float = 500.0,
) -> float:
    base = _finite_nonnegative("base_bps", base_bps)
    realized = _finite_nonnegative("realized_volatility", realized_volatility)
    reference = float(reference_volatility)
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("reference_volatility must be finite and positive")
    sensitivity = _finite_nonnegative("sensitivity", sensitivity)
    maximum = _finite_nonnegative("maximum_bps", maximum_bps)
    ratio = realized / reference
    adjusted = base * (1.0 + sensitivity * max(0.0, ratio - 1.0))
    return min(adjusted, maximum)


# Round 62 -------------------------------------------------------------------------------
def spread_impact_bps(
    *,
    bid: float,
    ask: float,
    participation_rate: float,
    exponent: float = 0.5,
) -> float:
    bid = float(bid)
    ask = float(ask)
    participation = float(participation_rate)
    if not all(math.isfinite(item) for item in (bid, ask, participation, exponent)):
        raise ValueError("spread inputs must be finite")
    if bid <= 0 or ask < bid or not 0 <= participation <= 1 or exponent <= 0:
        raise ValueError("invalid spread or participation inputs")
    mid = (bid + ask) / 2.0
    half_spread_bps = (ask - bid) / (2.0 * mid) * 10_000.0
    return half_spread_bps * (1.0 + participation**exponent)


# Round 63 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LiquidityCapResult:
    requested_quantity: float
    executable_quantity: float
    capped: bool
    participation_rate: float


def apply_liquidity_cap(
    *,
    requested_quantity: float,
    candle_volume: float,
    max_participation: float,
) -> LiquidityCapResult:
    requested = _finite_nonnegative("requested_quantity", requested_quantity)
    volume = _finite_nonnegative("candle_volume", candle_volume)
    participation = float(max_participation)
    if not math.isfinite(participation) or not 0 <= participation <= 1:
        raise ValueError("max_participation must be within [0, 1]")
    executable = min(requested, volume * participation)
    actual = executable / volume if volume > 0 else 0.0
    return LiquidityCapResult(requested, executable, executable + 1e-15 < requested, actual)


# Round 64 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LatencyModel:
    mean_milliseconds: float = 50.0
    jitter_milliseconds: float = 10.0
    minimum_milliseconds: float = 0.0

    def __post_init__(self) -> None:
        for name in ("mean_milliseconds", "jitter_milliseconds", "minimum_milliseconds"):
            _finite_nonnegative(name, getattr(self, name))

    def sample(self, *, seed: int) -> float:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        sample = np.random.default_rng(seed).normal(
            self.mean_milliseconds,
            self.jitter_milliseconds,
        )
        return max(self.minimum_milliseconds, float(sample))


# Round 65 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MakerTakerEstimate:
    maker_probability: float
    taker_probability: float


def estimate_maker_taker_probability(
    *,
    distance_to_mid_bps: float,
    urgency: float,
    queue_pressure: float,
) -> MakerTakerEstimate:
    distance = _finite_nonnegative("distance_to_mid_bps", distance_to_mid_bps)
    if not all(math.isfinite(float(item)) for item in (urgency, queue_pressure)):
        raise ValueError("urgency and queue_pressure must be finite")
    if not 0 <= urgency <= 1 or not -1 <= queue_pressure <= 1:
        raise ValueError("urgency or queue pressure outside supported range")
    logit = 2.5 - 0.35 * distance - 3.0 * urgency + 0.75 * queue_pressure
    maker = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, logit))))
    return MakerTakerEstimate(maker, 1.0 - maker)


# Round 66 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CancelLatencyPolicy:
    minimum_age_ms: float = 250.0
    replacement_cooldown_ms: float = 500.0

    def __post_init__(self) -> None:
        _finite_nonnegative("minimum_age_ms", self.minimum_age_ms)
        _finite_nonnegative("replacement_cooldown_ms", self.replacement_cooldown_ms)

    def can_cancel(self, *, order_age_ms: float) -> bool:
        return _finite_nonnegative("order_age_ms", order_age_ms) >= self.minimum_age_ms

    def can_replace(self, *, last_replace_age_ms: float) -> bool:
        return (
            _finite_nonnegative("last_replace_age_ms", last_replace_age_ms)
            >= self.replacement_cooldown_ms
        )


# Round 67 -------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RejectionModel:
    base_probability: float = 0.0
    stress_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not 0 <= self.base_probability <= 1:
            raise ValueError("base_probability must be within [0, 1]")
        if not math.isfinite(self.stress_multiplier) or self.stress_multiplier < 0:
            raise ValueError("stress_multiplier must be finite and non-negative")

    def probability(self, *, stress: float) -> float:
        if not math.isfinite(stress) or not 0 <= stress <= 1:
            raise ValueError("stress must be within [0, 1]")
        return min(1.0, self.base_probability * (1.0 + self.stress_multiplier * stress))

    def rejected(self, *, stress: float, seed: int) -> bool:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        return bool(np.random.default_rng(seed).random() < self.probability(stress=stress))


# Round 68 -------------------------------------------------------------------------------
def partial_fill_schedule(
    quantity: float,
    *,
    parts: int,
    front_load: float = 1.0,
) -> tuple[float, ...]:
    quantity = _finite_nonnegative("quantity", quantity)
    if parts < 1 or not math.isfinite(front_load) or front_load <= 0:
        raise ValueError("parts and front_load must be positive")
    weights = np.asarray([front_load ** (-index) for index in range(parts)], dtype=float)
    weights /= weights.sum()
    fills = quantity * weights
    fills[-1] += quantity - float(fills.sum())
    return tuple(float(item) for item in fills)


# Round 69 -------------------------------------------------------------------------------
def adverse_selection_cost(
    *,
    quantity: float,
    fill_price: float,
    post_fill_mark: float,
    is_buy: bool,
) -> float:
    quantity = _finite_nonnegative("quantity", quantity)
    fill = float(fill_price)
    mark = float(post_fill_mark)
    if not math.isfinite(fill) or not math.isfinite(mark) or min(fill, mark) <= 0:
        raise ValueError("fill and mark prices must be finite and positive")
    signed = (fill - mark) if is_buy else (mark - fill)
    return signed * quantity


# Round 70 -------------------------------------------------------------------------------
class ExecutionEventType(StrEnum):
    PREPARED = "PREPARED"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class ExecutionAuditEvent:
    sequence: int
    event_type: ExecutionEventType
    order_id: str
    timestamp: datetime
    payload: dict[str, object]
    previous_hash: str
    event_hash: str


class ExecutionAuditTrail:
    """Append-only hash chain suitable for deterministic simulation evidence."""

    def __init__(self) -> None:
        self._events: list[ExecutionAuditEvent] = []

    @staticmethod
    def _hash(
        sequence: int,
        event_type: ExecutionEventType,
        order_id: str,
        timestamp: datetime,
        payload: dict[str, object],
        previous_hash: str,
    ) -> str:
        encoded = json.dumps(
            {
                "sequence": sequence,
                "event_type": event_type.value,
                "order_id": order_id,
                "timestamp": timestamp.astimezone(UTC).isoformat(),
                "payload": payload,
                "previous_hash": previous_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return sha256(encoded.encode("utf-8")).hexdigest()

    def append(
        self,
        event_type: ExecutionEventType,
        *,
        order_id: str,
        payload: dict[str, object] | None = None,
        timestamp: datetime | None = None,
    ) -> ExecutionAuditEvent:
        if not order_id.strip():
            raise ValueError("order_id cannot be empty")
        when = datetime.now(UTC) if timestamp is None else timestamp
        if when.tzinfo is None:
            raise ValueError("audit timestamp must be timezone-aware")
        sequence = len(self._events)
        previous = self._events[-1].event_hash if self._events else "0" * 64
        values = dict(payload or {})
        event_hash = self._hash(sequence, event_type, order_id, when, values, previous)
        event = ExecutionAuditEvent(
            sequence,
            event_type,
            order_id,
            when,
            values,
            previous,
            event_hash,
        )
        self._events.append(event)
        return event

    def verify(self) -> bool:
        previous = "0" * 64
        for sequence, event in enumerate(self._events):
            expected = self._hash(
                sequence,
                event.event_type,
                event.order_id,
                event.timestamp,
                event.payload,
                previous,
            )
            if (
                event.sequence != sequence
                or event.previous_hash != previous
                or event.event_hash != expected
            ):
                return False
            previous = event.event_hash
        return True

    def events(self) -> tuple[ExecutionAuditEvent, ...]:
        return tuple(self._events)
