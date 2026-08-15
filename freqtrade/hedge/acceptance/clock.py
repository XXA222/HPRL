from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True, slots=True)
class ClockSample:
    local_send_ms: int
    server_ms: int
    local_receive_ms: int

    @property
    def round_trip_ms(self) -> int:
        return max(0, self.local_receive_ms - self.local_send_ms)

    @property
    def offset_ms(self) -> float:
        midpoint = self.local_send_ms + self.round_trip_ms / 2
        return float(self.server_ms - midpoint)


@dataclass(frozen=True, slots=True)
class ClockAudit:
    sample_count: int
    median_offset_ms: float
    max_abs_offset_ms: float
    median_rtt_ms: float
    max_rtt_ms: float
    synchronized: bool


def evaluate_clock(
    samples: Iterable[ClockSample], *, max_abs_skew_ms: float, max_rtt_ms: float
) -> ClockAudit:
    values = tuple(samples)
    if not values:
        raise ValueError("at least one clock sample is required")
    offsets = [item.offset_ms for item in values]
    rtts = [float(item.round_trip_ms) for item in values]
    maximum_offset = max(abs(value) for value in offsets)
    maximum_rtt = max(rtts)
    return ClockAudit(
        sample_count=len(values),
        median_offset_ms=float(median(offsets)),
        max_abs_offset_ms=maximum_offset,
        median_rtt_ms=float(median(rtts)),
        max_rtt_ms=maximum_rtt,
        synchronized=maximum_offset <= max_abs_skew_ms and maximum_rtt <= max_rtt_ms,
    )
