"""Deterministic candle cursor and data-integrity helpers for Hedge runtimes."""

from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256

from freqtrade.exchange import timeframe_to_seconds
from freqtrade.hedge.simulation.exchange import BarEvent


def bar_fingerprint(bar: BarEvent) -> str:
    """Return a stable fingerprint for one canonical OHLCV bar.

    The fingerprint intentionally includes the event identity and exact decimal
    wire values. It is used to detect exchange/data-provider revisions at an
    already committed durable cursor.
    """

    payload = "|".join(
        (
            bar.symbol,
            bar.timestamp.isoformat(),
            str(bar.open),
            str(bar.high),
            str(bar.low),
            str(bar.close),
            "" if bar.volume is None else str(bar.volume),
        )
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def expected_next_close(previous_close: datetime, timeframe: str) -> datetime:
    return previous_close + timedelta(seconds=timeframe_to_seconds(timeframe))


def missing_candle_count(
    previous_close: datetime,
    current_close: datetime,
    timeframe: str,
) -> int:
    """Return missing full candle slots between two close timestamps.

    Raises when timestamps are not aligned to an integer number of timeframe
    slots. This avoids silently accepting malformed or mixed-timeframe data.
    """

    seconds = timeframe_to_seconds(timeframe)
    delta = (current_close - previous_close).total_seconds()
    if delta <= 0:
        raise ValueError("candle timestamps must move forward")
    slots, remainder = divmod(int(delta), seconds)
    if remainder != 0 or delta != int(delta):
        raise ValueError("candle timestamps are not aligned to the configured timeframe")
    return max(slots - 1, 0)
