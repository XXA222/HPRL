from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from hashlib import sha256
from itertools import pairwise

from freqtrade.hedge.simulation.exchange import (
    BarEvent,
    FundingEvent,
    SignalEvent,
    SimulationInputEvent,
)

from .quality import validate_dataset_contract, validate_slice_bounds, validate_slice_nonempty
from .contracts import BacktestDataset
from .decimal_utils import canonical_json


def event_timestamp(event: SimulationInputEvent) -> datetime:
    return event.timestamp


def validate_event_stream(
    events: Iterable[SimulationInputEvent],
) -> tuple[SimulationInputEvent, ...]:
    materialized = tuple(events)
    if not materialized:
        raise ValueError("event stream cannot be empty")
    symbols = {event.symbol for event in materialized}
    if len(symbols) != 1:
        raise ValueError("a deterministic replay dataset must contain exactly one symbol")
    bars = [event for event in materialized if isinstance(event, BarEvent)]
    if not bars:
        raise ValueError("event stream must contain at least one bar")
    bar_times = [event.timestamp for event in bars]
    if any(right <= left for left, right in pairwise(bar_times)):
        raise ValueError("bar timestamps must be strictly increasing and unique")
    previous = materialized[0].timestamp
    for event in materialized[1:]:
        if event.timestamp < previous:
            raise ValueError("events must be globally chronological")
        previous = event.timestamp
    bar_keys = {(event.symbol, event.timestamp) for event in bars}
    for signal in (event for event in materialized if isinstance(event, SignalEvent)):
        if (signal.symbol, signal.timestamp) not in bar_keys:
            raise ValueError("every signal timestamp must have a matching bar")
    return materialized


def dataset_fingerprint(events: Iterable[SimulationInputEvent]) -> str:
    materialized = validate_event_stream(events)
    return sha256(canonical_json(materialized)).hexdigest()


def build_dataset(
    *,
    events: Iterable[SimulationInputEvent],
    dataset_id: str,
    timeframe: str,
    metadata: dict[str, str] | None = None,
) -> BacktestDataset:
    materialized = validate_event_stream(events)
    bars = tuple(event for event in materialized if isinstance(event, BarEvent))
    signals = sum(isinstance(event, SignalEvent) for event in materialized)
    fundings = sum(isinstance(event, FundingEvent) for event in materialized)
    dataset = BacktestDataset(
        events=materialized,
        dataset_id=dataset_id,
        symbol=bars[0].symbol,
        timeframe=timeframe,
        start=bars[0].timestamp,
        end=bars[-1].timestamp,
        fingerprint=sha256(canonical_json(materialized)).hexdigest(),
        bar_count=len(bars),
        signal_count=signals,
        funding_count=fundings,
        metadata=metadata or {},
    )
    validate_dataset_contract(dataset)
    return dataset


def slice_dataset(
    dataset: BacktestDataset,
    *,
    start: datetime,
    end: datetime,
    dataset_id: str | None = None,
) -> BacktestDataset:
    validate_slice_bounds(parent=dataset, start=start, end=end)
    events = tuple(event for event in dataset.events if start <= event.timestamp <= end)
    validate_slice_nonempty(events)
    return build_dataset(
        events=events,
        dataset_id=dataset_id or f"{dataset.dataset_id}:{start.isoformat()}:{end.isoformat()}",
        timeframe=dataset.timeframe,
        metadata={**dataset.metadata, "parent_fingerprint": dataset.fingerprint},
    )
