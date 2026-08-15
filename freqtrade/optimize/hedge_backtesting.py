"""Standard Freqtrade data/strategy adapter for deterministic Hedge backtesting.

The low-level :class:`HedgeBacktesting` facade remains useful for tests and
programmatic event replays.  ``run_freqtrade_hedge_backtest`` adds the missing
production-facing path: normal Freqtrade configuration, downloaded OHLCV,
strategy analysis, futures funding data, the shared Hedge planner/matcher, and a
reproducible JSON result artifact.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import (
    asdict,
    dataclass,
    fields,
    is_dataclass,
    replace,
)
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Any

from freqtrade.exceptions import OperationalException
from freqtrade.hedge.paper_config import PaperFundingSource, PaperSimulationConfig
from freqtrade.hedge.planning.context import PlannerConfig, StrategyPlanningPort
from freqtrade.hedge.simulation.exchange import (
    BarEvent,
    FundingEvent,
    MarketRules,
    SignalEvent,
    SimulationInputEvent,
    SimulationResult,
)
from freqtrade.hedge.simulation.matcher import MatchConfig
from freqtrade.hedge.simulation.replay import EventReplayEngine


DEFAULT_FUNDING_RATE_MULTIPLIER = Decimal(1)
DEFAULT_LEVERAGE = Decimal(3)
DEFAULT_FEE_RATE = Decimal("0.0004")
DEFAULT_LONG_SIGNAL = Decimal(1)
DEFAULT_SHORT_SIGNAL = Decimal(1)
DEFAULT_STREAM_CHUNK_BARS = 2048
_STREAM_FINGERPRINT_VERSION = b"hedge-event-stream-v2\0"


@dataclass(frozen=True, slots=True)
class HedgeBacktestDataset:
    events: tuple[SimulationInputEvent, ...]
    pair: str
    timeframe: str
    start: datetime
    end: datetime
    bar_count: int
    signal_count: int
    funding_count: int
    missing_candle_count: int = 0
    data_fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class HedgeBacktestRun:
    result: SimulationResult
    dataset: HedgeBacktestDataset
    export_path: Path
    strategy: str
    market_rule_source: str
    market_rule_version: str
    artifact_sha256: str = ""
    result_fingerprint: str = ""
    native_artifact: object | None = None
    memory_telemetry: tuple[Mapping[str, Any], ...] = ()


class HedgeBacktestEventChunks:
    """Single-use, bounded-memory event producer for analyzed OHLCV.

    The producer keeps the same Signal -> Funding -> Bar priority contract as
    ``EventReplayEngine`` but never materializes the full multi-year event list.
    It also computes the dataset fingerprint incrementally and validates candle
    chronology/missing slots while rows are consumed.
    """

    def __init__(
        self,
        *,
        pair: str,
        timeframe: str,
        frame: Any,
        funding_frame: Any | None = None,
        strategy_version: object = None,
        require_funding_data: bool = False,
        max_missing_candles: int = 0,
        funding_rate_multiplier: Decimal = DEFAULT_FUNDING_RATE_MULTIPLIER,
        chunk_bars: int = DEFAULT_STREAM_CHUNK_BARS,
    ) -> None:
        if chunk_bars < 1:
            raise ValueError("chunk_bars must be positive")
        if frame is None or frame.empty:
            raise OperationalException("Hedge backtest analyzed dataframe is empty")
        required = {"date", "open", "high", "low", "close"}
        columns = set(frame.columns)
        missing = sorted(required - columns)
        if missing:
            raise OperationalException(
                "Hedge backtest analyzed dataframe is missing: " + ", ".join(missing)
            )
        signal_columns = {
            "hedge_long_score",
            "hedge_short_score",
            "hedge_target_net",
            "enter_long",
            "enter_short",
        }
        if not signal_columns.intersection(columns):
            raise OperationalException(
                "Hedge backtest strategy produced no hedge_* or enter_long/enter_short columns"
            )

        multiplier = _decimal(
            funding_rate_multiplier, field="funding_rate_multiplier"
        )
        if multiplier < 0:
            raise OperationalException("funding_rate_multiplier cannot be negative")

        funding_missing = funding_frame is None or funding_frame.empty
        if require_funding_data and funding_missing:
            raise OperationalException(
                "Hedge futures backtest requires downloaded funding/mark data; "
                "run download-data with the futures configuration first"
            )
        if not funding_missing:
            funding_columns = set(funding_frame.columns)
            required_funding = {"date", "open_fund", "open_mark"}
            if not required_funding.issubset(funding_columns):
                raise OperationalException(
                    "Futures funding data must contain date/open_fund/open_mark"
                )

        from freqtrade.exchange import timeframe_to_seconds

        seconds = timeframe_to_seconds(timeframe)
        first_open = _aware(frame["date"].iloc[0], field="date")
        last_open = _aware(frame["date"].iloc[-1], field="date")
        from freqtrade.hedge.strategies.contract import HEDGE_SIGNAL_COLUMNS

        preferred = (
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            *HEDGE_SIGNAL_COLUMNS,
            "enter_long",
            "enter_short",
        )
        names = tuple(name for name in preferred if name in columns)
        # Copy only the narrow canonical replay surface.  This intentionally
        # detaches the multi-year stream from the often much wider analyzed
        # strategy dataframe so the caller can release indicator/intermediate
        # columns before event replay begins.  ExtensionArray.copy() preserves
        # compact timezone-aware datetime storage instead of creating Timestamp
        # object arrays.
        row_arrays = tuple(frame[name].array.copy() for name in names)
        funding_arrays = (
            None
            if funding_missing
            else tuple(
                funding_frame[name].array.copy()
                for name in ("date", "open_fund", "open_mark")
            )
        )

        self.pair = pair
        self.timeframe = timeframe
        self.strategy_version = strategy_version
        self._names = names
        self._row_arrays = row_arrays
        self._funding_arrays = funding_arrays
        self.require_funding_data = require_funding_data
        self.max_missing_candles = max_missing_candles
        self.funding_rate_multiplier = multiplier
        self.chunk_bars = chunk_bars
        self.start = first_open + timedelta(seconds=seconds)
        self.end = last_open + timedelta(seconds=seconds)
        self.bar_count = 0
        self.funding_count = 0
        self.missing_candle_count = 0
        self.max_chunk_input_events = 0
        self._columns = columns
        self._seconds = seconds
        self._consumed = False
        self._complete = False
        self._hasher = sha256(_STREAM_FINGERPRINT_VERSION)
        self.data_fingerprint = ""

    @staticmethod
    def _row_values(arrays: tuple[Any, ...]):
        return zip(*arrays, strict=True)

    def _funding_events(self):
        if self._funding_arrays is None:
            return
        previous: datetime | None = None
        for values in self._row_values(self._funding_arrays):
            timestamp = _aware(values[0], field="funding.date")
            if previous is not None and timestamp <= previous:
                raise OperationalException(
                    "Hedge funding dataframe must be strictly chronological and unique"
                )
            previous = timestamp
            if timestamp < self.start:
                continue
            if timestamp > self.end:
                break
            rate = (
                _decimal(values[1], field="funding.open_fund")
                * self.funding_rate_multiplier
            )
            mark = _decimal(values[2], field="funding.open_mark")
            if mark <= 0:
                raise OperationalException("funding.open_mark must be positive")
            yield FundingEvent(
                timestamp=timestamp,
                symbol=self.pair,
                rate=rate,
                mark_price=mark,
            )

    def _hash_event(self, event: SimulationInputEvent) -> None:
        # Event payloads are intentionally tiny.  A single compact bytes object is
        # cheaper here than retaining encoder chunk lists; large result payloads use
        # the streaming encoder below.
        payload = json.dumps(
            _json_value(event),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._hasher.update(len(payload).to_bytes(8, "big"))
        self._hasher.update(payload)

    def release_source(self) -> None:
        """Release detached historical arrays once the single-use stream is consumed."""
        self._row_arrays = ()
        self._funding_arrays = None

    def __iter__(self):
        if self._consumed:
            raise RuntimeError("HedgeBacktestEventChunks is single-use")
        self._consumed = True
        try:
            yield from self._iter_chunks()
        finally:
            # Even an aborted/failed replay must not keep the multi-year narrow
            # arrays alive through exception/report handling.
            self.release_source()

    def _iter_chunks(self):  # noqa: C901
        from freqtrade.hedge.integration.candle_cursor import missing_candle_count
        from freqtrade.hedge.integration.signal_provider import signal_from_analyzed_row

        names = self._names
        signal_columns = set(names)
        funding_iter = iter(self._funding_events() or ())
        next_funding = next(funding_iter, None)
        previous_open: datetime | None = None
        chunk: list[SimulationInputEvent] = []
        bars_in_chunk = 0
        row = dict.fromkeys(names)

        def append_event(event: SimulationInputEvent) -> None:
            chunk.append(event)
            self._hash_event(event)

        for values in self._row_values(self._row_arrays):
            for name, value in zip(names, values, strict=True):
                row[name] = value
            open_time = _aware(row["date"], field="date")
            if previous_open is not None:
                if open_time <= previous_open:
                    raise OperationalException(
                        "Hedge backtest candles must be strictly chronological and unique"
                    )
                self.missing_candle_count += missing_candle_count(
                    previous_open, open_time, self.timeframe
                )
                if self.missing_candle_count > self.max_missing_candles:
                    raise OperationalException(
                        f"Hedge backtest has {self.missing_candle_count} missing candle slots; "
                        f"limit={self.max_missing_candles}"
                    )
            previous_open = open_time

            signal = signal_from_analyzed_row(
                pair=self.pair,
                timeframe=self.timeframe,
                row=row,
                columns=signal_columns,
                feature_timestamp=open_time,
                strategy_version=self.strategy_version,
            )
            candle = signal.candle
            if candle is None:
                raise OperationalException("Hedge backtest could not build an OHLCV candle")
            signal_event = SignalEvent(
                timestamp=signal.candle_close_time,
                symbol=self.pair,
                long_signal=signal.long_score,
                short_signal=signal.short_score,
                target_net=signal.target_net,
                model_version=signal.model_version,
                reason=signal.strategy_reason or signal.reason,
                target_net_ratio=signal.target_net_ratio,
                confidence=signal.confidence,
                risk_scale=signal.risk_scale,
                long_exposure_scale=signal.long_exposure_scale,
                short_exposure_scale=signal.short_exposure_scale,
                allow_new_risk=signal.allow_new_risk,
                regime=signal.regime,
            )
            bar = candle.to_bar_event()

            while next_funding is not None and next_funding.timestamp < bar.timestamp:
                append_event(next_funding)
                self.funding_count += 1
                next_funding = next(funding_iter, None)

            append_event(signal_event)
            while next_funding is not None and next_funding.timestamp == bar.timestamp:
                append_event(next_funding)
                self.funding_count += 1
                next_funding = next(funding_iter, None)
            append_event(bar)

            self.bar_count += 1
            bars_in_chunk += 1
            if bars_in_chunk >= self.chunk_bars:
                self.max_chunk_input_events = max(
                    self.max_chunk_input_events, len(chunk)
                )
                yield tuple(chunk)
                chunk.clear()
                bars_in_chunk = 0

        if chunk:
            self.max_chunk_input_events = max(self.max_chunk_input_events, len(chunk))
            yield tuple(chunk)

        self.data_fingerprint = self._hasher.hexdigest()
        self._complete = True

    def dataset(self) -> HedgeBacktestDataset:
        if not self._complete:
            raise RuntimeError("event stream must be fully consumed before dataset()")
        return HedgeBacktestDataset(
            events=(),
            pair=self.pair,
            timeframe=self.timeframe,
            start=self.start,
            end=self.end,
            bar_count=self.bar_count,
            signal_count=self.bar_count,
            funding_count=self.funding_count,
            missing_candle_count=self.missing_candle_count,
            data_fingerprint=self.data_fingerprint,
        )


def _aware(value: object, *, field: str) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise OperationalException(f"{field} must contain datetime values")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _decimal(value: object, *, field: str) -> Decimal:
    if isinstance(value, bool):
        raise OperationalException(f"{field} must be numeric")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise OperationalException(f"{field} must be a valid decimal") from exc
    if not result.is_finite():
        raise OperationalException(f"{field} must be finite")
    return result


def _strict_dates(frame: Any, *, field: str = "date") -> tuple[datetime, ...]:
    if field not in frame.columns:
        raise OperationalException(f"Hedge backtest dataframe is missing {field!r}")
    values = tuple(_aware(value, field=field) for value in frame[field].tolist())
    if not values:
        raise OperationalException("Hedge backtest has no analyzed candles")
    if any(right <= left for left, right in pairwise(values)):
        raise OperationalException(
            "Hedge backtest candles must be strictly chronological and unique"
        )
    return values


def events_from_analyzed_dataframe(  # noqa: C901
    *,
    pair: str,
    timeframe: str,
    frame: Any,
    funding_frame: Any | None = None,
    strategy_version: object = None,
    require_funding_data: bool = False,
    max_missing_candles: int = 0,
    funding_rate_multiplier: Decimal = DEFAULT_FUNDING_RATE_MULTIPLIER,
) -> HedgeBacktestDataset:
    """Build the shared event stream from one analyzed Freqtrade dataframe.

    A signal and bar share the candle close timestamp.  The engine first matches
    orders accepted on earlier bars and only then creates orders from the current
    signal, so the current OHLC path can never fill an order derived from its own
    close or indicators.
    """

    from freqtrade.hedge.integration.candle_cursor import missing_candle_count
    from freqtrade.hedge.integration.signal_provider import signal_from_analyzed_row

    funding_rate_multiplier = _decimal(
        funding_rate_multiplier, field="funding_rate_multiplier"
    )
    if funding_rate_multiplier < 0:
        raise OperationalException("funding_rate_multiplier cannot be negative")

    if frame is None or frame.empty:
        raise OperationalException("Hedge backtest analyzed dataframe is empty")
    required = {"date", "open", "high", "low", "close"}
    columns = set(frame.columns)
    missing = sorted(required - columns)
    if missing:
        raise OperationalException(
            "Hedge backtest analyzed dataframe is missing: " + ", ".join(missing)
        )
    signal_columns = {
        "hedge_long_score",
        "hedge_short_score",
        "hedge_target_net",
        "enter_long",
        "enter_short",
    }
    if not signal_columns.intersection(columns):
        raise OperationalException(
            "Hedge backtest strategy produced no hedge_* or enter_long/enter_short columns"
        )
    dates = _strict_dates(frame)
    missing_slots = sum(
        missing_candle_count(left, right, timeframe)
        for left, right in pairwise(dates)
    )
    if missing_slots > max_missing_candles:
        raise OperationalException(
            f"Hedge backtest has {missing_slots} missing candle slots; "
            f"limit={max_missing_candles}"
        )

    events: list[SimulationInputEvent] = []
    bars: list[BarEvent] = []
    for _, row in frame.iterrows():
        signal = signal_from_analyzed_row(
            pair=pair,
            timeframe=timeframe,
            row=row,
            columns=columns,
            feature_timestamp=_aware(row.get("date"), field="date"),
            strategy_version=strategy_version,
        )
        candle = signal.candle
        if candle is None:  # guarded by the required columns above
            raise OperationalException("Hedge backtest could not build an OHLCV candle")
        events.append(
            SignalEvent(
                timestamp=signal.candle_close_time,
                symbol=pair,
                long_signal=signal.long_score,
                short_signal=signal.short_score,
                target_net=signal.target_net,
                model_version=signal.model_version,
                reason=signal.strategy_reason or signal.reason,
                target_net_ratio=signal.target_net_ratio,
                confidence=signal.confidence,
                risk_scale=signal.risk_scale,
                long_exposure_scale=signal.long_exposure_scale,
                short_exposure_scale=signal.short_exposure_scale,
                allow_new_risk=signal.allow_new_risk,
                regime=signal.regime,
            )
        )
        bar = candle.to_bar_event()
        bars.append(bar)
        events.append(bar)

    start = bars[0].timestamp
    end = bars[-1].timestamp
    funding_count = 0
    funding_missing = funding_frame is None or funding_frame.empty
    if require_funding_data and funding_missing:
        raise OperationalException(
            "Hedge futures backtest requires downloaded funding/mark data; "
            "run download-data with the futures configuration first"
        )
    if not funding_missing:
        funding_columns = set(funding_frame.columns)
        required_funding = {"date", "open_fund", "open_mark"}
        if not required_funding.issubset(funding_columns):
            raise OperationalException(
                "Futures funding data must contain date/open_fund/open_mark"
            )
        funding_dates = _strict_dates(funding_frame)
        for (_, row), timestamp in zip(
            funding_frame.iterrows(), funding_dates, strict=True
        ):
            if timestamp < start or timestamp > end:
                continue
            rate = (
                _decimal(row.get("open_fund"), field="funding.open_fund")
                * funding_rate_multiplier
            )
            mark = _decimal(row.get("open_mark"), field="funding.open_mark")
            if mark <= 0:
                raise OperationalException("funding.open_mark must be positive")
            events.append(
                FundingEvent(
                    timestamp=timestamp,
                    symbol=pair,
                    rate=rate,
                    mark_price=mark,
                )
            )
            funding_count += 1

    # Keep detailed and compact mode on one canonical replay order, then hash
    # incrementally so the detailed path does not create a second full JSON tree.
    events.sort(key=lambda event: (event.timestamp, _input_event_priority(event)))
    data_fingerprint = _fingerprint_events(events)
    return HedgeBacktestDataset(
        events=tuple(events),
        pair=pair,
        timeframe=timeframe,
        start=start,
        end=end,
        bar_count=len(bars),
        signal_count=len(bars),
        funding_count=funding_count,
        missing_candle_count=missing_slots,
        data_fingerprint=data_fingerprint,
    )


class HedgeBacktesting:
    """Freqtrade-facing adapter over the shared deterministic event engine."""

    def __init__(
        self,
        *,
        initial_balance: Decimal,
        planner_config: PlannerConfig | None = None,
        leverage: Decimal = DEFAULT_LEVERAGE,
        fee_rate: Decimal = DEFAULT_FEE_RATE,
        long_signal: Decimal = DEFAULT_LONG_SIGNAL,
        short_signal: Decimal = DEFAULT_SHORT_SIGNAL,
        target_net_quantity: Decimal | None = None,
        market_rules: MarketRules | None = None,
        planner: StrategyPlanningPort | None = None,
        match_config: MatchConfig | None = None,
    ) -> None:
        self.engine = EventReplayEngine(
            initial_balance=initial_balance,
            planner_config=planner_config,
            leverage=leverage,
            fee_rate=fee_rate,
            long_signal=long_signal,
            short_signal=short_signal,
            target_net_quantity=target_net_quantity,
            market_rules=market_rules,
            planner=planner,
            match_config=match_config,
        )

    def run(self, events: Iterable[SimulationInputEvent]) -> SimulationResult:
        return self.engine.replay(events)


    def run_compact(
        self,
        chunks: Iterable[Iterable[SimulationInputEvent]],
    ) -> SimulationResult:
        return self.engine.replay_ordered_chunks(chunks)


def _paper_match_config(config: PaperSimulationConfig) -> MatchConfig:
    return MatchConfig(
        maker_fee_rate=config.maker_fee_rate,
        taker_fee_rate=config.taker_fee_rate,
        volume_participation=config.volume_participation,
        market_slippage_bps=config.market_slippage_bps,
        price_tick=config.tick_size,
        qty_step=config.qty_step,
        min_fill_qty=config.min_qty,
        min_fill_notional=config.min_notional,
        max_entry_layers_per_bar=config.max_entry_layers_per_bar,
        max_reduce_layers_per_bar=config.max_reduce_layers_per_bar,
        max_fill_ratio_per_order=config.max_fill_ratio_per_order,
        max_fills_per_bar=config.max_fills_per_bar,
    )


def _input_event_priority(event: SimulationInputEvent) -> int:
    if isinstance(event, SignalEvent):
        return 0
    if isinstance(event, FundingEvent):
        return 1
    return 2


def _fingerprint_events(events: Iterable[SimulationInputEvent]) -> str:
    """Hash an event sequence without constructing a second full JSON tree."""

    hasher = sha256(_STREAM_FINGERPRINT_VERSION)
    for event in events:
        payload = json.dumps(
            _json_value(event),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        hasher.update(len(payload).to_bytes(8, "big"))
        hasher.update(payload)
    return hasher.hexdigest()


def _json_default(value: object) -> object:
    """Incremental JSON adapter matching ``_json_value`` scalar semantics."""
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: getattr(value, item.name) for item in fields(value)}
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_value(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _default_export_path(config: Mapping[str, Any]) -> Path:
    user_data = Path(str(config.get("user_data_dir", "user_data")))
    directory = user_data / "backtest_results"
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
    return directory / f"hedge-backtest-{stamp}.json"



def _json_sha256_stream(value: object) -> str:
    """Hash JSON incrementally without materializing one giant encoded string."""
    hasher = sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=_json_default,
    )
    for chunk in encoder.iterencode(value):
        hasher.update(chunk.encode("utf-8"))
    return hasher.hexdigest()


def _file_sha256_stream(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            hasher.update(block)
    return hasher.hexdigest()

def _write_result(
    *,
    path: Path,
    result: SimulationResult,
    dataset: HedgeBacktestDataset,
    strategy: str,
    market_rule_source: str,
    market_rule_version: str,
    export_events: bool,
) -> tuple[str, str, object]:
    from freqtrade.hedge.native.backtest import HedgeBacktestResultAdapter

    native_artifact = HedgeBacktestResultAdapter().build(
        result,
        strategy_name=strategy,
        pairs=(dataset.pair,),
        timeframe=dataset.timeframe,
        timerange=f"{dataset.start.isoformat()}-{dataset.end.isoformat()}",
        metadata={
            "start": dataset.start.isoformat(),
            "end": dataset.end.isoformat(),
            "bar_count": dataset.bar_count,
            "signal_count": dataset.signal_count,
            "funding_count": dataset.funding_count,
            "data_fingerprint": dataset.data_fingerprint,
            "market_rule_source": market_rule_source,
            "market_rule_version": market_rule_version,
            "execution_timing": "NEXT_BAR_NO_LOOKAHEAD",
            "replay_mode": result.report.get("replay_mode", "FULL_MATERIALIZED"),
            "retained_snapshot_count": result.report.get(
                "retained_snapshot_count", len(result.snapshots)
            ),
            "retained_event_count": result.report.get(
                "retained_event_count", len(result.events)
            ),
        },
    )
    export_artifact = native_artifact if export_events else replace(native_artifact, events=())
    deterministic_payload: dict[str, object] = {
        "schema_version": "hedge-backtest-result-v4",
        "execution_timing": "NEXT_BAR_NO_LOOKAHEAD",
        "pair": dataset.pair,
        "timeframe": dataset.timeframe,
        "start": dataset.start,
        "end": dataset.end,
        "strategy": strategy,
        "market_rule_source": market_rule_source,
        "market_rule_version": market_rule_version,
        "bar_count": dataset.bar_count,
        "signal_count": dataset.signal_count,
        "funding_count": dataset.funding_count,
        "missing_candle_count": dataset.missing_candle_count,
        "data_fingerprint": dataset.data_fingerprint,
        "report": result.report,
        "snapshots": result.snapshots,
        "hedge_native": export_artifact.to_dict(),
        "freqtrade_projection": export_artifact.frequi_projection(),
    }
    if export_events:
        deterministic_payload["events"] = result.events
    result_fingerprint = _json_sha256_stream(deterministic_payload)
    payload = {
        **deterministic_payload,
        "created_at": datetime.now(UTC),
        "result_fingerprint": result_fingerprint,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    temp.replace(path)
    digest = _file_sha256_stream(path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar_temp = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_temp.write_text(f"{digest}  {path.name}\n", encoding="ascii")
    sidecar_temp.replace(sidecar)
    return digest, result_fingerprint, native_artifact


def run_freqtrade_hedge_backtest(
    config: dict[str, Any],
    *,
    export_path: Path | None = None,
    export_events: bool = False,
) -> HedgeBacktestRun:
    """Run a single-pair Hedge backtest through Freqtrade's normal data stack."""

    from freqtrade.data.converter import trim_dataframe
    from freqtrade.hedge.config import validate_hedge_config
    from freqtrade.hedge.integration.market_data import exchange_market_rules
    from freqtrade.hedge.integration.paper_runtime import planner_config_from_mapping
    from freqtrade.hedge.memory_lifecycle import (
        HedgeMemoryPolicy,
        clear_dataprovider_caches,
        clear_exchange_caches,
        clear_strategy_caches,
        log_memory_snapshot,
        phase_boundary_cleanup as hedge_memory_cleanup,
    )
    from freqtrade.optimize.backtesting import Backtesting

    hedge_runtime = validate_hedge_config(config)
    paper = hedge_runtime.paper
    if not hedge_runtime.enabled or paper is None:
        raise OperationalException("hedge-backtesting requires Hedge mode and hedge.paper")
    if hedge_runtime.operation_mode not in {"paper", "shadow"}:
        raise OperationalException(
            "hedge-backtesting requires hedge.operation_mode=paper or shadow"
        )

    optimization_runtime = config.get("hedge_optimization_runtime", {})
    runtime_mapping = (
        optimization_runtime if isinstance(optimization_runtime, Mapping) else {}
    )
    memory_raw = runtime_mapping.get("memory", {})
    memory_mapping = memory_raw if isinstance(memory_raw, Mapping) else {}
    memory_policy = HedgeMemoryPolicy.from_mapping(dict(memory_mapping))
    # Freqtrade already provides the canonical dataframe footprint reducer.
    # Enable it for Hedge backtests unless the caller explicitly disables the
    # Hedge memory policy.  Use a shallow top-level copy so the caller's config
    # is not mutated.
    runtime_config = dict(config)
    pair = hedge_runtime.managed_pair
    if pair is None:
        raise OperationalException("hedge-backtesting requires hedge.managed_pair")
    if memory_policy.release_unmanaged_pair_data:
        # Scope Freqtrade's own data loader before it allocates historical frames.
        # Waiting until after load_bt_data() to discard unrelated pairs would reduce
        # steady-state memory but not the peak.  The caller's config is untouched.
        exchange_config = dict(runtime_config.get("exchange", {}))
        exchange_config["pair_whitelist"] = [pair]
        runtime_config["exchange"] = exchange_config
    runtime_optimization = dict(runtime_mapping)
    runtime_memory = dict(memory_mapping)
    runtime_memory.setdefault("backtesting_cache_mode", memory_policy.backtesting_cache_mode)
    runtime_memory.setdefault(
        "backtesting_cache_max_entries", memory_policy.backtesting_cache_max_entries
    )
    runtime_optimization["memory"] = runtime_memory
    runtime_config["hedge_optimization_runtime"] = runtime_optimization
    if memory_policy.reduce_dataframe_footprint:
        runtime_config["reduce_df_footprint"] = True

    memory_telemetry: list[Mapping[str, Any]] = []
    memory_telemetry.append(log_memory_snapshot("START"))
    backend = Backtesting(runtime_config)
    memory_telemetry.append(log_memory_snapshot("BACKEND_READY"))
    with backend.progress or nullcontext():
        data, timerange = backend.load_bt_data()
        memory_telemetry.append(log_memory_snapshot("DATA_LOADED"))
        if len(backend.strategylist) != 1:
            raise OperationalException(
                "hedge-backtesting currently supports exactly one strategy per run"
            )
        strategy = backend.strategylist[0]
        backend._set_strategy(strategy)
        if pair not in data:
            raise OperationalException(
                f"managed_pair {pair!r} is not available in the loaded backtest data"
            )
        if memory_policy.release_unmanaged_pair_data and len(data) > 1:
            managed_frame = data[pair]
            data.clear()
            data[pair] = managed_frame
            del managed_frame
            # Do not carry unrelated base-pair frames into indicator analysis.
            # Informative frames remain available through DataProvider on demand.
            hedge_memory_cleanup(policy=memory_policy)
            memory_telemetry.append(log_memory_snapshot("PAIR_SCOPED"))
        analyzed = strategy.advise_all_indicators(data)
        frame = strategy.ft_advise_signals(analyzed[pair], {"pair": pair})
        frame = trim_dataframe(
            frame,
            timerange,
            startup_candles=backend.required_startup,
        )
        del analyzed
        del data
        memory_telemetry.append(log_memory_snapshot("ANALYZED"))

    if len(frame) < 2:
        raise OperationalException(
            "hedge-backtesting requires at least two analyzed candles for next-bar execution"
        )
    version_attr = strategy.version if hasattr(strategy, "version") else None
    strategy_version = version_attr() if callable(version_attr) else version_attr
    funding_rate_multiplier = _decimal(
        runtime_mapping.get("funding_rate_multiplier", "1"),
        field="hedge_optimization_runtime.funding_rate_multiplier",
    )
    if funding_rate_multiplier < 0:
        raise OperationalException(
            "hedge_optimization_runtime.funding_rate_multiplier cannot be negative"
        )

    detailed_dataset: HedgeBacktestDataset | None = None
    compact_stream: HedgeBacktestEventChunks | None = None
    if export_events:
        detailed_dataset = events_from_analyzed_dataframe(
            pair=pair,
            timeframe=backend.timeframe,
            frame=frame,
            funding_frame=backend.futures_data.get(pair),
            strategy_version=strategy_version,
            require_funding_data=paper.funding_source is PaperFundingSource.EXCHANGE,
            max_missing_candles=paper.max_missing_candles,
            funding_rate_multiplier=funding_rate_multiplier,
        )
    else:
        compact_stream = HedgeBacktestEventChunks(
            pair=pair,
            timeframe=backend.timeframe,
            frame=frame,
            funding_frame=backend.futures_data.get(pair),
            strategy_version=strategy_version,
            require_funding_data=paper.funding_source is PaperFundingSource.EXCHANGE,
            max_missing_candles=paper.max_missing_candles,
            funding_rate_multiplier=funding_rate_multiplier,
        )
        # The compact stream owns the narrow replay surface.  Release every
        # upstream dataframe/cache owner before the million-bar replay starts.
        # Upstream DataProvider deliberately preserves historical data by default
        # for Hyperopt reuse; this one-shot compact path explicitly opts into full
        # release after detachment.
        del frame
        clear_strategy_caches(strategy)
        clear_dataprovider_caches(backend.dataprovider, memory_policy)
        backend.futures_data.clear()
        backend.price_pair_prec.clear()
        backend.detail_data.clear()
        strategy.dp = None
        hedge_memory_cleanup(policy=memory_policy)
        memory_telemetry.append(log_memory_snapshot("REPLAY_DETACHED"))
    raw_hedge = config.get("hedge", {})
    hedge_mapping = raw_hedge if isinstance(raw_hedge, Mapping) else {}
    planner_raw = hedge_mapping.get("planner", {})
    planner_mapping = planner_raw if isinstance(planner_raw, Mapping) else {}
    paper_raw = hedge_mapping.get("paper", {})
    paper_mapping = paper_raw if isinstance(paper_raw, Mapping) else {}
    rule_snapshot = exchange_market_rules(
        exchange=backend.exchange,
        pair=pair,
        fallback=paper_mapping,
    )
    if not export_events:
        clear_exchange_caches(backend.exchange, memory_policy)
        hedge_memory_cleanup(policy=memory_policy)
    market_rules = MarketRules(
        tick_size=rule_snapshot.tick_size,
        qty_step=rule_snapshot.qty_step,
        min_qty=rule_snapshot.min_qty,
        min_notional=rule_snapshot.min_notional,
    )
    maker_fee_multiplier = _decimal(
        runtime_mapping.get("maker_fee_multiplier", "1"),
        field="hedge_optimization_runtime.maker_fee_multiplier",
    )
    taker_fee_multiplier = _decimal(
        runtime_mapping.get("taker_fee_multiplier", "1"),
        field="hedge_optimization_runtime.taker_fee_multiplier",
    )
    if maker_fee_multiplier < 0 or taker_fee_multiplier < 0:
        raise OperationalException(
            "hedge_optimization_runtime fee multipliers cannot be negative"
        )
    match_config = _paper_match_config(paper)
    match_config = MatchConfig(
        **{
            **asdict(match_config),
            "maker_fee_rate": rule_snapshot.maker_fee_rate * maker_fee_multiplier,
            "taker_fee_rate": rule_snapshot.taker_fee_rate * taker_fee_multiplier,
            "price_tick": rule_snapshot.tick_size,
            "qty_step": rule_snapshot.qty_step,
            "min_fill_qty": rule_snapshot.min_qty,
            "min_fill_notional": rule_snapshot.min_notional,
        }
    )
    # No further upstream Backtesting state is needed after market rules and
    # detached replay inputs are prepared.  Dropping the owner here prevents
    # hidden references from keeping historical data alive during Hedge replay.
    if not export_events:
        del backend
        hedge_memory_cleanup(policy=memory_policy)
        memory_telemetry.append(log_memory_snapshot("BACKEND_RELEASED"))

    runner = HedgeBacktesting(
        initial_balance=paper.initial_balance,
        planner_config=planner_config_from_mapping(planner_mapping),
        leverage=paper.leverage,
        fee_rate=rule_snapshot.taker_fee_rate * taker_fee_multiplier,
        long_signal=paper.default_long_signal,
        short_signal=paper.default_short_signal,
        market_rules=market_rules,
        match_config=match_config,
    )
    if detailed_dataset is not None:
        dataset = detailed_dataset
        result = runner.run(dataset.events)
    else:
        if compact_stream is None:  # pragma: no cover - defensive invariant
            raise OperationalException("compact Hedge backtest stream was not initialized")
        result = runner.run_compact(compact_stream)
        dataset = compact_stream.dataset()
    memory_telemetry.append(log_memory_snapshot("REPLAY_DONE"))

    output = (export_path or _default_export_path(config)).expanduser().resolve()
    strategy_name = strategy.get_strategy_name()
    artifact_sha256, result_fingerprint, native_artifact = _write_result(
        path=output,
        result=result,
        dataset=dataset,
        strategy=strategy_name,
        market_rule_source=rule_snapshot.source,
        market_rule_version=rule_snapshot.version,
        export_events=export_events,
    )
    memory_telemetry.append(log_memory_snapshot("RESULT_WRITTEN"))
    return HedgeBacktestRun(
        result=result,
        dataset=dataset,
        export_path=output,
        strategy=strategy_name,
        market_rule_source=rule_snapshot.source,
        market_rule_version=rule_snapshot.version,
        artifact_sha256=artifact_sha256,
        result_fingerprint=result_fingerprint,
        native_artifact=native_artifact,
        memory_telemetry=tuple(memory_telemetry),
    )
