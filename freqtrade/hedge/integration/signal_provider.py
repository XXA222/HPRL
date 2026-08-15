"""Strategy/DataProvider adapter for deterministic Hedge signals and closed OHLCV."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from .candle_cursor import bar_fingerprint, missing_candle_count

from freqtrade.exchange import timeframe_to_seconds
from freqtrade.hedge.simulation.exchange import BarEvent


@dataclass(frozen=True, slots=True)
class AnalyzedCandle:
    symbol: str
    timeframe: str
    open_time: datetime
    close_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal | None
    source: str = "DATAPROVIDER_ANALYZED"

    def __post_init__(self) -> None:
        if self.open_time.tzinfo is None or self.close_time.tzinfo is None:
            raise ValueError("candle timestamps must be timezone-aware")
        if self.close_time <= self.open_time:
            raise ValueError("candle close_time must be after open_time")
        # Reuse the simulation event invariant checks so live Paper and backtest
        # cannot silently diverge on malformed OHLCV.
        BarEvent(
            timestamp=self.close_time,
            symbol=self.symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )

    def to_bar_event(self) -> BarEvent:
        return BarEvent(
            timestamp=self.close_time,
            symbol=self.symbol,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
        )


@dataclass(frozen=True, slots=True)
class SignalSnapshot:
    symbol: str
    timeframe: str
    candle_close_time: datetime
    feature_timestamp: datetime
    long_score: Decimal
    short_score: Decimal
    target_net: Decimal | None
    model_version: str
    reason: str
    target_net_ratio: Decimal | None = None
    confidence: Decimal = Decimal("1")
    risk_scale: Decimal = Decimal("1")
    long_exposure_scale: Decimal = Decimal("1")
    short_exposure_scale: Decimal = Decimal("1")
    allow_new_risk: bool = True
    regime: str = "UNSPECIFIED"
    strategy_reason: str = ""
    candle: AnalyzedCandle | None = None

    def __post_init__(self) -> None:
        for name in ("long_score", "short_score"):
            value = getattr(self, name)
            if not value.is_finite() or value < 0 or value > 1:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if self.target_net is not None and not self.target_net.is_finite():
            raise ValueError("target_net must be finite")
        if self.target_net_ratio is not None and (not self.target_net_ratio.is_finite() or self.target_net_ratio < -1 or self.target_net_ratio > 1):
            raise ValueError("target_net_ratio must be within [-1, 1]")
        for name in ("confidence", "risk_scale", "long_exposure_scale", "short_exposure_scale"):
            value = getattr(self, name)
            if not value.is_finite() or value < 0 or value > 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.candle_close_time.tzinfo is None or self.feature_timestamp.tzinfo is None:
            raise ValueError("signal timestamps must be timezone-aware")
        if self.candle is not None and self.candle.close_time != self.candle_close_time:
            raise ValueError("signal and candle close timestamps must match")


class HedgeSignalProviderPort(Protocol):
    def signals(self, pair: str, timeframe: str) -> SignalSnapshot: ...

    def signals_since(
        self,
        pair: str,
        timeframe: str,
        *,
        after: datetime | None,
        cursor_fingerprint: str | None = None,
        max_catchup_candles: int = 288,
        max_missing_candles: int = 0,
        reject_revised_candle: bool = True,
    ) -> tuple[SignalSnapshot, ...]: ...


def _decimal_score(value: object) -> Decimal:
    if value is None:
        return Decimal("0")
    try:
        result = Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return Decimal("0")
    if not result.is_finite():
        return Decimal("0")
    return min(Decimal("1"), max(Decimal("0"), result))


def _exact_decimal(value: object, *, field: str, optional: bool = False) -> Decimal | None:
    if value is None and optional:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"analyzed candle {field} is not a decimal") from exc
    if not result.is_finite():
        raise ValueError(f"analyzed candle {field} must be finite")
    return result


def _aware_datetime(value: object, fallback: datetime) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        return fallback
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _required_aware_datetime(value: object, *, field: str) -> datetime:
    if hasattr(value, "to_pydatetime"):
        value = value.to_pydatetime()
    if not isinstance(value, datetime):
        raise ValueError(f"analyzed candle {field} must be a datetime")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _candle_from_row(
    pair: str,
    timeframe: str,
    row: object,
    columns: set[str],
) -> AnalyzedCandle | None:
    required = {"date", "open", "high", "low", "close"}
    if not required.issubset(columns):
        return None
    open_time = _required_aware_datetime(
        row.get("date"), field="date"  # type: ignore[attr-defined]
    )
    close_time = open_time + timedelta(seconds=timeframe_to_seconds(timeframe))
    volume = (
        _exact_decimal(
            row.get("volume"),  # type: ignore[attr-defined]
            field="volume",
            optional=True,
        )
        if "volume" in columns
        else None
    )
    return AnalyzedCandle(
        symbol=pair,
        timeframe=timeframe,
        open_time=open_time,
        close_time=close_time,
        open=_exact_decimal(row.get("open"), field="open"),  # type: ignore[arg-type,attr-defined]
        high=_exact_decimal(row.get("high"), field="high"),  # type: ignore[arg-type,attr-defined]
        low=_exact_decimal(row.get("low"), field="low"),  # type: ignore[arg-type,attr-defined]
        close=_exact_decimal(
            row.get("close"),  # type: ignore[attr-defined]
            field="close",
        ),
        volume=volume,
    )


def signal_from_analyzed_row(
    *,
    pair: str,
    timeframe: str,
    row: object,
    columns: set[str],
    feature_timestamp: datetime,
    strategy_version: object = None,
) -> SignalSnapshot:
    """Convert one already-analyzed row into the canonical Hedge signal.

    This function is shared by live Paper and historical Hedge backtesting so
    both modes interpret ``hedge_*`` and legacy ``enter_*`` columns identically.
    It never shifts or reads a future row; next-bar order activation is enforced
    by the shared simulation/Paper execution engines.
    """

    candle = _candle_from_row(pair, timeframe, row, columns)
    if candle is None:
        raise ValueError(
            "analyzed dataframe must contain date/open/high/low/close for Hedge"
        )
    from freqtrade.hedge.strategies.contract import directive_from_values

    directive = directive_from_values(
        {name: row.get(name) for name in columns if name.startswith("hedge_")}  # type: ignore[attr-defined]
        | {
            "enter_long": row.get("enter_long", 0),  # type: ignore[attr-defined]
            "enter_short": row.get("enter_short", 0),  # type: ignore[attr-defined]
        }
    )
    target_net = directive.target_net_quantity

    version_value = (
        row.get("hedge_model_version")  # type: ignore[attr-defined]
        if "hedge_model_version" in columns
        else None
    )
    model_version = str(version_value or strategy_version or "strategy")[:128]
    reason = (
        "HEDGE_TARGET_COLUMNS"
        if {"hedge_long_score", "hedge_short_score"} & columns
        else "ENTER_SIGNAL_COMPATIBILITY"
    )
    return SignalSnapshot(
        symbol=pair,
        timeframe=timeframe,
        candle_close_time=candle.close_time,
        feature_timestamp=_aware_datetime(feature_timestamp, candle.close_time),
        long_score=directive.long_score,
        short_score=directive.short_score,
        target_net=target_net,
        model_version=directive.model_version if "hedge_model_version" in columns else model_version,
        reason=reason,
        target_net_ratio=directive.target_net_ratio,
        confidence=directive.confidence,
        risk_scale=directive.risk_scale,
        long_exposure_scale=directive.long_exposure_scale,
        short_exposure_scale=directive.short_exposure_scale,
        allow_new_risk=directive.allow_new_risk,
        regime=directive.regime,
        strategy_reason=directive.reason,
        candle=candle,
    )


class FreqtradeStrategySignalProvider:
    """Read one already-analyzed closed candle without creating future data.

    Preferred columns are ``hedge_long_score``, ``hedge_short_score`` and
    ``hedge_target_net``. Existing strategies remain compatible through
    ``enter_long``/``enter_short``. The exact OHLCV row used for the signal is
    returned with the signal so Paper matching and strategy decisions share one
    timestamp and one source of truth.
    """

    def __init__(
        self,
        dataprovider: Any,
        strategy: Any,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._dataprovider = dataprovider
        self._strategy = strategy
        self._clock = clock or (lambda: datetime.now(UTC))

    def _load(self, pair: str, timeframe: str) -> tuple[object, datetime, set[str]]:
        frame, refreshed_at = self._dataprovider.get_analyzed_dataframe(pair, timeframe)
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("signal-provider clock must be timezone-aware")
        now = now.astimezone(UTC)
        columns = set(getattr(frame, "columns", ())) if frame is not None else set()
        return frame, _aware_datetime(refreshed_at, now), columns

    def signals(self, pair: str, timeframe: str) -> SignalSnapshot:
        """Compatibility view of the latest analyzed row.

        Durable Paper uses :meth:`signals_since`, which filters unfinished
        candles. Existing read-only callers historically expected the literal
        latest analyzed row and retain that behavior here.
        """
        frame, refreshed_at, columns = self._load(pair, timeframe)
        if frame is not None and not getattr(frame, "empty", True):
            version_attr = getattr(self._strategy, "version", None)
            strategy_version = version_attr() if callable(version_attr) else version_attr
            return signal_from_analyzed_row(
                pair=pair,
                timeframe=timeframe,
                row=frame.iloc[-1],
                columns=columns,
                feature_timestamp=refreshed_at,
                strategy_version=strategy_version,
            )
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("signal-provider clock must be timezone-aware")
        now = now.astimezone(UTC)
        return SignalSnapshot(
            symbol=pair,
            timeframe=timeframe,
            candle_close_time=now,
            feature_timestamp=now,
            long_score=Decimal("0"),
            short_score=Decimal("0"),
            target_net=None,
            model_version="none",
            reason="NO_ANALYZED_DATA",
            candle=None,
        )

    def signals_since(
        self,
        pair: str,
        timeframe: str,
        *,
        after: datetime | None,
        cursor_fingerprint: str | None = None,
        max_catchup_candles: int = 288,
        max_missing_candles: int = 0,
        reject_revised_candle: bool = True,
    ) -> tuple[SignalSnapshot, ...]:
        """Return all unseen analyzed candles in deterministic close-time order.

        Timestamp validation scans the DataFrame, but expensive Decimal/OHLCV
        normalization is limited to the durable cursor and unseen rows. This
        avoids rebuilding every historical SignalSnapshot on each live tick.
        """

        if max_catchup_candles < 1:
            raise ValueError("max_catchup_candles must be positive")
        if max_missing_candles < 0:
            raise ValueError("max_missing_candles cannot be negative")
        frame, refreshed_at, columns = self._load(pair, timeframe)
        if frame is None or getattr(frame, "empty", True):
            return ()
        required = {"date", "open", "high", "low", "close"}
        if not required.issubset(columns):
            raise ValueError(
                "analyzed dataframe must contain date/open/high/low/close for Hedge"
            )

        timeframe_seconds = timeframe_to_seconds(timeframe)
        close_times: list[datetime] = []
        for raw_date in frame["date"]:
            open_time = _required_aware_datetime(raw_date, field="date")
            close_time = open_time + timedelta(seconds=timeframe_seconds)
            if close_times and close_time <= close_times[-1]:
                raise ValueError(
                    "analyzed DataProvider candles must be strictly ordered"
                )
            close_times.append(close_time)

        version_attr = getattr(self._strategy, "version", None)
        strategy_version = version_attr() if callable(version_attr) else version_attr

        as_of = self._clock()
        if as_of.tzinfo is None:
            raise ValueError("signal-provider clock must be timezone-aware")
        as_of = as_of.astimezone(UTC)
        closed_indices = tuple(
            index for index, value in enumerate(close_times) if value <= as_of
        )
        if not closed_indices:
            return ()

        if after is None:
            selected_indices = (closed_indices[-1],)
        else:
            if after.tzinfo is None:
                raise ValueError("durable candle cursor must be timezone-aware")
            after = after.astimezone(UTC)
            cursor_index = next(
                (index for index, value in enumerate(close_times) if value == after),
                None,
            )
            if cursor_fingerprint is not None and cursor_index is not None:
                cursor_snapshot = signal_from_analyzed_row(
                    pair=pair,
                    timeframe=timeframe,
                    row=frame.iloc[cursor_index],
                    columns=columns,
                    feature_timestamp=refreshed_at,
                    strategy_version=strategy_version,
                )
                if cursor_snapshot.candle is None:
                    raise ValueError("durable cursor candle is missing OHLCV")
                observed = bar_fingerprint(cursor_snapshot.candle.to_bar_event())
                if reject_revised_candle and observed != cursor_fingerprint:
                    raise ValueError("durable candle cursor OHLCV was revised")

            selected_indices = tuple(
                index
                for index in closed_indices
                if close_times[index] > after
            )
            if len(selected_indices) > max_catchup_candles:
                raise ValueError(
                    f"Paper catch-up requires {len(selected_indices)} candles; "
                    f"limit={max_catchup_candles}"
                )
            missing = 0
            previous_close = after
            for index in selected_indices:
                missing += missing_candle_count(
                    previous_close, close_times[index], timeframe
                )
                previous_close = close_times[index]
            if missing > max_missing_candles:
                raise ValueError(
                    f"Paper DataProvider history has {missing} missing candle slots; "
                    f"limit={max_missing_candles}"
                )

        return tuple(
            signal_from_analyzed_row(
                pair=pair,
                timeframe=timeframe,
                row=frame.iloc[index],
                columns=columns,
                feature_timestamp=refreshed_at,
                strategy_version=strategy_version,
            )
            for index in selected_indices
        )
