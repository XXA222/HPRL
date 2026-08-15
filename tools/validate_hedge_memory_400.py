from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
import sys
import types
from typing import Callable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _install_exchange_stub() -> None:
    if "freqtrade.exchange" in sys.modules:
        return
    module = types.ModuleType("freqtrade.exchange")

    def timeframe_to_seconds(timeframe: str) -> int:
        value = timeframe.strip().lower()
        amount = int(value[:-1])
        factor = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[value[-1]]
        return amount * factor

    module.timeframe_to_seconds = timeframe_to_seconds
    sys.modules["freqtrade.exchange"] = module


_install_exchange_stub()

from freqtrade.hedge.simulation.exchange import BarEvent, FundingEvent, SignalEvent
from freqtrade.optimize.hedge_backtesting import (
    HedgeBacktestEventChunks,
    HedgeBacktesting,
    events_from_analyzed_dataframe,
)


@dataclass(frozen=True, slots=True)
class CheckResult:
    number: int
    group: str
    case: str
    passed: bool
    detail: str = ""


def frame(
    count: int,
    *,
    long_score: float = 0.0,
    short_score: float = 0.0,
    start: datetime | None = None,
) -> pd.DataFrame:
    origin = start or datetime(2026, 1, 1, tzinfo=UTC)
    dates = pd.date_range(origin, periods=count, freq="min")
    price = np.full(count, 100.0, dtype="float32")
    return pd.DataFrame(
        {
            "date": dates,
            "open": price,
            "high": price + 1.0,
            "low": price - 1.0,
            "close": price,
            "volume": np.full(count, 100.0, dtype="float32"),
            "hedge_long_score": np.full(count, long_score, dtype="float32"),
            "hedge_short_score": np.full(count, short_score, dtype="float32"),
            "hedge_target_net_ratio": np.zeros(count, dtype="float32"),
        }
    )


def compact_run(dataframe: pd.DataFrame, *, chunk_bars: int = 4):
    stream = HedgeBacktestEventChunks(
        pair="BTC/USDT:USDT",
        timeframe="1m",
        frame=dataframe,
        chunk_bars=chunk_bars,
    )
    result = HedgeBacktesting(initial_balance=Decimal("1000")).run_compact(stream)
    return stream, result


def business_report(report: dict[str, object]) -> dict[str, object]:
    telemetry = {
        "replay_mode",
        "processed_chunk_count",
        "processed_input_event_count",
        "processed_bar_count",
        "retained_event_count",
        "retained_snapshot_count",
        "max_chunk_input_events",
    }
    return {key: value for key, value in report.items() if key not in telemetry}


def run() -> list[CheckResult]:
    checks: list[tuple[str, str, Callable[[], None]]] = []

    def add(group: str, case: str, fn: Callable[[], None]) -> None:
        checks.append((group, case, fn))

    # 001-020: stream counts across dataset sizes.
    for count in range(2, 22):
        def check(count=count) -> None:
            stream = HedgeBacktestEventChunks(
                pair="BTC/USDT:USDT",
                timeframe="1m",
                frame=frame(count),
                chunk_bars=5,
            )
            total = sum(len(chunk) for chunk in stream)
            dataset = stream.dataset()
            assert dataset.bar_count == count
            assert dataset.signal_count == count
            assert total == count * 2
            assert dataset.events == ()
        add("G01_STREAM_COUNTS", f"bars={count}", check)

    # 021-040: chunk-size bounds.
    for chunk_bars in range(1, 21):
        def check(chunk_bars=chunk_bars) -> None:
            stream = HedgeBacktestEventChunks(
                pair="BTC/USDT:USDT",
                timeframe="1m",
                frame=frame(43),
                chunk_bars=chunk_bars,
            )
            chunks = list(stream)
            assert chunks
            assert max(map(len, chunks)) <= chunk_bars * 2
            assert stream.max_chunk_input_events <= chunk_bars * 2
        add("G02_CHUNK_BOUNDS", f"chunk_bars={chunk_bars}", check)

    # 041-060: long signal changes fingerprint.
    baseline_long = None
    for index in range(1, 21):
        def check(index=index) -> None:
            nonlocal baseline_long
            score = index / 20.0
            stream = HedgeBacktestEventChunks(
                pair="BTC/USDT:USDT",
                timeframe="1m",
                frame=frame(5, long_score=score),
                chunk_bars=2,
            )
            list(stream)
            fingerprint = stream.dataset().data_fingerprint
            assert len(fingerprint) == 64
            if index == 1:
                baseline_long = fingerprint
            else:
                assert fingerprint != baseline_long
        add("G03_LONG_FINGERPRINT", f"score={index}/20", check)

    # 061-080: short signal changes fingerprint.
    baseline_short = None
    for index in range(1, 21):
        def check(index=index) -> None:
            nonlocal baseline_short
            score = index / 20.0
            stream = HedgeBacktestEventChunks(
                pair="BTC/USDT:USDT",
                timeframe="1m",
                frame=frame(5, short_score=score),
                chunk_bars=2,
            )
            list(stream)
            fingerprint = stream.dataset().data_fingerprint
            assert len(fingerprint) == 64
            if index == 1:
                baseline_short = fingerprint
            else:
                assert fingerprint != baseline_short
        add("G04_SHORT_FINGERPRINT", f"score={index}/20", check)

    # 081-100: funding multiplier is exact and fingerprinted.
    for index in range(1, 21):
        def check(index=index) -> None:
            funding = pd.DataFrame(
                [
                    {
                        "date": datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
                        "open_fund": "0.0001",
                        "open_mark": "100",
                    }
                ]
            )
            stream = HedgeBacktestEventChunks(
                pair="BTC/USDT:USDT",
                timeframe="1m",
                frame=frame(4),
                funding_frame=funding,
                funding_rate_multiplier=Decimal(index),
                chunk_bars=2,
            )
            events = [event for chunk in stream for event in chunk]
            funding_event = next(event for event in events if isinstance(event, FundingEvent))
            assert funding_event.rate == Decimal("0.0001") * index
            assert stream.dataset().funding_count == 1
        add("G05_FUNDING_MULTIPLIER", f"multiplier={index}", check)

    # 101-120: same-timestamp priority remains Signal, Funding, Bar.
    for minute in range(1, 21):
        def check(minute=minute) -> None:
            data = frame(22)
            timestamp = datetime(2026, 1, 1, 0, minute, tzinfo=UTC)
            funding = pd.DataFrame(
                [{"date": timestamp, "open_fund": "0.0001", "open_mark": "100"}]
            )
            stream = HedgeBacktestEventChunks(
                pair="BTC/USDT:USDT",
                timeframe="1m",
                frame=data,
                funding_frame=funding,
                chunk_bars=5,
            )
            events = [event for chunk in stream for event in chunk]
            same = [event for event in events if event.timestamp == timestamp]
            assert [type(event) for event in same] == [SignalEvent, FundingEvent, BarEvent]
        add("G06_PRIORITY", f"minute={minute}", check)

    # 121-140: missing-candle tolerance is enforced incrementally.
    for gap in range(1, 21):
        def check(gap=gap) -> None:
            data = frame(5)
            data.loc[3:, "date"] = data.loc[3:, "date"] + timedelta(minutes=gap)
            stream = HedgeBacktestEventChunks(
                pair="BTC/USDT:USDT",
                timeframe="1m",
                frame=data,
                max_missing_candles=gap,
                chunk_bars=3,
            )
            list(stream)
            assert stream.dataset().missing_candle_count == gap
        add("G07_MISSING_TOLERANCE", f"missing={gap}", check)

    # 141-160: detailed/compact business report parity.
    for index in range(20):
        def check(index=index) -> None:
            long_score = (index % 5) / 4.0
            short_score = ((index * 3) % 5) / 4.0
            data = frame(7, long_score=long_score, short_score=short_score)
            detailed_dataset = events_from_analyzed_dataframe(
                pair="BTC/USDT:USDT", timeframe="1m", frame=data
            )
            detailed = HedgeBacktesting(initial_balance=Decimal("1000")).run(
                detailed_dataset.events
            )
            _, compact = compact_run(data, chunk_bars=3)
            assert business_report(compact.report) == detailed.report
        add("G08_REPORT_PARITY", f"case={index}", check)

    # 161-180: compact result is chunk-size invariant.
    reference_report = None
    for chunk_bars in range(1, 21):
        def check(chunk_bars=chunk_bars) -> None:
            nonlocal reference_report
            _, result = compact_run(
                frame(11, long_score=0.75, short_score=0.75),
                chunk_bars=chunk_bars,
            )
            current = business_report(result.report)
            if chunk_bars == 1:
                reference_report = current
            else:
                assert current == reference_report
        add("G09_CHUNK_PARITY", f"chunk_bars={chunk_bars}", check)

    # 181-200: snapshot count is O(chunks), not O(bars).
    for index in range(1, 21):
        def check(index=index) -> None:
            count = 20 + index
            chunk_bars = 3 + index
            _, result = compact_run(frame(count), chunk_bars=chunk_bars)
            expected_max = (count + chunk_bars - 1) // chunk_bars
            assert len(result.snapshots) <= expected_max
        add("G10_SNAPSHOT_BOUND", f"case={index}", check)

    # 201-220: processed slot history is released after compact replay.
    for chunk_bars in range(1, 21):
        def check(chunk_bars=chunk_bars) -> None:
            runner = HedgeBacktesting(initial_balance=Decimal("1000"))
            stream = HedgeBacktestEventChunks(
                pair="BTC/USDT:USDT",
                timeframe="1m",
                frame=frame(13),
                chunk_bars=chunk_bars,
            )
            runner.run_compact(stream)
            assert runner.engine._processed_slots == set()
        add("G11_SLOT_RELEASE", f"chunk_bars={chunk_bars}", check)

    # 221-240: input Signal/Bar/Funding events are not retained.
    for index in range(20):
        def check(index=index) -> None:
            _, result = compact_run(
                frame(10, long_score=(index % 4) / 3.0),
                chunk_bars=2 + (index % 5),
            )
            assert not any(
                isinstance(event, (SignalEvent, FundingEvent, BarEvent))
                for event in result.events
            )
        add("G12_NO_INPUT_LEDGER", f"case={index}", check)

    # 241-260: deterministic compact report across repeated runs.
    for index in range(20):
        def check(index=index) -> None:
            data = frame(
                9,
                long_score=(index % 5) / 4.0,
                short_score=((index + 2) % 5) / 4.0,
            )
            _, one = compact_run(data, chunk_bars=4)
            _, two = compact_run(data, chunk_bars=4)
            assert one.report == two.report
        add("G13_COMPACT_DETERMINISM", f"case={index}", check)

    # 261-280: stream is single-use and completion-gated.
    for count in range(2, 22):
        def check(count=count) -> None:
            stream = HedgeBacktestEventChunks(
                pair="BTC/USDT:USDT",
                timeframe="1m",
                frame=frame(count),
                chunk_bars=4,
            )
            try:
                stream.dataset()
            except RuntimeError:
                pass
            else:
                raise AssertionError("dataset() must reject an unconsumed stream")
            list(stream)
            try:
                list(stream)
            except RuntimeError:
                pass
            else:
                raise AssertionError("stream must be single-use")
        add("G14_STREAM_LIFECYCLE", f"bars={count}", check)

    # 281-300: legacy detailed path remains available and deterministic.
    for count in range(2, 22):
        def check(count=count) -> None:
            one = events_from_analyzed_dataframe(
                pair="BTC/USDT:USDT", timeframe="1m", frame=frame(count)
            )
            two = events_from_analyzed_dataframe(
                pair="BTC/USDT:USDT", timeframe="1m", frame=frame(count)
            )
            assert len(one.events) == count * 2
            assert one.data_fingerprint == two.data_fingerprint
        add("G15_LEGACY_COMPAT", f"bars={count}", check)

    # 301-320: compact telemetry remains internally consistent.
    for count in range(2, 22):
        def check(count=count) -> None:
            chunk_bars = 4
            _, result = compact_run(frame(count), chunk_bars=chunk_bars)
            assert result.report["processed_bar_count"] == count
            assert result.report["processed_input_event_count"] == count * 2
            assert result.report["processed_chunk_count"] == (count + 3) // 4
            assert result.report["retained_snapshot_count"] == len(result.snapshots)
        add("G16_TELEMETRY", f"bars={count}", check)

    strategy_source = (
        ROOT / "config_examples/strategies/HedgeIndicatorMtfMemoryEfficient.py"
    ).read_text(encoding="utf-8")
    strategy_patterns = [
        "startup_candle_count = 80",
        "astype(\"float32\"",
        "DataFrame(",
        "mtf_trend",
        "mtf_rsi",
        "drop(columns=drop_columns, inplace=True)",
        "hedge_long_score",
        "hedge_short_score",
        "hedge_target_net_ratio",
        "hedge_confidence",
        "hedge_risk_scale",
        "hedge_long_exposure_scale",
        "hedge_short_exposure_scale",
        "hedge_allow_new_risk",
        "enter_long",
        "enter_short",
        "exit_long",
        "exit_short",
        "_configured_informatives",
        "merge_informative_pair",
    ]
    for pattern in strategy_patterns:
        add(
            "G17_STRATEGY_STATIC",
            pattern,
            lambda pattern=pattern: (
                None
                if pattern in strategy_source
                else (_ for _ in ()).throw(AssertionError(pattern))
            ),
        )

    backtest_source = (ROOT / "freqtrade/optimize/hedge_backtesting.py").read_text(
        encoding="utf-8"
    )
    backtest_patterns = [
        "class HedgeBacktestEventChunks",
        "DEFAULT_STREAM_CHUNK_BARS",
        "_STREAM_FINGERPRINT_VERSION",
        "self._row_values",
        "zip(*arrays, strict=True)",
        "self._hash_event(event)",
        "len(payload).to_bytes(8, \"big\")",
        "yield tuple(chunk)",
        "dataset = compact_stream.dataset()",
        "runner.run_compact(compact_stream)",
        "del analyzed",
        "del data",
        "del frame",
        "if export_events:",
        "events_from_analyzed_dataframe",
        "_fingerprint_events(events)",
        "for item in fields(value)",
        "replay_mode",
        "retained_snapshot_count",
        "retained_event_count",
    ]
    for pattern in backtest_patterns:
        add(
            "G18_BACKTEST_STATIC",
            pattern,
            lambda pattern=pattern: (
                None
                if pattern in backtest_source
                else (_ for _ in ()).throw(AssertionError(pattern))
            ),
        )

    replay_source = (ROOT / "freqtrade/hedge/simulation/replay.py").read_text(
        encoding="utf-8"
    )
    hyperopt_source = (ROOT / "freqtrade/hedge/native/hyperopt.py").read_text(
        encoding="utf-8"
    )
    contract_patterns = [
        (replay_source, "def replay_ordered_chunks("),
        (replay_source, "self._processed_slots.clear()"),
        (replay_source, "COMPACT_ORDERED_CHUNKS_V1"),
        (replay_source, "FillEvent"),
        (replay_source, "LiquidationEvent"),
        (replay_source, "max_chunk_input_events"),
        (replay_source, "processed_input_event_count"),
        (replay_source, "processed_bar_count"),
        (replay_source, "retain_material_events"),
        (replay_source, "retain_chunk_snapshots"),
        (hyperopt_source, "unstuck_trigger_gross_exposure"),
        (hyperopt_source, "paper[name] = serialized"),
        (hyperopt_source, 'hedge["paper"] = paper'),
        (hyperopt_source, "max_fill_ratio_per_order"),
        (hyperopt_source, "HedgeHyperoptSpace"),
        (hyperopt_source, "max_gross_wallet_exposure"),
        (hyperopt_source, "take_profit_spacing"),
        (hyperopt_source, "grid_spacing"),
        (hyperopt_source, "core_wallet_exposure_long"),
        (hyperopt_source, "core_wallet_exposure_short"),
    ]
    for source, pattern in contract_patterns:
        add(
            "G19_REPLAY_HYPEROPT_STATIC",
            pattern,
            lambda source=source, pattern=pattern: (
                None
                if pattern in source
                else (_ for _ in ()).throw(AssertionError(pattern))
            ),
        )

    from freqtrade.hedge.native.hyperopt import HedgeHyperoptSpace
    from freqtrade.hedge.optimization.config_patch import ALLOWED_PARAMETER_PATHS

    dynamic_contracts: list[Callable[[], None]] = []
    names = {space.name for space in HedgeHyperoptSpace().spaces}
    dynamic_contracts.extend(
        [
            lambda: None if "unstuck_trigger_gross_exposure" in names else (_ for _ in ()).throw(AssertionError()),
            lambda: None if "unstuck_threshold" not in names else (_ for _ in ()).throw(AssertionError()),
            lambda: None if "hedge.paper.long_signal" in ALLOWED_PARAMETER_PATHS else (_ for _ in ()).throw(AssertionError()),
            lambda: None if "hedge.paper.short_signal" in ALLOWED_PARAMETER_PATHS else (_ for _ in ()).throw(AssertionError()),
            lambda: None if "hedge.paper.default_long_signal" not in ALLOWED_PARAMETER_PATHS else (_ for _ in ()).throw(AssertionError()),
            lambda: None if "hedge.paper.default_short_signal" not in ALLOWED_PARAMETER_PATHS else (_ for _ in ()).throw(AssertionError()),
        ]
    )
    for value in ("0.05", "0.10", "0.15", "0.20", "0.25", "0.30", "0.35"):
        def contract(value=value) -> None:
            patched = HedgeHyperoptSpace.apply(
                {"hedge": {"planner": {}, "paper": {}}},
                {
                    "unstuck_trigger_gross_exposure": Decimal(value),
                    "max_fill_ratio_per_order": Decimal(value),
                },
            )
            assert patched["hedge"]["planner"]["unstuck_trigger_gross_exposure"] == value
            assert patched["hedge"]["paper"]["max_fill_ratio_per_order"] == value
            assert "max_fill_ratio_per_order" not in patched["hedge"]
        dynamic_contracts.append(contract)
    config_source = (ROOT / "freqtrade/hedge/config.py").read_text(encoding="utf-8")
    patch_source = (ROOT / "freqtrade/hedge/optimization/config_patch.py").read_text(
        encoding="utf-8"
    )
    for source, pattern in [
        (config_source, '"optimization"'),
        (config_source, '"unstuck_limit_only"'),
        (patch_source, '"long_signal"'),
        (patch_source, '"short_signal"'),
        (patch_source, "ALLOWED_PARAMETER_PATHS"),
        (patch_source, "PLANNER_FIELDS"),
        (patch_source, "PAPER_FIELDS"),
    ]:
        dynamic_contracts.append(
            lambda source=source, pattern=pattern: (
                None
                if pattern in source
                else (_ for _ in ()).throw(AssertionError(pattern))
            )
        )
    if len(dynamic_contracts) != 20:
        raise AssertionError(f"G20 must contain 20 checks, got {len(dynamic_contracts)}")
    for index, fn in enumerate(dynamic_contracts, start=1):
        add("G20_CONFIG_CONTRACTS", f"case={index}", fn)

    if len(checks) != 400:
        raise AssertionError(f"validator must contain exactly 400 checks, got {len(checks)}")

    results: list[CheckResult] = []
    for number, (group, case, fn) in enumerate(checks, start=1):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - audit must retain every failure
            results.append(CheckResult(number, group, case, False, repr(exc)))
        else:
            results.append(CheckResult(number, group, case, True, "PASS"))
    return results


def main() -> int:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    results = run()
    payload = {
        "schema": "hedge-memory-optimization-400-v1",
        "total": len(results),
        "passed": sum(item.passed for item in results),
        "failed": sum(not item.passed for item in results),
        "results": [asdict(item) for item in results],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(
        f"HEDGE MEMORY 400: {payload['passed']}/{payload['total']} PASS; "
        f"FAIL={payload['failed']}"
    )
    if payload["failed"]:
        for item in results:
            if not item.passed:
                print(f"FAIL {item.number:03d} {item.group} {item.case}: {item.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
