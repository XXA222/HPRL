from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
import tracemalloc
import types

import numpy as np
import pandas as pd


def _install_exchange_stub() -> None:
    if "freqtrade.exchange" in sys.modules:
        return
    module = types.ModuleType("freqtrade.exchange")
    def timeframe_to_seconds(timeframe: str) -> int:
        value = timeframe.strip().lower()
        return int(value[:-1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[value[-1]]
    module.timeframe_to_seconds = timeframe_to_seconds
    sys.modules["freqtrade.exchange"] = module


def _frame(count: int, *, active: bool = False) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=count, freq="min", tz="UTC")
    price = np.full(count, 100.0, dtype="float32")
    score = np.full(count, 0.8 if active else 0.0, dtype="float32")
    return pd.DataFrame({
        "date": dates,
        "open": price,
        "high": price + 1.0,
        "low": price - 1.0,
        "close": price,
        "volume": np.full(count, 100.0, dtype="float32"),
        "hedge_long_score": score,
        "hedge_short_score": score,
        "hedge_target_net_ratio": np.zeros(count, dtype="float32"),
    })


def _measure(fn):
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter()
    value = fn()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, elapsed, peak


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--builder-bars", type=int, default=10_000)
    parser.add_argument("--replay-bars", type=int, default=1_000)
    parser.add_argument("--chunk-bars", type=int, default=128)
    parser.add_argument("--wide-rows", type=int, default=100_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _install_exchange_stub()

    from decimal import Decimal
    from freqtrade.optimize.hedge_backtesting import (
        HedgeBacktestEventChunks,
        HedgeBacktesting,
        events_from_analyzed_dataframe,
    )
    from freqtrade.hedge.memory_lifecycle import memory_snapshot_dict

    builder_frame = _frame(args.builder_bars)
    def legacy_builder():
        ds = events_from_analyzed_dataframe(pair="BTC/USDT:USDT", timeframe="1m", frame=builder_frame)
        return len(ds.events), ds.data_fingerprint
    def compact_builder():
        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT", timeframe="1m", frame=builder_frame, chunk_bars=args.chunk_bars
        )
        seen = sum(len(chunk) for chunk in stream)
        ds = stream.dataset()
        return seen, len(ds.events), ds.data_fingerprint
    lb, lbs, lbp = _measure(legacy_builder)
    cb, cbs, cbp = _measure(compact_builder)

    replay_frame = _frame(args.replay_bars, active=True)
    def full_replay():
        ds = events_from_analyzed_dataframe(pair="BTC/USDT:USDT", timeframe="1m", frame=replay_frame)
        return HedgeBacktesting(initial_balance=Decimal("1000")).run(ds.events)
    def compact_replay():
        stream = HedgeBacktestEventChunks(
            pair="BTC/USDT:USDT", timeframe="1m", frame=replay_frame, chunk_bars=args.chunk_bars
        )
        return HedgeBacktesting(initial_balance=Decimal("1000")).run_compact(stream)
    fr, frs, frp = _measure(full_replay)
    cr, crs, crp = _measure(compact_replay)
    telemetry = {
        "replay_mode", "processed_chunk_count", "processed_input_event_count",
        "processed_bar_count", "retained_event_count", "retained_snapshot_count",
        "max_chunk_input_events",
    }
    report_equal = {k: v for k, v in cr.report.items() if k not in telemetry} == fr.report

    # Exercise Freqtrade's canonical wide-feature downcast when the full runtime
    # dependencies are available (they are present in the target Docker image).
    footprint = {"available": False}
    try:
        from freqtrade.data.converter.converter import reduce_dataframe_footprint
        wide = {"date": pd.date_range("2025-01-01", periods=args.wide_rows, freq="min", tz="UTC")}
        for name in ("open", "high", "low", "close", "volume"):
            wide[name] = np.ones(args.wide_rows, dtype=np.float64)
        for i in range(40):
            wide[f"feature_{i}"] = np.linspace(0.0, 1.0, args.wide_rows, dtype=np.float64)
        for i in range(10):
            wide[f"flag_{i}"] = np.arange(args.wide_rows, dtype=np.int64)
        wide_df = pd.DataFrame(wide)
        before = int(wide_df.memory_usage(deep=True).sum())
        reduced = reduce_dataframe_footprint(wide_df)
        after = int(reduced.memory_usage(deep=True).sum())
        footprint = {
            "available": True,
            "before_bytes": before,
            "after_bytes": after,
            "reduction_ratio": 1.0 - after / before if before else 0.0,
        }
        del wide_df, reduced, wide
        gc.collect()
    except (ImportError, ModuleNotFoundError):
        pass

    payload = {
        "schema": "hedge-memory-v15-benchmark-v1",
        "process": memory_snapshot_dict("BENCHMARK_DONE"),
        "builder": {
            "bars": args.builder_bars,
            "legacy_seconds": lbs,
            "legacy_peak_bytes": lbp,
            "compact_seconds": cbs,
            "compact_peak_bytes": cbp,
            "peak_reduction_ratio": 1.0 - cbp / lbp if lbp else None,
            "fingerprint_equal": lb[1] == cb[2],
            "compact_retained_events": cb[1],
        },
        "replay": {
            "bars": args.replay_bars,
            "full_seconds": frs,
            "full_peak_bytes": frp,
            "compact_seconds": crs,
            "compact_peak_bytes": crp,
            "peak_reduction_ratio": 1.0 - crp / frp if frp else None,
            "business_report_equal": report_equal,
            "full_snapshots": len(fr.snapshots),
            "compact_snapshots": len(cr.snapshots),
        },
        "dataframe_footprint": footprint,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0 if payload["builder"]["fingerprint_equal"] and report_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
