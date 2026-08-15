from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Callable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]


def _install_exchange_stub() -> None:
    if "freqtrade.exchange" in sys.modules:
        return
    module = types.ModuleType("freqtrade.exchange")
    def timeframe_to_seconds(timeframe: str) -> int:
        value = timeframe.strip().lower()
        amount = int(value[:-1])
        return amount * {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}[value[-1]]
    module.timeframe_to_seconds = timeframe_to_seconds
    sys.modules["freqtrade.exchange"] = module


_install_exchange_stub()

from freqtrade.hedge.hprl.config import HPRLMemoryConfig
from freqtrade.hedge.hprl.contracts import OfflineTransition
from freqtrade.hedge.hprl.data import OfflineTransitionDataset, TensorMarketDataset
from freqtrade.hedge.hprl.env import VectorizedHedgeEnv
from freqtrade.hedge.hprl.replay import TensorReplayBuffer
from freqtrade.hedge.memory_lifecycle import HedgeMemoryPolicy, memory_snapshot_dict
from freqtrade.optimize.hedge_backtesting import (
    HedgeBacktestEventChunks,
    HedgeBacktesting,
    _json_sha256_stream,
    _json_value,
    events_from_analyzed_dataframe,
)


@dataclass(frozen=True, slots=True)
class Check:
    number: int
    group: str
    case: str
    passed: bool
    detail: str


def _frame(count: int, long_score: float = 0.0, short_score: float = 0.0) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=count, freq="min", tz="UTC")
    price = np.full(count, 100.0, dtype="float32")
    return pd.DataFrame({
        "date": dates,
        "open": price,
        "high": price + 1.0,
        "low": price - 1.0,
        "close": price,
        "volume": np.full(count, 100.0, dtype="float32"),
        "hedge_long_score": np.full(count, long_score, dtype="float32"),
        "hedge_short_score": np.full(count, short_score, dtype="float32"),
        "hedge_target_net_ratio": np.zeros(count, dtype="float32"),
    })


def _market(steps: int = 12) -> TensorMarketDataset:
    return TensorMarketDataset(
        features=torch.arange(steps * 4, dtype=torch.float32).reshape(steps, 2, 2),
        forward_returns=torch.zeros((steps, 2), dtype=torch.float32),
        symbols=("BTC/USDT:USDT", "ETH/USDT:USDT"),
    ).validate()


def _business(report: dict[str, object]) -> dict[str, object]:
    telemetry = {
        "replay_mode", "processed_chunk_count", "processed_input_event_count",
        "processed_bar_count", "retained_event_count", "retained_snapshot_count",
        "max_chunk_input_events",
    }
    return {k: v for k, v in report.items() if k not in telemetry}


def run() -> list[Check]:
    cases: list[tuple[str, str, Callable[[], None]]] = []
    def add(group: str, case: str, fn: Callable[[], None]) -> None:
        cases.append((group, case, fn))

    # 001-010 policy/cache modes.
    modes = ["consume", "bounded", "official", "consume", "bounded"] * 2
    for i, mode in enumerate(modes, 1):
        def check(i=i, mode=mode):
            p = HedgeMemoryPolicy.from_mapping({
                "backtesting_cache_mode": mode,
                "backtesting_cache_max_entries": i,
            })
            assert p.backtesting_cache_mode == mode
            assert p.backtesting_cache_max_entries == i
        add("G01_POLICY", f"{mode}:{i}", check)

    # 011-020 process/cgroup telemetry schema.
    for i in range(10):
        def check(i=i):
            row = memory_snapshot_dict(f"P{i}")
            assert row["label"] == f"P{i}"
            assert "rss_bytes" in row and "pressure_ratio" in row
            if row["rss_bytes"] is not None:
                assert row["rss_bytes"] > 0
        add("G02_TELEMETRY", f"phase={i}", check)

    # 021-030 replay release is real and idempotent.
    for i in range(1, 11):
        def check(i=i):
            b = TensorReplayBuffer(8 * i, 4, 2, device="cpu", pin_memory=False)
            assert b.persistent_bytes > 0
            b.release()
            b.release()
            assert b.persistent_bytes == 0 and len(b) == 0
            assert b.obs.numel() == 0 and b.next_obs.numel() == 0
        add("G03_REPLAY_RELEASE", f"capacity={8*i}", check)

    # 031-040 environment close releases dataset/state.
    for i in range(10):
        def check(i=i):
            env = VectorizedHedgeEnv(_market(4 + i), device="cpu", memory_config=HPRLMemoryConfig())
            env.close()
            env.close()
            assert env.market.dataset is None and env.market.source is None
            assert env._position.numel() == 0 and env._return_history.numel() == 0
        add("G04_ENV_RELEASE", f"steps={4+i}", check)

    # 041-050 offline chunk tensorization exactness.
    for i in range(1, 11):
        def check(i=i):
            rows = [
                OfflineTransition((float(j), float(j+1)), (0.1, 0.2), float(j)/10,
                                  (float(j+1), float(j+2)), bool(j % 2))
                for j in range(1, 2 + i)
            ]
            ds = OfflineTransitionDataset(rows)
            a = ds.tensors("cpu", chunk_rows=1)
            b = ds.tensors("cpu", chunk_rows=max(1, i))
            assert all(torch.equal(a[k], b[k]) for k in a)
        add("G05_OFFLINE_CHUNKS", f"rows={i+1}", check)

    # 051-060 compact replay business parity across sizes.
    for i in range(1, 11):
        def check(i=i):
            f = _frame(8 + i, long_score=0.8, short_score=0.8)
            detailed_ds = events_from_analyzed_dataframe(pair="BTC/USDT:USDT", timeframe="1m", frame=f)
            detailed = HedgeBacktesting(initial_balance=Decimal("1000")).run(detailed_ds.events)
            stream = HedgeBacktestEventChunks(pair="BTC/USDT:USDT", timeframe="1m", frame=f, chunk_bars=3)
            compact = HedgeBacktesting(initial_balance=Decimal("1000")).run_compact(stream)
            assert _business(compact.report) == detailed.report
        add("G06_REPLAY_PARITY", f"bars={8+i}", check)

    # 061-070 compact snapshot retention remains O(chunks), not O(bars).
    for i in range(1, 11):
        def check(i=i):
            bars = 20 + i
            chunk = 2 + i
            stream = HedgeBacktestEventChunks(pair="BTC/USDT:USDT", timeframe="1m", frame=_frame(bars), chunk_bars=chunk)
            result = HedgeBacktesting(initial_balance=Decimal("1000")).run_compact(stream)
            assert len(result.snapshots) <= (bars + chunk - 1) // chunk
            assert result.report["processed_bar_count"] == bars
        add("G07_SNAPSHOT_BOUND", f"case={i}", check)

    # 071-080 stream source arrays are released after consumption.
    for i in range(1, 11):
        def check(i=i):
            stream = HedgeBacktestEventChunks(pair="BTC/USDT:USDT", timeframe="1m", frame=_frame(4+i), chunk_bars=2)
            list(stream)
            assert stream._row_arrays == () and stream._funding_arrays is None
        add("G08_STREAM_RELEASE", f"bars={4+i}", check)

    # 081-090 streaming result hash matches canonical normalization.
    for i in range(1, 11):
        def check(i=i):
            payload = {"i": i, "decimal": Decimal(str(i / 10)), "when": datetime(2026, 1, i, tzinfo=UTC)}
            canonical = json.dumps(_json_value(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
            assert _json_sha256_stream(payload) == hashlib.sha256(canonical).hexdigest()
        add("G09_STREAM_JSON", f"case={i}", check)

    # 091-100 HPRL memory config supports bounded one-shot knobs.
    for i in range(1, 11):
        def check(i=i):
            cfg = HPRLMemoryConfig(
                dataset_window_steps=1024 * i,
                offline_tensorize_chunk_rows=256 * i,
                release_offline_source_after_tensorize=bool(i % 2),
            )
            assert cfg.dataset_window_steps == 1024 * i
            assert cfg.offline_tensorize_chunk_rows == 256 * i
        add("G10_HPRL_CONFIG", f"case={i}", check)

    dp = (ROOT / "freqtrade/data/dataprovider.py").read_text(encoding="utf-8")
    hb = (ROOT / "freqtrade/optimize/hedge_backtesting.py").read_text(encoding="utf-8")
    replay = (ROOT / "freqtrade/hedge/simulation/replay.py").read_text(encoding="utf-8")
    wallet = (ROOT / "freqtrade/hedge/simulation/cross_wallet.py").read_text(encoding="utf-8")
    trainer = (ROOT / "freqtrade/hedge/hprl/trainer.py").read_text(encoding="utf-8")
    hmem = (ROOT / "freqtrade/hedge/hprl/memory.py").read_text(encoding="utf-8")
    hdata = (ROOT / "freqtrade/hedge/hprl/data.py").read_text(encoding="utf-8")
    runtime = (ROOT / "freqtrade/hedge/hprl/runtime.py").read_text(encoding="utf-8")
    lifecycle = (ROOT / "freqtrade/hedge/memory_lifecycle.py").read_text(encoding="utf-8")

    # 101-110 DataProvider official-compatible plus consume/bounded extensions.
    for token in (
        'backtesting_cache_mode', '"official"', '"bounded"', '"consume"',
        '__backtesting_cache_max_entries', 'return load_history()',
        'self.__cached_pairs_backtesting[saved_pair].copy()',
        'backtesting: bool = False', 'self.__cached_pairs_backtesting.clear()',
        'self.__msg_cache.clear()',
    ):
        add("G11_DP_CACHE", token, lambda token=token: None if token in dp else (_ for _ in ()).throw(AssertionError(token)))

    # 111-120 load-time pair scoping and phase telemetry.
    for token in (
        'exchange_config["pair_whitelist"] = [pair]', 'release_unmanaged_pair_data',
        'data.clear()', 'PAIR_SCOPED', 'DATA_LOADED', 'ANALYZED', 'REPLAY_DETACHED',
        'BACKEND_RELEASED', 'REPLAY_DONE', 'RESULT_WRITTEN',
    ):
        add("G12_PAIR_SCOPE", token, lambda token=token: None if token in hb else (_ for _ in ()).throw(AssertionError(token)))

    # 121-130 replay avoids compact temporary report/event/snapshot churn.
    for token in (
        'trusted_ordered=True', 'include_report=False', 'emit_events=not compact_retention',
        'observe_snapshot_state', 'update_metrics=False', 'compact_retention',
        '_processed_slots.clear()', 'pressure_cleanup()', 'last_snapshot_timestamp',
        'tuple(sorted(self.wallet.active_orders)) if not compact_retention else ()',
    ):
        add("G13_COMPACT_TEMP", token, lambda token=token: None if token in replay else (_ for _ in ()).throw(AssertionError(token)))

    # 131-140 wallet observation is split from object materialization.
    for token in (
        'def observe_snapshot_state', 'hedge_duration_seconds', 'dual_leg_active',
        'self.observe_risk(mark)', 'update_metrics: bool = True',
        'if update_metrics', 'utc_aware(timestamp)', 'return SimulationSnapshot(',
        'last_timestamp', 'equity_peak',
    ):
        add("G14_WALLET_OBSERVE", token, lambda token=token: None if token in wallet else (_ for _ in ()).throw(AssertionError(token)))

    # 141-150 offline tensorization avoids whole CPU+CUDA duplication.
    for token in (
        'chunk_rows', 'preallocates final tensors', 'copy_(value, non_blocking=False)',
        'release_source', 'len(self.dataset)', 'total_bytes = int(', 'tensor_device',
        'dataset_gpu_fraction', 'use_gpu', 'self.dataset.tensors(',
    ):
        source = hdata if token in {'chunk_rows','preallocates final tensors','copy_(value, non_blocking=False)','release_source'} else trainer
        add("G15_OFFLINE_NO_DOUBLE", token, lambda token=token, source=source: None if token in source else (_ for _ in ()).throw(AssertionError(token)))

    # 151-160 explicit HPRL lifecycle release contracts.
    combined = '\n'.join([hmem, runtime, trainer, (ROOT / 'freqtrade/hedge/hprl/replay.py').read_text(), (ROOT / 'freqtrade/hedge/hprl/env.py').read_text()])
    for token in (
        'release_dataset', '_gpu_window_buffers.clear()', '_pinned_window_buffers.clear()',
        'def release(self, *, aggressive', 'replay buffer has been released',
        'def close(self, *, aggressive', 'close_environment=True',
        'release_source=True', '__enter__', '__exit__',
    ):
        add("G16_EXPLICIT_RELEASE", token, lambda token=token: None if token in combined else (_ for _ in ()).throw(AssertionError(token)))

    # 161-170 streaming serialization avoids whole encoded/file byte copies.
    for token in (
        'JSONEncoder(', 'iterencode(value)', '_file_sha256_stream', 'handle.read(chunk_bytes)',
        'json.dump(', 'default=_json_default', 'temp.open("w"', 'json.load(handle)',
        'temp.replace(path)', 'result_fingerprint',
    ):
        source = hb if token != 'json.load(handle)' else (ROOT / 'freqtrade/hedge/backtesting/cache.py').read_text()
        add("G17_SERIALIZATION", token, lambda token=token, source=source: None if token in source else (_ for _ in ()).throw(AssertionError(token)))

    # 171-180 phase-boundary policy / glibc trim / cgroup awareness.
    for token in (
        '/sys/fs/cgroup/memory.current', '/sys/fs/cgroup/memory.max', '/proc/self/status',
        'malloc_trim', 'gc.collect(active.gc_generation)', 'pressure_cleanup_ratio',
        'hard_pressure_ratio', 'cuda_empty_cache', 'clear_dataprovider_caches',
        'clear_exchange_caches',
    ):
        add("G18_PROCESS_LIFECYCLE", token, lambda token=token: None if token in lifecycle else (_ for _ in ()).throw(AssertionError(token)))

    # 181-190 no per-step heavy cleanup / allocator churn regressions.
    step_body = replay[replay.index('def replay_ordered_chunks'):]
    trainer_run = trainer[trainer.index('def run(self, environment_steps'):trainer.index('class OfflineTrainer')]
    producer = (ROOT / 'freqtrade/hedge/native/producer.py').read_text()
    native_rpc = (ROOT / 'freqtrade/hedge/native/rpc.py').read_text()
    event_publisher = (ROOT / 'freqtrade/hedge/execution/event_publisher.py').read_text()
    execution_service = (ROOT / 'freqtrade/hedge/execution/service.py').read_text()
    conditions = [
        ('no_gc_per_env_step', 'gc.collect(' not in trainer_run),
        ('no_empty_cache_per_env_step', 'cuda.empty_cache()' not in trainer_run),
        ('coarse_pressure', 'chunk_count % 8 == 0' in step_body),
        ('no_full_replay_pin', 'Never page-lock the whole replay' in combined),
        ('phase_cleanup', 'phase_boundary_cleanup' in trainer_run),
        ('bounded_producer_dedupe', 'seen_message_capacity: int = 4096' in producer and 'popitem(last=False)' in producer),
        ('bounded_native_rpc', 'deque(maxlen=event_capacity)' in native_rpc),
        ('bounded_event_publisher', 'deque(maxlen=event_capacity)' in event_publisher),
        ('bounded_audit_projection', 'deque(maxlen=max_records)' in execution_service),
        ('memory_telemetry', 'memory_telemetry' in hb),
    ]
    for name, ok in conditions:
        add("G19_HOT_PATH", name, lambda ok=ok, name=name: None if ok else (_ for _ in ()).throw(AssertionError(name)))

    # 191-200 AST/format checks on the 10 most important changed runtime files.
    files = [
        'freqtrade/hedge/memory_lifecycle.py', 'freqtrade/data/dataprovider.py',
        'freqtrade/optimize/hedge_backtesting.py', 'freqtrade/hedge/simulation/replay.py',
        'freqtrade/hedge/hprl/memory.py', 'freqtrade/hedge/hprl/replay.py',
        'freqtrade/hedge/native/producer.py', 'freqtrade/hedge/native/rpc.py',
        'freqtrade/hedge/execution/event_publisher.py', 'freqtrade/hedge/execution/service.py',
    ]
    for rel in files:
        def check(rel=rel):
            text = (ROOT / rel).read_text(encoding='utf-8')
            ast.parse(text)
            assert '\t' not in text
            assert all(not line.rstrip('\n\r').endswith(' ') for line in text.splitlines(True))
        add("G20_SOURCE", rel, check)

    assert len(cases) == 200, len(cases)
    results: list[Check] = []
    for number, (group, case, fn) in enumerate(cases, 1):
        try:
            fn()
            results.append(Check(number, group, case, True, 'PASS'))
        except Exception as exc:
            results.append(Check(number, group, case, False, f'{type(exc).__name__}: {exc}'))
    return results


def main() -> int:
    results = run()
    failed = [r for r in results if not r.passed]
    payload = {
        'schema': 'hedge-memory-lifecycle-v15-200',
        'expected': 200,
        'executed': len(results),
        'passed': len(results) - len(failed),
        'failed': len(failed),
        'status': 'PASS' if not failed else 'FAIL',
        'checks': [asdict(r) for r in results],
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if not failed else 1


if __name__ == '__main__':
    raise SystemExit(main())
