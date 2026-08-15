"""Spawn-safe lease test worker isolated from optional exchange dependencies."""

from __future__ import annotations

import sys
import types
from pathlib import Path


def _bootstrap_concurrency_modules() -> None:
    root = Path(__file__).resolve().parents[3]
    package_paths = (
        ("freqtrade", root / "freqtrade"),
        ("freqtrade.hedge", root / "freqtrade" / "hedge"),
        ("freqtrade.hedge.concurrency", root / "freqtrade" / "hedge" / "concurrency"),
    )
    for name, path in package_paths:
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = [str(path)]
            sys.modules[name] = module


def acquire_worker(database_path: str, owner_id: str, start, release, output) -> None:
    _bootstrap_concurrency_modules()
    from freqtrade.hedge.concurrency.database_lease import SqliteDatabaseLeaseStore
    from freqtrade.hedge.concurrency.single_writer import SingleWriterGuard

    store = SqliteDatabaseLeaseStore(database_path)
    guard = SingleWriterGuard(store, owner_id=owner_id, ttl_ms=3000)
    start.wait(5)
    try:
        lease = guard.acquire()
    except Exception:
        output.put((False, None))
        return
    output.put((True, lease.fencing_token))
    release.wait(5)
    guard.release()
