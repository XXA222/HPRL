"""Process-wide memory lifecycle helpers for Hedge workloads.

The policy follows Freqtrade's own memory-management style: reduce dataframe
width/precision where semantics allow it, bound caches, and perform expensive
collection only at lifecycle boundaries instead of on every candle/step.

Linux/Docker additionally benefits from an optional ``malloc_trim(0)`` phase
boundary call.  Python GC can make objects unreachable, but glibc arenas may
otherwise keep those pages mapped in the process RSS for later reuse.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MIB = 1024**2
GIB = 1024**3

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessMemorySnapshot:
    rss_bytes: int | None
    cgroup_current_bytes: int | None
    cgroup_limit_bytes: int | None
    system_total_bytes: int | None

    @property
    def effective_limit_bytes(self) -> int | None:
        values = [
            value
            for value in (self.cgroup_limit_bytes, self.system_total_bytes)
            if value is not None and value > 0
        ]
        return min(values) if values else None

    @property
    def pressure_ratio(self) -> float | None:
        limit = self.effective_limit_bytes
        if not limit:
            return None
        used_candidates = [
            value
            for value in (self.cgroup_current_bytes, self.rss_bytes)
            if value is not None and value >= 0
        ]
        if not used_candidates:
            return None
        return max(used_candidates) / limit


@dataclass(frozen=True, slots=True)
class HedgeMemoryPolicy:
    """Global Hedge memory policy with conservative defaults."""

    reduce_dataframe_footprint: bool = True
    phase_gc: bool = True
    gc_generation: int = 2
    malloc_trim: bool = True
    malloc_trim_min_rss_bytes: int = 512 * MIB
    malloc_trim_pressure_ratio: float = 0.55
    pressure_cleanup_ratio: float = 0.78
    hard_pressure_ratio: float = 0.90
    clear_backtesting_cache: bool = True
    clear_external_cache: bool = True
    clear_message_cache: bool = True
    clear_exchange_cache: bool = True
    backtesting_cache_mode: str = "consume"
    backtesting_cache_max_entries: int = 1
    release_unmanaged_pair_data: bool = True

    def __post_init__(self) -> None:
        if self.gc_generation not in {0, 1, 2}:
            raise ValueError("gc_generation must be 0, 1 or 2")
        if self.malloc_trim_min_rss_bytes < 0:
            raise ValueError("malloc_trim_min_rss_bytes cannot be negative")
        for name in (
            "malloc_trim_pressure_ratio",
            "pressure_cleanup_ratio",
            "hard_pressure_ratio",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.pressure_cleanup_ratio > self.hard_pressure_ratio:
            raise ValueError("pressure_cleanup_ratio cannot exceed hard_pressure_ratio")
        cache_mode = self.backtesting_cache_mode.strip().lower()
        if cache_mode not in {"official", "bounded", "consume"}:
            raise ValueError("backtesting_cache_mode must be official/bounded/consume")
        if (
            not isinstance(self.backtesting_cache_max_entries, int)
            or isinstance(self.backtesting_cache_max_entries, bool)
            or self.backtesting_cache_max_entries < 1
        ):
            raise ValueError("backtesting_cache_max_entries must be >= 1")

    @classmethod
    def from_mapping(cls, values: object) -> "HedgeMemoryPolicy":
        mapping = values if isinstance(values, dict) else {}
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for name in (
            "reduce_dataframe_footprint",
            "phase_gc",
            "malloc_trim",
            "clear_backtesting_cache",
            "clear_external_cache",
            "clear_message_cache",
            "clear_exchange_cache",
            "release_unmanaged_pair_data",
        ):
            kwargs[name] = _strict_bool(
                mapping.get(name), default=getattr(defaults, name), field=name
            )
        for name in (
            "gc_generation",
            "malloc_trim_min_rss_bytes",
            "backtesting_cache_max_entries",
        ):
            kwargs[name] = int(mapping.get(name, getattr(defaults, name)))
        for name in (
            "malloc_trim_pressure_ratio",
            "pressure_cleanup_ratio",
            "hard_pressure_ratio",
        ):
            kwargs[name] = float(mapping.get(name, getattr(defaults, name)))
        kwargs["backtesting_cache_mode"] = str(
            mapping.get("backtesting_cache_mode", defaults.backtesting_cache_mode)
        )
        return cls(**kwargs)


def _strict_bool(value: object, *, default: bool, field: str) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "on", "1"}:
            return True
        if normalized in {"false", "no", "off", "0"}:
            return False
    raise ValueError(f"{field} must be a boolean")


def _read_int_file(path: str) -> int | None:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        raw = target.read_text(encoding="ascii").strip()
        if not raw or raw == "max":
            return None
        value = int(raw)
        # cgroup v1 sometimes exposes an effectively-unlimited sentinel.
        if value >= 2**60:
            return None
        return value
    except (OSError, ValueError):
        return None


def _rss_bytes() -> int | None:
    target = Path("/proc/self/status")
    if not target.is_file():
        return None
    try:
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _system_total_bytes() -> int | None:
    target = Path("/proc/meminfo")
    if not target.is_file():
        return None
    try:
        for line in target.read_text(encoding="ascii").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def process_memory_snapshot() -> ProcessMemorySnapshot:
    # cgroup v2 first, then v1 fallbacks.
    cgroup_current = _read_int_file("/sys/fs/cgroup/memory.current")
    cgroup_limit = _read_int_file("/sys/fs/cgroup/memory.max")
    if cgroup_current is None:
        cgroup_current = _read_int_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if cgroup_limit is None:
        cgroup_limit = _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    return ProcessMemorySnapshot(
        rss_bytes=_rss_bytes(),
        cgroup_current_bytes=cgroup_current,
        cgroup_limit_bytes=cgroup_limit,
        system_total_bytes=_system_total_bytes(),
    )


def memory_snapshot_dict(label: str) -> dict[str, Any]:
    snapshot = process_memory_snapshot()
    return {
        "label": str(label),
        "rss_bytes": snapshot.rss_bytes,
        "cgroup_current_bytes": snapshot.cgroup_current_bytes,
        "cgroup_limit_bytes": snapshot.cgroup_limit_bytes,
        "system_total_bytes": snapshot.system_total_bytes,
        "pressure_ratio": snapshot.pressure_ratio,
    }


def log_memory_snapshot(label: str) -> dict[str, Any]:
    row = memory_snapshot_dict(label)
    logger.info(
        "Hedge memory phase=%s rss=%s cgroup=%s/%s pressure=%s",
        row["label"],
        row["rss_bytes"],
        row["cgroup_current_bytes"],
        row["cgroup_limit_bytes"],
        row["pressure_ratio"],
    )
    return row


def malloc_trim() -> bool:
    """Ask glibc to return free heap pages to the OS when supported."""

    if os.name != "posix":
        return False
    try:
        libc = ctypes.CDLL(None)
        trim = getattr(libc, "malloc_trim", None)
        if trim is None:
            return False
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return bool(trim(0))
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _should_trim(snapshot: ProcessMemorySnapshot, policy: HedgeMemoryPolicy) -> bool:
    if not policy.malloc_trim:
        return False
    rss = snapshot.rss_bytes or 0
    ratio = snapshot.pressure_ratio
    return rss >= policy.malloc_trim_min_rss_bytes or (
        ratio is not None and ratio >= policy.malloc_trim_pressure_ratio
    )


def phase_boundary_cleanup(
    *,
    policy: HedgeMemoryPolicy | None = None,
    cuda_device: str | object | None = None,
    cuda_empty_cache: bool = False,
) -> dict[str, Any]:
    """Collect dead objects at an explicit phase boundary.

    ``torch.cuda.empty_cache`` is intentionally opt-in.  Steady-state CUDA code
    should reuse caching-allocator blocks rather than emptying them each step.
    """

    active = policy or HedgeMemoryPolicy()
    before = process_memory_snapshot()
    collected = gc.collect(active.gc_generation) if active.phase_gc else 0
    trimmed = malloc_trim() if _should_trim(before, active) else False
    cuda_released = False
    if cuda_empty_cache and cuda_device is not None:
        try:
            import torch

            device = torch.device(cuda_device)
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
                cuda_released = True
        except (ImportError, RuntimeError, TypeError):
            cuda_released = False
    after = process_memory_snapshot()
    return {
        "collected": int(collected),
        "malloc_trim": trimmed,
        "cuda_empty_cache": cuda_released,
        "rss_before": before.rss_bytes,
        "rss_after": after.rss_bytes,
        "pressure_before": before.pressure_ratio,
        "pressure_after": after.pressure_ratio,
    }


def pressure_cleanup(
    *,
    policy: HedgeMemoryPolicy | None = None,
    cuda_device: str | object | None = None,
) -> dict[str, Any] | None:
    """Run a boundary cleanup only when process/cgroup pressure is high."""

    active = policy or HedgeMemoryPolicy()
    snapshot = process_memory_snapshot()
    ratio = snapshot.pressure_ratio
    if ratio is None or ratio < active.pressure_cleanup_ratio:
        return None
    return phase_boundary_cleanup(
        policy=active,
        cuda_device=cuda_device,
        cuda_empty_cache=bool(ratio >= active.hard_pressure_ratio),
    )


def clear_dataprovider_caches(dataprovider: object, policy: HedgeMemoryPolicy | None = None) -> None:
    active = policy or HedgeMemoryPolicy()
    clear = getattr(dataprovider, "clear_cache", None)
    if not callable(clear):
        return
    try:
        clear(
            backtesting=active.clear_backtesting_cache,
            external=active.clear_external_cache,
            messages=active.clear_message_cache,
        )
    except TypeError:
        # Compatibility with an unextended upstream DataProvider.
        clear()


def clear_exchange_caches(exchange: object, policy: HedgeMemoryPolicy | None = None) -> None:
    """Release only well-known volatile exchange caches after data detachment."""

    active = policy or HedgeMemoryPolicy()
    if not active.clear_exchange_cache or exchange is None:
        return
    for name in (
        "_klines",
        "_trades",
        "_pairs_last_refresh_time",
        "_expiring_candle_cache",
    ):
        value = getattr(exchange, name, None)
        clear = getattr(value, "clear", None)
        if callable(clear):
            clear()


def clear_strategy_caches(strategy: object) -> None:
    informative = getattr(strategy, "_ft_informative_cache", None)
    expire = getattr(informative, "expire", None)
    if callable(expire):
        expire()
    grouped = getattr(strategy, "_cached_grouped_trades_per_pair", None)
    clear = getattr(grouped, "clear", None)
    if callable(clear):
        clear()
