"""Capacity and sustained-runtime acceptance for two-year HPRL histories."""

from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence


MIB = 1024**2
GIB = 1024**3


def historical_bars(start: datetime, end: datetime, *, timeframe_minutes: int = 1) -> int:
    """Return exact bar count for a left-closed/right-open chronological interval."""
    if end <= start:
        raise ValueError("end must be after start")
    if not isinstance(timeframe_minutes, int) or isinstance(timeframe_minutes, bool):
        raise ValueError("timeframe_minutes must be an integer")
    if timeframe_minutes < 1:
        raise ValueError("timeframe_minutes must be positive")
    seconds = (end - start).total_seconds()
    step_seconds = timeframe_minutes * 60
    quotient, remainder = divmod(seconds, step_seconds)
    if remainder != 0:
        raise ValueError("history interval must align exactly to the timeframe")
    return int(quotient)


def canonical_two_year_minute_bars() -> int:
    """Bars in [2024-01-01, 2026-01-01) at one-minute resolution."""
    return historical_bars(datetime(2024, 1, 1), datetime(2026, 1, 1), timeframe_minutes=1)


@dataclass(frozen=True, slots=True)
class TwoYearCapacityConfig:
    symbols: int
    features_per_symbol: int
    observation_dim: int
    action_dim: int
    replay_capacity: int = 1_000_000
    dataset_window_steps: int = 16_384
    include_funding: bool = True
    include_available_notional: bool = True
    replay_on_cpu: bool = True
    host_memory_bytes: int = 32 * GIB
    cuda_memory_bytes: int = 8 * GIB
    max_host_fraction: float = 0.70
    max_cuda_fraction: float = 0.70
    model_and_runtime_cuda_bytes: int = 2 * GIB

    def __post_init__(self) -> None:
        integers = (
            self.symbols,
            self.features_per_symbol,
            self.observation_dim,
            self.action_dim,
            self.replay_capacity,
            self.dataset_window_steps,
            self.host_memory_bytes,
            self.cuda_memory_bytes,
            self.model_and_runtime_cuda_bytes,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integers):
            raise ValueError("capacity dimensions and byte budgets must be integers")
        if min(self.symbols, self.features_per_symbol, self.observation_dim, self.action_dim) < 1:
            raise ValueError("capacity dimensions must be positive")
        if self.replay_capacity < 1 or self.dataset_window_steps < 2:
            raise ValueError("replay_capacity/window_steps are invalid")
        if min(self.host_memory_bytes, self.cuda_memory_bytes) < 1:
            raise ValueError("memory budgets must be positive")
        if self.model_and_runtime_cuda_bytes < 0:
            raise ValueError("model_and_runtime_cuda_bytes cannot be negative")
        if not 0 < self.max_host_fraction <= 1 or not 0 < self.max_cuda_fraction <= 1:
            raise ValueError("memory fractions must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class TwoYearCapacityPlan:
    schema: str
    bars: int
    transitions: int
    market_dataset_bytes: int
    replay_bytes: int
    cuda_window_bytes: int
    estimated_host_persistent_bytes: int
    estimated_cuda_persistent_bytes: int
    host_budget_bytes: int
    cuda_budget_bytes: int
    host_headroom_bytes: int
    cuda_headroom_bytes: int
    host_fit: bool
    cuda_fit: bool
    recommended_dataset_mode: str
    recommended_replay_device: str


def replay_bytes(capacity: int, observation_dim: int, action_dim: int) -> int:
    if min(capacity, observation_dim, action_dim) < 1:
        raise ValueError("replay dimensions must be positive")
    floats = observation_dim * 2 + action_dim + 2
    return int(capacity * floats * 4)


def market_dataset_bytes(
    bars: int,
    *,
    symbols: int,
    features_per_symbol: int,
    include_funding: bool,
    include_available_notional: bool,
) -> int:
    if min(bars, symbols, features_per_symbol) < 1:
        raise ValueError("market dataset dimensions must be positive")
    floats_per_bar = symbols * features_per_symbol + symbols
    if include_funding:
        floats_per_bar += symbols
    if include_available_notional:
        floats_per_bar += symbols
    return int(bars * floats_per_bar * 4)


def build_two_year_capacity_plan(
    config: TwoYearCapacityConfig,
    *,
    bars: int | None = None,
) -> TwoYearCapacityPlan:
    resolved_bars = canonical_two_year_minute_bars() if bars is None else int(bars)
    if resolved_bars < 2:
        raise ValueError("history requires at least two bars")
    dataset_size = market_dataset_bytes(
        resolved_bars,
        symbols=config.symbols,
        features_per_symbol=config.features_per_symbol,
        include_funding=config.include_funding,
        include_available_notional=config.include_available_notional,
    )
    replay_size = replay_bytes(config.replay_capacity, config.observation_dim, config.action_dim)
    window_size = market_dataset_bytes(
        min(config.dataset_window_steps, resolved_bars),
        symbols=config.symbols,
        features_per_symbol=config.features_per_symbol,
        include_funding=config.include_funding,
        include_available_notional=config.include_available_notional,
    )

    host_persistent = dataset_size + (replay_size if config.replay_on_cpu else 0)
    cuda_persistent = (
        window_size
        + (0 if config.replay_on_cpu else replay_size)
        + config.model_and_runtime_cuda_bytes
    )
    host_budget = int(config.host_memory_bytes * config.max_host_fraction)
    cuda_budget = int(config.cuda_memory_bytes * config.max_cuda_fraction)
    host_headroom = host_budget - host_persistent
    cuda_headroom = cuda_budget - cuda_persistent
    return TwoYearCapacityPlan(
        schema="hprl-two-year-capacity-plan-v1",
        bars=resolved_bars,
        transitions=resolved_bars - 1,
        market_dataset_bytes=dataset_size,
        replay_bytes=replay_size,
        cuda_window_bytes=window_size,
        estimated_host_persistent_bytes=host_persistent,
        estimated_cuda_persistent_bytes=cuda_persistent,
        host_budget_bytes=host_budget,
        cuda_budget_bytes=cuda_budget,
        host_headroom_bytes=host_headroom,
        cuda_headroom_bytes=cuda_headroom,
        host_fit=host_headroom >= 0,
        cuda_fit=cuda_headroom >= 0,
        recommended_dataset_mode="windowed",
        recommended_replay_device="cpu" if config.replay_on_cpu else "cuda",
    )


@dataclass(frozen=True, slots=True)
class RuntimeScaleSample:
    processed_steps: int
    elapsed_seconds: float
    rss_bytes: int | None = None
    cuda_allocated_bytes: int | None = None
    cuda_reserved_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.processed_steps < 0:
            raise ValueError("processed_steps cannot be negative")
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        for value in (self.rss_bytes, self.cuda_allocated_bytes, self.cuda_reserved_bytes):
            if value is not None and value < 0:
                raise ValueError("memory samples cannot be negative")


@dataclass(frozen=True, slots=True)
class RuntimeScaleGateConfig:
    expected_steps: int = canonical_two_year_minute_bars() - 1
    min_full_run_fraction: float = 0.999
    max_projected_hours: float = 24.0
    max_tail_throughput_drop: float = 0.30
    max_robust_cv: float = 0.30
    max_rss_growth_bytes: int = 2 * GIB
    max_cuda_reserved_growth_bytes: int = 512 * MIB

    def __post_init__(self) -> None:
        if self.expected_steps < 1:
            raise ValueError("expected_steps must be positive")
        if not 0 < self.min_full_run_fraction <= 1:
            raise ValueError("min_full_run_fraction must be in (0, 1]")
        if not math.isfinite(self.max_projected_hours) or self.max_projected_hours <= 0:
            raise ValueError("max_projected_hours must be positive")
        if not 0 <= self.max_tail_throughput_drop < 1:
            raise ValueError("max_tail_throughput_drop must be in [0, 1)")
        if self.max_robust_cv < 0:
            raise ValueError("max_robust_cv cannot be negative")
        if min(self.max_rss_growth_bytes, self.max_cuda_reserved_growth_bytes) < 0:
            raise ValueError("memory growth gates cannot be negative")


@dataclass(frozen=True, slots=True)
class RuntimeScaleReport:
    schema: str
    verdict: str
    reasons: tuple[str, ...]
    processed_steps: int
    expected_steps: int
    observed_fraction: float
    elapsed_seconds: float
    overall_steps_per_second: float
    projected_full_run_seconds: float
    projected_full_run_hours: float
    window_steps_per_second: tuple[float, ...]
    throughput_tail_to_head_ratio: float
    throughput_robust_cv: float
    rss_growth_bytes: int | None
    cuda_reserved_growth_bytes: int | None
    full_history_observed: bool
    throughput_stable: bool
    memory_stable: bool
    runtime_budget_pass: bool


def _window_rates(samples: Sequence[RuntimeScaleSample]) -> tuple[float, ...]:
    result: list[float] = []
    for previous, current in zip(samples, samples[1:]):
        delta_steps = current.processed_steps - previous.processed_steps
        delta_seconds = current.elapsed_seconds - previous.elapsed_seconds
        if delta_steps <= 0 or delta_seconds <= 0:
            continue
        result.append(delta_steps / delta_seconds)
    return tuple(result)


def _robust_cv(values: Sequence[float]) -> float:
    positives = [float(value) for value in values if float(value) > 0]
    if len(positives) < 2:
        return 0.0
    median = statistics.median(positives)
    if median <= 0:
        return math.inf
    mad = statistics.median(abs(value - median) for value in positives)
    return 1.4826 * mad / median


def _growth(samples: Sequence[RuntimeScaleSample], field: str) -> int | None:
    values = [getattr(sample, field) for sample in samples if getattr(sample, field) is not None]
    if len(values) < 2:
        return None
    return int(values[-1]) - int(values[0])


def assess_runtime_scale(
    samples: Sequence[RuntimeScaleSample],
    *,
    config: RuntimeScaleGateConfig | None = None,
) -> RuntimeScaleReport:
    cfg = config or RuntimeScaleGateConfig()
    values = tuple(samples)
    if len(values) < 2:
        raise ValueError("runtime scale assessment needs at least two samples")
    if any(b.processed_steps <= a.processed_steps for a, b in zip(values, values[1:])):
        raise ValueError("processed_steps must increase strictly")
    if any(b.elapsed_seconds <= a.elapsed_seconds for a, b in zip(values, values[1:])):
        raise ValueError("elapsed_seconds must increase strictly")

    final = values[-1]
    elapsed = max(final.elapsed_seconds, 1e-12)
    rate = final.processed_steps / elapsed
    projected_seconds = cfg.expected_steps / rate if rate > 0 else math.inf
    projected_hours = projected_seconds / 3600.0
    observed_fraction = final.processed_steps / cfg.expected_steps
    full_history = observed_fraction >= cfg.min_full_run_fraction

    rates = _window_rates(values)
    edge_count = max(1, len(rates) // 5) if rates else 1
    if rates:
        head = statistics.median(rates[:edge_count])
        tail = statistics.median(rates[-edge_count:])
        tail_to_head = tail / head if head > 0 else 0.0
    else:
        tail_to_head = 0.0
    robust_cv = _robust_cv(rates)
    min_tail_ratio = 1.0 - cfg.max_tail_throughput_drop
    throughput_stable = tail_to_head >= min_tail_ratio and robust_cv <= cfg.max_robust_cv

    rss_growth = _growth(values, "rss_bytes")
    cuda_growth = _growth(values, "cuda_reserved_bytes")
    rss_ok = rss_growth is None or rss_growth <= cfg.max_rss_growth_bytes
    cuda_ok = cuda_growth is None or cuda_growth <= cfg.max_cuda_reserved_growth_bytes
    memory_stable = rss_ok and cuda_ok
    runtime_ok = projected_hours <= cfg.max_projected_hours

    reasons: list[str] = []
    if not full_history:
        reasons.append("full_history_not_observed")
    if tail_to_head < min_tail_ratio:
        reasons.append("throughput_tail_degradation")
    if robust_cv > cfg.max_robust_cv:
        reasons.append("throughput_variance")
    if not rss_ok:
        reasons.append("rss_growth")
    if not cuda_ok:
        reasons.append("cuda_reserved_growth")
    if not runtime_ok:
        reasons.append("projected_runtime_exceeds_budget")

    if full_history and throughput_stable and memory_stable and runtime_ok:
        verdict = "PASS"
    elif not full_history and throughput_stable and memory_stable and runtime_ok:
        verdict = "PROVISIONAL"
    else:
        verdict = "FAIL"

    return RuntimeScaleReport(
        schema="hprl-two-year-runtime-scale-v1",
        verdict=verdict,
        reasons=tuple(reasons),
        processed_steps=final.processed_steps,
        expected_steps=cfg.expected_steps,
        observed_fraction=observed_fraction,
        elapsed_seconds=elapsed,
        overall_steps_per_second=rate,
        projected_full_run_seconds=projected_seconds,
        projected_full_run_hours=projected_hours,
        window_steps_per_second=rates,
        throughput_tail_to_head_ratio=tail_to_head,
        throughput_robust_cv=robust_cv,
        rss_growth_bytes=rss_growth,
        cuda_reserved_growth_bytes=cuda_growth,
        full_history_observed=full_history,
        throughput_stable=throughput_stable,
        memory_stable=memory_stable,
        runtime_budget_pass=runtime_ok,
    )


class RuntimeScaleMonitor:
    """Low-frequency monitor intended for historical training/backtest loops.

    Call ``sample`` at bounded intervals (for example every 16k environment steps). The monitor
    itself performs no per-step synchronization and therefore does not distort the hot path.
    """

    def __init__(self) -> None:
        self._started = time.perf_counter()
        self._samples: list[RuntimeScaleSample] = []

    def sample(
        self,
        processed_steps: int,
        *,
        rss_bytes: int | None = None,
        cuda_allocated_bytes: int | None = None,
        cuda_reserved_bytes: int | None = None,
    ) -> RuntimeScaleSample:
        sample = RuntimeScaleSample(
            processed_steps=int(processed_steps),
            elapsed_seconds=time.perf_counter() - self._started,
            rss_bytes=rss_bytes,
            cuda_allocated_bytes=cuda_allocated_bytes,
            cuda_reserved_bytes=cuda_reserved_bytes,
        )
        if self._samples and processed_steps <= self._samples[-1].processed_steps:
            raise ValueError("processed_steps must increase strictly")
        self._samples.append(sample)
        return sample

    @property
    def samples(self) -> tuple[RuntimeScaleSample, ...]:
        return tuple(self._samples)
