"""Pure helpers for HPRL hardware calibration and confidence-aware profile selection."""

from __future__ import annotations

import math
import os
from pathlib import Path
import random
import statistics
from typing import Iterable, Mapping, Sequence


def quantile(values: Sequence[float], q: float) -> float:
    data = sorted(float(v) for v in values)
    if not data:
        return 0.0
    if len(data) == 1:
        return data[0]
    q = min(1.0, max(0.0, float(q)))
    pos = q * (len(data) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return data[lo]
    weight = pos - lo
    return data[lo] * (1.0 - weight) + data[hi] * weight


def distribution_summary(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(v) for v in values]
    if not data:
        return {
            "count": 0,
            "median": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "mad": 0.0,
            "cv": 0.0,
        }
    median = statistics.median(data)
    mean = statistics.mean(data)
    mad = statistics.median(abs(value - median) for value in data)
    cv = statistics.stdev(data) / abs(mean) if len(data) > 1 and mean else 0.0
    return {
        "count": len(data),
        "median": median,
        "mean": mean,
        "min": min(data),
        "max": max(data),
        "p10": quantile(data, 0.10),
        "p90": quantile(data, 0.90),
        "mad": mad,
        "cv": cv,
    }


def bootstrap_superiority_probability(
    best: Sequence[float],
    runner_up: Sequence[float],
    *,
    samples: int = 4000,
    seed: int = 42,
) -> float:
    """Probability that a bootstrap median of ``best`` exceeds ``runner_up``."""
    a = [float(v) for v in best]
    b = [float(v) for v in runner_up]
    if not a or not b:
        return 0.0
    rng = random.Random(int(seed))
    wins = 0.0
    total = max(1, int(samples))
    for _ in range(total):
        ma = statistics.median(rng.choice(a) for _ in range(len(a)))
        mb = statistics.median(rng.choice(b) for _ in range(len(b)))
        if ma > mb:
            wins += 1.0
        elif ma == mb:
            wins += 0.5
    return wins / total


def winner_confidence(
    best_rates: Sequence[float],
    runner_rates: Sequence[float],
    *,
    bootstrap_samples: int = 4000,
    seed: int = 42,
) -> dict[str, float | str]:
    best_summary = distribution_summary(best_rates)
    runner_summary = distribution_summary(runner_rates)
    best_median = float(best_summary["median"])
    runner_median = float(runner_summary["median"])
    margin = (
        100.0 * (best_median / runner_median - 1.0)
        if runner_median > 0.0
        else 0.0
    )
    probability = bootstrap_superiority_probability(
        best_rates, runner_rates, samples=bootstrap_samples, seed=seed
    )
    noise_pct = 100.0 * max(float(best_summary["cv"]), float(runner_summary["cv"]))
    if probability >= 0.97 and margin >= max(5.0, 0.75 * noise_pct):
        label = "high"
    elif probability >= 0.90 and margin >= max(2.0, 0.35 * noise_pct):
        label = "medium"
    else:
        label = "low"
    return {
        "label": label,
        "margin_pct": margin,
        "bootstrap_superiority": probability,
        "noise_cv_pct": noise_pct,
    }


def choose_threads_with_confidence(
    points: Sequence[Mapping[str, object]],
    *,
    previous_threads: int | None,
    bootstrap_samples: int = 4000,
    seed: int = 42,
) -> dict[str, object]:
    valid = [p for p in points if p.get("status") == "PASS" and p.get("runs")]
    if not valid:
        return {
            "candidate_winner_threads": None,
            "recommended_threads": previous_threads,
            "confidence": "none",
            "fallback_used": previous_threads is not None,
        }
    ranked = sorted(
        valid,
        key=lambda p: float(p.get("median_updates_per_second", 0.0)),
        reverse=True,
    )
    best = ranked[0]
    winner = int(best["cpu_interop_threads"])
    if len(ranked) == 1:
        confidence = {"label": "single", "margin_pct": 0.0, "bootstrap_superiority": 1.0, "noise_cv_pct": 0.0}
    else:
        runner = ranked[1]
        confidence = winner_confidence(
            [float(r["updates_per_second"]) for r in best["runs"]],
            [float(r["updates_per_second"]) for r in runner["runs"]],
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    label = str(confidence["label"])
    fallback = label == "low" and previous_threads is not None and previous_threads != winner
    recommended = int(previous_threads) if fallback else winner
    return {
        "candidate_winner_threads": winner,
        "recommended_threads": recommended,
        "confidence": label,
        "fallback_used": fallback,
        **confidence,
    }



def mad_inlier_mask(values: Sequence[float], *, z_threshold: float = 3.5) -> tuple[bool, ...]:
    """Return a robust MAD-based inlier mask without discarding small-sample ties.

    The modified-z-score rule is deliberately deterministic so calibration evidence is
    reproducible.  When MAD is zero, points equal to the median are kept and unequal
    points are only rejected when there are at least three exact-median observations.
    """
    data = [float(v) for v in values]
    if not data:
        return ()
    if z_threshold <= 0:
        raise ValueError("z_threshold must be positive")
    center = statistics.median(data)
    deviations = [abs(value - center) for value in data]
    mad = statistics.median(deviations)
    if mad <= 1e-12:
        exact = sum(dev <= 1e-12 for dev in deviations)
        if exact >= 3:
            return tuple(dev <= 1e-12 for dev in deviations)
        return tuple(True for _ in data)
    scale = 0.6744897501960817 / mad
    return tuple((dev * scale) <= float(z_threshold) for dev in deviations)


def robust_distribution_summary(
    values: Iterable[float], *, z_threshold: float = 3.5, trim_fraction: float = 0.10
) -> dict[str, float | int]:
    """Distribution summary with MAD rejection and trimmed/winsorized locations."""
    data = [float(v) for v in values]
    raw = distribution_summary(data)
    if not data:
        return {**raw, "inliers": 0, "outliers": 0, "robust_median": 0.0,
                "trimmed_mean": 0.0, "winsorized_mean": 0.0}
    mask = mad_inlier_mask(data, z_threshold=z_threshold)
    inliers = [v for v, keep in zip(data, mask, strict=True) if keep]
    if not inliers:
        inliers = data
    ordered = sorted(inliers)
    fraction = min(0.25, max(0.0, float(trim_fraction)))
    trim = min(len(ordered) // 2, int(math.floor(len(ordered) * fraction)))
    trimmed = ordered[trim:len(ordered)-trim] if trim else ordered
    if trim and len(ordered) > 2 * trim:
        lower = ordered[trim]
        upper = ordered[-trim-1]
        winsorized = [min(upper, max(lower, v)) for v in ordered]
    else:
        winsorized = ordered
    return {
        **raw,
        "inliers": len(inliers),
        "outliers": len(data) - len(inliers),
        "robust_median": statistics.median(inliers),
        "trimmed_mean": statistics.mean(trimmed),
        "winsorized_mean": statistics.mean(winsorized),
        "robust_cv": (statistics.stdev(inliers) / abs(statistics.mean(inliers))
                      if len(inliers) > 1 and statistics.mean(inliers) else 0.0),
    }


def paired_speedup_summary(
    candidate_rates: Sequence[float], baseline_rates: Sequence[float]
) -> dict[str, float | int]:
    """Summarize matched-round candidate/baseline speedups.

    Pairing removes much of the slow drift caused by laptop GPU power state and host
    scheduling because each ratio compares observations from the same interleaved round.
    """
    if len(candidate_rates) != len(baseline_rates):
        raise ValueError("paired rate vectors must have equal length")
    ratios = [
        float(a) / float(b)
        for a, b in zip(candidate_rates, baseline_rates, strict=True)
        if float(a) > 0.0 and float(b) > 0.0
    ]
    summary = robust_distribution_summary(ratios)
    return {
        "count": int(summary["count"]),
        "median_speedup": float(summary["median"]),
        "robust_median_speedup": float(summary["robust_median"]),
        "p10_speedup": float(summary["p10"]),
        "p90_speedup": float(summary["p90"]),
        "mad": float(summary["mad"]),
        "cv": float(summary["cv"]),
        "outliers": int(summary["outliers"]),
    }


def paired_bootstrap_superiority_probability(
    candidate_rates: Sequence[float],
    baseline_rates: Sequence[float],
    *,
    samples: int = 4000,
    seed: int = 42,
) -> float:
    if len(candidate_rates) != len(baseline_rates):
        raise ValueError("paired rate vectors must have equal length")
    pairs = [(float(a), float(b)) for a, b in zip(candidate_rates, baseline_rates, strict=True)
             if float(a) > 0.0 and float(b) > 0.0]
    if not pairs:
        return 0.0
    rng = random.Random(int(seed))
    wins = 0.0
    total = max(1, int(samples))
    for _ in range(total):
        selected = [rng.choice(pairs) for _ in range(len(pairs))]
        ratio = statistics.median(a / b for a, b in selected)
        if ratio > 1.0:
            wins += 1.0
        elif ratio == 1.0:
            wins += 0.5
    return wins / total


def paired_winner_confidence(
    candidate_rates: Sequence[float],
    baseline_rates: Sequence[float],
    *,
    bootstrap_samples: int = 4000,
    seed: int = 42,
) -> dict[str, float | str | int]:
    summary = paired_speedup_summary(candidate_rates, baseline_rates)
    probability = paired_bootstrap_superiority_probability(
        candidate_rates, baseline_rates, samples=bootstrap_samples, seed=seed
    )
    margin = 100.0 * (float(summary["robust_median_speedup"]) - 1.0)
    paired_noise = 100.0 * float(summary["cv"])
    if probability >= 0.98 and margin >= max(3.0, 0.50 * paired_noise):
        label = "high"
    elif probability >= 0.92 and margin >= max(1.5, 0.25 * paired_noise):
        label = "medium"
    else:
        label = "low"
    return {
        "label": label,
        "paired_margin_pct": margin,
        "paired_bootstrap_superiority": probability,
        "paired_noise_cv_pct": paired_noise,
        **summary,
    }


def balanced_interleaved_orders(
    candidates: Sequence[int], repeats: int, *, seed: int = 42
) -> tuple[tuple[int, ...], ...]:
    """Generate balanced per-round candidate orders to suppress temporal drift."""
    values = tuple(dict.fromkeys(int(v) for v in candidates))
    if not values or any(v <= 0 for v in values):
        raise ValueError("candidates must contain positive integers")
    if int(repeats) < 1:
        raise ValueError("repeats must be positive")
    rng = random.Random(int(seed))
    base = list(values)
    rng.shuffle(base)
    orders: list[tuple[int, ...]] = []
    n = len(base)
    for round_index in range(int(repeats)):
        shift = round_index % n
        order = base[shift:] + base[:shift]
        if (round_index // n) % 2:
            order = list(reversed(order))
        orders.append(tuple(order))
    return tuple(orders)

def compile_cache_environment(
    base_env: Mapping[str, str] | None,
    *,
    cache_state: str,
    cache_dir: str | os.PathLike[str],
) -> dict[str, str]:
    """Build an isolated TorchInductor environment for cold or warm calibration."""
    state = str(cache_state).strip().lower()
    if state not in {"cold", "warm"}:
        raise ValueError("cache_state must be cold or warm")
    env = dict(base_env or os.environ)
    target = str(Path(cache_dir))
    env["TORCHINDUCTOR_CACHE_DIR"] = target
    env["TRITON_CACHE_DIR"] = str(Path(target) / "triton")
    env["TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"] = "0"
    env["TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE"] = "0"
    env["TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE"] = "0"
    if state == "cold":
        env["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
    else:
        env.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
    return env


def paired_scope_confidence_decision(
    candidate_rates: Sequence[float],
    baseline_rates: Sequence[float],
    *,
    min_speedup: float = 1.03,
    bootstrap_threshold: float = 0.95,
    bootstrap_samples: int = 8000,
    seed: int = 42,
) -> dict[str, object]:
    """Hardware-gated scope promotion decision from matched AB/BA measurements.

    A candidate is promoted only when its robust paired median clears the minimum material
    speedup and bootstrap superiority threshold.  This deliberately treats small laptop-GPU
    wins as noise until repeated matched rounds prove otherwise.
    """
    if float(min_speedup) <= 1.0:
        raise ValueError("min_speedup must exceed 1.0")
    if not 0.5 < float(bootstrap_threshold) <= 1.0:
        raise ValueError("bootstrap_threshold must be within (0.5, 1]")
    confidence = paired_winner_confidence(
        candidate_rates, baseline_rates,
        bootstrap_samples=max(1000, int(bootstrap_samples)), seed=int(seed),
    )
    median_speedup = float(confidence["robust_median_speedup"])
    probability = float(confidence["paired_bootstrap_superiority"])
    promote = median_speedup >= float(min_speedup) and probability >= float(bootstrap_threshold)
    return {
        **confidence,
        "min_speedup": float(min_speedup),
        "bootstrap_threshold": float(bootstrap_threshold),
        "promote": bool(promote),
        "recommended": "candidate" if promote else "baseline",
    }
