#!/usr/bin/env python3
"""V2.2 cache-state/compiler-policy acceptance: 200 executable checks."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl.calibration import (  # noqa: E402
    bootstrap_superiority_probability,
    choose_threads_with_confidence,
    compile_cache_environment,
    distribution_summary,
    quantile,
    winner_confidence,
)
from freqtrade.hedge.hprl.config import HPRLTrainingConfig  # noqa: E402
from freqtrade.hedge.hprl.performance import (  # noqa: E402
    compile_break_even_updates,
    compile_policy_thresholds,
    configure_training_runtime,
    estimate_compile_break_even_updates,
    estimate_compile_startup_seconds,
    host_interop_profile_info,
    normalize_compile_cache_state,
    resolve_compile_mode,
    resolve_host_interop_threads,
    resolve_rebrac_flow_precision,
)

ALGORITHMS = ("fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2")
COLD = {"fast_td3": 18000, "fast_dsac": 10000, "simba_sac": 10000, "xqc": 3000, "rebrac_v2": 3000}
WARM = {"fast_td3": 800, "fast_dsac": 400, "simba_sac": 300, "xqc": 900, "rebrac_v2": 300}
EAGER_THREADS = {"fast_td3": 1, "fast_dsac": 16, "simba_sac": 4, "xqc": 1, "rebrac_v2": 16}
COMPILED_THREADS = {"fast_td3": 8, "fast_dsac": 1, "simba_sac": 32, "xqc": 1, "rebrac_v2": 16}


def check(rows: list[dict], group: str, case: str, passed: bool, detail: str = "") -> None:
    rows.append({"number": len(rows)+1, "group": group, "case": case, "passed": bool(passed), "detail": detail or ("PASS" if passed else "FAIL")})


def cfg(**kwargs):
    base = dict(device="cpu", batch_size=8, replay_capacity=64, hidden_dim=16, hidden_depth=1, compile_mode="off")
    base.update(kwargs)
    return HPRLTrainingConfig(**base)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    rows: list[dict] = []

    # G01: compile-cache configuration contract (20)
    valid = ["cold", "warm", "auto"]
    for i in range(10):
        value = valid[i % len(valid)]
        c = cfg(compile_cache_state=value)
        check(rows, "G01_CACHE_CONFIG", f"valid:{i}:{value}", c.compile_cache_state == value)
    invalid = ["hot", "disk", "none", "1", "true", "cache", "coldish", "warmish", "disabled", "unknown"]
    for value in invalid:
        try:
            cfg(compile_cache_state=value)
            ok = False
        except Exception:
            ok = True
        check(rows, "G01_CACHE_CONFIG", f"invalid:{value}", ok)

    # G02: dual RTX5070 thresholds + conservative auto (20)
    for i in range(20):
        alg = ALGORITHMS[i % 5]
        quadrant = i // 5
        thresholds = compile_policy_thresholds(alg, "rtx5070_laptop")
        if quadrant == 0:
            ok = thresholds == {"cold": COLD[alg], "warm": WARM[alg]}
        elif quadrant == 1:
            ok = compile_break_even_updates(alg, "rtx5070_laptop", "cold") == COLD[alg]
        elif quadrant == 2:
            ok = compile_break_even_updates(alg, "rtx5070_laptop", "warm") == WARM[alg]
        else:
            ok = normalize_compile_cache_state("auto") == "cold"
        check(rows, "G02_DUAL_THRESHOLD", f"{alg}:{quadrant}", ok)

    # G03: state-aware auto compile policy (20)
    for i in range(20):
        alg = ALGORITHMS[i % 5]
        quadrant = i // 5
        state = "cold" if quadrant < 2 else "warm"
        threshold = COLD[alg] if state == "cold" else WARM[alg]
        horizon = threshold - 1 if quadrant % 2 == 0 else threshold
        expected = "off" if horizon < threshold else "reduce-overhead"
        resolved = resolve_compile_mode(
            "auto", alg, "cuda:0", expected_updates=horizon,
            hardware_profile="rtx5070_laptop", compile_cache_state=state,
        )
        check(rows, "G03_AUTO_POLICY", f"{alg}:{state}:{horizon}", resolved == expected)

    # G04: corrected startup estimator (20)
    for i in range(20):
        rate = 100.0 + i
        warmup_n = 10 + i
        pure_start = 1.0 + i * 0.05
        total = pure_start + warmup_n / rate
        estimated = estimate_compile_startup_seconds(
            compiled_warmup_seconds=total,
            compiled_updates_per_second=rate,
            warmup_iterations=warmup_n,
        )
        check(rows, "G04_STARTUP", f"case={i}", math.isclose(estimated, pure_start, rel_tol=0.0, abs_tol=1e-12))

    # G05: corrected break-even and quantization (20)
    for i in range(20):
        eager = 50.0 + i
        compiled = eager * 2.0
        warmup_n = 20
        startup = 2.0 + 0.05 * i
        compiled_warmup = startup + warmup_n / compiled
        q = 100 if i % 2 == 0 else 500
        threshold = estimate_compile_break_even_updates(
            eager_updates_per_second=eager,
            compiled_updates_per_second=compiled,
            compiled_warmup_seconds=compiled_warmup,
            warmup_iterations=warmup_n,
            safety_margin=1.25,
            quantum=q,
        )
        raw = startup / ((1.0 / eager) - (1.0 / compiled)) * 1.25
        expected = int(math.ceil(raw / q) * q)
        check(rows, "G05_BREAK_EVEN", f"case={i}:q={q}", threshold == expected)

    # G06: distribution statistics / quantiles (20)
    for i in range(20):
        values = [float(i+j) for j in range(1,6)]
        summary = distribution_summary(values)
        ok = summary["count"] == 5 and summary["median"] == float(i+3)
        ok = ok and summary["p10"] <= summary["median"] <= summary["p90"]
        ok = ok and summary["mad"] >= 0 and summary["cv"] >= 0
        ok = ok and math.isclose(quantile(values, 0.5), float(i+3))
        check(rows, "G06_STATS", f"case={i}", ok)

    # G07: bootstrap winner confidence and conservative fallback (20)
    for i in range(20):
        strong = i % 2 == 0
        best_rates = [120+i, 121+i, 122+i, 123+i, 124+i]
        runner = ([90+i,91+i,92+i,93+i,94+i] if strong else [119+i,121+i,120+i,122+i,118+i])
        confidence = winner_confidence(best_rates, runner, bootstrap_samples=500, seed=100+i)
        points = [
            {"status":"PASS","cpu_interop_threads":8,"median_updates_per_second":float(sorted(best_rates)[2]),"runs":[{"updates_per_second":x} for x in best_rates]},
            {"status":"PASS","cpu_interop_threads":1,"median_updates_per_second":float(sorted(runner)[2]),"runs":[{"updates_per_second":x} for x in runner]},
        ]
        choice = choose_threads_with_confidence(points, previous_threads=1, bootstrap_samples=500, seed=100+i)
        if strong:
            ok = confidence["label"] in {"medium","high"} and choice["recommended_threads"] == 8
        else:
            ok = confidence["label"] == "low" and choice["recommended_threads"] == 1 and choice["fallback_used"]
        check(rows, "G07_CONFIDENCE", f"case={i}:strong={strong}", ok)

    # G08: cold/warm environment isolation (20)
    for i in range(20):
        state = "cold" if i % 2 == 0 else "warm"
        env = compile_cache_environment({"KEEP":"1", "TORCHINDUCTOR_FORCE_DISABLE_CACHES":"1"}, cache_state=state, cache_dir=f"/tmp/hprl-v22-{i}")
        ok = env["KEEP"] == "1" and env["TORCHINDUCTOR_CACHE_DIR"].endswith(str(i))
        ok = ok and env["TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"] == "0"
        ok = ok and env["TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE"] == "0"
        ok = ok and env["TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE"] == "0"
        if state == "cold":
            ok = ok and env.get("TORCHINDUCTOR_FORCE_DISABLE_CACHES") == "1"
        else:
            ok = ok and "TORCHINDUCTOR_FORCE_DISABLE_CACHES" not in env
        check(rows, "G08_CACHE_ENV", f"case={i}:{state}", ok)

    # G09: RTX5070 host profile metadata and mode split (20)
    for i in range(20):
        alg = ALGORITHMS[i % 5]
        compiled = i // 5 in {1,3}
        mode = "reduce-overhead" if compiled else "off"
        expected_threads = COMPILED_THREADS[alg] if compiled else EAGER_THREADS[alg]
        resolved = resolve_host_interop_threads(0, alg, "cuda:0", "rtx5070_laptop", compile_mode=mode)
        meta = host_interop_profile_info(alg, "rtx5070_laptop", compile_mode=mode)
        ok = resolved == expected_threads and meta is not None and int(meta["threads"]) == expected_threads
        ok = ok and str(meta["confidence"]) in {"low","medium","high"}
        check(rows, "G09_HOST_PROFILE", f"{alg}:{mode}:{i}", ok)

    # G10: integration/release contracts (20)
    files = [
        "freqtrade/hedge/hprl/calibration.py",
        "freqtrade/hedge/hprl/performance.py",
        "freqtrade/hedge/hprl/config.py",
        "freqtrade/hedge/hprl/cli.py",
        "tools/run_hprl_performance_v22_rtx5070_calibration.py",
        "tools/validate_hprl_performance_v22_200.py",
        "tools/validate_hprl_performance_400.py",
        "config_examples/hprl.gpu.example.json",
    ]
    for rel in files:
        check(rows, "G10_INTEGRATION", rel, (ROOT/rel).is_file())
    cases = [
        cfg(compile_cache_state="cold").compile_cache_state == "cold",
        cfg(compile_cache_state="warm").compile_cache_state == "warm",
        resolve_rebrac_flow_precision("auto", "cuda:0", "rtx5070_laptop", mixed_precision_enabled=True) == "fp32",
        resolve_rebrac_flow_precision("auto", "cuda:0", "generic_cuda", mixed_precision_enabled=True) == "mixed",
    ]
    for i, ok in enumerate(cases):
        check(rows, "G10_INTEGRATION", f"policy={i}", ok)
    # 8 more runtime/source contracts = 20 total in G10
    source = (ROOT/"freqtrade/hedge/hprl/performance.py").read_text()
    tool = (ROOT/"tools/run_hprl_performance_v22_rtx5070_calibration.py").read_text()
    extras = [
        "_COMPILE_MIN_UPDATES_BY_CACHE_STATE" in source,
        "estimate_compile_startup_seconds" in source,
        "compile_cache_state" in source,
        "TORCHINDUCTOR_FORCE_DISABLE_CACHES" in tool,
        "TORCHINDUCTOR_CACHE_DIR" in tool,
        "bootstrap superiority" in tool,
        "random.Random(args.seed).shuffle" in tool,
        'parser.add_argument("--repeats", type=int, default=5)' in tool,
    ]
    for i, ok in enumerate(extras):
        check(rows, "G10_INTEGRATION", f"source={i}", ok)

    if len(rows) != 200:
        raise RuntimeError(f"V2.2 matrix construction error: expected 200 got {len(rows)}")
    failed = [row for row in rows if not row["passed"]]
    result = {"schema":"hprl-performance-v2.2-runtime-200-v1","expected":200,"executed":len(rows),"passed":len(rows)-len(failed),"failed":len(failed),"status":"PASS" if not failed else "FAIL"}
    if args.summary_only:
        print(f"HPRL PERFORMANCE V2.2 RUNTIME 200: {result['passed']}/200 {'PASS' if not failed else 'FAIL'}; FAIL={len(failed)}")
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps({**result,"checks":rows}, ensure_ascii=False, sort_keys=True))
    return 0 if not failed else 1

if __name__ == "__main__":
    raise SystemExit(main())
