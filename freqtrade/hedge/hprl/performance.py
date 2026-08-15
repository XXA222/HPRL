"""CPU/CUDA throughput helpers for HPRL training hot paths."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import time

from .device import require_torch, torch_device


torch = require_torch()
_ASSOCIATIVE_SCAN_VERIFIED: dict[str, bool] = {}


# Compile policy distinguishes true cold-start compilation from disk-cache reuse.
# RTX 5070 thresholds below are frozen from the V2.2 true cold/warm hardware calibration:
# cold disables/isolates compiler caches; warm reuses a seeded disk cache in fresh processes.
# Break-even uses the corrected startup estimator and 1.25 safety margin.
_COMPILE_MIN_UPDATES_BY_CACHE_STATE: dict[str, dict[str, dict[str, int]]] = {
    "rtx5070_laptop": {
        "cold": {
            "fast_td3": 7_500,
            "fast_dsac": 6_000,
            "simba_sac": 3_000,
            "xqc": 6_000,
            "rebrac_v2": 3_000,
        },
        "warm": {
            "fast_td3": 900,
            "fast_dsac": 500,
            "simba_sac": 300,
            "xqc": 1_100,
            "rebrac_v2": 400,
        },
    },
    "generic_cuda": {
        # No hardware-specific warm-cache evidence: warm remains conservative.
        "cold": {
            "fast_td3": 25_000,
            "fast_dsac": 20_000,
            "simba_sac": 20_000,
            "xqc": 5_000,
            "rebrac_v2": 5_000,
        },
        "warm": {
            "fast_td3": 25_000,
            "fast_dsac": 20_000,
            "simba_sac": 20_000,
            "xqc": 5_000,
            "rebrac_v2": 5_000,
        },
    },
}

# V2.2 RTX 5070 confidence-aware profile.  Low-confidence candidates were already
# reverted to the prior stable thread value by the paired calibration policy.
_RTX5070_HOST_INTEROP_PROFILE: dict[str, dict[str, dict[str, float | int | str]]] = {
    "off": {
        "fast_td3": {"threads": 1, "margin_pct": 5.821, "cv_pct": 20.159, "confidence": "low"},
        "fast_dsac": {"threads": 16, "margin_pct": 0.0, "cv_pct": 0.0, "confidence": "low"},
        "simba_sac": {"threads": 4, "margin_pct": 0.0, "cv_pct": 0.0, "confidence": "low"},
        "xqc": {"threads": 1, "margin_pct": 17.240, "cv_pct": 21.824, "confidence": "low"},
        "rebrac_v2": {"threads": 16, "margin_pct": 0.0, "cv_pct": 0.0, "confidence": "low"},
    },
    "reduce-overhead": {
        "fast_td3": {"threads": 8, "margin_pct": 7.502, "cv_pct": 6.666, "confidence": "low"},
        "fast_dsac": {"threads": 1, "margin_pct": 0.0, "cv_pct": 20.698, "confidence": "low"},
        "simba_sac": {"threads": 32, "margin_pct": 0.0, "cv_pct": 0.0, "confidence": "medium"},
        "xqc": {"threads": 1, "margin_pct": 0.0, "cv_pct": 17.262, "confidence": "low"},
        "rebrac_v2": {"threads": 16, "margin_pct": 0.0, "cv_pct": 0.0, "confidence": "low"},
    },
}


def _policy_device(device):
    """Normalize a policy device without requiring CUDA availability on build hosts."""
    try:
        return torch.device(device)
    except (TypeError, RuntimeError):
        return torch_device(device)

def normalize_algorithm(algorithm: str) -> str:
    return str(algorithm).strip().lower().replace("-", "_")


def resolve_hardware_profile(requested: str, device) -> str:
    value = str(requested or "auto").strip().lower()
    resolved = _policy_device(device)
    if value != "auto":
        return value
    if resolved.type != "cuda":
        return "cpu"
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(resolved).lower()
            if "rtx 5070" in name and "laptop" in name:
                return "rtx5070_laptop"
        except Exception:
            pass
    return "generic_cuda"


def normalize_compile_cache_state(value: str | None) -> str:
    state = str(value or "cold").strip().lower()
    if state == "auto":
        # Unknown cache residency must never opt into the optimistic warm threshold.
        return "cold"
    if state not in {"cold", "warm"}:
        raise ValueError("compile cache state must be cold/warm/auto")
    return state


def compile_break_even_updates(
    algorithm: str, hardware_profile: str, cache_state: str = "cold"
) -> int | None:
    profile_name = str(hardware_profile).strip().lower()
    profile = _COMPILE_MIN_UPDATES_BY_CACHE_STATE.get(profile_name)
    if profile is None:
        profile = _COMPILE_MIN_UPDATES_BY_CACHE_STATE["generic_cuda"]
    state = normalize_compile_cache_state(cache_state)
    thresholds = profile.get(state) or profile["cold"]
    return thresholds.get(normalize_algorithm(algorithm))


def compile_policy_thresholds(algorithm: str, hardware_profile: str) -> dict[str, int | None]:
    return {
        "cold": compile_break_even_updates(algorithm, hardware_profile, "cold"),
        "warm": compile_break_even_updates(algorithm, hardware_profile, "warm"),
    }


def auto_compile_policy(
    hardware_profile: str = "generic_cuda", cache_state: str = "cold"
) -> dict[str, str]:
    state = normalize_compile_cache_state(cache_state)
    profile = _COMPILE_MIN_UPDATES_BY_CACHE_STATE.get(
        str(hardware_profile).strip().lower(),
        _COMPILE_MIN_UPDATES_BY_CACHE_STATE["generic_cuda"],
    )
    thresholds = profile.get(state) or profile["cold"]
    return {name: "reduce-overhead" for name in thresholds}


def estimate_compile_startup_seconds(
    *,
    compiled_warmup_seconds: float,
    compiled_updates_per_second: float,
    warmup_iterations: int,
) -> float:
    """Estimate compiler/cache startup cost from a timed warmup interval.

    ``warmup_seconds`` includes both compile/cache lookup and the warmup optimizer updates.
    Subtracting the steady-state time of those updates avoids the V2.1 bias where a faster
    compiled loop made startup appear artificially cheap.
    """
    rate = float(compiled_updates_per_second)
    count = max(0, int(warmup_iterations))
    if rate <= 0.0:
        raise ValueError("compiled_updates_per_second must be positive")
    return max(0.0, float(compiled_warmup_seconds) - (float(count) / rate))


def estimate_compile_break_even_updates(
    *,
    eager_updates_per_second: float,
    compiled_updates_per_second: float,
    eager_warmup_seconds: float = 0.0,
    compiled_warmup_seconds: float,
    warmup_iterations: int | None = None,
    safety_margin: float = 1.25,
    quantum: int = 500,
) -> int | None:
    """Estimate the update horizon where compile startup cost is amortized.

    When ``warmup_iterations`` is supplied, use the corrected V2.2 startup estimator.
    The legacy warmup-delta estimate remains available for callers without iteration metadata.
    """
    eager_rate = float(eager_updates_per_second)
    compiled_rate = float(compiled_updates_per_second)
    if eager_rate <= 0.0 or compiled_rate <= eager_rate:
        return None
    if warmup_iterations is None:
        startup = max(0.0, float(compiled_warmup_seconds) - float(eager_warmup_seconds))
    else:
        startup = estimate_compile_startup_seconds(
            compiled_warmup_seconds=compiled_warmup_seconds,
            compiled_updates_per_second=compiled_rate,
            warmup_iterations=warmup_iterations,
        )
    per_update_gain = (1.0 / eager_rate) - (1.0 / compiled_rate)
    if per_update_gain <= 0.0:
        return None
    raw = startup / per_update_gain
    padded = max(0.0, raw * max(1.0, float(safety_margin)))
    q = max(1, int(quantum))
    return int(math.ceil(padded / q) * q)


def resolve_compile_mode(
    requested: str,
    algorithm: str,
    device,
    *,
    expected_updates: int = 0,
    hardware_profile: str = "auto",
    compile_cache_state: str = "cold",
) -> str:
    value = str(requested).strip().lower()
    if value != "auto":
        return value
    resolved = _policy_device(device)
    if resolved.type != "cuda":
        return "off"
    profile = resolve_hardware_profile(hardware_profile, resolved)
    threshold = compile_break_even_updates(algorithm, profile, compile_cache_state)
    horizon = max(0, int(expected_updates or 0))
    if threshold is None or horizon < threshold:
        return "off"
    return "reduce-overhead"


def resolve_host_interop_threads(
    requested: int,
    algorithm: str,
    device,
    hardware_profile: str = "auto",
    *,
    compile_mode: str = "off",
) -> int:
    if int(requested) > 0:
        return int(requested)
    resolved = _policy_device(device)
    current = int(torch.get_num_interop_threads())
    if resolved.type != "cuda":
        return current
    profile = resolve_hardware_profile(hardware_profile, resolved)
    key = normalize_algorithm(algorithm)
    mode = str(compile_mode or "off").strip().lower()
    if profile == "rtx5070_laptop":
        table = _RTX5070_HOST_INTEROP_PROFILE.get(mode)
        if table is None and mode != "off":
            table = _RTX5070_HOST_INTEROP_PROFILE.get("reduce-overhead")
        if table is None:
            table = _RTX5070_HOST_INTEROP_PROFILE["off"]
        tuned = table.get(key)
        if tuned is not None:
            return int(tuned["threads"])
    return current


def host_interop_profile_info(
    algorithm: str, hardware_profile: str, *, compile_mode: str = "off"
) -> dict[str, float | int | str] | None:
    if str(hardware_profile).strip().lower() != "rtx5070_laptop":
        return None
    mode = str(compile_mode or "off").strip().lower()
    table = _RTX5070_HOST_INTEROP_PROFILE.get(mode)
    if table is None and mode != "off":
        table = _RTX5070_HOST_INTEROP_PROFILE.get("reduce-overhead")
    if table is None:
        return None
    value = table.get(normalize_algorithm(algorithm))
    return dict(value) if value is not None else None


def resolve_rebrac_flow_precision(
    requested: str,
    device,
    hardware_profile: str = "auto",
    *,
    mixed_precision_enabled: bool = False,
) -> str:
    """Resolve ReBRAC likelihood precision from explicit policy and hardware evidence.

    RTX 5070 V2.0 acceptance found FP32 + reduce-overhead + separate materially faster
    than the mixed path, while also removing fine-grained dtype copies.  Explicit
    ``mixed`` remains available for research/other GPUs; only ``auto`` changes.
    """
    value = str(requested or "auto").strip().lower()
    if value not in {"auto", "fp32", "mixed"}:
        raise ValueError("ReBRAC flow precision must be auto/fp32/mixed")
    resolved = _policy_device(device)
    if value == "fp32":
        return "fp32"
    if value == "mixed":
        if resolved.type != "cuda" or not mixed_precision_enabled:
            raise ValueError("flow_likelihood_precision='mixed' requires CUDA mixed_precision")
        return "mixed"
    if resolved.type != "cuda" or not mixed_precision_enabled:
        return "fp32"
    profile = resolve_hardware_profile(hardware_profile, resolved)
    if profile == "rtx5070_laptop":
        return "fp32"
    return "mixed"


@dataclass(frozen=True, slots=True)
class RuntimePerformanceInfo:
    device: str
    optimizer_backend: str
    polyak_backend: str
    grad_clip_backend: str
    compile_mode: str
    compile_scope: str
    hardware_profile: str
    compile_cache_state: str
    expected_updates: int
    compile_break_even_updates: int | None
    compile_cold_break_even_updates: int | None
    compile_warm_break_even_updates: int | None
    host_dispatch_confidence: str
    host_dispatch_margin_pct: float
    cpu_threads: int
    cpu_interop_threads: int
    host_dispatch_tuned: bool


def resolve_optimizer_backend(requested: str, device) -> str:
    value = str(requested).strip().lower()
    if value == "auto":
        return "fused" if torch.device(device).type == "cuda" else "for_loop"
    if value not in {"fused", "foreach", "for_loop"}:
        raise ValueError("optimizer backend must be auto/fused/foreach/for_loop")
    if value == "fused" and torch.device(device).type not in {"cuda", "cpu"}:
        raise ValueError("fused Adam requires CPU or CUDA")
    return value



def resolve_polyak_backend(requested: str, device) -> str:
    value = str(requested).strip().lower()
    if value == "auto":
        return "foreach" if torch.device(device).type == "cuda" else "for_loop"
    if value not in {"foreach", "for_loop"}:
        raise ValueError("polyak backend must be auto/foreach/for_loop")
    return value


def resolve_polyak_foreach(requested: str, device) -> bool:
    return resolve_polyak_backend(requested, device) == "foreach"


def resolve_grad_clip_backend(requested: str, device) -> str:
    value = str(requested).strip().lower()
    if value == "auto":
        return "foreach" if torch.device(device).type == "cuda" else "for_loop"
    if value not in {"foreach", "for_loop"}:
        raise ValueError("grad clip backend must be auto/foreach/for_loop")
    return value


def resolve_grad_clip_foreach(requested: str, device) -> bool:
    return resolve_grad_clip_backend(requested, device) == "foreach"


def make_adam(parameters, *, lr: float, device, backend: str = "auto"):
    resolved = resolve_optimizer_backend(backend, device)
    kwargs = {"lr": float(lr)}
    if resolved == "fused":
        kwargs["fused"] = True
    elif resolved == "foreach":
        kwargs["foreach"] = True
    else:
        kwargs["foreach"] = False
        kwargs["fused"] = False
    optimizer = torch.optim.Adam(parameters, **kwargs)
    setattr(optimizer, "_hprl_backend", resolved)
    return optimizer


def configure_cpu_threads(cpu_threads: int, interop_threads: int) -> tuple[int, int]:
    if cpu_threads > 0:
        torch.set_num_threads(int(cpu_threads))
    # PyTorch only allows changing inter-op threads before parallel work begins.  Preserve a
    # working runtime instead of crashing if another subsystem initialized the pool first.
    current_interop = int(torch.get_num_interop_threads())
    if int(interop_threads) != current_interop:
        try:
            torch.set_num_interop_threads(int(interop_threads))
        except RuntimeError:
            pass
    return int(torch.get_num_threads()), int(torch.get_num_interop_threads())


def configure_training_runtime(config, device) -> RuntimePerformanceInfo:
    resolved = torch_device(device)
    before_threads = int(torch.get_num_threads())
    before_interop = int(torch.get_num_interop_threads())
    # Host dispatch still matters for CUDA training.  In particular, a large inter-op pool can
    # add scheduling overhead to the many small Python-dispatched RL kernels.  Apply explicit
    # thread policy on both CPU and CUDA; cpu_threads=0 preserves the current intra-op setting.
    profile = resolve_hardware_profile(
        getattr(config, "hardware_profile", "auto"), resolved
    )
    expected = int(getattr(config, "expected_updates", 0) or 0)
    cache_state = normalize_compile_cache_state(getattr(config, "compile_cache_state", "cold"))
    thresholds = compile_policy_thresholds(config.algorithm, profile)
    threshold = compile_break_even_updates(config.algorithm, profile, cache_state)
    resolved_compile_mode = resolve_compile_mode(
        config.compile_mode,
        config.algorithm,
        resolved,
        expected_updates=expected,
        hardware_profile=profile,
        compile_cache_state=cache_state,
    )
    resolved_compile_scope = resolve_compile_scope(
        getattr(config, "compile_scope", "auto"),
        config.algorithm,
        hardware_profile=profile,
    )
    requested_interop = resolve_host_interop_threads(
        config.cpu_interop_threads,
        config.algorithm,
        resolved,
        profile,
        compile_mode=resolved_compile_mode,
    )
    threads, interop = configure_cpu_threads(config.cpu_threads, requested_interop)
    tuned = (threads != before_threads) or (interop != before_interop) or (
        int(requested_interop) == interop
    )
    host_profile = host_interop_profile_info(
        config.algorithm, profile, compile_mode=resolved_compile_mode
    ) or {}
    return RuntimePerformanceInfo(
        device=str(resolved),
        optimizer_backend=resolve_optimizer_backend(config.optimizer_backend, resolved),
        polyak_backend=resolve_polyak_backend(config.polyak_backend, resolved),
        grad_clip_backend=resolve_grad_clip_backend(config.grad_clip_backend, resolved),
        compile_mode=resolved_compile_mode,
        compile_scope=resolved_compile_scope,
        hardware_profile=profile,
        compile_cache_state=cache_state,
        expected_updates=expected,
        compile_break_even_updates=threshold,
        compile_cold_break_even_updates=thresholds["cold"],
        compile_warm_break_even_updates=thresholds["warm"],
        host_dispatch_confidence=str(host_profile.get("confidence", "unmeasured")),
        host_dispatch_margin_pct=float(host_profile.get("margin_pct", 0.0)),
        cpu_threads=threads,
        cpu_interop_threads=interop,
        host_dispatch_tuned=bool(tuned),
    )


def maybe_compile_callable(fn, config, device):
    requested = str(config.compile_mode).strip().lower()
    mode = resolve_compile_mode(
        requested, config.algorithm, device,
        expected_updates=getattr(config, "expected_updates", 0),
        hardware_profile=getattr(config, "hardware_profile", "auto"),
        compile_cache_state=getattr(config, "compile_cache_state", "cold"),
    )
    if mode == "off":
        return fn
    resolved = torch_device(device)
    try:
        return torch.compile(
            fn,
            backend="inductor",
            mode=mode,
            dynamic=bool(config.compile_dynamic),
            fullgraph=bool(config.compile_fullgraph),
        )
    except Exception:
        # Compilation is a performance policy, never a correctness dependency.
        if requested == "auto":
            return fn
        raise


def maybe_cudagraph_mark_step_begin(agent) -> None:
    """Mark a new training iteration for compiled CUDA-graph hot paths when supported."""
    info = getattr(agent, "performance_info", None)
    mode = str(getattr(info, "compile_mode", "off")).strip().lower()
    device = torch.device(getattr(agent, "device", "cpu"))
    if device.type != "cuda" or mode not in {"auto", "reduce-overhead", "max-autotune"}:
        return
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None) if compiler is not None else None
    if callable(marker):
        marker()


def discounted_returns_scan(reward, done, gamma: float, *, backend: str = "auto"):
    """Compute reverse discounted returns with an associative GPU scan when available.

    The recurrence ``G_t = r_t + gamma * (1-done_t) * G_{t+1}`` can be represented as
    affine maps ``(a, b)`` with associative composition ``(a,b) o (c,d) =
    (a + b*c, b*d)``.  PyTorch 2.13 can parallelize this with associative_scan.  A
    conservative loop fallback preserves compatibility with older Torch builds.
    """
    value = str(backend).strip().lower()
    if value not in {"auto", "associative", "loop"}:
        raise ValueError("return scan backend must be auto/associative/loop")
    reward = reward.reshape(-1)
    done = done.reshape(-1)
    if reward.numel() != done.numel() or reward.numel() < 1:
        raise ValueError("reward/done must be non-empty matching vectors")

    scan = getattr(torch, "associative_scan", None)
    use_scan = callable(scan) and (value == "associative" or (value == "auto" and reward.is_cuda))
    if use_scan:
        a = reward
        b = (1.0 - done) * float(gamma)

        def combine(left, right):
            la, lb = left
            ra, rb = right
            return la + lb * ra, lb * rb

        try:
            scanned_a, _ = scan(
                combine,
                (a, b),
                dim=0,
                reverse=True,
                combine_mode="pointwise",
            )
            key = f"{reward.device}:{reward.dtype}"
            verified = _ASSOCIATIVE_SCAN_VERIFIED.get(key)
            if verified is None:
                probe_reward = torch.tensor(
                    [0.1, -0.2, 0.3, 0.4, -0.1],
                    device=reward.device,
                    dtype=reward.dtype,
                )
                probe_done = torch.tensor(
                    [0.0, 1.0, 0.0, 0.0, 1.0],
                    device=reward.device,
                    dtype=reward.dtype,
                )
                pa = probe_reward
                pb = (1.0 - probe_done) * float(gamma)
                scan_probe, _ = scan(
                    combine,
                    (pa, pb),
                    dim=0,
                    reverse=True,
                    combine_mode="pointwise",
                )
                loop_probe = torch.empty_like(probe_reward)
                running = torch.zeros((), device=reward.device, dtype=reward.dtype)
                for index in range(probe_reward.numel() - 1, -1, -1):
                    running = (
                        probe_reward[index]
                        + float(gamma) * (1.0 - probe_done[index]) * running
                    )
                    loop_probe[index] = running
                verified = bool(torch.allclose(scan_probe, loop_probe, rtol=1e-5, atol=1e-6))
                _ASSOCIATIVE_SCAN_VERIFIED[key] = verified
            if verified:
                return scanned_a
            if value == "associative":
                raise RuntimeError("torch.associative_scan failed HPRL discounted-return parity")
        except Exception:
            if value == "associative":
                raise

    returns = torch.empty_like(reward)
    running = torch.zeros((), device=reward.device, dtype=reward.dtype)
    for index in range(reward.numel() - 1, -1, -1):
        running = reward[index] + float(gamma) * (1.0 - done[index]) * running
        returns[index] = running
    return returns



def condition_cuda_device(
    device, *, milliseconds: int = 0, matrix_size: int = 512
) -> dict[str, float | int | bool]:
    """Warm GPU clocks with bounded GEMM work before a benchmark measurement.

    This is intentionally outside the timed benchmark interval.  It does not alter model
    parameters or compiler cache state; it only reduces idle->boost clock ramp variance that
    is material for the small HPRL MLP update kernels on laptop GPUs.
    """
    resolved = torch_device(device)
    requested_ms = max(0, int(milliseconds))
    size = max(64, int(matrix_size))
    if resolved.type != "cuda" or requested_ms <= 0:
        return {
            "enabled": False,
            "requested_milliseconds": requested_ms,
            "matrix_size": size,
            "iterations": 0,
            "seconds": 0.0,
        }
    synchronize(resolved)
    dtype = torch.float16
    left = torch.randn((size, size), device=resolved, dtype=dtype)
    right = torch.randn((size, size), device=resolved, dtype=dtype)
    out = torch.empty((size, size), device=resolved, dtype=dtype)
    target = float(requested_ms) / 1000.0
    started = time.perf_counter()
    iterations = 0
    with torch.no_grad():
        while True:
            torch.mm(left, right, out=out)
            left, out = out, left
            iterations += 1
            if iterations % 8 == 0:
                synchronize(resolved)
                if time.perf_counter() - started >= target:
                    break
    synchronize(resolved)
    elapsed = time.perf_counter() - started
    return {
        "enabled": True,
        "requested_milliseconds": requested_ms,
        "matrix_size": size,
        "iterations": iterations,
        "seconds": elapsed,
    }

def agent_finite_state(agent) -> dict[str, object]:
    """Validate finite trainable/target module state outside timed benchmark regions."""
    nonfinite: list[str] = []
    checked = 0
    seen: set[int] = set()
    for module_name in ("actor", "critic", "actor_target", "critic_target"):
        module = getattr(agent, module_name, None)
        if module is None or not hasattr(module, "parameters"):
            continue
        for index, tensor in enumerate(module.parameters()):
            identity = id(tensor)
            if identity in seen:
                continue
            seen.add(identity)
            checked += 1
            if not bool(torch.isfinite(tensor.detach()).all().item()):
                nonfinite.append(f"{module_name}.parameter[{index}]")
        for index, tensor in enumerate(module.buffers()):
            if not (torch.is_floating_point(tensor) or torch.is_complex(tensor)):
                continue
            identity = id(tensor)
            if identity in seen:
                continue
            seen.add(identity)
            checked += 1
            if not bool(torch.isfinite(tensor.detach()).all().item()):
                nonfinite.append(f"{module_name}.buffer[{index}]")
    for name in ("log_alpha",):
        tensor = getattr(agent, name, None)
        if torch.is_tensor(tensor):
            checked += 1
            if not bool(torch.isfinite(tensor.detach()).all().item()):
                nonfinite.append(name)
    return {
        "checked_tensors": checked,
        "parameters_finite": not nonfinite,
        "nonfinite": nonfinite,
    }


def synchronize(device) -> None:
    resolved = torch_device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)


def prepare_steady_state_agent(agent) -> int:
    """Move staged algorithms to their full update path for throughput measurement only."""
    warmup_updates = int(getattr(agent, "warmup_updates", 0) or 0)
    if warmup_updates > 0 and hasattr(agent, "update_count"):
        current = int(getattr(agent, "update_count", 0) or 0)
        setattr(agent, "update_count", max(current, warmup_updates))
    return warmup_updates


def timed_iterations(fn, *, warmup: int, iterations: int, device) -> dict[str, float]:
    result = timed_iterations_detailed(fn, warmup=warmup, iterations=iterations, device=device)
    return {
        "iterations": result["iterations"],
        "seconds": result["seconds"],
        "iterations_per_second": result["iterations_per_second"],
    }


def timed_iterations_detailed(
    fn, *, warmup: int, iterations: int, device, samples_per_iteration: int = 1
) -> dict[str, float]:
    if warmup < 0 or iterations < 1 or samples_per_iteration < 1:
        raise ValueError("timing warmup/iterations/sample count are invalid")
    synchronize(device)
    warmup_started = time.perf_counter()
    for _ in range(int(warmup)):
        fn()
    synchronize(device)
    warmup_seconds = time.perf_counter() - warmup_started
    started = time.perf_counter()
    for _ in range(int(iterations)):
        fn()
    synchronize(device)
    elapsed = time.perf_counter() - started
    rate = float(iterations) / max(elapsed, 1e-12)
    return {
        "warmup_iterations": int(warmup),
        "warmup_seconds": warmup_seconds,
        "iterations": int(iterations),
        "seconds": elapsed,
        "iterations_per_second": rate,
        "samples_per_second": rate * int(samples_per_iteration),
    }


def profile_iterations(
    fn,
    *,
    device,
    wait: int = 1,
    warmup: int = 1,
    active: int = 3,
    profile_memory: bool = False,
    row_limit: int = 30,
) -> list[dict[str, object]]:
    """Profile a short steady-state window without retaining shapes/stacks by default."""
    resolved = torch_device(device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    sort_key = "self_cpu_time_total"
    if resolved.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        sort_key = "self_cuda_time_total"
    schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1)
    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        profile_memory=bool(profile_memory),
        record_shapes=False,
        with_stack=False,
        acc_events=True,
    ) as prof:
        total = int(wait + warmup + active)
        for _ in range(total):
            fn()
            prof.step()
    events = list(prof.key_averages())
    events.sort(key=lambda item: float(getattr(item, sort_key, 0.0)), reverse=True)
    rows: list[dict[str, object]] = []
    for event in events[: max(1, int(row_limit))]:
        rows.append(
            {
                "name": str(event.key),
                "count": int(event.count),
                "self_cpu_time_us": float(getattr(event, "self_cpu_time_total", 0.0)),
                "self_cuda_time_us": float(getattr(event, "self_cuda_time_total", 0.0)),
                "cpu_memory_bytes": int(getattr(event, "self_cpu_memory_usage", 0)),
                "cuda_memory_bytes": int(getattr(event, "self_device_memory_usage", 0)),
            }
        )
    return rows



def summarize_profile_operations(rows: list[dict[str, object]]) -> dict[str, object]:
    """Aggregate profiler rows into launch/orchestration surfaces."""
    categories = {
        "cuda_kernel_launches": 0,
        "cuda_graph_launches": 0,
        "foreach_ops": 0,
        "copy_ops": 0,
        "dtype_copy_ops": 0,
        "optimizer_steps": 0,
        "compiled_regions": 0,
    }
    interesting: dict[str, dict[str, object]] = {}
    for row in rows:
        name = str(row.get("name", ""))
        count = int(row.get("count", 0) or 0)
        lowered = name.lower()
        if name == "cudaLaunchKernel":
            categories["cuda_kernel_launches"] += count
        if name == "cudaGraphLaunch":
            categories["cuda_graph_launches"] += count
        if "_foreach_" in name or "foreach" in lowered:
            categories["foreach_ops"] += count
        if "copy" in lowered:
            categories["copy_ops"] += count
        if name in {"aten::_to_copy", "aten::copy_"} or "_to_copy" in name:
            categories["dtype_copy_ops"] += count
        if "optimizer.step" in lowered:
            categories["optimizer_steps"] += count
        if "compiledfxgraph" in lowered or "torch-compiled region" in lowered:
            categories["compiled_regions"] += count
        if any(token in lowered for token in (
            "cudalaunchkernel", "cudagraphlaunch", "_foreach_", "optimizer.step",
            "_to_copy", "aten::copy_", "compiledfxgraph", "torch-compiled region"
        )):
            interesting[name] = dict(row)
    return {"categories": categories, "interesting_ops": interesting}

def suggested_cpu_threads() -> tuple[int, ...]:
    logical = max(1, int(os.cpu_count() or 1))
    candidates = {1, 2, 4, 8, max(1, logical // 2), logical}
    return tuple(sorted(value for value in candidates if value <= logical))



_RTX5070_LOSS_SCOPE_ALGORITHMS = frozenset({"fast_dsac", "simba_sac", "rebrac_v2"})


def resolve_compile_scope(
    requested: str,
    algorithm: str,
    *,
    hardware_profile: str = "generic_cuda",
) -> str:
    value = str(requested or "auto").strip().lower()
    if value not in {"auto", "module", "loss", "loss_post", "xqc_fused"}:
        raise ValueError("compile scope must be auto/module/loss/loss_post/xqc_fused")
    if value != "auto":
        return value
    profile = str(hardware_profile or "generic_cuda").strip().lower()
    if (
        profile == "rtx5070_laptop"
        and normalize_algorithm(algorithm) in _RTX5070_LOSS_SCOPE_ALGORITHMS
    ):
        return "loss"
    return "module"

def compile_agent_hotpaths(agent, config, device) -> tuple[str, ...]:
    """Compile the methods actually used by HPRL hot paths, not merely module.forward()."""
    requested = str(config.compile_mode).strip().lower()
    resolved = torch_device(device)
    mode = resolve_compile_mode(
        requested, config.algorithm, resolved,
        expected_updates=getattr(config, "expected_updates", 0),
        hardware_profile=getattr(config, "hardware_profile", "auto"),
        compile_cache_state=getattr(config, "compile_cache_state", "cold"),
    )
    if mode == "off":
        return ()
    compiled: list[str] = []

    def compile_method(owner, method_name: str, label: str) -> None:
        method = getattr(owner, method_name, None)
        if not callable(method):
            return
        try:
            wrapped = torch.compile(
                method,
                backend="inductor",
                mode=mode,
                dynamic=bool(config.compile_dynamic),
                fullgraph=bool(config.compile_fullgraph),
            )
            setattr(owner, method_name, wrapped)
            compiled.append(label)
        except Exception:
            if requested != "auto":
                raise

    profile = resolve_hardware_profile(getattr(config, "hardware_profile", "auto"), resolved)
    scope = resolve_compile_scope(
        getattr(config, "compile_scope", "auto"), config.algorithm, hardware_profile=profile
    )
    if scope in {"loss", "loss_post"}:
        for method_name, label in (
            ("_critic_loss_surface", "update.critic_loss_surface"),
            ("_actor_loss_surface", "update.actor_loss_surface"),
        ):
            compile_method(agent, method_name, label)
        if scope == "loss_post":
            compile_method(agent, "_post_update_surface", "update.post_update_surface")
        if compiled:
            return tuple(compiled)

    if scope == "xqc_fused" and normalize_algorithm(config.algorithm) == "xqc":
        for method_name, label in (
            ("_xqc_target_value_surface", "xqc.target_value_surface"),
            ("_xqc_critic_loss_surface", "xqc.critic_loss_surface"),
            ("_xqc_actor_q_surface", "xqc.actor_q_surface"),
        ):
            compile_method(agent, method_name, label)
        if compiled:
            return tuple(compiled)

    actor = getattr(agent, "actor", None)
    actor_target = getattr(agent, "actor_target", None)
    for name, module in (("actor", actor), ("actor_target", actor_target)):
        if module is None:
            continue
        if callable(getattr(module, "distribution", None)):
            compile_method(module, "distribution", f"{name}.distribution")
        elif callable(getattr(module, "sample", None)) and not callable(
            getattr(module, "forward", None)
        ):
            compile_method(module, "sample", f"{name}.sample")
        elif callable(getattr(module, "sample", None)) and module.__class__.__name__.startswith(
            "ConditionalFlow"
        ):
            compile_method(module, "sample", f"{name}.sample")
            if callable(getattr(module, "log_prob", None)):
                compile_method(module, "log_prob", f"{name}.log_prob")
            if (
                bool(getattr(config, "flow_obs_projection_reuse", False))
                and callable(getattr(module, "sample_and_data_log_prob", None))
            ):
                compile_method(
                    module,
                    "sample_and_data_log_prob",
                    f"{name}.sample_and_data_log_prob",
                )
        else:
            compile_method(module, "forward", f"{name}.forward")

    for name in ("critic", "critic_target"):
        module = getattr(agent, name, None)
        if module is None:
            continue
        if callable(getattr(module, "logits", None)):
            compile_method(module, "logits", f"{name}.logits")
        else:
            compile_method(module, "forward", f"{name}.forward")
        if name == "critic" and callable(getattr(module, "twin_cross_entropy", None)):
            compile_method(module, "twin_cross_entropy", f"{name}.twin_cross_entropy")
        if name == "critic" and callable(getattr(module, "twin_nll", None)):
            compile_method(module, "twin_nll", f"{name}.twin_nll")

    # SAC-family tier entropy/CDF helpers are launch-heavy pointwise surfaces and sit
    # outside the actor module. Compile their bound callables when an agent exposes them.
    for attr, label in (
        ("_tier_entropy_fn", "tier_entropy"),
        ("_selected_tier_log_prob_fn", "selected_tier_log_prob"),
    ):
        compile_method(agent, attr, label)
    return tuple(compiled)

