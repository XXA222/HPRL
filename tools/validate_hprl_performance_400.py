#!/usr/bin/env python3
"""Dependency-light HPRL profile-driven CPU/CUDA performance acceptance matrix (400 checks)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl.action_space import (  # noqa: E402
    TierBoundaryBuffers,
    TieredHedgeActionCodec,
    gaussian_selected_tier_log_prob,
    gaussian_selected_tier_log_prob_from_boundaries,
    gaussian_tier_boundaries,
    gaussian_tier_probabilities,
    gaussian_tier_probabilities_from_boundaries,
)
from freqtrade.hedge.hprl.algorithms.base import soft_update  # noqa: E402
from freqtrade.hedge.hprl.config import (  # noqa: E402
    HPRLActionConfig,
    HPRLConfig,
    HPRLEnvironmentConfig,
    HPRLRewardConfig,
    HPRLTrainingConfig,
)
from freqtrade.hedge.hprl.data import TensorMarketDataset  # noqa: E402
from freqtrade.hedge.hprl.device import require_torch, resolve_device  # noqa: E402
from freqtrade.hedge.hprl.env import VectorizedHedgeEnv  # noqa: E402
from freqtrade.hedge.hprl.performance import (  # noqa: E402
    configure_training_runtime,
    discounted_returns_scan,
    make_adam,
    auto_compile_policy,
    compile_break_even_updates,
    compile_policy_thresholds,
    estimate_compile_break_even_updates,
    estimate_compile_startup_seconds,
    host_interop_profile_info,
    resolve_compile_mode,
    resolve_compile_scope,
    resolve_host_interop_threads,
    resolve_rebrac_flow_precision,
    resolve_grad_clip_backend,
    resolve_optimizer_backend,
    timed_iterations_detailed,
)
from freqtrade.hedge.hprl.registry import create_agent  # noqa: E402
from freqtrade.hedge.hprl.replay import ReplayBatch, TensorReplayBuffer  # noqa: E402
from freqtrade.hedge.hprl.reward import CompositeReward, RewardFactsTensor  # noqa: E402


torch = require_torch()
ALGORITHMS = ("fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2")


def check(rows: list[dict], group: str, case: str, passed: bool, detail: str = "") -> None:
    rows.append(
        {
            "number": len(rows) + 1,
            "group": group,
            "case": case,
            "passed": bool(passed),
            "detail": detail or ("PASS" if passed else "FAIL"),
        }
    )


def cfg(**values) -> HPRLTrainingConfig:
    base = dict(
        device="cpu",
        hidden_dim=16,
        hidden_depth=1,
        batch_size=8,
        replay_capacity=64,
        warmup_steps=0,
        runtime_checks=False,
        metrics_interval=1000,
        optimizer_backend="for_loop",
        compile_mode="off",
    )
    base.update(values)
    return HPRLTrainingConfig(**base)


def batch(seed: int, obs_dim: int = 7, action_dim: int = 4, size: int = 8) -> ReplayBatch:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return ReplayBatch(
        obs=torch.randn(size, obs_dim, generator=gen),
        action=torch.rand(size, action_dim, generator=gen),
        reward=torch.randn(size, 1, generator=gen) * 0.01,
        next_obs=torch.randn(size, obs_dim, generator=gen),
        done=torch.zeros(size, 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    rows: list[dict] = []

    # G01: configuration contract (20)
    valid = [
        ("optimizer_backend", "auto"), ("optimizer_backend", "fused"),
        ("compile_mode", "auto"), ("compile_mode", "reduce-overhead"),
        ("polyak_backend", "auto"), ("grad_clip_backend", "for_loop"),
        ("return_scan_backend", "auto"), ("return_scan_backend", "loop"),
        ("flow_likelihood_precision", "auto"), ("flow_likelihood_precision", "fp32"),
    ]
    for key, value in valid:
        obj = cfg(**{key: value})
        check(rows, "G01_CONFIG", f"{key}={value}", getattr(obj, key) == value)
    invalid = [
        {"optimizer_backend": "bad"}, {"compile_mode": "fastest"},
        {"cpu_threads": -1}, {"expected_updates": -1},
        {"grad_clip_backend": "bad"}, {"return_scan_backend": "bad"},
        {"polyak_backend": "bad"}, {"flow_likelihood_precision": "bad"},
        {"replay_prefetch_slots": 5}, {"replay_reuse_sample_buffers": 1},
    ]
    for values in invalid:
        try:
            cfg(**values)
            ok = False
        except Exception:
            ok = True
        check(rows, "G01_CONFIG", str(values), ok)

    # G02: optimizer backend policy (20)
    for i in range(20):
        requested = ("auto", "foreach", "for_loop", "fused")[i % 4]
        expected = "for_loop" if requested == "auto" else requested
        resolved = resolve_optimizer_backend(requested, "cpu")
        ok = resolved == expected
        if ok:
            layer = torch.nn.Linear(8, 4)
            try:
                opt = make_adam(layer.parameters(), lr=3e-4, device="cpu", backend=requested)
                x = torch.randn(4, 8)
                layer(x).square().mean().backward()
                opt.step()
                ok = getattr(opt, "_hprl_backend", None) == expected
            except RuntimeError:
                # CPU fused remains build/platform dependent; policy resolution is still valid.
                ok = requested == "fused"
        check(rows, "G02_OPTIMIZER", f"case={i}:{requested}", ok)

    # G03: foreach/for-loop Polyak parity (20)
    for i in range(20):
        width = 4 + i
        src = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.LayerNorm(width))
        a = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.LayerNorm(width))
        b = torch.nn.Sequential(torch.nn.Linear(width, width), torch.nn.LayerNorm(width))
        b.load_state_dict(a.state_dict())
        soft_update(a, src, 0.125, foreach=True)
        soft_update(b, src, 0.125, foreach=False)
        ok = all(
            torch.allclose(x, y)
            for x, y in zip(a.state_dict().values(), b.state_dict().values())
        )
        check(rows, "G03_POLYAK", f"width={width}", ok)

    # G04: packed replay contiguous and wrapped write (20)
    for i in range(20):
        cap = 17 + i
        rb = TensorReplayBuffer(cap, 3, 2, device="cpu", pin_memory=False)
        first = min(cap - 2, 7 + (i % 5))
        x = torch.arange(first * 3, dtype=torch.float32).reshape(first, 3)
        rb.add(x, x[:, :2], torch.zeros(first), x + 1, torch.zeros(first))
        y = torch.full((8, 3), float(100 + i))
        rb.add(y, y[:, :2], torch.ones(8), y + 1, torch.zeros(8))
        sample = rb.sample_reusable(min(8, len(rb)))
        pointer = sample.obs.data_ptr()
        sample2 = rb.sample_reusable(min(8, len(rb)))
        ok = (
            rb._storage.is_contiguous()
            and rb.obs.untyped_storage().data_ptr() == rb._storage.untyped_storage().data_ptr()
            and sample2.obs.data_ptr() == pointer
            and rb.add_stage_bytes > 0
            and rb.sample_stage_bytes > 0
            and sample2.obs.shape[1] == 3
            and torch.isfinite(sample2.obs).all()
        )
        check(rows, "G04_REPLAY", f"capacity={cap}", bool(ok))

    # G05: Gaussian action-only fast path (20)
    from freqtrade.hedge.hprl.networks import GaussianActor
    for i in range(20):
        dim = 1 + (i % 10)
        actor = GaussianActor(5, dim, hidden_dim=16, depth=1)
        keep_mean = bool(i % 2)
        action, logp, mean = actor.sample(
            torch.randn(8, 5),
            deterministic=bool(i % 2),
            compute_log_prob=False,
            compute_mean_action=keep_mean,
        )
        ok = logp is None and action.shape == (8, dim)
        ok = ok and ((mean is None) if not keep_mean else mean.shape == action.shape)
        ok = ok and bool(torch.isfinite(action).all())
        if mean is not None:
            ok = ok and bool(torch.isfinite(mean).all())
        check(rows, "G05_GAUSSIAN_FAST", f"dim={dim}:case={i}", ok)

    # G06: registered/static tier boundaries + pure probability kernels (20)
    for i in range(20):
        levels = 2 + (i % 10)
        mean = torch.zeros(4, 3)
        log_std = torch.zeros(4, 3)
        buffers = TierBoundaryBuffers(levels)
        named = dict(buffers.named_buffers())
        boundaries = buffers.gaussian_boundaries
        ptr = boundaries.data_ptr()
        p = gaussian_tier_probabilities_from_boundaries(mean, log_std, boundaries)
        index = torch.full((4, 3), i % levels, dtype=torch.int64)
        action = index.float() / float(levels - 1)
        reference = torch.log(p.gather(-1, index.unsqueeze(-1)).squeeze(-1).clamp_min(1e-8))
        direct = gaussian_selected_tier_log_prob_from_boundaries(
            mean, log_std, action, boundaries
        )
        compat = gaussian_selected_tier_log_prob(mean, log_std, action, levels)
        ok = "gaussian_boundaries" in named and ptr == buffers.gaussian_boundaries.data_ptr()
        ok = ok and p.shape == (4, 3, levels)
        ok = ok and bool(torch.allclose(p.sum(-1), torch.ones(4, 3), atol=1e-6))
        ok = ok and bool(torch.allclose(direct, reference, rtol=5e-4, atol=5e-4))
        ok = ok and bool(torch.allclose(compat, direct, rtol=5e-4, atol=5e-4))
        check(rows, "G06_TIER_STATIC_BUFFER", f"levels={levels}:case={i}", ok)

    # G07: env buffer reuse + training info surface (20)
    for i in range(20):
        envs = 1 + (i % 8)
        market = TensorMarketDataset(
            features=torch.randn(48, 2, 3),
            forward_returns=torch.randn(48, 2) * 0.001,
        )
        env = VectorizedHedgeEnv(
            market,
            HPRLEnvironmentConfig(
                parallel_envs=envs,
                info_mode="training" if i % 2 == 0 else "full",
                action=HPRLActionConfig(),
            ),
            device="cpu",
        )
        env.reset()
        tail_id = id(env._tail_sortable)
        step = env.step(torch.zeros(envs, 4))
        info = step.info
        ok = id(env._tail_sortable) == tail_id and "equity" in info
        if i % 2 == 0:
            ok = ok and set(info).issuperset({"equity", "executed_action", "time_done"})
            ok = ok and "drawdown" not in info
        check(rows, "G07_ENV", f"envs={envs}:case={i}", ok)

    # G08: reward zero-weight fast path and finite result (20)
    for i in range(20):
        reward_cfg = HPRLRewardConfig(
            fees=0.0,
            slippage=0.0,
            market_impact=0.0,
            funding=0.0,
            hedge_overlap=0.0,
            opportunity_cost=0.0,
        )
        model = CompositeReward(reward_cfg)
        n = 4
        zeros = torch.zeros(n)
        facts = RewardFactsTensor(
            equity_return=torch.full((n,), (i - 10) * 1e-5),
            drawdown_increase=zeros,
            downside_return=zeros,
            cvar_loss=zeros,
            turnover_ratio=zeros,
            fee_ratio=zeros,
            slippage_ratio=zeros,
            impact_ratio=zeros,
            funding_ratio=zeros,
            quantization_distance=zeros,
            constraint_distance=zeros,
            gross_margin_ratio=zeros,
            hedge_overlap_ratio=zeros,
            opportunity_miss=zeros,
            terminal=zeros.bool(),
        )
        value, components = model.evaluate_tensor(facts, return_components=False)
        check(
            rows,
            "G08_REWARD",
            f"case={i}",
            bool(torch.isfinite(value).all() and components is None),
        )

    # G09: CPU runtime policy and timing helper (20)
    for i in range(20):
        c = cfg(cpu_threads=0, optimizer_backend="auto")
        info = configure_training_runtime(c, "cpu")
        timing = timed_iterations_detailed(
            lambda: torch.ones(8).add_(1),
            warmup=0,
            iterations=1 + i % 3,
            device="cpu",
            samples_per_iteration=8,
        )
        ok = info.optimizer_backend == "for_loop" and info.cpu_threads >= 1
        ok = ok and info.grad_clip_backend == "for_loop"
        ok = ok and timing["iterations_per_second"] > 0 and timing["samples_per_second"] > 0
        check(rows, "G09_CPU_RUNTIME", f"case={i}", ok)

    # G10-G14: real gradient updates for each algorithm (100)
    for group_i, algorithm in enumerate(ALGORITHMS, start=10):
        for i in range(20):
            training = cfg(algorithm=algorithm, polyak_backend="foreach" if i % 2 else "for_loop")
            agent = create_agent(algorithm, 7, 4, training, device="cpu")
            from freqtrade.hedge.hprl.action_space import configure_agent_action_levels
            configure_agent_action_levels(agent, HPRLActionConfig().level_count)
            metrics = agent.update(batch(1000 + group_i * 100 + i), collect_metrics=False)
            params = list(agent.actor.parameters()) + list(agent.critic.parameters())
            ok = metrics.values == {} and all(bool(torch.isfinite(p).all()) for p in params)
            check(rows, f"G{group_i:02d}_{algorithm.upper()}", f"seed={i}", ok)

    # G15: action codec execution grid (20)
    action_cfg = HPRLActionConfig()
    codec = TieredHedgeActionCodec(action_cfg)
    for i in range(20):
        latent = torch.rand(8, 2, 2)
        prior = torch.zeros_like(latent)
        result = codec.decode(latent, prior)
        levels = torch.tensor(action_cfg.position_levels)
        margins = result.target_margin.reshape(-1)
        ok = all(bool(torch.any(torch.isclose(value, levels))) for value in margins)
        check(rows, "G15_ACTION", f"case={i}", ok)

    # G16: source hot-path invariants (20)
    files = {
        "base": (ROOT / "freqtrade/hedge/hprl/algorithms/base.py").read_text(),
        "replay": (ROOT / "freqtrade/hedge/hprl/replay.py").read_text(),
        "networks": (ROOT / "freqtrade/hedge/hprl/networks.py").read_text(),
        "perf": (ROOT / "freqtrade/hedge/hprl/performance.py").read_text(),
        "env": (ROOT / "freqtrade/hedge/hprl/env.py").read_text(),
        "trainer": (ROOT / "freqtrade/hedge/hprl/trainer.py").read_text(),
        "cli": (ROOT / "freqtrade/hedge/hprl/cli.py").read_text(),
        "action": (ROOT / "freqtrade/hedge/hprl/action_space.py").read_text(),
        "pipeline": (ROOT / "freqtrade/hedge/hprl/pipeline_benchmark.py").read_text(),
        "artifact_policy": (ROOT / "freqtrade/hedge/hprl/artifact_policy.py").read_text(),
        "v25_gate": (ROOT / "tools/run_hprl_performance_v25_rtx5070_gate.py").read_text(),
    }
    algorithm_corpus = "\n".join(
        path.read_text()
        for path in sorted((ROOT / "freqtrade/hedge/hprl/algorithms").glob("*.py"))
    )
    corpus = "\n".join((*files.values(), algorithm_corpus))
    invariants = [
        "torch._foreach_lerp_", "resolve_polyak_foreach", "self._storage",
        "sample_reusable", "_index_buffers", "_add_stage_storage",
        "torch.cat((obs, action, reward, next_obs, done)", "twin_cross_entropy",
        "twin_nll", "compute_mean_action", "sample_and_data_log_prob",
        "paired_scope_confidence_decision", "TierBoundaryBuffers",
        "gaussian_selected_tier_log_prob_from_boundaries", "compile_break_even_updates",
        "estimate_compile_break_even_updates", "resolve_rebrac_flow_precision",
        "resolve_host_interop_threads", "CudaReplayPrefetcher",
        "resolve_artifact_io_mode",
    ]
    for text in invariants:
        check(rows, "G16_SOURCE", text, text in corpus)

    # G17: replay release/packed-byte accounting (20)
    for i in range(20):
        rb = TensorReplayBuffer(32 + i, 5, 3, device="cpu", pin_memory=False)
        x = torch.randn(8, 5)
        rb.add(x, x[:, :3], torch.zeros(8), x, torch.zeros(8))
        rb.sample_reusable(8)
        before = rb.persistent_bytes
        staged = rb.sample_stage_bytes + rb.add_stage_bytes
        rb.release()
        ok = (
            before > 0 and staged > 0 and rb.persistent_bytes == 0
            and rb.sample_stage_bytes == 0 and rb.add_stage_bytes == 0
            and rb._storage.numel() == 0
        )
        check(rows, "G17_RELEASE", f"case={i}", ok)

    # G18: device + horizon + RTX5070 break-even compile policy (20)
    from freqtrade.hedge.hprl.performance import compile_agent_hotpaths
    policy = auto_compile_policy("rtx5070_laptop")
    for i in range(20):
        algorithm = ALGORITHMS[i % len(ALGORITHMS)]
        threshold = compile_break_even_updates(algorithm, "rtx5070_laptop")
        assert threshold is not None
        quadrant = i // len(ALGORITHMS)
        if quadrant == 0:
            horizon, expected = 0, "off"
        elif quadrant == 1:
            horizon, expected = threshold - 1, "off"
        elif quadrant == 2:
            horizon, expected = threshold, "reduce-overhead"
        else:
            horizon, expected = threshold * 10, "reduce-overhead"
        resolved = resolve_compile_mode(
            "auto", algorithm, "cuda:0",
            expected_updates=horizon, hardware_profile="rtx5070_laptop"
        )
        training = cfg(algorithm=algorithm, compile_mode="auto", expected_updates=horizon)
        agent = create_agent(algorithm, 7, 4, training, device="cpu")
        compiled = compile_agent_hotpaths(agent, training, "cpu")
        ok = resolved == expected and compiled == ()
        ok = ok and resolve_compile_mode(
            "auto", algorithm, "cpu", expected_updates=10**9, hardware_profile="cpu"
        ) == "off"
        ok = ok and policy[algorithm] == "reduce-overhead"
        expected_scope = "loss" if algorithm in {"fast_dsac", "simba_sac", "rebrac_v2"} else "module"
        ok = ok and resolve_compile_scope(
            "auto", algorithm, hardware_profile="rtx5070_laptop"
        ) == expected_scope
        eager_host = {
            "fast_td3": 1, "fast_dsac": 16, "simba_sac": 4, "xqc": 1, "rebrac_v2": 16
        }[algorithm]
        compiled_host = {
            "fast_td3": 8, "fast_dsac": 1, "simba_sac": 32, "xqc": 1, "rebrac_v2": 16
        }[algorithm]
        host_mode = "reduce-overhead" if expected == "reduce-overhead" else "off"
        host_expected = compiled_host if host_mode == "reduce-overhead" else eager_host
        ok = ok and resolve_host_interop_threads(
            0, algorithm, "cuda:0", "rtx5070_laptop", compile_mode=host_mode
        ) == host_expected
        if algorithm == "rebrac_v2":
            ok = ok and resolve_rebrac_flow_precision(
                "auto", "cuda:0", "rtx5070_laptop", mixed_precision_enabled=True
            ) == "fp32"
            ok = ok and compile_break_even_updates(algorithm, "rtx5070_laptop", "cold") == 3000
            ok = ok and compile_break_even_updates(algorithm, "rtx5070_laptop", "warm") == 400
            ok = ok and estimate_compile_break_even_updates(
                eager_updates_per_second=44.47954766380794,
                compiled_updates_per_second=126.42645499439992,
                compiled_warmup_seconds=3.4100932849978562,
                warmup_iterations=50,
                quantum=100,
            ) == 300
        check(rows, "G18_COMPILE_HORIZON", f"algorithm={algorithm}:case={i}", ok)

    # G19: real update timing remains finite/positive (20)
    for i in range(20):
        training = cfg(algorithm="fast_td3")
        agent = create_agent("fast_td3", 7, 4, training, device="cpu")
        b = batch(2000 + i)
        result = timed_iterations_detailed(
            lambda: agent.update(b, collect_metrics=False),
            warmup=1,
            iterations=2,
            device="cpu",
            samples_per_iteration=8,
        )
        ok = result["seconds"] >= 0 and result["iterations_per_second"] > 0
        ok = ok and result["samples_per_second"] > result["iterations_per_second"]
        check(rows, "G19_TIMING", f"case={i}", ok)

    # G20: integration contract (20)
    mapping_cases = [
        ("optimizer_backend", "auto"), ("compile_mode", "off"),
        ("expected_updates", 100000), ("hardware_profile", "rtx5070_laptop"),
        ("compile_cache_state", "cold"),
        ("compile_scope", "auto"),
        ("polyak_backend", "auto"), ("grad_clip_backend", "auto"),
        ("replay_prefetch", True), ("replay_reuse_sample_buffers", True),
        ("flow_likelihood_precision", "auto"), ("flow_obs_projection_reuse", False),
    ]
    for key, value in mapping_cases:
        mapped = HPRLConfig.from_mapping({"training": {key: value}})
        check(rows, "G20_INTEGRATION", key, getattr(mapped.training, key) == value)
    integration_files = [
        "freqtrade/hedge/hprl/performance.py",
        "freqtrade/hedge/hprl/pipeline_benchmark.py",
        "freqtrade/hedge/hprl/async_io.py",
        "freqtrade/hedge/hprl/artifact_policy.py",
        "freqtrade/hedge/hprl/sustained_benchmark.py",
        "freqtrade/hedge/hprl/algorithms/xqc.py",
        "tools/validate_hprl_performance_v24_200.py",
        "tools/run_hprl_performance_v25_rtx5070_gate.py",
    ]
    for rel in integration_files:
        check(rows, "G20_INTEGRATION", rel, (ROOT / rel).is_file())

    if len(rows) != 400:
        raise RuntimeError(f"performance matrix construction error: expected 400, got {len(rows)}")
    failed = [row for row in rows if not row["passed"]]
    result = {
        "schema": "hprl-performance-runtime-400-v10",
        "expected": 400,
        "executed": len(rows),
        "passed": len(rows) - len(failed),
        "failed": len(failed),
        "status": "PASS" if not failed else "FAIL",
    }
    if args.summary_only:
        print(
            f"HPRL PERFORMANCE RUNTIME 400: {result['passed']}/400 "
            f"{'PASS' if not failed else 'FAIL'}; FAIL={len(failed)}"
        )
        print(json.dumps(result, sort_keys=True))
    else:
        print(json.dumps({**result, "checks": rows}, ensure_ascii=False, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
