#!/usr/bin/env python3
"""Executable 200-check V2.4 deep quality matrix (10 domains x 20 checks)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl.action_space import configure_agent_action_levels
from freqtrade.hedge.hprl.algorithms.base import OptimizerStepPlan, soft_update
from freqtrade.hedge.hprl.async_io import AsyncArtifactWriter
from freqtrade.hedge.hprl.checkpoint import load_checkpoint
from freqtrade.hedge.hprl.config import HPRLActionConfig, HPRLTrainingConfig
from freqtrade.hedge.hprl.device import PrecisionManager, require_torch
from freqtrade.hedge.hprl.performance import agent_finite_state, resolve_compile_scope
from freqtrade.hedge.hprl.pipeline_benchmark import benchmark_training_pipeline
from freqtrade.hedge.hprl.registry import create_agent
from freqtrade.hedge.hprl.replay import ReplayBatch, TensorReplayBuffer
from freqtrade.hedge.hprl.stage_profiling import StageRecorder
from freqtrade.hedge.hprl.sustained_benchmark import benchmark_sustained_training

torch = require_torch()
ALGORITHMS = ("fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2")
TARGET = ("fast_dsac", "simba_sac", "rebrac_v2")
checks: list[dict[str, object]] = []


def record(group: str, name: str, fn) -> None:
    index = len(checks) + 1
    try:
        detail = fn()
        checks.append({"index": index, "group": group, "name": name, "status": "PASS", "detail": detail})
    except Exception as exc:
        checks.append({"index": index, "group": group, "name": name, "status": "FAIL", "error": f"{type(exc).__name__}: {exc}"})


def cfg(algorithm: str, *, batch: int = 8) -> HPRLTrainingConfig:
    return HPRLTrainingConfig(
        algorithm=algorithm, device="cpu", replay_device="same", batch_size=batch,
        replay_capacity=max(32, batch * 4), warmup_steps=0, hidden_dim=16,
        hidden_depth=1, compile_mode="off", metrics_interval=1000,
    )


def batch(rows: int = 8) -> ReplayBatch:
    return ReplayBatch(
        obs=torch.randn(rows, 8), action=torch.rand(rows, 4),
        reward=torch.randn(rows, 1) * 0.01, next_obs=torch.randn(rows, 8),
        done=torch.zeros(rows, 1),
    )


# G01: compile-scope production policy (20)
for alg in ALGORITHMS:
    expected = "loss" if alg in TARGET else "module"
    record("G01-scope", f"rtx-auto-{alg}", lambda a=alg, e=expected: (
        resolve_compile_scope("auto", a, hardware_profile="rtx5070_laptop") == e
    ) or (_ for _ in ()).throw(AssertionError("rtx auto scope mismatch")))
for alg in ALGORITHMS:
    record("G01-scope", f"generic-auto-{alg}", lambda a=alg: (
        resolve_compile_scope("auto", a, hardware_profile="generic_cuda") == "module"
    ) or (_ for _ in ()).throw(AssertionError("generic auto must remain module")))
for alg in ALGORITHMS:
    record("G01-scope", f"explicit-loss-{alg}", lambda a=alg: (
        resolve_compile_scope("loss", a, hardware_profile="rtx5070_laptop") == "loss"
    ) or (_ for _ in ()).throw(AssertionError("explicit loss changed")))
for alg in ALGORITHMS:
    record("G01-scope", f"explicit-loss-post-{alg}", lambda a=alg: (
        resolve_compile_scope("loss_post", a, hardware_profile="rtx5070_laptop") == "loss_post"
    ) or (_ for _ in ()).throw(AssertionError("explicit loss_post changed")))

# G02: pre-bound optimizer/clip plan parity (20)
for seed in range(20):
    def check_optimizer(seed=seed):
        torch.manual_seed(1000 + seed)
        left = torch.nn.Linear(4 + seed % 3, 2 + seed % 2)
        right = torch.nn.Linear(4 + seed % 3, 2 + seed % 2)
        right.load_state_dict(left.state_dict())
        x = torch.randn(5 + seed % 4, 4 + seed % 3)
        y = torch.randn(5 + seed % 4, 2 + seed % 2)
        lo = torch.optim.Adam(left.parameters(), lr=1e-3)
        ro = torch.optim.Adam(right.parameters(), lr=1e-3)
        lp, rp = PrecisionManager("cpu"), PrecisionManager("cpu")
        ll, rl = (left(x) - y).square().mean(), (right(x) - y).square().mean()
        n1 = lp.backward_step(ll, lo, tuple(left.parameters()), 10.0)
        n2 = OptimizerStepPlan(rp, ro, tuple(right.parameters()), 10.0).step(rl)
        assert torch.equal(n1, n2)
        assert all(torch.equal(a, b) for a, b in zip(left.parameters(), right.parameters(), strict=True))
        return {"seed": seed, "norm": float(n1)}
    record("G02-optimizer-plan", f"optimizer-parity-{seed:02d}", check_optimizer)

# G03: real algorithm update finite/state contracts (20)
for i in range(20):
    alg = ("fast_dsac", "simba_sac", "rebrac_v2", "xqc")[i % 4]
    def check_update(i=i, alg=alg):
        torch.manual_seed(2000 + i)
        agent = create_agent(alg, 8, 4, cfg(alg), device="cpu")
        configure_agent_action_levels(agent, HPRLActionConfig().level_count)
        if hasattr(agent, "warmup_updates"):
            agent.update_count = agent.warmup_updates
        if alg == "xqc":
            agent.update_count = agent.policy_delay - 1
        metrics = agent.update(batch(), collect_metrics=True)
        finite = agent_finite_state(agent)
        assert metrics.values and finite["parameters_finite"]
        return {"algorithm": alg, "metrics": sorted(metrics.values)}
    record("G03-real-update", f"real-update-{i:02d}-{alg}", check_update)

# G04: post-update target semantics parity (20)
for i in range(20):
    alg = TARGET[i % len(TARGET)]
    def check_post(i=i, alg=alg):
        torch.manual_seed(3000 + i)
        left = create_agent(alg, 8, 4, cfg(alg), device="cpu")
        torch.manual_seed(3000 + i)
        right = create_agent(alg, 8, 4, cfg(alg), device="cpu")
        right.critic.load_state_dict(left.critic.state_dict())
        right.critic_target.load_state_dict(left.critic_target.state_dict())
        with torch.no_grad():
            for p in left.critic.parameters(): p.add_(0.01 * (i + 1))
            for p in right.critic.parameters(): p.add_(0.01 * (i + 1))
        soft_update(left.critic_target, left.critic, left.config.tau, foreach=left._foreach_polyak)
        if hasattr(left, "log_alpha"):
            with torch.no_grad():
                left.log_alpha.clamp_(-20.0, 5.0)
        right._post_update_surface()
        ls, rs = left.critic_target.state_dict(), right.critic_target.state_dict()
        assert ls.keys() == rs.keys() and all(torch.equal(ls[k], rs[k]) for k in ls)
        return {"algorithm": alg, "tau": left.config.tau}
    record("G04-post-update", f"post-update-parity-{i:02d}-{alg}", check_post)

# G05: async artifact consistency/backpressure/flush (20)
for i in range(20):
    alg = ALGORITHMS[i % len(ALGORITHMS)]
    def check_async(i=i, alg=alg):
        torch.manual_seed(4000 + i)
        source = create_agent(alg, 8, 4, cfg(alg), device="cpu")
        target = create_agent(alg, 8, 4, cfg(alg), device="cpu")
        with tempfile.TemporaryDirectory(prefix="hprl-v24-async-") as tmp:
            root = Path(tmp); ckpt = root / "agent.pt"; log = root / "train.jsonl"
            with AsyncArtifactWriter(queue_size=1 + i % 4) as writer:
                writer.submit_metrics(log, {"iteration": i, "algorithm": alg})
                writer.submit_checkpoint(ckpt, source, {"iteration": i, "algorithm": alg})
            st = writer.stats()
            meta = load_checkpoint(ckpt, target)
            assert st.submitted == st.completed == 2 and not st.failed
            assert meta["iteration"] == i and log.stat().st_size > 0 and ckpt.stat().st_size > 0
            return {"algorithm": alg, "queue_high_water": st.queue_high_water}
    record("G05-async-io", f"async-io-{i:02d}-{alg}", check_async)

# G06: XQC production-vs-diagnostic exact parity (20)
for i in range(20):
    def check_xqc(i=i):
        c = cfg("xqc")
        torch.manual_seed(5000 + i); left = create_agent("xqc", 8, 4, c, device="cpu")
        torch.manual_seed(5000 + i); right = create_agent("xqc", 8, 4, c, device="cpu")
        for name in ("actor", "critic", "critic_target"):
            getattr(right, name).load_state_dict(getattr(left, name).state_dict())
        right.actor_opt.load_state_dict(left.actor_opt.state_dict())
        right.critic_opt.load_state_dict(left.critic_opt.state_dict())
        right.alpha_opt.load_state_dict(left.alpha_opt.state_dict())
        right.log_alpha.data.copy_(left.log_alpha.data)
        configure_agent_action_levels(left, HPRLActionConfig().level_count)
        configure_agent_action_levels(right, HPRLActionConfig().level_count)
        # Cycle all policy-delay states, including actor/alpha updates.
        left.update_count = right.update_count = i % left.policy_delay
        torch.manual_seed(5100 + i); b = batch()
        torch.manual_seed(5200 + i); lm = left.update(b, collect_metrics=True)
        torch.manual_seed(5200 + i); rm = right.profile_update_stages(b, StageRecorder("cpu"), collect_metrics=True)
        assert lm.values == rm.values
        for name in ("actor", "critic", "critic_target"):
            ls, rs = getattr(left, name).state_dict(), getattr(right, name).state_dict()
            assert all(torch.equal(ls[k], rs[k]) for k in ls)
        assert torch.equal(left.log_alpha, right.log_alpha)
        return {"delay_state": i % left.policy_delay}
    record("G06-xqc-decompose", f"xqc-parity-{i:02d}", check_xqc)

# G07: replay/staging lifecycle identity (20)
for i in range(20):
    def check_replay(i=i):
        rows = 8 + (i % 4) * 4
        buffer = TensorReplayBuffer(64, 8, 4, device="cpu", pin_memory=False)
        for _ in range(8):
            buffer.add(torch.randn(8, 8), torch.rand(8, 4), torch.randn(8, 1), torch.randn(8, 8), torch.zeros(8, 1))
        buffer.sample_reusable(rows)
        before = buffer.staging_identity()
        for _ in range(10): buffer.sample_reusable(rows)
        after = buffer.staging_identity()
        assert before == after and before["storage"] != 0
        buffer.release(); assert buffer.persistent_bytes == 0
        return {"batch": rows, "stage": before["sample"]}
    record("G07-replay-lifecycle", f"replay-identity-{i:02d}", check_replay)

# G08: end-to-end pipeline async/sync correctness (20)
for i in range(20):
    alg = ALGORITHMS[i % len(ALGORITHMS)]
    def check_pipeline(i=i, alg=alg):
        torch.manual_seed(6000 + i)
        agent = create_agent(alg, 8, 4, cfg(alg), device="cpu")
        configure_agent_action_levels(agent, HPRLActionConfig().level_count)
        result = benchmark_training_pipeline(
            agent, obs_dim=8, action_dim=4, batch_size=8, iterations=2, warmup=1,
            replay_capacity=32, replay_device="same", metrics_interval=1,
            checkpoint_interval=2, diagnostic_iterations=1,
            async_artifacts=bool(i % 2 == 0), artifact_queue_size=1 + i % 3,
        )
        assert result.samples == 16 and result.metrics_events == 2 and result.checkpoints == 1
        assert result.checkpoint_bytes > 0 and result.samples_per_second > 0
        return {"algorithm": alg, "async": result.async_artifacts}
    record("G08-pipeline", f"pipeline-{i:02d}-{alg}", check_pipeline)

# G09: sustained pipeline drift/memory/retention invariants (20)
for i in range(20):
    def check_sustained(i=i):
        c = cfg("fast_td3")
        agent = create_agent("fast_td3", 8, 4, c, device="cpu")
        configure_agent_action_levels(agent, HPRLActionConfig().level_count)
        keep = 1 + (i % 2)
        result = benchmark_sustained_training(
            agent, obs_dim=8, action_dim=4, batch_size=8, iterations=10, warmup=1,
            window_size=5, replay_capacity=32, replay_device="same", metrics_interval=5,
            checkpoint_interval=5, checkpoint_keep_last=keep, artifact_queue_size=1 + i % 3,
        )
        assert result.parameters_finite and result.replay_staging_stable and result.memory_plateau
        assert result.checkpoints_retained <= keep and result.checkpoint_deletions == 2 - result.checkpoints_retained
        return {"keep": keep, "drift": result.throughput_drift_ratio}
    record("G09-sustained", f"sustained-{i:02d}", check_sustained)

# G10: source/CLI/integration/static contracts (20)
source_checks = [
    ("async-module", "freqtrade/hedge/hprl/async_io.py", "class AsyncArtifactWriter"),
    ("stage-profiler", "freqtrade/hedge/hprl/stage_profiling.py", "class StageRecorder"),
    ("sustained-module", "freqtrade/hedge/hprl/sustained_benchmark.py", "benchmark_sustained_training"),
    ("xqc-profiler", "freqtrade/hedge/hprl/algorithms/xqc.py", "profile_update_stages"),
    ("optimizer-plan", "freqtrade/hedge/hprl/algorithms/base.py", "class OptimizerStepPlan"),
    ("fastdsac-post", "freqtrade/hedge/hprl/algorithms/fast_dsac.py", "_post_update_surface"),
    ("simba-post", "freqtrade/hedge/hprl/algorithms/simba_sac.py", "_post_update_surface"),
    ("rebrac-post", "freqtrade/hedge/hprl/algorithms/rebrac_v2.py", "_post_update_surface"),
    ("loss-post-config", "freqtrade/hedge/hprl/config.py", "loss_post"),
    ("rtx-loss-default", "freqtrade/hedge/hprl/performance.py", "_RTX5070_LOSS_SCOPE_ALGORITHMS"),
    ("pipeline-v2", "freqtrade/hedge/hprl/cli.py", "pipeline_efficiency_same_scope"),
    ("module-baseline", "freqtrade/hedge/hprl/cli.py", "end_to_end_speedup_vs_module_baseline"),
    ("xqc-command", "freqtrade/hedge/hprl/cli.py", "perf-xqc-decomposition"),
    ("sustained-command", "freqtrade/hedge/hprl/cli.py", "perf-sustained-pipeline"),
    ("v24-hw-gate", "tools/run_hprl_performance_v24_rtx5070_gate.py", "deep-sustained"),
    ("checkpoint-capture", "freqtrade/hedge/hprl/checkpoint.py", "capture_checkpoint_payload"),
    ("checkpoint-write", "freqtrade/hedge/hprl/checkpoint.py", "write_checkpoint_payload"),
    ("replay-identity", "freqtrade/hedge/hprl/replay.py", "staging_identity"),
    ("bounded-worker", "freqtrade/hedge/hprl/async_io.py", "Queue(maxsize=queue_size)"),
    ("flush-close", "freqtrade/hedge/hprl/async_io.py", "self.flush()"),
]
for name, relative, token in source_checks:
    def check_source(relative=relative, token=token):
        path = ROOT / relative
        text = path.read_text(encoding="utf-8")
        compile(text, str(path), "exec")
        assert token in text
        return {"file": relative, "token": token}
    record("G10-source-integration", name, check_source)

assert len(checks) == 200, len(checks)
failed = [row for row in checks if row["status"] != "PASS"]
report = {
    "schema": "hprl-performance-v2.4-runtime-200-v1",
    "expected": 200, "executed": len(checks), "passed": len(checks) - len(failed),
    "failed": len(failed), "status": "PASS" if not failed else "FAIL", "checks": checks,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"HPRL PERFORMANCE V2.4 RUNTIME 200: {report['passed']}/200 PASS; FAIL={report['failed']}")
    print(json.dumps({k: report[k] for k in ("schema", "expected", "executed", "passed", "failed", "status")}, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
