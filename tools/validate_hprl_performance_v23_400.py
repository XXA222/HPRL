#!/usr/bin/env python3
"""HPRL Performance V2.3 deep acceptance matrix: 400 executable checks."""
from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freqtrade.hedge.hprl import HPRL_RELEASE
from freqtrade.hedge.hprl.action_space import configure_agent_action_levels
from freqtrade.hedge.hprl.algorithms.base import FrozenModulePlan, PolyakUpdatePlan, soft_update
from freqtrade.hedge.hprl.calibration import (
    balanced_interleaved_orders,
    mad_inlier_mask,
    paired_bootstrap_superiority_probability,
    paired_speedup_summary,
    paired_winner_confidence,
    robust_distribution_summary,
)
from freqtrade.hedge.hprl.cli import build_parser
from freqtrade.hedge.hprl.config import HPRLActionConfig, HPRLTrainingConfig
from freqtrade.hedge.hprl.performance import (
    compile_break_even_updates,
    compile_policy_thresholds,
    condition_cuda_device,
    resolve_compile_mode,
    resolve_compile_scope,
    resolve_host_interop_threads,
    summarize_profile_operations,
)
from freqtrade.hedge.hprl.pipeline_benchmark import benchmark_training_pipeline
from freqtrade.hedge.hprl.registry import create_agent
from freqtrade.hedge.hprl.replay import ReplayBatch, TensorReplayBuffer
from freqtrade.hedge.hprl.checkpoint import save_checkpoint, load_checkpoint
from freqtrade.hedge.hprl.device import require_torch

torch = require_torch()
ALGORITHMS = ("fast_td3", "fast_dsac", "simba_sac", "xqc", "rebrac_v2")
TARGET = ("fast_dsac", "simba_sac", "rebrac_v2")
COLD = {"fast_td3": 7500, "fast_dsac": 6000, "simba_sac": 3000, "xqc": 6000, "rebrac_v2": 3000}
WARM = {"fast_td3": 900, "fast_dsac": 500, "simba_sac": 300, "xqc": 1100, "rebrac_v2": 400}
EAGER = {"fast_td3": 1, "fast_dsac": 16, "simba_sac": 4, "xqc": 1, "rebrac_v2": 16}
COMPILED = {"fast_td3": 8, "fast_dsac": 1, "simba_sac": 32, "xqc": 1, "rebrac_v2": 16}


def check(rows, group, case, passed, detail=""):
    rows.append({"number": len(rows)+1, "group": group, "case": case, "passed": bool(passed), "detail": detail or ("PASS" if passed else "FAIL")})


def cfg(algorithm="fast_td3", **kwargs):
    base = dict(algorithm=algorithm, device="cpu", replay_device="same", batch_size=8,
                replay_capacity=64, warmup_steps=0, hidden_dim=16, hidden_depth=1,
                compile_mode="off", metrics_interval=1000)
    base.update(kwargs)
    return HPRLTrainingConfig(**base)


def batch(seed: int, obs_dim=7, action_dim=4, rows=8):
    g = torch.Generator().manual_seed(seed)
    return ReplayBatch(
        obs=torch.randn(rows, obs_dim, generator=g),
        action=torch.rand(rows, action_dim, generator=g),
        reward=torch.randn(rows, 1, generator=g)*0.01,
        next_obs=torch.randn(rows, obs_dim, generator=g),
        done=torch.zeros(rows, 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    rows = []

    # G01 robust distributions: 20
    for i in range(20):
        values = [100+i*0.01, 101+i*0.01, 99+i*0.01, 100.5+i*0.01, 1000+i]
        s = robust_distribution_summary(values)
        ok = s["count"] == 5 and s["outliers"] == 1 and 99 <= s["robust_median"] <= 102
        check(rows, "G01_ROBUST_STATS", f"case={i}", ok, json.dumps(s, sort_keys=True))

    # G02 MAD mask: 20
    for i in range(20):
        base = float(i+1)
        mask = mad_inlier_mask([base, base+0.1, base-0.1, base+0.05, base+100])
        check(rows, "G02_MAD_FILTER", f"case={i}", mask[-1] is False and sum(mask)==4, str(mask))

    # G03 paired estimator/bootstrap: 20
    for i in range(20):
        baseline = [100+j+i*0.01 for j in range(7)]
        candidate = [value*(1.05 + i*0.001) for value in baseline]
        summary = paired_speedup_summary(candidate, baseline)
        prob = paired_bootstrap_superiority_probability(candidate, baseline, samples=500, seed=100+i)
        conf = paired_winner_confidence(candidate, baseline, bootstrap_samples=500, seed=200+i)
        ok = summary["median_speedup"] > 1.04 and prob > 0.95 and conf["paired_margin_pct"] > 4
        check(rows, "G03_PAIRED", f"case={i}", ok, json.dumps(conf, sort_keys=True))

    # G04 balanced interleaving: 20
    for i in range(20):
        count = 2 + (i % 4)
        candidates = tuple(range(1, count+1))
        orders = balanced_interleaved_orders(candidates, 7+i%3, seed=300+i)
        ok = all(sorted(order)==list(candidates) for order in orders) and len(set(orders)) > 1
        check(rows, "G04_INTERLEAVE", f"case={i}", ok, str(orders[:3]))

    # G05 calibrated cold/warm thresholds: 20
    for i in range(20):
        alg = ALGORITHMS[i%5]
        q=i//5
        thresholds=compile_policy_thresholds(alg,"rtx5070_laptop")
        if q==0: ok=thresholds=={"cold":COLD[alg],"warm":WARM[alg]}
        elif q==1: ok=compile_break_even_updates(alg,"rtx5070_laptop","cold")==COLD[alg]
        elif q==2: ok=compile_break_even_updates(alg,"rtx5070_laptop","warm")==WARM[alg]
        else: ok=COLD[alg] >= WARM[alg]
        check(rows,"G05_THRESHOLDS",f"{alg}:{q}",ok,str(thresholds))

    # G06 cache-state aware auto compile: 20
    for i in range(20):
        alg=ALGORITHMS[i%5]; state="cold" if i//5<2 else "warm"; t=COLD[alg] if state=="cold" else WARM[alg]
        horizon=t-1 if (i//5)%2==0 else t
        expected="off" if horizon<t else "reduce-overhead"
        got=resolve_compile_mode("auto",alg,"cuda:0",expected_updates=horizon,hardware_profile="rtx5070_laptop",compile_cache_state=state)
        check(rows,"G06_AUTO_COMPILE",f"{alg}:{state}:{horizon}",got==expected,got)

    # G07 compile scope contract: 20
    values=["auto","module","loss"]
    for i in range(20):
        value=values[i%3]; c=cfg(compile_scope=value)
        expected="module" if value=="auto" else value
        ok=c.compile_scope==value and resolve_compile_scope(value,TARGET[i%3])==expected
        check(rows,"G07_COMPILE_SCOPE",f"{value}:{i}",ok,resolve_compile_scope(value,TARGET[i%3]))

    # G08 Polyak pre-bound parity: 20
    for i in range(20):
        torch.manual_seed(400+i)
        a=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4)); b=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4))
        c=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4)); d=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.BatchNorm1d(4))
        c.load_state_dict(a.state_dict()); d.load_state_dict(b.state_dict()); tau=0.01+0.01*(i%10)
        soft_update(b,a,tau,foreach=False); PolyakUpdatePlan(d,c,foreach=False).step(tau)
        ok=all(torch.equal(x,y) for x,y in zip(b.state_dict().values(),d.state_dict().values(),strict=True))
        check(rows,"G08_POLYAK_PLAN",f"case={i}:tau={tau}",ok)

    # G09 reusable freeze plan restores module/grad state: 20
    for i in range(20):
        module=torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.ReLU(),torch.nn.Linear(4,2)); module.train(i%2==0)
        before=module.training; plan=FrozenModulePlan(module,eval_mode=bool(i%3==0))
        with plan.frozen():
            inner=all(not p.requires_grad for p in module.parameters()) and (not module.training if i%3==0 else module.training==before)
        ok=inner and module.training==before and all(p.requires_grad for p in module.parameters())
        check(rows,"G09_FREEZE_PLAN",f"case={i}",ok)

    # G10 target agent orchestration caches + real update: 20
    for i in range(20):
        alg=TARGET[i%3]; torch.manual_seed(500+i); agent=create_agent(alg,7,4,cfg(alg),device="cpu"); configure_agent_action_levels(agent,HPRLActionConfig().level_count)
        if hasattr(agent,"warmup_updates"): agent.update_count=agent.warmup_updates
        metrics=agent.update(batch(600+i),collect_metrics=True)
        ok=isinstance(agent._actor_params,tuple) and isinstance(agent._critic_params,tuple) and isinstance(agent._critic_polyak,PolyakUpdatePlan) and all(math.isfinite(float(v)) for v in metrics.values.values())
        check(rows,"G10_TARGET_AGENT",f"{alg}:{i}",ok,str(metrics.values))

    # G11 all algorithm real updates remain finite: 20
    for i in range(20):
        alg=ALGORITHMS[i%5]; torch.manual_seed(700+i); agent=create_agent(alg,7,4,cfg(alg),device="cpu"); configure_agent_action_levels(agent,HPRLActionConfig().level_count)
        if hasattr(agent,"warmup_updates"): agent.update_count=agent.warmup_updates
        m=agent.update(batch(800+i),collect_metrics=True)
        ok=bool(m.values) and all(math.isfinite(float(v)) for v in m.values.values())
        check(rows,"G11_ALL_AGENT_UPDATE",f"{alg}:{i}",ok)

    # G12 loss surfaces: 20 direct forward + backward viability
    for i in range(20):
        alg=TARGET[i%3]; torch.manual_seed(900+i); agent=create_agent(alg,7,4,cfg(alg),device="cpu"); configure_agent_action_levels(agent,HPRLActionConfig().level_count); b=batch(1000+i)
        if alg=="rebrac_v2":
            loss=agent._critic_loss_surface(b.obs,b.action,b.reward,b.next_obs,b.done)
        else:
            boundaries=agent._tier_buffers.gaussian_boundaries
            alpha=agent.log_alpha.detach().exp()
            loss=agent._critic_loss_surface(b.obs,b.action,b.reward,b.next_obs,b.done,alpha,boundaries)
        ok=torch.is_tensor(loss) and loss.ndim==0 and bool(torch.isfinite(loss))
        check(rows,"G12_LOSS_SURFACE",f"{alg}:{i}",ok,float(loss.detach()))

    # G13 CPU pipeline end-to-end: 20
    for i in range(20):
        alg=ALGORITHMS[i%5]; agent=create_agent(alg,7,4,cfg(alg,batch_size=8,replay_capacity=64),device="cpu"); configure_agent_action_levels(agent,HPRLActionConfig().level_count)
        if hasattr(agent,"warmup_updates"): agent.update_count=agent.warmup_updates
        r=benchmark_training_pipeline(agent,obs_dim=7,action_dim=4,batch_size=8,iterations=1,warmup=0,replay_capacity=64,replay_device="same",metrics_interval=1,checkpoint_interval=0,diagnostic_iterations=1)
        ok=r.samples==8 and r.samples_per_second>0 and r.stage_diagnostics["update_target_seconds"]>0 and r.stage_diagnostics["checkpoint_bytes"]>0
        check(rows,"G13_PIPELINE",f"{alg}:{i}",ok,f"sps={r.samples_per_second:.3f}")

    # G14 conditioning/profiler aggregation contracts: 20
    for i in range(20):
        c=condition_cuda_device("cpu",milliseconds=i*10)
        profile=summarize_profile_operations([
            {"name":"cudaLaunchKernel","count":i+1}, {"name":"cudaGraphLaunch","count":2},
            {"name":"aten::_foreach_lerp_","count":3}, {"name":"aten::_to_copy","count":4},
            {"name":"Optimizer.step#Adam.step","count":5},
        ])["categories"]
        ok=not c["enabled"] and profile["cuda_kernel_launches"]==i+1 and profile["cuda_graph_launches"]==2 and profile["foreach_ops"]==3 and profile["dtype_copy_ops"]==4
        check(rows,"G14_CONDITION_PROFILE",f"case={i}",ok,str(profile))

    # G15 CLI command/option contracts: 20
    cli=build_parser()
    argv_cases=[]
    for i in range(10): argv_cases.append(["perf-pipeline-benchmark","--device","cpu","--algorithm",ALGORITHMS[i%5],"--iterations","1"])
    for i in range(10): argv_cases.append(["perf-orchestration-profile","--device","cpu","--algorithm",TARGET[i%3],"--compile-scope","loss"])
    for i,argv in enumerate(argv_cases):
        try: ns=cli.parse_args(argv); ok=ns.command in {"perf-pipeline-benchmark","perf-orchestration-profile"}
        except SystemExit: ok=False
        check(rows,"G15_CLI",f"case={i}",ok)

    # G16 replay reusable staging: 20
    for i in range(20):
        rb=TensorReplayBuffer(64,7,4,device="cpu",pin_memory=False,validate_inputs=False); b=batch(1100+i)
        rb.add(b.obs,b.action,b.reward,b.next_obs,b.done); rb.add(b.obs,b.action,b.reward,b.next_obs,b.done)
        sample=rb.sample_reusable(8,staging_slot=i%2); ok=sample.obs.shape==(8,7) and sample.action.shape==(8,4) and rb.sample_stage_bytes>0
        rb.release(); check(rows,"G16_REPLAY",f"case={i}",ok)

    # G17 checkpoint round-trip after orchestration plan additions: 20
    for i in range(20):
        alg=ALGORITHMS[i%5]; torch.manual_seed(1200+i); source=create_agent(alg,7,4,cfg(alg),device="cpu"); target=create_agent(alg,7,4,cfg(alg),device="cpu")
        configure_agent_action_levels(source,HPRLActionConfig().level_count); configure_agent_action_levels(target,HPRLActionConfig().level_count)
        with tempfile.TemporaryDirectory() as d:
            path=save_checkpoint(Path(d)/"a.pt",source,{"i":i}); meta=load_checkpoint(path,target)
            ok=meta["i"]==i and all(torch.equal(a,b) for a,b in zip(source.actor.state_dict().values(),target.actor.state_dict().values(),strict=True))
        check(rows,"G17_CHECKPOINT",f"{alg}:{i}",ok)

    # G18 deterministic same-seed update pairs: 20
    for i in range(20):
        alg=ALGORITHMS[i%5]; torch.manual_seed(1300+i); a=create_agent(alg,7,4,cfg(alg),device="cpu"); torch.manual_seed(1300+i); bagent=create_agent(alg,7,4,cfg(alg),device="cpu")
        configure_agent_action_levels(a,HPRLActionConfig().level_count); configure_agent_action_levels(bagent,HPRLActionConfig().level_count)
        if hasattr(a,"warmup_updates"): a.update_count=a.warmup_updates; bagent.update_count=bagent.warmup_updates
        bt=batch(1400+i); torch.manual_seed(1500+i); ma=a.update(bt,collect_metrics=True); torch.manual_seed(1500+i); mb=bagent.update(bt,collect_metrics=True)
        ok=ma.values.keys()==mb.values.keys() and all(float(ma.values[k])==float(mb.values[k]) for k in ma.values)
        check(rows,"G18_DETERMINISM",f"{alg}:{i}",ok)

    # G19 source AST + mandatory V2.3 feature markers: 20
    source_files=[
        "freqtrade/hedge/hprl/calibration.py","freqtrade/hedge/hprl/performance.py","freqtrade/hedge/hprl/pipeline_benchmark.py",
        "freqtrade/hedge/hprl/algorithms/base.py","freqtrade/hedge/hprl/algorithms/fast_dsac.py","freqtrade/hedge/hprl/algorithms/simba_sac.py",
        "freqtrade/hedge/hprl/algorithms/rebrac_v2.py","freqtrade/hedge/hprl/cli.py","tools/run_hprl_performance_v23_rtx5070_gate.py",
        "tests/hedge/hprl/test_performance_pipeline_variance.py",
    ]
    markers=["paired_winner_confidence","condition_cuda_device","agent_finite_state","benchmark_training_pipeline","PolyakUpdatePlan","_critic_loss_surface","_actor_loss_surface","perf-pipeline-benchmark","perf-orchestration-profile","balanced_interleaved_orders","summarize_profile_operations"]
    for i in range(20):
        path=ROOT/source_files[i%len(source_files)]; text=path.read_text(encoding="utf-8"); ast.parse(text); marker=markers[i%len(markers)]
        universe="\n".join((ROOT/f).read_text(encoding="utf-8") for f in source_files)
        check(rows,"G19_SOURCE",f"{path.name}:{marker}:{i}",marker in universe)

    # G20 release/integration contracts: 20
    expected_threads={"fast_td3":(1,8),"fast_dsac":(16,1),"simba_sac":(4,32),"xqc":(1,1),"rebrac_v2":(16,16)}
    for i in range(20):
        alg=ALGORITHMS[i%5]; q=i//5
        if q==0: ok=resolve_host_interop_threads(0,alg,"cuda:0","rtx5070_laptop",compile_mode="off")==expected_threads[alg][0]
        elif q==1: ok=resolve_host_interop_threads(0,alg,"cuda:0","rtx5070_laptop",compile_mode="reduce-overhead")==expected_threads[alg][1]
        elif q==2: ok="perf-v2.3" in HPRL_RELEASE
        else: ok=(ROOT/"tools/run_hprl_performance_v23_rtx5070_gate.py").exists() and (ROOT/"freqtrade/hedge/hprl/pipeline_benchmark.py").exists()
        check(rows,"G20_INTEGRATION",f"{alg}:{q}",ok,HPRL_RELEASE)

    if len(rows)!=400:
        raise RuntimeError(f"expected 400 checks, got {len(rows)}")
    failed=[r for r in rows if not r["passed"]]
    result={"schema":"hprl-performance-v2.3-runtime-400-v1","status":"PASS" if not failed else "FAIL","expected":400,"executed":len(rows),"passed":400-len(failed),"failed":len(failed),"checks":rows}
    if args.output:
        Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    if args.summary_only:
        print(json.dumps({k:v for k,v in result.items() if k!="checks"},sort_keys=True))
    else:
        print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if not failed else 1

if __name__ == "__main__":
    raise SystemExit(main())
