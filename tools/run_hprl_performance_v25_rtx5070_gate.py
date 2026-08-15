#!/usr/bin/env python3
"""HPRL V2.5 RTX5070 paired-confidence, workload-I/O and XQC compute gate."""
from __future__ import annotations

import argparse, contextlib, io, json, os, statistics, subprocess, sys
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from freqtrade.hedge.hprl.calibration import paired_scope_confidence_decision
from freqtrade.hedge.hprl.cli import build_parser

INTEROP={"rebrac_v2":16,"xqc":1,"fast_td3":8,"fast_dsac":1,"simba_sac":32}
ALGORITHMS=("fast_td3","fast_dsac","simba_sac","xqc","rebrac_v2")

def run_json(cmd:list[str]):
    cp=subprocess.run(cmd,cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True,check=False)
    if cp.returncode: return None,(cp.stderr or cp.stdout)[-12000:]
    try: return json.loads(cp.stdout),None
    except Exception as e: return None,f"json error {e}: {cp.stdout[-3000:]}"


def profiler_scope_capability(scopes: tuple[str, ...]) -> dict[str, object]:
    """Fail-fast contract: every hardware candidate scope must parse in orchestration profiler CLI."""
    results: dict[str, bool] = {}
    errors: list[dict[str, str]] = []
    parser = build_parser()
    for scope in scopes:
        sink = io.StringIO()
        try:
            with contextlib.redirect_stderr(sink):
                parser.parse_args(["perf-orchestration-profile", "--compile-scope", scope])
            results[scope] = True
        except SystemExit:
            results[scope] = False
            errors.append({"scope": scope, "error": sink.getvalue().strip() or "argparse_rejected_scope"})
    return {
        "schema": "hprl-v252-profiler-scope-capability-v1",
        "results": results,
        "errors": errors,
        "pass": not errors and all(results.values()),
    }

def precision_flags(a): return ["--flow-likelihood-precision","fp32"] if a=="rebrac_v2" else ["--mixed-precision"]

def bench(args,a,scope):
    return [sys.executable,"-m","freqtrade.hedge.hprl","perf-benchmark","--device",args.device,
      "--algorithm",a,"--batch-size",str(args.batch_size),"--hidden-dim",str(args.hidden_dim),
      "--hidden-depth",str(args.hidden_depth),"--warmup",str(args.warmup),"--iterations",str(args.iterations),
      "--compile-mode","reduce-overhead","--compile-scope",scope,"--expected-updates","10000",
      "--compile-cache-state","warm","--hardware-profile","rtx5070_laptop","--cpu-interop-threads",str(INTEROP[a]),
      "--obs-dim","32","--action-dim","4",*precision_flags(a)]

def profile(args,a,scope):
    return [sys.executable,"-m","freqtrade.hedge.hprl","perf-orchestration-profile","--device",args.device,
      "--algorithm",a,"--batch-size",str(args.batch_size),"--hidden-dim",str(args.hidden_dim),"--hidden-depth",str(args.hidden_depth),
      "--compile-mode","reduce-overhead","--compile-scope",scope,"--compile-cache-state","warm",
      "--hardware-profile","rtx5070_laptop","--cpu-interop-threads",str(INTEROP[a]),"--active","5",*precision_flags(a)]

def paired_scope(args,a,baseline,candidate,min_speedup):
    base=[]; cand=[]; rounds=[]; errors=[]
    for i in range(args.scope_repeats):
        order=((baseline,candidate) if i%2==0 else (candidate,baseline))
        row={"round":i,"order":list(order)}
        for scope in order:
            payload,error=run_json(bench(args,a,scope))
            if error or payload is None:
                errors.append({"round":i,"scope":scope,"error":error or "missing_payload"}); continue
            rate=float(payload.get("iterations_per_second",0.0))
            finite=bool(payload.get("parameters_finite",False))
            observed=str(payload.get("compile_scope",""))
            if rate<=0.0 or not finite or observed!=scope:
                errors.append({"round":i,"scope":scope,"error":"invalid_benchmark_contract",
                    "rate":rate,"parameters_finite":finite,"observed_scope":observed})
                continue
            row[scope]=rate
        if baseline in row and candidate in row:
            base.append(row[baseline]); cand.append(row[candidate]); rounds.append(row)
    complete_pairs=len(base)==args.scope_repeats
    if complete_pairs:
        decision=paired_scope_confidence_decision(cand,base,min_speedup=min_speedup,
            bootstrap_threshold=args.bootstrap_threshold,bootstrap_samples=args.bootstrap_samples,seed=2500)
    else:
        decision={"promote":False,"reason":"incomplete_paired_measurements",
            "required_pairs":args.scope_repeats,"observed_pairs":len(base),
            "min_speedup":min_speedup,"bootstrap_threshold":args.bootstrap_threshold}
    bp,be=run_json(profile(args,a,baseline)); cp,ce=run_json(profile(args,a,candidate))
    if be: errors.append({"profile":baseline,"error":be})
    if ce: errors.append({"profile":candidate,"error":ce})
    for scope,payload in ((baseline,bp),(candidate,cp)):
        if payload is not None and str(payload.get("compile_scope",""))!=scope:
            errors.append({"profile":scope,"error":"profile_scope_mismatch",
                "observed_scope":payload.get("compile_scope")})
    bc=((bp or {}).get("categories") or {}); cc=((cp or {}).get("categories") or {})
    profiles_valid=bool(bp and cp and bc and cc)
    launch_ok=profiles_valid and int(cc.get("cuda_kernel_launches",0))<=int(bc.get("cuda_kernel_launches",0))
    copy_ok=profiles_valid and int(cc.get("dtype_copy_ops",0))<=int(bc.get("dtype_copy_ops",0))
    promote=bool(complete_pairs and not errors and decision.get("promote",False) and launch_ok and copy_ok)
    return {"algorithm":a,"baseline":baseline,"candidate":candidate,"rounds":rounds,
      "baseline_rates":base,"candidate_rates":cand,"decision":decision,"complete_pairs":complete_pairs,
      "baseline_profile":bp,"candidate_profile":cp,"profiles_valid":profiles_valid,
      "kernel_launch_nonincrease":launch_ok,"dtype_copy_nonincrease":copy_ok,
      "recommended_scope":candidate if promote else baseline,"promotion_eligible":promote,
      "errors":errors,"pass":complete_pairs and profiles_valid and not errors}

def pipe(args,a,mode):
    return [sys.executable,"-m","freqtrade.hedge.hprl","perf-pipeline-benchmark","--device",args.device,
      "--algorithm",a,"--batch-size",str(args.batch_size),"--hidden-dim",str(args.hidden_dim),"--hidden-depth",str(args.hidden_depth),
      "--warmup","20","--iterations",str(args.pipeline_iterations),"--replay-capacity","16384","--replay-device","cpu",
      "--prefetch-slots","2","--metrics-interval",str(args.metrics_interval),"--checkpoint-interval",str(args.checkpoint_interval),
      "--diagnostic-iterations","1","--compile-mode","auto","--compile-scope","auto","--compile-cache-state","warm",
      "--hardware-profile","rtx5070_laptop","--cpu-interop-threads",str(INTEROP[a]),"--artifact-io-mode",mode,
      "--artifact-queue-size","8","--skip-micro-reference",*precision_flags(a)]

def io_gate(args):
    results={}; errors=[]
    for a in ALGORITHMS:
        auto,ae=run_json(pipe(args,a,"auto")); sync,se=run_json(pipe(args,a,"sync")); asyncp,xe=run_json(pipe(args,a,"async"))
        if ae or se or xe or not auto or not sync or not asyncp:
            errors.append({"algorithm":a,"auto":ae,"sync":se,"async":xe}); continue
        sr=float(sync["samples_per_second"]); ar=float(asyncp["samples_per_second"])
        auto_mode=auto.get("artifact_io_resolved"); policy=auto.get("artifact_io_policy") or {}
        auto_finite=bool(auto.get("parameters_finite",False)); sync_finite=bool(sync.get("parameters_finite",False)); async_finite=bool(asyncp.get("parameters_finite",False))
        policy_match=str(policy.get("resolved",auto_mode))==str(auto_mode)
        results[a]={"auto":auto,"sync":sync,"async":asyncp,"async_vs_sync":ar/sr if sr else 0.0,
          "auto_mode":auto_mode,"policy":policy,"policy_matches_runtime":policy_match,
          "all_parameters_finite":auto_finite and sync_finite and async_finite,
          "pass":auto.get("artifact_io_requested")=="auto" and auto_mode in {"sync","async"}
                 and policy_match and auto_finite and sync_finite and async_finite and sr>0.0 and ar>0.0}
    return {"schema":"hprl-v25-artifact-io-gate-v1","results":results,"errors":errors,
      "pass":not errors and all(v["pass"] for v in results.values())}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',default='cuda'); ap.add_argument('--batch-size',type=int,default=1024)
    ap.add_argument('--hidden-dim',type=int,default=256); ap.add_argument('--hidden-depth',type=int,default=2)
    ap.add_argument('--warmup',type=int,default=50); ap.add_argument('--iterations',type=int,default=200)
    ap.add_argument('--scope-repeats',type=int,default=9); ap.add_argument('--bootstrap-samples',type=int,default=8000)
    ap.add_argument('--bootstrap-threshold',type=float,default=0.95); ap.add_argument('--pipeline-iterations',type=int,default=300)
    ap.add_argument('--metrics-interval',type=int,default=50); ap.add_argument('--checkpoint-interval',type=int,default=100)
    args=ap.parse_args()
    if args.scope_repeats<7: ap.error('scope-repeats must be >=7')
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],capture_output=True,text=True,check=False).stdout.strip().splitlines()
    gpu_name=gpu[0] if gpu else 'unknown'
    capability=profiler_scope_capability(('module','loss','loss_post','xqc_fused'))
    if capability['pass']:
        rebrac=paired_scope(args,'rebrac_v2','loss','loss_post',1.03)
        xqc=paired_scope(args,'xqc','module','xqc_fused',1.05)
        io=io_gate(args)
    else:
        rebrac={"pass":False,"errors":[{"error":"profiler_scope_capability_failed"}]}
        xqc={"pass":False,"errors":[{"error":"profiler_scope_capability_failed"}]}
        io={"pass":False,"errors":[{"error":"hardware_gate_preflight_failed"}]}
    ok='RTX 5070' in gpu_name and capability['pass'] and rebrac['pass'] and xqc['pass'] and io['pass']
    payload={"schema":"hprl-performance-v2.5.2-rtx5070-gate","status":"PASS" if ok else "FAIL",
      "gpu_name":gpu_name,"profiler_scope_capability":capability,
      "rebrac_paired_scope":rebrac,"xqc_compute_scope":xqc,"artifact_io_auto":io}
    print(json.dumps(payload,indent=2,sort_keys=True))
    return 0 if ok else 1
if __name__=='__main__': raise SystemExit(main())
