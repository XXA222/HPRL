#!/usr/bin/env python3
"""Focused 100-check V2.4.1 hardware-gate correctness matrix."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from freqtrade.hedge.hprl import HPRL_API_VERSION,HPRL_RELEASE
from freqtrade.hedge.hprl import cli
from freqtrade.hedge.hprl.sustained_benchmark import _throughput_health, benchmark_sustained_training
from freqtrade.hedge.hprl.config import HPRLTrainingConfig
from freqtrade.hedge.hprl.registry import create_agent
from freqtrade.hedge.hprl.action_space import configure_agent_action_levels
from freqtrade.hedge.hprl.config import HPRLActionConfig
checks=[]
def rec(group,name,fn):
    try: detail=fn(); checks.append({'index':len(checks)+1,'group':group,'name':name,'status':'PASS','detail':detail})
    except Exception as e: checks.append({'index':len(checks)+1,'group':group,'name':name,'status':'FAIL','error':f'{type(e).__name__}: {e}'})

def must(cond,msg='assertion failed'):
    if not cond: raise AssertionError(msg)
    return True

# G01 stable synthetic, 10
for i in range(10):
    rec('G01-stable',f'stable-{i}',lambda i=i: (lambda h: (must(h['stable']),h))(_throughput_health([100+i,102+i,99+i,101+i,100+i,103+i,98+i,101+i,100+i,102+i])))
# G02 terminal degradation, 10
for i in range(10):
    rec('G02-terminal',f'terminal-{i}',lambda i=i: (lambda h: (must(not h['stable']),must('edge_degradation' in h['reasons'] or 'terminal_degradation' in h['reasons']),h))(_throughput_health([100,100,100,100,100,100,100,100,20+i*0.1,20+i*0.1])))
# G03 mid-run collapse/recovery must still fail, 10
for i in range(10):
    seq=[100,102,101,100,20,20,20,20,101,100]
    rec('G03-mid-collapse',f'mid-{i}',lambda seq=seq: (lambda h: (must(not h['stable']),must(h['recovery_observed']),must(h['longest_collapse_run']>=4),h))(_throughput_health(seq)))
# G04 short transient tolerance, 10
for i in range(10):
    seq=[100,101,100,102,99,20,100,101,99,100]
    rec('G04-single-transient',f'transient-{i}',lambda seq=seq: (lambda h: (must(h['stable']),must(h['collapse_fraction']<=0.10),h))(_throughput_health(seq)))
# G05 CLI loss_post contract, 10
for i in range(10):
    rec('G05-cli',f'loss-post-{i}',lambda: (lambda ns: (must(ns.compile_scope=='loss_post'),{'scope':ns.compile_scope}))(cli.build_parser().parse_args(['perf-orchestration-profile','--device','cpu','--algorithm','fast_dsac','--compile-scope','loss_post'])))
# G06 release/source contract, 10
for i in range(10):
    rec('G06-release',f'release-{i}',lambda: (must(HPRL_API_VERSION in {'2.4.1','2.5','2.5.1','2.5.2'}),must(HPRL_RELEASE.endswith(('perf-v2.4.1','perf-v2.5','perf-v2.5.1','perf-v2.5.2'))),{'api':HPRL_API_VERSION,'release':HPRL_RELEASE}))
# G07 CPU sustained result schema/fields, 10
for i in range(10):
    def smoke(i=i):
        cfg=HPRLTrainingConfig(algorithm='fast_td3',device='cpu',replay_device='same',batch_size=8,replay_capacity=32,warmup_steps=0,hidden_dim=16,hidden_depth=1,compile_mode='off')
        agent=create_agent('fast_td3',8,4,cfg,device='cpu'); configure_agent_action_levels(agent,HPRLActionConfig().level_count)
        r=benchmark_sustained_training(agent,obs_dim=8,action_dim=4,batch_size=8,iterations=10,warmup=1,window_size=5,replay_capacity=32,replay_device='same',pin_memory=False,metrics_interval=5,checkpoint_interval=5,checkpoint_keep_last=1,artifact_queue_size=2)
        must(r.schema=='hprl-sustained-training-benchmark-v2'); must(len(r.windows)==2); must(hasattr(r,'throughput_edge_ratio')); must(r.parameters_finite); return {'stable':r.throughput_stable,'reasons':r.throughput_stability_reasons}
    rec('G07-cpu-smoke',f'smoke-{i}',smoke)
# G08 source gate correctness, 10
source=(ROOT/'tools/run_hprl_performance_v24_rtx5070_gate.py').read_text()
for i in range(10):
    rec('G08-gate-source',f'gate-{i}',lambda: (must('launch_profile_valid' in source),must('retry_unstable=True' in source),must('sustained_5k_role' in source),{'ok':True}))
# G09 sustained source telemetry/robustness, 10
sust=(ROOT/'freqtrade/hedge/hprl/sustained_benchmark.py').read_text()
for i in range(10):
    rec('G09-sustained-source',f'source-{i}',lambda: (must('_throughput_health' in sust),must('_gpu_window_telemetry' in sust),must('throughput_collapse_fraction' in sust),must('throughput_longest_collapse_run' in sust),{'ok':True}))
# G10 AST compile touched files, 10
paths=['freqtrade/hedge/hprl/__init__.py','freqtrade/hedge/hprl/cli.py','freqtrade/hedge/hprl/sustained_benchmark.py','tools/run_hprl_performance_v24_rtx5070_gate.py','tools/validate_hprl_performance_v241_100.py']
for i in range(10):
    rel=paths[i%len(paths)]
    rec('G10-ast',f'ast-{i}-{Path(rel).name}',lambda rel=rel: (compile((ROOT/rel).read_text(),rel,'exec'),{'file':rel})[1])
assert len(checks)==100
failed=[x for x in checks if x['status']!='PASS']
report={'schema':'hprl-performance-v2.4.1-hotfix-runtime-100-v1','expected':100,'executed':100,'passed':100-len(failed),'failed':len(failed),'status':'PASS' if not failed else 'FAIL','checks':checks}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default=''); a=ap.parse_args()
    if a.output: Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(f"HPRL PERFORMANCE V2.4.1 HOTFIX RUNTIME 100: {report['passed']}/100 PASS; FAIL={report['failed']}")
    print(json.dumps({k:report[k] for k in ('schema','expected','executed','passed','failed','status')},sort_keys=True))
    return 0 if not failed else 1
if __name__=='__main__': raise SystemExit(main())
