#!/usr/bin/env python3
from __future__ import annotations
import ast, hashlib, importlib.util, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from freqtrade.hedge.hprl import HPRL_API_VERSION,HPRL_RELEASE
from freqtrade.hedge.hprl.cli import build_parser
from freqtrade.hedge.hprl.config import HPRLTrainingConfig
from freqtrade.hedge.hprl.performance import resolve_compile_scope
checks=[]
def rec(group,name,fn):
    try: detail=fn(); ok=True
    except Exception as e: ok=False; detail=f"{type(e).__name__}: {e}"
    checks.append({"group":group,"name":name,"pass":ok,"detail":detail})
def must(x,msg="contract_failed"):
    if not x: raise AssertionError(msg)
    return True
pairs=[('xqc','xqc_fused'),('xqc','module'),('rebrac_v2','loss'),('rebrac_v2','loss_post'),('fast_dsac','loss'),('simba_sac','loss'),('fast_td3','module'),('xqc','loss'),('fast_dsac','loss_post'),('simba_sac','loss_post')]
for i,(alg,scope) in enumerate(pairs):
    rec('G01-parser-scope',f'parser-{i:02d}',lambda alg=alg,scope=scope:(lambda a:(must(a.algorithm==alg),must(a.compile_scope==scope),{'algorithm':alg,'scope':scope}))(build_parser().parse_args(['perf-orchestration-profile','--algorithm',alg,'--compile-scope',scope])))
spec=importlib.util.spec_from_file_location('v252gate',ROOT/'tools/run_hprl_performance_v25_rtx5070_gate.py')
gate=importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(gate)
cap=gate.profiler_scope_capability(('module','loss','loss_post','xqc_fused'))
A=type('A',(),{'device':'cuda','batch_size':1024,'hidden_dim':256,'hidden_depth':2})
cap_tests=[lambda:must(cap['pass'] is True),lambda:must(cap['errors']==[]),lambda:must(cap['results']['module'] is True),lambda:must(cap['results']['loss'] is True),lambda:must(cap['results']['loss_post'] is True),lambda:must(cap['results']['xqc_fused'] is True),lambda:must(cap['schema']=='hprl-v252-profiler-scope-capability-v1'),lambda:must('--compile-scope' in gate.profile(A(),'xqc','xqc_fused')),lambda:must('xqc_fused' in gate.profile(A(),'xqc','xqc_fused')),lambda:must(gate.INTEROP['xqc']==1)]
for i,fn in enumerate(cap_tests): rec('G02-gate-preflight',f'preflight-{i:02d}',fn)
res_tests=[lambda:must(HPRLTrainingConfig(algorithm='xqc',compile_scope='xqc_fused').compile_scope=='xqc_fused'),lambda:must(resolve_compile_scope('xqc_fused','xqc',hardware_profile='rtx5070_laptop')=='xqc_fused'),lambda:must(resolve_compile_scope('module','xqc',hardware_profile='rtx5070_laptop')=='module'),lambda:must(resolve_compile_scope('loss_post','rebrac_v2',hardware_profile='rtx5070_laptop')=='loss_post'),lambda:must(resolve_compile_scope('loss','rebrac_v2',hardware_profile='rtx5070_laptop')=='loss'),lambda:must(HPRL_API_VERSION=='2.5.2'),lambda:must(HPRL_RELEASE.endswith('perf-v2.5.2')),lambda:must('xqc_fused' in (ROOT/'freqtrade/hedge/hprl/cli.py').read_text()),lambda:must('profiler_scope_capability' in (ROOT/'tools/run_hprl_performance_v25_rtx5070_gate.py').read_text()),lambda:must('hprl-performance-v2.5.2-rtx5070-gate' in (ROOT/'tools/run_hprl_performance_v25_rtx5070_gate.py').read_text())]
for i,fn in enumerate(res_tests): rec('G03-runtime-contract',f'runtime-{i:02d}',fn)
source_files=[ROOT/'freqtrade/hedge/hprl/cli.py',ROOT/'freqtrade/hedge/hprl/__init__.py',ROOT/'tools/run_hprl_performance_v25_rtx5070_gate.py',ROOT/'tools/validate_hprl_performance_v25_200.py',ROOT/'tools/validate_hprl_performance_v241_100.py',ROOT/'tests/hedge/hprl/test_performance_compute_io.py']
for i,p in enumerate(source_files):
    def ast_check(p=p):
        ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
        return str(p.relative_to(ROOT))
    rec('G04-source-ast',f'ast-{i:02d}',ast_check)
for j,needle in enumerate(['choices=("module", "loss", "loss_post", "xqc_fused")','profiler_scope_capability','profiles_valid','dtype_copy_nonincrease']): rec('G04-source-ast',f'source-{j:02d}',lambda needle=needle:must(needle in ((ROOT/'freqtrade/hedge/hprl/cli.py').read_text()+(ROOT/'tools/run_hprl_performance_v25_rtx5070_gate.py').read_text()),needle))
EXPECTED={'freqtrade/hedge/hprl/algorithms/fast_dsac.py': 'c220b97875bfab7ba023b33e057e318b2365ba390b973d6d9c233f356ee23173', 'freqtrade/hedge/hprl/algorithms/simba_sac.py': 'ee752d0ae54d7d18eea4a898998c13d0eaf279cdd610a0708c96f572420890dc', 'freqtrade/hedge/hprl/algorithms/rebrac_v2.py': 'a35b39b01b4d106b912b2b90070f35ce1a4b3f9fb3846ee8195be7ae15312af0', 'freqtrade/hedge/hprl/algorithms/xqc.py': 'a04805238916481b7b78ac17ed970974801b99dc78618a8be89c3e279b4445a2', 'freqtrade/hedge/hprl/networks.py': 'a0dddb0a1c932b55a76ddc92abcd44726567a61e1d0a16c2715c798073ad0f64', 'freqtrade/hedge/hprl/artifact_policy.py': '047281c40c0c2d5e41a6dd192e6bedf69b8dae43728d88dd5f6b00597d3fbf38', 'freqtrade/hedge/hprl/pipeline_benchmark.py': 'e5de76e4be7368d42a6b6bd3ca2a08cb41c805e2643b4a8ce6c06b5c6247d67d', 'freqtrade/hedge/hprl/calibration.py': '05d2de5de4e3d5f482889a78fd958db06455a1c91507c2f085b60fc7f31791ee'}
for i,(rel,expected) in enumerate(EXPECTED.items()): rec('G05-isolation',f'sha-{i:02d}',lambda rel=rel,expected=expected:(lambda got:(must(got==expected,f'sha mismatch {rel}'),{'path':rel,'sha256':got}))(hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()))
rec('G05-isolation','release-08',lambda:must(HPRL_RELEASE.endswith('perf-v2.5.2')))
rec('G05-isolation','count-09',lambda:must(len(checks)==49))
failed=[c for c in checks if not c['pass']]
payload={'schema':'hprl-performance-v2.5.2-hotfix-runtime-50-v1','status':'PASS' if not failed else 'FAIL','expected':50,'executed':len(checks),'passed':len(checks)-len(failed),'failed':len(failed),'checks':checks}
print(json.dumps(payload,sort_keys=True))
print(f"HPRL PERFORMANCE V2.5.2 HOTFIX RUNTIME 50: {payload['passed']}/50 PASS; FAIL={len(failed)}")
raise SystemExit(0 if not failed else 1)
