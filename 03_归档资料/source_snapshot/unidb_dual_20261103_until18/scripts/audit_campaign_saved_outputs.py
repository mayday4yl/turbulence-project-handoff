"""Extension CPU-only wrapper around the immutable noon saved-output auditor.

The original audit_one algorithm, tolerances, metric source and source SHA stay
unchanged, so canonical noon caches remain reusable after every source SHA scan.
Only this wrapper controls approved read roots and new-root output locations.
"""
import argparse,hashlib,importlib.util,json,os,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
LEGACY_AUDITOR_SHA='a9c684bf42fdab2a1cf34670d717d7d3d1e4dc21bccc5b7f63bc18beaddf3950'

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def approved_roots():
 a=json.loads((ROOT/'protocol/USER_AUTHORITY.json').read_text())
 old=Path(a['previous_campaign_root']).resolve()
 assert old.name=='unidb_dual_20261103_until_noon'
 return ROOT.resolve(),old

def main():
 p=argparse.ArgumentParser();p.add_argument('--validation',action='append',default=[]);p.add_argument('--out',required=True)
 p.add_argument('--threads',type=int,default=8);p.add_argument('--self-test',action='store_true');a=p.parse_args()
 own,previous=approved_roots();out=Path(a.out).resolve()
 assert out.is_relative_to(own) and 1<=a.threads<=8
 out.parent.mkdir(parents=True,exist_ok=True)
 legacy=previous/'scripts/audit_saved_outputs.py'
 assert sha(legacy)==LEGACY_AUDITOR_SHA,'Immutable noon auditor source changed.'
 os.environ.update(ASCEND_RT_VISIBLE_DEVICES='',CUDA_VISIBLE_DEVICES='',TORCH_DEVICE_BACKEND_AUTOLOAD='0',
  PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(a.threads),OPENBLAS_NUM_THREADS=str(a.threads),MKL_NUM_THREADS=str(a.threads))
 spec=importlib.util.spec_from_file_location('immutable_noon_saved_output_auditor',legacy)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 # Importing the original defines a legacy cache path but performs no model work.
 # Redirect all subsequently constructed metric/cache state to the new campaign.
 os.environ['XDG_CACHE_HOME']=str(own/'cache/cpu_output_audit')
 if a.self_test:
  m.self_test(out)
  q=json.loads(out.read_text());q.update(wrapper_sha256=sha(__file__),legacy_auditor_sha256=LEGACY_AUDITOR_SHA,
   allowed_read_roots=[str(own),str(previous)],all_outputs_inside_new_root=True)
  m.write(out,q);print('CAMPAIGN_CPU_WRAPPER_SELFTEST_PASS');return
 paths=list(dict.fromkeys(Path(x).resolve() for x in a.validation));assert paths
 for path in paths:
  assert path.is_dir() and any(path.is_relative_to(r/'runs') for r in [own,previous]),path
  assert path.name.endswith('_ema_full'),path
 protocol={'wrapper_sha256':sha(__file__),'legacy_auditor':str(legacy),'legacy_auditor_sha256':sha(legacy),
  'validation_paths':[str(x) for x in paths],'allowed_read_roots':[str(own),str(previous)],
  'out_root':str(own),'old_assets_writes_allowed':False,'tolerances_frozen_before_metric_evaluation':m.TOL,
  'batch_size':4,'CPU_threads':a.threads,'metric_source_sha256':sha(m.METRIC),
  'restoration_model_construction_forbidden':True,'NPU_calls':0,'frozen_unix':time.time()}
 m.write(out.with_name(out.stem+'_PROTOCOL.json'),protocol)
 try:
  expected,source_identity=m.load_expected();metric,metric_identity=m.cpu_metric_module(a.threads);results=[]
  for path in paths:
   tag=hashlib.sha256(str(path).encode()).hexdigest()[:12];prefix=out.with_name(out.stem+'_'+tag)
   result=m.audit_one(path,prefix,expected,source_identity,metric,metric_identity)
   assert result['status']=='PASS',result.get('failure_counts')
   # Materialize the invocation receipt and copied dependencies in the new root;
   # the original immutable signature and all metric values remain unchanged.
   for key,sha_key in [('per_image_CSV','per_image_CSV_sha256'),('source_SHA_manifest',None)]:
    src=Path(result[key]);target=own/'evidence'/src.name
    if not target.exists():
     assert src.is_file()
     target.write_bytes(src.read_bytes())
    if sha_key:assert sha(target)==result[sha_key]
    else:assert sha(target)==sha(src)
    result[key]=str(target)
   result['report_path']=str(prefix.with_name(prefix.name+'_RESULT.json'))
   result['wrapper_sha256']=sha(__file__);result['all_audit_outputs_inside_new_root']=True
   m.write(result['report_path'],result);results.append(result)
  aggregate={'status':'PASS','protocol':protocol,'unique_validations':len(results),'validations':results,
   'audited_summaries':{str(Path(r['summary_path']).resolve()):r['summary_sha256'] for r in results},
   'reports':{r['report_path']:sha(r['report_path']) for r in results},'pure_CPU':True,'CPU_metric_network_allowed':True,
   'CPU_metric_model_constructions':1,'CPU_metric_forward_batches_this_invocation':sum(r['CPU_metric_forward_batches_this_invocation'] for r in results),
   'historical_completed_audits_reused':sum(bool(r.get('reused_completed_audit')) for r in results),
   'restoration_model_constructed':False,'restoration_network_calls':0,'NPU_calls':0,'optimizer_steps':0}
  m.write(out,aggregate);print('CAMPAIGN_SAVED_OUTPUT_CPU_AUDIT_COMPLETE',len(results),flush=True)
 except BaseException as exc:
  m.write(out,{'status':'FAIL','protocol':protocol,'error':repr(exc),'restoration_network_calls':0,'NPU_calls':0,'optimizer_steps':0})
  raise

if __name__=='__main__':main()
