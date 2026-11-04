"""Independent saved-output CPU audit. Never constructs a restoration model.

LPIPS-Alex is a metric network and is explicitly recomputed on CPU for every
saved output. Duplicate validation paths within one call are audited once.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='8'
os.environ['ASCEND_RT_VISIBLE_DEVICES']=''
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter,defaultdict
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'unidb_d0_20261102'
os.environ['XDG_CACHE_HOME']=str(ROOT/'cache/cpu_output_audit')
METRIC=Path('/data2/yanglin/v5_concertormer_10k_20261021/training/image_metrics.py')
TORCH_CACHE=Path('/data2/yanglin/v6_diffuir_l_20261027/cache/torch')
GS=[.5,1,2,3,4]
KEYS=['psnr','ssim','lpips','mae']
TOL={'psnr':{'atol':1e-8,'rtol':0.},'ssim':{'atol':1e-8,'rtol':0.},
     'mae':{'atol':1e-8,'rtol':0.},'lpips':{'atol':2e-4,'rtol':2e-4}}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def derive(*values):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,values)).encode()).digest()[:8],'little')%(2**63-1)


def match(actual,expected,key):
    return abs(actual-expected)<=TOL[key]['atol']+TOL[key]['rtol']*abs(expected)


def simple_metrics(pred,clean):
    d=pred.astype(np.float64)-clean.astype(np.float64);mse=float(np.mean(d*d))
    return {'psnr':-10*math.log10(max(mse,1e-30)),'mae':float(np.mean(np.abs(d)))}


def load_expected():
    gate_path=OLD/'evidence/DATA_GATE.json';gate=json.loads(gate_path.read_text());assert gate['status']=='PASS'
    manifest=Path(gate['val_manifest']);assert sha(manifest)==gate['val_manifest_sha256']
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    assert len(rows)==1000 and len({r['sample_id'] for r in rows})==1000
    sources=defaultdict(list)
    for r in rows:sources[r['source_id']].append(r)
    assert len(sources)==250 and all(len(v)==4 for v in sources.values())
    expected={(r['sample_id'],float(g)):r for r in rows for g in GS}
    return expected,{'data_gate_sha256':sha(gate_path),'val_manifest':str(manifest),'val_manifest_sha256':sha(manifest)}


def coverage(rows,summary,expected):
    assert summary['status']=='PASS' and summary['full'] is True and summary['count']==5000
    assert summary['per_g']==1000 and summary['sources_per_g']==250 and summary['nfe_per_image']==100
    assert summary['sampler']=='native_reverse_sde' and summary['image_clip_crop_resize'] is False
    assert summary.get('test_used') is False
    assert len(rows)==5000 and len({(r['sample_id'],float(r['g'])) for r in rows})==5000
    assert {(r['sample_id'],float(r['g'])) for r in rows}==set(expected)
    assert Counter(float(r['g']) for r in rows)=={g:1000 for g in GS}
    groups=defaultdict(list)
    for r in rows:
        g=float(r['g']);e=expected[(r['sample_id'],g)];key=f'{g:g}'
        assert r['source_id']==e['source_id'] and r['update']==summary['update']
        assert r['branch']==summary['branch'] and r['weights']==summary['weights']
        assert Path(r['input_path']).resolve()==Path(e['states'][key]['path']).resolve()
        assert r['input_sha256']==e['states'][key]['sha256']
        assert Path(r['clean_path']).resolve()==Path(e['clean_path']).resolve()
        assert r['clean_sha256']==e['clean_sha256']
        assert r['group_seed']==derive(843001,r['source_id'],'four_realizations_native_SDE')
        assert r['group_batch']==4 and r['nfe']==100 and 0<=r['group_position']<4
        assert all(math.isfinite(float(r['metrics'][k])) for k in KEYS)
        groups[(g,r['source_id'])].append(r)
    assert len(groups)==1250
    for (g,source),group in groups.items():
        assert len(group)==4 and {r['group_position'] for r in group}=={0,1,2,3}
        assert len({r['sample_id'] for r in group})==4
        # Native validation orders each source's four realizations by sample_id;
        # the position controls which part of the seeded batch RNG it receives.
        for position,row in enumerate(sorted(group,key=lambda r:r['sample_id'])):
            assert row['group_position']==position,('Native seeded batch order mismatch',source,row['sample_id'])
    for g in GS:
        q=[r for r in rows if float(r['g'])==g]
        assert len({r['source_id'] for r in q})==250
        for k in KEYS:
            assert abs(float(np.mean([r['metrics'][k] for r in q]))-summary['by_g'][f'{g:g}'][k])<=1e-10
    return groups


def cpu_metric_module(threads):
    import torch
    torch.set_num_threads(threads)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    # Require the existing pretrained file before model construction: no download
    # and no writes to original assets are needed.
    alex=TORCH_CACHE/'hub/checkpoints/alexnet-owt-7be5be79.pth'
    assert alex.is_file(),'Original cached pretrained AlexNet must already exist; do not download during final audit.'
    alex_sha=sha(alex);assert alex_sha.startswith('7be5be79')
    os.environ['TORCH_HOME']=str(TORCH_CACHE)
    spec=importlib.util.spec_from_file_location('dual_original_cpu_metrics',METRIC)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.perceptual_model('cpu')
    assert all(p.device.type=='cpu' for p in model.parameters())
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return module,{'metric_source':str(METRIC),'metric_source_sha256':sha(METRIC),
        'metric_SPEC':{k:module.SPEC[k] for k in ['psnr','ssim','lpips']},'LPIPS_network':'official AlexNet v0.1, pretrained, normalize=True, eval/FP32 CPU',
        'metric_parameter_digest_full':h.hexdigest(),'alexnet_pretrained_file_sha256':alex_sha,
        'torch_version':torch.__version__,'torch_threads':threads,'interop_threads':1}


def audit_one(path,outprefix,expected,source_identity,metric,metric_identity):
    import torch
    begin=time.time();summary_path=path/'SUMMARY.json';summary=json.loads(summary_path.read_text());rows=[];rank_ids=[];ranks=[]
    for file in sorted(path.glob('rank*.jsonl')):
        rank=int(file.stem[4:]);rank_ids.append(rank)
        done=json.loads((path/f'DONE_rank{rank}.json').read_text());assert done['status']=='PASS'
        actual=sha(file);assert done['sha256']==actual
        rr=[json.loads(s) for s in file.read_text().splitlines()];assert done['count']==len(rr)
        rows.extend(rr);ranks.append({'file':str(file),'sha256':actual,'records':len(rr)})
    assert sorted(rank_ids)==list(range(len(rank_ids)))
    assert len(rank_ids)==(3 if summary['branch']=='original' else 5)
    groups=coverage(rows,summary,expected)
    identity={'validation':str(path),'SUMMARY_sha256':sha(summary_path),'rank_files':ranks,
              'source_identity':source_identity,'metric_identity':metric_identity,'auditor_sha256':sha(__file__)}
    signature=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    checked={};shape_checked=set();sourcefiles=[]
    def check_file(file,expected_sha):
        file=Path(file).resolve();key=str(file)
        if key in checked:assert checked[key]==expected_sha
        else:
            actual=sha(file);assert actual==expected_sha,(str(file),actual,expected_sha)
            checked[key]=actual;sourcefiles.append({'path':key,'sha256':actual,'bytes':file.stat().st_size})
    # Full source SHA scan always runs, including before any completed cache reuse.
    for r in rows:
        expected_pred=(path/'predictions'/f"G{float(r['g']):g}"/(r['sample_id']+'.npy')).resolve()
        assert Path(r['prediction']).resolve()==expected_pred
        for file,key in [(r['prediction'],'prediction_sha256'),(r['input_path'],'input_sha256'),(r['clean_path'],'clean_sha256')]:check_file(file,r[key])
    csv_path=outprefix.with_name(outprefix.name+'_PER_IMAGE.csv')
    cached_path=outprefix.with_name(outprefix.name+'_RESULT.json')
    if cached_path.is_file():
        cached=json.loads(cached_path.read_text())
        if cached.get('status')=='PASS' and cached.get('signature')==signature:
            assert Path(cached['per_image_CSV']).is_file() and sha(cached['per_image_CSV'])==cached['per_image_CSV_sha256']
            cached.update(reused_completed_audit=True,source_SHA_revalidated=True,reuse_seconds=time.time()-begin,
                          CPU_metric_forward_batches_this_invocation=0)
            return cached
    result_rows=[];maxerr={k:0. for k in KEYS};failures=Counter();examples=[];cpu_means={};record_means={};metric_forward_batches=0
    def array(file):
        a=np.load(file,allow_pickle=False)
        assert a.shape==(3,256,256) and a.dtype==np.float32 and np.isfinite(a).all()
        return a
    for index,((g,source),group) in enumerate(sorted(groups.items()),1):
        group=sorted(group,key=lambda r:r['group_position'])
        pred=[array(r['prediction']) for r in group];clean=[array(r['clean_path']) for r in group]
        inputs=[array(r['input_path']) for r in group]
        p=torch.from_numpy(np.stack(pred));c=torch.from_numpy(np.stack(clean));assert p.device.type==c.device.type=='cpu'
        ss,lp=metric.image_metrics(p,c);metric_forward_batches+=1
        for r,pa,ca,ya,s,l in zip(group,pred,clean,inputs,ss,lp):
            actual={**simple_metrics(pa,ca),'ssim':float(s),'lpips':float(l)}
            input_actual=simple_metrics(ya,ca)
            item={k:r[k] for k in ['branch','weights','update','source_id','sample_id','g','group_seed','group_position','prediction','prediction_sha256','input_path','input_sha256','clean_path','clean_sha256']}
            for k in KEYS:
                old=float(r['metrics'][k]);error=abs(actual[k]-old);ok=match(actual[k],old,k)
                item.update({f'recorded_{k}':old,f'CPU_recomputed_{k}':actual[k],f'abs_error_{k}':error,f'within_tolerance_{k}':ok})
                maxerr[k]=max(maxerr[k],error)
                if not ok:
                    failures[k]+=1
                    if len(examples)<20:examples.append({'sample_id':r['sample_id'],'g':g,'metric':k,'recorded':old,'CPU':actual[k],'abs_error':error})
            item.update(CPU_input_psnr=input_actual['psnr'],CPU_input_mae=input_actual['mae'])
            assert abs(input_actual['psnr']-r['input_metrics']['psnr'])<=1e-8
            assert abs(input_actual['mae']-r['input_metrics']['mae'])<=1e-8
            result_rows.append(item)
        if index%25==0:
            write(outprefix.with_name(outprefix.name+'_PROGRESS.json'),{'groups':index,'total_groups':1250,'images':len(result_rows),'seconds':time.time()-begin,'restoration_calls':0,'CPU_metric_batches':metric_forward_batches})
    with csv_path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(result_rows[0]));writer.writeheader();writer.writerows(result_rows)
    mean_errors={k:0. for k in KEYS}
    for g in GS:
        q=[r for r in result_rows if float(r['g'])==g];key=f'{g:g}';cpu_means[key]={};record_means[key]={}
        for k in KEYS:
            value=float(np.mean([r[f'CPU_recomputed_{k}'] for r in q]));old=float(summary['by_g'][key][k])
            cpu_means[key][k]=value;record_means[key][k]=old;mean_errors[k]=max(mean_errors[k],abs(value-old))
            if not match(value,old,k):failures['mean_'+k]+=1
        assert abs(float(np.mean([r['CPU_input_psnr'] for r in q]))-summary['by_g'][key]['input_psnr'])<=1e-8
    manifests=outprefix.with_name(outprefix.name+'_SOURCE_SHA.json');write(manifests,{'status':'PASS','unique_files_checked':len(sourcefiles),'files':sourcefiles})
    result={'status':'PASS' if not failures else 'REVIEW_NUMERICAL_AGREEMENT','validation':str(path),'validation_dir':str(path),'summary_path':str(summary_path),
        'summary_sha256':sha(summary_path),'report_path':str(cached_path),'signature':signature,
        'identity':identity,'images':5000,'sources_per_strength':250,'realizations_per_source':4,'strengths':GS,
        'coverage':'exact canonical1000samples x5strengths; no duplicate or missing identity; group seeds/positions/source SHA matched',
        'all_prediction_input_clean_SHA_verified':True,'unique_file_SHA_checks':len(sourcefiles),'per_image_max_absolute_errors':maxerr,
        'CPU_recomputed_means':cpu_means,'recorded_SUMMARY_means':record_means,'mean_max_absolute_errors':mean_errors,
        'fixed_tolerances':TOL,'failure_counts':dict(failures),'failure_examples':examples,
        'per_image_CSV':str(csv_path),'per_image_CSV_sha256':sha(csv_path),'source_SHA_manifest':str(manifests),
        'restoration_network_calls':0,'NPU_calls':0,'optimizer_steps':0,'CPU_metric_forward_batches':metric_forward_batches,
        'CPU_LPIPS_images':len(result_rows),'CPU_metric_forward_batches_this_invocation':metric_forward_batches,
        'pure_CPU':True,'CPU_metric_network_allowed':True,'restoration_model_constructed':False,'input_metrics_scope':'All input SHA + PSNR/MAE freshly recomputed; primary output SSIM/LPIPS freshly evaluated for all5000. Input SSIM/LPIPS were not newly evaluated.',
        'seconds':time.time()-begin,'reused_completed_audit':False,
        'limits':'Independent CPU recomputation against recorded CPU metrics uses fixed declared descriptive tolerances for thread/version/floating implementation differences, not mathematical identity. A PASS proves saved-output provenance/coverage/metrics at this protocol, not restoration success, convergence or generalization.'}
    write(cached_path,result);return result


def self_test(out):
    a=np.zeros((3,16,16),dtype=np.float32);b=np.ones_like(a)*.25;m=simple_metrics(a,b)
    assert abs(m['mae']-.25)<1e-15 and abs(m['psnr']-(-10*math.log10(.0625)))<1e-15
    assert match(1.,1.+1e-10,'psnr') and not match(1.,1.01,'psnr')
    # Synthetic exact5000 identity/group coverage; no image/model access.
    expected={};rows=[]
    for source in range(250):
        for xi in range(4):
            sid=f's{source}_r{xi}';src=f'source{source}';e={'sample_id':sid,'source_id':src,'clean_path':f'/toy/clean{source}.npy','clean_sha256':'c','states':{f'{g:g}':{'path':f'/toy/G{g:g}/{sid}.npy','sha256':'i'} for g in GS}}
            for g in GS:
                expected[(sid,float(g))]=e;rows.append({'sample_id':sid,'source_id':src,'g':g,'update':2,'branch':'improved','weights':'EMA','input_path':e['states'][f'{g:g}']['path'],'input_sha256':'i','clean_path':e['clean_path'],'clean_sha256':'c','group_seed':derive(843001,src,'four_realizations_native_SDE'),'group_batch':4,'nfe':100,'group_position':xi,'metrics':{k:1. for k in KEYS}})
    summary={'status':'PASS','full':True,'count':5000,'per_g':1000,'sources_per_g':250,'nfe_per_image':100,'sampler':'native_reverse_sde','image_clip_crop_resize':False,'test_used':False,'update':2,'branch':'improved','weights':'EMA','by_g':{f'{g:g}':{k:1. for k in KEYS} for g in GS}}
    assert len(coverage(rows,summary,expected))==1250
    broken=list(rows);broken[-1]=dict(rows[0]);rejected=False
    try:coverage(broken,summary,expected)
    except AssertionError:rejected=True
    assert rejected
    wrong_position=[dict(r) for r in rows]
    source0=[i for i,r in enumerate(wrong_position) if r['source_id']=='source0' and r['g']==.5]
    a0,a1=source0[:2]
    wrong_position[a0]['group_position'],wrong_position[a1]['group_position']=wrong_position[a1]['group_position'],wrong_position[a0]['group_position']
    rejected=False
    try:coverage(wrong_position,summary,expected)
    except AssertionError:rejected=True
    assert rejected
    write(out,{'status':'PASS','toy_PSNR_MAE':True,'synthetic_exact5000_coverage':True,'duplicate_missing_rejected':True,'swapped_seeded_group_positions_rejected':True,'metric_or_restoration_models_constructed':0,'NPU_calls':0,'auditor_sha256':sha(__file__)})


def main():
    p=argparse.ArgumentParser();p.add_argument('--validation',action='append',default=[]);p.add_argument('--out',required=True);p.add_argument('--threads',type=int,default=8);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    out=Path(a.out).resolve();assert out.is_relative_to(ROOT) and 1<=a.threads<=8
    out.parent.mkdir(parents=True,exist_ok=True)
    if a.self_test:self_test(out);print('CPU_OUTPUT_AUDITOR_SELFTEST_PASS');return
    paths=list(dict.fromkeys(Path(x).resolve() for x in a.validation));assert paths,'--validation DIR required'
    assert all(x.is_relative_to(ROOT) and x.is_dir() for x in paths)
    protocol={'auditor_sha256':sha(__file__),'validation_paths':[str(x) for x in paths],'duplicate_paths_removed':len(a.validation)-len(paths),
              'tolerances_frozen_before_metric_evaluation':TOL,'batch_size':4,'CPU_threads':a.threads,'metric_source_sha256':sha(METRIC),
              'PSNR_MAE':'float64 RGB raw, no crop/clip/resize; per-image mean and range1',
              'LPIPS_difference':'Independent CPU recomputation against recorded CPU metrics; fixed thread/version/floating implementation tolerances. REVIEW on violation, never relax automatically.',
              'restoration_model_construction_forbidden':True,'NPU_calls':0,'frozen_unix':time.time()}
    write(out.with_name(out.stem+'_PROTOCOL.json'),protocol)
    try:
        expected,source_identity=load_expected();metric,metric_identity=cpu_metric_module(a.threads);results=[]
        for path in paths:
            tag=hashlib.sha256(str(path).encode()).hexdigest()[:12];prefix=out.with_name(out.stem+'_'+tag)
            results.append(audit_one(path,prefix,expected,source_identity,metric,metric_identity))
        aggregate={'status':'PASS' if all(r['status']=='PASS' for r in results) else 'REVIEW_NUMERICAL_AGREEMENT',
            'protocol':protocol,'unique_validations':len(results),'validations':results,
            'audited_summaries':{r['summary_path']:r['summary_sha256'] for r in results},
            'reports':{r['report_path']:sha(r['report_path']) for r in results},
            'pure_CPU':True,'CPU_metric_network_allowed':True,'CPU_metric_model_constructions':1,
            'CPU_metric_forward_batches_this_invocation':sum(r['CPU_metric_forward_batches_this_invocation'] for r in results),
            'restoration_model_constructed':False,'restoration_network_calls':0,'NPU_calls':0,'optimizer_steps':0}
        if len(results)==1:
            for k in ['validation_dir','summary_path','summary_sha256','per_image_CSV','per_image_CSV_sha256']:
                aggregate[k]=results[0][k]
        write(out,aggregate)
        print('SAVED_OUTPUT_CPU_AUDIT_COMPLETE',len(results),flush=True)
    except BaseException as exc:
        write(out,{'status':'FAIL','protocol':protocol,'error':repr(exc),'restoration_network_calls':0,'NPU_calls':0,'optimizer_steps':0})
        raise


if __name__=='__main__':main()
