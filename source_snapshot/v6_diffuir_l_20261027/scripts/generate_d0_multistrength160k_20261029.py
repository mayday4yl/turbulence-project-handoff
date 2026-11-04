"""Generate matched G1..4 for every original xi, reusing verified old work."""
import argparse,json,os,shutil,sys,time
from pathlib import Path
import numpy as np
import torch
from scripts.runtime_utils import ROOT,sha,write_json,process_identity
from scripts.generate_g4 import raw_noise
from scripts.prepare_d0_multistrength160k_20261029 import EX,PREVIOUS,rows

def partial_rows(path):
    data=[]
    if not path.exists():return data
    for line in path.read_text().splitlines():
        try:data.append(json.loads(line))
        except json.JSONDecodeError:break
    return data
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--safety-file',required=True);a=ap.parse_args()
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE']);assert world==4
    safety=json.load(open(a.safety_file));assert safety['physical_devices']==[4,5,6,7]
    for n,h in json.load(open(EX/'GENERATION_SOURCE_MANIFEST.json'))['files'].items():assert sha(ROOT/n)==h,n
    plan=json.load(open(EX/'DATA_PLAN.json'));assert sha(EX/'TRAIN_PLAN.jsonl')==plan['train_plan_sha256']
    import torch_npu
    torch.set_num_threads(2);torch.npu.set_device(local);device=torch.device(f'npu:{local}')
    torch.npu.set_compile_mode(jit_compile=False);torch.npu.config.allow_internal_format=True
    torch.npu.conv.allow_hf32=False;torch.npu.matmul.allow_hf32=False
    if hasattr(torch.npu,'aclnn'):torch.npu.aclnn.allow_hf32=False
    torch.npu.set_option({'ACL_PRECISION_MODE':'must_keep_origin_dtype','NPU_FUZZY_COMPILE_BLACKLIST':'Conv2DBackpropInput'})
    free,total=torch.npu.mem_get_info();torch.npu.set_per_process_memory_fraction(safety['cards'][str(4+rank)]['allocator_cap']/total)
    out=ROOT/'distributed'/a.run;write_json(out/f'RANK_{rank}.json',{**process_identity(os.getpid()),'physical_device':4+rank})
    simroot=ROOT.parent/'v5_v1_simulator_20261020';assert sha(simroot/'simulator_v5.py')=='0f2f6f54a578b3c36cb69861815c7eee5476f75ccd91cac863bd001345bd9636'
    sys.path.insert(0,str(simroot));from simulator_v5 import build_simulator,Realization
    sim=build_simulator(device=str(device)).eval().requires_grad_(False)
    assert sim.smooth_psf and sim.pixel_center_grid and sim.tilt_fft=='legacy_full'
    allrows=rows(EX/'TRAIN_PLAN.jsonl');mine=[r for r in allrows if r['source_index']%world==rank];assert len(mine)==40000
    reused={(r['sample_id'],r['g']):r for r in partial_rows(PREVIOUS/f'GENERATED_TRAIN_rank{rank}.jsonl')}
    started=time.time();count=0;copied=0;created=0;previous=None;current_xi=None;raw=None;xi=None;replays=[]
    def save(dest,array):
        dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();tmp=dest.with_suffix('.npy.tmp')
        with tmp.open('xb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
        os.replace(tmp,dest);return sha(dest)
    with torch.no_grad(),(EX/f'GENERATED_TRAIN_rank{rank}.jsonl').open('x') as log:
        for row in mine:
            assert not (EX/'STOP_REQUEST.json').exists(),'User STOP'
            if previous!=row['source_id']:
                assert sha(row['clean_path'])==row['clean_sha256'];clean=np.load(row['clean_path'],allow_pickle=False)
                assert clean.shape==(3,256,256) and clean.dtype==np.float32 and np.isfinite(clean).all()
                image=torch.from_numpy(clean[None]).to(device);prepared=sim.prepare_image(image);previous=row['source_id']
            if current_xi!=row['sample_id']:raw=None;xi=None;current_xi=row['sample_id'];hashes=row.get('raw_sha256')
            dest=Path(row['observation_path']);g=row['g'];reuse=reused.get((row['sample_id'],g));source=None;replay=None
            if g==4:source=row['old_g4_path'];expected=row['old_g4_sha256']
            elif reuse:
                for k in ['source_id','realization_index','clean_sha256','seed_zernike','seed_tilt']:assert reuse[k]==row[k]
                source=reuse['observation_path'];expected=reuse['observation_sha256'];hashes=reuse['raw_sha256']
            if source:
                assert sha(source)==expected;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists()
                shutil.copyfile(source,dest);digest=sha(dest);assert digest==expected;array=np.load(dest,allow_pickle=False);copied+=1
            else:
                if raw is None:
                    raw,hashes=raw_noise(row)
                    if row.get('raw_sha256'):assert hashes==row['raw_sha256']
                    else:assert hashes==json.load(open(ROOT/'datasets/g4_records'/(row['sample_id']+'.json')))['raw_sha256']
                    xi=Realization(*(v.to(device) for v in raw))
                    if len(replays)<8:
                        ref=sim.render_prepared(prepared,xi,4.).output[0].cpu().numpy()
                        assert sha(row['old_g4_path'])==row['old_g4_sha256']
                        replay=float(np.max(np.abs(ref-np.load(row['old_g4_path'],allow_pickle=False))));assert replay<=2e-5
                        replays.append({'sample_id':row['sample_id'],'g4_maxabs':replay,'raw_sha256':hashes})
                        assert torch.equal(sim.render_prepared(prepared,xi,0.).output,image)
                value=sim.render_prepared(prepared,xi,float(g)).output;array=value[0].cpu().numpy();digest=save(dest,array);created+=1;del value
            assert array.shape==(3,256,256) and array.dtype==np.float32 and np.isfinite(array).all()
            receipt={**row,'observation_sha256':digest,'raw_sha256':hashes,'finite':True,
                'cache_source':'verified_existing_copy' if source else 'fresh_exact_physics',
                'reused_from':str(source) if source else None,'g4_replay_maxabs':replay}
            log.write(json.dumps(receipt,ensure_ascii=False,allow_nan=False)+'\n');log.flush();count+=1
            if count%400==0:
                info={'status':'GENERATING','rank':rank,'completed':count,'total':len(mine),'fresh':created,'copied':copied,'seconds':time.time()-started}
                write_json(EX/f'GENERATION_rank{rank}.json',info);print(json.dumps(info),flush=True)
    with torch.no_grad(),(EX/f'GENERATED_VAL_rank{rank}.jsonl').open('x') as log:
        for row in rows(EX/'VAL_PLAN.jsonl')[rank::world]:
            assert not (EX/'STOP_REQUEST.json').exists()
            assert sha(row['clean_path'])==row['clean_sha256'];clean=np.load(row['clean_path'],allow_pickle=False)
            image=torch.from_numpy(clean[None]).to(device);prepared=sim.prepare_image(image);raw,hashes=raw_noise(row);assert hashes==row['raw_sha256']
            xi=Realization(*(v.to(device) for v in raw));array=sim.render_prepared(prepared,xi,.5).output[0].cpu().numpy();assert np.isfinite(array).all()
            row['states']['0.5']['sha256']=save(row['states']['0.5']['path'],array)
            log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
    result={'status':'PASS','train_count':count,'val_g05_count':250,'fresh':created,'copied':copied,'seconds':time.time()-started,
        'replays':replays,'optimizer_updates':0,'physical_device':4+rank,'peak_allocated':torch.npu.max_memory_allocated(),'peak_reserved':torch.npu.max_memory_reserved()}
    write_json(out/f'RESULT_rank{rank}.json',result);write_json(EX/f'GENERATION_rank{rank}.json',result)
    if rank==0:write_json(out/'RESULT.json',{'status':'PASS','scope':'generation completed; independent CPU data gate still required'})
if __name__=='__main__':main()
