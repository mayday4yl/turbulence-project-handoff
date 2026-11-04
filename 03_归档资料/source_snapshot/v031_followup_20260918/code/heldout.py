"""Fixed unseen-source preparation and direct four-severity replay, no model."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys,tarfile,time
import numpy as np
import torch
from v031.runtime import save_json,sha256,verify_source,configure_npu
from v031.data import read_jsonl,StartDataset

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'shared/heldout32'
POOL=Path('/data2/yanglin/physical_bridge/data/places_p2s_bridge_v01/manifests')
ARCHIVE=Path('/data2/yanglin/physical_bridge/data/source_archives/val_256.tar')
BUILDER=Path('/data2/yanglin/v03_overfit100_g4321_n4/code/build_dataset.py')

def digest(b):return hashlib.sha256(b).hexdigest()
def rows(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in x));tmp.replace(p)
def array(p,x):
    a=np.asarray(x,dtype=np.float32)
    if not np.isfinite(a).all():raise FloatingPointError('Nonfinite heldout array')
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.with_suffix('.tmp').open('wb') as f:np.save(f,a,allow_pickle=False)
    p.with_suffix('.tmp').replace(p)
def seed(s,r,b):
    v=json.dumps(['v031-followup-heldout',20260918,s,r,b],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(v).digest()[:8],'big')&((1<<63)-1)

def prepare():
    if (OUT/'PREPARED.json').exists():return
    cfg=json.loads((ROOT/'shared/original_train_config.json').read_text());verify_source(cfg)
    pool=read_jsonl(POOL/'source_train.jsonl');val=read_jsonl(POOL/'source_val.jsonl');test=read_jsonl(POOL/'source_test.jsonl')
    mapping={r['source_image_id']:r for r in pool}
    assert len(mapping)==len(pool) and all(r['source_split']=='train' for r in pool)
    assert set(mapping)=={r['source_image_id'] for r in read_jsonl(POOL/'train.jsonl')}
    old=read_jsonl(Path(cfg['dataset_root'])/'source_manifest.jsonl')
    excluded={r['source_id'] for r in old}|{r['source_image_id'] for r in val+test}
    chosen=sorted(set(mapping)-excluded,key=lambda s:(digest(('v031-followup-heldout|20260918|'+s).encode()),s))[:32]
    if len(chosen)!=32:raise ValueError('Fewer than32 verified unused sources')
    identity={'source_ids':chosen,'selection_rule':'SHA256(v031-followup-heldout|20260918|+source_id) ascending',
        'seed_algorithm':'compact JSON [v031-followup-heldout,20260918,source_id,r,block], SHA256 first8 bytes big endian masked63bits',
        'pool_hashes':{p.name:sha256(p) for p in (POOL/'source_train.jsonl',POOL/'source_val.jsonl',POOL/'source_test.jsonl',POOL/'train.jsonl')},
        'builder_sha256':sha256(BUILDER),'original_dataset_root':cfg['dataset_root'],'frozen_unix':time.time()}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'FROZEN_SOURCES.json').exists():
        saved=json.loads((OUT/'FROZEN_SOURCES.json').read_text())
        assert {k:v for k,v in saved.items() if k!='frozen_unix'}=={k:v for k,v in identity.items() if k!='frozen_unix'}
    else:save_json(OUT/'FROZEN_SOURCES.json',identity)
    # Call precisely the accepted preprocessing function; no old build entrypoint.
    sys.path.insert(0,str(BUILDER.parent))
    spec=importlib.util.spec_from_file_location('accepted_data_builder',BUILDER);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sources=[];protected_raw=set();protected_clean=set();selected_raw=set();selected_clean=set()
    with tarfile.open(ARCHIVE,'r') as archive:
        for r in old:
            protected_raw.add(r['original_content_sha256'])
            clean=np.load(Path(cfg['dataset_root'])/r['clean_ref'],allow_pickle=False)
            protected_clean.add(digest(clean.tobytes()))
        for r in val+test:
            raw=archive.extractfile(r['source_member']).read();clean,_=module.preprocess(raw)
            protected_raw.add(digest(raw));protected_clean.add(digest(clean.tobytes()))
        for sid in chosen:
            r=mapping[sid];raw=archive.extractfile(r['source_member']).read();clean,prep=module.preprocess(raw)
            rh,ch=digest(raw),digest(clean.tobytes())
            if rh in protected_raw|selected_raw or ch in protected_clean|selected_clean:
                raise ValueError('Fixed source content duplicate: '+sid)
            selected_raw.add(rh);selected_clean.add(ch)
            ref=f'clean/{sid}.npy';array(OUT/ref,clean)
            sources.append({'source_id':sid,'original_split':'train','clean_ref':ref,'clean_sha256':sha256(OUT/ref),
                'clean_content_sha256':ch,'original_content_sha256':rh,'preprocessing':prep,
                'original_source_ref':{'archive':str(ARCHIVE),'member':r['source_member']}})
    planned=[]
    for s in sources:
        for r in (1,2):
            sid=s['source_id'];tid=f'{Path(sid).stem}_heldout_r{r:02d}'
            planned.append({'trajectory_id':tid,'source_id':sid,'realization_index':r,'clean_ref':s['clean_ref'],
                'clean_sha256':s['clean_sha256'],'seed_zernike':seed(sid,r,'epsilon_zernike'),'seed_tilt':seed(sid,r,'epsilon_tilt'),
                **{b+'_ref':f'xi/{tid}/{b}.npy' for b in ('epsilon_zernike','epsilon_tilt')},
                **{f'x{g}_ref':f'states/{tid}/X_{g}.npy' for g in (4,3,2,1)}})
    assert len({r[k] for r in planned for k in ('seed_zernike','seed_tilt')})==128
    old_seeds={r[k] for r in read_jsonl(Path(cfg['dataset_root'])/'trajectory_manifest.jsonl') for k in ('seed_zernike','seed_tilt')}
    assert not old_seeds&{r[k] for r in planned for k in ('seed_zernike','seed_tilt')}
    rows(OUT/'source_manifest.jsonl',sources);rows(OUT/'planned_trajectories.jsonl',planned)
    save_json(OUT/'PREPARED.json',{'status':'cpu_sources_prepared','source_count':32,'trajectory_count':64,
        'id_and_raw_and_clean_overlap':0,'checked_against':'overfit100, original val/test, within heldout32',
        'source_manifest_sha256':sha256(OUT/'source_manifest.jsonl'),'planned_sha256':sha256(OUT/'planned_trajectories.jsonl'),
        'code_sha256':sha256(__file__),'data_only_no_model':True})

def generate(rank,shards):
    from physical_bridge.simulator import PiRNP2SSimulator,Realization
    import torch_npu
    prep=json.loads((OUT/'PREPARED.json').read_text())
    assert prep['planned_sha256']==sha256(OUT/'planned_trajectories.jsonl')
    cfg=json.loads((ROOT/'shared/original_train_config.json').read_text());verify_source(cfg);configure_npu(cfg)
    torch.npu.set_per_process_memory_fraction(float(os.environ['FOLLOWUP_OWN_CAP_GIB'])/64,0)
    sim=PiRNP2SSimulator(cfg['asset_dir'],**cfg['physics'],device='npu:0').eval().requires_grad_(False)
    planned=read_jsonl(OUT/'planned_trajectories.jsonl')
    with torch.no_grad():
        for n,r in enumerate(planned):
            if n%shards!=rank:continue
            done=OUT/'completed'/f"{r['trajectory_id']}.json"
            if done.exists():
                proof=json.loads(done.read_text());assert all(sha256(OUT/p)==h for p,h in proof['file_hashes'].items());continue
            c=torch.from_numpy(np.load(OUT/r['clean_ref'],allow_pickle=False)).unsqueeze(0).to('npu:0')
            xi=sim.sample_realization(1,seed_zernike=r['seed_zernike'],seed_tilt=r['seed_tilt'])
            for b in ('epsilon_zernike','epsilon_tilt'):array(OUT/r[b+'_ref'],getattr(xi,b).cpu().numpy()[0])
            for g in (4,3,2,1):array(OUT/r[f'x{g}_ref'],sim.render(c,xi,g).output.cpu().numpy()[0])
            # Replay from saved xi and independent render calls, not cached generation outputs.
            loaded=Realization(*(torch.from_numpy(np.load(OUT/r[b+'_ref'],allow_pickle=False)).unsqueeze(0).to('npu:0') for b in ('epsilon_zernike','epsilon_tilt')))
            again=sim.sample_realization(1,seed_zernike=r['seed_zernike'],seed_tilt=r['seed_tilt'])
            assert all(torch.equal(getattr(loaded,b),getattr(again,b)) for b in ('epsilon_zernike','epsilon_tilt'))
            errors=[]
            for g in (4,3,2,1):
                original=np.load(OUT/r[f'x{g}_ref'],allow_pickle=False)
                replay=sim.render(c,loaded,g).output.cpu().numpy()[0];d=replay.astype(np.float64)-original.astype(np.float64)
                metrics={'g':g,'max_abs':float(np.abs(d).max()),'MSE':float((d*d).mean()),'exact':bool(np.array_equal(original,replay))}
                assert metrics['max_abs']<=1e-6 and metrics['MSE']<=1e-12;errors.append(metrics)
            refs=[r[b+'_ref'] for b in ('epsilon_zernike','epsilon_tilt')]+[r[f'x{g}_ref'] for g in (4,3,2,1)]
            save_json(done,{'record':r,'file_hashes':{p:sha256(OUT/p) for p in refs},'seed_replay_exact':True,'replay':errors})
            save_json(OUT/f'progress_rank{rank}.json',{'completed':n,'total':64,'time':time.time()})
    if rank==0:
        deadline=time.time()+7200
        while not all((OUT/'completed'/f"{r['trajectory_id']}.json").exists() for r in planned):
            if time.time()>deadline:raise TimeoutError('Heldout render shards incomplete')
            time.sleep(10)
        finalize()

def finalize():
    planned=read_jsonl(OUT/'planned_trajectories.jsonl');starts=[];proofs=[]
    for r in planned:
        proof=json.loads((OUT/'completed'/f"{r['trajectory_id']}.json").read_text());proofs.append(proof)
        for g in (4,3,2,1):
            starts.append({'start_id':r['trajectory_id']+f'_G{g}','trajectory_id':r['trajectory_id'],'source_id':r['source_id'],
                'initial_G':g,'initial_state_ref':r[f'x{g}_ref'],'clean_ref':r['clean_ref'],'severity_path':list(range(g,0,-1)),'num_predictions':g})
    rows(OUT/'trajectory_manifest.jsonl',planned);rows(OUT/'start_manifest.jsonl',starts)
    data=StartDataset(OUT);assert len(data)==256
    hashes={str(p.relative_to(OUT)):sha256(p) for p in OUT.rglob('*.npy')}
    save_json(OUT/'hash_manifest.json',hashes)
    save_json(OUT/'READY.json',{'status':'ready','sources':32,'trajectories':64,'starts':256,'hash_manifest_sha256':sha256(OUT/'hash_manifest.json'),
        'replay_max_abs':max(m['max_abs'] for p in proofs for m in p['replay']),'seed_replay_exact':True,'replay_states':256,'time':time.time()})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('prepare','generate'));ap.add_argument('--rank',type=int,default=0);ap.add_argument('--shards',type=int,default=1);a=ap.parse_args()
    try:prepare() if a.mode=='prepare' else generate(a.rank,a.shards)
    except Exception as e:
        import traceback
        save_json(OUT/'FAILURE.json',{'error':repr(e),'traceback':traceback.format_exc(),'only_heldout_blocked':True});raise
