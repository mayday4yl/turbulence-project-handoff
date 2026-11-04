"""RAW full train8000 backfill from a saved milestone; zero training updates."""
import argparse,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'deps'),str(ROOT/'scripts')]
os.environ['TORCH_HOME']=str(ROOT/'assets/torch')
import torch
from v4.matched import Store,build
from v4.runtime import configure_cuda,sha256,save_json
from v4.evaluate_matched import Evaluator
from v4.refinement import module_hash
from train_worker import verify_frozen

def main(a):
    frozen=verify_frozen();configure_cuda(20260929);run=Path(frozen['results_root'])
    cfg=json.loads((ROOT/'configs/runtime.json').read_text());store=Store(cfg)
    path=run/a.group/'checkpoints'/f'weights_U{a.update:05d}.pt';identity=json.loads(path.with_suffix('.identity.json').read_text())
    assert sha256(path)==identity['sha256']
    model,sim,obj=build(cfg,a.group);saved=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);del saved
    before=module_hash(model);evaluator=Evaluator(store,frozen['protocol_hash'],frozen['environment_identity'])
    destination=run/a.group/'evaluations'/f'U{a.update:05d}'/'raw/train8000'
    evaluator.evaluate(obj,store.train,destination,update=a.update,weights='raw',checkpoint_hash=identity['sha256'])
    assert module_hash(model)==before
    save_json(destination/'ENDPOINT_BACKFILL.json',{'optimizer_updates':0,'checkpoint':str(path),'checkpoint_hash':identity['sha256'],
         'reason':'Final best-val auxiliary full train8000 RAW; val800 reused','evaluation_batch':2,'time':time.time()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--group',required=True);p.add_argument('--update',type=int,choices=list(range(0,2001,250)),required=True);a=p.parse_args()
    try:main(a)
    except BaseException as e:
        save_json(ROOT/'reports/RUN_PAUSED.json',{'gate':'early_endpoint_backfill','error':repr(e),'traceback':traceback.format_exc()});raise
