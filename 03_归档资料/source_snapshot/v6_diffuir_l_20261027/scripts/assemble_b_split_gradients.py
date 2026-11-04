"""CPU-only merge with exact equivalence to actual Pilot U0 and unchanged examples."""
import argparse,json
from pathlib import Path
import torch
from grad_helpers import state_digest
from scripts.runtime_utils import ROOT,sha,write_json
EX=ROOT/'execution_20261027_five_npu'
def read(p):return json.loads(Path(p).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot-run',required=True);a=ap.parse_args();torch.set_num_threads(2)
    p=ROOT/'distributed'/a.pilot_run
    initial,end=read(EX/'B_BRANCH_STAGE_U000000.json'),read(EX/'B_BRANCH_STAGE_U000100.json')
    assert initial['status']==end['status']=='PASS' and len(initial['records'])==len(end['records'])==15
    saved=torch.load(p/'checkpoints/U000000.pt',map_location='cpu',weights_only=False)
    assert state_digest(saved['core'])==initial['model_core_digest']==end['initial_core_digest']
    assert initial['logical_B_update']==0 and end['logical_B_update']==100
    x=sorted(initial['records'],key=lambda r:r['global_probe_position']);y=sorted(end['records'],key=lambda r:r['global_probe_position'])
    for before,after in zip(x,y):
        for k in ['sample_id','source_id','seed','global_probe_position']:assert before[k]==after[k]
        for row in [before,after]:assert row['finite_nonzero'] and all(g['finite'] and g['nonzero'] for g in row['terminal_Full_BPTT'])
    assert all(v['changed_tensors']>0 and v['l2']>0 for v in end['parameter_changes'].values())
    write_json(EX/'B_PILOT_BRANCH_GRADIENTS.json',{'status':'PASS','scope':'same original fixed 15 Train examples at initialization and raw Pilot endpoint; initialization exactly matched actual saved U0',
        'records':x+y,'parameter_changes':end['parameter_changes'],'pilot_run':a.pilot_run,'initial_check_parallel_world_size':4,'endpoint_check_world_size':5,
        'actual_Pilot_U0_core_exact':True,'actual_Pilot_U0_sha256':sha(p/'checkpoints/U000000.pt'),'examples_per_checkpoint':15,
        'additional_optimizer_updates':0,'test_used':False,'source_sha256':sha(__file__)})
    print('Split initial/endpoint gradients, same 15 examples and actual Pilot U0 identity PASS')
if __name__=='__main__':main()
