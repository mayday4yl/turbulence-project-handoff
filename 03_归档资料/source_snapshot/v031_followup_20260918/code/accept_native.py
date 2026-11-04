"""New semantics only, on original real FP32 samples/backend. Zero updates."""
from pathlib import Path
import argparse,json,os
import torch
from physical_bridge.model import DualHeadUNet
from physical_bridge.simulator import PiRNP2SSimulator
from v031.runtime import configure_npu,verify_source,state_hash,save_json
from v031.followup import CleanOnly,RolloutModule
from v031.data import StartDataset
from v031.core import full_image_mse,severity_path
from v031.training import memory_stats

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--task',choices=('T1_DIRECT','T2_FEEDBACK_R'),required=True)
    ap.add_argument('--output',required=True);a=ap.parse_args()
    cfg=json.loads((ROOT/'configs'/f'{a.task}_mb1.json').read_text())
    import torch_npu
    configure_npu(cfg);torch.npu.set_per_process_memory_fraction(float(os.environ['FOLLOWUP_OWN_CAP_GIB'])/64,0)
    verify_source(cfg);torch.random.default_generator.manual_seed(cfg['seed']);torch.npu.manual_seed(cfg['seed'])
    model=DualHeadUNet(**cfg['network']);model.load_state_dict(torch.load(cfg['initial_state_path'],map_location='cpu',weights_only=True))
    assert state_hash(model)==cfg['expected_full_initial_network_sha256']
    model=model.to('npu:0');model.followup_task=a.task
    data=StartDataset(cfg['dataset_root']);chosen=[next(i for i,r in enumerate(data.records) if r['initial_G']==g) for g in (4,3,2,1)]
    samples=[data[i] for i in chosen]
    c=torch.stack([x['clean'] for x in samples]).to('npu:0');y=torch.stack([x['observation'] for x in samples]).to('npu:0');g=y.new_tensor([4,3,2,1])
    result={'task':a.task,'optimizer_updates':0,'start_ids':[x['start_id'] for x in samples]}
    if a.task=='T1_DIRECT':
        small=CleanOnly(model)
        with torch.no_grad():reference=model(y,g).clean;actual=small(y,g)
        result['max_abs']=float((actual-reference).abs().max());result['MSE']=float((actual-reference).square().mean())
        assert torch.allclose(actual,reference,atol=1e-6,rtol=1e-5)
        out=RolloutModule(small,torch.nn.Identity())(c,y,g)
        independent=(small(y,g)-c).square().flatten(1).mean(1)
        assert torch.allclose(out['loss_mean'],independent.mean(),atol=1e-7,rtol=1e-6)
        out['loss_mean'].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in small.parameters())
        assert out['prediction_count']==4 and out['logical_render_count']==0
        result.update(clean_only_equivalent=True,full_image_mean=True,all_required_gradients_connected=True)
    else:
        sim=PiRNP2SSimulator(cfg['asset_dir'],**cfg['physics'],device='npu:0').eval().requires_grad_(False)
        # Native G4 full graph; mixed-G denominator is independently covered by DDP32 acceptance.
        out=RolloutModule(model,sim)(c[:1],y[:1],g[:1],retain_intermediates=True)
        tr=out['trace'];terminal=tr[-1]['clean_losses'].sum()
        gc=torch.autograd.grad(terminal,[x['clean'] for x in tr[:-1]],retain_graph=True)
        assert all(torch.isfinite(x).all() and x.norm()>0 for x in gc)
        gx=torch.autograd.grad(terminal,[x[k] for x in tr[:-1] for k in ('epsilon_zernike','epsilon_tilt')],allow_unused=True,retain_graph=True)
        assert all(x is None for x in gx)
        own=[]
        for x in tr:
            gg=torch.autograd.grad(x['reconstruction_losses'].sum(),[x[k] for k in ('clean','epsilon_zernike','epsilon_tilt')],retain_graph=True)
            assert all(torch.isfinite(v).all() for v in gg)
            own.append([float(v.norm()) for v in gg])
        final_value=tr[-1]['clean'].detach();del out,tr,terminal,gc,gx,x,gg
        torch.npu.empty_cache()
        negative=[]
        for cut in range(3):
            z=y[:1];predictions=[]
            for k,s in enumerate((4,3,2,1)):
                p=model(z,z.new_tensor([s]));predictions.append(p.clean)
                z=p.clean.detach() if k==cut else p.clean
            loss=full_image_mse(predictions[-1],c[:1]).mean()
            assert torch.autograd.grad(loss,predictions[cut],allow_unused=True)[0] is None
            assert torch.allclose(predictions[-1],final_value,atol=1e-6,rtol=1e-5)
            negative.append(True)
        result.update(clean_feedback_connected=True,terminal_to_early_xi_absent=True,
                      own_LR_input_gradient_norms=own,negative_controls=negative)
    result.update(status='passed',memory=memory_stats(torch.device('npu:0')))
    save_json(a.output,result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
