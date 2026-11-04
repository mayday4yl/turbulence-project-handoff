"""A/B stage gates and bounded training, called by persistent continuation controller."""
import json,time
from pathlib import Path
from scripts.runtime_utils import ROOT,sha,write_json
EX=ROOT/'execution_20261027_continuation'
def read(p):return json.loads(Path(p).read_text())
def validate_run(p,expected,world=6):
    r=read(p/'RESULT.json');assert r['status']=='PASS'
    assert r.get('completed_updates')==expected
    for rank in range(world):
        q=read(p/f'RESULT_rank{rank}.json');assert q['status']=='PASS' and q.get('finite_grad_rate')==1
    return r

def report_stage(stage,stage_root,gate):
    r=read(stage_root/'RESULT.json');best=read(stage_root/'BEST.json')
    losses=[json.loads(s) for s in (stage_root/'UPDATES_rank0.jsonl').read_text().splitlines()]
    body=f'''# V6 {stage} 正式阶段完成报告

真实完成{r['completed_updates']}次optimizer更新，{r['presentations']}次轨迹呈现。数据Train10000/40K、FP32和完整联合方法保持。工程PASS不代表恢复质量提高。

实测更新吞吐{r['images_per_sec']:.6f}轨迹/秒；全阶段墙钟{r['seconds']/3600:.3f}小时；最低运行时剩余显存{r['min_device_free']/2**30:.3f}GiB。完整逐rank Loss、梯度、时间、显存及数据游标见{stage_root.relative_to(ROOT)}。

固定Val1000、固定4→3→2→1→0四步最终PSNR选出的EMA best为U{best['update']}：PSNR{best['validation']['means']['psnr']:.6f}dB；Step4−Step1={best['validation']['step4_minus_step1_psnr']:.6f}dB。负结果如实记录，不改变预算或Loss。

权重：{best['path']}；SHA256：{best['sha256']}。历史全部checkpoint保留，每250更新保存，full Val1000每1000/终点，Val100仅辅助观察。完整配置/源码/数据/噪声身份与CPU读回均在run内。

未使用Test1000；完整训练图与恢复及Pilot证据在execution_20261027_continuation。详细Gate：{gate.name}。
'''
    (ROOT/'reports'/f'{stage}_AUTHORIZED_TRAINING_REPORT_ZH.md').write_text(body)

def run_remaining(d0_best,run_job,run_cpu,state,stopped):
    for path,expected in read(EX/'SOURCE_MANIFEST.json')['files'].items():
        assert sha(ROOT/path)==expected,('Frozen execution source changed',path)
    previous=d0_best
    for stage,mb,budget,pilot_steps,seed in [('A',4,6000,200,741001),('B',1,10000,100,742001)]:
        if stopped():
            state('USER_STOPPED_BEFORE_NEXT_STAGE',stage=stage);raise SystemExit(0)
        initial=previous['path'];assert sha(initial)==previous['sha256']
        prefix=f'{stage}_authorized'
        if not (EX/f'{stage}_FROZEN.json').exists():
            base={'stage':stage,'microbatch':mb,'accumulation':2,'world_size':6,'effective_batch':mb*12,
                'precision':'FP32','amp':False,'checkpointing':False,'loss':'residual+0.9*Clean_S3+0.1*physics_current_g' if stage=='A' else '0.9*mean4_Clean+0.1*mean4_physics_G4+0.1*TF_residual',
                'lr_peaks':[1e-5,5e-5,5e-5] if stage=='A' else [5e-6,1e-5,1e-5],
                'initial_checkpoint':initial,'initial_checkpoint_sha256':previous['sha256'],'psf_backend':'exact_frozen_psf_adjoint',
                'checkpoint_every':250,'stop_request':str(EX/'STOP_REQUEST.json')}
            configs={m:{**base,'mode':m,'updates':n,'seed':seed+i,'scheduler_total_updates':budget if m=='resume_probe' else n}
                for i,(m,n) in enumerate([('resume_probe',2),('pilot',pilot_steps),('formal',budget)])}
            write_json(EX/f'{stage}_FROZEN.json',configs)
            for mode,cfg in configs.items():write_json(EX/f'{stage}_{mode}.json',cfg)
        configs=read(EX/f'{stage}_FROZEN.json');assert configs['formal']['initial_checkpoint_sha256']==previous['sha256']
        if not (EX/f'G2_{stage}.json').exists():
            p,r=run_job(prefix+'_graph01','audit_ab_authorized.py',['--stage',stage,'--initial-checkpoint',initial],hours=2,devices='2')
            assert r['completed_updates']==3 and r['schedule_buffers_unchanged']
            if stage=='B':assert all(v['finite'] and v['nonzero'] for v in r['terminal_clean_only_cross_stage'])
            source_paths=['method_impl/ab.py','method_impl/ab_data.py','method_impl/continuation_factory.py','method_impl/psf_adjoint.py',
                'optimizations/dispatch_20261027/aclnn_runtime.py','optimizations/dispatch_20261027/guards.py','scripts/audit_ab_authorized.py']
            write_json(EX/f'G2_{stage}.json',{'status':'PASS','scope':f'{stage} actual inherited checkpoint complete-graph real-gradient acceptance',
                'model_code_sha256':{n:sha(ROOT/n) for n in source_paths},'run':p.name,'result_sha256':sha(p/'RESULT.json'),
                'initial_checkpoint_sha256':previous['sha256'],'formal_updates':0})
        if not (EX/f'G3_{stage}.json').exists():
            p,r=run_job(prefix+'_batch_stable100_01','bench_ab_authorized.py',['--stage',stage,'--batch',str(mb),'--accumulation','2','--steps','100',
                '--guard-mode','batched','--task-queue','1','--runtime-mode','aclnn','--initial-checkpoint',initial],hours=2)
            assert r['completed_updates']==103 and r['timed_updates']==100 and r['rank_parameters_equal_exact']
            validate_run(p,103)
            write_json(EX/f'G3_{stage}.json',{'status':'PASS','scope':'actual inherited checkpoint complete joint-loss 6-card batch100 stability',
                'microbatch':mb,'accumulation':2,'world_size':6,'effective_batch':mb*12,'result':r,'run':p.name,'result_sha256':sha(p/'RESULT.json')})
        if not (EX/f'{stage}_DDP_RESUME_AUDIT.json').exists():
            ref=prefix+'_resume_reference01';res=prefix+'_resume_restarted01'
            run_job(ref,'run_ab_authorized.py',['--config',EX/f'{stage}_resume_probe.json'],hours=2)
            run_job(res,'run_ab_authorized.py',['--config',EX/f'{stage}_resume_probe.json','--resume',ROOT/'distributed'/ref/'checkpoints/U000001.pt'],hours=2)
            run_cpu('verify_ab_authorized_resume.py',['--stage',stage,'--reference',ref,'--resumed',res])
        assert read(EX/f'{stage}_DDP_RESUME_AUDIT.json')['status']=='PASS'
        if not (EX/f'{stage}_PILOT.json').exists():
            name=prefix+f'_pilot{pilot_steps}_01'
            p,r=run_job(name,'run_ab_authorized.py',['--config',EX/f'{stage}_pilot.json'],hours=5)
            validate_run(p,pilot_steps)
            if not (EX/f'{stage}_PILOT_BRANCH_GRADIENTS.json').exists():
                run_job(prefix+'_pilot_branch_gradients01','diagnose_ab_authorized_pilot.py',['--pilot-run',name],hours=2)
            gradients=read(EX/f'{stage}_PILOT_BRANCH_GRADIENTS.json');assert gradients['status']=='PASS'
            vals=[read(p/'validation'/f'U{u:06d}_1000/SUMMARY.json') for u in [0,pilot_steps]]
            assert all(v['count']==1000 and v['selection_scope']=='fixed Val1000 Step4 final PSNR' for v in vals)
            if stage=='B':assert all(all(x['finite'] and x['nonzero'] for x in z['terminal_Full_BPTT']) for z in gradients['records'])
            write_json(EX/f'{stage}_PILOT.json',{'status':'PASS','scope':'engineering only; no recovery quality continuation threshold',
                'run':name,'updates':pilot_steps,'pilot_weights_inherited':False,'result_sha256':sha(p/'RESULT.json'),'gradients_sha256':sha(EX/f'{stage}_PILOT_BRANCH_GRADIENTS.json'),
                'full_val_u0':vals[0],'full_val_endpoint':vals[1]})
            (ROOT/'reports'/f'{stage}_AUTHORIZED_PILOT_REPORT_ZH.md').write_text(f'# {stage} 独立Pilot\n\n{pilot_steps}次真实完整联合更新、初终完整Val1000、分支及参数更新诊断PASS。临时Pilot权重不继承到正式阶段；恢复质量仅记录。原件见execution_20261027_continuation/{stage}_PILOT.json及对应run。\n')
        p,r=run_job(prefix+f'_formal{budget}_01','run_ab_authorized.py',['--config',EX/f'{stage}_formal.json'],hours=30)
        validate_run(p,budget)
        best=read(p/'BEST.json');assert sha(best['path'])==best['sha256']
        assert best['validation']['full_val1000'] and best['validation']['physical_path']==[4,3,2,1,0]
        gate=EX/f'{stage}_FORMAL_GATE.json';write_json(gate,{'status':'PASS','scope':'full authorized budget completed, engineering valid; recovery quality reported separately',
            'updates':budget,'run':p.name,'best':best,'result_sha256':sha(p/'RESULT.json')})
        report_stage(stage,p,gate);previous=best
    run_cpu('finalize_authorized_project.py')
