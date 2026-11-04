"""Final paired source-level results; no fitting, selection, or extra evaluation."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from v4.runtime import save_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def source_scores(result):
    rows=result['summary']['source_G'];ids=sorted({r['source_id'] for r in rows})
    return {sid:sum(r['psnr'] for r in rows if r['source_id']==sid)/4 for sid in ids}


def main():
    protocol=json.loads((ROOT/'protocol/FROZEN.json').read_text());base=Path(protocol['comparison_dir'])
    results={};training={}
    for group in ('D0','R0','R08'):
        complete=json.loads((base/group/'COMPLETE.json').read_text())
        assert complete['E']==32000 and complete['protocol_hash']==protocol['protocol_hash']
        results[group]=json.loads((base/group/'evaluation/E32000_val800_raw_ref0/results.json').read_text())
        training[group]=json.loads((base/group/'evaluation/E32000_train1600_raw_ref0/results.json').read_text())
        assert results[group]['protocol_hash']==protocol['protocol_hash']
    pairs={}
    for a,b in [('R0','D0'),('R08','R0'),('R08','D0')]:
        first,second=source_scores(results[a]),source_scores(results[b]);assert first.keys()==second.keys()
        delta={sid:first[sid]-second[sid] for sid in first}
        pairs[f'{a}-{b}']={'mean_dB':sum(delta.values())/len(delta),
                          'source_improvement_fraction':sum(x>0 for x in delta.values())/len(delta),
                          'per_source_dB':delta}
    refinement={}
    for group in ('R0','R08'):
        on=json.loads((base/group/'evaluation/E32000_val800_raw_ref1/results.json').read_text())
        assert on['protocol_hash']==protocol['protocol_hash'] and on['checkpoint_identity']==results[group]['checkpoint_identity']
        first,second=source_scores(on),source_scores(results[group]);delta={sid:first[sid]-second[sid] for sid in first}
        refinement[group]={'mean_dB':sum(delta.values())/len(delta),'per_source_dB':delta,
                           'source_improvement_fraction':sum(v>0 for v in delta.values())/len(delta),
                           'on_summary':on['summary'],'on_seconds':on['seconds'],'off_seconds':results[group]['seconds']}
    totals={}
    for group in ('D0','R0','R08'):
        updates=[json.loads(line) for line in (base/group/'updates.jsonl').read_text().splitlines()]
        # Latest occurrence of a committed update wins after an explicitly logged checkpoint restart.
        committed={r['E']:r for r in updates if r['E']<=32000}
        assert sorted(committed)==list(range(protocol['batch'],32001,protocol['batch']))
        counts={g:sum(r['by_G'][g-1][0] for r in committed.values()) for g in (1,2,3,4)}
        assert set(counts.values())=={8000.}
        totals[group]={'updates':len(committed),'exposures':32000,'successful_update_seconds':sum(r['seconds'] for r in committed.values()),
                       'peak_GiB':max(r['peak_GiB_per_card'] for r in committed.values()),
                       'network_calls':32000 if group=='D0' else 80000,
                       'training_simulator_renders':32000 if group=='D0' else 128000}
    save_json(base/'PAIRED_RESULTS.json',{'status':'COMPLETE','protocol_hash':protocol['protocol_hash'],
        'train':{k:v['summary'] for k,v in training.items()},'val':{k:v['summary'] for k,v in results.items()},
        'pairs':pairs,'refinement':refinement,'costs':totals,
        'limitations':['one training seed','one evaluation eta mapping','equal exposures not equal compute',
                       '20 epochs does not establish convergence','CUDA backward not bitwise deterministic',
                       'R08-R0 changes mixing and t conditioning; not isolated Gaussian effect']})
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    colors={'D0':'#555555','R0':'#1678b8','R08':'#d06b1e'}
    for axis,list_name in zip(axes,('train64','val64')):
        for group in ('D0','R0','R08'):
            for weight,style in [('raw','-'),('ema','--')]:
                points=[]
                for path in sorted((base/group/'evaluation').glob(f'E*_{list_name}_{weight}_ref0/results.json')):
                    item=json.loads(path.read_text());points.append((item['E'],item['summary']['overall']['psnr']))
                points.sort();axis.plot([p[0] for p in points],[p[1] for p in points],style,color=colors[group],label=f'{group} {weight}')
        axis.set(title=list_name,xlabel='Start-record exposures',ylabel='PSNR (dB)');axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.savefig(base/'learning_curves.png',dpi=180);plt.close(fig)
    with (base/'last32k_source_G.csv').open('w') as stream:
        fields=['group','source_id','G','psnr','input_psnr','gain_over_input','first_psnr','progressive_gain']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for group,result in results.items():
            for row in result['summary']['source_G']:writer.writerow({'group':group,**{k:row[k] for k in fields if k!='group'}})
    lines=['# V4 Overfit100 paired comparison','',f"Protocol: `{protocol['protocol_hash']}`",'',
           '| Group | Train PSNR | Val PSNR | Val gain over Y |','|---|---:|---:|---:|']
    for group in ('D0','R0','R08'):
        t=training[group]['summary']['overall'];v=results[group]['summary']['overall']
        lines.append(f"| {group} | {t['psnr']:.4f} | {v['psnr']:.4f} | {v['gain_over_input']:+.4f} |")
    lines+=['','Paired source-level differences:']
    for name,value in pairs.items():lines.append(f"- {name}: {value['mean_dB']:+.4f} dB; improved sources {value['source_improvement_fraction']:.1%}.")
    lines+=['','Refinement, last_32k raw, identical data/eta mapping:']
    for name,value in refinement.items():lines.append(f"- {name}: on minus off {value['mean_dB']:+.4f} dB.")
    lines+=['','All three runs completed 32,000 exposures. Training never used refinement. Test200 was not used.',
            'Single training seed and one evaluation eta mapping; no convergence, multi-seed robustness, isolated Gaussian-noise benefit or equal-compute claim.',
            'Raw and EMA curves, per-source data and complete metrics accompany this report. No additional training is authorized by completion.']
    (base/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    save_json(base/'STATUS.json',{'state':'COMPLETE','protocol_hash':protocol['protocol_hash'],
                               'groups':['R08','R0','D0'],'E_per_group':32000,'report':'FINAL_REPORT.md'})


if __name__=='__main__':main()
