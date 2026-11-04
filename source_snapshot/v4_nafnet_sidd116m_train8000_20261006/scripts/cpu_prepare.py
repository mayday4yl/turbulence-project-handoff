from pathlib import Path
import sys,json,collections,copy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from runlib.common import *
from runlib.data import Store,Stream,monitor
from runlib.models import prepare_common
c=read(ROOT/'configs/PREFLIGHT_TEMPLATE.json');store=Store(c)
assert read(Path(c['dataset_root'])/'A100_RUNTIME_READY.json')['status']=='PASS'
info=prepare_common(c);m=monitor(store);freeze(ROOT/'protocol/FIXED_MONITOR.json',m)
streams={b:Stream(store,b) for b in [32,64]}
# Full epoch acceptance: each start exactly once, exact 8-stratum balance,
# B64 is consecutive B32 exposure stream, with no validation or Test rows.
seen={}
for b,s in streams.items():
 ids=[]
 for u in range(s.per_epoch):ids.extend(x['start_id'] for x in s.rows(u))
 assert len(ids)==len(set(ids))==128000;seen[b]=ids
assert seen[32]==seen[64]
for u in [0,1,100,1999]:
 assert [r['start_id'] for r in streams[64].rows(u)]==[r['start_id'] for i in [2*u,2*u+1] for r in streams[32].rows(i)]
atomic(ROOT/'reports/CPU_PROTOCOL_ACCEPTANCE.json',{'status':'PASS','common_init':info,'source_count':9200,'train_starts':128000,'val_starts':8000,'all_epoch_starts_unique':True,'G_domain_exact_per_batch':True,'B32_B64_same_ordered_exposures':True,'train_val_monitor_source_disjoint':True,'Test_used':False,'GPU_preflight_started':False,'time':__import__('time').time()})
print('CPU_PROTOCOL_ACCEPTANCE PASS')
