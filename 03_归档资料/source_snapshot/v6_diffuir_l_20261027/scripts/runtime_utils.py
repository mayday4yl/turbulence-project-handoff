"""V6 process ownership, physical HBM and immutable-run helpers."""
from pathlib import Path
import hashlib,json,os,re,subprocess,time

ROOT=Path(__file__).resolve().parents[1]
GIB=1<<30

def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):result.update(block)
    return result.hexdigest()

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,path)

def snapshot():
    # Same npu-smi table schema verified by the protected Stage1 benchmark.
    raw=subprocess.check_output(['npu-smi','info'],text=True,timeout=20)
    cards={};processes={};pending=None
    for line in raw.splitlines():
        head=re.match(r'^\|\s*(\d+)\s+910B3\s*\|',line)
        if head:pending=int(head[1]);continue
        if pending is not None:
            pairs=re.findall(r'(\d+)\s*/\s*(\d+)',line)
            if pairs and int(pairs[-1][1])>0:
                used,total=map(int,pairs[-1]);cards[pending]={'used':used*(1<<20),'total':total*(1<<20),'free':(total-used)*(1<<20)};pending=None
        proc=re.match(r'^\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|[^|]*\|\s*(\d+)\s*\|',line)
        if proc:
            card,pid,mb=map(int,proc.groups());processes.setdefault(card,[]).append({'pid':pid,'bytes':mb*(1<<20)})
    if len(cards)!=8:raise RuntimeError('Cannot safely parse NPU inventory')
    return {'unix':time.time(),'cards':cards,'processes':processes,'raw':raw}

def process_identity(pid):
    p=Path(f'/proc/{pid}')
    return {'pid':pid,'start_ticks':p.joinpath('stat').read_text().split()[21],'cmdline':p.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode()}

def is_same_process(identity):
    try:
        stat=Path(f"/proc/{identity['pid']}/stat").read_text().split()
        return stat[2]!='Z' and process_identity(identity['pid'])['start_ticks']==identity['start_ticks']
    except FileNotFoundError:return False

def is_descendant(pid,ancestor):
    if not is_same_process(ancestor):return False
    current=pid
    for _ in range(32):
        if current==ancestor['pid']:return True
        if current<=1:return False
        try:current=int(Path(f'/proc/{current}/stat').read_text().split()[3])
        except FileNotFoundError:return False
    return False
