"""Per-physical-GPU cooperative optimizer-boundary barrier and evaluation lease.

Every caller CUDA-synchronizes before acknowledging a boundary. No worker starts
another update while any evaluation ticket exists. A peer death is a hard pause.
The JSON state is protected by a POSIX flock; this is not a GPU-memory lock alone.
"""
from contextlib import contextmanager
import fcntl,json,os,time
from pathlib import Path
from .runtime import save_json

class EvaluationMutex:
    def __init__(self, directory, group, gpu, members, sync, timeout=14400):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.group=group;self.gpu=gpu;self.members=members;self.sync=sync;self.timeout=timeout
        self.path=self.directory/f'gpu{gpu}.json'
        self.lock_path=self.directory/f'gpu{gpu}.lock'
        self.events=self.directory/f'gpu{gpu}.events.jsonl'
        with self.locked() as state:
            previous=state['workers'].get(group)
            if previous and self.alive(previous['pid']):raise RuntimeError('Duplicate live worker')
            if state.get('failed'):raise RuntimeError('Failed mutex requires reviewed recovery in new session')
            state['workers'][group]={'pid':os.getpid(),'status':'REGISTERED','update':None,'exposures':None}
            self.event('register',state)

    @staticmethod
    def alive(pid):
        try:os.kill(pid,0);return True
        except ProcessLookupError:return False

    @contextmanager
    def locked(self):
        with self.lock_path.open('a+') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            state=json.loads(self.path.read_text()) if self.path.exists() else {
                'workers':{},'queue':[],'owner':None,'next_ticket':0,'failed':None,'committed':[]}
            try:yield state
            finally:
                save_json(self.path,state)
                fcntl.flock(handle,fcntl.LOCK_UN)

    def event(self, kind, state, **extra):
        entry={'time_ns':time.time_ns(),'event':kind,'group':self.group,'gpu':self.gpu,
               'owner':state['owner'],'workers':{g:dict(w) for g,w in state['workers'].items()},**extra}
        with self.events.open('a') as f:f.write(json.dumps(entry)+'\n')

    def health(self,state):
        if state['failed']:raise RuntimeError(f'Mutex failed: {state["failed"]}')
        for g,w in state['workers'].items():
            if w['status']!='TERMINAL' and not self.alive(w['pid']):
                state['failed']=f'Peer {g} died outside terminal boundary'
                self.event('peer_died',state);raise RuntimeError(state['failed'])

    def boundary(self, update, evaluation_id=None):
        self.sync()
        with self.locked() as state:
            self.health(state)
            worker=state['workers'][self.group]
            worker.update(status='PAUSED',update=update,exposures=32*update)
            if evaluation_id is not None:
                key=f'{self.group}:{evaluation_id}'
                if key in state['committed']:raise RuntimeError('Duplicate committed evaluation request')
                if any(q['group']==self.group for q in state['queue']):raise RuntimeError('Duplicate pending ticket')
                state['queue'].append({'group':self.group,'id':evaluation_id,'ticket':state['next_ticket'],
                                       'priority':self.members.index(self.group)})
                state['next_ticket']+=1
            self.event('boundary_ack',state,evaluation_id=evaluation_id)
        started=time.monotonic()
        while True:
            with self.locked() as state:
                self.health(state)
                if set(state['workers'])==set(self.members):
                    if state['owner']==self.group:
                        state['workers'][self.group]['status']='EVALUATING'
                        self.event('lease_acquired',state);return True
                    if state['owner'] is None:
                        if state['queue']:
                            ready=all(w['status'] in ('PAUSED','TERMINAL') for w in state['workers'].values())
                            if ready:
                                ticket=min(state['queue'],key=lambda q:(q['ticket'],q['priority']))
                                state['owner']=ticket['group'];self.event('grant',state,ticket=ticket)
                        else:
                            state['workers'][self.group]['status']='RUNNING'
                            self.event('update_permitted',state);return False
            if time.monotonic()-started>self.timeout:
                self.fail('Boundary ACK/evaluation timeout');raise TimeoutError('Evaluation mutex timeout')
            time.sleep(.1)

    def complete(self, evaluation_id):
        self.sync()
        with self.locked() as state:
            self.health(state)
            assert state['owner']==self.group
            ticket=[q for q in state['queue'] if q['group']==self.group]
            assert len(ticket)==1 and ticket[0]['id']==evaluation_id
            state['committed'].append(f'{self.group}:{evaluation_id}')
            state['queue'].remove(ticket[0]);state['owner']=None
            state['workers'][self.group]['status']='PAUSED'
            self.event('evaluation_committed_release',state,evaluation_id=evaluation_id)

    def terminal(self,update):
        self.sync()
        with self.locked() as state:
            self.health(state)
            assert state['owner']!=self.group
            assert not any(q['group']==self.group for q in state['queue'])
            state['workers'][self.group].update(status='TERMINAL',update=update,exposures=32*update)
            self.event('terminal_ack',state)

    def park(self,update):
        """Stage wait ACK without retaining a lease or blocking peer evaluation."""
        self.sync()
        with self.locked() as state:
            self.health(state)
            assert state['owner']!=self.group
            assert not any(q['group']==self.group for q in state['queue'])
            state['workers'][self.group].update(status='PAUSED',update=update,exposures=32*update)
            self.event('stage_park_ack',state)

    def fail(self,reason):
        with self.locked() as state:
            state['failed']=str(reason)
            state['workers'][self.group]['status']='FAILED'
            self.event('failure_pause',state,reason=str(reason))
