#!/usr/bin/env python3
"""Revalidate a resume-only checker repair; reuse provably unchanged execution."""
import ast
import copy
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import traceback
from run_stage2_smokes import (ROOT, REPORT_NAME, LEDGER_NAME, SelectionDriver, evidence,
                              memory_failure_reason, now, read, require, sha256, write)

ARCHIVE='artifacts/STAGE2_RUNTIME_SELECTION_before_resume_validation_fix.json'
INITIAL='artifacts/stage2_ddp_gradients_mb1_resume03/results.json'

def training_fingerprint(text):
    tree=ast.parse(text)
    tree.body=[n for n in tree.body if not (isinstance(n,ast.FunctionDef) and n.name=='compare_resume_reexecution')]
    for node in ast.walk(tree):
        if isinstance(node,ast.If) and ast.unparse(node.test)=="args.mode == 'resume-check'":
            node.body=[ast.Pass()]
    return ast.dump(tree,include_attributes=False)

def checked(root,record):
    p=Path(record['path']).resolve()
    require(p.is_relative_to(root.resolve()),'Evidence path leaves project')
    require(sha256(p)==record['sha256'],f'Changed preserved evidence: {p}')
    return read(p) if p.suffix=='.json' else p.read_text()

class ResumeValidationRecovery(SelectionDriver):
    def __init__(self,root,previous):
        super().__init__(root,'resume03',Path(root)/INITIAL)
        self.previous=previous
        self.legacy_initial=checked(self.root,previous['initial_ddp_normalization'])
        self.report.update(maximum_new_smoke_reservation=3,
                           recovery={'prior_report':evidence(self.root/ARCHIVE),
                                     'scope':'resume-check acceptance only; production/performance AST unchanged',
                                     'policy':evidence(self.root/'configs/resume_acceptance_policy.json'),
                                     'source':evidence(Path(__file__).resolve()),
                                     'all_previous_failures_preserved':True})
    def verify_prerequisites(self):
        super().verify_prerequisites()
        require(self.previous['status']=='blocked' and self.previous['phase']=='resume','Unexpected prior failure')
        old=self.root/'artifacts/stage2_before_resume_debug/train_worker.py'
        old_hash=self.legacy_initial['rank_results'][0]['identity']['actual_code_sha256']['code/train_worker.py']
        require(sha256(old)==old_hash,'Original worker snapshot does not match accepted evidence')
        require(training_fingerprint(old.read_text())==training_fingerprint((self.root/'code/train_worker.py').read_text()),
                'Code outside resume-only validation changed; reuse is forbidden')
        require(self.previous['candidate_configs']==self.report['candidate_configs'],'Candidate configuration changed')
        legacy=self.legacy_initial['rank_results'][0]['identity']['actual_code_sha256']
        current=self.initial_ddp['rank_results'][0]['identity']['actual_code_sha256']
        require(legacy.keys()==current.keys() and all(legacy[k]==current[k] for k in legacy if k!='code/train_worker.py'),
                'A mathematical/runtime module changed beyond the validation-only repair')
        ledger=read(self.root/LEDGER_NAME)
        require(ledger['conservative_consumed_updates']+3<=60,'Insufficient smoke budget')
        tests=read(self.root/'artifacts/STAGE2_RESUME_REEXECUTION_TESTS.json')
        require(tests['status']=='passed' and tests['tests_run']==8,'Typed resume acceptance tests failed')
        require(tests['worker_sha256']==sha256(self.root/'code/train_worker.py'),'Tested validation bytes changed')
        for rank in range(4):
            boundary=read(self.root/f'artifacts/stage2_resume_mb1_hccl02/resume_boundary_parity_rank{rank}.json')
            require(boundary['exact'] and boundary['checked_before_any_restored_optimizer_attempt'],
                    'Original checkpoint restoration was not exact')
    def launch_smoke(self,mode,mb,label,measure):
        if mode=='resume-check':
            return super().launch_smoke(mode,mb,label,measure)
        prefix=f'stage2_{label}_mb{mb}_'
        matching=[x for x in self.previous['attempts'] if x['id'].startswith(prefix)
                  and (x['status']=='passed' or (mode=='memory-smoke' and mb==8))]
        require(len(matching)==1,f'Expected exactly one preserved qualified attempt: {prefix}')
        attempt=copy.deepcopy(matching[0]);self.verify_current_code()
        ledger=read(self.root/LEDGER_NAME)
        require(next(x for x in ledger['attempts'] if x['id']==attempt['id'])==attempt['ledger_row'],
                'Prior smoke accounting changed')
        guard=checked(self.root,attempt['guard_evidence'])
        require(guard.get('own_npu_processes_released') is True,'Prior resources were not released')
        output=self.root/'artifacts'/attempt['id']
        if attempt['status']=='passed':
            require(guard['status']=='completed','Prior successful guard did not complete')
            checked(self.root,attempt['result_evidence'])
            for item in attempt['rank_log_evidence']:checked(self.root,item)
            current_initial,current_identity=self.initial_ddp,self.common_identity
            try:
                # Validate untouched old records under their actual original byte
                # identity. Never rewrite old records to claim the new worker hash.
                self.initial_ddp=self.legacy_initial
                self.common_identity=dict(self.legacy_initial['rank_results'][0]['identity'])
                self.common_identity.pop('numerical_config_sha256')
                result,logs=self.validate_success(output,mode,mb,measure)
            finally:
                self.initial_ddp=current_initial
                self.common_identity=dict(current_initial['rank_results'][0]['identity'])
                self.common_identity.pop('numerical_config_sha256')
            answer={'passed':True,'result':result,'logs':logs,'attempt':attempt}
        else:
            require(mode=='memory-smoke' and mb==8,'Only explicit original memory OOM is reusable')
            errors=[read(p) for p in output.glob('error_rank*.json')]
            reason=memory_failure_reason(guard,errors,(self.root/'runs'/attempt['id']/'worker.log').read_text())
            require(reason and reason==attempt.get('candidate_exclusion_reason'),'OOM reason changed')
            answer={'passed':False,'reason':reason,'attempt':attempt}
        attempt['reused_after_validation_only_change']=True
        self.report['attempts'].append(attempt);self.persist()
        return answer

def main():
    with (ROOT/'.stage2_selection_driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (ROOT/'STOP_REQUESTED').exists(),'Project stop marker is present')
        require(not (ROOT/ARCHIVE).exists(),'This explicit recovery already exists')
        previous=read(ROOT/REPORT_NAME);write(ROOT/ARCHIVE,previous)
        name='stage2_ddp_gradients_mb1_resume03'
        command=[sys.executable,'code/guard_distributed.py','--run-dir',str(ROOT/'runs'/name),
                 '--timeout-seconds','3600','--memory-limit-gib','61','--command','bash','code/run_distributed_env.sh',
                 '-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=4',
                 'code/check_distributed_gradients.py','--config','configs/stage2_candidate_mb1.json',
                 '--output',str(ROOT/'artifacts'/name),'--microbatch','1']
        subprocess.run(command,cwd=ROOT,stdin=subprocess.DEVNULL,check=True)
        guard=read(ROOT/'runs'/name/'guard_result.json')
        require(guard['status']=='completed' and guard['own_npu_processes_released'],'Final-byte gradient check failed')
        driver=ResumeValidationRecovery(ROOT,previous)
        try:
            driver.run()
            require(driver.report['selected_microbatch']==previous['provisional_selected_microbatch'],
                    'Unchanged performance evidence changed the runtime choice')
            p=ROOT/'artifacts/STAGE2_HCCL_TIMEOUT_REPAIR.json';repair=read(p)
            repair.update(status='verified_by_completed_runtime_selection',verified_utc=now(),
                          retest=evidence(ROOT/'artifacts/stage2_performance_mb4_hccl02/results.json'),
                          runtime_selection=evidence(ROOT/REPORT_NAME))
            write(p,repair)
        except BaseException as exc:
            driver.report.update(status='blocked',error=repr(exc),traceback=traceback.format_exc(),completed_utc=now())
            driver.persist();raise

if __name__=='__main__':main()
