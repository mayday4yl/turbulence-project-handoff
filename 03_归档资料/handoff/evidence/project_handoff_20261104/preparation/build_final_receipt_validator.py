"""Generate readable local metadata gate from the independently audited package
source. Generation only; no receipt is created, no final state is asserted.
"""
import ast,hashlib
from pathlib import Path
P=Path(__file__).resolve().parent
SOURCE=P/'package_preparation/build_handoff_package.py'
EXPECTED='41ed23a8e75e88ffaf091e47ea2c6d5023e743e515cbef76389c036d9b557fd9'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==EXPECTED
source=SOURCE.read_text();tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='final_readiness')
tries=[n for n in fn.body if isinstance(n,ast.Try)]
assert len(tries)==3
native=tries[0].body
assert isinstance(native[0],ast.Assign) and ast.unparse(native[0]).startswith('report = jload(')
related=tries[1].body
end=next(i for i,n in enumerate(related) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='docqa' for t in n.targets))
related=related[:end]
# SLOTS is a local tuple of only the three upstream scientific/exit/backup
# adapters; document/page/content-refresh are downstream and never fabricated.
header='''"""Strict final experimental metadata gate for the handoff registers.
Copied readable gate statements from package helper SHA41ed23a8, retaining its
native CPU4-role/SUMMARY/Adam/RNG/full-output/figure/exit/backup requirements.
No document QA or package publication: those are downstream independent gates.
Only local small JSON/figures/source SHA; large checkpoints are stat-only.
"""
import hashlib
import importlib.util
from pathlib import Path
PACKAGE_SOURCE_SHA256 = '41ed23a8e75e88ffaf091e47ea2c6d5023e743e515cbef76389c036d9b557fd9'
SOURCE_DERIVATION = 'native final_readiness first try and upstream adapter/exit/backup portion of its second try; exact audited source only'
SLOTS = ('experiment_full_delivery_receipt', 'all_owned_process_exit_audit', 'unique_checkpoint_backup_audit')

def validate(preparation, registry):
    prep = Path(preparation).resolve()
    package_source = prep / 'package_preparation/build_handoff_package.py'
    source_sha = hashlib.sha256(package_source.read_bytes()).hexdigest()
    if source_sha != PACKAGE_SOURCE_SHA256:
        raise ValueError('Reviewed package metadata validator source changed; independent review required')
    spec = importlib.util.spec_from_file_location('_handoff_readonly_package_gate', package_source)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    need, sha, jload = tool.need, tool.sha, tool.jload
    project_path, local_from_remote = tool.project_path, tool.local_from_remote
    CAMPAIGN, DEADLINE = tool.CAMPAIGN, tool.DEADLINE
    NATIVE_SMALL, ORIGINAL_PNGS = tool.NATIVE_SMALL, tool.ORIGINAL_PNGS
    need(registry.get('current_experiment_final_acceptance') is True,
         'A final receipt gate requires explicit final request; false remains preparation')
    receipt = registry.get('final_experiment_receipt')
    related_receipts = registry.get('final_related_receipts', {})
    need(set(related_receipts) == set(SLOTS[1:]), 'Final registers require both exact owned-exit and actual-unique-backup adapter bindings')
    cfg = {'status':'READY_FOR_PACKAGE', 'receipts':{'experiment_full_delivery_receipt':receipt, **related_receipts}}
    proof = {'required_packaging_files':[]}
    for slot in SLOTS:
        path, q = tool.require_adapter(slot, cfg)
        proof[slot] = {'path':str(path), 'sha256':sha(path), 'receipt':q}
        proof['required_packaging_files'] += [str(path)] + [str(project_path(item['path'])) for item in q['evidence_bindings']]
'''
body='\n'.join('    '+line for n in native+related for line in ast.unparse(n).splitlines())
footer='''
    files = []
    for raw in sorted(set(proof['required_packaging_files'])):
        path = project_path(raw)
        files.append({'path':str(path), 'bytes':path.stat().st_size, 'sha256':sha(path),
                      'kind':'fixed_final_figure' if path.suffix.lower() == '.png' else 'final_metadata'})
    return {'status':'PASS_EXPLICIT_NATIVE_FINAL_RECEIPT', 'identities':proof['native_identities'],
            'native_CPU_exit_unix':proof['native_CPU_exit_unix'], 'files':files,
            'package_gate_source':{'path':str(package_source),'sha256':source_sha},
            'adapters':{slot:{'path':proof[slot]['path'],'sha256':proof[slot]['sha256']} for slot in SLOTS},
            'network_calls':0, 'model_calls':0, 'optimizer_updates':0, 'checkpoint_payload_reads':0,
            'scope':'Experimental native final acceptance only; final handoff document/package delivery remain separate'}
'''
out=P/'validate_unidb_final_receipt.py';out.write_text(header+body+'\n'+footer)
ast.parse(out.read_text());print(out,hashlib.sha256(out.read_bytes()).hexdigest())
