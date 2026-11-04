"""Apply actual native-final metadata to UniDB handoff records. Local only.
No models, training, image generation, network or checkpoint payload reads.
Scientific old D0/forensic/noon/18 records and old pending snapshot stay intact.
"""
import csv,datetime,hashlib,importlib.util,json
from pathlib import Path
P=Path(__file__).resolve().parent
PROJECT=P.parent.parent
ROOT=PROJECT/'unidb_dual_20261104_until00'
ARCHIVE=P/'register_preparation_snapshot_20261104'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bound(path):
 path=Path(path);return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
def load(path):return json.loads(Path(path).read_text())
def local(path):
 p=Path(path)
 if p.is_relative_to('/data2/yanglin'):p=PROJECT/p.relative_to('/data2/yanglin')
 return p

def run():
 upstream=ROOT/'delivery/FINAL_RECEIPT_REGISTRY_INPUT.json';request=load(upstream)
 # Actual checked chain; this cannot be satisfied by the bool alone.
 spec=importlib.util.spec_from_file_location('native_final_gate',P/'validate_unidb_final_receipt.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
 proof=gate.validate(P,request)
 assert proof['status']=='PASS_EXPLICIT_NATIVE_FINAL_RECEIPT'
 cpu=load(ROOT/'evidence/CPU_POSTPROCESS_ACCEPTANCE.json');audit=load(ROOT/'evidence/FINAL_OUTPUTS_CPU_AUDIT.json')
 exitq=load(request['final_related_receipts']['all_owned_process_exit_audit']['path']);backups=load(request['final_related_receipts']['unique_checkpoint_backup_audit']['path'])
 assert exitq['historical_union_identity_count']==534 and exitq['historical_identity_set_difference_verified']
 assert backups['unique_file_count']==3 and backups['logical_role_count']==4
 rows=list(csv.DictReader((ROOT/'report/FULL_QUALITY_COMPARISON.csv').open()))
 def row(line,role,g):return next(x for x in rows if x['line']==line and x['role']==role and float(x['g'])==g)
 for ref in cpu['refs']:
  summary=load(local(ref['summary']));assert sha(local(ref['summary']))==ref['summary_sha256']
  for g,v in summary['by_g'].items():
   x=row(ref['branch'],'latest_full' if ref['role']=='latest' else 'best_full',float(g))
   for k in ['psnr','ssim','mae','lpips']:assert float(x[k])==v[k]
 base=(ARCHIVE/'unidb_history.md').read_text().split('## 3＋5双路线：')[0]
 base=base.replace('# UniDB 主线交接工作稿','# UniDB 主线交接说明')
 base=base.replace('证据整理日期：2026-11-04。此稿的历史部分来自已交付原件；双路线仅确认午夜安全停训，最终质量、CPU验收与交付尚待补齐。','证据整理日期：2026-11-04。D0和封存forensic保留原历史边界；双路线已完成停训、最终评价、独立CPU验收、全部自有进程退出和实际unique完整权重备份。实验原件交接与全项目主文档、资料包和聊天交付分别记录，后者不由本章自动宣称完成。')
 labels=[('Input','input_baseline','Input'),('U0','external_fixed_baseline','官方SR U0'),('parentFinal','external_fixed_baseline','父D0 Final U45411'),('original','latest_full','原路线best=latest U67180'),('improved','best_full','改进full-best U10000'),('improved','latest_full','改进latest U40848')]
 table='|同G4 Val1000 EMA|PSNR↑|SSIM↑|MAE↓|LPIPS↓|\n|---|---:|---:|---:|---:|\n'
 for line,role,label in labels:
  x=row(line,role,4);table+='|'+label+'|'+ '|'.join(f'{float(x[k]):.9f}' for k in ['psnr','ssim','mae','lpips'])+'|\n'
 five='|G 每档n1000|Input|官方SR U0|父D0 U45411|原best/latest U67180|改best U10000|改latest U40848|\n|---:|---:|---:|---:|---:|---:|---:|\n'
 for g in [.5,1,2,3,4]:five+='|'+str(g)+'|'+'|'.join(f'{float(row(line,role,g)["psnr"]):.6f}' for line,role,label in labels)+'|\n'
 ids=proof['identities'];orig=row('original','latest_full',4);inp=row('Input','input_baseline',4)
 tail=f'''## 3＋5双路线与最终实验验收

原路线从D0 U45411完整状态继续；改进路线从官方SR U0重开，并非从退化后的D0优化器继续。随后两次延期均在独立root完整验收接续，旧源和历史结果不覆盖。

|阶段|原路线起止U|改进路线起止U|各阶段角色|
|---|---:|---:|---|
|until_noon|45411→55760|0→20000|原3卡保持L1/LR1e-5；改5卡epsilon_opt MSE，200更新warmup到2e-6，墙钟cosine降到2e-7、clip1|
|until18|55760→61764|20000→30000|全部状态恢复；改进固定既有floor2e-7，不重warmup|
|until00|61764→67180|30000→40848|全部状态及祖先调度保持；午夜安全停训，终评、CPU验收、退出和实际完整权重备份均完成|

两线B96/FP32、原生网络和桥、等比例G1–G4保持。原3物理rank保留八逻辑rank RNG语义，改进保留五物理rank；同3/5真实恢复+重启工程门通过后丢弃门内更新，从验收父正式接续。两路线起点、loss、LR与卡数均不同，是综合路线尝试，不能作单一loss或LR的因果消融。

午夜最终更新的8个rank均在1793721600之前开始，只完成在途更新后以`reason=deadline`结束。原U67180/cursor6449280，改U40848/cursor3921408。原路线累计6449280呈现、相当于40.308轮160K组合，U45411之后新增21769更新/2089824呈现/13.0614轮；改进累计40848更新/3921408呈现/24.5088轮。午夜延长段分别5416更新/519936呈现与10848更新/1041408呈现。这里是样本重复呈现的等效轮数，不表示独立内容数量，也不是收敛结论。

### 实际最终角色与全量质量

原路线best和latest都为U67180，SHA `{ids['original:latest']['sha256']}`。改进latest为U40848，SHA `{ids['improved:latest']['sha256']}`；eligible full-best保留noon U10000，SHA `{ids['improved:best']['sha256']}`。四逻辑角色只对应三个unique完整checkpoint和三个unique全量验证，不能当四份独立模型证据或重复计算备份/CPU成本。

best按同一条3＋5路线全部已完成5000 full候选的G4 PSNR、LPIPS等既定规则选择，官方U0和父D0只作外部基线；没有为monitor U3000增加新full候选。改U3000的G4 monitor100=18.615861是小监控最优，未full确认，不能替换U10000或当泛化成功。

{table}
完整五档PSNR如下；同一节点所有SSIM/MAE/LPIPS均保留于`unidb_dual_20261104_until00/report/FULL_QUALITY_COMPARISON.csv`，没有分别按各指标择优。

{five}
原U67180相对父D0 U45411的G4 PSNR改善{float(orig['delta_psnr_vs_parentFinal']):.9f}dB，但仍比Input低{float(inp['psnr'])-float(orig['psnr']):.9f}dB、比官方SR U0低{float(row('U0','external_fixed_baseline',4)['psnr'])-float(orig['psnr']):.9f}dB。原LPIPS {float(orig['lpips']):.9f}低于matched Input {float(inp['lpips']):.9f}，同时PSNR/SSIM/MAE较Input差；不能称四项全退化，也不能把较低LPIPS或更锐利视为正确几何恢复。改进两选中节点在强G4的PSNR/SSIM/MAE/LPIPS均差于Input，U40848的PSNR又低于其full-best U10000。两条综合配方未建立强湍流恢复优势或外部真实域泛化成功。

EMA full每G1000=250独立来源×4ξ，五G总5000；RAW仅G1/G4各monitor100，另列于`report/VALIDATION_HISTORY.csv`（筛选weights=RAW/monitor）和最终报告。其人口、RAW/EMA和指标不能混用。科学结论不由工程PASS、loss下降或完成更多更新代替。

### 完整性 CPU成本 退出与备份

`evidence/STATE_AUDIT_{{original,improved}}_{{FINAL,BEST}}.json`完成四角色RAW/Adam/EMA/8逻辑或5物理RNG/cursor/日程/source检查；实际checkpoint SHA→EMA全参数digest→对应full SUMMARY均逐项绑定。`evidence/FINAL_OUTPUTS_CPU_AUDIT.json`对三个unique完整5000候选的全部预测/Input/clean SHA及PSNR/SSIM/MAE/全部LPIPS复核PASS，共15000角色去重记录。两个新终点实际CPU metric前向2500 batches，U10000完整已审缓存复用1250 batches；缓存没有被重复计成新调用，新增恢复模型/optimizer/NPU调用均0。CPU finalizer于unix {load(ROOT/'evidence/CPU_FINALIZER_EXIT.json')['unix']}退出。

全部自有进程终验采用历史来源并集534身份：原生本次审计468，加上01:12预检中额外66个老loader/resource_tracker精确PID/start/PGID/SID身份；差集已逐项核相等，补查全部gone，无same/zombie/未解释存活或信号。37类来源覆盖、历史PGID/SID、rank、loader/resource_tracker、controller、CPU与本机8个指定传输PID均核退出；外部任务未动，card1 driver context不当OS存活断言。最终补查unix {exitq['checked_unix']}，不能把旧534预检中的活跃sender混入最终结论。

实际本机unique备份如下，均已根代理全文件字节SHA核过，当前size/mtime及最终角色重新绑定；本次记录刷新不重新读2.19GB载荷。旧U10000完整备份合法复用，不重复下载。

'''
 for x in backups['files']:tail+=f"- {x['branch']} {'/'.join(x['roles'])} U{x['update']}：`{x['local_path']}`，{x['bytes']} bytes，SHA `{x['sha256']}`。\n"
 tail+='''
全部unique字节合计6584061917。旧noon其他候选仍按原收据保留，既非额外最新权重，也不因已备份而获得删除许可。原报告/CPU收尾器里保留的VISUAL_LOCAL_PENDING是它们当时的历史状态，后续`delivery/experiment_full_delivery_receipt.json`、`all_owned_process_exit_audit.json`和`unique_checkpoint_backup_audit.json`合并关闭实际最终实验接收；不覆盖这些原件。

总图`report/VISUAL_QA.json`及原路线`delivery/original_route/VISUAL_QA.json`已root逐图绑定新SHA，分别三张总曲线/固定图与七张原路线全程曲线/G1–G4图。只引用已验字节，不重复绘图或运行恢复模型。19点PPT及21点状态图仍保留为历史快照，不能代替这些最终附件。

## 接手入口与科学局限

本地/远端以相同root名定位：`unidb_dual_20261103_until_noon`、`unidb_dual_20261103_until18`、`unidb_dual_20261104_until00`；远端host `liteserver-de36`、root前缀`/data2/yanglin/`。先读当前root `report/FINAL_REPORT_ZH.md`、`FULL_QUALITY_COMPARISON.csv`、`CPU_POSTPROCESS_WORK.csv`、四角色STATE_AUDIT、最终保存预测CPU审计以及上述三个实际交付adapter。数据、权重、状态与脚本入口由ASSET/WEIGHT/CODE_ENV/EVIDENCE注册表追溯；旧CURRENT_EXECUTION早期时间戳按历史快照理解，不复活旧PID、STOP或controller。

本阶段原生完整性、评价、退出与完整权重接收已经闭合；全项目主文档、可编辑DOCX、文档逐页QA、便携包和聊天交付另有流程，不能由实验终验自动宣布完成。`UNIDB_PENDING_FINAL.json`原件及其准备表收据保留为旧捕获，在当前实验表改归HISTORICAL_SNAPSHOT，不冒充最新状态。

科学局限仍包括模拟器和源域范围、训练一步目标与free-running状态差异、SR迁移/优化/强度混合及随机路径的未分离因果份额。当前独立全量Val不等于锁定Test或外部真实湍流部署成功。封存forensic的支持与反证不因后续综合配方失败而改写；没有证据把唯一根因定位为warmup、L1、Adam或backbone。环境多site-packages继承、未完全锁定wheel/OS/ISA/CANN，及诊断大数组主要留远端等复现边界继续保留。
'''
 history=P/'unidb_history.md';history.write_text(base+tail)
 old=load(ARCHIVE/'unidb_evidence.json');paths=[PROJECT/x['path'] for x in old['sources']]
 paths += [ROOT/'report'/x for x in ['FINAL_REPORT_ZH.md','FULL_QUALITY_COMPARISON.csv','VALIDATION_HISTORY.csv','CPU_AUDIT_VALIDATION_PROVENANCE.csv','HANDOFF_INTERVALS.csv','ACTIVITY_AND_COST.csv','CPU_POSTPROCESS_WORK.csv','REPORT_STATUS.json','FIGURE_MANIFEST.json','VISUAL_QA.json'] if (ROOT/'report'/x).is_file()]
 paths += [local(ref['summary']) for ref in cpu['refs']]
 paths += [Path(x['path']) for x in proof['files'] if Path(x['path']).suffix!='.png']
 paths += [upstream,ROOT/'delivery/EXPERIMENT_FINAL_RECEIPT_VALIDATION.json',P/'refresh_final_unidb_records.py']
 sources=[]
 for path in sorted(set(x.resolve() for x in paths)):
  assert path.suffix.lower() not in ['.pt','.pth','.npy','.npz','.png','.zip']
  sources.append({'path':str(path.relative_to(PROJECT)),'absolute_path':str(path),'bytes':path.stat().st_size,'sha256':sha(path),'local_exists':True,
   'check_scope':'Small local evidence bytes; checkpoint payloads only stat through actual accepted byte-backup receipt; no tensor read by refresh'})
 old.update({'status':'FINAL_EXPERIMENT_SOURCES_VERIFIED','prepared_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'document':'unidb_history.md','document_sha256':sha(history),'sources':sources,'current_experiment_final_acceptance':True,'gaps':[],
  'scientific_limits':['No strong-turbulence or external-real generalization success established','Different starts/objective/LR/clip/resources prevent single-cause attribution','Multi-site-packages and physical simulation/source-split limitations retained'],
  'final_experiment_receipt':request['final_experiment_receipt'],'final_related_receipts':request['final_related_receipts'],
  'actual_final_identities':proof['identities'],'overall_project_handoff_complete':False,'chat_delivery_performed':False,
  'generation_script':bound(__file__),'new_model_calls':0,'new_optimizer_updates':0,'checkpoint_payload_reads':0})
 (P/'unidb_evidence.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
 state={'schema':'unidb_final_handoff_content_state_v1','status':'FINAL_EXPERIMENT_DELIVERY_COMPLETE','campaign':ROOT.name,
  'handover_not_complete':False,'identities':proof['identities'],'evidence':bound(P/'unidb_evidence.json'),'history':bound(history),
  'upstream_registry':bound(upstream),'native_validation_receipt':bound(ROOT/'delivery/EXPERIMENT_FINAL_RECEIPT_VALIDATION.json'),
  'updated_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Actual final experimental evidence, owned exit and unique byte backups; overall project Word/package/chat delivery separate',
  'overall_project_handoff_complete':False,'chat_delivery_performed':False,'network_calls':0,'model_calls':0,'checkpoint_payload_reads':0}
 (P/'UNIDB_FINAL_HANDOFF_STATE.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
 result={'status':'PASS_FINAL_UNIDB_HISTORY_EVIDENCE_STATE_REFRESH','history':bound(history),'evidence':bound(P/'unidb_evidence.json'),'state':bound(P/'UNIDB_FINAL_HANDOFF_STATE.json'),'source_count':len(sources),'identities':proof['identities'],'network_calls':0,'model_calls':0,'checkpoint_payload_reads':0,'old_pending_snapshot_modified':False}
 (P/'UNIDB_FINAL_RECORD_REFRESH_RECEIPT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':run()
