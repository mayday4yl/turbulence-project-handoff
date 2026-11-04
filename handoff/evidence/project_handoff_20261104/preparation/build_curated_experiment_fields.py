"""Local text-only field curation; preserve exact excerpts and source SHA.
No training, image, checkpoint, network or inference IO. This is a handoff
preparation component, not historical experiment acceptance.
"""
import collections, datetime, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent.parent
OUT=[]
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def cite(file, anchor):
 p=Path(file); p=p if p.is_absolute() else HERE/p
 text=p.read_text()
 blocks=text.split('\n\n')
 candidates=[s for s in blocks if anchor in s]
 assert len(candidates)==1,(file,anchor,len(candidates))
 return {'path':str(p.resolve()),'sha256':digest(p),'exact_excerpt':candidates[0]}
def add(group,bid,field,value,*citations):
 assert citations
 OUT.append({'experiment_id':group+':'+bid,'source_group':group,'branch_local_id':bid,
 'field':field,'value':value,'citations':list(citations),
 'basis':'EXPLICIT_LOCAL_TEXT_OR_STRUCTURED_FACT; no execution status override'})
def pair(g,b,method,pop,manchors,panchors=None):
 name={'v5':'v5_history.md','v6':'v6_history.md','unidb':'unidb_history.md','extra':'extra_branches.md'}[g]
 mc=[cite(name,a) for a in manchors]
 pc=[cite(name,a) for a in (panchors or manchors)]
 add(g,b,'method_description',method,*mc)
 add(g,b,'dataset_or_population_json',{'description':pop,'scope':'Exact branch-specific described workload; not a new asset or whole-project dataset claim'},*pc)
# V5: formal training, no-update diagnostics, engineering and archival scopes remain separate.
pair('v5','simulator','V5三通道当前状态Z+连续物理时间t；原生raw Zernike/tilt及P2S分布，Smooth/Tilt-Warp/FFT模拟；研究数值验收，非正式训练。','数值和梯度验收fixtures（含g=1e-4近零边界）；接收任务无正式数据生成/训练，未声明独立source数。',['V5沿用预测clean/raw-ξ后','Simulator保留原生raw','接收任务本身无正式数据生成/训练'])
pair('v5','backbone_review','Concertormer full/Lite、BioIR-L、AdaIR的CPU/static结构及参数量审查；没有GPU、checkpoint下载或optimizer执行。','静态模型源码与CPU结构审查，不是图像恢复数据集评价；模型运行0、下载checkpoint0。',['是CPU/static审查'])
pair('v5','a100_preparation','A100镜像与benchmark静态/CPU准备；默认launch只打印，原报告未执行GPU benchmark/正式训练。','32样本预备数据；不等于完整新数据，未宣称真实GPU训练消费。',['截至原报告仅静态/CPU准备'])
pair('v5','psf_input_gradient','固定PSF输入梯度等价普通卷积优化，保留FP32前向/反射边界；CPU64/NPU/完整四步/多卡工程门。','原四卡B32工程工作负载，D0/R0更新耗时对照；报告未声明该门独立source总数，不是正式六卡质量人口。',['保持前向、反射边界和FP32数学'])
pair('v5','concertormer_D0_A_B','Concertormer：D0纯RGB Clean MAE一次恢复10k；A物理中间态连续时间单步6k；B真实四步物理Full-BPTT 4k，A/B目标0.9Clean+0.1回投。','V5 train2000/val250/test250 clean，各4ξ、G4单帧；Train8000对、Val1000；Test1000锁定未用。',['**D0**：纯RGB一次直接恢复','**A**：从D0','**B**：从A best'],['数据为新V5 clean train2000'])
pair('v5','concertormer_stepwise_replay','训练后三阶段PSNR-best（D0U8250/AU5000/BU2750）固定样例逐步重放，0更新；不是训练时逐update快照。','4个固定Train+4个Val样例，共8例；非全Val结论，Test未用。',['训练后补做的'])
pair('v5','resshift_stage1','官方journal ResShift Deblur/VQ f4四步latent diffusion零样本候选，FP16 autocast/HF32 off；未正式训练。','官方720p演示28tile/112UNet调用；固定湍流Val4图；没有完整Val1000或GoPro复现。',['固定journal commit','官方演示720p'])
pair('v5','resshift_grad_vq','ResShift FP32四步恢复反传与固定真实ξ的两阶段物理反传；detach负对照；0 optimizer；未验收预测ξ完整四物理阶段。','梯度fixture独立source数未声明；官方VQ自重建Val250 clean与50弱态；这些codec指标不是恢复效果。',['零optimizer','严格官方VQ路径Val250'])
pair('v5','resshift_codec_repeat','官方冻结VQ编码/解码连续四轮往返，0 UNet/Simulator/训练。','250 clean×4轮=1000 codec调用；同图重复，不是1000独立内容。',['对250clean连续四次codec往返'])
pair('v5','resshift_archive','ResShift三个原目录原地双SHA及证据快照归档，非新增完整备份，未删除源资产。','资产操作人口：3目录7321文件、4.56GiB；不是训练/验证数据人口。',['对三目录7321文件/4.56GiB原地双重SHA核验'])
pair('v5','diffuir_L_stage1','官方DiffUIR-L Large64-1248 pixel-space六通道条件、随机τ原生raw residual L1，S3=999→665→332/eta0/initnoise0.1；官方RAW fresh EMA起的100更新pilot，无物理外层。','400pair冻结G4池中实际200realization/91source/49类；EMA Val50，另固定Val4技术检查；无锁定Test1000/大规模训练。',['这是pixel-space六通道','随后正式短pilot100更新'],['NPU2 FP32固定Val4','随后正式短pilot100更新','U0/25/50/100完整checkpoint读回'])
pair('v5','diffuir_benchmark','DiffUIR-L原生FP32一步训练与完整S3采样反传工程benchmark；12成功case，144测试更新全丢弃，6环境失败0更新。','Train8000冻结池实际986呈现/240pair/229source/50类；这是工程工作负载，不是全部池训练。',['12次成功有界运行','Train8000冻结池中实际测试覆盖'])
# V6: population is kept tied to the branch, not inferred from every shared asset.
base='初始G4 Train10000clean×4ξ=40000pairs'
pair('v6','method_review','V6方法规范/代码映射静态审查：DiffUIR-L六通道condition+τ/SDT，D0内部S3；A/B预测raw-ξ物理外层。审查本身无训练。','方法规范及源码审查；无实际训练人口消费，不把规范中的训练数据等同此审查已运行。',['保留用户完整规范、代码对应和未决'])
pair('v6','initial_joint_D0','DiffUIR-L D0联合目标0.5×原生raw残差L1+三步末端Clean MAE；4 denoiser训练、S3推理，FP32/B60。','G4 10000 clean×4ξ=40000训练pairs；固定Val250source×4=1000；U10000/600000呈现。',['初始D0联合目标 |','六卡NPU2–7，FP32'],[base,'六卡NPU2–7，FP32'])
pair('v6','joint_D0_continuation','继承初始D0联合目标完整状态的续训：U10000→13334→14927暂停；不是官方U0起的新独立对照。','沿初始G4 Train10000clean×4ξ=40000/Val1000；+3334更新200040呈现，再+1593更新95580呈现。',['续训U13334比U10000' if False else 'U13334比U10000仅','四卡限时段从13334到14927'],[base,'U13334比U10000仅','四卡限时段从13334到14927'])
pair('v6','A','V6物理恢复A训练阶段，模型选择用固定4→3→2→1→0真实四步Val1000末级PSNR；6000更新，六卡转五卡改变有效batch。本条不补造完成报告未展开的训练目标公式。','原Val1000=250source×4；A实际259280呈现（前2410步B48、后B40），不能用6000×48反推。',['确认A实际6000更新'],[base,'确认A实际6000更新'])
OUT[-2]['citations'].append(cite(PROJECT/'v6_diffuir_l_20261027/reports/A_COMPLETION_REPORT_ZH.md','固定Val1000、4→3→2→1→0四步最终PSNR最佳'))
pair('v6','B','从A best EMA初始化的V6四步物理Full-BPTT B；工程pilot丢弃，实际U9500暂停，终点未找到完整Val1000。','沿G4 40000pairs/Val1000定义；B U9500/cursor95000；bestU7000、最近fullU9000，U9500不冒充full。',['V6 A/B为预测raw-ξ驱动的物理外层','B从A best EMA新初始化','用户要求暂停B、继续D0'],[base,'用户要求暂停B、继续D0','停止点U9500未找到完整Val1000'])
OUT[-2]['citations'].append(cite(PROJECT/'v6_diffuir_l_20261027/reports/B_PREFLIGHT_PILOT_AND_STARTUP_REPORT_ZH.md','五卡micro1×acc2，有效batch10'))
pair('v6','official_residual_D0','联合目标U14927完整RAW/Adam/EMA/RNG起续训，移除Clean分支后只用原生随机τ raw residual L1系数1；FP32/B60/LR2e-6，无新warmup；1NFE训练/S3推理。','G4 Train40000pairs/Val1000；本段+80617有效更新4837020呈现，累计5732640；非官方U0纯原生对照。',['此段继承联合目标训练过的RAW','唯一目标修订为原生raw残差L1系数1'],[base,'唯一目标修订为原生raw残差L1系数1'])
pair('v6','mixed_strength160k','从U95544完整状态续训；只将G4专训改G1–4等比例，FP32/B60/constant2e-6/nativeL1/1NFE训练/S3推理不改。','10000clean×4ξ×4G=160000组合；五档G0.5/1/2/3/4各Val1000=250source×4ξ；G0.5不训练。',[ '从U95544接续至2026-10-30 18:00'],[base,'从U95544接续至2026-10-30 18:00','各档Val1000=250source×4ξ'])
pair('v6','Control_G4biased','同完整父U157505的Control均匀25%/G与G4-biased（G1/2/3各1/6、G4一半）两支各5k；loss/LR2e-6/EMA/seed保持，native残差L1。','160K组合共享池；各300000呈现，五档各Val1000；Control与biased实际分档呈现另列，不称均匀遍历全池。',['两支从同一U157505完整父状态'],[base,'原报告`execution_20261030_d0_g4_reweight','实际Control呈现G1'])
pair('v6','turbnet_candidate','TurbNet官方源码/权重可获取性静态审查，官方权重当时缺失后终止；0环境/模型/smoke/Val/optimizer。','NOT_RUN：没有实际恢复数据集评价；未运行随机初始化替代。',['**TurbNet 2026-10-28。**'])
pair('v6','turbwnet_initial','官方完整TurbWNet权重CPU strict load+一次真实forward；NPU资源guard后停止，无本分支NPU fullVal或训练。','CPU一次技术forward；报告此段未声明独立source数；NPU完整smoke/Val1000未做；后续OTIS4328属独立分支。',['**TurbWNet原分支。**'])
pair('v6','OTIS_ATD_U95544','V6 U95544 EMA原生S3单帧外部0训练评价；灰度luma+有效mask/full-native无resize/tile，LPIPSvalid64。','OTIS5195恢复：Pattern16序列4328有maskGT，Door300/Car567定性；ATD38视频×5固定帧=190无GT。',['，零训练：OTIS共5195恢复','这是灰度luma+mask'])
pair('v6','official13_forgetting','官方EMA vs V6综合best U109544原生S3五任务13子集遗忘诊断；按原任务Y/ECC/RGB LPIPS协议，0训练。','13完整子集，43321配对输入、86642恢复、259926UNet调用；不能任意平均不同PSNR定义。',['13/13完整子集','量化Y/RealBlur ECC'])
pair('v6','external6groups','V6 best109544/Control5k/biased5k三EMA单帧S3外部对照；另TurbWNet OTIS全量；0训练/物理A-B，OCR/伪GT/NIQE限定口径。','六组OTIS4328/Text400/ATD190/CLEAR64/RLR225/BVI15=5222输入另Door3；三D0共15675恢复，TurbWNet另4328。',['六固定组OTIS4328/Text400'])
pair('v6','ATSyn999_best_final','ATSyn-Static固定单帧turb24/gt，Input vs best109544/final157505 EMA原生512/S3；非论文20帧流程复现。','官方1000test场景，去掉1个与Places近重复后999；1998模型恢复+999 Input记录；非额外训练人口。',['官方1000test场景固定','不是DATUM论文默认20帧'])
pair('v6','diffusion_dependency','U13334与官方EMA的固定状态/时间干预和C1→C3采样依赖诊断，0更新；不证明后验分布能力。','50独立source×4seed固定子集；不是完整Val1000或独立训练恢复器对照。',['**D0_diffusion_dependency_20261027_1955**'])
pair('v6','diffusion_mechanism','官方/U10000/U13334三CPU副本，NFE1/2/3/4/6、合法q状态、敏感性和真实分支梯度审计；0optimizer/NPU。','12source固定CPU诊断子集，3模型；不冒称完整NPU验证重现。',['**diffusion_mechanism_audit_20261028**'])
pair('v6','train_val_generalization','U14927固定Train/Val匹配协议泛化诊断，非全量Train10000；结果不能证明无过拟合或必然欠训练。','固定Train300source×4=1200；已保存Val250source×4=1000。',['**d0_u14927_train_val_generalization_20261028**'])
pair('v6','U95544_strength','U95544 G4-only EMA原生单帧恢复器在G1–G4四档full评价，0训练；只提供混强度动机。','G1/G2/G3/G4各1000，合计4000恢复；G4逐图与原验证对齐。',['**d0_u95544_g1_g4_val1000_20261029**'])
pair('v6','residual_train_val_loss','U95544 EMA固定τ/noise原生残差L1 Val vs最近在线RAW train；S3末端Clean MAE独立列，非同目标历史曲线。','完整Val1000及最近1000在线RAW train日志；RAW在线与EMA固定采样并非完全等价人口。',['**d0_train_val_loss_20261029**'])
pair('v6','strength_preview_inventory','同source/原ξ直接物理render六强度与数据库存审查，0恢复模型/训练，不是对前一退化连续再退化。','32source×6G（0.25/0.5/1/2/3/4）=192退化+32clean；非完整质量人口。',['**strength_preview_g025_g4_20261029**'])
pair('v6','post_meeting_audit','V6源码/loss/sampler/评价/数据历史只读审计；受审范围未确认新实现bug，不作P2S物理真值证明。','既有160K组合与source-disjoint检查、5000观测cache及250clean重SHA、150000旧逐图及额外5000结果复核。',['**post_meeting_audit_20261101**'])
pair('v6','candidate_research','像素空间候选/去模糊/PixelDiT/latent约束方向的源码与权重入口元数据审查；0候选模型/权重载荷/推理/训练。','论文、源码、README、HEAD/LFS/权重入口元数据，非湍流恢复图像测试；优先级不是质量排名。',['v2按用户取消','上述研究均零候选模型实例'])
# UniDB shared protocol is quoted per applicable branch; do not alter pending status.
umethod='官方源为`UniDB-SOC/UniDB@'
udata='- 仍是Places365'
uval='- 正式完整验证为'
uprot='- 主评价为EMA'
pair('unidb','section-ab02f5f9df99','官方DIV2K SR gamma_1e7初始化，UniDB六通道condition+扩散t，native MatchingLoss L1全参数微调；FP32/B96/Adam1e-5，100步SDE评价，无prompt/Adapter/rawξ/物理递归。','Places50类10000train clean×4ξ×G1–4=160000组合；五档each Val1000；G1/G4 each monitor100；完整D0止U45411。',[umethod,'官方网络/桥/MatchingLoss L1保持'],[udata,uval,'- 监控为G1/G4各100图'])
pair('unidb','section-387924b72318','封存forensic代码/数学/优化器/只读梯度/teacher-forced与SDE/mean-ODE诊断；optimizer0，不重训。','多协议子集：正式6节点×5G×1000记录；50Val G4来源×4=200固定采样干预；4seed；具体诊断8来源/50来源另列，不等于全Train。',['这套已封存审计有前向和只读反传','**有干预支持但有范围限制：**'],['**有干预支持但有范围限制：**', '- 监控为G1/G4各100图'])
# Add original forensic table as exact source for the 6-node count.
OUT[-1]['citations'].append(cite(PROJECT/'unidb_d0_20261102/forensic_audit_20261102/FINAL_FORENSIC_AUDIT.md','|正式完整验证|'))
for bid in ['section-e71430a96f82','current_midnight_boundary']:
 pair('unidb',bid,'3+5综合双路线：原从D0 U45411完整状态沿native MatchingLoss L1/1e-5；改从官方SR U0用epsilon_opt MSE、200步warmup2e-7→2e-6/墙钟cosine至2e-7/clip1；18和00段保留floor2e-7及祖先状态，无新warmup。','共用160K组合：10000clean×4ξ×G1–4；五档各Val1000=250source×4，G1/G4各monitor100；午夜原67180/改40848安全停训；人口字段不声明终验完成。',['原路线从D0 U45411完整状态继续','|until_noon|','两线B96/FP32','午夜最终更新的8个rank'],[udata,uval,'- 监控为G1/G4各100图','午夜最终更新的8个rank'])
# External branches: acquisition, illustrations, diagnostics and literature are different populations.
pair('extra','section-b0bbc41eebff','DMAT所链ATD/OTIS官方包获取完整性及GT-mask适用性审查，Heat公开入口获取核查；非模型训练/排行榜。','ATD38原视频、采样表18075帧，无cleanGT；OTIS21序列5195退化帧+16GT+583Mask；Heat原包未验收、实际人口未知。',['|DMAT官方Real入口所指ATD|','|OTIS PNG Color|','|Heat-Chamber|','OTIS共有21序列'])
pair('extra','section-8030cf70bd52','V6 U14927 EMA原生S3/eta0 vs官方完整TurbWNet三阶段，本地CPU FP32同输入单帧固定视觉预览，0训练。','预固定OTIS10指定Pattern首帧+ATD10指定视频frame0；中心≤256ROI/小图原生reflect补边；非全量OTIS/ATD或高分辨率整图恢复。',['预先固定OTIS10图','这是中心区域最多256'])
pair('extra','section-ce047829061b','V6 U14927细节抹除受控诊断：历史复现、训练原预处理、RGB/范围/EMA/S3保存、noise敏感性等，0训练。','同固定20真实图中指定故障样例及全20训练预处理复跑；80有效模型调用不是80独立图；另4合成G4正向对照。',['完成80次有效模型调用的受控定位','复用训练原函数后','固定4张合成G4正向检查'])
pair('extra','section-0271212c8e6e','匹配小集合的V6 D0旧/新EMA与官方TurbWNet对照；U95544/G4及U153544五档D0复用保存输出，TurbWNet另50 CPU前向，非新训练。','原G4固定10图；独立五档G0.5/1/2/3/4各相同10=50样例；不是TurbWNet五档each Val1000。',['沿用原G4固定10图','固定D0 U153544 EMA及每档相同10图'])
pair('extra','section-1c796dffde36','AT-DDPM/V6方法重合定向文献与代码审查；另UniDB未应用warmup补丁/CPU scheduler自检，无端到端训练或质量验证。','论文/源码/规范与调度测试，不是共同训练或图像恢复评价人口；后续双路线综合配方不是独立warmup因果实验。',['是针对当时完整V6的定向重合核查','是独立未应用补丁和CPU调度自检'])
# Dates only where a source explicitly provides a date, rather than inferred directory names.
add('unidb','section-ab02f5f9df99','historical_date','2026-11-02（正式D0开始日期；非后续双路线完成日）',cite('unidb_history.md','正式于2026-11-02开始'))
for bid, path, anchor, val in [
 ('section-b0bbc41eebff','dmat_audit_20261029/reports/DMAT_REAL_ATD_DOWNLOAD_ZH.txt','DMAT 官方真实数据（ATD）下载验收','2026-10-29（获取验收；后续外部评价另列）'),
 ('section-8030cf70bd52','real20_d0_turbwnet_20261029/reports/RESULTS_ZH.txt','完成日期：2026-10-29','2026-10-29（Asia/Shanghai）'),
 ('section-ce047829061b','d0_real20_diagnosis_20261029/reports/DIAGNOSIS_ZH.txt','日期：2026-10-29','2026-10-29（Asia/Shanghai）'),
 ('section-0271212c8e6e','g05_g4_fixed10_turbwnet_d0_20261030/reports/RESULTS_ZH.txt','D0固定为2026-10-30 17:04','2026-10-30 17:04（五档对照固定checkpoint的检查时点；非统一任务完成时点）')]:
 add('extra',bid,'historical_date',val,cite(PROJECT/path,anchor))
for bid in ['section-e71430a96f82','current_midnight_boundary']:
 add('unidb',bid,'historical_date','2026-11-04T00:00:00+08:00（第三段训练截止；非终评或交接的统一完成时点）',cite(PROJECT/'unidb_dual_20261104_until00/protocol/USER_AUTHORITY.json','"deadline_local": "2026-11-04T00:00:00+08:00"'),cite('unidb_history.md','午夜最终更新的8个rank'))

def main():
 cpath=HERE/'curated_v4_experiment_fields.json'
 child=json.loads(cpath.read_text()) if cpath.exists() else {}
 citems=child.get('overrides',child.get('entries',[])) if isinstance(child,dict) else child
 if not citems: raise SystemExit('V4 component not ready; not writing combined final file')
 combined=OUT+citems
 seen=set()
 for x in combined:
  ident=(x['experiment_id'],x['field']); assert ident not in seen,ident; seen.add(ident)
  for c in x['citations']:
   p=Path(c['path']); assert digest(p)==c['sha256']; assert c['exact_excerpt'] and c['exact_excerpt'] in p.read_text()
 q={'schema_version':1,'status':'CURATED_EXPLICIT_FIELDS_ONLY_NOT_EXPERIMENT_ACCEPTANCE',
    'prepared_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'scope':'Small local UTF8 text/JSON only; explicit branch-bound method/population/date. Missing historical dates remain UNKNOWN; no status or final acceptance overrides.',
    'generation_script':{'path':str(Path(__file__).resolve()),'sha256':digest(Path(__file__))},
    'components':[{'path':str(cpath),'sha256':digest(cpath),'overrides':len(citems)}],
    'overrides':combined,'counts':dict(collections.Counter(x['field'] for x in combined)),
    'network_calls':0,'model_calls':0,'optimizer_updates':0,'checkpoint_payload_reads':0}
 (HERE/'curated_experiment_fields.json').write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n')
 print(q['counts'])
if __name__=='__main__': main()
