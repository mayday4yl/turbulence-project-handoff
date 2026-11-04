# 无版本前缀的真实数据、外部对照与方法审查

这是2026-11-04的交接工作稿，依据2026-10-29至11-02已有本地报告整理；没有重新获取数据、运行模型或联网更新文献。各路径和SHA见`extra_evidence.json`。以下“D0”在真实图和TurbWNet比较中指 **V6 DiffUIR-L D0**，不是11月2日的UniDB D0。

## 1. 外部真实数据已经取得到哪一步

|项目|实际已完成|可以如何使用|尚缺什么|
|---|---|---|---|
|DMAT官方Real入口所指ATD|38段原始MP4及说明下载、大小/ZIP CRC/逐文件SHA验收，采样表合计18075帧；首帧均解码|自然场景真实湍流的定性素材；保留原始视频与固定选帧记录|没有逐帧clean GT；未逐帧解码全视频；38视频也非38独立场景。CLEAR1/CLEAR2是算法结果，不能当GT|
|OTIS PNG Color|283564104字节官方原包与解压完整验收，5794张PNG|16静态Pattern序列共4328帧可作图案恢复候选；Door和Car可作无GT定性素材|GT为数字图案经缩放/人工配准，非同相机无湍流clean；此获取审查在10月29日时尚未冻结评价协议，后续V6已完成冻结评价，见下文承接|
|Heat-Chamber|保存官方入口及扩展检索证据；2026-10-29官方入口404|候选外部实验室真实数据方向|未验收原始包，实际场景/退化/GT数量未知；扩展取得的演示包不能替代官方完整数据|

ATD实存范围是原始视频约1.341GB；完整ZIP标称约7.835GB，本轮没有完整保存或核其整包MD5。另一个DMAT synthetic数据和权重入口只保存元信息，不能登记为已下载。已有首帧PNG是解码衍生物，不是原包提供的原始PNG序列。

OTIS共有21序列、5195退化帧：Pattern16序列4328帧、Door300帧、Car四序列567帧；另外16张恢复GT和583张Mask。Car的GT目录是检测/跟踪目标框Mask，不是恢复参考。Pattern主要是两类打印图案，不能当4328张独立自然内容。独立官方Mask不等于GT alpha阈值；相关视频帧也不能用于独立同分布置信区间。

优先入口：
- `dmat_audit_20261029/reports/DMAT_REAL_ATD_DOWNLOAD_ZH.txt`、`ORIGINAL_VIDEO_MANIFEST.csv`、`evidence/ORIGINAL_VIDEO_RECEIPTS.json`。
- `otis_audit_20261029/reports/OTIS_AUDIT_ZH.txt`、`SAVED_DATA_ACCEPTANCE.json`、`FILE_INVENTORY.csv`、`PAIRED_STATIC_CANDIDATES.csv`。
- `heat_chamber_audit_20261029/reports/ACQUISITION_AND_SUITABILITY_ZH.md`、`EXTENDED_PUBLIC_SEARCH_ZH.md`。

这些都是带日期的获取状态，不能从历史404断言今天永远不可下载；本轮交接不为刷新结论重复下载大数据。

后续执行已经发生：V6 U95544在`v6_diffuir_l_20261027/external_eval_20261029_otis_atd/OTIS_ATD_EVALUATION_REPORT_ZH.md`冻结灰度、有效Mask、SSIM与LPIPSvalid64协议并完成OTIS Pattern4328帧定量；随后`v6_diffuir_l_20261027/diagnostics/external_singleframe_20261031/HANDOFF_CURRENT_ZH.md`记录三候选及TurbWNet的OTIS4328完整评价。获取审查的“尚未冻结”只描述10月29日当时的子任务，不能覆盖这些后来完成的实验。对应结果与限制见`v6_history.md`。

## 2. 固定20张真实图：已经完成的视觉对照

`real20_d0_turbwnet_20261029/`预先固定OTIS10图（指定Pattern首帧）和ATD10图（指定视频第0帧），对V6旧D0 U14927 EMA与官方完整TurbWNet各输出20图。本机独立CPU FP32、batch1，无训练；D0原S3/NFE3/eta0，TurbWNet完整deblur/detilt/final三阶段。两模型同一RGB输入，没有GT输入网络。

这是中心区域最多256×256的低成本预览，大图不作为整图恢复，小图保留原生尺寸后统一reflect补边。旧D0明显过平滑、一些文字和细杆丢失；TurbWNet保留更多结构，但形变和局部伪影仍在。此结论只对应20个固定样例，未形成OTIS或ATD完整集排名，也没有合法的ATD full-reference指标。

资产入口：`inputs/FIXED20.json`、`evidence/MODEL_ACCEPTANCE.json`、`evidence/D0_CANONICAL_REPLAY.json`、`outputs/RESULT.json`、`reports/RESULTS_ZH.txt`和`reports/COMPARISON_GALLERY.html`。D0 U14927 SHA `0c7903726f262c56fb40333dcaf70e0e205109c618456bb39e216987bd48d9b6`；TurbWNet官方完整权重SHA `a039c67176a0704a9db22cfefe6f58e78d09681e50c03ce73bfad34290b0843f`。历史CPU环境为该目录`env/`与`requirements-cpu.txt`，不覆盖服务器环境。

## 3. 真实图丢字/抹杆诊断：预处理差异是真实限制

`d0_real20_diagnosis_20261029/`完成80次有效模型调用的受控定位：包含历史数值基准、同一组故障样例的反事实和全20张精确训练预处理复跑。80是调用数，不是80独立内容。

核对未见RGB/BGR、范围、EMA、S3输出索引或保存错误；但原预览采用原像素中心ROI/小图补边，训练源流程为短边缩放256再中心crop，空间尺度确有不同。复用训练原函数后，小OTIS图明显变化；ATD文字/杆件仍被抹掉。训练空间变换也不自动是外部真实测试的唯一正确协议。

固定4张合成G4正向检查Input19.186→CPU D021.574dB，但恢复仍平滑；不能把所有细节问题归因纯真实域差异。把S3初噪声std从正式0.1改到1.0使这4图PSNR下降1.151dB，未修复真实图，不能将增噪当作已验证修复。4图8bit量化平均降低约.132dB，说明输入统计敏感，却不等于归一化实现出错。首步clean估计已削弱文字/杆件，支持细节丢失发生在模型预测与clean重建内部，但未隔离唯一loss或容量原因。

已保存历史CPU轨迹的同初态数值检查，与跨CPU/NPU仅同名义seed的检查分开；后者随机数流不同，不声称逐像素精确等价。最初读错旧NPZ的fixture比较已归档纠正，当前报告不采用错误差值。

入口：`reports/DIAGNOSIS_ZH.txt`、`DIAGNOSIS_RESULT.json`、`ALL80_CONTROLS.csv`、`evidence/FINAL_ACCEPTANCE.json`、固定图与原始NPY。诊断已完成，交接不重跑模型、不覆盖原20图。

## 4. 固定合成图上的D0与TurbWNet：小集合与全量分列

`latest_d0_turbwnet_20261029/`沿用原G4固定10图，将第三列更新为当时已完成Val1000的V6 D0 U95544 EMA输出，0新推理。相同10图PSNR：Input17.903898、旧U14927 20.177678、TurbWNet18.177932、新U95544 20.185760；新旧D0平均仅+.008082dB。D0全Val1000 21.165142是另一人口，不能和TurbWNet固定10图拼成全量对照。

`g05_g4_fixed10_turbwnet_d0_20261030/`固定D0 U153544 EMA及每档相同10图，五档50个TurbWNet CPU前向；D0复用已保存正式输出，四指标同CPU实现重算。该样例表为：

|G（每档n=10）|Input PSNR|TurbWNet PSNR|D0 U153544 PSNR|
|---:|---:|---:|---:|
|0.5|23.894973|28.574888|28.523672|
|1|21.172043|28.162771|26.102622|
|2|19.378935|26.379081|23.529562|
|3|18.578579|21.321070|21.573448|
|4|17.903898|18.177934|20.122902|

四指标完整表见`reports/SUMMARY_BY_G.csv`和`RESULTS_ZH.txt`；PSNR、SSIM与LPIPS不总是同向。数据与模型调用在不同设备上，不能比较速度。这些图提供样例层面的强弱档差异，不是五档各1000的TurbWNet全量验证。

## 5. 方法重合与warmup：保留为审查，不伪造实验证明

`at_ddpm_overlap_20261029/FULL_COMPARISON_ZH.md`是针对当时完整V6的定向重合核查。它认为“条件扩散用于单帧湍流”已有先例，不能以换数据域或骨干声称首次；V6的learned raw-ξ物理递归/自生成观测/full-BPTT是机制差异，但收益仍待实验。这不是全网新颖性放行，也不是当前UniDB带有raw-ξ的证据。全文与逐步检索证据保留，接手者应按最终实际方法重新审查claim。

`unidb_warmup_review_20261102/REVIEW_ZH.md`是独立未应用补丁和CPU调度自检。所查官方SR参考配置`warmup_iter=-1`，不能称本次漏掉官方已开warmup。U45411后加warmup也不能追溯检验U0阶段无warmup的原因。该包当时没有端到端NPU/Adam/EMA恢复或质量验证。后来双路线改进同时改起点、loss、LR、clip并用了warmup，是新的综合配方；不能把其结果当原独立warmup补丁的受控验证。

## 交接边界

这些支线不改变主训练授权、冻结Test或历史结果。接手后可先阅读保存的图/CSV/验收，不必先加载模型。已有V6外部完整评价沿用其冻结选帧、GT/Mask、原生尺寸与指标协议。只有尚未执行的新模型、新人口或新协议需要另行明确；本次不补做未授权评价，也不向数据作者发送请求。
