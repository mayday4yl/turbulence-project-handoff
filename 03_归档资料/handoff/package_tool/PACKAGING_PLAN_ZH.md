# 小文件便携交接包：准备方案与严格准入接口

本工具只整理现存本地小文件，不联网、不执行shell/旧launcher、不导入torch/NumPy、不运行恢复或指标模型、不读取checkpoint张量。当前只完成准备；`--plan`和`--dry-run`仅写本目录，不创建`delivery`，不发布当前draft。

## 当前可执行命令

项目根下：

```bash
python3 project_handoff_20261104/preparation/package_preparation/build_handoff_package.py --plan
python3 project_handoff_20261104/preparation/package_preparation/build_handoff_package.py --dry-run
python3 project_handoff_20261104/preparation/package_preparation/test_package_preparation.py
```

`--plan`生成`PACKAGING_PLAN.json`，列出选择的来源/bytes/SHA/包内路径、明确省略和缺少的最终门槛，不创建资料成品。`--dry-run`复核完全相同的门槛；任一未齐返回exit2并生成`PACKAGING_DRY_RUN.json`。exit0只表示以后可按当前证据打包，不是用户已收到资料或实验科学成功。

最终入口是同脚本`--final`，**本轮不得用当前draft作为其输入**。工具会先查四个逻辑槽和原生实验收据，失败exit2；成功后先在本目录临时stage复制/改写链接/全SHA回读/ZIP CRC，再创建`project_handoff_20261104/delivery/portable_handoff/`与`portable_handoff.zip`。已有成品一律拒绝覆盖，不启动传输。当前准备态已经实际验证该最终入口拒绝且没有创建delivery。

## 选择与稳定包内结构

输入是`EVIDENCE_REGISTER.csv`、各history/evidence JSON、各注册表、`UNIDB_PENDING_FINAL.json`、已出现的当前实验小收据和显式最终文档QA附件。旧NOON权重只引用，不复制。

| 包内位置 | 内容 |
|---|---|
| `README.md` | 主入口、最终文档/注册表链接、明确不含payload |
| `documents/` | 真正FINAL主稿、DOCX、PDF；拒绝DRAFT文件名与DRAFT QA |
| `document_pages/` | root已逐页检查并绑定的所有最终页PNG |
| `registers/` | 七类总注册表，原绝对路径继续用于溯源 |
| `history/`、`provenance/` | 按版本的叙述与证据JSON |
| `evidence/<原项目相对路径>` | 注册表选定小原件、实际最终state/SUMMARY/验收/固定图等 |
| `original_text/<原项目相对路径>` | MD的未改写原始字节，保留其原SHA及历史链接作证据 |
| `SOURCE_MAP.csv` | 原绝对/项目相对路径、原bytes/SHA、包内路径、包内bytes/SHA与MD原字节副本位置 |
| `NOT_INCLUDED.md` | 未包含大资产和原引用的位置/原因；不是文件已下载或远端现存证明 |
| `MANIFEST.json` | 除manifest自身外所有包内文件的bytes/SHA与链接检查计数 |

MD可读副本的本地链接在**新包副本**中改写；原始MD另存原字节。未选入附件/大权重/原始预测等引用改成`NOT_INCLUDED.md`中的明确原位置条目，不伪造文件或显示空白图；原报告不修改。HTTP/HTTPS/mailto保留且本轮不联网验证。CSV/JSON里的原始绝对路径不改，不能据此把包当作可直接恢复的完整环境。

包只允许常见文本/CSV/JSON/YAML和保存为证据的py/sh；二进制附件仅允许显式最终DOCX/PDF/PNG/JPEG。`.pt/.pth/.ckpt/.safetensors/.bin/.npy/.npz`、视频、ZIP/tar载荷及checkpoints/predictions/arrays等批量目录不复制。单文件32MiB、全包128MiB、含MD原文副本总文件数2000。超限资产明确列省略；若超限者属于必需最终证据则最终拒绝，不静默丢弃。

路径只接受项目根内的规范文件；拒绝`..`、控制字符、每一层symlink、私有凭据/运行目录及包内绝对/冲突路径。远端路径只把已知`/data2/yanglin/...`翻译成本机项目相对位置，不建立连接。最终小JSON若仍只留远端，必须由主收尾按原SHA同步其实际小文件镜像；工具不会代跑远端采集。canonical历史symlink应指向实际已同步的小原件路径并显式绑定，不为打包改远端链接或复制权重。

## 最终四个逻辑槽：显式adapter，不按任意键名猜PASS

`PACKAGE_FINAL_INPUTS.json`当前是准备模板，所有占位都故意不存在。root最终把**真实原生验收**转接为下面四种小adapter，再把其项目内path及完整SHA填入`receipts`。adapter是原证据索引，不是把未完成项手改PASS；每个槽均要求`schema`和`scope`精确相同、`status="PASS"`、`final=true`，以及非空`evidence_bindings:[{"path":"项目内真实JSON","sha256":"64hex"}]`。所有绑定逐字节核SHA。

1. `experiment_full_delivery_receipt`：schema/scope=`unidb_midnight_full_delivery_v1`。必须含`campaign="unidb_dual_20261104_until00"`、`deadline_unix=1793721600`、`identities`四项`original:best/original:latest/improved:best/improved:latest`，每项`update`及checkpoint`sha256`要精确等于真实CPU收尾refs。应绑定主实验最终交付、实际最终验收、小文件本地同步等原收据。
2. `all_owned_process_exit_audit`：schema/scope=`all_campaign_owned_process_exit_v1`。必须含`all_owned_exited=true`、`live_owned_actors=[]`、`external_signals_sent=0`、`checked_unix`不早于实际CPU finalizer退出，`campaign_roots`精确含noon/18/00三root名。原证据应覆盖PGID/SID、rank、loader/resource_tracker/controller/CPU/helper，而不只NPU rank；不是在本工具内查询或停止进程。
3. `unique_checkpoint_backup_audit`：schema/scope=`actual_final_unique_checkpoint_backup_v1`。`files`每项含`local_path,branch,roles,update,bytes,local_mtime_ns,sha256,source_sha256,status,SHA_verified_unix,final_role_binding_verified_unix`；item status精确为`PASS_BYTE_IDENTICAL_FINAL`。只stat本机已验过的payload路径，不重读大权重；size/mtime须保持、两SHA相等，最终角色绑定时间在截止之后，所有四角色精确匹配实际最终身份、不得重复。best=latest可用一个文件的两个roles；旧完整备份若成为真实最终best可复用原已核SHA，只需新的最终角色绑定与未变size/mtime，不能重复下载。
4. `handoff_final_document_visual_qa`：schema/scope=`handoff_final_document_visual_qa_v1`。必须含`reviewer="root"`、`draft=false`、`page_count`、`pages_reviewed=[1..page_count]`；`artifacts`逐项`path,bytes,sha256`，覆盖真正FINAL DOCX、PDF、每页PNG；每PNG还必须有唯一整数`page_index=1..page_count`，所有artifact path必须唯一，重复page1不能凑足页数。`source_markdown`同样以path/bytes/SHA绑定最终MD，不能保留本次“准备稿/当前终评未完”等固定段落。所有这些文件名不得带DRAFT。现有`DRAFT_VISUAL_QA.json/PASS_DRAFT_LAYOUT_ONLY`不接受，已有draft布局QA不能代替最终数据/文字更新后的全页QA。

`final_documents`和`extra_small_assets`可显式增补其他最终小附件，item为`path,destination,sha256`。主要文档/页PNG从最终QA自动收集；不能只给一个PASS而不附文件。

## 当前叙述与注册表刷新：不把旧pending快照当current

`PACKAGE_FINAL_INPUTS.final_handoff_refresh={path,sha256}`另绑定一个**收尾内容刷新索引**，不是新增模型验收或代替上游四类收据。其`schema=scope="handoff_final_content_refresh_v1"`、`status="PASS"`、`final=true`、`identities`必须匹配原生CPU的四个真实最终角色。

该索引的`current_unidb_history/current_unidb_evidence/current_handoff_state/consolidation_acceptance`每项均为`{path,bytes,sha256}`；`registers`为七注册表文件名到同样binding的字典。路径可指向新最终文件或已实际刷新的当前文件，不强制修改旧准备快照。

- 当前UniDB evidence必须`current_experiment_final_acceptance=true,gaps=[]`，且`document_sha256`等于当前history的SHA；history不能仍含本次当前终评未完成的固定准备态段落。
- 当前状态小JSON必须`campaign="unidb_dual_20261104_until00"`、`status="COMPLETE"`或`"FINAL_EXPERIMENT_DELIVERY_COMPLETE"`、`handover_not_complete=false`。这是实验已验收内容状态，不声称资料已在聊天发给用户。
- 整合验收必须`current_UniDB_final_acceptance=true,errors=[]`，其outputs的SHA与指定CSV匹配。EXPERIMENT_REGISTER须至少一条当前UniDB行（`version_id=version:UniDB,status_scope`以`CURRENT_`开头），全部此类current行的`current_final_acceptance=True`且scope/status无PENDING；旧快照如保留必须标历史。不能只把bool手改true去绕过consolidator的受验final分支。

工具从刷新绑定选择当前history/evidence/七注册表与整合验收，放入稳定包内位置；旧`UNIDB_PENDING_FINAL.json`以及被替代的准备原件只进入`historical_preparation/`，README明确它们不是current。原生门全齐但当前叙述/注册表尚未刷新，仍exit2。consolidator先消费真实上游实验最终收据生成这些文件，再由刷新索引绑定输出；不让consolidator消费自身输出，避免循环。

凭据筛查只保守匹配明显token前缀或secret/password赋值（包括短quoted password），发现时仅报告path/line/count、不输出值，并拒绝打包该来源；这不是穷尽性保密证明。证据绑定只读小`.json`，先排除payload/批量目录后才做SHA，不因其内容长得像JSON就读取`.pt/.npy`。工具生成的`original_text/`与README/manifest等命名空间保留；所有来源/目标路径控制字符和symlink父链均拒绝，final输出父链也重新核验。

## 同时独立检查的原生实验门槛

即使四adapter齐全，工具仍直接读取当前root已有真实schema并拒绝以下缺项：

- `report/REPORT_STATUS.json`必须`FINAL_READY/final_requested/final_requirements_pass/visual_qa_pass`，pending空；两线`TRAINING_AND_MODEL_VALIDATION_COMPLETE`、controller及CPU finalizer完成/退出需真实绑定。
- `CPU_POSTPROCESS_ACCEPTANCE.refs`精确四角色；各`STATE_AUDIT`的auditor源码SHA、151个RAW/EMA张量/137147523参数、FP32 finite、Adam step、EMA step、8/5 RNG bank/cursor正确；其checkpoint SHA→EMA digest→对应full SUMMARY一致。
- 每实际best/latest SUMMARY必须EMA原生SDE100步、五档各1000/250source、总5000、同branch/update，无clip/crop/resize且Test未用。`FINAL_OUTPUTS_CPU_AUDIT`须pure CPU、0新恢复调用、每候选全5000的SHA与全部LPIPS覆盖并绑定同一SUMMARY。
- `FIGURE_MANIFEST`与root `VISUAL_QA`绑定三张新实验PNG。
- `delivery/original_route/ORIGINAL_ATTACHMENT_RECEIPT.json`必须绑定实际最终original best/latest；其七张PNG（全程train、monitor/full四指标、G1–G4固定图）必须与该目录新`VISUAL_QA.json`的`status=PASS,figures={文件名:SHA}`一致。该source receipt本身保留`ARTIFACTS_GENERATED_ROOT_VISUAL_QA_PENDING`历史状态，由后续独立QA完成，不改其原件。

实际4角色state/SUMMARY、四adapter及原证据、全部最终主文档/页图和上述10张关键图缺一不能生成成品。工具只重新核小文件与已生成图，不重跑模型、CPU LPIPS或大checkpoint SHA。

## 准备完成的验收边界

本目录的合成测试检查越界、symlink、凭据目录、包内路径、payload/超限排除、链接重写、缺文件拒错、DRAFT QA及旧候选拒绝。真实`--dry-run`和缺证时`--final`须exit2，`project_handoff_20261104/delivery`不因准备而创建。没有执行通过全部门槛的最终包构建或外部发布。

本工具是后续root收尾入口；当前源和准备验收SHA记录在`PACKAGING_PREPARATION_ACCEPTANCE.json`。若最终实际原生schema有新增字段或合法变体，先让独立review核实并更新**本新工具**，不得改冻结训练/后处理源码或放宽原实验容差来迎合打包。
