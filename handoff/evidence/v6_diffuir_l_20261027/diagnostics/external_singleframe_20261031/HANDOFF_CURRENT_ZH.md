## 2026-11-01 最终交付完成（覆盖下文全部历史活动状态）

- 权威入口：本目录 FINAL_DELIVERY.json；两份最终可编辑 PPT 和760项文件已放入 output/V6_最终组会离线包，ZIP 为 output/V6_最终组会离线包.zip。已完成最终7页重导入逐页目视、全部原生表/图缓存及36真实图片源像素核验、760文件SHA、ZIP CRC和365离线HTML链接核验。
- G4一页审计完成：原5000步双方日志、实际强度比例、26checkpoint和五档逐图结果复核，旧两项FAIL原件保留，不修改训练数学，不续训。
- 外部六组固定范围完成：OTIS4328、Text400、ATD190、CLEAR64、RLR225、BVI2026十五帧，共5222输入记录；另3Door预定视觉。三EMA原尺寸S3共15675恢复/47025 U-Net调用，TurbWNet另完成全部4328OTIS。Text8000词OCR与各无GT共同支持NIQE完成。全原始源、版本、GT限制、尺度、采样与指标口径见最终包。
- Heat官方404与ATSyn配额/Box404如实跳过，未伪称完成Places重合核查或CW-SSIM。旧BVI门户超时证据保留，新作者2026版本单列，BVI/CLEAR内容重合不跨集汇总。
- 全部本轮10个有限NN run、master/RLR CPU/BVI CPU均已退出，0训练/0微调/A/B未恢复，禁止重启。实际 automation_update 删除本轮 v6，deleteStatus=deleted，见 evidence/AUTOMATION_DELETION_TOOL_RESULT.json。旧运行状态仅作历史，不可据其再起任务。
- 本地离线文件已提前就绪，远端镜像与最终读回见 evidence/FINAL_REMOTE_DELIVERY_READBACK.json。大型原始归档/权重/全量浮点预测保留服务器，离线包内有实际路径和SHA，不伪称已打包这些大文件。

## 2026-11-01 01:43最新覆盖：新BVI作者再发布来源；最终文件未打最终包

- **新发现有效BVI作者2026再发布 https://zenodo.org/records/21550759**，不要沿用“BVI全部受阻”。旧Bristol门户仍超时原证据不变；新版本必须另行登记，12个source版本。evidence/BVI_Zenodo_2026_record.json/README.txt已存；metadata CC BY4.0但README研究专用冲突，按更严格research-only用，勿宣称无限制权利。作者Anantrasirichai/Bull，真实Mirage50/Monument100/MovingCar163，共313原PNG无GT；受控gas-hob其余大包不下载不评测。
- downloads/BVI2026已取得三个小型ZIP共35.7MB，官方MD5/ZIPCRC/SHA全PASS（不是1.1GB全库）。evidence/BVI2026_DATA_IDENTITY.json完整313源PNG/15预定等距帧验证；data/BVI2026_FIVE_FIXED.jsonl/.gate已上传，原尺寸256²/512²/256×512，没有本轮resize。MovingCar与CLEAR2025共享物理内容，但5对选帧RGB像素均不相同，版本单列，不合并总PSNR/独立场景数。
- **唯一新BVI持久队列** build/pipeline_bvi2026.py CPU2598855/start575286056，BVI2026_PIPELINE_LAUNCH01/STATUS/log；BVI2026_SOURCE_MANIFEST五文件已冻结。invoke/d0_dataset函数从原master原字节抽取，原worker/三EMA/CONFIG/源manifest/数学不变。等 MASTER_NN_COMPLETE + RLR所有rank退出后自动BVI2026单卡smoke→4卡15帧formal。严禁第二次。再CPU summarize_bvi.py + audit_extra_final_v2.py，只BVI标签/manifest支持扩展，旧代码不改。
- RLR原master2438036已经全部45视频原生提225帧，当前进入单卡原尺寸smoke，后全量225；不要重复。原档4896959499B SHA16d67218dac97d20de51dd47a4b640dfb07523db869ca6eb26e2730a6d8f45b0。Text/CLEAR/RLR **三个完整原档均CRC/SHA PASS，Mac resume_v3已实际退出(session1325 exit0)**，CLEAR V3 JSON已PASS，不再是下载中。远端MASTER仍原活作业。
- 已接有限纯CPU后处理 build/finish_rlr_cpu.py CPU2582159/start575228869，RLR45_FINAL_CPU_QUEUE_LAUNCH01/STATUS/log及RLR45_CPU_COMPLETION_SOURCE_MANIFEST五源。它等待唯一RLR_NN_COMPLETE后只运行 summarize_extra --slug RLR45、audit_extra_final --slug RLR45、collect_run_closure(原8run+3archives)，0新NPU。不重复启动RLR CPU汇总。原8-run闭合只是这些8项，不替代新增BVI的2run闭合。
- **Task1最终一页进一步完善** output/weighting_review_final_v2/V6_G4_Weighting_Review.pptx，SHAfafda5764be9cb5ae3c6a6676fc454398e52b274666067d14687c6e78b8bd249；新增可见Mixed U95544→157505/+61961/best109544，LPIPS代价明确。1页原生表+bar最终重导入PNG已实际目视通过，output/已验证逐页预览/G4_final/第1页.png。旧stage/final_v1均留存。
- 外部build/build_external_final_v2.mjs是6页待运行版，目前只支持原五组，不含新BVI，**需新v3**：加BVI第6组/12source版本，page5无GT表加BVI15和共同支持；自然视觉可4组×(Input/best)132px方ROI，保持可读。page1 CLEAR2025/BVI2026双版本完成，旧门户仍失效；page6边界去掉BVI全部受阻。预算原5210+15=5225输入(含3Door)，3EMA15675恢复/47025U-Net；只算原生formal，smoke另计。不说独立场景合计，BVI/CLEAR物理场景有重合。
- build/package_final.py已写但**尚未运行**，需要补BVI：6组/5222benchmark行(另3Door)/15675恢复47025calls、BVI CSV/manifest/源/3小ZIP回执、视觉及新2run闭合；原8run审计仍保留，不把旧8误说全部10。其G4路径已改final_v2，预览标签G4_final；要求build/FINAL_VISUAL_REVIEW.json真实7页目视PASS后才pack。final_benchmark_register.py待新v2加BVI2026独立第12行，同时保留旧门户blocked行。
- OCR8000全PASS、CLEAR共同59/12表已完成，详下文。最终源表/摘要/图/讲稿必须用最新完成事实，不用旧5页阶段文字。最终目标仍05:30本地离线包/用户06:00断网，当前尚早；尽量全部CPU/worker结束、逐页QA/本地远端SHA同步、实际删除automation v6后最终交付。已交付旧阶段ZIP不覆盖。

## 2026-11-01 01:21最新覆盖：OCR及共同支持已完成

- TEXT_OCR_PER_WORD.jsonl全8000行已真实完成且独立核对官方500词/框/8000唯一键/编辑距离。TEXT_OCR_FINAL_AUDIT PASS；SUMMMARY/PER_FRAME/PER_SCENE/PAIRED_DELTA已生成，不重复OCR。WordAcc Input1.05% / best10.00% / Control2.40% / biased2.85%，CER96.437/79.212/91.760/91.524%。AppleVision补充协议，不称论文AWDR。
- Text_CLEAR_NIQE_COMMON_SUPPORT_SUMMARY已生成：59帧12序列共同finite，Input6.75124/best7.26563/Control6.37996/biased6.49833；原64帧13序列和5条Input undefined保留。新主表不使用原不等分母marginal。固定Text1/50/100帧50全图/词crop和OCR已在visuals/Text，CLEAR固定图也已拉本地。
- Text_CLEAR_INDEPENDENT_FINAL_AUDIT远端刚重新全部464输入/1392floatSHA/数据/源/三严格EMA/NFE/退出PASS，待同步新回执。RLR完整4896959499B已PASS CRC/SHA16d67218dac97d20de51dd47a4b640dfb07523db869ca6eb26e2730a6d8f45b0；master正45视频提225帧，不另起。CLEAR完整原档仍下载，保留实时V3状态。
- build/build_external_final.mjs已经准备6页最终稿，尚未执行，需RLR45_NIQE_COMMON_SUPPORT_SUMMARY和RLR45_INDEPENDENT_FINAL_AUDIT真实PASS后才运行。stage包已给用户且不覆盖。新audit_extra_final.py只读CPU核全量，已跑Text_CLEAR，RLR待正式退出。

## 最新覆盖：2026-11-01 01时（优先于下文旧状态）

- 本地已实际交付可离线阶段包 output/V6_已完成实测_离线阶段包.zip，17,681,707B SHA3f51da5c5564b518814940b1d851aa433724de860e847156f42ea7989197ebb2。包含最终G4一页+外部5页(仅完整OTIS/ATD/TurbWNet)及CSV/48图/原配置/回执/旧G4附录/离线图库。5页阶段PPT SHA0559687287bfa3bb6eb62987a2e064651a95304130eb9001b3eaf09a464c760e；最终G4在output/weighting_review_final，SHA902e2890bb55433661582ad674994ab34876bd86dc614bd64148fb108d49050a，新增五档均PSNR−.093011dB。所有最终页重导入实际目视；原生bar轴位置/图片抗锯齿重导入后变化，未伪称逐像素一致。已在commentary给用户离线包，仍继续扩充，05:30前最终离线交付/06:00断网。不得把此stage当7集全完。
- OTIS/ATD NN与13563浮点SHA/finite/NFE/所有worker退出独立PASS；NIQE760条全部有限。OTIS16序列宏：Input10.372096/.311215/.274517；best10.056591/.284050/.246401；Control9.701717/.242966/.240611；biased9.770948/.250429/.241546。TurbWNet全部4328也独立PASS，9.464645/.214299/.269797，CPU审计/视觉source SHA已在本地evidence、results。ATDNIQE Input6.72025 / best7.43962 /Control6.74882 /biased6.80478，无GT，不造PSNR。
- Text原完整zip v3已PASS CRC/SHA：1984598056B，SHA2c016ec6e773405c95638e8311d3e1937a5978d0d2ee329edba752ae3c365e7a，MD50c9fa7aa7cb8bf36b05c347a1808bdea；原档在server，不重下。RLR约3.9GB/4.897GB、CLEAR约1.1GB/2.208GB是旧进度，继续查*_DOWNLOAD_V3与download_v3_console.log。curl稳定恢复，勿再重启原requests/v2。
- 原prepare_selected v3因官方CLEAR ZIP中一个CRC-valid但0字节01233.png失败，原文件/日志保留。独立v4只将此无效原始成员从所有方法共同排除，无替补、不看输出选图；65预选CLEAR有64有效，Text400全部有效。evidence/SELECTED_INVALID_SOURCE_MEMBERS.json为准，完整source归档尚另在下载。
- watcher watch_upload_selected.py已PASS原PNG逐SHA/形状server重核并写SELECTED_PUBLIC_REMOTE_READY；MASTER已实际Text_CLEAR smoke/formal完成464输入×3EMA(400Text+64CLEAR)，PROCESS_COMPLETE exit0/222秒。Text_CLEAR_NN_COMPLETE PASS，不重复NN。MASTER目前等RLR原档完整acceptance。
- EXTRA_SUMMARY_Text_CLEAR_LAUNCH01 CPU2517801已/将完成：build/summarize_extra.py核每个floatSHA/NFE/配对/worker退出；本地data/OCR_EVAL_IMAGES.jsonl已取1600图源SHA；本地ocr_inputs已拉全部PNG，OCR_INPUT_ACCEPTANCE PASS1600源/400帧/100场景/500词框。
- 本地全量OCR现已启动：build/ocr_fixed build/OCR_FULL_INPUT.json results/TEXT_OCR_PER_WORD.jsonl，日志evidence/ocr_full.log；应8000word records(2000/方法)。AppleVision revision3 accurate en-US CPUOnly，语言校正false，严格固定官方框；ASCII lowerletters规范化；WordAcc/CER，非TurbNet CTPN+CRNN AWDR/AD-LCS。原sandbox smoke0行nilError失败保留；native获准smoke5词真实完成(第一场景重模糊均空，不能当CPU故障或填成功)。全量记录逐行flush。不要重复启动全量OCR，先看实际PID/cmdline和行数；若全量结束重算每方法400图/2000词/100场景、CER/LCS等实际，保存coverage与缺识别错误。
- CLEAR目前NIQE per-method margins不等支持：Input59/64有限，其余64/64。不得用不同分母直接公平排名。需基于results/Text_CLEAR_NIQE_PER_FRAME.csv算四方法共同finite交集(预计59)，保留原permethodCSV/coverage，主表common support、序列宏均值并明示原64帧/13序列与实际支持数。不要换NIQE参数或删原数据以掩盖NaN。ATD190×4全部finite无此问题；RLR完成后同样算共同支持。
- 下一步：TextOCR完成审计+固定第1/第50/第100场景中心帧图/OCR字框样例；CLEAR可用固定首/中/末序列中帧图由summarize_extra生成，需rsync visuals/新manifest并目视。当前已5页阶段版完整，扩充最终6页可结构：来源总览、OTIS量化、OTIS同帧、TextOCR、ATD/CLEAR/RLR无GT/NIQE共同支持与视觉、总结/限制。要保留阶段版，最终新output/final...路径里同名PPT。所有最终页重导入逐页目视、完整CSV/文字/源/PNG/native对象/输出SHA验收。本轮marker已用expected2，不再mark。
- RLR master源码包含45MP4按5等距decode、ffmpeg/ffprobe已在server可用；正确run_env+PIL。原档CRC/SHA后自动提225→smoke/full，实际资源4–7，若原尺寸OOM/数值问题只停受影响任务，保留现场，不缩图/切块/换模型。其余已完成不重跑。最后需启动CPU summarize_extra.py --slug RLR45(只一次/看launch防重)再作NIQE共同支持/图QA/本地同步。

# 当前执行交接（2026-11-01）

本轮唯一目标：既有G4加权一页审计PPT + 外部single-frame turbulence 4–6页实测PPT，附CSV/源/配置/图。不训练、不改方法、不A/B、不读V6 Test。用户06:00离线，05:30之前优先本地离线交付；不能把网络受阻写成已评测。自动跟进v6已实际创建，半小时；不是旧已删除v6-d0-160k。完成交付与任务状态闭合后删除。

先读本目录RUN_PLAN_ZH.md和当前最新人类消息。所有权威状态须核真实PID/start_ticks/完整cmdline或Mac lstart。远端ROOT=/data2/yanglin/v6_diffuir_l_20261027；本地为项目V6根。只4–7实时排他推理，不动0–3/外部任务。旧policy及checkpoint/冻结实验报告保持原样。

## 已真实完成

- G4两原始UPDATES各5000条/300000呈现、配对位置/原pool/τ/microseed及26CP SHA再审PASS，evidence/G4_RAW_LOG_REAUDIT.json。5000逐图重算三指标与delta，results/G4_*CSV。
- output/V6_G4_Weighting_Review.pptx：1页，原生表+delta柱图，SHA706b24fdb3dac34b0cd15f756d1d35b1e28312511944a95111a3f297ecb621ac；目视清晰、finalizer无警告，最终需重导入渲染。build/build_g4.mjs。图示49.9993% G4，Control24.9587%；G4 delta+.08770705dB /+.00534580SSIM/+ .00539515 LPIPS(坏)，弱档相对Control低、相对父仍略升。不要因冷/热Adam诊断改写原FAIL。
- OTIS全部4328Pattern、ATD38×5=190及预选Door3，共4521输入×3EMA=13563恢复、40689 U-Net调用。唯一distributed/V6_External_SingleFrame_OTIS_ATD_20261031_formal01/PROCESS_COMPLETE exit0；EXISTING_NN_COMPLETE PASS，所有rank到满数，绝不重启。权重为U109544/Control5k/biased5k，详CONFIG。原CPU launch01包路径错误0worker/0模型，v2调用python -m修，旧现场保留。
- 所有七入口/许可/GT/协议已查公开来源，证据evidence；Text官方labels_final.csv 500词框已Range/CRC取得。RLR45MP4、CLEAR distorted/restored目录由严格HTTP206 ZIP目录确认，只是metadata，不等于完整下载。

## 当前活作业，勿重复启动

- CPU_SUMMARY_LAUNCH01：原PID2449191，正确run_env+TORCH_DEVICE_BACKEND_AUTOLOAD=0，build/summarize_existing.py；核全部13563浮点SHA/NFE/shape/finite及worker退出，先写results/OTIS_SUMMARY.csv等，然后4CPU算190×4 NIQE。CPU_SUMMARY_STATUS/cpu_summary.log为准。NIQE用既有DiffUIR源码的96block+OpenCV bilinear缩半近似，明确诊断实现；不叫原MATLAB逐位。CPU失败只修CPU、不重跑模型。
- MASTER_QUEUE_LAUNCH01：原CPU2438036，正确run_env，build/master_queue.py，MASTER_QUEUE_STATUS/master_queue.log为准。当前已自动TurbWNet原尺寸OTIS smoke PASS并进入唯一V6_External_TurbWNet_OTIS_20261031_formal01全4328（4–7）。官方权重a039c671...复用原server，无重下；strict709 tensors/32,430,816参数、原3阶段，64倍reflect/crop，无resize。每图1完整网络/3stage，不能写DiffUIR3NFE。baseline失败仅该baseline记录失败。
- 此master后续先等SELECTED_PUBLIC_REMOTE_READY.json再Text/CLEAR固定成员S3 smoke/full；再等downloads/RLR_AT_Small.zip.acceptance.json真实完整CRC/SHA，提45原视频各5等距native帧=225，最后S3 smoke/full。新run不能抢现卡/开第二队列；无需另起Text或RLR。
- Mac完整下载：build/resume_public_v3.py，日志evidence/download_v3_console.log和*_DOWNLOAD_V3.json，原requests/v2 TLS EOF/超时均已退出，partial保留。v3 curl HTTP1.1证书不关闭、严格206/Content-Range，前缀SHA+三片段对源核对，仅补缺失字节。Text1.984GB/RLR4.897GB/CLEAR2.208GB；历史prefix约1.05GB/1.1GB/2MB，未来实时为准。完成必须CRC/SHA/MD5+服务器*.zip.acceptance.json。原v1/v2失败不覆盖。若本机断网relay不能继续，须在截止前保存真实断点，不能称完整。
- Mac选帧成员准备：build/prepare_selected_public_v3.py，evidence/selected_public_preparation_v3.log。v1/v2已退出且日志保留；保留114已CRC PNG，curl v3只补缺少。固定Text全部100场景中心49–52=400（原生440图），CLEAR每distorted场景5等距，原字节/CRC/SHA/形状；全归档进度另列，不把此称1万帧全评。完成data/SELECTED_TEXT_CLEAR.jsonl/.gate、evidence/SELECTED_PUBLIC_PREPARATION.json PASS后：先rsync data/Text和data/CLEAR PNG，再清单和gate，服务器独立逐SHA/形状核验，最后写SELECTED_PUBLIC_REMOTE_READY.json让唯一master推进。
- OCR build/ocr_fixed.swift已编译build/ocr_fixed。固定AppleVision revision3/accurate/en-US/无语言校正/CPUOnly，官方5词框仅评分、不得恢复输入。原论文CTPN+CRNN源码协议未复现，所以如实际跑只标补充WordAcc/CER/非原论文AWDR。需要生成manifest含items(sample_id,method,image_path,scene)和labels(scene→5个text/box)，把Text同样本四方法PNG拉本地后实际运行。标准化ASCII小写字母，CER真实Levenshtein，不能虚构OCR结果。

## 剩余交付

1. benchmark来源CSV/简洁中文清单：7核心+3候选，原论文/官方入口/许可/实际下载/结构/GT/原测试与本次协议分开。Heat404；ATSyn quota+Box404、Places源泄漏未核不能PASS；BVI旧URL超时；CLEAR restored是多帧算法伪参考，不真GT；Bristol38=已存在ATD；RLR无GT。TurbNet无可验证weights、PiRN官方Box404，不随机替代。TMT/DATUM/TMFS为多帧文献背景，不能混公平表；候选SOTIS15.8GB官方GT但无确认固定test，别下载无关大包。
2. 先完成当前完整OTIS/ATD的source/SHA/均值/全部worker/13563记录独立验收与图片：固定OTIS Pattern1和Pattern3/4首或旧预选中帧，Input/D0best/Control/biased/GT及同位置ROI；ATD固定预定Bin/Cabin/Road帧，同4方法、无GT、NIQE诊断。不能按效果挑图。raw图server，已有Input本地。先给完整实测版，其他按实际状态列清；05:30离线交付比追求7集齐全优先。
3. output/V6_External_Generalization_Evaluation.pptx 4–6页，中文16:9、原生可编辑表图、真实图溯源。不要把GT全无的当前范围叫带clean合成GT；OTIS是数字图案ref，mask亮度PSNR-M/SSIM-M11/LPIPS-valid64不同于RGB Val。
4. 05:30前本地输出两PPT、CSV、源表、配置/命令/状态、实际visuals、摘要ZIP和SHA。保留preliminary版本快照，不用远端旧草稿覆盖已补正文。后续能及时完成Text/CLEAR/RLR就扩充，不能为其无限等待；用户6点断网后只远端已启动作业可以继续，本地relay/新交互会受阻。
5. 本轮已用Presentations技能且artifact create marker已成功expected-output-count=2，不重复mark。bundled @oai/artifact-tool；Hiragino Sans GB；private build/final output分离。所有最终页逐页渲染目视、重导入检查、图表/workbook/数据追溯。未使用python-pptx，0训练/改方法。最终readiness审查后交付；v6自动跟进实际删除再记录。
