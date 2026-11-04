# 共享资产、环境与只读接手操作说明

整理日期：2026-11-04。这里交接的是**如何找到与辨认资产**，不是新的训练、推理、恢复或清理指令。历史运行命令只用于理解来源；午夜 UniDB 的终评与最终权重备份由本次实验收尾单独确认，不以旧 noon 候选备份替代。

配套文件：`ASSET_REGISTER.csv` 为逐资产台账；`asset_evidence.json` 保存本次本地stat、来源SHA、历史验收口径和缺口；`REMOTE_PATH_CHECK_PLAN.json` 是原本地资产任务制定的最小远端stat白名单；后续root的独立执行结果见下述补证，不改写原任务的0远端调用口径。原始资产和旧报告没有改动。

`build_asset_register.py`保留台账生成方式：只用stdlib读取明确本地证据、stat已声明路径并写本preparation输出；不联网、不导入torch、不读取大checkpoint内容。它是手动维护入口，本次交接无需重跑；将来重跑会更新台账时间/本地stat，需重新审阅新增差异和QA，不可冒充本次冻结快照。

## 1. 新接手者先走这条路径

1. 从项目根 `EXPERIMENT_INDEX.md` 确认版本。早期v1/v1.5/v2/v2.5/v3与后来的V4/V5/V6/UniDB不能只看目录名字；旧名字、数据ID和checkpoint路径保留。
2. 读对应最终报告、科学配置、source/数据manifest及完整checkpoint验收。README里的旧“未启动”“正在运行”和PID都是历史快照，不能代替最终记录或当前进程。
3. 按台账找到代码、数据、预训练权重、已训练checkpoint与环境。`local_payload_state=PRESENT`仅说明这个路径本次存在；目录存在不表示其中大数组完整，本地资料包常只含manifest/小图/报告。
4. 先核验小文件/元数据和白名单路径stat。大checkpoint不靠文件名判断best/latest，数据不靠seed或`DATA_READY`单个文件判断完整。
5. 若只是接收研究成果，到这里即可。历史launcher、dataset generator、训练/评估脚本、精修solver、STOP、旧cleanup均无需运行；即使旧脚本写有“resume”，也不是当前授权。

早期完整历史见 `legacy_v1_v3.md`；本说明不重复各实验结果表。

## 2. 平台与 canonical root

| 平台 | 历史明确主机标识 | 代码/数据根与注意事项 |
|---|---|---|
| 本机Mac | 本地 | `/Users/lucas/Desktop/turbulence IR`：配置、证据、报告、部分权重/备份；不是所有训练数据的完整镜像 |
| 早期NPU | `liteserver-40b3` | v1源 `/data0/yanglin/physical_bridge`，run/data `/data2/yanglin/physical_bridge`；v1.5新协议在其子目录，不能凭同名推断de36也有相同内容 |
| 后续NPU | `liteserver-de36` | 基础源/环境仍有 `/data0/yanglin/...`；大多数新实验 `/data2/yanglin/<独立实验>`，canonical datasets另在 `/data2/yanglin/datasets/`；V5/V6数据有独立根 |
| A100 CUDA | `h3c-gpu-server01`（文档也记 `10.94.2.1`） | `/home/zhuzhiyu/yanglin/turbulence_ir`；shared/assets、shared/data/generated、shared/envs与versions分离。共享账号/UID不等于本项目拥有其他人的文件或进程 |

SSH别名由接手者使用现有获授权连接确认。本交接不读取或打包SSH配置、私钥、token、登录状态或镜像临时凭据。文档中的主机标识不是连接成功证明。

## 3. 模拟器、P2S、basis与raw ξ：必须一起绑定

原PiRN/P2S模拟器有三份关键资产：

| 文件 | 作用 | 历史固定SHA256 |
|---|---|---|
| `P2S_model.pt` | 固定前向物理映射网络，**不是图像恢复预训练权重** | `578c3cd09c5c44e66c1086ffc0aadcab99542b74d27b932e9ca601a3faf6a4d5` |
| `dictionary.pt` | PSF均值/字典basis | `2204ace3e4b99f31d62d21d0ba19bae95c606393ba7df5111e2526172e6b24c6` |
| `I0_I2.pt` | 固定物理统计资产 | `aa99db54950df99465e5c8b7b7b16b7ed9236603b5c1a63d82dd8831b5712f2e` |

三份文件本地在 `v5_v1_simulator_20261020/assets/`现存，本次对小文件字节SHA重新核对；其他版本仍须按自身asset lock查路径，不为统一命名移动文件。

**同三份资产不等于同一个renderer。** v1/V4 renderer的tilt使用原`irfft2`约定；V5研究接收路径默认legacy full-complex IFFT，另有Smooth/pixel-center语义和固定buffer。V5原Gate未全PASS与其研究接收决定分别保留。不能把V4观测改挂到V5模型上，也不能为复现而偷偷修改renderer“修正”历史约定。

跨平台重放还有两层已知陷阱：

- 原ARM/scalar与x86 AVX的CPU `torch.randn`即使seed相同，也曾不逐字节相等。Train15000保留了scalar RNG兼容方案及20项raw精确SHA核验；新环境必须按原generation identity核验，不能只抄seed。
- 用同源码/同资产在另一主机重新构建固定simulator buffer也曾产生差异。`CANONICAL_CPU_REFERENCE.json/.npz`与兼容REVISION固定原buffer；8项RGB重放在原容差内通过。不要重建这份参考后仍称同一basis，亦不能把其生成专用兼容补丁随意套入所有训练代码。

V1/V3和V4部分canonical保留全raw数组；V5/V6为节省数百GiB至TiB，主要保存每样本/每kind独立CPU种子、固定shape、torch/RNG版本与raw字节SHA，不落盘全部raw。原形状为Zernike `[1,36,512,512]`、tilt `[1,512,512,2]`。seed-replay元数据也是科学资产；它与噪声的effective field/PSF系数不是同一种坐标表示，不能直接替换网络head或监督。

## 4. 数据版本与source split不能互换

| 数据人口 | 固定规模/用途 | 入口与锁定边界 |
|---|---|---|
| v1 `places_p2s_bridge_v01` | train2000/val200/test500 sources，4/2/2 realizations，六档state | 早期40b3数据/归档报告；terminal test1000与6000-state teacher-forced另算 |
| v2.5 source-half view | train1000×4、val100×2、test250×2 | 只是引用原tar的view；test只完整性核验；U204不是训练完成 |
| v3 Overfit100 | 100train sources×4ξ×4G=1600starts | `v03_overfit100...`为v3数据；后补heldout32来自未见train-pool source，已反复用于诊断，不是未来独立测试 |
| A100旧train500 | ID `places_p2s_a100_20260928_g4321_s500_v100_t200_v1` | 不能与Train2000/Train8000归一化混用；部分旧raw约97GiB是明确保留资产 |
| V4 Train2000 | `places_p2s_canonical_20261003_c50_s2000_v250_t200_g4321_v1`；2000/250/200 source | de36唯一canonical，A100字节镜像；train-only normalization，test锁定；A100旧xi副本曾清理，de36原xi历史全量SHA通过 |
| V4 Train8000 | `places_lsdir_p2s_canonical_20261006_p4000_l4000_v1000_t200_g4321_v1` | train Places4000+LSDIR4000，val各500，test Places200；LSDIR500来自官方train池heldout，不称官方val500；统计只用128000train states，runtime不含raw/test |
| V4 Train15000扩展 | 旧Train8000＋新增7000source | 新增original/clean/states已从A100归档到de36；历史指针不改。恢复原run前必须先按归档清单恢复缺失文件，旧DATA_READY不能自行授权续训 |
| V5 G4 | `v5_places_c50_s2000_v250_t250_g4_r4_20261021`；2000/250/250×4 | 复用clean身份但重生全部V5观测，不复用旧V4 Y4；default test loader锁定 |
| V6 Train10000 G4 | 10000train sources/40000pairs，固定val250×4与test250×4 | 保留V5旧2000source/8000pairs，新增8000source/32000pairs；official train archive与旧val来源域不同，不能认为同类别就同分布 |
| V6/UniDB G1—G4 | 160000train pairs、10000clean；val每G1000、250source | 同source/ξ渲染四G；UniDB DATA_GATE绑定Train/Val两个manifest；G0.5只作额外验证，不当训练已见强度 |

`SOURCE_READY`只表示来源，`DATA_READY`是某时点数据冻结验收，不是恢复模型完成标志。读取所需manifest的SHA、source split、generation identity和actual file audit必须同时匹配。V6旧DATA_READY某些诊断字段曾记录raw replay false，而G1报告引用后续独立重放；不得仅挑`status=PASS`就宣称所有逐项重放为真，应读取对应原始/后续验收，仍不能唯一对应的条目标未核。

验证与Test锁定是**实验协议**。不要因为目录能读就打开未授权Test恢复评价；已经被用于模型选择/诊断的heldout也不能在新调参后继续称从未看过的test。多强度数据、V5/V6与UniDB共用clean/val来源，跨实验比较需说明这种复用。

## 5. 环境是按实验绑定的，不存在通用run_env

| 分支 | 历史解释器/入口 | 已记录版本与局限 |
|---|---|---|
| de36早期v2.5/v3 | `/data0/yanglin/envs/physical_bridge/bin/python`；各自`scripts/run_env.sh`或`code/run_env.sh` | Python3.10、torch2.7.1+cpu＋torch_npu2.7.1.post2、CANN8.5.2；`.cpu`标签不表示不能跑NPU，实际由扩展接设备 |
| A100 V4 | `/home/zhuzhiyu/yanglin/turbulence_ir/shared/envs/py310_torch271_cu126_v1/bin/python` | Python3.10.20、torch2.7.1+cu126/CUDA12.6、torchvision 0.22.1+cu126；metrics deps/cache另有link；不等于de36二进制环境 |
| V5 Concertormer | 本分支`run_env.sh`＋base physical_bridge环境 | 需要V5 simulator PYTHONPATH及私有kernel cache；queue0/等价backend属于分支证据，勿抄其他launcher覆盖 |
| ResShift→DiffUIR Stage1 | 各新root `env/bin/python`＋各`scripts/run_env.sh` | `.pth`复用base torch/NPU；DiffUIR还复用ResShift包目录，不是完全封闭venv。上游CUDA/xformers环境未逐项复刻 |
| V6及UniDB D0 | V6 `env/bin/python`；V6/UniDB各自run_env | 实际sys.path还经过DiffUIR Stage1/ResShift/base；UniDB ema-pytorch 0.3.2的warmup语义会影响验收，不能只看beta |

**依赖没有完全锁定。** 有commit、科学源码SHA、环境快照和部分pip freeze/安装版本，但本次没有发现一个覆盖全部实验、wheel二进制、`.pth`继承链、OS/ISA、CANN/driver和系统库的可离线重建容器锁。不要用`pip install -U`“修环境”，也不要在基础环境卸载共享包；先记录解释器、模块实际路径、`.pth`链和对应snapshot，按分支另建隔离环境并重新完成所需验收，不能承诺一条requirements就逐bit复现。

run_env常会`mkdir`缓存、改变设备可见性/线程/JIT、source CANN并`exec`传入脚本。**不要把source/run_env当只读检查**，也不要把旧脚本注释的卡号当今天预留资源。主进程、CPU checkpoint读取乃至torch import都有NPU自动加载风险；接手初查只用系统python3的stdlib stat/JSON，不导入torch、不反序列化checkpoint。

## 6. 官方权重、训练权重和指标网络分别找

- P2S三件套是模拟器资产，不代表Restormer已有恢复预训练。V4 Restormer的common U0是随机共同初始状态；NAFNet则使用官方SIDD width64主体，V4 RGB/ξ适配头另初始化，完整官方文件保留。
- Concertormer官方GoPro、DiffUIR `model-300.pt`、UniDB SR `gamma_1e7.pth`是不同模型/任务/权重格式。DiffUIR `model-130`的UDC分支不能混用；UniDB本轮明确选gamma_1e7，其他SR候选不因同属SR就可替换。
- LPIPS AlexNet `alexnet-owt-7be5be79.pth`是指标网络权重，cache路径与SHA也必须保留，离线计算LPIPS依赖它；“恢复模型0调用”不等于“所有神经网络0调用”。
- TurbWNet完整`net_g_300000.pth`已经含全部恢复阶段，不能再用单独deblur/detilt覆盖；TurbNet官方预训练本轮未取得，应登记缺失而非随机补参。
- 已训练checkpoint须看完整状态验收与对应SUMMARY/parameter digest，RAW与EMA、best full与monitor best、final/latest分别标记。文件size相同或checkpoint名带final都不足以说明相同参数。

台账对大权重只做本地stat/大小复核、复用历史SHA，不为整理重复读取数十GiB。少量P2S/basis小资产另标本次字节SHA实核。

## 7. 归档与备份：哪些仍可取回

| 历史操作 | 本次核实的接手含义 |
|---|---|
| A100 20261002：93中间权重先本机归档再删源 | 本地93/93存在、size匹配，约43.26GB；它们现在是被删源的保留副本，不是可随意清空的cache |
| 20261012授权/13完成：旧Train2000 A100 rawξ清理 | de36原17800份xi历史全SHA通过；另86+7+221旧checkpoint未找到归档目的地，不能承诺全部可恢复；final/U0/官方预训练等另保留 |
| 20261020：33文件转de36后删A100 | archive `/data2/yanglin/storage_archives/A100_cleanup_20261020_33files`；本地只保留receipt，不是33份payload本机备份 |
| 20261025：126050文件转de36后删A100 | 其中126000是Train15000新增7000source的original/clean/states，其余9临时数据+41旧评价权重；文件保留原项目相对路径，恢复前另核清单/不得覆盖内容不同文件 |
| NAFNet Train15000 10/18停止点 | 本地四份完整权重存在、size与receipt一致：D0/R0 U40000、R08 U32891、L08 U21559；它们是当时latest，不等于任何后续实验最终best |
| UniDB旧noon候选四份 | 本地传输完成、size匹配；是原U55411/U55760、改U10000/U20000，不可用来声称午夜实际best/latest都已备份。旧18指定local delivery路径本次未发现 |
| ResShift原地归档 | 7321文件完整清单与历史回读、33小证据副本；大资产仍在原目录。属于原地保全，不是额外异地完整备份 |

历史cleanup/restore脚本**不再运行**。只有匹配archive acceptance＋cleanup receipt才说明当时“先验后删”完成；已删且无归档的文件应直接标不可保证恢复。磁盘free、用户声明2TB限额与共享文件UID不是当前可用配额或项目归属证明。本交接不删除、不重排、不新建硬链或迁移数据。

## 8. 可直接审阅的只读命令模板

以下是**模板，尚未执行**。先将`<SSH_ALIAS>`替换为已获授权的现有别名；只使用白名单中明确路径。不要遍历`/`、`/data*`或其他用户目录，不启动模型、驱动查询、数据生成、checkpoint反序列化、评估或任何STOP操作。

本机检查小证据或权重路径存在与大小，目录只stat不递归：

```bash
python3 - <<'PY'
from pathlib import Path
p = Path('/Users/lucas/Desktop/turbulence IR/unidb_d0_20261102/weights/sr_gamma_1e7.pth')
print({'path': str(p), 'exists': p.exists(), 'is_file': p.is_file(),
       'bytes': p.stat().st_size if p.is_file() else None})
PY
```

远端只读stat单一已知路径（不读取payload，不import torch）：

```bash
ssh '<SSH_ALIAS>' python3 - <<'PY'
from pathlib import Path
import json
p = Path('/data2/yanglin/unidb_d0_20261102/weights/sr_gamma_1e7.pth')
out = {'path': str(p), 'exists': p.exists(), 'symlink': p.is_symlink()}
if p.exists():
    s = p.stat()
    out.update(bytes=s.st_size, mtime_ns=s.st_mtime_ns,
               is_file=p.is_file(), is_dir=p.is_dir())
print(json.dumps(out))
PY
```

这只能核存在/大小，不能升级为SHA或完整性通过。只在明确小文件（如manifest、环境JSON、receipt）上另算SHA；对大weights、rawξ和全部预测的字节核验应由已授权最终验收或后续明确范围承担。

查看launcher内容应使用 `sed -n '1,160p' '/明确分支/run_env.sh'`，不source、不执行。当前进程审查若必要，应核PID＋`/proc/<PID>/stat`的start_ticks＋cmdline归属，不能只按旧PID发信号；本资产任务未运行远端进程检查。

## 9. 本次仍未核实、须在总交接注明

原本地资产整理任务没有远端调用。其后root已按白名单仅做stat：`REMOTE_PATH_CHECK_RESULT.json` 记录de36的22项及40b3的1项在该次检查时存在；这是路径/类型/大小/时间元数据，不是内容SHA、整个目录完整性或环境可运行验收。A100的5项经历史地址 `zhuzhiyu@10.94.2.1` 连接超时，`REMOTE_PATH_CHECK_RESULT_A100.json` 明确为 `NOT_VERIFIED_CONNECTION_FAILED`；它们仍未核，连接失败不等于资产不存在。各补证时间及原始source SHA由统一注册表单独引用；磁盘空间、账号配额和全部远端内容完整性仍未知。当前午夜UniDB终评与最终best/latest备份仍等主任务完成记录补齐；旧候选不冒充它们。部分旧文件已按历史保留策略删除且无恢复目的地，环境依赖链不完全封闭，V6原始raw replay字段与后续独立验收须按具体ID对应。局部数值/CPU/GPU PASS不能外推为整个环境或所有数据已验证。

本次没有读取凭据、访问远端、调用模型/指标网络、执行optimizer、训练/推理/STOP或清理；只在这份新交接的preparation目录写入台账与说明。


## 10. 最终补充：午夜实验已验收的三份完整权重

本补充生成于 `2026-11-03T18:12:49.997292+00:00`。前文及原77条资产的stat、旧noon候选与旧18路径缺失是准备期历史观察，保留原口径；本节补充后来通过的实际终评、完整状态/EMA身份、保存预测CPU复算、root目视、备份和全部自有进程退出证据，不改写旧receipt的pending字段。实验内容完成不代表总交接文档/资料包或用户消息已交付。

| 最终角色 | Update | 本地完整权重位置 | 接手含义 |
|---|---:|---|---|
| original best＋latest | 67180 | `/Users/lucas/Desktop/turbulence IR/unidb_dual_20261104_until00/checkpoint_backup_staging/original/U00067180.pt` | 完整RAW/Adam/EMA/RNG/cursor/scheduler；完整字节SHA复用已验receipt，本刷新只stat |
| improved latest | 40848 | `/Users/lucas/Desktop/turbulence IR/unidb_dual_20261104_until00/checkpoint_backup_staging/improved/U00040848.pt` | 完整RAW/Adam/EMA/RNG/cursor/scheduler；完整字节SHA复用已验receipt，本刷新只stat |
| improved best | 10000 | `/Users/lucas/Desktop/turbulence IR/unidb_dual_20261103_until_noon/delivery/checkpoints/improved/U00010000.pt` | 完整RAW/Adam/EMA/RNG/cursor/scheduler；完整字节SHA复用已验receipt，本刷新只stat |

原线best=latest U67180；改进latest U40848，full-best U10000来自旧noon。四个逻辑角色只需要三份unique文件，共6,584,061,917字节。U10000复用旧候选，不重新下载或复制大checkpoint。质量报告与best/latest角色不能相互替代，也不能把monitor U3000当full-best。

全部自有退出证据绑定 `534` 个精确身份及三段controller/rank/loader/CPU/helper范围，外部signals=0；这不是按历史PID猜测，也不对drivercontext推断OS存活。

只读入口：当前实验 `delivery/experiment_full_delivery_receipt.json`、`delivery/all_owned_process_exit_audit.json`、`delivery/unique_checkpoint_backup_audit.json` 与 `delivery/EXPERIMENT_FINAL_RECEIPT_VALIDATION.json`；其绝对路径/SHA在asset_evidence.json的新来源与final_unidb_update中。无需运行旧launcher、重新算模型、重复CPU指标或恢复训练。

本次使用 `refresh_final_unidb_assets.py` 从 `asset_preparation_snapshot_20261104` 增量刷新。旧 `build_asset_register.py` 仅作历史生成器；**不要为最终状态重跑旧builder**，它会覆盖远端计划并破坏已完成stat补证的planSHA绑定。原REMOTE_PATH_CHECK_PLAN及两份RESULT字节保持不变，A100连接未核局限不因本次UniDB终评而消失。
