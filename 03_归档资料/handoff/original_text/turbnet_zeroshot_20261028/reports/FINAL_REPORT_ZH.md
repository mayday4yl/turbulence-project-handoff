# TurbNet 官方预训练 Zero-shot 核查结果

2026-10-28，Asia/Shanghai。**当前公开官方渠道未找到可下载的预训练 checkpoint，按用户要求停止 TurbNet 测试。尚不能判断其在 PiRN/P2S G4 上的真实恢复表现。**

本轮完成了来源可用性、静态源码和共享资源审查。官方预训练权重下载、架构与权重的严格匹配、Smoke Test、完整 Val1000 及逐图配对比较均没有完成，不能将本报告视为性能验证通过。

## 官方权重的具体阻碍

| 核查位置 | 本轮直接观察 | 证据 |
|---|---|---|
| [官方仓库](https://github.com/VITA-Group/TurbNet/tree/4ec7239b47328d286a834d50636411cf82741ee8) | 已下载源码，固定 commit `4ec7239b47328d286a834d50636411cf82741ee8`；17个跟踪文件，无 checkpoint | `evidence/SOURCE_AUDIT.json` |
| [官方配置](https://github.com/VITA-Group/TurbNet/blob/4ec7239b47328d286a834d50636411cf82741ee8/config.yml) | 期望 `./weights/turb_current49.pth`，仓库中没有该文件或 weights 目录 | 同上 |
| [README 指定的 Drive](https://drive.google.com/drive/folders/1ijaHurveoWKBJrpDAzpeqGtVy-9-36Pw) | 连接器成功读取目录，当前只列出一个 `PiRN Models` 文件夹 | `evidence/OFFICIAL_DRIVE_ROOT.json` |
| [PiRN Models 子目录](https://drive.google.com/drive/folders/1J_HrCYV1FNLrl0QEjQrsLSKk7eZK8wp_) | 成功读取，返回 `files: []`、`is_empty: true` | `evidence/OFFICIAL_DRIVE_CHILD.json` |
| [GitHub Releases API](https://api.github.com/repos/VITA-Group/TurbNet/releases?per_page=100) | 返回空列表，没有发布版本可供获取权重 | `evidence/GITHUB_RELEASES.json` |

这是本轮可访问官方位置的观察，不能据此断言作者从未训练或私人渠道没有权重。没有发现可供发起二进制下载的官方权重文件 ID，因此也没有权重 SHA256。该阻碍是缺少权重文件，而非模型推理 OOM、NPU 算子报错或性能不佳。

官方历史反馈与本轮结果一致：[Issue #2](https://github.com/VITA-Group/TurbNet/issues/2) 中作者提供了同一个 Drive 地址，后续有人反馈其中没有模型文件；[Issue #6](https://github.com/VITA-Group/TurbNet/issues/6) 仍处于 open，报告 `turb_current49.pth` 缺失。这些反馈作为旁证，主要结论来自本轮仓库、Releases、Drive 目录读取。原始响应已保存。

## 执行边界

| 阶段 | 本轮状态 |
|---|---|
| 官方源码下载和固定版本 | 已完成，源码工作树干净，逐文件 SHA256 已存 |
| 官方预训练权重获取 | 阻塞，无可下载文件 |
| 架构与官方 state_dict 严格加载 | 未执行，无法验证；静态语法解析不能代替加载成功 |
| 独立模型运行环境 | 未创建，在安装依赖和模型运行前即停止 |
| 单 RGB 图像 Smoke Test | 未执行，预处理、形状、数值范围尚未获得运行证据 |
| Val1000 PSNR/SSIM/LPIPS | 未执行 |
| 与输入 G4、旧 D0 U14927 EMA 的逐图比较 | 未执行 |
| TurbNet 恢复图与逐图指标 | 没有产出，不以输入图或仓库示例图冒充 |
| 微调、新训练、随机权重推理、替换其他模型 | 均未执行 |

本轮新增 NPU 进程为0，模型推理图像数为0，optimizer 更新数为0，远端文件写入为0。独立产物均位于本地 `turbnet_zeroshot_20261028/`。没有改动 D0 执行入口、环境、配置、STOP、checkpoint 或调度器。

## 单帧接口和官方脚本静态问题

以下仅为源码结论，尚未经过官方权重运行验证：

- `TurbulenceNet.forward(self, inp_img)` 只接受一张图像张量；恢复结果为返回值第二项 `out_J`（官方 evaluate 中命名为 `J`）。不需要 GT、模拟器参数或额外帧作为模型输入。
- 官方读取使用 PIL `convert('RGB')` 和 `ToTensor()`，对应 RGB、浮点 `[0,1]` 输入，没有均值/标准差归一化。实际数值核对仍须等待 Smoke Test。
- 源码含三次 `PixelUnshuffle(2)` 下采样，推断高宽应为8的倍数；输出 `out_J` 没有最终 clamp。实际分辨率保持、pad/crop 必要性和输出范围均未验证。
- `utils/data_loader.py` 的 `EvaluateData.get_images()` 注释掉了 `turb_reshaped`、`gt_reshaped` 的赋值，却继续引用它们；原示例不能直接当作已验证评估入口。
- `restore.py` 虽然计算了 `J`，但调用 `save_images(tiles, confs)`，实际保存的是输入 tiles；默认权重名又是 `turb_current37.pth`。后续若权重可得，独立适配器应取 `J`，并检查结果确实来自网络输出。

这些问题已记录，官方源码未修改。没有为了绕过权重门槛而实例化随机模型或修改现有 D0 代码。

## NPU4–7 是否适合临时共卡

北京时间 **21:42:54** 开始保存的只读快照中，D0 正式 run 为 `V6_D0_U14927_official_residual_microopt_01`，四个模型 worker PID 为 `2503018`–`2503021`；STATUS 报 U20020，读取期间四 rank 更新日志已到 U20024，最近记录有限值检查正常。

| 物理卡 | HBM 已用 / 总量（MiB） | 瞬时 AICore |
|---|---:|---:|
| NPU4 | 17209 / 65536 | 84% |
| NPU5 | 17209 / 65536 | 93% |
| NPU6 | 17210 / 65536 | 56% |
| NPU7 | 17211 / 65536 | 3% |

有约47 GiB/卡的显存余量，但这不能证明共卡推理影响小。AICore 是瞬时采样，NPU7 的3%不能视为空闲卡；四卡 DDP 同步会受较慢 rank 约束，在其中一张卡加入推理可能增加整个 D0 的更新时间。独立环境也不能隔离同卡算力竞争。

**本轮没有共卡试跑，因此没有 TurbNet 耗时、完整1000图时长或 D0 降速的实测结论。** 在权重未获取前，共用训练卡没有验证价值。本轮没有占用其他卡；即使快照显示部分卡当时无模型进程，也没有把它们认定为可用资源。

旧 D0 U14927 checkpoint 的本轮只读 SHA256 为：

`0c7903726f262c56fb40333dcaf70e0e205109c618456bb39e216987bd48d9b6`

与既有验收一致，大小580,755,715字节，读取前后元信息稳定。D0 源码、指标文件、运行配置及相关 PID 另存于 `evidence/REMOTE_READONLY_SNAPSHOT.json`。这是只读完整性核查，并非对共卡影响的 A/B 测速。

## 原 Val1000 协议和旧 D0 对照定位

已定位原固定 Val1000：1000个 G4 observation，250个不同 Clean source。原入口为 `v6_diffuir_l_20261027/method_impl/evaluation.py` 的 `validation_rows()`，仍使用原有 manifest 和 GT。不得以 Train300、Val100、Val50 或新抽样替代。

现有 U14927 EMA 全量结果目录：

`/data2/yanglin/v6_diffuir_l_20261027/distributed/V6_D0_U14927_baseline01/validation/U014927_1000/`

本轮只读取、核验已有结果，没有重跑 D0：

- `SUMMARY.json` SHA256：`197bee731d274d097e389da173bcadbcddb8d458b51865ba685db3f56f8daaba`。
- `PER_IMAGE.jsonl` SHA256：`4fded91c853b4d471597e4c5ec3881fba97e48f2cd5b0a70ad8446c5d5f7dc24`；确认1000行、250源。
- 原有参考值：输入 G4 PSNR 18.79593752 dB；D0 U14927 EMA PSNR 21.06020173 dB、SSIM 0.54878696、LPIPS 0.52907033。**这是历史结果的当前文件核验，不是本轮新评估，也没有 TurbNet 配对结论。**

若将来获得可追溯的官方权重，所需协议已记录在 `RESULT.json`，但本轮不会自动恢复：

1. 新独立环境中严格匹配官方架构及权重，记录来源、文件 SHA256、键名差异和运行版本；不接受缺失参数的随机补齐。
2. 小规模 Smoke Test 校验单 RGB 输入、通道顺序、归一化、原分辨率、有限值及输出范围；GT 仅进入模型运行后的评估函数。
3. 沿用同一 Val1000、同一 GT 和旧指标实现：PSNR 为逐图 RGB float64 MSE/range1 后求平均 dB；SSIM 为 RGB Gaussian 11×11/sigma1.5/population covariance；LPIPS 为官方 AlexNet v0.1、normalize=True、FP32。
4. 指标保持无 clamp、resize 或 crop，PNG 仅为展示裁限。按 `sample_id`、`source_id` 和数据哈希与输入及 U14927 配对；保存原始浮点输出和视觉图。

指标实现固定为 `/data2/yanglin/v5_concertormer_10k_20261021/training/image_metrics.py`，SHA256：`ea241b2477a07b67d2704d47f8db9e1e38a47ac9593524203f222807db05ef2d`。

当前下一步依赖是取得可验证来源的官方 checkpoint。没有发邮件、创建 Issue、安排自动重试、替换模型或启动微调。
