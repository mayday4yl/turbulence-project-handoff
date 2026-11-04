# Heat-Chamber 官方数据获取与固定测试集适用性核查

核查日期：2026-10-29，北京时间。来源快照汇总时间：14:40。

## 结论

**当前无法通过核查到的官方公开入口下载 Heat-Chamber。实际场景数、退化帧数、GT 数均无法从原始文件验收，暂不能把它登记为已准备好的固定真实测试集。**

这表示公开获取受阻，不能据此认定作者已彻底删除文件或不存在私人备份。首次官方入口检查时未下载到数据；后续扩大检索取得的是演示包，详情见 [扩大公开检索结果](<EXTENDED_PUBLIC_SEARCH_ZH.md>)。HTTP 错误页面不是数据包，演示包也未作为原始 Heat-Chamber 接收。

## 已执行的获取检查

| 检查项 | 实际结果 | 证据 |
|---|---|---|
| 官方 Google Drive 文件页面 | 最终 HTTP 404；正文提示请求的文件不存在 | `../evidence/official_drive_view.headers`、`official_drive_view.html` |
| 同一文件的标准 `uc?export=download` 端点 | 跟随重定向后最终 HTTP 404，返回 HTML 错误页 | `../evidence/official_drive_download.headers`、`official_drive_download.response` |
| VITA-Group/TurbNet README | 仍指向同一 Drive 文件 ID | `../evidence/TurbNet_README.md` |
| 原作者实验室 i2lab-purdue/Turbulence-Text-Dataset | 仍指向同一 Drive 文件 ID；仓库根目录仅 README，没有数据文件 | `../evidence/i2lab_dataset_README.md`、`OFFICIAL_REPOSITORY_CHECKS.json` |
| 同团队 DATUM、MambaTM 数据入口 | 都引用同一个失效 ID，没有独立副本入口 | `../evidence/DATUM_README.md`、`MambaTM_README.md` |
| TurbNet Releases 与仓库树 | Releases 为空；树中没有 Heat-Chamber 数据包或逐图清单 | `../evidence/OFFICIAL_REPOSITORY_CHECKS.json` |
| TurbNet issue #7 | 2026-06-02 提交的失效反馈仍 open，评论数为 0 | `../evidence/TurbNet_issue7.json` |
| 实验室数据仓库 issue #1 | 2026-06-21 请求新的数据链接；仍 open，评论数为 0 | `../evidence/OFFICIAL_REPOSITORY_CHECKS.json` |

这些页面中出现相同链接，属于同一上游入口的重复引用，不是多个独立可用镜像。TMT README 也已检查，未发现 Heat-Chamber 的备用入口。此次未采用来源不明的第三方压缩包，也没有联系或向作者发送消息。

官方文件 ID：`14iVachB95bCCtke8ONPD9CCH20JO75v2`。

## 数量能核实到哪一步

| 项目 | 当前证据允许的结论 |
|---|---|
| 原论文规模描述 | 第 4.2 节写有 2,400 张不同图像，并说明每个场景含 100 帧。属于文献描述。 |
| 原始包实际场景数 | 未知：未取得包及目录清单。 |
| 实际退化图数、GT 数、配对数 | 未知，不能将文献中的 2,400 当作已实数的退化帧数量。 |
| 是否为 24 场景 × 100 帧 | 未核实，不作此推算。 |
| PiRN 表 2 使用的精确场景和帧集合 | 未核实；PiRN 当前仓库没有提交评分脚本引用的两个测试列表文件。 |
| 数据包文件名、字节数、SHA256 | 未取得，均为 null。证据清单中的 SHA256 仅用于保存页面与核查文件。 |

原论文依据：[ECCV 2022，第 4.2 节](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136790419.pdf)。本地已存文本 `../../papers/turbnet_2022.txt` 第 441–467 行。PiRN 评分入口：[estimate_psnr.py](https://github.com/VITA-Group/PiRN/blob/main/assets/estimate_psnr.py#L36-L38)。

## 能否成为我们的固定真实测试集

**研究用途适合，数据接收尚未通过。** 原始采集是在受控实验室光路中产生真实热湍流，并在关闭热源后获得清晰参考图，因此可作为合成训练到实验室真实退化的外部配对测试。它覆盖的是实验室静态场景，不能单独代表室外长距离自然湍流。

拿到来源可追溯的原始包后，才可完成以下接收工作：

1. 保留原包、下载来源、时间、完整字节数和 SHA256；保留原目录，不重编码原图。
2. 全量清点场景、退化帧和 GT；检查解码、尺寸、通道、逐帧 GT 配对与缺失项；按文件和解码像素检查重复项。
3. 人工检查少量固定配对，确认清晰参考图、裁切和坐标对应关系。若配对或几何基准存在问题，先记录，不用 GT 对单个模型的输出作优化配准或挑选。
4. 冻结所有合格场景和帧的排序清单、文件哈希和排除理由。选择依据限于数据完整性，不依据模型得分。多帧模型和单帧模型分开规定输入协议。
5. 对当前单帧任务，每次前向只输入一张退化 RGB 图像；GT 只用于统一评分。事先冻结尺寸处理、裁边、数值范围与 PSNR/SSIM/LPIPS/MAE 实现。
6. 保持真实测试集不参与训练、微调、选 checkpoint 或调后处理；汇报逐图指标，并同时给出全帧均值和按场景等权均值，避免同场景多帧被当作独立场景。
7. 检查并记录各预训练模型是否已使用该 benchmark 训练或调参。若缺少训练数据声明，保留这一可比性限制。

即使未来完成上述冻结，也应称为“我们的 Heat-Chamber 固定测试清单”；在缺少 PiRN 原始测试清单时，不能声称精确复现其 20.54 dB 的样本协议。

## 执行范围与复现

本轮只使用本机文件读写、公开网页和 GitHub 只读接口。没有连接训练服务器，没有加载任何恢复模型，没有启动推理、训练或微调，没有修改 D0、G4 Val1000 或锁定 Test 数据。

在项目根目录执行以下命令可在新建目录中重新保存官方入口的响应（这是入口检查，不是完整数据包下载）：

```bash
bash heat_chamber_audit_20261029/scripts/recheck_official.sh /tmp/heat_chamber_recheck
```

`evidence/SOURCE_INDEX.json` 保存来源 URL 与 README blob SHA；`reports/ACQUISITION_RESULT.json` 保存机器可读结果；`SHA256SUMS.txt` 固定本轮全部核查文件。当前没有可冻结的逐图测试 manifest。
